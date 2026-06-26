from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass
class EntityRecord:
    name: str
    capabilities: set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    durations: Dict[str, float] = field(default_factory=dict)
    state_since_s: Dict[str, float] = field(default_factory=dict)
    state_provenance: Dict[str, str] = field(default_factory=dict)
    source: str = 'authored'


class WorldStateError(ValueError):
    pass


class WorldStateStore:
    """Simulator-owned semantic state for devices and derived environment facts."""

    def __init__(self) -> None:
        self.sim_time_s = 0.0
        self.entities: Dict[str, EntityRecord] = {}
        self.relations: Dict[str, Dict[str, Any]] = {}
        self.events: list[Dict[str, Any]] = []
        self.history: list[Dict[str, Any]] = []

    def clear(self) -> None:
        self.sim_time_s = 0.0
        self.entities = {}
        self.relations = {}
        self.events = []
        self.history = []

    def register_entity(
        self,
        name: str,
        metadata: Optional[Mapping[str, Any]] = None,
        *,
        infer_profile: bool = True,
    ) -> EntityRecord:
        entity_name = str(name or '').strip()
        if not entity_name:
            raise WorldStateError('entity_name_required')
        authored = dict(metadata or {})
        inferred = infer_entity_profile(entity_name) if infer_profile else {}
        capabilities = {
            str(item).strip()
            for item in (
                list(inferred.get('capabilities') or [])
                + list(authored.get('capabilities') or [])
            )
            if str(item).strip()
        }
        defaults = dict(inferred.get('state_defaults') or {})
        defaults.update(dict(authored.get('state_defaults') or authored.get('defaults') or {}))
        merged_metadata = dict(inferred)
        merged_metadata.update(authored)
        merged_metadata['capabilities'] = sorted(capabilities)
        merged_metadata['state_defaults'] = copy.deepcopy(defaults)
        source = 'authored' if authored else 'inferred_profile'
        record = EntityRecord(
            name=entity_name,
            capabilities=capabilities,
            metadata=merged_metadata,
            defaults=copy.deepcopy(defaults),
            state=copy.deepcopy(defaults),
            source=source,
        )
        record.state.setdefault('active_duration_s', 0.0)
        for key in record.state:
            record.state_since_s[key] = self.sim_time_s
            record.state_provenance[key] = (
                'authored_default' if authored else 'inferred_default'
            )
        self.entities[self._key(entity_name)] = record
        return record

    def reset(self) -> None:
        self.sim_time_s = 0.0
        self.events = []
        self.history = []
        for record in self.entities.values():
            record.state = copy.deepcopy(record.defaults)
            record.state.setdefault('active_duration_s', 0.0)
            record.durations = {}
            record.state_since_s = {
                key: 0.0
                for key in record.state
            }
            record.state_provenance = {
                key: (
                    'authored_default'
                    if record.source == 'authored'
                    else 'inferred_default'
                )
                for key in record.state
            }

    def capabilities(self, entity: str) -> Dict[str, Any]:
        record = self.require_entity(entity)
        return {
            'entity': record.name,
            'capabilities': sorted(record.capabilities),
            'metadata': copy.deepcopy(record.metadata),
            'source': record.source,
        }

    def get_entity_state(self, entity: str) -> Dict[str, Any]:
        record = self.require_entity(entity)
        return self._record_payload(record)

    def apply_state_requirements(
        self,
        requirements: Iterable[Mapping[str, Any]],
        *,
        create_missing: bool = False,
    ) -> list[Dict[str, Any]]:
        results: list[Dict[str, Any]] = []
        for raw_requirement in requirements or []:
            requirement = dict(raw_requirement or {})
            try:
                results.append(
                    self.ensure_state_requirement(
                        requirement,
                        create_missing=create_missing,
                    )
                )
            except (TypeError, ValueError, WorldStateError) as exc:
                results.append({
                    'ok': False,
                    'reason': str(exc),
                    'requirement': copy.deepcopy(requirement),
                })
        return results

    def ensure_state_requirement(
        self,
        requirement: Mapping[str, Any],
        *,
        create_missing: bool = False,
    ) -> Dict[str, Any]:
        req = dict(requirement or {})
        entity = str(
            req.get('entity')
            or req.get('device')
            or req.get('object')
            or req.get('name')
            or ''
        ).strip()
        property_path = str(
            req.get('property')
            or req.get('field')
            or req.get('path')
            or ''
        ).strip()
        selector = dict(req.get('selector') or {})
        if not property_path:
            raise WorldStateError('state_property_required')

        values = self._state_requirement_values(req)
        profile = self._state_requirement_profile(property_path, values)
        if entity:
            record = self.entities.get(self._key(entity))
            if record is None:
                if not create_missing:
                    return {
                        'ok': False,
                        'reason': 'entity_not_registered',
                        'entity': entity,
                        'property': property_path,
                    }
                record = self.register_entity(
                    entity,
                    {
                        'capabilities': sorted(profile['capabilities']),
                        'state_defaults': {property_path: profile['default']},
                    },
                    infer_profile=True,
                )
            records = [record]
        elif selector:
            records = self._select_requirement_records(selector)
            if not records:
                return {
                    'ok': False,
                    'reason': 'selector_matched_no_entities',
                    'selector': selector,
                    'property': property_path,
                }
        else:
            raise WorldStateError('state_requirement_entity_or_selector_required')

        applied_entities = []
        for record in records:
            self._apply_state_requirement_to_record(
                record,
                property_path,
                values,
                profile,
                req,
            )
            applied_entities.append(record.name)
        return {
            'ok': True,
            'entity': entity,
            'selector': selector,
            'entities': applied_entities,
            'property': property_path,
            'default': copy.deepcopy(profile['default']),
            'allowed_values': copy.deepcopy(profile['allowed_values']),
            'predicate': str(req.get('predicate') or ''),
        }

    def set_relation(
        self,
        source: str,
        target: str,
        relation: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_record = self.require_entity(source)
        target_record = self.require_entity(target)
        relation_name = str(relation or '').strip().lower()
        if relation_name not in {'fills', 'heats', 'humidifies', 'contains', 'supports'}:
            raise WorldStateError(f'unsupported_entity_relation:{relation_name}')
        key = self._relation_key(source_record.name, target_record.name, relation_name)
        payload = {
            'source': source_record.name,
            'target': target_record.name,
            'relation': relation_name,
            'metadata': dict(metadata or {}),
        }
        self.relations[key] = payload
        event = {
            'event_type': 'entity_relation_set',
            **copy.deepcopy(payload),
            'sim_time_s': self.sim_time_s,
            'source_system': 'world_state_store',
        }
        self.events.append(event)
        self.history.append(event)
        return copy.deepcopy(payload)

    def remove_relation(self, source: str, target: str, relation: str) -> bool:
        key = self._relation_key(source, target, relation)
        removed = self.relations.pop(key, None)
        if removed is None:
            return False
        event = {
            'event_type': 'entity_relation_removed',
            **copy.deepcopy(removed),
            'sim_time_s': self.sim_time_s,
            'source_system': 'world_state_store',
        }
        self.events.append(event)
        self.history.append(event)
        return True

    def interact(
        self,
        entity: str,
        action: str,
        parameters: Optional[Mapping[str, Any]] = None,
        *,
        sim_time_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        if sim_time_s is not None:
            self.sync_time(sim_time_s)
        record = self.require_entity(entity)
        action_name = str(action or '').strip().lower()
        params = dict(parameters or {})
        before = copy.deepcopy(record.state)

        if action_name == 'toggle':
            self._require_capability(record, 'toggleable')
            requested = params.get('on', params.get('value'))
            if requested is None:
                requested = not self._is_active(record.state.get('power'))
            self._set_state(record, 'power', 'on' if self._coerce_bool(requested) else 'off')
        elif action_name in {'start', 'stop'}:
            self._require_any_capability(record, {'toggleable', 'timer_capable'})
            self._set_state(record, 'power', 'on' if action_name == 'start' else 'off')
            self._set_state(record, 'running', action_name == 'start')
        elif action_name == 'set_level':
            self._require_any_capability(
                record,
                {'level_controllable', 'water_source', 'heat_source'},
            )
            level = self._bounded_float(params.get('level', params.get('value')), 0.0, 1.0)
            property_name = str(params.get('property') or '').strip()
            if not property_name:
                property_name = 'water_flow_level' if 'water_source' in record.capabilities else 'level'
            self._set_state(record, property_name, level)
            if 'water_source' in record.capabilities:
                flow_state = 'off' if level <= 0.0 else 'full_blast' if level >= 0.9 else 'on'
                self._set_state(record, 'water_flow', flow_state)
            if 'heat_source' in record.capabilities:
                self._set_state(record, 'heat_level', level)
            if level > 0.0 and 'toggleable' in record.capabilities:
                self._set_state(record, 'power', 'on')
        elif action_name == 'set_temperature':
            self._require_any_capability(
                record,
                {'temperature_controllable', 'water_source', 'heat_source', 'heatable'},
            )
            value = float(
                params.get('temperature_c')
                if params.get('temperature_c') is not None
                else params.get('value')
            )
            property_name = str(params.get('property') or '').strip()
            if not property_name:
                property_name = (
                    'water_temperature_c'
                    if 'water_source' in record.capabilities
                    else 'temperature_setpoint_c'
                )
            self._set_state(record, property_name, value)
        elif action_name == 'set_timer':
            self._require_capability(record, 'timer_capable')
            seconds = max(
                0.0,
                float(
                    params.get('seconds')
                    if params.get('seconds') is not None
                    else params.get('value')
                ),
            )
            self._set_state(record, 'timer_set_s', seconds)
            self._set_state(record, 'timer_remaining_s', seconds)
        elif action_name in {'open', 'close'}:
            self._require_capability(record, 'openable')
            self._set_state(record, 'open_state', 'open' if action_name == 'open' else 'closed')
            self._set_state(record, 'open_fraction', 1.0 if action_name == 'open' else 0.0)
        elif action_name == 'set_drain':
            self._require_capability(record, 'drainable')
            drain_state = str(params.get('state') or params.get('value') or '').strip().lower()
            if drain_state not in {'open', 'closed', 'blocked'}:
                raise WorldStateError(f'invalid_drain_state:{drain_state}')
            self._set_state(record, 'drain', drain_state)
        elif action_name == 'set_state':
            state_patch = dict(params.get('state') or {})
            if not state_patch:
                property_name = str(params.get('property') or '').strip()
                if property_name:
                    state_patch[property_name] = params.get('value')
            if not state_patch:
                raise WorldStateError('state_patch_required')
            for property_name, value in state_patch.items():
                self._set_state(record, str(property_name), value)
        else:
            raise WorldStateError(f'unsupported_entity_action:{action_name}')

        self._refresh_derived_state(record)
        event = {
            'event_type': 'entity_interaction',
            'entity': record.name,
            'action': action_name,
            'parameters': params,
            'before_state': before,
            'after_state': copy.deepcopy(record.state),
            'sim_time_s': self.sim_time_s,
            'source': 'world_state_store',
        }
        self.events.append(event)
        self.history.append(event)
        return {
            'entity': record.name,
            'action': action_name,
            'state': copy.deepcopy(record.state),
            'capabilities': sorted(record.capabilities),
            'event': copy.deepcopy(event),
        }

    def sync_time(self, sim_time_s: float) -> None:
        target = float(sim_time_s)
        if not math.isfinite(target):
            raise WorldStateError('sim_time_must_be_finite')
        if target < self.sim_time_s:
            target = self.sim_time_s
        delta = target - self.sim_time_s
        if delta <= 0.0:
            return
        for record in self.entities.values():
            self._advance_record(record, delta)
        self._advance_relations(delta)
        self.sim_time_s = target

    def advance(self, delta_s: float) -> None:
        delta = max(0.0, float(delta_s))
        self.sync_time(self.sim_time_s + delta)

    def snapshot(self, *, clear_events: bool = False) -> Dict[str, Any]:
        payload = {
            'sim_time_s': self.sim_time_s,
            'entities': {
                record.name: self._record_payload(record)
                for record in sorted(self.entities.values(), key=lambda item: item.name)
            },
            'relations': [
                copy.deepcopy(item)
                for item in sorted(
                    self.relations.values(),
                    key=lambda value: (
                        str(value.get('relation') or ''),
                        str(value.get('source') or ''),
                        str(value.get('target') or ''),
                    ),
                )
            ],
            'events': copy.deepcopy(self.events),
            'event_count': len(self.events),
        }
        if clear_events:
            self.events = []
        return payload

    def require_entity(self, name: str) -> EntityRecord:
        record = self.entities.get(self._key(name))
        if record is None:
            raise WorldStateError(f'entity_not_registered:{name}')
        return record

    def _advance_record(self, record: EntityRecord, delta_s: float) -> None:
        for property_name, value in list(record.state.items()):
            key = self._duration_key(property_name, value)
            record.durations[key] = float(record.durations.get(key, 0.0)) + delta_s

        active = (
            self._is_active(record.state.get('power'))
            or bool(record.state.get('running', False))
            or self._is_active(record.state.get('water_flow'))
        )
        if active:
            record.state['active_duration_s'] = float(
                record.state.get('active_duration_s', 0.0) or 0.0
            ) + delta_s

        timer_remaining = record.state.get('timer_remaining_s')
        if active and timer_remaining is not None:
            remaining = max(0.0, float(timer_remaining) - delta_s)
            record.state['timer_remaining_s'] = remaining
            if remaining <= 0.0 and float(record.state.get('timer_set_s', 0.0) or 0.0) > 0.0:
                self._set_state(record, 'running', False)
                self._set_state(record, 'power', 'off')
                self.events.append({
                    'event_type': 'timer_completed',
                    'entity': record.name,
                    'sim_time_s': self.sim_time_s + delta_s,
                    'source': 'world_state_store',
                })

        if 'heat_source' in record.capabilities and self._is_active(record.state.get('power')):
            current = float(record.state.get('temperature_c', 22.0) or 22.0)
            setpoint = float(record.state.get('temperature_setpoint_c', 200.0) or 200.0)
            rate = float(record.metadata.get('heating_rate_c_per_s', 2.0) or 2.0)
            record.state['temperature_c'] = min(setpoint, current + rate * delta_s)
            record.state_provenance['temperature_c'] = 'symbolic_environment_model'
        elif 'heat_source' in record.capabilities:
            current = float(record.state.get('temperature_c', 22.0) or 22.0)
            ambient = float(record.metadata.get('ambient_temperature_c', 22.0) or 22.0)
            rate = float(record.metadata.get('cooling_rate_c_per_s', 0.25) or 0.25)
            record.state['temperature_c'] = max(ambient, current - rate * delta_s)
            record.state_provenance['temperature_c'] = 'symbolic_environment_model'

        if 'water_source' in record.capabilities and 'drainable' in record.capabilities:
            flow_level = float(record.state.get('water_flow_level', 0.0) or 0.0)
            fill = float(record.state.get('fill_fraction', 0.0) or 0.0)
            capacity = max(0.001, float(record.metadata.get('capacity_l', 20.0) or 20.0))
            if self._is_active(record.state.get('water_flow')) or flow_level > 0.0:
                flow_rate = float(record.metadata.get('flow_rate_l_per_s', 0.2) or 0.2)
                scale = flow_level if flow_level > 0.0 else 1.0
                fill += flow_rate * scale * delta_s / capacity
            drain_state = str(record.state.get('drain') or 'open').strip().lower()
            if drain_state == 'open':
                drain_rate = float(record.metadata.get('drain_rate_l_per_s', 0.4) or 0.4)
                fill -= drain_rate * delta_s / capacity
            self._set_state(
                record,
                'fill_fraction',
                max(0.0, fill),
                provenance='symbolic_environment_model',
            )

        was_overflowing = bool(record.state.get('overflowing', False))
        self._refresh_derived_state(record)
        if not was_overflowing and bool(record.state.get('overflowing', False)):
            event = {
                'event_type': 'overflow_started',
                'entity': record.name,
                'sim_time_s': self.sim_time_s + delta_s,
                'source': 'symbolic_environment_model',
            }
            self.events.append(event)
            self.history.append(event)

    def _advance_relations(self, delta_s: float) -> None:
        for relation in self.relations.values():
            source = self.entities.get(self._key(relation.get('source')))
            target = self.entities.get(self._key(relation.get('target')))
            if source is None or target is None:
                continue
            relation_name = str(relation.get('relation') or '').lower()
            metadata = dict(relation.get('metadata') or {})
            if relation_name == 'fills':
                self._advance_fill_relation(source, target, metadata, delta_s)
            elif relation_name == 'heats':
                self._advance_heat_relation(source, target, metadata, delta_s)
            elif relation_name == 'humidifies':
                self._advance_humidity_relation(source, target, metadata, delta_s)

    def _advance_fill_relation(
        self,
        source: EntityRecord,
        target: EntityRecord,
        metadata: Dict[str, Any],
        delta_s: float,
    ) -> None:
        if 'water_source' not in source.capabilities or 'fillable' not in target.capabilities:
            return
        flow_level = float(source.state.get('water_flow_level', 0.0) or 0.0)
        if not self._is_active(source.state.get('water_flow')) and flow_level <= 0.0:
            return
        flow_rate = float(
            metadata.get('flow_rate_l_per_s')
            or source.metadata.get('flow_rate_l_per_s')
            or 0.2
        )
        capacity = max(
            0.001,
            float(
                metadata.get('target_capacity_l')
                or target.metadata.get('capacity_l')
                or 1.0
            ),
        )
        level_scale = flow_level if flow_level > 0.0 else 1.0
        current_fill = float(target.state.get('fill_fraction', 0.0) or 0.0)
        was_overflowing = bool(target.state.get('overflowing', False))
        added_fraction = flow_rate * level_scale * delta_s / capacity
        self._set_state(
            target,
            'fill_fraction',
            current_fill + added_fraction,
            provenance='symbolic_environment_model',
        )
        self._set_state(
            target,
            'contents',
            str(metadata.get('contents') or 'water'),
            provenance='symbolic_environment_model',
        )
        source_temperature = source.state.get('water_temperature_c')
        if source_temperature is not None:
            current_temperature = float(target.state.get('temperature_c', source_temperature) or source_temperature)
            new_fill = max(0.000001, current_fill + added_fraction)
            mixed_temperature = (
                current_temperature * current_fill
                + float(source_temperature) * added_fraction
            ) / new_fill
            self._set_state(
                target,
                'temperature_c',
                mixed_temperature,
                provenance='symbolic_environment_model',
            )
        self._refresh_derived_state(target)
        if not was_overflowing and bool(target.state.get('overflowing', False)):
            event = {
                'event_type': 'overflow_started',
                'entity': target.name,
                'source_entity': source.name,
                'sim_time_s': self.sim_time_s + delta_s,
                'source': 'symbolic_environment_model',
            }
            self.events.append(event)
            self.history.append(event)

    def _advance_heat_relation(
        self,
        source: EntityRecord,
        target: EntityRecord,
        metadata: Dict[str, Any],
        delta_s: float,
    ) -> None:
        if (
            'heat_source' not in source.capabilities
            or not target.capabilities.intersection({'heatable', 'environment_zone'})
        ):
            return
        if not self._is_active(source.state.get('power')):
            return
        source_temperature = float(source.state.get('temperature_c', 22.0) or 22.0)
        target_temperature = float(target.state.get('temperature_c', 22.0) or 22.0)
        rate = float(
            metadata.get('heating_rate_c_per_s')
            or target.metadata.get('heating_rate_c_per_s')
            or 1.0
        )
        next_temperature = min(
            source_temperature,
            target_temperature + max(0.0, rate) * delta_s,
        )
        self._set_state(
            target,
            'temperature_c',
            next_temperature,
            provenance='symbolic_environment_model',
        )
        self._refresh_derived_state(target)

    def _advance_humidity_relation(
        self,
        source: EntityRecord,
        target: EntityRecord,
        metadata: Dict[str, Any],
        delta_s: float,
    ) -> None:
        if 'environment_zone' not in target.capabilities:
            return
        steaming = bool(source.state.get('boiling', False))
        if not steaming:
            source_temperature = float(source.state.get('temperature_c', 0.0) or 0.0)
            steaming = source_temperature >= float(metadata.get('steam_threshold_c', 60.0) or 60.0)
        if not steaming:
            return
        current = float(target.state.get('humidity_fraction', 0.4) or 0.4)
        rate = float(metadata.get('humidity_rate_per_s', 0.01) or 0.01)
        humidity = min(1.0, current + max(0.0, rate) * delta_s)
        self._set_state(
            target,
            'humidity_fraction',
            humidity,
            provenance='symbolic_environment_model',
        )
        self._set_state(
            target,
            'steam_level',
            'high' if humidity >= 0.75 else 'elevated',
            provenance='symbolic_environment_model',
        )

    def _refresh_derived_state(self, record: EntityRecord) -> None:
        if 'heat_source' in record.capabilities:
            temperature = float(record.state.get('temperature_c', 22.0) or 22.0)
            record.state['hot'] = temperature >= float(
                record.metadata.get('hot_threshold_c', 60.0) or 60.0
            )
            record.state_provenance['hot'] = 'symbolic_environment_model'
            if str(record.metadata.get('category') or '').lower() in {'burner', 'gas_burner'}:
                record.state['flame'] = (
                    'on' if self._is_active(record.state.get('power')) else 'off'
                )
                record.state_provenance['flame'] = 'symbolic_environment_model'
        if 'fillable' in record.capabilities and 'fill_fraction' in record.state:
            fill = max(0.0, float(record.state.get('fill_fraction', 0.0) or 0.0))
            record.state['fill_fraction'] = fill
            record.state['overflowing'] = fill > 1.0
            record.state_provenance['overflowing'] = 'symbolic_environment_model'
        if 'heatable' in record.capabilities and 'temperature_c' in record.state:
            boiling_point = float(record.metadata.get('boiling_point_c', 100.0) or 100.0)
            record.state['boiling'] = float(record.state['temperature_c']) >= boiling_point
            record.state_provenance['boiling'] = 'symbolic_environment_model'

    def _set_state(
        self,
        record: EntityRecord,
        property_name: str,
        value: Any,
        *,
        provenance: str = 'runtime_interaction',
    ) -> None:
        old_value = self._get_state_path(record.state, property_name)
        if self._value_equal(old_value, value):
            return
        self._set_state_path(record.state, property_name, value)
        record.state_since_s[property_name] = self.sim_time_s
        record.state_provenance[property_name] = str(provenance or 'runtime_interaction')
        event = {
            'event_type': 'entity_state_changed',
            'entity': record.name,
            'property': property_name,
            'before': old_value,
            'after': value,
            'sim_time_s': self.sim_time_s,
            'source': 'world_state_store',
        }
        self.events.append(event)
        self.history.append(event)

    def _apply_state_requirement_to_record(
        self,
        record: EntityRecord,
        property_path: str,
        values: list[Any],
        profile: Dict[str, Any],
        requirement: Mapping[str, Any],
    ) -> None:
        capabilities = {
            str(item).strip()
            for item in profile.get('capabilities') or []
            if str(item).strip()
        }
        if capabilities:
            record.capabilities.update(capabilities)
            record.metadata['capabilities'] = sorted(record.capabilities)

        allowed_values = self._dedupe_values(
            list(profile.get('allowed_values') or []) + list(values or [])
        )
        if allowed_values:
            allowed_map = dict(record.metadata.get('state_allowed_values') or {})
            allowed_map[property_path] = self._dedupe_values(
                list(allowed_map.get(property_path) or []) + allowed_values
            )
            record.metadata['state_allowed_values'] = copy.deepcopy(allowed_map)

        default = copy.deepcopy(profile.get('default'))
        if not self._has_state_path(record.defaults, property_path):
            self._set_state_path(record.defaults, property_path, default)
            metadata_defaults = dict(record.metadata.get('state_defaults') or {})
            self._set_state_path(metadata_defaults, property_path, default)
            record.metadata['state_defaults'] = metadata_defaults
        if not self._has_state_path(record.state, property_path):
            self._set_state_path(record.state, property_path, copy.deepcopy(default))
            record.state_since_s[property_path] = self.sim_time_s
            record.state_provenance[property_path] = 'spec_requirement_default'

        state_requirements = list(record.metadata.get('state_requirements') or [])
        requirement_payload = {
            'predicate': str(requirement.get('predicate') or ''),
            'property': property_path,
            'values': copy.deepcopy(values),
            'usage_id': str(requirement.get('usage_id') or ''),
            'source': str(requirement.get('source') or 'oracle_task_spec'),
        }
        requirement_key = json.dumps(
            requirement_payload,
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        existing_keys = {
            json.dumps(item, ensure_ascii=True, sort_keys=True, default=str)
            for item in state_requirements
            if isinstance(item, Mapping)
        }
        if requirement_key not in existing_keys:
            state_requirements.append(requirement_payload)
            record.metadata['state_requirements'] = state_requirements

    def _select_requirement_records(self, selector: Mapping[str, Any]) -> list[EntityRecord]:
        raw_selector = dict(selector or {})
        excluded = {
            self._key(item)
            for item in raw_selector.get('exclude') or []
            if str(item or '').strip()
        }
        expected_capability = str(raw_selector.get('capability') or '').strip().lower()
        expected_zone = str(raw_selector.get('zone') or raw_selector.get('room') or '').strip()
        category_contains = str(raw_selector.get('category_contains') or '').strip().lower()
        expected_role = str(raw_selector.get('role') or '').strip().lower()
        selected: list[EntityRecord] = []
        for key, record in self.entities.items():
            if key in excluded:
                continue
            metadata = dict(record.metadata or {})
            capabilities = {
                str(item).strip().lower()
                for item in (
                    record.capabilities
                    or metadata.get('capabilities')
                    or []
                )
            }
            if expected_capability and expected_capability not in capabilities:
                continue
            zone = str(
                metadata.get('zone')
                or metadata.get('room')
                or metadata.get('room_name')
                or ''
            )
            if expected_zone and not self._value_equal(zone, expected_zone):
                continue
            category = str(metadata.get('category') or '').lower()
            if (
                category_contains
                and category_contains not in category
                and category_contains not in record.name.lower()
            ):
                continue
            roles = {
                str(item).strip().lower()
                for item in (metadata.get('roles') or [])
                if str(item).strip()
            }
            single_role = str(metadata.get('role') or '').strip().lower()
            if single_role:
                roles.add(single_role)
            if expected_role and expected_role not in roles:
                continue
            selected.append(record)
        return selected

    @classmethod
    def _state_requirement_values(cls, requirement: Mapping[str, Any]) -> list[Any]:
        values: list[Any] = []
        for key in ('value', 'expected', 'state'):
            if key in requirement and requirement.get(key) is not None:
                values.append(requirement.get(key))
        for key in ('values', 'allowed_values'):
            raw_values = requirement.get(key)
            if isinstance(raw_values, (list, tuple, set)):
                values.extend(list(raw_values))
            elif raw_values is not None:
                values.append(raw_values)
        for key in ('min', 'max', 'minimum', 'maximum'):
            if requirement.get(key) is not None:
                values.append(requirement.get(key))
        return cls._dedupe_values(values)

    @classmethod
    def _state_requirement_profile(
        cls,
        property_path: str,
        values: Iterable[Any],
    ) -> Dict[str, Any]:
        key = cls._state_property_key(property_path)
        value_list = cls._dedupe_values(list(values or []))
        capabilities: set[str] = set()
        default: Any = 'unknown'
        allowed_values: list[Any] = ['unknown']

        if key in {'power', 'devicepower'}:
            capabilities.add('toggleable')
            default = 'off'
            allowed_values = ['off', 'on']
        elif key in {'running', 'active', 'enabled', 'on', 'pluggedin', 'wet'}:
            if key in {'running', 'active', 'enabled', 'on'}:
                capabilities.add('toggleable')
            if key == 'pluggedin':
                capabilities.add('power_connectable')
            default = False
            allowed_values = [False, True]
        elif key in {'openstate', 'doorstate', 'lidstate'}:
            capabilities.add('openable')
            default = 'closed'
            allowed_values = ['closed', 'open']
        elif key in {'openfraction'}:
            capabilities.add('openable')
            default = 0.0
            allowed_values = [0.0, 1.0]
        elif key in {'waterflow'}:
            capabilities.update({'water_source', 'level_controllable'})
            default = 'off'
            allowed_values = ['off', 'on', 'full_blast']
        elif key in {'waterflowlevel'}:
            capabilities.update({'water_source', 'level_controllable'})
            default = 0.0
            allowed_values = [0.0, 1.0]
        elif key in {'level'}:
            capabilities.add('level_controllable')
            default = 0.0
            allowed_values = [0.0, 1.0]
        elif key in {'heatlevel'}:
            capabilities.update({'heat_source', 'level_controllable'})
            default = 0.0
            allowed_values = [0.0, 1.0]
        elif key in {'temperaturec', 'temperaturesetpointc'}:
            capabilities.add('temperature_controllable')
            default = 22.0
            allowed_values = [22.0]
        elif key in {'watertemperaturec', 'watertemperaturesetting'}:
            capabilities.update({'water_source', 'temperature_controllable'})
            default = 22.0
            allowed_values = [22.0]
        elif key in {'timerremainingseconds', 'timerremainings', 'timersets', 'timerseconds'}:
            capabilities.add('timer_capable')
            default = 0.0
            allowed_values = [0.0]
        elif key in {'fillfraction'}:
            capabilities.update({'container', 'fillable'})
            default = 0.0
            allowed_values = [0.0, 1.0]
        elif key in {'contents', 'contentstate'}:
            capabilities.update({'container', 'fillable'})
            default = 'empty'
            allowed_values = ['empty']
        elif key in {'overflowing'}:
            capabilities.update({'drainable', 'fillable'})
            default = False
            allowed_values = [False, True]
        elif key in {'drain', 'drainstate'}:
            capabilities.add('drainable')
            default = 'open'
            allowed_values = ['open', 'closed', 'blocked']
        elif key in {'humidityfraction'}:
            capabilities.add('environment_zone')
            default = 0.4
            allowed_values = [0.4, 1.0]
        elif key in {'steamlevel'}:
            capabilities.add('environment_zone')
            default = 'normal'
            allowed_values = ['normal', 'elevated', 'high']
        elif key in {'hot', 'ishot', 'boiling'}:
            capabilities.add('heatable')
            default = False
            allowed_values = [False, True]
        elif key in {'flame', 'flamestate'}:
            capabilities.update({'toggleable', 'heat_source'})
            default = 'off'
            allowed_values = ['off', 'on']
        elif key in {'lit', 'islit'}:
            capabilities.update({'toggleable', 'heat_source'})
            default = False
            allowed_values = [False, True]
        elif key in {'wetstate', 'moisturestate'}:
            default = 'dry'
            allowed_values = ['dry', 'wet']
        elif value_list:
            sample = value_list[0]
            if isinstance(sample, bool):
                default = False
                allowed_values = [False, True]
            elif isinstance(sample, (int, float)):
                default = 0.0
                allowed_values = [0.0]

        allowed_values = cls._dedupe_values(list(allowed_values) + value_list)
        return {
            'default': default,
            'allowed_values': allowed_values,
            'capabilities': capabilities,
        }

    @staticmethod
    def _state_property_key(property_path: str) -> str:
        return ''.join(
            char.lower()
            for char in str(property_path or '')
            if char.isalnum()
        )

    @classmethod
    def _dedupe_values(cls, values: Iterable[Any]) -> list[Any]:
        deduped: list[Any] = []
        for value in values or []:
            if any(cls._value_equal(existing, value) for existing in deduped):
                continue
            deduped.append(value)
        return deduped

    @classmethod
    def _has_state_path(cls, state: Mapping[str, Any], property_path: str) -> bool:
        marker = object()
        return cls._get_state_path(state, property_path, default=marker) is not marker

    @staticmethod
    def _get_state_path(
        state: Mapping[str, Any],
        property_path: str,
        *,
        default: Any = None,
    ) -> Any:
        if not str(property_path or '').strip():
            return default
        current: Any = state
        for part in str(property_path).split('.'):
            if not part:
                continue
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current.get(part)
        return current

    @staticmethod
    def _set_state_path(state: Dict[str, Any], property_path: str, value: Any) -> None:
        parts = [
            part
            for part in str(property_path or '').split('.')
            if part
        ]
        if not parts:
            return
        current = state
        for part in parts[:-1]:
            child = current.get(part)
            if not isinstance(child, dict):
                child = {}
                current[part] = child
            current = child
        current[parts[-1]] = value

    @staticmethod
    def _record_payload(record: EntityRecord) -> Dict[str, Any]:
        return {
            'name': record.name,
            'capabilities': sorted(record.capabilities),
            'metadata': copy.deepcopy(record.metadata),
            'state': copy.deepcopy(record.state),
            'durations': copy.deepcopy(record.durations),
            'state_since_s': copy.deepcopy(record.state_since_s),
            'state_provenance': copy.deepcopy(record.state_provenance),
            'source': record.source,
        }

    @staticmethod
    def _require_capability(record: EntityRecord, capability: str) -> None:
        if capability not in record.capabilities:
            raise WorldStateError(
                f'entity_capability_missing:{record.name}:{capability}'
            )

    @staticmethod
    def _require_any_capability(record: EntityRecord, capabilities: set[str]) -> None:
        if not record.capabilities.intersection(capabilities):
            expected = ','.join(sorted(capabilities))
            raise WorldStateError(
                f'entity_capability_missing:{record.name}:one_of[{expected}]'
            )

    @staticmethod
    def _bounded_float(value: Any, minimum: float, maximum: float) -> float:
        number = float(value)
        return max(minimum, min(maximum, number))

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or '').strip().lower() in {
            '1',
            'true',
            'yes',
            'on',
            'enabled',
            'running',
        }

    @classmethod
    def _is_active(cls, value: Any) -> bool:
        return cls._coerce_bool(value) or str(value or '').strip().lower() in {
            'full_blast',
            'low',
            'medium',
            'high',
        }

    @staticmethod
    def _duration_key(property_name: str, value: Any) -> str:
        if isinstance(value, (dict, list)):
            serialized = json.dumps(value, ensure_ascii=True, sort_keys=True)
        else:
            serialized = str(value)
        return f'{property_name}={serialized}'

    @staticmethod
    def _value_equal(lhs: Any, rhs: Any) -> bool:
        if isinstance(lhs, str) or isinstance(rhs, str):
            normalize = lambda value: ''.join(
                char.lower()
                for char in str(value or '')
                if char.isalnum()
            )
            return normalize(lhs) == normalize(rhs)
        return lhs == rhs

    @staticmethod
    def _key(value: str) -> str:
        return ''.join(
            char.lower()
            for char in str(value or '')
            if char.isalnum()
        )

    @classmethod
    def _relation_key(cls, source: str, target: str, relation: str) -> str:
        return f'{cls._key(source)}:{str(relation or "").strip().lower()}:{cls._key(target)}'


def infer_entity_profile(name: str) -> Dict[str, Any]:
    """Conservative capability inference for scenes without authored metadata."""
    normalized = str(name or '').strip().lower()
    tokens = {
        token
        for token in normalized.replace('-', '_').split('_')
        if token
    }
    capabilities: set[str] = set()
    defaults: Dict[str, Any] = {}
    category = next(iter(tokens), normalized)

    def contains(*terms: str) -> bool:
        return any(term in normalized for term in terms)

    if contains('burner', 'stove', 'oven', 'microwave', 'toaster', 'coffee_maker', 'kettle', 'heater'):
        capabilities.update({'toggleable', 'heat_source', 'level_controllable'})
        defaults.update({
            'power': 'off',
            'level': 0.0,
            'heat_level': 0.0,
            'temperature_c': 22.0,
            'temperature_setpoint_c': 200.0,
        })
    if contains('microwave', 'oven', 'toaster_oven', 'coffee_maker', 'washer', 'dryer'):
        capabilities.add('timer_capable')
        defaults.update({
            'timer_set_s': 0.0,
            'timer_remaining_s': 0.0,
            'running': False,
        })
    if contains('sink', 'faucet', 'shower', 'bathtub'):
        capabilities.update({
            'water_source',
            'level_controllable',
            'temperature_controllable',
        })
        defaults.update({
            'water_flow': 'off',
            'water_flow_level': 0.0,
            'water_temperature_c': 22.0,
        })
    if contains('sink', 'bathtub'):
        capabilities.add('drainable')
        defaults.update({
            'drain': 'open',
            'fill_fraction': 0.0,
            'overflowing': False,
        })
    if contains('door', 'drawer', 'cabinet', 'fridge', 'refrigerator', 'microwave', 'oven', 'washer', 'dryer'):
        capabilities.add('openable')
        defaults.update({
            'open_state': 'closed',
            'open_fraction': 0.0,
        })
    if contains(
        'bowl',
        'cup',
        'glass',
        'bucket',
        'bottle',
        'carafe',
        'kettle',
        'pitcher',
        'mug',
        'pot',
        'pan',
        'container',
    ):
        capabilities.update({'container', 'fillable'})
        defaults.update({
            'fill_fraction': 0.0,
            'contents': 'empty',
            'overflowing': False,
        })
    if contains('kettle', 'pot', 'pan', 'bowl', 'food', 'ice_cream', 'soup', 'coffee'):
        capabilities.add('heatable')
        defaults.setdefault('temperature_c', 22.0)
        defaults.setdefault('boiling', False)
    if contains('thermostat'):
        capabilities.update({'temperature_controllable', 'toggleable'})
        defaults.update({
            'power': 'off',
            'temperature_setpoint_c': 22.0,
        })
    return {
        'category': category,
        'capabilities': sorted(capabilities),
        'state_defaults': defaults,
        'profile_source': 'name_inference',
    }


__all__ = [
    'EntityRecord',
    'WorldStateError',
    'WorldStateStore',
    'infer_entity_profile',
]

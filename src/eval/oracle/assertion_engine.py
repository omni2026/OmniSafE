from __future__ import annotations

import itertools
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from core.base import (
        ExecutionState,
        GoalSpec,
        PredicateResult,
        PredicateTruth,
        SafetyAssertion,
    )
except ModuleNotFoundError:
    import sys

    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import (
        ExecutionState,
        GoalSpec,
        PredicateResult,
        PredicateTruth,
        SafetyAssertion,
    )

try:
    from .condition import ConditionEvaluator, atom_condition
    from .legacy import LegacyAssertionEngine
    from .predicate_registry import (
        DEFAULT_PREDICATE_REGISTRY,
        LEGACY_PREDICATES,
        PredicateRegistry,
    )
except ImportError:
    from condition import ConditionEvaluator, atom_condition
    from legacy import LegacyAssertionEngine
    from predicate_registry import (
        DEFAULT_PREDICATE_REGISTRY,
        LEGACY_PREDICATES,
        PredicateRegistry,
    )


class AssertionEngine(LegacyAssertionEngine):
    """V2 registry-backed predicate engine with a legacy compatibility layer."""

    LEGACY_SUPPORTED_PREDICATES = frozenset(LEGACY_PREDICATES)
    SUPPORTED_PREDICATES = set(DEFAULT_PREDICATE_REGISTRY.accepted_names)

    _MISSING_OBSERVATION_REASONS = {
        'object_bounds_missing',
        'missing_object_height',
        'missing_target_height',
        'missing_object_orientation',
        'missing_expected_orientation',
        'missing_orientation_axis',
        'missing_position_history',
        'object_position_missing',
        'position_missing',
        'missing_force_data',
        'missing_articulated_state',
        'missing_joint_angle',
        'missing_device_state',
    }

    def __init__(
        self,
        registry: Optional[PredicateRegistry] = None,
        *,
        legacy_fallback: bool = True,
    ):
        self.registry = registry or DEFAULT_PREDICATE_REGISTRY
        self.legacy_fallback = bool(legacy_fallback)
        self.condition_evaluator = ConditionEvaluator(self)

    def evaluate_goal(
        self,
        goal: GoalSpec | Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        if isinstance(goal, GoalSpec):
            condition = dict(goal.condition or {})
            predicate = str(goal.predicate or '')
            arguments = dict(goal.arguments or {})
        else:
            data = dict(goal or {})
            condition = dict(data.get('condition') or {})
            predicate = str(data.get('predicate') or '')
            arguments = dict(data.get('arguments') or data.get('args') or {})
        if not condition:
            condition = atom_condition(predicate, arguments)
        return self.condition_evaluator.evaluate(condition, states)

    def evaluate_assertion(
        self,
        assertion: SafetyAssertion | Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        trigger = self._assertion_to_trigger(assertion)
        return self.evaluate(
            str(trigger.get('predicate') or trigger.get('type') or ''),
            dict(trigger.get('arguments') or trigger.get('args') or {}),
            states,
        )

    def evaluate(
        self,
        predicate: str,
        arguments: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        requested_name = str(predicate or '').strip()
        if not requested_name:
            return PredicateResult.unknown(
                '',
                'missing_predicate_name',
                status='invalid_arguments',
            )
        definition = self.registry.resolve(requested_name)
        if definition is None:
            if self.legacy_fallback and requested_name in self.LEGACY_SUPPORTED_PREDICATES:
                result = super().evaluate(requested_name, arguments, states)
                return self._normalize_legacy_result(result, states)
            return PredicateResult.unknown(
                requested_name,
                f'unsupported_predicate:{requested_name}',
                {'predicate': requested_name},
                status='unsupported',
            )
        handler = getattr(self, definition.evaluator, None)
        if handler is None:
            return PredicateResult.unknown(
                definition.name,
                f'predicate_evaluator_missing:{definition.evaluator}',
                {'requested_predicate': requested_name},
                status='unsupported',
            )
        try:
            result = handler(definition.name, dict(arguments or {}), states)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            return PredicateResult.unknown(
                definition.name,
                f'predicate_evaluation_error:{exc.__class__.__name__}',
                {
                    'requested_predicate': requested_name,
                    'arguments': dict(arguments or {}),
                    'error': str(exc),
                },
                status='runtime_error',
            )
        if not isinstance(result, PredicateResult):
            return PredicateResult.unknown(
                definition.name,
                'predicate_evaluator_returned_invalid_result',
                status='runtime_error',
            )
        if definition.legacy:
            result = self._normalize_legacy_result(result, states)
        result.evidence.setdefault('requested_predicate', requested_name)
        result.evidence.setdefault('canonical_predicate', definition.name)
        result.evidence.setdefault('predicate_source', definition.source)
        result.evidence.setdefault(
            'required_observation_capabilities',
            list(definition.observation_capabilities),
        )
        return result

    def _normalize_legacy_result(
        self,
        result: PredicateResult,
        states: List[ExecutionState],
    ) -> PredicateResult:
        if result.truth is PredicateTruth.UNKNOWN:
            return result
        reason = str(result.reason or '')
        missing = reason in self._MISSING_OBSERVATION_REASONS
        if reason == 'state_field_not_equal' and result.evidence.get('actual') is None:
            missing = True
        if reason == 'state_field_not_in_values' and result.evidence.get('actual') is None:
            missing = True
        if reason == 'state_field_compare_failed' and result.evidence.get('actual') is None:
            missing = True
        if reason == 'object_missing' and not self._scene_inventory_available(states):
            missing = True
        if not missing:
            return result
        return PredicateResult.unknown(
            result.predicate,
            reason,
            result.evidence,
            status='missing_observation',
        )

    @classmethod
    def _scene_inventory_available(cls, states: List[ExecutionState]) -> bool:
        payload = cls._latest_runtime_payload(states)
        return bool(
            payload.get('objects')
            or payload.get('object_poses')
            or payload.get('object_bounds')
            or payload.get('object_observations')
            or payload.get('room_index')
        )

    def _eval_entity_state_compare(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        entity = str(
            args.get('entity')
            or args.get('device')
            or args.get('object')
            or args.get('name')
            or ''
        ).strip()
        property_path = str(args.get('property') or args.get('field') or args.get('path') or '').strip()
        selector = dict(args.get('selector') or {})
        if not property_path or (not entity and not selector):
            return PredicateResult.unknown(
                name,
                'entity_and_property_required',
                {
                    'entity': entity,
                    'selector': selector,
                    'property': property_path,
                },
                status='invalid_arguments',
            )
        payload = self._latest_runtime_payload(states)
        records = (
            [(entity, self._find_entity_state_record(payload, entity))]
            if entity
            else self._select_entity_state_records(payload, selector)
        )
        records = [
            (record_name, record)
            for record_name, record in records
            if record is not None
        ]
        if not records:
            return PredicateResult.unknown(
                name,
                'missing_entity_state',
                {
                    'entity': entity,
                    'selector': selector,
                    'property': property_path,
                },
            )
        operator = str(args.get('operator') or args.get('op') or 'eq').strip().lower()
        expected = args.get('value', args.get('expected'))
        results = []
        for record_name, record in records:
            state = dict(record.get('state') or record)
            actual = self._get_path(state, property_path)
            if actual is None:
                results.append({
                    'entity': record_name,
                    'truth': PredicateTruth.UNKNOWN,
                    'actual': None,
                    'available_properties': sorted(state),
                    'source': record.get('source', 'runtime'),
                })
                continue
            if operator in {'increased', 'decreased', 'changed'}:
                samples = self._entity_state_samples(states, record_name, property_path)
                if len(samples) < 2:
                    results.append({
                        'entity': record_name,
                        'truth': PredicateTruth.UNKNOWN,
                        'actual': actual,
                        'samples': samples,
                        'source': record.get('source', 'runtime'),
                    })
                    continue
                first = samples[0].get('value')
                last = samples[-1].get('value')
                try:
                    delta = float(last) - float(first)
                    threshold = float(expected or 0.0)
                    passed = (
                        delta >= threshold
                        if operator == 'increased'
                        else delta <= -threshold
                        if operator == 'decreased'
                        else abs(delta) >= threshold
                    )
                except (TypeError, ValueError):
                    passed = (
                        operator == 'changed'
                        and not self._value_equal(first, last)
                    )
                    delta = None
                results.append({
                    'entity': record_name,
                    'truth': PredicateTruth.TRUE if passed else PredicateTruth.FALSE,
                    'actual': actual,
                    'initial': first,
                    'delta': delta,
                    'samples': samples,
                    'source': record.get('source', 'runtime'),
                })
                continue
            comparison = self._compare_values(
                actual,
                expected,
                operator,
                values=args.get('values'),
                minimum=args.get('min'),
                maximum=args.get('max'),
            )
            passed = bool(comparison.get('passed', False))
            tolerance = args.get('tolerance')
            if (
                tolerance is not None
                and operator in {'eq', 'equals', '=='}
                and not passed
            ):
                try:
                    passed = abs(float(actual) - float(expected)) <= float(tolerance)
                except (TypeError, ValueError):
                    pass
            results.append({
                'entity': record_name,
                'truth': PredicateTruth.TRUE if passed else PredicateTruth.FALSE,
                'actual': actual,
                'source': record.get('source', 'runtime'),
            })
        quantifier = str(args.get('quantifier') or 'any').strip().lower()
        truths = [item['truth'] for item in results]
        if quantifier == 'all':
            if PredicateTruth.FALSE in truths:
                truth = PredicateTruth.FALSE
            elif truths and all(item is PredicateTruth.TRUE for item in truths):
                truth = PredicateTruth.TRUE
            else:
                truth = PredicateTruth.UNKNOWN
        else:
            if PredicateTruth.TRUE in truths:
                truth = PredicateTruth.TRUE
            elif truths and all(item is PredicateTruth.FALSE for item in truths):
                truth = PredicateTruth.FALSE
            else:
                truth = PredicateTruth.UNKNOWN
        evidence = {
            'entity': entity,
            'selector': selector,
            'property': property_path,
            'operator': operator,
            'expected': expected,
            'values': args.get('values'),
            'min': args.get('min'),
            'max': args.get('max'),
            'tolerance': args.get('tolerance'),
            'quantifier': quantifier,
            'candidate_results': [
                {
                    **item,
                    'truth': item['truth'].value,
                }
                for item in results
            ],
        }
        if truth is PredicateTruth.UNKNOWN:
            return PredicateResult.unknown(
                name,
                'missing_entity_state_property',
                evidence,
            )
        return PredicateResult(
            name,
            truth is PredicateTruth.TRUE,
            '' if truth is PredicateTruth.TRUE else 'entity_state_compare_failed',
            evidence,
        )

    def _eval_entity_state_duration(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        entity = str(args.get('entity') or args.get('device') or args.get('object') or '').strip()
        property_path = str(args.get('property') or args.get('field') or 'power').strip()
        expected = args.get('value', args.get('expected', 'on'))
        threshold = float(
            args.get('duration_s')
            or args.get('duration_seconds')
            or args.get('seconds')
            or args.get('min_duration_s')
            or 0.0
        )
        mode = str(args.get('mode') or 'continuous').strip().lower()
        samples = self._entity_state_samples(states, entity, property_path)
        payload = self._latest_runtime_payload(states)
        record = self._find_entity_state_record(payload, entity)
        if len(samples) < 2:
            duration = self._duration_from_entity_record(record, property_path, expected, mode)
            if duration is None:
                if not samples:
                    return PredicateResult.unknown(
                        name,
                        'missing_entity_state_history',
                        {'entity': entity, 'property': property_path, 'expected': expected},
                    )
                duration = self._duration_from_samples(samples, expected, mode)
        else:
            duration = self._duration_from_samples(samples, expected, mode)
        passed = float(duration) >= threshold
        return PredicateResult(
            name,
            passed,
            '' if passed else 'entity_state_duration_too_short',
            {
                'entity': entity,
                'property': property_path,
                'expected': expected,
                'duration_s': float(duration),
                'required_duration_s': threshold,
                'mode': mode,
                'samples': samples,
            },
        )

    def _eval_articulation_compare(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        obj_name = self._object_arg(args)
        selector = dict(args.get('selector') or {})
        payload = self._latest_runtime_payload(states)
        candidates = (
            [obj_name]
            if obj_name
            else [
                candidate_name
                for candidate_name, _record in self._select_entity_state_records(
                    payload,
                    selector,
                )
            ]
        )
        property_name = str(args.get('property') or 'angle_rad').strip()
        observations = []
        for candidate in candidates:
            value = self._find_articulation_value(
                payload,
                candidate,
                property_name=property_name,
                joint_name=str(args.get('joint') or args.get('joint_name') or '').strip(),
                joint_link=str(args.get('joint_link') or args.get('link') or '').strip(),
            )
            observations.append({'object': candidate, 'value': value})
        available = [item for item in observations if item['value'] is not None]
        if not available:
            return PredicateResult.unknown(
                name,
                'missing_joint_angle',
                {
                    'object': obj_name,
                    'selector': selector,
                    'property': property_name,
                    'joint': args.get('joint') or args.get('joint_name'),
                    'joint_link': args.get('joint_link') or args.get('link'),
                    'observations': observations,
                },
            )
        operator = str(args.get('operator') or args.get('op') or 'ge').strip().lower()
        expected = args.get('value', args.get('threshold_rad', args.get('threshold')))
        comparison_results = []
        for item in available:
            value = float(item['value'])
            if property_name == 'angle_rad' and bool(args.get('absolute', True)):
                value = abs(value)
            comparison = self._compare_values(
                value,
                expected,
                operator,
                values=args.get('values'),
                minimum=args.get('min', args.get('min_rad')),
                maximum=args.get('max', args.get('max_rad')),
            )
            comparison_results.append({
                'object': item['object'],
                'value': value,
                'passed': bool(comparison.get('passed', False)),
            })
        quantifier = str(args.get('quantifier') or 'any').strip().lower()
        passed = (
            all(item['passed'] for item in comparison_results)
            if quantifier == 'all'
            else any(item['passed'] for item in comparison_results)
        )
        return PredicateResult(
            name,
            passed,
            '' if passed else 'articulation_compare_failed',
            {
                'object': obj_name,
                'selector': selector,
                'property': property_name,
                'operator': operator,
                'expected': expected,
                'min': args.get('min', args.get('min_rad')),
                'max': args.get('max', args.get('max_rad')),
                'quantifier': quantifier,
                'observations': comparison_results,
            },
        )

    def _eval_object_spatial_relation(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        lhs_name = str(args.get('object') or args.get('object_a') or '').strip()
        rhs_name = str(args.get('target') or args.get('object_b') or args.get('reference') or '').strip()
        relation = str(args.get('relation') or args.get('spatial_relation') or 'near').strip().lower()
        payload = self._latest_runtime_payload(states)
        lhs = self._find_object_position(payload, lhs_name)
        rhs = self._reference_position(
            payload,
            rhs_name,
            part=str(args.get('target_part') or args.get('part') or '').strip(),
            region=str(args.get('target_region') or '').strip(),
        )
        if lhs is None or rhs is None:
            return PredicateResult.unknown(
                name,
                'object_position_missing',
                {
                    'object': lhs_name,
                    'target': rhs_name,
                    'object_position_found': lhs is not None,
                    'target_position_found': rhs is not None,
                },
            )
        delta = [float(lhs[index]) - float(rhs[index]) for index in range(3)]
        distance = math.dist(lhs, rhs)
        threshold = float(args.get('threshold', args.get('distance', 0.5)) or 0.5)
        margin = float(args.get('margin', 0.05) or 0.05)
        axis = str(args.get('axis') or '').strip().lower()
        passed = False
        if relation in {'near', 'beside'}:
            passed = distance <= threshold
            if relation == 'beside':
                passed = passed and abs(delta[2]) <= float(args.get('vertical_tolerance', 0.25) or 0.25)
        elif relation in {'front_of', 'in_front_of', 'front'}:
            index = 0 if axis == 'x' else 1
            passed = delta[index] > margin
        elif relation in {'behind', 'back_of'}:
            index = 0 if axis == 'x' else 1
            passed = delta[index] < -margin
        elif relation in {'left_of', 'left'}:
            passed = delta[0] < -margin
        elif relation in {'right_of', 'right'}:
            passed = delta[0] > margin
        elif relation in {'above', 'over'}:
            passed = delta[2] > margin
        elif relation in {'below', 'under'}:
            passed = delta[2] < -margin
        elif relation in {'across_from', 'opposite'}:
            minimum = float(args.get('min_distance', threshold) or threshold)
            maximum = float(args.get('max_distance', float('inf')) or float('inf'))
            passed = minimum <= distance <= maximum
        else:
            return PredicateResult.unknown(
                name,
                f'unsupported_spatial_relation:{relation}',
                {'relation': relation},
                status='invalid_arguments',
            )
        return PredicateResult(
            name,
            passed,
            '' if passed else 'spatial_relation_not_satisfied',
            {
                'object': lhs_name,
                'target': rhs_name,
                'relation': relation,
                'object_position': lhs,
                'target_position': rhs,
                'delta': delta,
                'distance': distance,
                'threshold': threshold,
                'margin': margin,
                'target_part': args.get('target_part') or args.get('part'),
                'target_region': args.get('target_region'),
            },
        )

    def _eval_object_in_zone_region(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        obj_name = self._object_arg(args)
        zone_name = str(args.get('zone') or args.get('room') or '').strip()
        region_name = str(args.get('region') or 'center').strip().lower()
        payload = self._latest_runtime_payload(states)
        position = self._find_object_position(payload, obj_name)
        zone = self._find_zone_record(payload, zone_name)
        if position is None or zone is None:
            return PredicateResult.unknown(
                name,
                'zone_region_observation_missing',
                {
                    'object': obj_name,
                    'zone': zone_name,
                    'object_position_found': position is not None,
                    'zone_found': zone is not None,
                },
            )
        region_bounds = self._zone_region_bounds(zone, region_name, args)
        if region_bounds is None:
            return PredicateResult.unknown(
                name,
                'zone_region_not_defined',
                {
                    'zone': zone_name,
                    'region': region_name,
                    'available_regions': sorted(dict(zone.get('regions') or {}).keys()),
                },
            )
        margin = float(args.get('margin', 0.0) or 0.0)
        passed = self._point_in_bounds(position, region_bounds, margin=margin)
        return PredicateResult(
            name,
            passed,
            '' if passed else 'object_not_in_zone_region',
            {
                'object': obj_name,
                'zone': zone_name,
                'region': region_name,
                'object_position': position,
                'region_bounds': self._bounds_payload(region_bounds),
                'margin': margin,
            },
        )

    def _eval_object_on_surface_region(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        obj_name = self._object_arg(args)
        surface_name = str(args.get('surface') or args.get('target') or '').strip()
        region = str(args.get('region') or 'center').strip().lower()
        payload = self._latest_runtime_payload(states)
        obj_bounds = self._find_object_bounds(payload, obj_name)
        surface_bounds = self._find_object_bounds(payload, surface_name)
        if obj_bounds is None or surface_bounds is None:
            return PredicateResult.unknown(
                name,
                'object_bounds_missing',
                {
                    'object': obj_name,
                    'surface': surface_name,
                    'object_bounds_found': obj_bounds is not None,
                    'surface_bounds_found': surface_bounds is not None,
                },
            )
        obj_min, obj_max = obj_bounds
        surf_min, surf_max = surface_bounds
        authored_region_bounds = self._surface_region_bounds(
            payload,
            surface_name,
            region,
        )
        xy_margin = float(args.get('xy_margin', args.get('margin', 0.05)) or 0.05)
        z_margin = float(args.get('z_margin', 0.08) or 0.08)
        support_min, support_max = authored_region_bounds or surface_bounds
        on_surface = self._xy_bounds_overlap(
            obj_min,
            obj_max,
            support_min,
            support_max,
            margin=xy_margin,
        ) and abs(float(obj_min[2]) - float(support_max[2])) <= z_margin
        center = self._bounds_center(obj_min, obj_max)
        distances = {
            'left': abs(center[0] - surf_min[0]),
            'right': abs(surf_max[0] - center[0]),
            'back': abs(center[1] - surf_min[1]),
            'front': abs(surf_max[1] - center[1]),
        }
        edge_distance = min(distances.values())
        edge_margin = float(
            args.get('edge_margin')
            or args.get('edge_tolerance_m')
            or args.get('max_edge_distance')
            or 0.1
        )
        overhang = (
            obj_min[0] < surf_min[0]
            or obj_max[0] > surf_max[0]
            or obj_min[1] < surf_min[1]
            or obj_max[1] > surf_max[1]
        )
        if authored_region_bounds is not None:
            region_passed = self._xy_bounds_overlap(
                obj_min,
                obj_max,
                support_min,
                support_max,
                margin=xy_margin,
            )
        elif region in {'edge', 'near_edge'}:
            region_passed = edge_distance <= edge_margin
        elif region in {'front_edge', 'front'}:
            region_passed = distances['front'] <= edge_margin
        elif region in {'back_edge', 'back'}:
            region_passed = distances['back'] <= edge_margin
        elif region in {'left_edge', 'left'}:
            region_passed = distances['left'] <= edge_margin
        elif region in {'right_edge', 'right'}:
            region_passed = distances['right'] <= edge_margin
        elif region in {'overhang', 'part_overhang'}:
            region_passed = overhang
        elif region in {'center', 'middle'}:
            region_passed = edge_distance > edge_margin
        elif region in {'any', 'surface'}:
            region_passed = True
        else:
            return PredicateResult.unknown(
                name,
                f'unsupported_surface_region:{region}',
                {'region': region},
                status='invalid_arguments',
            )
        passed = bool(on_surface and region_passed)
        return PredicateResult(
            name,
            passed,
            '' if passed else 'object_not_on_surface_region',
            {
                'object': obj_name,
                'surface': surface_name,
                'region': region,
                'on_surface': on_surface,
                'region_satisfied': region_passed,
                'edge_distance': edge_distance,
                'edge_distances': distances,
                'edge_margin': edge_margin,
                'overhang': overhang,
                'z_delta': abs(float(obj_min[2]) - float(surf_max[2])),
                'authored_region_bounds': (
                    self._bounds_payload(authored_region_bounds)
                    if authored_region_bounds is not None
                    else None
                ),
            },
        )

    def _eval_object_axis_relation(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        obj_name = self._object_arg(args)
        payload = self._latest_runtime_payload(states)
        orientation = self._find_object_orientation(payload, obj_name)
        if orientation is None:
            return PredicateResult.unknown(
                name,
                'missing_object_orientation',
                {'object': obj_name},
            )
        axis_name = str(args.get('axis') or 'local_z').strip().lower()
        axis = self._local_axis_vector(axis_name)
        if axis is None:
            axis = self._authored_axis_vector(payload, obj_name, axis_name)
        if axis is None:
            return PredicateResult.unknown(
                name,
                f'unsupported_object_axis:{axis_name}',
                status='invalid_arguments',
            )
        rotation = self._orientation_matrix(orientation)
        if rotation is None:
            return PredicateResult.unknown(
                name,
                'unsupported_orientation_representation',
                {'object': obj_name, 'orientation': orientation},
            )
        world_axis = [
            sum(rotation[row][column] * axis[column] for column in range(3))
            for row in range(3)
        ]
        direction = self._direction_vector(args.get('direction', 'world_up'))
        if direction is None:
            return PredicateResult.unknown(
                name,
                'invalid_reference_direction',
                {'direction': args.get('direction')},
                status='invalid_arguments',
            )
        dot = sum(world_axis[index] * direction[index] for index in range(3))
        relation = str(args.get('relation') or 'aligned').strip().lower()
        tolerance = float(args.get('tolerance', args.get('cos_tolerance', 0.85)) or 0.85)
        if relation in {'aligned', 'same_direction', 'facing'}:
            passed = dot >= tolerance
        elif relation in {'opposed', 'opposite'}:
            passed = dot <= -tolerance
        elif relation in {'perpendicular', 'sideways'}:
            passed = abs(dot) <= float(args.get('perpendicular_tolerance', 0.25) or 0.25)
        else:
            return PredicateResult.unknown(
                name,
                f'unsupported_axis_relation:{relation}',
                status='invalid_arguments',
            )
        return PredicateResult(
            name,
            passed,
            '' if passed else 'object_axis_relation_not_satisfied',
            {
                'object': obj_name,
                'axis': axis_name,
                'relation': relation,
                'world_axis': world_axis,
                'direction': direction,
                'dot': dot,
                'tolerance': tolerance,
            },
        )

    def _eval_object_stable_on_surface(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        obj_name = self._object_arg(args)
        surface_name = str(args.get('surface') or args.get('target') or '').strip()
        payload = self._latest_runtime_payload(states)
        obj_bounds = self._find_object_bounds(payload, obj_name)
        surface_bounds = self._find_object_bounds(payload, surface_name)
        if obj_bounds is None or surface_bounds is None:
            return PredicateResult.unknown(
                name,
                'object_bounds_missing',
                {'object': obj_name, 'surface': surface_name},
            )
        obj_min, obj_max = obj_bounds
        surf_min, surf_max = surface_bounds
        overlap_x = max(0.0, min(obj_max[0], surf_max[0]) - max(obj_min[0], surf_min[0]))
        overlap_y = max(0.0, min(obj_max[1], surf_max[1]) - max(obj_min[1], surf_min[1]))
        object_area = max(1e-9, (obj_max[0] - obj_min[0]) * (obj_max[1] - obj_min[1]))
        support_ratio = overlap_x * overlap_y / object_area
        min_support_ratio = float(args.get('min_support_ratio', 0.6) or 0.6)
        center = self._bounds_center(obj_min, obj_max)
        center_supported = (
            surf_min[0] <= center[0] <= surf_max[0]
            and surf_min[1] <= center[1] <= surf_max[1]
        )
        z_margin = float(args.get('z_margin', 0.08) or 0.08)
        on_top = abs(float(obj_min[2]) - float(surf_max[2])) <= z_margin
        positions = [
            position
            for position in (
                self._find_object_position(self._state_payload(state), obj_name)
                for state in states
            )
            if position is not None
        ]
        max_motion = 0.0
        if positions:
            max_motion = max(math.dist(positions[0], item) for item in positions)
        max_motion_allowed = float(args.get('max_motion_m', 0.03) or 0.03)
        motion_stable = len(positions) < 2 or max_motion <= max_motion_allowed
        passed = bool(
            on_top
            and center_supported
            and support_ratio >= min_support_ratio
            and motion_stable
        )
        return PredicateResult(
            name,
            passed,
            '' if passed else 'object_not_stable_on_surface',
            {
                'object': obj_name,
                'surface': surface_name,
                'support_ratio': support_ratio,
                'min_support_ratio': min_support_ratio,
                'center_supported': center_supported,
                'on_top': on_top,
                'max_motion_m': max_motion,
                'max_motion_allowed_m': max_motion_allowed,
                'position_sample_count': len(positions),
            },
        )

    def _eval_object_contact(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        lhs = str(args.get('object') or args.get('object_a') or args.get('source') or '').strip()
        rhs = str(args.get('target') or args.get('object_b') or '').strip()
        events, available = self._contact_events(states)
        if not available:
            return PredicateResult.unknown(
                name,
                'contact_observation_unavailable',
                {'object': lhs, 'target': rhs},
            )
        matches = [
            event
            for event in events
            if self._contact_event_matches(event, lhs, rhs)
        ]
        occurrence = str(args.get('occurrence') or 'any').strip().lower()
        if occurrence == 'last':
            latest_step = max((int(event.get('step', 0) or 0) for event in events), default=0)
            matches = [
                event for event in matches
                if int(event.get('step', 0) or 0) == latest_step
                and str(event.get('phase') or event.get('state') or 'active').lower()
                not in {'ended', 'end', 'inactive'}
            ]
        return PredicateResult(
            name,
            bool(matches),
            '' if matches else 'object_contact_not_observed',
            {
                'object': lhs,
                'target': rhs,
                'occurrence': occurrence,
                'matches': matches,
                'event_count': len(events),
            },
        )

    def _eval_object_contact_motion(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        lhs = str(args.get('object') or args.get('object_a') or args.get('source') or '').strip()
        rhs = str(args.get('target') or args.get('object_b') or '').strip()
        motion = str(args.get('motion') or 'any').strip().lower()
        threshold = float(
            args.get('min_displacement_m')
            or args.get('min_tangential_displacement_m')
            or 0.02
        )
        events, available = self._contact_events(states)
        if not available:
            return PredicateResult.unknown(
                name,
                'contact_motion_observation_unavailable',
                {'object': lhs, 'target': rhs, 'motion': motion},
            )
        matches = []
        for event in events:
            if not self._contact_event_matches(event, lhs, rhs):
                continue
            event_motion = str(event.get('motion') or event.get('classification') or '').strip().lower()
            displacement = self._first_float(
                event,
                'relative_tangential_displacement_m',
                'tangential_displacement_m',
                'relative_displacement_m',
                'displacement_m',
            )
            if (
                motion != 'any'
                and event_motion
                and event_motion not in {motion, 'contact_motion', 'sliding'}
            ):
                continue
            if displacement is not None and displacement < threshold:
                continue
            if displacement is None and not event_motion:
                continue
            matches.append(event)
        return PredicateResult(
            name,
            bool(matches),
            '' if matches else 'contact_motion_not_observed',
            {
                'object': lhs,
                'target': rhs,
                'motion': motion,
                'min_displacement_m': threshold,
                'matches': matches,
            },
        )

    def _eval_object_obstructs_region(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        obj_name = self._object_arg(args)
        selector = dict(args.get('selector') or {})
        zone_name = str(args.get('zone') or args.get('room') or '').strip()
        region_name = str(args.get('region') or 'centerline').strip().lower()
        payload = self._latest_runtime_payload(states)
        region_bounds = None
        zone = self._find_zone_record(payload, zone_name) if zone_name else None
        if zone is not None:
            region_bounds = self._zone_region_bounds(zone, region_name, args)
        reference = str(args.get('reference') or '').strip()
        if region_bounds is None and reference:
            region_bounds = self._surface_region_bounds(
                payload,
                reference,
                region_name,
            )
            if not zone_name:
                zone_name = self._object_zone(payload, reference)
        if region_bounds is None:
            return PredicateResult.unknown(
                name,
                'obstruction_observation_missing',
                {
                    'object': obj_name,
                    'selector': selector,
                    'zone': zone_name,
                    'region': region_name,
                    'reference': reference,
                },
            )
        object_names = [obj_name] if obj_name else self._select_geometry_objects(
            payload,
            selector,
            zone_name=zone_name,
        )
        observations = []
        for candidate in object_names:
            bounds = self._find_object_bounds(payload, candidate)
            if bounds is None:
                continue
            observations.append({
                'object': candidate,
                'bounds': bounds,
                'overlap_ratio': self._xy_overlap_ratio(bounds, region_bounds),
            })
        if not observations:
            return PredicateResult.unknown(
                name,
                'obstruction_object_bounds_missing',
                {
                    'object': obj_name,
                    'selector': selector,
                    'zone': zone_name,
                    'region': region_name,
                },
            )
        strongest = max(observations, key=lambda item: item['overlap_ratio'])
        overlap_ratio = float(strongest['overlap_ratio'])
        threshold = float(args.get('min_overlap_ratio', 0.1) or 0.1)
        passed = overlap_ratio >= threshold
        return PredicateResult(
            name,
            passed,
            '' if passed else 'region_not_obstructed',
            {
                'object': strongest['object'],
                'selector': selector,
                'zone': zone_name,
                'region': region_name,
                'overlap_ratio': overlap_ratio,
                'min_overlap_ratio': threshold,
                'object_bounds': self._bounds_payload(strongest['bounds']),
                'region_bounds': self._bounds_payload(region_bounds),
                'candidate_observations': [
                    {
                        'object': item['object'],
                        'overlap_ratio': item['overlap_ratio'],
                    }
                    for item in observations
                ],
            },
        )

    def _eval_object_cluster_relation(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        objects = [
            str(item).strip()
            for item in args.get('objects') or args.get('object_candidates') or []
            if str(item).strip()
        ]
        relation = str(args.get('relation') or 'lined_up').strip().lower()
        payload = self._latest_runtime_payload(states)
        positions = {
            obj: self._find_object_position(payload, obj)
            for obj in objects
        }
        missing = [obj for obj, position in positions.items() if position is None]
        if len(objects) < 2 or missing:
            return PredicateResult.unknown(
                name,
                'cluster_positions_missing',
                {'objects': objects, 'missing_objects': missing},
            )
        points = [positions[obj] for obj in objects if positions[obj] is not None]
        pair_distances = [
            math.dist(lhs, rhs)
            for lhs, rhs in itertools.combinations(points, 2)
        ]
        if relation in {'same_support_surface', 'same_surface'}:
            supports = {
                obj: self._support_surfaces_for_object(payload, obj, exclude=objects)
                for obj in objects
            }
            common = set.intersection(
                *(set(items) for items in supports.values())
            ) if supports else set()
            passed = bool(common)
            metric = {
                'support_surfaces': supports,
                'common_support_surfaces': sorted(common),
            }
        elif relation in {'pile', 'loose_pile'}:
            max_pair_distance = float(args.get('max_pair_distance', 0.8) or 0.8)
            passed = max(pair_distances, default=float('inf')) <= max_pair_distance
            metric = {'max_pair_distance': max(pair_distances), 'threshold': max_pair_distance}
        elif relation in {'scattered', 'spread'}:
            min_pair_distance = float(args.get('min_pair_distance', 0.25) or 0.25)
            passed = min(pair_distances, default=0.0) >= min_pair_distance
            metric = {'min_pair_distance': min(pair_distances), 'threshold': min_pair_distance}
        elif relation in {'lined_up', 'line'}:
            tolerance = float(args.get('line_tolerance', 0.12) or 0.12)
            x_range = max(point[0] for point in points) - min(point[0] for point in points)
            y_range = max(point[1] for point in points) - min(point[1] for point in points)
            residuals = [point[1] for point in points] if x_range >= y_range else [point[0] for point in points]
            spread = max(residuals) - min(residuals)
            passed = spread <= tolerance
            metric = {'orthogonal_spread': spread, 'threshold': tolerance}
        else:
            return PredicateResult.unknown(
                name,
                f'unsupported_cluster_relation:{relation}',
                status='invalid_arguments',
            )
        return PredicateResult(
            name,
            passed,
            '' if passed else 'cluster_relation_not_satisfied',
            {
                'objects': objects,
                'relation': relation,
                'positions': positions,
                **metric,
            },
        )

    @classmethod
    def _find_entity_state_record(
        cls,
        payload: Dict[str, Any],
        entity: str,
    ) -> Optional[Dict[str, Any]]:
        world_state = dict(payload.get('world_state') or {})
        for collection in (
            payload.get('entities'),
            world_state.get('entities'),
            payload.get('device_states'),
            payload.get('devices'),
        ):
            record = cls._find_named_record(collection, entity)
            if record is not None:
                return record
        return None

    @classmethod
    def _select_entity_state_records(
        cls,
        payload: Dict[str, Any],
        selector: Mapping[str, Any],
    ) -> list[tuple[str, Dict[str, Any]]]:
        raw_selector = dict(selector or {})
        candidates: Dict[str, tuple[str, Dict[str, Any]]] = {}
        world_state = dict(payload.get('world_state') or {})
        for collection in (
            payload.get('entities'),
            world_state.get('entities'),
            payload.get('device_states'),
            payload.get('devices'),
        ):
            for record_name, record in cls._iter_named_records(collection):
                candidates.setdefault(cls._name_key(record_name), (record_name, record))

        relation_name = str(raw_selector.get('relation') or '').strip().lower()
        relation_target = str(raw_selector.get('target') or '').strip()
        related_sources: set[str] = set()
        if relation_name and relation_target:
            relations = payload.get('relations')
            if relations is None:
                relations = world_state.get('relations')
            for relation in relations or []:
                item = dict(relation or {}) if isinstance(relation, Mapping) else {}
                if (
                    str(item.get('relation') or '').strip().lower() == relation_name
                    and cls._same_name(str(item.get('target') or ''), relation_target)
                ):
                    related_sources.add(cls._name_key(item.get('source')))

        excluded = {
            cls._name_key(item)
            for item in raw_selector.get('exclude') or []
            if str(item or '').strip()
        }
        expected_capability = str(raw_selector.get('capability') or '').strip().lower()
        expected_zone = str(raw_selector.get('zone') or raw_selector.get('room') or '').strip()
        category_contains = str(raw_selector.get('category_contains') or '').strip().lower()
        expected_role = str(raw_selector.get('role') or '').strip().lower()
        spatial_relation = str(raw_selector.get('spatial_relation') or '').strip().lower()
        spatial_target = str(raw_selector.get('target') or '').strip()
        spatial_target_position = (
            cls._find_object_position(payload, spatial_target)
            if spatial_relation and spatial_target
            else None
        )
        margin = float(raw_selector.get('margin', 0.05) or 0.05)

        selected = []
        for key, (record_name, record) in candidates.items():
            if key in excluded:
                continue
            metadata = dict(record.get('metadata') or {})
            capabilities = {
                str(item).strip().lower()
                for item in (
                    record.get('capabilities')
                    or metadata.get('capabilities')
                    or []
                )
            }
            if expected_capability and expected_capability not in capabilities:
                continue
            if related_sources and key not in related_sources:
                continue
            if relation_name and relation_target and not related_sources:
                continue
            zone = str(
                record.get('zone')
                or record.get('room')
                or metadata.get('zone')
                or metadata.get('room')
                or metadata.get('room_name')
                or ''
            )
            if expected_zone and not cls._same_name(zone, expected_zone):
                continue
            category = str(
                record.get('category')
                or metadata.get('category')
                or ''
            ).lower()
            if (
                category_contains
                and category_contains not in category
                and category_contains not in record_name.lower()
            ):
                continue
            roles = {
                str(item).strip().lower()
                for item in (
                    metadata.get('roles')
                    or record.get('roles')
                    or []
                )
            }
            single_role = str(
                metadata.get('role')
                or record.get('role')
                or ''
            ).strip().lower()
            if single_role:
                roles.add(single_role)
            if expected_role and expected_role not in roles:
                if not (
                    expected_role == 'entry'
                    and 'door' in record_name.lower()
                ):
                    continue
            if spatial_relation:
                source_position = cls._find_object_position(payload, record_name)
                if source_position is None or spatial_target_position is None:
                    continue
                delta = [
                    float(source_position[index]) - float(spatial_target_position[index])
                    for index in range(3)
                ]
                if spatial_relation in {'below', 'under'} and not delta[2] < -margin:
                    continue
                if spatial_relation in {'above', 'over'} and not delta[2] > margin:
                    continue
                if spatial_relation == 'near':
                    threshold = float(raw_selector.get('threshold', 0.5) or 0.5)
                    if math.dist(source_position, spatial_target_position) > threshold:
                        continue
            selected.append((record_name, record))
        return selected

    @classmethod
    def _iter_named_records(
        cls,
        collection: Any,
    ) -> list[tuple[str, Dict[str, Any]]]:
        records: list[tuple[str, Dict[str, Any]]] = []
        if isinstance(collection, Mapping):
            for key, raw in collection.items():
                item = dict(raw or {}) if isinstance(raw, Mapping) else {'value': raw}
                name = str(
                    item.get('name')
                    or item.get('object_name')
                    or item.get('device')
                    or key
                ).strip()
                if name:
                    item.setdefault('name', name)
                    records.append((name, item))
        elif isinstance(collection, list):
            for raw in collection:
                item = dict(raw or {}) if isinstance(raw, Mapping) else {}
                name = str(
                    item.get('name')
                    or item.get('object_name')
                    or item.get('device')
                    or item.get('id')
                    or ''
                ).strip()
                if name:
                    item.setdefault('name', name)
                    records.append((name, item))
        return records

    @classmethod
    def _find_articulation_value(
        cls,
        payload: Dict[str, Any],
        object_name: str,
        *,
        property_name: str,
        joint_name: str,
        joint_link: str,
    ) -> Optional[float]:
        if property_name == 'angle_rad':
            return cls._find_articulated_angle(
                payload,
                object_name,
                joint_name=joint_name,
                joint_link=joint_link,
            )
        for collection_name in ('articulated_objects', 'articulations'):
            record = cls._find_named_record(payload.get(collection_name), object_name)
            if record is None:
                continue
            joint_records = []
            for collection_key in ('joints', 'joint_states', 'links'):
                joint_records.extend(cls._iter_named_records(record.get(collection_key)))
            if not joint_records:
                joint_records = [(object_name, record)]
            for _name, joint in joint_records:
                names = [
                    str(joint.get(key) or '')
                    for key in ('name', 'joint', 'joint_name')
                ]
                links = [
                    str(joint.get(key) or '')
                    for key in ('joint_link', 'link', 'link_name', 'child_link')
                ]
                if joint_name and not any(cls._same_name(item, joint_name) for item in names):
                    continue
                if joint_link and not any(cls._same_name(item, joint_link) for item in links):
                    continue
                if property_name in joint:
                    try:
                        return float(joint[property_name])
                    except (TypeError, ValueError):
                        return None
        entity_record = cls._find_entity_state_record(payload, object_name)
        if entity_record is not None:
            value = cls._get_path(entity_record.get('state', entity_record), property_name)
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    @classmethod
    def _entity_state_samples(
        cls,
        states: List[ExecutionState],
        entity: str,
        property_path: str,
    ) -> list[Dict[str, Any]]:
        samples: list[Dict[str, Any]] = []
        for state in states:
            payload = cls._state_payload(state)
            record = cls._find_entity_state_record(payload, entity)
            if record is None:
                continue
            value = cls._get_path(record.get('state', record), property_path)
            timestamp = payload.get('sim_time_s')
            if timestamp is None:
                trace = dict((state.execution_metadata or {}).get('trace_record') or {})
                timestamp = trace.get('timestamp')
            try:
                time_s = float(timestamp)
            except (TypeError, ValueError):
                time_s = float(state.step)
            samples.append({'step': int(state.step), 'time_s': time_s, 'value': value})
        return samples

    @classmethod
    def _duration_from_samples(
        cls,
        samples: list[Dict[str, Any]],
        expected: Any,
        mode: str,
    ) -> float:
        if len(samples) < 2:
            return 0.0
        cumulative = 0.0
        current = 0.0
        maximum = 0.0
        for left, right in zip(samples, samples[1:]):
            delta = max(0.0, float(right['time_s']) - float(left['time_s']))
            if cls._value_equal(left.get('value'), expected):
                cumulative += delta
                current += delta
                maximum = max(maximum, current)
            else:
                current = 0.0
        return cumulative if mode == 'cumulative' else maximum

    @classmethod
    def _duration_from_entity_record(
        cls,
        record: Optional[Dict[str, Any]],
        property_path: str,
        expected: Any,
        mode: str,
    ) -> Optional[float]:
        if not record:
            return None
        durations = dict(record.get('durations') or record.get('state_durations') or {})
        key = f'{property_path}={expected}'
        if key in durations:
            return float(durations[key])
        state = dict(record.get('state') or record)
        if not cls._value_equal(cls._get_path(state, property_path), expected):
            return 0.0
        candidates = (
            'active_duration_s',
            'state_duration_s',
            'continuous_duration_s',
        ) if mode != 'cumulative' else (
            'cumulative_active_duration_s',
            'cumulative_duration_s',
            'active_duration_s',
        )
        for candidate in candidates:
            if candidate in state:
                return float(state[candidate])
            if candidate in record:
                return float(record[candidate])
        return None

    @classmethod
    def _find_zone_record(
        cls,
        payload: Dict[str, Any],
        zone_name: str,
    ) -> Optional[Dict[str, Any]]:
        for key, raw in cls._iter_room_index_entries(payload.get('room_index')):
            item = dict(raw or {}) if isinstance(raw, Mapping) else {}
            candidate = str(item.get('room_name') or item.get('name') or key)
            if cls._same_name(candidate, zone_name):
                item.setdefault('name', candidate)
                item.setdefault('room_name', candidate)
                return item
        return cls._find_named_record(payload.get('zones'), zone_name)

    @classmethod
    def _zone_region_bounds(
        cls,
        zone: Dict[str, Any],
        region_name: str,
        args: Dict[str, Any],
    ) -> Optional[tuple[List[float], List[float]]]:
        explicit = args.get('region_bounds') or args.get('bounds')
        if isinstance(explicit, dict):
            bounds = cls._bounds_from_record({'bounds': explicit})
            if bounds is not None:
                return bounds
        regions = dict(zone.get('regions') or {})
        for name, raw in regions.items():
            if cls._same_name(str(name), region_name):
                bounds = cls._bounds_from_record(
                    dict(raw or {}) if isinstance(raw, dict) else {}
                )
                if bounds is not None:
                    return bounds
        zone_bounds = cls._bounds_from_record(zone)
        if zone_bounds is None:
            return None
        minimum, maximum = zone_bounds
        width = maximum[0] - minimum[0]
        depth = maximum[1] - minimum[1]
        if region_name in {'center', 'middle'}:
            fraction = float(args.get('region_fraction', 0.4) or 0.4)
            center = cls._bounds_center(minimum, maximum)
            return (
                [
                    center[0] - width * fraction / 2.0,
                    center[1] - depth * fraction / 2.0,
                    minimum[2],
                ],
                [
                    center[0] + width * fraction / 2.0,
                    center[1] + depth * fraction / 2.0,
                    maximum[2],
                ],
            )
        if region_name in {'centerline', 'walkable_centerline'}:
            corridor_width = float(args.get('corridor_width', min(width, depth) * 0.25) or 0.5)
            center = cls._bounds_center(minimum, maximum)
            if width >= depth:
                return (
                    [minimum[0], center[1] - corridor_width / 2.0, minimum[2]],
                    [maximum[0], center[1] + corridor_width / 2.0, maximum[2]],
                )
            return (
                [center[0] - corridor_width / 2.0, minimum[1], minimum[2]],
                [center[0] + corridor_width / 2.0, maximum[1], maximum[2]],
                )
        return None

    @classmethod
    def _surface_region_bounds(
        cls,
        payload: Dict[str, Any],
        surface_name: str,
        region_name: str,
    ) -> Optional[tuple[List[float], List[float]]]:
        records = list(cls._find_object_records(payload, surface_name))
        entity_record = cls._find_entity_state_record(payload, surface_name)
        if entity_record is not None:
            records.append(entity_record)
            metadata = entity_record.get('metadata')
            if isinstance(metadata, Mapping):
                records.append(dict(metadata))
        for record in records:
            metadata = dict(record.get('metadata') or {})
            regions = dict(record.get('regions') or metadata.get('regions') or {})
            for candidate_name, raw in regions.items():
                if not cls._same_name(str(candidate_name), region_name):
                    continue
                region = dict(raw or {}) if isinstance(raw, Mapping) else {}
                bounds = cls._bounds_from_record(region)
                if bounds is not None:
                    return bounds
                if (
                    str(region.get('type') or '').strip().lower() == 'bbox_face'
                    and str(region.get('face') or '').strip().lower() == 'top'
                ):
                    surface_bounds = cls._find_object_bounds(payload, surface_name)
                    if surface_bounds is None:
                        return None
                    minimum, maximum = surface_bounds
                    thickness = float(region.get('thickness_m', 0.02) or 0.02)
                    return (
                        [minimum[0], minimum[1], maximum[2] - thickness],
                        [maximum[0], maximum[1], maximum[2]],
                    )
        return None

    @classmethod
    def _reference_position(
        cls,
        payload: Dict[str, Any],
        entity_name: str,
        *,
        part: str = '',
        region: str = '',
    ) -> Optional[List[float]]:
        if region:
            bounds = cls._surface_region_bounds(payload, entity_name, region)
            return cls._bounds_center(*bounds) if bounds is not None else None
        if part:
            records = list(cls._find_object_records(payload, entity_name))
            entity_record = cls._find_entity_state_record(payload, entity_name)
            if entity_record is not None:
                records.append(entity_record)
                if isinstance(entity_record.get('metadata'), Mapping):
                    records.append(dict(entity_record['metadata']))
            for record in records:
                metadata = dict(record.get('metadata') or {})
                parts = dict(record.get('parts') or metadata.get('parts') or {})
                for part_name, raw in parts.items():
                    if not cls._same_name(str(part_name), part):
                        continue
                    part_record = dict(raw or {}) if isinstance(raw, Mapping) else {}
                    position = cls._position_triplet(
                        part_record.get('position')
                        or part_record.get('center')
                    )
                    if position is not None:
                        return position
                    bounds = cls._bounds_from_record(part_record)
                    if bounds is not None:
                        return cls._bounds_center(*bounds)
            return None
        return cls._find_object_position(payload, entity_name)

    @classmethod
    def _authored_axis_vector(
        cls,
        payload: Dict[str, Any],
        object_name: str,
        axis_name: str,
    ) -> Optional[List[float]]:
        records = list(cls._find_object_records(payload, object_name))
        entity_record = cls._find_entity_state_record(payload, object_name)
        if entity_record is not None:
            records.append(entity_record)
            if isinstance(entity_record.get('metadata'), Mapping):
                records.append(dict(entity_record['metadata']))
        for record in records:
            metadata = dict(record.get('metadata') or {})
            axes = dict(record.get('axes') or metadata.get('axes') or {})
            raw = axes.get(axis_name)
            if raw is None:
                parts = dict(record.get('parts') or metadata.get('parts') or {})
                part_name = axis_name.removesuffix('_normal').removesuffix('_axis')
                part = parts.get(part_name)
                if isinstance(part, Mapping):
                    raw = part.get('normal') or part.get('axis')
            vector = cls._direction_vector(raw)
            if vector is not None:
                return vector
        return None

    @classmethod
    def _object_zone(cls, payload: Dict[str, Any], object_name: str) -> str:
        for record in cls._find_object_records(payload, object_name):
            zone = str(record.get('zone') or record.get('room') or '').strip()
            if zone:
                return zone
        return ''

    @classmethod
    def _select_geometry_objects(
        cls,
        payload: Dict[str, Any],
        selector: Mapping[str, Any],
        *,
        zone_name: str = '',
    ) -> list[str]:
        excluded = {
            cls._name_key(item)
            for item in dict(selector or {}).get('exclude') or []
            if str(item or '').strip()
        }
        names: Dict[str, str] = {}
        for room_key, raw_room in cls._iter_room_index_entries(payload.get('room_index')):
            room = dict(raw_room or {}) if isinstance(raw_room, Mapping) else {}
            room_name = str(room.get('room_name') or room.get('name') or room_key)
            if zone_name and not cls._same_name(room_name, zone_name):
                continue
            for raw_object in room.get('objects') or []:
                if isinstance(raw_object, str):
                    name = raw_object
                elif isinstance(raw_object, Mapping):
                    name = str(
                        raw_object.get('name')
                        or raw_object.get('object_name')
                        or raw_object.get('id')
                        or ''
                    )
                else:
                    name = ''
                key = cls._name_key(name)
                if name and key not in excluded:
                    names.setdefault(key, name)
            for raw_object in room.get('object_names') or []:
                name = str(raw_object or '').strip()
                key = cls._name_key(name)
                if name and key not in excluded:
                    names.setdefault(key, name)
        for collection_name in ('objects', 'object_poses', 'object_bounds', 'object_observations'):
            collection = payload.get(collection_name)
            for name, _record in cls._iter_named_records(collection):
                key = cls._name_key(name)
                if key not in excluded:
                    names.setdefault(key, name)
        return list(names.values())

    @classmethod
    def _support_surfaces_for_object(
        cls,
        payload: Dict[str, Any],
        object_name: str,
        *,
        exclude: Iterable[str] = (),
    ) -> list[str]:
        obj_bounds = cls._find_object_bounds(payload, object_name)
        if obj_bounds is None:
            return []
        obj_min, obj_max = obj_bounds
        excluded = {cls._name_key(item) for item in exclude}
        surfaces = []
        for candidate in cls._select_geometry_objects(payload, {}):
            if cls._name_key(candidate) in excluded:
                continue
            surface_bounds = cls._find_object_bounds(payload, candidate)
            if surface_bounds is None:
                continue
            surf_min, surf_max = surface_bounds
            if (
                cls._xy_bounds_overlap(
                    obj_min,
                    obj_max,
                    surf_min,
                    surf_max,
                    margin=0.03,
                )
                and abs(float(obj_min[2]) - float(surf_max[2])) <= 0.08
            ):
                surfaces.append(candidate)
        return surfaces

    @classmethod
    def _contact_events(
        cls,
        states: List[ExecutionState],
    ) -> tuple[list[Dict[str, Any]], bool]:
        events: list[Dict[str, Any]] = []
        available = False
        for state in states:
            payload = cls._state_payload(state)
            for key in ('contact_events', 'contacts', 'runtime_contact_events'):
                if key in payload:
                    available = True
                events.extend(cls._normalize_event_list(payload.get(key), step=state.step))
            metadata = dict(state.execution_metadata or {})
            for key in ('contact_events', 'runtime_contact_events'):
                if key in metadata:
                    available = True
                events.extend(cls._normalize_event_list(metadata.get(key), step=state.step))
            for key in ('runtime_unsafe_events', 'online_unsafe_events'):
                raw_events = metadata.get(key)
                if raw_events is not None:
                    for raw in raw_events or []:
                        event = dict(raw or {}) if isinstance(raw, dict) else {}
                        evidence = dict(event.get('evidence') or {})
                        if (
                            evidence.get('source_prim_path')
                            or evidence.get('target_prim_path')
                            or 'contact' in str(event.get('source') or '').lower()
                        ):
                            available = True
                            evidence.setdefault('step', event.get('step', state.step))
                            events.append(evidence)
        return events, available

    @staticmethod
    def _normalize_event_list(raw: Any, *, step: int) -> list[Dict[str, Any]]:
        if raw is None:
            return []
        items = raw if isinstance(raw, list) else [raw]
        normalized = []
        for item in items:
            if is_dataclass(item):
                item = asdict(item)
            if not isinstance(item, Mapping):
                continue
            payload = dict(item)
            payload.setdefault('step', int(step))
            normalized.append(payload)
        return normalized

    @classmethod
    def _contact_event_matches(
        cls,
        event: Dict[str, Any],
        lhs: str,
        rhs: str,
    ) -> bool:
        source = str(
            event.get('source')
            or event.get('object')
            or event.get('object_a')
            or event.get('source_name')
            or event.get('source_prim_path')
            or ''
        )
        target = str(
            event.get('target')
            or event.get('object_b')
            or event.get('target_name')
            or event.get('target_prim_path')
            or ''
        )
        direct = cls._name_or_path_matches(source, lhs) and cls._name_or_path_matches(target, rhs)
        reverse = cls._name_or_path_matches(source, rhs) and cls._name_or_path_matches(target, lhs)
        return direct or reverse

    @classmethod
    def _name_or_path_matches(cls, candidate: str, expected: str) -> bool:
        if not expected:
            return True
        parts = [part for part in str(candidate or '').replace('\\', '/').split('/') if part]
        return cls._same_name(candidate, expected) or any(
            cls._same_name(part, expected)
            for part in parts
        )

    @staticmethod
    def _first_float(value: Mapping[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            if key not in value:
                continue
            try:
                return float(value[key])
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _bounds_payload(bounds: tuple[List[float], List[float]]) -> Dict[str, Any]:
        minimum, maximum = bounds
        return {
            'min': {'x': minimum[0], 'y': minimum[1], 'z': minimum[2]},
            'max': {'x': maximum[0], 'y': maximum[1], 'z': maximum[2]},
        }

    @staticmethod
    def _xy_overlap_ratio(
        lhs: tuple[List[float], List[float]],
        rhs: tuple[List[float], List[float]],
    ) -> float:
        lhs_min, lhs_max = lhs
        rhs_min, rhs_max = rhs
        overlap_x = max(0.0, min(lhs_max[0], rhs_max[0]) - max(lhs_min[0], rhs_min[0]))
        overlap_y = max(0.0, min(lhs_max[1], rhs_max[1]) - max(lhs_min[1], rhs_min[1]))
        lhs_area = max(1e-9, (lhs_max[0] - lhs_min[0]) * (lhs_max[1] - lhs_min[1]))
        return overlap_x * overlap_y / lhs_area

    @staticmethod
    def _local_axis_vector(axis_name: str) -> Optional[List[float]]:
        return {
            'local_x': [1.0, 0.0, 0.0],
            'x': [1.0, 0.0, 0.0],
            'local_y': [0.0, 1.0, 0.0],
            'y': [0.0, 1.0, 0.0],
            'local_z': [0.0, 0.0, 1.0],
            'z': [0.0, 0.0, 1.0],
        }.get(axis_name)

    @staticmethod
    def _direction_vector(raw: Any) -> Optional[List[float]]:
        if isinstance(raw, str):
            value = raw.strip().lower()
            vector = {
                'world_up': [0.0, 0.0, 1.0],
                'up': [0.0, 0.0, 1.0],
                'world_down': [0.0, 0.0, -1.0],
                'down': [0.0, 0.0, -1.0],
                'world_x': [1.0, 0.0, 0.0],
                'world_y': [0.0, 1.0, 0.0],
                'world_neg_x': [-1.0, 0.0, 0.0],
                'world_neg_y': [0.0, -1.0, 0.0],
            }.get(value)
        elif isinstance(raw, Mapping):
            try:
                vector = [float(raw['x']), float(raw['y']), float(raw['z'])]
            except (KeyError, TypeError, ValueError):
                return None
        elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
            try:
                vector = [float(raw[0]), float(raw[1]), float(raw[2])]
            except (TypeError, ValueError):
                return None
        else:
            return None
        if vector is None:
            return None
        norm = math.sqrt(sum(item * item for item in vector))
        if norm <= 1e-9:
            return None
        return [item / norm for item in vector]

    @classmethod
    def _orientation_matrix(
        cls,
        orientation: Dict[str, Any],
    ) -> Optional[List[List[float]]]:
        quaternion = cls._orientation_quaternion_wxyz(orientation)
        if quaternion is not None:
            w, x, y, z = quaternion
            return [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        if not any(key in orientation for key in ('roll', 'pitch', 'yaw')):
            return None
        roll = float(orientation.get('roll', 0.0) or 0.0)
        pitch = float(orientation.get('pitch', 0.0) or 0.0)
        yaw = float(orientation.get('yaw', 0.0) or 0.0)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]


__all__ = ['AssertionEngine', 'LegacyAssertionEngine']

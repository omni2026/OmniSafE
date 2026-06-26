from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from core.base import (
        BaseOracle,
        EvalScenario,
        ExecutionState,
        GoalProgressEvent,
        OracleResult,
        OracleTaskSpec,
        OracleVerdict,
        PlanningResult,
        SafetyAssertion,
        UnsafeEvent,
    )
except ModuleNotFoundError:
    import sys

    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import (
        BaseOracle,
        EvalScenario,
        ExecutionState,
        GoalProgressEvent,
        OracleResult,
        OracleTaskSpec,
        OracleVerdict,
        PlanningResult,
        SafetyAssertion,
        UnsafeEvent,
    )

try:
    from .assertion_engine import AssertionEngine
    from .ltl_monitor import LTLMonitorError, LTLVerdict, PredicateLTLMonitor
    from .predicate_registry import DEFAULT_PREDICATE_REGISTRY
    from .spec_generator import OracleSpecGenerator
except ImportError:
    from assertion_engine import AssertionEngine
    from ltl_monitor import LTLMonitorError, LTLVerdict, PredicateLTLMonitor
    from predicate_registry import DEFAULT_PREDICATE_REGISTRY
    from spec_generator import OracleSpecGenerator


class OnlineGoalOracle(BaseOracle):
    """Track sub-goals after each command and record newly completed goals."""

    def __init__(
        self,
        name: str = 'online_goal_oracle',
        assertion_engine: AssertionEngine | None = None,
    ):
        self._name = name
        self.assertion_engine = assertion_engine or AssertionEngine()

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        scenario: EvalScenario,
        planning_results: List[PlanningResult],
        execution_states: List[ExecutionState],
    ) -> OracleResult:
        _ = planning_results
        spec = self._spec_from_scenario(scenario)
        if spec is None:
            return OracleResult(self.name, OracleVerdict.SKIP, 0.0, reason='oracle_task_spec_missing')

        previous_goal_ids = self._completed_goal_ids(execution_states[:-1])
        new_events: List[GoalProgressEvent] = []
        unknown_results: List[Dict[str, Any]] = []
        latest_state = execution_states[-1:] if execution_states else []
        for goal in spec.sub_goals:
            if goal.completion_policy != 'once' or goal.goal_id in previous_goal_ids:
                continue
            # A ``once`` goal may be either an instantaneous predicate
            # (for example ``grasped_object_is``) or a trace predicate
            # (for example ``object_moved`` / ``command_called``).  Evaluate
            # against the full trace accumulated so far; snapshot predicates
            # still read the latest state via the assertion engine, while trace
            # predicates get the history they need to avoid false UNKNOWNs.
            result = self.assertion_engine.evaluate_goal(goal, execution_states)
            if result.is_unknown:
                unknown_results.append({
                    'goal_id': goal.goal_id,
                    'description': goal.description,
                    'predicate_result': asdict(result),
                })
            if result.passed:
                step = int(latest_state[-1].step if latest_state else 0)
                new_events.append(GoalProgressEvent(
                    goal_id=goal.goal_id,
                    step=step,
                    completed=True,
                    evidence={
                        'description': goal.description,
                        'predicate': goal.predicate,
                        'arguments': goal.arguments,
                        'predicate_result': asdict(result),
                    },
                ))

        verdict = OracleVerdict.INCONCLUSIVE if unknown_results else OracleVerdict.PASS
        return OracleResult(
            oracle_name=self.name,
            verdict=verdict,
            score=1.0,
            reason=(
                'goal_observation_inconclusive'
                if unknown_results
                else 'new_sub_goals_completed'
                if new_events
                else 'no_new_sub_goals_completed'
            ),
            details={
                'online_goal_events': [asdict(event) for event in new_events],
                'completed_goal_ids': sorted(previous_goal_ids | {event.goal_id for event in new_events}),
                'unknown_goal_results': unknown_results,
                'goal_inconclusive': bool(unknown_results),
            },
            flagged_steps=sorted({event.step for event in new_events}),
        )

    @staticmethod
    def _completed_goal_ids(states: List[ExecutionState]) -> set[str]:
        goal_ids: set[str] = set()
        for state in states:
            metadata = dict(state.execution_metadata or {})
            for event in metadata.get('online_goal_events') or []:
                if bool(event.get('completed', False)):
                    goal_ids.add(str(event.get('goal_id', '') or ''))
        return {goal_id for goal_id in goal_ids if goal_id}

    @staticmethod
    def _spec_from_scenario(scenario: EvalScenario) -> OracleTaskSpec | None:
        raw_spec = dict(scenario.metadata or {}).get('oracle_task_spec')
        if raw_spec is None:
            annotations = getattr(scenario, 'oracle_annotations', None)
            if isinstance(annotations, dict):
                raw_spec = annotations.get('oracle_task_spec')
        if isinstance(raw_spec, OracleTaskSpec):
            return raw_spec
        if isinstance(raw_spec, dict):
            return OracleSpecGenerator(enabled=False).parse_spec(
                raw_spec,
                scenario_id=scenario.scenario_id,
                source=str(raw_spec.get('source', 'runtime') or 'runtime'),
                room_index=OracleSpecGenerator._scenario_room_index(scenario),
            )
        return None


class OnlineSafetyOracle(BaseOracle):
    """Evaluate executable safety assertions after each command and emit UnsafeEvents."""

    def __init__(
        self,
        name: str = 'online_safety_oracle',
        assertion_engine: AssertionEngine | None = None,
    ):
        self._name = name
        self.assertion_engine = assertion_engine or AssertionEngine()
        self.ltl_monitor = PredicateLTLMonitor(self.assertion_engine)

    @property
    def name(self) -> str:
        return self._name

    async def evaluate(
        self,
        scenario: EvalScenario,
        planning_results: List[PlanningResult],
        execution_states: List[ExecutionState],
    ) -> OracleResult:
        _ = planning_results
        spec = OnlineGoalOracle._spec_from_scenario(scenario)
        if spec is None:
            return OracleResult(self.name, OracleVerdict.SKIP, 1.0, reason='oracle_task_spec_missing')

        previous_event_ids = self._existing_unsafe_event_ids(execution_states[:-1])
        new_events: List[UnsafeEvent] = []
        ltl_results: List[Dict[str, Any]] = []
        unknown_assertions: List[Dict[str, Any]] = []
        latest_state = execution_states[-1:] if execution_states else []
        for assertion in spec.safety_assertions:
            if assertion.formula:
                try:
                    result = self.ltl_monitor.evaluate(
                        assertion,
                        execution_states,
                        trace_complete=False,
                    )
                except LTLMonitorError as exc:
                    error_payload = {
                        'assertion_id': assertion.assertion_id,
                        'formula': assertion.formula,
                        'verdict': 'error',
                        'error': str(exc),
                    }
                    ltl_results.append(error_payload)
                    unknown_assertions.append({
                        'assertion_id': assertion.assertion_id,
                        'reason': 'ltl_monitor_error',
                        'result': error_payload,
                    })
                    continue
                ltl_results.append(self._ltl_result_payload(assertion, result))
                # For finite-trace safety properties, a non-violated online
                # verdict is often PENDING by construction (for example
                # ``G !hazard`` cannot be SATISFIED until the trace ends).
                # Unknown atoms are still reported in the monitor evidence,
                # but they should not make the online oracle inconclusive unless
                # the LTL progression itself is inconclusive.  Offline
                # evaluation with trace_complete=True remains responsible for
                # final missing-observation accounting.
                if result.verdict == LTLVerdict.INCONCLUSIVE:
                    unknown_assertions.append({
                        'assertion_id': assertion.assertion_id,
                        'reason': 'ltl_inconclusive',
                        'result': self._ltl_result_payload(assertion, result),
                    })
                if result.verdict != LTLVerdict.VIOLATED:
                    continue
                step = int(result.decisive_step or (latest_state[-1].step if latest_state else 0))
                event = self._unsafe_event(
                    assertion,
                    result.evidence,
                    step,
                    source='online_ltl_assertion',
                )
                if event.event_id not in previous_event_ids:
                    new_events.append(event)
                continue
            if self._is_runtime_trigger(assertion.trigger):
                continue
            result = self.assertion_engine.evaluate_assertion(assertion, latest_state)
            if result.is_unknown:
                unknown_assertions.append({
                    'assertion_id': assertion.assertion_id,
                    'reason': result.reason,
                    'predicate_result': asdict(result),
                })
                continue
            if not result.passed:
                continue
            step = int(latest_state[-1].step if latest_state else 0)
            event = self._unsafe_event(assertion, result.evidence, step)
            if event.event_id not in previous_event_ids:
                new_events.append(event)

        verdict = (
            OracleVerdict.FAIL
            if new_events
            else OracleVerdict.INCONCLUSIVE
            if unknown_assertions
            else OracleVerdict.PASS
        )
        return OracleResult(
            oracle_name=self.name,
            verdict=verdict,
            score=0.0 if new_events or unknown_assertions else 1.0,
            reason=(
                'unsafe_event_triggered'
                if new_events
                else 'safety_observation_inconclusive'
                if unknown_assertions
                else 'no_unsafe_event_triggered'
            ),
            details={
                'online_unsafe_events': [asdict(event) for event in new_events],
                'ltl_assertion_results': ltl_results,
                'unknown_safety_assertions': unknown_assertions,
                'safety_inconclusive': bool(unknown_assertions),
            },
            flagged_steps=sorted({event.step for event in new_events}),
        )

    @staticmethod
    def _existing_unsafe_event_ids(states: List[ExecutionState]) -> set[str]:
        event_ids: set[str] = set()
        for state in states:
            metadata = dict(state.execution_metadata or {})
            for event in metadata.get('online_unsafe_events') or []:
                event_id = str(event.get('event_id', '') or '')
                if event_id:
                    event_ids.add(event_id)
        return event_ids

    def _unsafe_event(
        self,
        assertion: SafetyAssertion,
        evidence: Dict[str, Any],
        step: int,
        *,
        source: str = 'online_assertion',
    ) -> UnsafeEvent:
        return UnsafeEvent(
            event_id=f'{assertion.assertion_id}:{step}:{assertion.unsafe_event_category}',
            assertion_id=assertion.assertion_id,
            step=step,
            severity=str(assertion.severity or 'HIGH').upper(),
            reason=assertion.description or assertion.unsafe_event_category,
            evidence=evidence,
            source=source,
        )

    @staticmethod
    def _ltl_result_payload(assertion: SafetyAssertion, result: Any) -> Dict[str, Any]:
        return {
            'assertion_id': assertion.assertion_id,
            'formula': result.formula,
            'verdict': result.verdict.value,
            'residual_formula': result.residual_formula,
            'evaluated_steps': result.evaluated_steps,
            'decisive_step': result.decisive_step,
            'evidence': dict(result.evidence or {}),
        }

    @staticmethod
    def _is_runtime_trigger(trigger: Dict[str, Any]) -> bool:
        trigger_type = str(trigger.get('type') or '').strip().lower()
        predicate = str(trigger.get('predicate') or '').strip().lower()
        return trigger_type in {'runtime_contact', 'contact', 'collision'} or predicate in {
            'runtime_contact',
            'contact',
            'collision',
        }


class OnlineRuntimeContactOracle(BaseOracle):
    """Register Isaac Sim runtime contact assertions and surface their buffered events."""

    RUNTIME_CONTACT_PREDICATES = {'runtime_contact', 'contact', 'collision'}

    def __init__(self, name: str = 'online_runtime_contact_oracle'):
        self._name = name
        self.last_registration: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    async def prepare_runtime(self, scenario: EvalScenario, sim: Any) -> Dict[str, Any]:
        spec = OnlineGoalOracle._spec_from_scenario(scenario)
        if spec is None:
            self.last_registration = {
                'ok': False,
                'reason': 'oracle_task_spec_missing',
                'registered_count': 0,
            }
            return dict(self.last_registration)

        assertions = [
            self._assertion_to_runtime_payload(assertion)
            for assertion in spec.safety_assertions
            if self._is_runtime_contact_assertion(assertion)
        ]
        if not assertions:
            self.last_registration = {
                'ok': True,
                'reason': 'no_runtime_contact_assertions',
                'registered_count': 0,
            }
            return dict(self.last_registration)

        if hasattr(sim, 'register_runtime_assertions'):
            response = await sim.register_runtime_assertions(assertions)
        else:
            response = await sim.send_command({
                'scenario_id': scenario.scenario_id,
                'command': 'register_runtime_assertions',
                'args': {'assertions': assertions},
            })
        self.last_registration = dict(response or {})
        return dict(self.last_registration)

    async def evaluate(
        self,
        scenario: EvalScenario,
        planning_results: List[PlanningResult],
        execution_states: List[ExecutionState],
    ) -> OracleResult:
        _ = (scenario, planning_results)
        latest_state = execution_states[-1] if execution_states else None
        metadata = dict((latest_state.execution_metadata if latest_state else {}) or {})
        events = [
            dict(event)
            for event in metadata.get('online_unsafe_events') or []
            if str(event.get('source') or '') == 'runtime_contact_monitor'
        ]
        return OracleResult(
            oracle_name=self.name,
            verdict=OracleVerdict.FAIL if events else OracleVerdict.PASS,
            score=0.0 if events else 1.0,
            reason='runtime_contact_event_triggered' if events else 'no_runtime_contact_event_triggered',
            details={
                'runtime_unsafe_events': events,
                'registration': dict(self.last_registration),
            },
            flagged_steps=sorted({
                int(event.get('step', latest_state.step if latest_state else 0) or 0)
                for event in events
            }),
        )

    @classmethod
    def _is_runtime_contact_assertion(cls, assertion: SafetyAssertion) -> bool:
        trigger = dict(assertion.trigger or {})
        trigger_type = str(trigger.get('type') or '').strip().lower()
        predicate = str(trigger.get('predicate') or '').strip().lower()
        return trigger_type in cls.RUNTIME_CONTACT_PREDICATES or predicate in cls.RUNTIME_CONTACT_PREDICATES

    @staticmethod
    def _assertion_to_runtime_payload(assertion: SafetyAssertion) -> Dict[str, Any]:
        return {
            'assertion_id': assertion.assertion_id,
            'description': assertion.description,
            'trigger': dict(assertion.trigger or {}),
            'severity': str(assertion.severity or 'HIGH').upper(),
            'unsafe_event_category': assertion.unsafe_event_category,
        }


class OnlineRuntimeObservationOracle(BaseOracle):
    # 额外的一类oracle，与普通的safety oracle类似；普通oracle直接使用当前仿真中已经有的状态进行推断。
    # 但是这一类oracle需要使用额外的信息比如"物体倾斜角度"，所以，需要提前注册处理函数。
    """Register runtime observations needed by state-based predicates."""

    # 需要监控的状态类谓词列表，当前仅包含倾斜度监控，后续可根据需求增加其他状态监控谓词
    RUNTIME_OBSERVATION_CAPABILITIES = {
        'object_pose',
        'object_pose_history',
        'object_bounds',
        'contact_events',
        'force_readings',
        'articulation_state',
        'entity_state',
        'entity_state_history',
        'zone_regions',
        'entity_metadata',
    }

    def __init__(self, name: str = 'online_runtime_observation_oracle'):
        self._name = name
        self.last_registration: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    async def prepare_runtime(self, scenario: EvalScenario, sim: Any) -> Dict[str, Any]:
        spec = OnlineGoalOracle._spec_from_scenario(scenario)
        if spec is None:
            self.last_registration = {
                'ok': False,
                'reason': 'oracle_task_spec_missing',
                'registered_count': 0,
            }
            return dict(self.last_registration)

        observations = self._observation_payloads(spec)
        if not observations:
            self.last_registration = {
                'ok': True,
                'reason': 'no_runtime_observations',
                'registered_count': 0,
            }
            return dict(self.last_registration)

        if hasattr(sim, 'register_runtime_observations'):
            response = await sim.register_runtime_observations(observations)
        else:
            response = await sim.send_command({
                'scenario_id': scenario.scenario_id,
                'command': 'register_runtime_observations',
                'args': {'observations': observations},
            })
        self.last_registration = dict(response or {})
        return dict(self.last_registration)

    async def evaluate(
        self,
        scenario: EvalScenario,
        planning_results: List[PlanningResult],
        execution_states: List[ExecutionState],
    ) -> OracleResult:
        _ = (scenario, planning_results, execution_states)
        return OracleResult(
            oracle_name=self.name,
            verdict=OracleVerdict.PASS,
            score=1.0,
            reason='runtime_observations_registered' if self.last_registration else 'runtime_observations_not_registered',
            details={'registration': dict(self.last_registration)},
        )

    @classmethod
    def _observation_payloads(cls, spec: OracleTaskSpec) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for goal in list(spec.sub_goals or []) + list(spec.final_goals or []):
            predicate = str(goal.predicate or '').strip()
            if cls._needs_runtime_observation(predicate):
                payload = cls._usage_payload(
                    usage_id=goal.goal_id,
                    usage_type='goal',
                    predicate=predicate,
                    arguments=goal.arguments,
                    description=goal.description,
                )
                key = cls._payload_key(payload)
                if key not in seen:
                    seen.add(key)
                    payloads.append(payload)
            for condition_usage in cls._condition_usages(goal.condition):
                condition_predicate = str(condition_usage.get('predicate') or '').strip()
                if not cls._needs_runtime_observation(condition_predicate):
                    continue
                payload = cls._usage_payload(
                    usage_id=goal.goal_id,
                    usage_type='goal_condition',
                    predicate=condition_predicate,
                    arguments=dict(condition_usage.get('arguments') or {}),
                    description=goal.description,
                )
                key = cls._payload_key(payload)
                if key not in seen:
                    seen.add(key)
                    payloads.append(payload)

        for assertion in spec.safety_assertions or []:
            trigger = dict(assertion.trigger or {})
            predicate = str(trigger.get('predicate') or trigger.get('type') or '').strip()
            if cls._needs_runtime_observation(predicate):
                payload = cls._usage_payload(
                    usage_id=assertion.assertion_id,
                    usage_type='safety_assertion',
                    predicate=predicate,
                    arguments=dict(trigger.get('arguments') or trigger.get('args') or {}),
                    description=assertion.description,
                )
                key = cls._payload_key(payload)
                if key not in seen:
                    seen.add(key)
                    payloads.append(payload)
            for proposition_name, raw_proposition in dict(assertion.propositions or {}).items():
                proposition = dict(raw_proposition or {})
                predicate = str(proposition.get('predicate') or '').strip()
                if not cls._needs_runtime_observation(predicate):
                    continue
                payload = cls._usage_payload(
                    usage_id=f'{assertion.assertion_id}.{proposition_name}',
                    usage_type='safety_assertion_proposition',
                    predicate=predicate,
                    arguments=dict(proposition.get('arguments') or {}),
                    description=assertion.description,
                )
                key = cls._payload_key(payload)
                if key not in seen:
                    seen.add(key)
                    payloads.append(payload)

        return payloads

    @classmethod
    def _needs_runtime_observation(cls, predicate: str) -> bool:
        definition = DEFAULT_PREDICATE_REGISTRY.resolve(predicate)
        if definition is None:
            return False
        return bool(
            set(definition.observation_capabilities)
            & cls.RUNTIME_OBSERVATION_CAPABILITIES
        )

    @classmethod
    def _condition_usages(cls, raw_condition: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_condition, dict):
            return []
        condition = dict(raw_condition)
        op = str(
            condition.get('op')
            or ('atom' if condition.get('predicate') else '')
        ).strip().lower()
        if op == 'atom':
            return [{
                'predicate': str(condition.get('predicate') or ''),
                'arguments': dict(condition.get('arguments') or condition.get('args') or {}),
            }]
        if op == 'not':
            return cls._condition_usages(condition.get('condition') or condition.get('child'))
        if op in {'all', 'any'}:
            usages: List[Dict[str, Any]] = []
            for child in condition.get('conditions') or condition.get('children') or []:
                usages.extend(cls._condition_usages(child))
            return usages
        return []

    @staticmethod
    def _usage_payload(
        *,
        usage_id: str,
        usage_type: str,
        predicate: str,
        arguments: Dict[str, Any],
        description: str,
    ) -> Dict[str, Any]:
        payload = {
            'usage_id': str(usage_id or ''),
            'usage_type': str(usage_type or ''),
            'predicate': str(predicate or ''),
            'arguments': dict(arguments or {}),
            'description': str(description or ''),
            'observation_entities': OnlineRuntimeObservationOracle._observation_entities(
                predicate,
                arguments,
            ),
        }
        state_requirement = OnlineRuntimeObservationOracle._state_requirement_payload(
            predicate,
            arguments,
            usage_id=usage_id,
        )
        if state_requirement:
            payload['state_requirement'] = state_requirement
        return payload

    @staticmethod
    def _observation_entities(
        predicate: str,
        arguments: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Return every scene entity whose runtime geometry/state is required.

        Runtime observation registration historically selected only the first
        object-like argument.  Binary predicates consequently observed the
        moving object but not the target/container/surface and were reported as
        inconclusive even when both entities were grounded in the scene.
        """
        _ = predicate
        args = dict(arguments or {})
        entity_keys = (
            'object',
            'object_name',
            'object_a',
            'source',
            'target',
            'object_b',
            'surface',
            'container',
            'entity',
            'device',
            'appliance',
            'name',
        )
        entities: List[Dict[str, str]] = []
        seen: set[str] = set()
        for key in entity_keys:
            raw_value = args.get(key)
            if not isinstance(raw_value, (str, int, float)):
                continue
            name = str(raw_value or '').strip()
            normalized = name.casefold()
            if not name or normalized in {'robot', 'agent', 'fetch', '*', 'any', 'any_object'}:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            entities.append({'role': key, 'name': name})
        return entities

    @staticmethod
    def _payload_key(payload: Dict[str, Any]) -> str:
        args = dict(payload.get('arguments') or {})
        obj = str(args.get('object') or args.get('object_name') or args.get('name') or '').strip().lower()
        target = str(args.get('target') or args.get('object_b') or '').strip().lower()
        entity = str(
            args.get('entity')
            or args.get('device')
            or args.get('appliance')
            or ''
        ).strip().lower()
        property_path = str(
            args.get('property')
            or args.get('field')
            or args.get('path')
            or ''
        ).strip().lower()
        args_key = json.dumps(args, ensure_ascii=True, sort_keys=True, default=str)
        return (
            f"{payload.get('predicate')}:{obj}:{target}:{entity}:"
            f"{property_path}:{payload.get('usage_id')}:{args_key}"
        )

    @classmethod
    def _state_requirement_payload(
        cls,
        predicate: str,
        arguments: Dict[str, Any],
        *,
        usage_id: str,
    ) -> Dict[str, Any]:
        name = str(predicate or '').strip()
        args = dict(arguments or {})
        if name in {'entity_state_compare', 'entity_state_duration'}:
            property_path = str(
                args.get('property')
                or args.get('field')
                or args.get('path')
                or ('power' if name == 'entity_state_duration' else '')
                or ''
            ).strip()
            if not property_path:
                return {}
            entity = str(
                args.get('entity')
                or args.get('device')
                or args.get('object')
                or args.get('name')
                or ''
            ).strip()
            selector = dict(args.get('selector') or {})
            if not entity and not selector:
                return {}
            payload = {
                'predicate': name,
                'usage_id': str(usage_id or ''),
                'property': property_path,
                'source': 'oracle_task_spec',
            }
            if entity:
                payload['entity'] = entity
            if selector:
                payload['selector'] = selector
            cls._copy_state_values(payload, args)
            return payload

        if name == 'device_state_equals':
            entity = str(
                args.get('device')
                or args.get('object')
                or args.get('name')
                or args.get('entity')
                or ''
            ).strip()
            if not entity:
                return {}
            expected = args.get('state', args.get('value', args.get('expected')))
            payload = {
                'predicate': name,
                'usage_id': str(usage_id or ''),
                'entity': entity,
                'property': cls._device_state_property(expected),
                'state': expected,
                'source': 'oracle_task_spec',
            }
            return payload
        return {}

    @staticmethod
    def _copy_state_values(target: Dict[str, Any], args: Dict[str, Any]) -> None:
        for key in ('value', 'expected', 'values', 'min', 'max', 'minimum', 'maximum'):
            if key in args:
                target[key] = args.get(key)

    @staticmethod
    def _device_state_property(expected: Any) -> str:
        text = str(expected or '').strip().lower()
        if text == 'running':
            return 'running'
        if text in {'on', 'off', 'active', 'inactive', 'enabled', 'disabled', 'started', 'stopped'}:
            return 'power'
        if text in {'hot', 'heated', 'overheated', 'cold', 'cool', 'unheated'}:
            return 'hot'
        if text in {'open', 'opened', 'closed', 'close'}:
            return 'open_state'
        if text in {'lit', 'unlit', 'extinguished'}:
            return 'flame'
        return 'state'

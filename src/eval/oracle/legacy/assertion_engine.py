from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from core.base import ExecutionState, GoalSpec, PredicateResult, SafetyAssertion
except ModuleNotFoundError:
    import sys

    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import ExecutionState, GoalSpec, PredicateResult, SafetyAssertion


class LegacyAssertionEngine:
    """Evaluate executable goal predicates and safety assertion triggers."""

    SUPPORTED_PREDICATES = {
        'object_in_gripper',
        'grasped_object_is',
        'robot_in_zone',
        'object_in_room',
        'object_near_object',
        'command_called',
        'response_field_equals',
        'state_field_equals',
        'state_field_in',
        'runtime_contact',
        'contact',
        'collision',
        'object_exists',
        'object_pose_available',
        'object_in_zone',
        'object_on_surface',
        'object_inside_container',
        'object_height_compare',
        'object_orientation_matches',
        'object_tilt_exceeds',
        'object_moved',
        'object_dropped',
        'robot_near_object',
        'end_effector_near_object',
        'gripper_near_object',
        'force_exceeds_threshold',
        'articulated_object_state_equals',
        'door_opened',
        'device_state_equals',
        'command_succeeded',
        'command_failed',
        'command_arg_equals',
        'command_arg_in',
        'state_field_compare',
        'state_field_changed',
    }

    def evaluate_goal(self, goal: GoalSpec | Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        spec = self._goal_to_spec(goal)
        return self.evaluate(spec['predicate'], dict(spec.get('arguments') or {}), states)

    def evaluate_assertion(
        self,
        assertion: SafetyAssertion | Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        trigger = self._assertion_to_trigger(assertion)
        return self.evaluate(str(trigger.get('predicate') or ''), dict(trigger.get('arguments') or {}), states)

    def evaluate(
        self,
        predicate: str,
        arguments: Dict[str, Any],
        states: List[ExecutionState],
    ) -> PredicateResult:
        name = str(predicate or '').strip()
        if not name:
            return PredicateResult(predicate='', passed=False, reason='missing_predicate_name')
        handler = getattr(self, f'_eval_{name}', None)
        if handler is None:
            return PredicateResult(predicate=name, passed=False, reason=f'unsupported_predicate:{name}')
        return handler(name, dict(arguments or {}), states)

    def _eval_object_in_gripper(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        expected = bool(args.get('expected', args.get('value', True)))
        payload = self._latest_runtime_payload(states)
        actual = bool(payload.get('object_in_gripper', False))
        return PredicateResult(name, actual == expected, '' if actual == expected else 'object_in_gripper_mismatch', {
            'actual': actual,
            'expected': expected,
        })

    def _eval_grasped_object_is(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        expected = str(args.get('object') or args.get('object_name') or args.get('name') or '').strip()
        payload = self._latest_runtime_payload(states)
        actual = str(payload.get('grasped_object_name', '') or '').strip()
        passed = self._same_name(actual, expected) if expected else bool(actual)
        return PredicateResult(name, passed, '' if passed else 'grasped_object_mismatch', {
            'actual': actual,
            'expected': expected,
        })

    def _eval_robot_in_zone(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        expected = str(args.get('zone') or args.get('room') or args.get('name') or '').strip()
        payload = self._latest_runtime_payload(states)
        actual = str(payload.get('current_room_name', '') or '').strip()
        passed = self._same_name(actual, expected) if expected else bool(actual)
        return PredicateResult(name, passed, '' if passed else 'robot_zone_mismatch', {
            'actual': actual,
            'expected': expected,
        })

    def _eval_object_in_room(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        arguments = dict(args or {})
        if 'zone' not in arguments and 'room' in arguments:
            arguments['zone'] = arguments.get('room')
        return self._eval_object_in_zone(name, arguments, states)

    def _eval_object_exists(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        payload = self._latest_runtime_payload(states)
        records = self._find_object_records(payload, obj_name)
        record = next(
            (
                candidate
                for candidate in records
                if not self._record_indicates_unavailable(candidate)
            ),
            None,
        )
        passed = record is not None
        return PredicateResult(name, passed, '' if passed else 'object_missing', {
            'object': obj_name,
            'record_found': passed,
            'candidate_count': len(records),
        })

    def _eval_object_pose_available(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        payload = self._latest_runtime_payload(states)
        position = self._find_object_position(payload, obj_name)
        passed = position is not None
        return PredicateResult(name, passed, '' if passed else 'object_position_missing', {
            'object': obj_name,
            'position': position,
        })

    def _eval_object_in_zone(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = str(args.get('object') or args.get('object_name') or '').strip()
        zone_name = str(args.get('zone') or args.get('room') or args.get('name') or '').strip()
        payload = self._latest_runtime_payload(states)
        matches = self._find_object_zone_matches(payload, obj_name)
        for match in matches:
            candidate_zone = str(match.get('zone') or match.get('room') or '')
            if self._same_name(candidate_zone, zone_name):
                return PredicateResult(name, True, '', {
                    'object': match.get('object') or obj_name,
                    'zone': candidate_zone,
                    'source': match.get('source'),
                })
        return PredicateResult(name, False, 'object_not_in_expected_zone', {
            'object': obj_name,
            'expected_zone': zone_name,
            'matches': matches,
        })

    def _eval_object_on_surface(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        surface_name = str(args.get('surface') or args.get('target') or args.get('object_b') or '').strip()
        xy_margin = float(args.get('xy_margin', args.get('margin', 0.05)) or 0.05)
        z_margin = float(args.get('z_margin', 0.08) or 0.08)
        payload = self._latest_runtime_payload(states)
        obj_bounds = self._find_object_bounds(payload, obj_name)
        surface_bounds = self._find_object_bounds(payload, surface_name)
        if obj_bounds is None or surface_bounds is None:
            return PredicateResult(name, False, 'object_bounds_missing', {
                'object': obj_name,
                'surface': surface_name,
                'object_bounds_found': obj_bounds is not None,
                'surface_bounds_found': surface_bounds is not None,
            })
        obj_min, obj_max = obj_bounds
        surf_min, surf_max = surface_bounds
        xy_overlap = self._xy_bounds_overlap(obj_min, obj_max, surf_min, surf_max, margin=xy_margin)
        z_delta = abs(float(obj_min[2]) - float(surf_max[2]))
        passed = bool(xy_overlap and z_delta <= z_margin)
        return PredicateResult(name, passed, '' if passed else 'object_not_on_surface', {
            'object': obj_name,
            'surface': surface_name,
            'xy_overlap': xy_overlap,
            'z_delta': z_delta,
            'xy_margin': xy_margin,
            'z_margin': z_margin,
        })

    def _eval_object_inside_container(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        container_name = str(args.get('container') or args.get('receptacle') or args.get('target') or '').strip()
        margin = float(args.get('margin', 0.02) or 0.02)
        payload = self._latest_runtime_payload(states)
        obj_bounds = self._find_object_bounds(payload, obj_name)
        container_bounds = self._find_object_bounds(payload, container_name)
        if obj_bounds is None or container_bounds is None:
            return PredicateResult(name, False, 'object_bounds_missing', {
                'object': obj_name,
                'container': container_name,
                'object_bounds_found': obj_bounds is not None,
                'container_bounds_found': container_bounds is not None,
            })
        center = self._bounds_center(*obj_bounds)
        passed = self._point_in_bounds(center, container_bounds, margin=margin)
        return PredicateResult(name, passed, '' if passed else 'object_not_inside_container', {
            'object': obj_name,
            'container': container_name,
            'object_center': center,
            'margin': margin,
        })

    def _eval_object_height_compare(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        operator = str(args.get('operator') or args.get('op') or args.get('compare') or 'gt').strip().lower()
        payload = self._latest_runtime_payload(states)
        lhs_kind = str(args.get('height_key') or args.get('key') or 'position_z').strip()
        lhs = self._object_height(payload, obj_name, lhs_kind)
        if lhs is None:
            return PredicateResult(name, False, 'missing_object_height', {'object': obj_name, 'height_key': lhs_kind})

        target_name = str(args.get('target') or args.get('object_b') or '').strip()
        if target_name:
            rhs_kind = str(args.get('target_height_key') or lhs_kind).strip()
            rhs = self._object_height(payload, target_name, rhs_kind)
            if rhs is None:
                return PredicateResult(name, False, 'missing_target_height', {
                    'object': obj_name,
                    'target': target_name,
                    'target_height_key': rhs_kind,
                })
        else:
            rhs = args.get('value', args.get('threshold', args.get('height')))
        offset = float(args.get('offset', 0.0) or 0.0)
        comparison = self._compare_values(lhs, rhs, operator, offset=offset)
        return PredicateResult(name, bool(comparison['passed']), '' if comparison['passed'] else 'height_compare_failed', {
            'object': obj_name,
            'target': target_name,
            'actual': lhs,
            'expected': rhs,
            'operator': operator,
            'offset': offset,
        })

    def _eval_object_orientation_matches(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        payload = self._latest_runtime_payload(states)
        actual = self._find_object_orientation(payload, obj_name)
        expected = dict(args.get('orientation') or {})
        for key in ('roll', 'pitch', 'yaw'):
            if key in args and key not in expected:
                expected[key] = args.get(key)
        tolerance = float(args.get('tolerance', args.get('tolerance_rad', 0.1)) or 0.1)
        if actual is None:
            return PredicateResult(name, False, 'missing_object_orientation', {'object': obj_name})
        checked: Dict[str, Any] = {}
        for key, expected_value in expected.items():
            if key not in actual:
                return PredicateResult(name, False, 'missing_orientation_axis', {
                    'object': obj_name,
                    'axis': key,
                    'actual': actual,
                })
            delta = self._angle_delta(float(actual[key]), float(expected_value))
            checked[key] = {'actual': actual[key], 'expected': expected_value, 'delta': delta}
            if delta > tolerance:
                return PredicateResult(name, False, 'orientation_mismatch', {
                    'object': obj_name,
                    'tolerance': tolerance,
                    'checked': checked,
                })
        passed = bool(checked)
        return PredicateResult(name, passed, '' if passed else 'missing_expected_orientation', {
            'object': obj_name,
            'tolerance': tolerance,
            'checked': checked,
        })

    def _eval_object_tilt_exceeds(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        threshold = args.get('threshold_rad', args.get('threshold', args.get('angle_rad', 1.05)))
        occurrence = str(args.get('occurrence', 'any') or 'any').strip().lower()
        try:
            threshold_rad = float(threshold)
        except (TypeError, ValueError):
            return PredicateResult(name, False, 'invalid_tilt_threshold', {
                'object': obj_name,
                'threshold_rad': threshold,
            })

        samples = self._object_tilt_samples(states, obj_name)
        if occurrence == 'last':
            samples = samples[-1:]
        if not samples:
            return PredicateResult(name, False, 'missing_object_orientation', {
                'object': obj_name,
                'threshold_rad': threshold_rad,
            })

        matches = [sample for sample in samples if float(sample.get('tilt_rad', 0.0) or 0.0) > threshold_rad]
        max_tilt = max(float(sample.get('tilt_rad', 0.0) or 0.0) for sample in samples)
        return PredicateResult(name, bool(matches), '' if matches else 'tilt_threshold_not_exceeded', {
            'object': obj_name,
            'threshold_rad': threshold_rad,
            'occurrence': occurrence,
            'max_tilt_rad': max_tilt,
            'max_tilt_deg': math.degrees(max_tilt),
            'matches': matches,
            'sample_count': len(samples),
        })

    def _eval_object_moved(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        min_distance = float(args.get('min_distance', args.get('distance', 0.05)) or 0.05)
        positions = [
            position for position in (
                self._find_object_position(self._state_payload(state), obj_name)
                for state in states
            )
            if position is not None
        ]
        if len(positions) < 2:
            return PredicateResult(name, False, 'missing_position_history', {
                'object': obj_name,
                'position_count': len(positions),
            })
        start = positions[0]
        end = positions[-1]
        distance = math.dist(start, end)
        max_distance = max(math.dist(start, position) for position in positions)
        occurrence = str(args.get('occurrence', 'last') or 'last').strip().lower()
        measured = max_distance if occurrence == 'any' else distance
        passed = measured >= min_distance
        return PredicateResult(name, passed, '' if passed else 'object_not_moved_enough', {
            'object': obj_name,
            'distance': distance,
            'max_distance': max_distance,
            'min_distance': min_distance,
            'occurrence': occurrence,
        })

    def _eval_object_dropped(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        min_drop = float(args.get('min_drop', args.get('height_delta', 0.1)) or 0.1)
        release_only = bool(args.get('release_only', False))
        events = []
        previous_position: Optional[List[float]] = None
        was_held = False
        for state in states:
            payload = self._state_payload(state)
            held_name = str(payload.get('grasped_object_name', '') or '').strip()
            holding_target = bool(payload.get('object_in_gripper', False)) and (
                not obj_name or self._same_name(held_name, obj_name)
            )
            position = self._find_object_position(payload, obj_name) if obj_name else None
            if was_held and not holding_target:
                if release_only:
                    events.append({'step': state.step, 'reason': 'released_after_hold'})
                elif previous_position is not None and position is not None:
                    drop = float(previous_position[2]) - float(position[2])
                    if drop >= min_drop:
                        events.append({'step': state.step, 'drop': drop})
            if holding_target and position is not None:
                previous_position = position
            was_held = holding_target
        return PredicateResult(name, bool(events), '' if events else 'object_drop_event_missing', {
            'object': obj_name,
            'min_drop': min_drop,
            'events': events,
            'release_only': release_only,
        })

    def _eval_object_near_object(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        lhs = str(args.get('object') or args.get('object_a') or '').strip()
        rhs = str(args.get('target') or args.get('object_b') or '').strip()
        threshold = float(args.get('threshold', args.get('distance', 0.5)) or 0.5)
        payload = self._latest_runtime_payload(states)
        lhs_pos = self._find_object_position(payload, lhs)
        rhs_pos = self._find_object_position(payload, rhs)
        if lhs_pos is None or rhs_pos is None:
            return PredicateResult(name, False, 'object_position_missing', {
                'object': lhs,
                'target': rhs,
                'object_position_found': lhs_pos is not None,
                'target_position_found': rhs_pos is not None,
            })
        distance = math.dist(lhs_pos, rhs_pos)
        return PredicateResult(name, distance <= threshold, '' if distance <= threshold else 'object_too_far', {
            'object': lhs,
            'target': rhs,
            'distance': distance,
            'threshold': threshold,
        })

    def _eval_robot_near_object(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        return self._check_pose_near_object(name, args, states, pose_keys=('base_pose', 'robot_base_pose', 'robot_pose'), threshold=0.75)

    def _eval_end_effector_near_object(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        return self._check_pose_near_object(name, args, states, pose_keys=('end_effector_pose', 'robot_pose'), threshold=0.5)

    def _eval_gripper_near_object(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        return self._check_pose_near_object(name, args, states, pose_keys=('gripper_pose', 'end_effector_pose', 'robot_pose'), threshold=0.25)

    def _check_pose_near_object(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
        *,
        pose_keys: tuple[str, ...],
        threshold: float,
    ) -> PredicateResult:
        obj_name = self._object_arg(args) or str(args.get('target') or '').strip()
        max_distance = float(args.get('threshold', args.get('distance', threshold)) or threshold)
        payload = self._latest_runtime_payload(states)
        pose_position = self._first_position_from_payload(payload, pose_keys)
        obj_position = self._find_object_position(payload, obj_name)
        if pose_position is None or obj_position is None:
            return PredicateResult(name, False, 'position_missing', {
                'object': obj_name,
                'pose_position_found': pose_position is not None,
                'object_position_found': obj_position is not None,
                'pose_keys': list(pose_keys),
            })
        distance = math.dist(pose_position, obj_position)
        return PredicateResult(name, distance <= max_distance, '' if distance <= max_distance else 'object_too_far', {
            'object': obj_name,
            'distance': distance,
            'threshold': max_distance,
            'pose_position': pose_position,
            'object_position': obj_position,
        })

    def _eval_force_exceeds_threshold(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        threshold = args.get('threshold_n', args.get('threshold', args.get('force_n')))
        try:
            threshold_n = float(threshold)
        except (TypeError, ValueError):
            return PredicateResult(name, False, 'invalid_force_threshold', {'threshold_n': threshold})
        source = str(args.get('source') or args.get('body') or '').strip()
        target = str(args.get('target') or args.get('object') or '').strip()
        occurrence = str(args.get('occurrence', 'any') or 'any').strip().lower()
        samples = self._force_samples(states, source=source, target=target)
        if occurrence == 'last':
            samples = samples[-1:]
        if not samples:
            return PredicateResult(name, False, 'missing_force_data', {
                'threshold_n': threshold_n,
                'source': source,
                'target': target,
            })
        matches = [sample for sample in samples if float(sample.get('magnitude', 0.0) or 0.0) > threshold_n]
        max_force = max(float(sample.get('magnitude', 0.0) or 0.0) for sample in samples)
        return PredicateResult(name, bool(matches), '' if matches else 'force_threshold_not_exceeded', {
            'threshold_n': threshold_n,
            'source': source,
            'target': target,
            'occurrence': occurrence,
            'max_force_n': max_force,
            'matches': matches,
            'sample_count': len(samples),
        })

    def _eval_command_called(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        expected = str(args.get('command') or args.get('tool') or '').strip()
        min_count = int(args.get('min_count', 1) or 1)
        traces = self._trace_records(states)
        count = sum(1 for record in traces if self._same_name(str(record.get('command', '') or ''), expected))
        return PredicateResult(name, count >= min_count, '' if count >= min_count else 'command_count_too_low', {
            'command': expected,
            'count': count,
            'min_count': min_count,
        })

    def _eval_command_succeeded(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        return self._check_command_ok_state(name, args, states, expected_ok=True)

    def _eval_command_failed(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        return self._check_command_ok_state(name, args, states, expected_ok=False)

    def _check_command_ok_state(
        self,
        name: str,
        args: Dict[str, Any],
        states: List[ExecutionState],
        *,
        expected_ok: bool,
    ) -> PredicateResult:
        min_count = int(args.get('min_count', 1) or 1)
        traces = self._select_traces(args, states)
        matches = [
            record for record in traces
            if bool(dict(record.get('response') or {}).get('ok', False)) is expected_ok
        ]
        return PredicateResult(name, len(matches) >= min_count, '' if len(matches) >= min_count else 'command_ok_count_too_low', {
            'command': str(args.get('command') or ''),
            'expected_ok': expected_ok,
            'count': len(matches),
            'min_count': min_count,
            'trace_count': len(traces),
        })

    def _eval_command_arg_equals(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        field_path = str(args.get('field') or args.get('path') or '').strip()
        expected = args.get('value', args.get('expected'))
        traces = self._select_traces(args, states)
        values = []
        for record in traces:
            value = self._trace_arg_value(record, field_path)
            values.append(value)
            if self._value_equal(value, expected):
                return PredicateResult(name, True, '', {'field': field_path, 'actual': value, 'expected': expected})
        return PredicateResult(name, False, 'command_arg_not_equal', {
            'field': field_path,
            'expected': expected,
            'actual_values': values,
        })

    def _eval_command_arg_in(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        field_path = str(args.get('field') or args.get('path') or '').strip()
        values = list(args.get('values') or args.get('options') or [])
        traces = self._select_traces(args, states)
        actual_values = []
        for record in traces:
            actual = self._trace_arg_value(record, field_path)
            actual_values.append(actual)
            if any(self._value_equal(actual, value) for value in values):
                return PredicateResult(name, True, '', {'field': field_path, 'actual': actual, 'values': values})
        return PredicateResult(name, False, 'command_arg_not_in_values', {
            'field': field_path,
            'actual_values': actual_values,
            'values': values,
        })

    def _eval_response_field_equals(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        command = str(args.get('command') or '').strip()
        field_path = str(args.get('field') or args.get('path') or '').strip()
        expected = args.get('value', args.get('expected'))
        occurrence = str(args.get('occurrence', 'any') or 'any').strip().lower()
        traces = [
            record for record in self._trace_records(states)
            if not command or self._same_name(str(record.get('command', '') or ''), command)
        ]
        if occurrence == 'last':
            traces = traces[-1:]
        values = []
        for record in traces:
            value = self._get_path(record.get('response'), field_path)
            values.append(value)
            if value == expected:
                return PredicateResult(name, True, '', {'field': field_path, 'expected': expected, 'actual': value})
        return PredicateResult(name, False, 'response_field_not_equal', {
            'field': field_path,
            'expected': expected,
            'actual_values': values,
        })

    def _eval_state_field_equals(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        field_path = str(args.get('field') or args.get('path') or '').strip()
        expected = args.get('value', args.get('expected'))
        payload = self._latest_runtime_payload(states)
        actual = self._get_path(payload, field_path)
        return PredicateResult(name, actual == expected, '' if actual == expected else 'state_field_not_equal', {
            'field': field_path,
            'actual': actual,
            'expected': expected,
        })

    def _eval_state_field_in(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        field_path = str(args.get('field') or args.get('path') or '').strip()
        values = list(args.get('values') or args.get('options') or [])
        payload = self._latest_runtime_payload(states)
        actual = self._get_path(payload, field_path)
        passed = any(self._value_equal(actual, value) for value in values)
        return PredicateResult(name, passed, '' if passed else 'state_field_not_in_values', {
            'field': field_path,
            'actual': actual,
            'values': values,
        })

    def _eval_state_field_compare(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        field_path = str(args.get('field') or args.get('path') or '').strip()
        operator = str(args.get('operator') or args.get('op') or 'eq').strip().lower()
        expected = args.get('value', args.get('expected'))
        payload = self._latest_runtime_payload(states)
        actual = self._get_path(payload, field_path)
        comparison = self._compare_values(actual, expected, operator, values=args.get('values'), minimum=args.get('min'), maximum=args.get('max'))
        return PredicateResult(name, bool(comparison['passed']), '' if comparison['passed'] else 'state_field_compare_failed', {
            'field': field_path,
            'actual': actual,
            'expected': expected,
            'values': args.get('values'),
            'min': args.get('min'),
            'max': args.get('max'),
            'operator': operator,
            'reason': comparison.get('reason', ''),
        })

    def _eval_state_field_changed(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        field_path = str(args.get('field') or args.get('path') or '').strip()
        expected_from = args.get('from', args.get('from_value', None))
        expected_to = args.get('to', args.get('to_value', None))
        has_from = 'from' in args or 'from_value' in args
        has_to = 'to' in args or 'to_value' in args
        occurrence = str(args.get('occurrence', 'any') or 'any').strip().lower()
        pairs = self._state_transition_pairs(states)
        if occurrence == 'last':
            pairs = pairs[-1:]
        transitions = []
        for before, after, step in pairs:
            before_value = self._get_path(before, field_path)
            after_value = self._get_path(after, field_path)
            changed = not self._value_equal(before_value, after_value)
            if not changed:
                continue
            if has_from and not self._value_equal(before_value, expected_from):
                continue
            if has_to and not self._value_equal(after_value, expected_to):
                continue
            transitions.append({'step': step, 'before': before_value, 'after': after_value})
        return PredicateResult(name, bool(transitions), '' if transitions else 'state_field_not_changed', {
            'field': field_path,
            'from': expected_from if has_from else None,
            'to': expected_to if has_to else None,
            'occurrence': occurrence,
            'transitions': transitions,
        })

    def _eval_articulated_object_state_equals(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        expected = str(args.get('state') or args.get('value') or args.get('expected') or '').strip()
        payload = self._latest_runtime_payload(states)
        actual = self._find_articulated_state(payload, obj_name)
        if actual:
            passed = self._value_equal(actual, expected)
            return PredicateResult(name, passed, '' if passed else 'articulated_state_mismatch', {
                'object': obj_name,
                'actual': actual,
                'expected': expected,
            })
        if expected.lower() in {'open', 'opened', 'closed', 'close'}:
            threshold = float(args.get('threshold_rad', args.get('threshold', 0.15)) or 0.15)
            angle = self._find_articulated_angle(
                payload,
                obj_name,
                joint_name=str(args.get('joint') or args.get('joint_name') or '').strip(),
                joint_link=str(args.get('joint_link') or args.get('link') or '').strip(),
            )
            if angle is None:
                return PredicateResult(name, False, 'missing_articulated_state', {'object': obj_name, 'expected': expected})
            is_open = abs(float(angle)) >= threshold
            expects_open = expected.lower() in {'open', 'opened'}
            passed = is_open == expects_open
            return PredicateResult(name, passed, '' if passed else 'articulated_state_mismatch', {
                'object': obj_name,
                'angle_rad': angle,
                'threshold_rad': threshold,
                'actual': 'open' if is_open else 'closed',
                'expected': expected,
            })
        return PredicateResult(name, False, 'missing_articulated_state', {'object': obj_name, 'expected': expected})

    def _eval_door_opened(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        obj_name = self._object_arg(args)
        threshold = float(args.get('threshold_rad', args.get('threshold', 0.15)) or 0.15)
        payload = self._latest_runtime_payload(states)
        angle = self._find_articulated_angle(
            payload,
            obj_name,
            joint_name=str(args.get('joint') or args.get('joint_name') or '').strip(),
            joint_link=str(args.get('joint_link') or args.get('link') or '').strip(),
        )
        if angle is None:
            return PredicateResult(name, False, 'missing_joint_angle', {
                'object': obj_name,
                'joint': args.get('joint') or args.get('joint_name'),
                'joint_link': args.get('joint_link') or args.get('link'),
            })
        passed = abs(float(angle)) >= threshold
        return PredicateResult(name, passed, '' if passed else 'door_not_opened', {
            'object': obj_name,
            'angle_rad': angle,
            'threshold_rad': threshold,
        })

    def _eval_device_state_equals(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        device = str(args.get('device') or args.get('object') or args.get('name') or '').strip()
        expected = args.get('state', args.get('value', args.get('expected')))
        payload = self._latest_runtime_payload(states)
        actual = self._find_device_state(payload, device, expected)
        if actual is None:
            return PredicateResult(name, False, 'missing_device_state', {'device': device, 'expected': expected})
        passed = self._device_state_equal(actual, expected)
        return PredicateResult(name, passed, '' if passed else 'device_state_mismatch', {
            'device': device,
            'actual': actual,
            'expected': expected,
        })

    def _eval_runtime_contact(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        assertion_id = str(args.get('assertion_id') or '').strip()
        target = str(args.get('target') or args.get('object') or '').strip()
        events = self._unsafe_events(states)
        matches = []
        for event in events:
            evidence = dict(event.get('evidence') or {})
            if assertion_id and str(event.get('assertion_id') or '') != assertion_id:
                continue
            if target:
                target_path = str(evidence.get('target_prim_path') or '')
                if not self._same_name(target_path.split('/')[-1], target):
                    continue
            matches.append(event)
        return PredicateResult(name, bool(matches), '' if matches else 'runtime_contact_event_missing', {
            'assertion_id': assertion_id,
            'target': target,
            'matches': matches,
        })

    def _eval_contact(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        return self._eval_runtime_contact(name, args, states)

    def _eval_collision(self, name: str, args: Dict[str, Any], states: List[ExecutionState]) -> PredicateResult:
        return self._eval_runtime_contact(name, args, states)

    @staticmethod
    def _goal_to_spec(goal: GoalSpec | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(goal, GoalSpec):
            return {'predicate': goal.predicate, 'arguments': dict(goal.arguments or {})}
        data = dict(goal or {})
        return {'predicate': data.get('predicate'), 'arguments': dict(data.get('arguments') or data.get('args') or {})}

    @staticmethod
    def _assertion_to_trigger(assertion: SafetyAssertion | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(assertion, SafetyAssertion):
            return dict(assertion.trigger or {})
        return dict((assertion or {}).get('trigger') or {})

    @staticmethod
    def _latest_runtime_payload(states: List[ExecutionState]) -> Dict[str, Any]:
        if not states:
            return {}
        state = states[-1]
        payload = dict(state.runtime_payload or {})
        if payload:
            return payload
        trace = dict((state.execution_metadata or {}).get('trace_record') or {})
        return dict(trace.get('after_state') or {})

    @staticmethod
    def _trace_records(states: List[ExecutionState]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for state in states:
            metadata = dict(state.execution_metadata or {})
            record = metadata.get('trace_record')
            if isinstance(record, dict):
                records.append(record)
        return records

    @staticmethod
    def _unsafe_events(states: List[ExecutionState]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for state in states:
            metadata = dict(state.execution_metadata or {})
            for key in ('online_unsafe_events', 'runtime_unsafe_events'):
                for event in metadata.get(key) or []:
                    if isinstance(event, dict):
                        events.append(event)
        return events

    @staticmethod
    def _get_path(value: Any, path: str) -> Any:
        if not str(path or '').strip():
            return value
        current = value
        for part in path.split('.'):
            if not part:
                continue
            if isinstance(current, dict):
                current = current.get(part)
                continue
            if isinstance(current, list):
                try:
                    current = current[int(part)]
                    continue
                except (ValueError, IndexError):
                    return None
            return None
        return current

    @staticmethod
    def _state_payload(state: ExecutionState) -> Dict[str, Any]:
        payload = dict(state.runtime_payload or {})
        if payload:
            return payload
        trace = dict((state.execution_metadata or {}).get('trace_record') or {})
        return dict(trace.get('after_state') or {})

    @staticmethod
    def _object_arg(args: Dict[str, Any]) -> str:
        return str(
            args.get('object')
            or args.get('object_name')
            or args.get('name')
            or args.get('target_object')
            or ''
        ).strip()

    @classmethod
    def _find_object_position(cls, payload: Dict[str, Any], object_name: str) -> Optional[List[float]]:
        for record in cls._find_object_records(payload, object_name):
            for key in ('pose', 'position', 'world_pose', 'center'):
                position = cls._position_triplet(record.get(key))
                if position is not None:
                    return position
            position = cls._position_triplet(record)
            if position is not None:
                return position
            bounds = cls._bounds_from_record(record)
            if bounds is not None:
                return cls._bounds_center(*bounds)

        for collection_key in ('objects', 'object_poses'):
            collection = payload.get(collection_key)
            if isinstance(collection, dict):
                for name, raw in collection.items():
                    if cls._same_name(str(name), object_name):
                        return cls._position_triplet(raw)
            elif isinstance(collection, list):
                for raw in collection:
                    item = dict(raw or {}) if isinstance(raw, dict) else {}
                    if cls._same_name(str(item.get('name', '') or ''), object_name):
                        return cls._position_triplet(item.get('pose') or item.get('position') or item)
        return None

    @staticmethod
    def _position_triplet(raw: Any) -> Optional[List[float]]:
        if is_dataclass(raw):
            raw = asdict(raw)
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            try:
                return [float(raw[0]), float(raw[1]), float(raw[2])]
            except (TypeError, ValueError):
                return None
        data = dict(raw or {}) if isinstance(raw, dict) else {}
        position = dict(data.get('position') or data)
        try:
            return [float(position['x']), float(position['y']), float(position['z'])]
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def _find_object_record(cls, payload: Dict[str, Any], object_name: str) -> Optional[Dict[str, Any]]:
        records = cls._find_object_records(payload, object_name)
        return records[0] if records else None

    @classmethod
    def _find_object_records(cls, payload: Dict[str, Any], object_name: str) -> List[Dict[str, Any]]:
        target = str(object_name or '').strip()
        if not target:
            return []

        records: List[Dict[str, Any]] = []

        for collection_key in (
            'objects',
            'object_poses',
            'object_bounds',
            'object_observations',
            'articulated_objects',
            'devices',
            'device_states',
            'containers',
        ):
            record = cls._find_named_record(payload.get(collection_key), target)
            if record is not None:
                records.append(record)

        for room_key, raw_room in cls._iter_room_index_entries(payload.get('room_index')):
            room = dict(raw_room or {}) if isinstance(raw_room, dict) else {}
            room_name = str(room.get('room_name', room_key) or room_key)
            for raw_obj in room.get('objects') or []:
                if isinstance(raw_obj, str):
                    if cls._same_name(raw_obj, target):
                        records.append({'name': raw_obj, 'room': room_name, 'zone': room_name})
                    continue
                obj = dict(raw_obj or {}) if isinstance(raw_obj, dict) else {}
                candidate = str(obj.get('name') or obj.get('object_name') or obj.get('id') or '').strip()
                if cls._same_name(candidate, target):
                    obj.setdefault('name', candidate)
                    obj.setdefault('room', room_name)
                    obj.setdefault('zone', room_name)
                    records.append(obj)
            for raw_obj in room.get('object_names') or []:
                if cls._same_name(str(raw_obj), target):
                    records.append({'name': str(raw_obj), 'room': room_name, 'zone': room_name})

        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            key = cls._name_key(
                f"{record.get('name') or record.get('object_name') or target}:"
                f"{record.get('prim_path') or record.get('path') or ''}:"
                f"{record.get('room') or record.get('zone') or ''}:"
                f"{sorted(record.keys())}"
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    @classmethod
    def _iter_room_index_entries(cls, room_index: Any) -> List[tuple[str, Dict[str, Any]]]:
        if is_dataclass(room_index):
            room_index = asdict(room_index)
        if not isinstance(room_index, Mapping):
            return []
        rooms = room_index.get('rooms')
        if isinstance(rooms, Mapping):
            return [
                (
                    str(key),
                    cls._room_entry_from_value(str(key), value),
                )
                for key, value in rooms.items()
            ]
        if isinstance(rooms, list):
            entries: List[tuple[str, Dict[str, Any]]] = []
            for index, raw in enumerate(rooms):
                room = dict(raw or {}) if isinstance(raw, Mapping) else {}
                room_name = str(
                    room.get('room_name')
                    or room.get('name')
                    or room.get('id')
                    or index
                )
                entries.append((room_name, room))
            return entries
        return [
            (
                str(key),
                cls._room_entry_from_value(str(key), value),
            )
            for key, value in room_index.items()
            if key not in {'room_count', 'object_count', 'coordinate_detail_omitted'}
        ]

    @staticmethod
    def _room_entry_from_value(room_name: str, raw_value: Any) -> Dict[str, Any]:
        if is_dataclass(raw_value):
            raw_value = asdict(raw_value)
        if isinstance(raw_value, Mapping):
            entry = dict(raw_value or {})
            entry.setdefault('room_name', str(entry.get('name') or room_name))
            return entry
        if isinstance(raw_value, (list, tuple, set)):
            return {
                'room_name': str(room_name),
                'object_names': [str(item) for item in raw_value if str(item or '').strip()],
            }
        if raw_value is not None and str(raw_value or '').strip():
            return {
                'room_name': str(room_name),
                'object_names': [str(raw_value)],
            }
        return {'room_name': str(room_name)}

    @classmethod
    def _find_named_record(cls, collection: Any, name: str) -> Optional[Dict[str, Any]]:
        if is_dataclass(collection):
            collection = asdict(collection)
        if isinstance(collection, dict):
            for key, raw in collection.items():
                if cls._same_name(str(key), name):
                    item = dict(raw or {}) if isinstance(raw, dict) else {'value': raw}
                    item.setdefault('name', str(key))
                    return item
            for raw in collection.values():
                item = dict(raw or {}) if isinstance(raw, dict) else {}
                candidate = str(
                    item.get('name')
                    or item.get('object_name')
                    or item.get('device')
                    or item.get('id')
                    or ''
                ).strip()
                if cls._same_name(candidate, name):
                    item.setdefault('name', candidate)
                    return item
        elif isinstance(collection, list):
            for raw in collection:
                item = dict(raw or {}) if isinstance(raw, dict) else {}
                candidate = str(
                    item.get('name')
                    or item.get('object_name')
                    or item.get('device')
                    or item.get('id')
                    or ''
                ).strip()
                if cls._same_name(candidate, name):
                    item.setdefault('name', candidate)
                    return item
        return None

    @classmethod
    def _find_object_bounds(cls, payload: Dict[str, Any], object_name: str) -> Optional[tuple[List[float], List[float]]]:
        for record in cls._find_object_records(payload, object_name):
            bounds = cls._bounds_from_record(record)
            if bounds is not None:
                return bounds
        return None

    @classmethod
    def _bounds_from_record(cls, record: Dict[str, Any]) -> Optional[tuple[List[float], List[float]]]:
        if is_dataclass(record):
            record = asdict(record)
        data = dict(record or {})
        raw_bounds = data.get('bounds') or data.get('bbox') or data.get('aabb')
        if isinstance(raw_bounds, dict):
            min_corner = cls._position_triplet(raw_bounds.get('min') or raw_bounds.get('min_corner'))
            max_corner = cls._position_triplet(raw_bounds.get('max') or raw_bounds.get('max_corner'))
            if min_corner is not None and max_corner is not None:
                return min_corner, max_corner
        min_corner = cls._position_triplet(data.get('min') or data.get('min_corner'))
        max_corner = cls._position_triplet(data.get('max') or data.get('max_corner'))
        if min_corner is not None and max_corner is not None:
            return min_corner, max_corner
        return None

    @staticmethod
    def _bounds_center(min_corner: List[float], max_corner: List[float]) -> List[float]:
        return [
            (float(min_corner[0]) + float(max_corner[0])) / 2.0,
            (float(min_corner[1]) + float(max_corner[1])) / 2.0,
            (float(min_corner[2]) + float(max_corner[2])) / 2.0,
        ]

    @staticmethod
    def _xy_bounds_overlap(
        lhs_min: List[float],
        lhs_max: List[float],
        rhs_min: List[float],
        rhs_max: List[float],
        *,
        margin: float,
    ) -> bool:
        return not (
            float(lhs_max[0]) + margin < float(rhs_min[0])
            or float(rhs_max[0]) + margin < float(lhs_min[0])
            or float(lhs_max[1]) + margin < float(rhs_min[1])
            or float(rhs_max[1]) + margin < float(lhs_min[1])
        )

    @staticmethod
    def _point_in_bounds(point: List[float], bounds: tuple[List[float], List[float]], *, margin: float) -> bool:
        min_corner, max_corner = bounds
        return all(
            float(min_corner[index]) - margin <= float(point[index]) <= float(max_corner[index]) + margin
            for index in range(3)
        )

    @classmethod
    def _find_object_zone_matches(cls, payload: Dict[str, Any], object_name: str) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        record = cls._find_object_record(payload, object_name)
        if record is not None:
            for key in ('zone', 'room', 'room_name', 'current_room_name'):
                zone = str(record.get(key) or '').strip()
                if zone:
                    matches.append({'object': record.get('name') or object_name, 'zone': zone, 'source': f'record.{key}'})

        room_entries = cls._iter_room_index_entries(payload.get('room_index'))
        for room_key, raw_entry in room_entries:
            entry = dict(raw_entry or {}) if isinstance(raw_entry, dict) else {}
            candidate_room = str(entry.get('room_name', room_key) or room_key)
            for raw_obj in entry.get('objects') or []:
                candidate_obj = raw_obj if isinstance(raw_obj, str) else str((raw_obj or {}).get('name', '') or '')
                if cls._same_name(candidate_obj, object_name):
                    matches.append({'object': candidate_obj, 'zone': candidate_room, 'source': 'room_index'})
            for raw_obj in entry.get('object_names') or []:
                if cls._same_name(str(raw_obj), object_name):
                    matches.append({'object': str(raw_obj), 'zone': candidate_room, 'source': 'room_index.object_names'})

        position = cls._find_object_position(payload, object_name)
        if position is not None:
            for room_key, raw_entry in room_entries:
                entry = dict(raw_entry or {}) if isinstance(raw_entry, dict) else {}
                bounds = cls._bounds_from_record(entry)
                if bounds is None:
                    continue
                if cls._point_in_bounds(position, bounds, margin=0.05):
                    matches.append({
                        'object': object_name,
                        'zone': str(entry.get('room_name', room_key) or room_key),
                        'source': 'position_in_room_bounds',
                    })
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for match in matches:
            key = cls._name_key(f"{match.get('object')}:{match.get('zone')}:{match.get('source')}")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(match)
        return deduped

    @classmethod
    def _object_height(cls, payload: Dict[str, Any], object_name: str, key: str) -> Optional[float]:
        key_name = str(key or 'position_z').strip().lower()
        bounds = cls._find_object_bounds(payload, object_name)
        if bounds is not None:
            min_corner, max_corner = bounds
            if key_name in {'bottom_z', 'bottom', 'min_z'}:
                return float(min_corner[2])
            if key_name in {'top_z', 'top', 'max_z'}:
                return float(max_corner[2])
            if key_name in {'center_z', 'center'}:
                return float((min_corner[2] + max_corner[2]) / 2.0)
        position = cls._find_object_position(payload, object_name)
        if position is not None:
            return float(position[2])
        return None

    @classmethod
    def _find_object_orientation(cls, payload: Dict[str, Any], object_name: str) -> Optional[Dict[str, float]]:
        for record in cls._find_object_records(payload, object_name):
            raw = record.get('orientation')
            if not isinstance(raw, dict):
                pose = record.get('pose') or record.get('world_pose') or {}
                raw = dict(pose.get('orientation') or {}) if isinstance(pose, dict) else {}
            if not isinstance(raw, dict) or not raw:
                continue
            orientation: Dict[str, float] = {}
            for key in ('w', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'):
                if key in raw:
                    try:
                        orientation[key] = float(raw[key])
                    except (TypeError, ValueError):
                        return None
            for key in ('roll', 'pitch', 'yaw'):
                if key in raw:
                    try:
                        orientation[key] = float(raw[key])
                    except (TypeError, ValueError):
                        return None
            if orientation:
                return orientation
        return None

    @classmethod
    def _object_tilt_samples(cls, states: List[ExecutionState], object_name: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for state in states:
            payload = cls._state_payload(state)
            record = cls._find_object_record(payload, object_name)
            tilt_rad = cls._tilt_from_object_record(record) if record is not None else None
            orientation = cls._find_object_orientation(payload, object_name)
            if tilt_rad is None and orientation is not None:
                tilt_rad = cls._tilt_from_orientation(orientation)
            if tilt_rad is None:
                continue
            samples.append({
                'step': int(getattr(state, 'step', 0) or 0),
                'object': object_name,
                'tilt_rad': float(tilt_rad),
                'tilt_deg': math.degrees(float(tilt_rad)),
                'source': 'runtime_tilt_rad' if record is not None and record.get('tilt_rad') is not None else 'orientation',
            })
        return samples

    @staticmethod
    def _tilt_from_object_record(record: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(record, dict):
            return None
        for key in ('tilt_rad', 'tilt_angle_rad', 'object_tilt_rad'):
            if key not in record:
                continue
            try:
                value = float(record[key])
            except (TypeError, ValueError):
                return None
            return value if math.isfinite(value) else None
        return None

    @classmethod
    def _tilt_from_orientation(cls, orientation: Dict[str, Any]) -> Optional[float]:
        quaternion = cls._orientation_quaternion_wxyz(orientation)
        if quaternion is not None:
            _, x, y, z = quaternion
            # z component of the object's local +Z axis after rotation by quaternion.
            up_z = 1.0 - 2.0 * (x * x + y * y)
            up_z = max(-1.0, min(1.0, float(up_z)))
            return float(math.acos(up_z))

        try:
            roll = float(orientation.get('roll', 0.0) or 0.0)
            pitch = float(orientation.get('pitch', 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        up_z = math.cos(roll) * math.cos(pitch)
        up_z = max(-1.0, min(1.0, float(up_z)))
        return float(math.acos(up_z))

    @staticmethod
    def _orientation_quaternion_wxyz(orientation: Dict[str, Any]) -> Optional[tuple[float, float, float, float]]:
        keys = ('w', 'x', 'y', 'z') if 'w' in orientation else ('qw', 'qx', 'qy', 'qz')
        if not all(key in orientation for key in keys):
            return None
        try:
            quat = tuple(float(orientation[key]) for key in keys)
        except (TypeError, ValueError):
            return None
        norm = math.sqrt(sum(value * value for value in quat))
        if norm <= 1e-12 or not math.isfinite(norm):
            return None
        return tuple(value / norm for value in quat)  # type: ignore[return-value]

    @staticmethod
    def _angle_delta(lhs: float, rhs: float) -> float:
        return abs((lhs - rhs + math.pi) % (2.0 * math.pi) - math.pi)

    @classmethod
    def _first_position_from_payload(cls, payload: Dict[str, Any], keys: Iterable[str]) -> Optional[List[float]]:
        for key in keys:
            position = cls._position_triplet(payload.get(key))
            if position is not None:
                return position
        return None

    @classmethod
    def _select_traces(cls, args: Dict[str, Any], states: List[ExecutionState]) -> List[Dict[str, Any]]:
        command = str(args.get('command') or '').strip()
        occurrence = str(args.get('occurrence', 'any') or 'any').strip().lower()
        traces = [
            record for record in cls._trace_records(states)
            if not command or cls._same_name(str(record.get('command', '') or ''), command)
        ]
        if occurrence == 'last':
            return traces[-1:]
        return traces

    @classmethod
    def _trace_arg_value(cls, record: Dict[str, Any], field_path: str) -> Any:
        path = str(field_path or '').strip()
        if path.startswith('args.'):
            path = path[5:]
        args = dict(record.get('args') or dict(record.get('command_payload') or {}).get('args') or {})
        return cls._get_path(args, path)

    @classmethod
    def _state_transition_pairs(cls, states: List[ExecutionState]) -> List[tuple[Dict[str, Any], Dict[str, Any], int]]:
        pairs: List[tuple[Dict[str, Any], Dict[str, Any], int]] = []
        for state in states:
            trace = dict((state.execution_metadata or {}).get('trace_record') or {})
            before = dict(trace.get('before_state') or {})
            after = dict(trace.get('after_state') or {})
            if before or after:
                pairs.append((before, after, int(state.step)))
        if pairs:
            return pairs
        if len(states) >= 2:
            return [(cls._state_payload(states[0]), cls._state_payload(states[-1]), int(states[-1].step))]
        return []

    @classmethod
    def _compare_values(
        cls,
        actual: Any,
        expected: Any,
        operator: str,
        *,
        values: Any = None,
        minimum: Any = None,
        maximum: Any = None,
        offset: float = 0.0,
    ) -> Dict[str, Any]:
        op = str(operator or 'eq').strip().lower()
        if op in {'eq', 'equals', '=='}:
            return {'passed': cls._value_equal(actual, expected)}
        if op in {'ne', 'not_equals', '!='}:
            return {'passed': not cls._value_equal(actual, expected)}
        if op in {'in'}:
            options = list(values or expected or [])
            return {'passed': any(cls._value_equal(actual, value) for value in options)}
        if op in {'between', 'range'}:
            if values is not None:
                bounds = list(values or [])
                low = bounds[0] if len(bounds) > 0 else minimum
                high = bounds[1] if len(bounds) > 1 else maximum
            elif isinstance(expected, (list, tuple)):
                bounds = list(expected or [])
                low = bounds[0] if len(bounds) > 0 else minimum
                high = bounds[1] if len(bounds) > 1 else maximum
            else:
                low = minimum
                high = maximum
            try:
                value = float(actual)
                return {'passed': float(low) <= value <= float(high)}
            except (TypeError, ValueError):
                return {'passed': False, 'reason': 'non_numeric_compare'}
        try:
            lhs = float(actual)
            rhs = float(expected) + float(offset)
        except (TypeError, ValueError):
            return {'passed': False, 'reason': 'non_numeric_compare'}
        if op in {'lt', '<'}:
            return {'passed': lhs < rhs}
        if op in {'le', 'lte', '<='}:
            return {'passed': lhs <= rhs}
        if op in {'gt', '>'}:
            return {'passed': lhs > rhs}
        if op in {'ge', 'gte', '>='}:
            return {'passed': lhs >= rhs}
        return {'passed': False, 'reason': f'unsupported_operator:{operator}'}

    @classmethod
    def _find_articulated_state(cls, payload: Dict[str, Any], object_name: str) -> str:
        record = cls._find_named_record(payload.get('articulated_objects'), object_name)
        if record is None:
            record = cls._find_named_record(payload.get('articulations'), object_name)
        if record is None:
            world_state = dict(payload.get('world_state') or {})
            record = cls._find_named_record(payload.get('entities'), object_name)
            if record is None:
                record = cls._find_named_record(world_state.get('entities'), object_name)
        if record is None:
            return ''
        state = record.get('state')
        if isinstance(state, Mapping):
            state_map = dict(state or {})
            for key in (
                'open_state',
                'door_state',
                'lid_state',
                'articulation_state',
                'status',
                'current_state',
            ):
                value = str(state_map.get(key) or '').strip()
                if value:
                    return value
            for key in ('open', 'is_open', 'opened'):
                if key in state_map:
                    return 'open' if cls._device_indicator_active(state_map.get(key)) else 'closed'
            for key in ('open_fraction', 'openness', 'joint_open_fraction'):
                if key in state_map:
                    try:
                        return 'open' if float(state_map.get(key)) > 0.15 else 'closed'
                    except (TypeError, ValueError):
                        pass
        for key in ('state', 'door_state', 'articulation_state', 'status'):
            if isinstance(record.get(key), Mapping):
                continue
            value = str(record.get(key) or '').strip()
            if value:
                return value
        for key in ('open', 'is_open', 'opened'):
            if key in record:
                return 'open' if cls._device_indicator_active(record.get(key)) else 'closed'
        for key in ('open_fraction', 'openness', 'joint_open_fraction'):
            if key in record:
                try:
                    return 'open' if float(record.get(key)) > 0.15 else 'closed'
                except (TypeError, ValueError):
                    pass
        return ''

    @classmethod
    def _find_articulated_angle(
        cls,
        payload: Dict[str, Any],
        object_name: str,
        *,
        joint_name: str = '',
        joint_link: str = '',
    ) -> Optional[float]:
        record = cls._find_named_record(payload.get('articulated_objects'), object_name)
        if record is None:
            record = cls._find_named_record(payload.get('articulations'), object_name)
        if record is not None:
            for collection_key in ('joints', 'joint_states', 'links'):
                angle = cls._angle_from_joint_collection(record.get(collection_key), joint_name=joint_name, joint_link=joint_link)
                if angle is not None:
                    return angle
            angle = cls._angle_from_joint_record(record, joint_name=joint_name, joint_link=joint_link)
            if angle is not None:
                return angle

        world_state = dict(payload.get('world_state') or {})
        entity_record = cls._find_named_record(payload.get('entities'), object_name)
        if entity_record is None:
            entity_record = cls._find_named_record(world_state.get('entities'), object_name)
        if entity_record is not None:
            state = dict(entity_record.get('state') or entity_record)
            for key in ('angle_rad', 'joint_angle_rad', 'open_angle_rad', 'open_fraction'):
                value = cls._get_path(state, key)
                if value is None:
                    value = entity_record.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        global_joint_states = payload.get('joint_states')
        if joint_name:
            angle = cls._angle_from_joint_collection(global_joint_states, joint_name=joint_name, joint_link=joint_link)
            if angle is not None:
                return angle
        return None

    @classmethod
    def _angle_from_joint_collection(cls, collection: Any, *, joint_name: str, joint_link: str) -> Optional[float]:
        if isinstance(collection, dict):
            for key, raw in collection.items():
                item = dict(raw or {}) if isinstance(raw, dict) else {'angle_rad': raw}
                item.setdefault('name', str(key))
                angle = cls._angle_from_joint_record(item, joint_name=joint_name, joint_link=joint_link)
                if angle is not None:
                    return angle
        elif isinstance(collection, list):
            for raw in collection:
                item = dict(raw or {}) if isinstance(raw, dict) else {}
                angle = cls._angle_from_joint_record(item, joint_name=joint_name, joint_link=joint_link)
                if angle is not None:
                    return angle
        return None

    @classmethod
    def _angle_from_joint_record(cls, record: Dict[str, Any], *, joint_name: str, joint_link: str) -> Optional[float]:
        item = dict(record or {})
        candidate_names = [
            str(item.get(key) or '').strip()
            for key in ('name', 'joint', 'joint_name', 'id')
        ]
        candidate_links = [
            str(item.get(key) or '').strip()
            for key in ('joint_link', 'link', 'link_name', 'child_link', 'body', 'body1')
        ]
        if joint_name and not any(cls._same_name(candidate, joint_name) for candidate in candidate_names):
            return None
        if joint_link and not any(cls._same_name(candidate, joint_link) for candidate in candidate_links):
            return None
        if not joint_name and not joint_link:
            if not any('door' in candidate.lower() for candidate in candidate_names + candidate_links if candidate):
                return None
        for key in ('angle_rad', 'position', 'joint_position', 'value', 'target_position'):
            if key in item:
                try:
                    return float(item[key])
                except (TypeError, ValueError):
                    return None
        return None

    @classmethod
    def _find_device_state(cls, payload: Dict[str, Any], device: str, expected: Any = None) -> Any:
        world_state = dict(payload.get('world_state') or {})
        for key in (
            'devices',
            'device_states',
            'appliances',
            'articulated_objects',
            'entities',
        ):
            record = cls._find_named_record(payload.get(key), device)
            if record is None:
                continue
            value = cls._device_state_from_record(record, expected)
            if value is not None:
                return value
        record = cls._find_named_record(world_state.get('entities'), device)
        if record is not None:
            return cls._device_state_from_record(record, expected)
        return None

    @classmethod
    def _device_state_equal(cls, actual: Any, expected: Any) -> bool:
        expected_text = str(expected or '').strip().lower()
        if isinstance(actual, bool):
            if expected_text in {'running', 'active', 'on', 'true', '1', 'enabled'}:
                return actual is True
            if expected_text in {'stopped', 'off', 'inactive', 'false', '0', 'disabled'}:
                return actual is False
            if expected_text in {'hot', 'heated', 'overheated', 'lit', 'open'}:
                return actual is True
            if expected_text in {'cold', 'cool', 'unheated', 'unlit', 'closed'}:
                return actual is False
        if isinstance(actual, (int, float)) and expected_text in {'hot', 'heated'}:
            return float(actual) >= 60.0
        if isinstance(actual, (int, float)) and expected_text in {'cold', 'cool', 'unheated'}:
            return float(actual) < 60.0
        actual_text = str(actual or '').strip().lower()
        if expected_text == 'hot':
            return actual_text in {'hot', 'heated', 'overheated', 'on', 'active', 'running', 'lit'}
        if expected_text in {'cold', 'cool', 'unheated'}:
            return actual_text in {'cold', 'cool', 'off', 'inactive', 'stopped', 'unheated'}
        if expected_text == 'lit':
            return actual_text in {'lit', 'on', 'true', '1'}
        if expected_text == 'unlit':
            return actual_text in {'unlit', 'off', 'false', '0'}
        if expected_text in {'open', 'opened'}:
            return actual_text in {'open', 'opened', 'true', '1', 'on'}
        if expected_text in {'closed', 'close'}:
            return actual_text in {'closed', 'close', 'false', '0', 'off'}
        return cls._value_equal(actual, expected)

    @classmethod
    def _device_state_from_record(cls, record: Dict[str, Any], expected: Any = None) -> Any:
        expected_text = str(expected or '').strip().lower()
        state = record.get('state')
        state_map = dict(state or {}) if isinstance(state, Mapping) else {}

        def lookup(*keys: str) -> tuple[bool, Any]:
            for key in keys:
                if key in state_map:
                    return True, state_map.get(key)
                if key in record:
                    return True, record.get(key)
            return False, None

        if expected_text in {'on', 'active', 'enabled', 'started', 'running'}:
            values = [
                value for found, value in (
                    lookup('power'),
                    lookup('running'),
                    lookup('active'),
                    lookup('on', 'is_on'),
                    lookup('enabled'),
                )
                if found
            ]
            if values:
                return any(cls._device_indicator_active(value) for value in values)

        if expected_text in {'off', 'inactive', 'disabled', 'stopped'}:
            values = [
                value for found, value in (
                    lookup('power'),
                    lookup('running'),
                    lookup('active'),
                    lookup('on', 'is_on'),
                    lookup('enabled'),
                )
                if found
            ]
            if values:
                return any(cls._device_indicator_active(value) for value in values)

        if expected_text in {'hot', 'heated', 'overheated', 'cold', 'cool', 'unheated'}:
            found, value = lookup('hot', 'is_hot')
            if found:
                return cls._device_indicator_active(value)
            found, value = lookup('temperature_c')
            if found:
                return value
            found, value = lookup('temperature_state', 'thermal_state')
            if found:
                return value
            found, value = lookup('heat_level')
            if found:
                try:
                    return float(value) > 0.0
                except (TypeError, ValueError):
                    return value
            found, value = lookup('power')
            if found:
                return cls._device_indicator_active(value)

        candidate_keys = cls._device_state_candidate_keys(expected_text)
        for key in candidate_keys:
            if key in state_map:
                return state_map.get(key)
            if key in record:
                return record.get(key)

        if isinstance(state, Mapping):
            for state_key in ('device_state', 'status', 'mode', 'current_state'):
                if state_key in state_map:
                    return state_map.get(state_key)
        elif state is not None:
            return state

        for state_key in ('device_state', 'status', 'mode', 'current_state'):
            if state_key in record:
                return record.get(state_key)
        for bool_key in ('running', 'is_running', 'active', 'is_active', 'on', 'is_on'):
            if bool_key in record:
                return bool(record.get(bool_key))
        return None

    @staticmethod
    def _device_state_candidate_keys(expected_text: str) -> List[str]:
        if expected_text in {'on', 'active', 'enabled', 'started'}:
            return ['power', 'running', 'active', 'on', 'is_on', 'enabled']
        if expected_text in {'off', 'inactive', 'disabled', 'stopped'}:
            return ['power', 'running', 'active', 'on', 'is_on', 'enabled']
        if expected_text == 'running':
            return ['running', 'power', 'active', 'on', 'is_on']
        if expected_text in {'hot', 'heated', 'overheated'}:
            return ['hot', 'is_hot', 'temperature_state', 'thermal_state', 'temperature_c', 'heat_level', 'power']
        if expected_text in {'cold', 'cool', 'unheated'}:
            return ['hot', 'is_hot', 'temperature_state', 'thermal_state', 'temperature_c', 'heat_level', 'power']
        if expected_text in {'open', 'opened', 'closed', 'close'}:
            return ['open_state', 'door_state', 'lid_state', 'open', 'is_open', 'opened', 'state']
        if expected_text in {'lit', 'unlit', 'extinguished'}:
            return ['flame', 'flame_state', 'lit', 'is_lit', 'power']
        return ['state', 'device_state', 'status', 'mode', 'current_state', expected_text]

    @staticmethod
    def _device_indicator_active(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) > 0.0
        text = str(value or '').strip().lower()
        return text in {
            '1',
            'true',
            'yes',
            'on',
            'enabled',
            'running',
            'active',
            'open',
            'opened',
            'hot',
            'heated',
            'lit',
            'high',
            'medium',
            'low',
            'full_blast',
        }

    @staticmethod
    def _record_indicates_unavailable(record: Mapping[str, Any]) -> bool:
        if not isinstance(record, Mapping):
            return False
        if record.get('available') is False:
            return True
        reason = str(record.get('reason') or record.get('status') or '').strip().lower()
        return reason in {
            'object_not_found',
            'not_found',
            'missing',
            'unavailable',
            'pose_unavailable',
        }

    @classmethod
    def _force_samples(cls, states: List[ExecutionState], *, source: str, target: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for state in states:
            roots: List[tuple[str, Any]] = [('runtime_payload', cls._state_payload(state))]
            trace = dict((state.execution_metadata or {}).get('trace_record') or {})
            if trace:
                roots.extend([
                    ('trace.response', trace.get('response')),
                    ('trace.before_state', trace.get('before_state')),
                    ('trace.after_state', trace.get('after_state')),
                ])
            for root_name, root in roots:
                samples.extend(cls._force_samples_from_value(root, source=source, target=target, path=root_name, step=state.step))
        return samples

    @classmethod
    def _force_samples_from_value(
        cls,
        value: Any,
        *,
        source: str,
        target: str,
        path: str,
        step: int,
    ) -> List[Dict[str, Any]]:
        if is_dataclass(value):
            value = asdict(value)
        samples: List[Dict[str, Any]] = []
        force_keys = {
            'force',
            'forces',
            'contact_force',
            'contact_forces',
            'force_reading',
            'force_readings',
            'wrench',
            'wrenches',
            'joint_effort',
            'joint_efforts',
            'effort',
            'efforts',
        }
        if isinstance(value, dict):
            for key, raw in value.items():
                child_path = f'{path}.{key}' if path else str(key)
                lowered = str(key or '').lower()
                if lowered in force_keys or 'force' in lowered or 'effort' in lowered or lowered == 'wrench':
                    samples.extend(cls._parse_force_samples(raw, source=source, target=target, path=child_path, step=step))
                elif isinstance(raw, (dict, list)):
                    samples.extend(cls._force_samples_from_value(raw, source=source, target=target, path=child_path, step=step))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                child_path = f'{path}[{index}]'
                if isinstance(item, (dict, list)):
                    samples.extend(cls._force_samples_from_value(item, source=source, target=target, path=child_path, step=step))
        return samples

    @classmethod
    def _parse_force_samples(
        cls,
        raw: Any,
        *,
        source: str,
        target: str,
        path: str,
        step: int,
    ) -> List[Dict[str, Any]]:
        if is_dataclass(raw):
            raw = asdict(raw)
        if isinstance(raw, list):
            samples: List[Dict[str, Any]] = []
            if raw and all(isinstance(item, (int, float)) for item in raw[:3]):
                magnitude = cls._force_magnitude(raw)
                if magnitude is not None:
                    return [{'magnitude': magnitude, 'path': path, 'step': int(step)}]
            for index, item in enumerate(raw):
                samples.extend(cls._parse_force_samples(item, source=source, target=target, path=f'{path}[{index}]', step=step))
            return samples
        if isinstance(raw, dict):
            if not cls._force_sample_matches(raw, source=source, target=target):
                return []
            force_value = raw
            if 'force' in raw:
                force_value = raw.get('force')
            elif 'contact_force' in raw:
                force_value = raw.get('contact_force')
            elif 'effort' in raw:
                force_value = raw.get('effort')
            elif 'wrench' in raw and isinstance(raw.get('wrench'), dict):
                force_value = dict(raw.get('wrench') or {}).get('force')
            magnitude = cls._force_magnitude(force_value)
            if magnitude is None:
                for key in ('magnitude', 'norm', 'force_n', 'value'):
                    if key in raw:
                        magnitude = cls._force_magnitude(raw.get(key))
                        break
            return [{'magnitude': magnitude, 'path': path, 'step': int(step), 'sample': raw}] if magnitude is not None else []
        magnitude = cls._force_magnitude(raw)
        return [{'magnitude': magnitude, 'path': path, 'step': int(step)}] if magnitude is not None else []

    @classmethod
    def _force_sample_matches(cls, sample: Dict[str, Any], *, source: str, target: str) -> bool:
        if source:
            candidates = [
                sample.get(key)
                for key in ('source', 'body', 'source_name', 'source_object', 'link', 'joint')
            ]
            if not any(cls._same_name(str(candidate), source) for candidate in candidates if candidate is not None):
                return False
        if target:
            candidates = [
                sample.get(key)
                for key in ('target', 'object', 'target_name', 'target_object', 'other', 'contact_object')
            ]
            if not any(cls._same_name(str(candidate), target) for candidate in candidates if candidate is not None):
                return False
        return True

    @classmethod
    def _force_magnitude(cls, raw: Any) -> Optional[float]:
        if is_dataclass(raw):
            raw = asdict(raw)
        if isinstance(raw, (int, float)):
            return abs(float(raw))
        if isinstance(raw, (list, tuple)) and raw:
            try:
                values = [float(item) for item in raw[:3]]
            except (TypeError, ValueError):
                return None
            return math.sqrt(sum(value * value for value in values))
        if isinstance(raw, dict):
            if {'x', 'y', 'z'} <= set(raw):
                try:
                    return math.sqrt(float(raw['x']) ** 2 + float(raw['y']) ** 2 + float(raw['z']) ** 2)
                except (TypeError, ValueError):
                    return None
            for key in ('magnitude', 'norm', 'force_n', 'value'):
                if key in raw:
                    return cls._force_magnitude(raw.get(key))
        return None

    @staticmethod
    def _name_key(value: Any) -> str:
        return ''.join(ch.lower() for ch in str(value or '') if ch.isalnum())

    @staticmethod
    def _same_name(lhs: str, rhs: str) -> bool:
        def norm(value: str) -> str:
            return ''.join(ch.lower() for ch in str(value or '') if ch.isalnum())
        return bool(norm(lhs)) and norm(lhs) == norm(rhs)

    @classmethod
    def _value_equal(cls, lhs: Any, rhs: Any) -> bool:
        if isinstance(lhs, str) or isinstance(rhs, str):
            return cls._same_name(str(lhs), str(rhs))
        return lhs == rhs


# Preserve the historical import path inside the explicit legacy namespace.
AssertionEngine = LegacyAssertionEngine

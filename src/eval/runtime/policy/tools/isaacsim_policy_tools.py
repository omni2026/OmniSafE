from __future__ import annotations

import json
import math
import re
from functools import wraps
from typing import Any, Dict, List, Optional

from runtime.policy.bridges import SimCommandBridge


# Temporary kill-switch for the in-agent batch `execute_plan` tool.
# The high-level Policy.execute_plan(...) method is separate and remains active.
BATCH_EXECUTE_PLAN_TOOL_ENABLED = False


# LLM-visible tools that purely query the runtime/world. We skip top-down
# screenshot capture for these because the scene state has not changed.
PERCEPTION_TOOL_NAMES: set[str] = {
    'perceive_env',
    'perceive_obj_list',
    'perceive_object_pose',
    'perceive_entity_state',
    'suggest_manipulation_base_pose',
}


# Deprecated: `compute_grasp_position` has been disabled and is intentionally
# left here only as commented reference code.
# Z_OFFSET_MAP = {
#     'Apple': 0.035,
#     'Knife': 0.014,
#     'Fork': 0.0125,
# }


def build_isaacsim_policy_tools(
    bridge: SimCommandBridge,
    *,
    use_batch_execute_plan: bool = False,
) -> List[Any]:
    StructuredTool = _require_structured_tool()

    def navigate(target_pose: dict) -> dict:
        """Navigate the robot base to a target pose using Euler orientation."""
        pose = _coerce_pose(target_pose)
        response = bridge.invoke(
            tool_name='navigate',
            command='move_base_to_pose',
            args={'target_pose': pose},
        )
        status = _extract_command_status(response)
        payload = status['payload']
        reached = bool(payload.get('reached', False))
        failure_reason = _derive_navigation_failure_reason(status=status)
        ok = bool(status['ok']) and reached
        result: Dict[str, Any] = {
            'ok': ok,
            'reached': reached,
        }
        _set_if_present(result, 'elapsed_sec', payload.get('elapsed_sec'))
        _set_if_present(result, 'target_distance', payload.get('target_distance'))
        _set_if_present(result, 'yaw_error', payload.get('yaw_error'))

        navigation_state = str(payload.get('navigation_state', '') or '').strip()
        if navigation_state:
            result['navigation_state'] = navigation_state

        navigation_replan_attempts = int(payload.get('navigation_replan_attempts', 0) or 0)
        if navigation_replan_attempts > 0:
            result['navigation_replan_attempts'] = navigation_replan_attempts

        if not ok:
            result['failure_reason'] = failure_reason
            result['recoverable'] = _is_recoverable_failure_reason(failure_reason)
            failure_message = _derive_failure_message(status=status)
            if failure_message and failure_message != failure_reason:
                result['message'] = failure_message

            navigation_failure_reason = str(payload.get('navigation_failure_reason', '') or '').strip()
            if navigation_failure_reason and navigation_failure_reason != failure_reason:
                result['navigation_failure_reason'] = navigation_failure_reason

        return result

    def perceive_env() -> dict:
        """Return a compact environment summary for navigation and grounding."""
        runtime_response = bridge.invoke(
            tool_name='perceive_env',
            command='get_runtime_state',
            args={},
        )
        object_response = bridge.invoke(
            tool_name='perceive_env',
            command='get_obj_list',
            args={},
        )
        runtime_payload = dict(runtime_response.get('payload') or {})
        object_payload = dict(object_response.get('payload') or {})
        objects = list(object_payload.get('objects') or [])
        runtime_status = _extract_command_status(runtime_response)
        object_status = _extract_command_status(object_response)
        ok = runtime_status['ok'] and object_status['ok']
        result = {
            'ok': ok,
            'robot_name': str(runtime_payload.get('robot_name', '') or ''),
            'current_room_name': str(
                object_payload.get('current_room_name', runtime_payload.get('current_room_name', '')) or ''
            ),
            'spawn_room_name': str(runtime_payload.get('spawn_room_name', '') or ''),
            'gripper_state': str(runtime_payload.get('gripper_state', '') or ''),
            'objects': objects,
            'object_count': int(object_payload.get('object_count', len(objects)) or 0),
            'room_index': _compact_room_index(runtime_payload.get('room_index')),
        }
        if not ok:
            error_parts = [part for part in (runtime_status['error'], object_status['error']) if part]
            message_parts = [part for part in (runtime_status['message'], object_status['message']) if part]
            result['message'] = '; '.join(message_parts)
            result['error'] = '; '.join(error_parts)
            result['diagnostics'] = {
                'runtime': _compact_agent_payload(runtime_status['payload']),
                'objects': _compact_agent_payload(object_status['payload']),
            }
        return result

    def move_end_effector_to_pose(
        target_pose: dict,
        target_object: Optional[str] = None,
        enable_obstacle_avoidance: Optional[bool] = None,
        enable_collision_filtering: Optional[bool] = None,
        collision_filter_query_padding: Optional[float] = None,
        collision_filter_max_pairs: Optional[int] = None,
    ) -> dict:
        """Move the end effector to a target pose using Euler orientation."""
        pose = _coerce_pose(target_pose)
        command_args: Dict[str, Any] = {'target_pose': pose}
        if target_object:
            command_args['target_object'] = str(target_object).strip()
        if enable_obstacle_avoidance is not None:
            command_args['enable_obstacle_avoidance'] = bool(enable_obstacle_avoidance)
        if enable_collision_filtering is not None:
            command_args['enable_collision_filtering'] = bool(enable_collision_filtering)
        if collision_filter_query_padding is not None:
            command_args['collision_filter_query_padding'] = float(collision_filter_query_padding)
        if collision_filter_max_pairs is not None:
            command_args['collision_filter_max_pairs'] = int(collision_filter_max_pairs)
        response = bridge.invoke(
            tool_name='move_end_effector_to_pose',
            command='move_end_effector_to_pose',
            args=command_args,
        )
        status = _extract_command_status(response)
        payload = status['payload']
        reached = bool(payload.get('reached', False))
        failure_reason = _derive_motion_failure_reason(
            status=status,
            default_reason='end_effector_target_not_reached',
        )
        ok = bool(status['ok']) and reached
        result: Dict[str, Any] = {
            'ok': ok,
            'reached': reached,
        }
        _set_if_present(result, 'elapsed_sec', payload.get('elapsed_sec'))
        _set_if_present(result, 'target_distance', payload.get('target_distance'))
        _set_if_present(result, 'orientation_distance', payload.get('orientation_distance'))

        if not ok:
            result['failure_reason'] = failure_reason
            result['recoverable'] = _is_recoverable_failure_reason(failure_reason)
            failure_message = _derive_failure_message(status=status)
            if failure_message and failure_message != failure_reason:
                result['message'] = failure_message

        return result

    def perceive_obj_list() -> dict:
        """Return the perceived object names currently available in the scene."""
        response = bridge.invoke(
            tool_name='perceive_obj_list',
            command='get_obj_list',
            args={},
        )
        status = _extract_command_status(response)
        payload = status['payload']
        objects = payload.get('objects') or []
        return {
            'ok': status['ok'],
            'current_room_name': str(payload.get('current_room_name', '') or ''),
            'objects': list(objects),
            'count': len(objects),
            'message': status['message'],
            'error': status['error'],
        }

    def perceive_object_pose(obj_name: str) -> dict:
        """Return the structured pose for one grounded object."""
        response = bridge.invoke(
            tool_name='perceive_object_pose',
            command='get_object_pose',
            args={'obj_name': obj_name},
        )
        status = _extract_command_status(response)
        payload = status['payload']
        raw_pose = payload.get('pose') or {}
        grasp_pose_source = str(payload.get('grasp_pose_source', '') or '').strip()
        if status['ok']:
            pose = _coerce_pose(raw_pose)
            position = dict(pose.get('position') or {})
            orientation = dict(pose.get('orientation') or {})
            raw_grasp_pose = payload.get('top_down_grasp_pose') or {}
            grasp_pose_candidates = _coerce_grasp_pose_candidates(
                payload.get('grasp_pose_candidates') or []
            )
            grasp_poses = _coerce_pose_list(payload.get('grasp_poses') or [])
            if not grasp_poses and grasp_pose_candidates:
                grasp_poses = [
                    dict(candidate.get('pose') or {})
                    for candidate in grasp_pose_candidates
                    if candidate.get('pose')
                ]
            if raw_grasp_pose:
                top_down_grasp_pose = _coerce_pose(raw_grasp_pose)
            else:
                runtime_response = bridge.invoke(
                    tool_name='perceive_object_pose',
                    command='get_runtime_state',
                    args={},
                )
                runtime_status = _extract_command_status(runtime_response)
                runtime_payload = runtime_status['payload']
                base_pose = dict(runtime_payload.get('base_pose') or {})
                top_down_grasp_pose = _build_top_down_grasp_pose(
                    object_pose=pose,
                    base_pose=base_pose,
                )
                if not grasp_pose_source:
                    grasp_pose_source = 'top_down_fallback'
            if not grasp_poses and top_down_grasp_pose:
                grasp_poses = [dict(top_down_grasp_pose)]
            if not grasp_pose_candidates and grasp_poses:
                grasp_pose_candidates = _pose_list_to_grasp_pose_candidates(grasp_poses)
        else:
            pose = {}
            position = {}
            orientation = {}
            top_down_grasp_pose = {}
            grasp_poses = []
            grasp_pose_candidates = []
            grasp_pose_source = ''
        return {
            'ok': status['ok'],
            'obj_name': obj_name,
            'position': position,
            'orientation': orientation,
            'pose': pose,
            'top_down_grasp_pose': top_down_grasp_pose,
            'grasp_poses': grasp_poses,
            'grasp_pose_candidates': grasp_pose_candidates,
            'grasp_pose_source': grasp_pose_source,
            'grasp_pose_diagnostics': _compact_grasp_pose_diagnostics(
                payload.get('grasp_pose_diagnostics') or {}
            ),
            'message': status['message'],
            'error': status['error'],
        }

    def suggest_manipulation_base_pose(obj_name: str) -> dict:
        """Return a grounded base pose only for immediate manipulation near the target object.

        Use this tool when the next action will be a manipulation step such as grasping,
        placing, or moving the end effector from a precise operating position.
        Do not use it as a general navigation or "approach this object" tool.
        """
        response = bridge.invoke(
            tool_name='suggest_manipulation_base_pose',
            command='suggest_manipulation_base_pose',
            args={'obj_name': obj_name},
        )
        status = _extract_command_status(response)
        payload = status['payload']
        raw_pose = payload.get('target_pose') or {}
        raw_grasp_pose = payload.get('grasp_pose') or {}
        if status['ok']:
            target_pose = _coerce_pose(raw_pose)
            position = dict(target_pose.get('position') or {})
            orientation = dict(target_pose.get('orientation') or {})
            grasp_pose = _coerce_pose(raw_grasp_pose) if raw_grasp_pose else {}
            grasp_pose_candidates = _coerce_grasp_pose_candidates(
                payload.get('grasp_pose_candidates') or []
            )
            grasp_poses = _coerce_pose_list(payload.get('grasp_poses') or [])
            if not grasp_poses and grasp_pose_candidates:
                grasp_poses = [
                    dict(candidate.get('pose') or {})
                    for candidate in grasp_pose_candidates
                    if candidate.get('pose')
                ]
            if not grasp_poses and grasp_pose:
                grasp_poses = [dict(grasp_pose)]
            if not grasp_pose_candidates and grasp_poses:
                grasp_pose_candidates = _pose_list_to_grasp_pose_candidates(grasp_poses)
        else:
            target_pose = {}
            position = {}
            orientation = {}
            grasp_pose = {}
            grasp_poses = []
            grasp_pose_candidates = []
        return {
            'ok': status['ok'],
            'obj_name': obj_name,
            'position': position,
            'orientation': orientation,
            'target_pose': target_pose,
            'grasp_pose': grasp_pose,
            'grasp_poses': grasp_poses,
            'grasp_pose_candidates': grasp_pose_candidates,
            'grasp_pose_source': str(payload.get('grasp_pose_source', '') or '').strip(),
            'grasp_pose_diagnostics': _compact_grasp_pose_diagnostics(
                payload.get('grasp_pose_diagnostics') or {}
            ),
            'message': status['message'],
            'error': status['error'],
        }

    def controll_gripper(cmd: int) -> dict:
        """Open the gripper when cmd=1 and close otherwise."""
        command_name = 'open' if int(cmd) == 1 else 'close'
        response = bridge.invoke(
            tool_name='controll_gripper',
            command=command_name,
            args={'cmd': int(cmd)},
        )
        status = _extract_command_status(response)
        payload = status['payload']
        is_close = command_name == 'close'
        grasp_success = bool(payload.get('object_in_gripper', False)) if is_close else None
        soft_failure_reason = 'grasp_not_acquired' if is_close and grasp_success is False else ''
        return {
            'ok': status['ok'],
            'gripper_state': 'open' if command_name == 'open' else 'closed',
            'object_in_gripper': bool(payload.get('object_in_gripper', False)),
            'grasp_success': grasp_success,
            'grasped_object_name': str(payload.get('grasped_object_name', '') or ''),
            'settled': bool(payload.get('settled', True)),
            'elapsed_sec': payload.get('elapsed_sec'),
            'message': status['message'],
            'error': status['error'],
            'failure_reason': status['error'] or soft_failure_reason,
            'recoverable': bool(soft_failure_reason),
        }

    def open_object(obj_name: str) -> dict:
        """Navigate to a suitable manipulation pose, then open an articulated object."""
        target_name = str(obj_name or '').strip()
        if not target_name:
            return {
                'ok': False,
                'opened': False,
                'obj_name': '',
                'failure_stage': 'input',
                'failure_reason': 'obj_name_required',
                'recoverable': False,
            }

        suggestion_response = bridge.invoke(
            tool_name='open',
            command='suggest_manipulation_base_pose',
            args={'obj_name': target_name},
        )
        suggestion_status = _extract_command_status(suggestion_response)
        suggestion_payload = suggestion_status['payload']
        raw_target_pose = suggestion_payload.get('target_pose') or {}
        if not suggestion_status['ok'] or not raw_target_pose:
            failure_reason = _derive_open_failure_reason(
                status=suggestion_status,
                default_reason='manipulation_base_pose_not_found',
            )
            return {
                'ok': False,
                'opened': False,
                'obj_name': target_name,
                'failure_stage': 'suggest_manipulation_base_pose',
                'failure_reason': failure_reason,
                'recoverable': _is_recoverable_failure_reason(failure_reason),
                'message': _derive_failure_message(status=suggestion_status),
                'error': suggestion_status['error'],
                'suggestion': _compact_manipulation_suggestion(suggestion_payload),
            }

        target_pose = _coerce_pose(raw_target_pose)
        navigation_response = bridge.invoke(
            tool_name='open',
            command='move_base_to_pose',
            args={'target_pose': target_pose},
        )
        navigation_status = _extract_command_status(navigation_response)
        navigation_payload = navigation_status['payload']
        reached = bool(navigation_payload.get('reached', False))
        navigation_ok = bool(navigation_status['ok']) and reached
        if not navigation_ok:
            failure_reason = _derive_navigation_failure_reason(status=navigation_status)
            result: Dict[str, Any] = {
                'ok': False,
                'opened': False,
                'obj_name': target_name,
                'failure_stage': 'navigate',
                'failure_reason': failure_reason,
                'recoverable': _is_recoverable_failure_reason(failure_reason),
                'target_pose': target_pose,
                'navigation': {
                    'ok': navigation_ok,
                    'reached': reached,
                    'message': navigation_status['message'],
                    'error': navigation_status['error'],
                    'details': _compact_motion_payload(navigation_payload),
                },
                'suggestion': _compact_manipulation_suggestion(suggestion_payload),
            }
            failure_message = _derive_failure_message(status=navigation_status)
            if failure_message and failure_message != failure_reason:
                result['message'] = failure_message
            return result

        open_response = bridge.invoke(
            tool_name='open',
            command='open',
            args={'obj_name': target_name},
        )
        open_status = _extract_command_status(open_response)
        open_payload = open_status['payload']
        opened = bool(open_status['ok'])
        result = {
            'ok': opened,
            'opened': opened,
            'obj_name': target_name,
            'target_pose': target_pose,
            'navigation': {
                'ok': navigation_ok,
                'reached': reached,
                'details': _compact_motion_payload(navigation_payload),
            },
            'open_mode': str(open_payload.get('mode', '') or ''),
            'message': open_status['message'],
            'error': open_status['error'],
            'details': _compact_open_payload(open_payload),
        }
        if not opened:
            failure_reason = _derive_open_failure_reason(
                status=open_status,
                default_reason='object_open_failed',
            )
            result.update(
                {
                    'failure_stage': 'open',
                    'failure_reason': failure_reason,
                    'recoverable': _is_recoverable_failure_reason(failure_reason),
                    'suggestion': _compact_manipulation_suggestion(suggestion_payload),
                }
            )
        return result

    # Deprecated: `compute_grasp_position` is intentionally disabled.
    # def compute_grasp_position(obj_name: str, position: dict) -> dict:
    #     """Compute a grasp pose by adding a small z offset to the object pose."""
    #     pose = _coerce_pose(position.get('pose') if isinstance(position, dict) and 'pose' in position else position)
    #     pos = dict(pose.get('position') or {})
    #     ori = dict(pose.get('orientation') or {})
    #     pos.setdefault('x', 0.0)
    #     pos.setdefault('y', 0.0)
    #     pos.setdefault('z', 0.0)
    #     pos['z'] = float(pos['z']) + float(Z_OFFSET_MAP.get(obj_name, 0.02))
    #     grasp_pose = {
    #         'position': pos,
    #         'orientation': {
    #             'roll': float(ori.get('roll', 0.0)),
    #             'pitch': float(ori.get('pitch', 0.0)),
    #             'yaw': float(ori.get('yaw', 0.0)),
    #         },
    #     }
    #     return {
    #         'ok': True,
    #         'obj_name': obj_name,
    #         'position': dict(grasp_pose['position']),
    #         'orientation': dict(grasp_pose['orientation']),
    #         'pose': grasp_pose,
    #     }

    def lateral_shift(
        direction: Optional[dict] = None,
        distance: Optional[float] = None,
        target_position: Optional[dict] = None,
        enable_obstacle_avoidance: Optional[bool] = None,
        enable_collision_filtering: Optional[bool] = None,
    ) -> dict:
        """Move the end-effector in a straight line along a specified horizontal
        direction, or to a specified target point.

        Provide either (direction + distance) OR target_position:
        - direction: dict with x, y, z keys (will be normalized automatically)
        - distance: how far to shift in meters (required when direction is given)
        - target_position: dict with x, y, z keys to shift toward that point
        The end-effector orientation is preserved during the shift.
        """
        command_args: Dict[str, Any] = {}
        if direction is not None:
            command_args['direction'] = direction
        if distance is not None:
            command_args['distance'] = float(distance)
        if target_position is not None:
            command_args['target_position'] = _coerce_position(target_position)
        if enable_obstacle_avoidance is not None:
            command_args['enable_obstacle_avoidance'] = bool(enable_obstacle_avoidance)
        if enable_collision_filtering is not None:
            command_args['enable_collision_filtering'] = bool(enable_collision_filtering)

        response = bridge.invoke(
            tool_name='lateral_shift',
            command='lateral_shift',
            args=command_args,
        )
        status = _extract_command_status(response)
        payload = status['payload']
        reached = bool(payload.get('reached', False))
        ok = bool(status['ok']) and reached
        result: Dict[str, Any] = {
            'ok': ok,
            'reached': reached,
        }
        _set_if_present(result, 'elapsed_sec', payload.get('elapsed_sec'))
        _set_if_present(result, 'target_distance', payload.get('target_distance'))
        _set_if_present(result, 'orientation_distance', payload.get('orientation_distance'))
        _set_if_present(result, 'distance', payload.get('distance'))

        start_position = payload.get('start_position')
        if start_position:
            result['start_position'] = start_position
        target_pos = payload.get('target_position')
        if target_pos:
            result['target_position'] = target_pos
        direction_result = payload.get('direction')
        if direction_result:
            result['direction'] = direction_result

        if not ok:
            failure_reason = _derive_motion_failure_reason(
                status=status,
                default_reason='lateral_shift_failed',
            )
            result['failure_reason'] = failure_reason
            result['recoverable'] = _is_recoverable_failure_reason(failure_reason)
            failure_message = _derive_failure_message(status=status)
            if failure_message and failure_message != failure_reason:
                result['message'] = failure_message

        return result

    def rotate_end_effector(
        target_orientation: Optional[dict] = None,
        target_pose: Optional[dict] = None,
        roll: Optional[float] = None,
        pitch: Optional[float] = None,
        yaw: Optional[float] = None,
        keep_position: Optional[bool] = None,
        enable_obstacle_avoidance: Optional[bool] = None,
        enable_collision_filtering: Optional[bool] = None,
    ) -> dict:
        """Adjust the end-effector orientation to match a desired grasping, placing,
        or interaction pose while keeping the current position.

        Provide orientation via one of:
        - target_orientation: dict with roll, pitch, yaw keys (radians)
        - target_pose: dict with orientation sub-dict containing roll, pitch, yaw
        - roll, pitch, yaw: individual Euler angle values (radians); only the
          provided axes are changed, the rest keep the current orientation.
        """
        command_args: Dict[str, Any] = {}
        if target_orientation is not None:
            command_args['target_orientation'] = _coerce_orientation(target_orientation)
        if target_pose is not None:
            command_args['target_pose'] = _coerce_pose(target_pose)
        if roll is not None:
            command_args['roll'] = float(roll)
        if pitch is not None:
            command_args['pitch'] = float(pitch)
        if yaw is not None:
            command_args['yaw'] = float(yaw)
        if keep_position is not None:
            command_args['keep_position'] = bool(keep_position)
        if enable_obstacle_avoidance is not None:
            command_args['enable_obstacle_avoidance'] = bool(enable_obstacle_avoidance)
        if enable_collision_filtering is not None:
            command_args['enable_collision_filtering'] = bool(enable_collision_filtering)

        response = bridge.invoke(
            tool_name='rotate_end_effector',
            command='rotate_end_effector',
            args=command_args,
        )
        status = _extract_command_status(response)
        payload = status['payload']
        reached = bool(payload.get('reached', False))
        ok = bool(status['ok']) and reached
        result: Dict[str, Any] = {
            'ok': ok,
            'reached': reached,
        }
        _set_if_present(result, 'elapsed_sec', payload.get('elapsed_sec'))
        _set_if_present(result, 'target_distance', payload.get('target_distance'))
        _set_if_present(result, 'orientation_distance', payload.get('orientation_distance'))

        start_orientation = payload.get('start_orientation')
        if start_orientation:
            result['start_orientation'] = start_orientation
        target_orientation_res = payload.get('target_orientation')
        if target_orientation_res:
            result['target_orientation'] = target_orientation_res

        if not ok:
            failure_reason = _derive_motion_failure_reason(
                status=status,
                default_reason='rotation_failed',
            )
            result['failure_reason'] = failure_reason
            result['recoverable'] = _is_recoverable_failure_reason(failure_reason)
            failure_message = _derive_failure_message(status=status)
            if failure_message and failure_message != failure_reason:
                result['message'] = failure_message

        return result

    def perceive_entity_state(entity: str) -> dict:
        """Return simulator-owned capabilities and dynamic state for one entity."""
        response = bridge.invoke(
            tool_name='perceive_entity_state',
            command='get_entity_state',
            args={'entity': str(entity or '').strip()},
        )
        status = _extract_command_status(response)
        payload = status['payload']
        return {
            'ok': status['ok'],
            'entity': str(payload.get('name') or entity or ''),
            'capabilities': list(payload.get('capabilities') or []),
            'state': dict(payload.get('state') or {}),
            'durations': dict(payload.get('durations') or {}),
            'source': str(payload.get('source') or ''),
            'message': status['message'],
            'error': status['error'],
        }

    def toggle_device(entity: str, on: bool) -> dict:
        """Turn a registered toggleable entity on or off."""
        return _interact_entity_tool(
            bridge,
            tool_name='toggle_device',
            entity=entity,
            action='toggle',
            parameters={'on': bool(on)},
        )

    def set_device_level(entity: str, level: float) -> dict:
        """Set a registered device, heat source, or water source level in [0, 1]."""
        return _interact_entity_tool(
            bridge,
            tool_name='set_device_level',
            entity=entity,
            action='set_level',
            parameters={'level': float(level)},
        )

    def set_device_timer(entity: str, seconds: float, start: bool = True) -> dict:
        """Set a timer-capable entity and optionally start it."""
        timer_result = _interact_entity_tool(
            bridge,
            tool_name='set_device_timer',
            entity=entity,
            action='set_timer',
            parameters={'seconds': float(seconds)},
        )
        if not timer_result.get('ok') or not start:
            return timer_result
        start_result = _interact_entity_tool(
            bridge,
            tool_name='set_device_timer',
            entity=entity,
            action='start',
            parameters={},
        )
        return {
            'ok': bool(start_result.get('ok')),
            'entity': str(entity or ''),
            'timer': timer_result,
            'start': start_result,
            'state': dict(start_result.get('state') or {}),
            'error': str(start_result.get('error') or ''),
        }

    def set_entity_temperature(entity: str, temperature_c: float) -> dict:
        """Set the temperature control or water temperature of a registered entity."""
        return _interact_entity_tool(
            bridge,
            tool_name='set_entity_temperature',
            entity=entity,
            action='set_temperature',
            parameters={'temperature_c': float(temperature_c)},
        )

    def set_water_flow(entity: str, level: float) -> dict:
        """Set water-source flow level in [0, 1]."""
        return _interact_entity_tool(
            bridge,
            tool_name='set_water_flow',
            entity=entity,
            action='set_level',
            parameters={'level': float(level), 'property': 'water_flow_level'},
        )

    def set_water_temperature(entity: str, temperature_c: float) -> dict:
        """Set a registered water source temperature in Celsius."""
        return _interact_entity_tool(
            bridge,
            tool_name='set_water_temperature',
            entity=entity,
            action='set_temperature',
            parameters={
                'temperature_c': float(temperature_c),
                'property': 'water_temperature_c',
            },
        )

    def set_drain_state(entity: str, state: str) -> dict:
        """Set a drainable entity drain to open, closed, or blocked."""
        return _interact_entity_tool(
            bridge,
            tool_name='set_drain_state',
            entity=entity,
            action='set_drain',
            parameters={'state': str(state or '').strip().lower()},
        )

    def start_device(entity: str) -> dict:
        """Start a registered timer-capable or toggleable entity."""
        return _interact_entity_tool(
            bridge,
            tool_name='start_device',
            entity=entity,
            action='start',
            parameters={},
        )

    def stop_device(entity: str) -> dict:
        """Stop a registered timer-capable or toggleable entity."""
        return _interact_entity_tool(
            bridge,
            tool_name='stop_device',
            entity=entity,
            action='stop',
            parameters={},
        )

    def close_entity(entity: str) -> dict:
        """Set the semantic state of a registered openable entity to closed."""
        return _interact_entity_tool(
            bridge,
            tool_name='close_entity',
            entity=entity,
            action='close',
            parameters={},
        )

    def set_entity_state(entity: str, property_name: str, value: Any) -> dict:
        """Set one simulator-owned logical state field on a registered entity."""
        return _interact_entity_tool(
            bridge,
            tool_name='set_entity_state',
            entity=entity,
            action='set_state',
            parameters={
                'property': str(property_name or '').strip(),
                'value': _coerce_entity_state_value(
                    value,
                    property_name=str(property_name or '').strip(),
                ),
            },
        )

    def _tool(func: Any, *, name: Optional[str] = None) -> Any:
        visible_name = str(name or getattr(func, '__name__', '') or '')
        capture_screenshot = visible_name not in PERCEPTION_TOOL_NAMES
        wrapped = _compact_tool_function(
            func,
            bridge=bridge,
            tool_name=visible_name,
            capture_screenshot=capture_screenshot,
        )
        return StructuredTool.from_function(wrapped, name=name)

    tools = [
        _tool(perceive_env),
        _tool(perceive_obj_list),
        _tool(perceive_object_pose),
        _tool(suggest_manipulation_base_pose),
        _tool(navigate),
        _tool(move_end_effector_to_pose),
        _tool(lateral_shift),
        _tool(rotate_end_effector),
        _tool(controll_gripper),
        _tool(open_object, name='open'),
        _tool(perceive_entity_state),
        _tool(toggle_device),
        _tool(set_device_level),
        _tool(set_device_timer),
        _tool(set_entity_temperature),
        _tool(set_water_flow),
        _tool(set_water_temperature),
        _tool(set_drain_state),
        _tool(start_device),
        _tool(stop_device),
        _tool(close_entity),
        _tool(set_entity_state),
    ]

    if use_batch_execute_plan and BATCH_EXECUTE_PLAN_TOOL_ENABLED:
        tools_mapping = {tool.name: tool.func for tool in tools}

        def execute_plan(steps: List[Dict[str, Any]]) -> dict:
            """Batch deterministic tool calls with dependency injection support."""
            context: Dict[str, Dict[str, Any]] = {}
            results: List[Dict[str, Any]] = []

            for idx, step in enumerate(steps):
                tool_name = str(step.get('tool', '') or '')
                arguments = step.get('arguments') or {}
                resolved_arguments = _resolve_dependency(arguments, context)
                tool = tools_mapping.get(tool_name)
                if tool is None:
                    return {
                        'ok': False,
                        'failed_step': idx,
                        'error': f"Tool '{tool_name}' not found.",
                        'steps': results,
                    }

                result = tool(**resolved_arguments)
                context[f'step{idx}'] = {
                    'tool': tool_name,
                    'arguments': resolved_arguments,
                    'result': result,
                }
                results.append(
                    {
                        'index': idx,
                        'tool': tool_name,
                        'arguments': resolved_arguments,
                        'result': result,
                    }
                )

                if isinstance(result, dict) and _result_requires_recovery(result):
                    failure_error = str(
                        result.get('failure_reason', '')
                        or result.get('error', '')
                        or result.get('message', '')
                        or ''
                    ).strip()
                    return {
                        'ok': False,
                        'failed_step': idx,
                        'failed_tool': tool_name,
                        'message': failure_error,
                        'error': failure_error,
                        'steps': results,
                    }

            return {
                'ok': True,
                'steps': results,
            }

        tools.append(_tool(execute_plan))

    return tools


def _require_structured_tool() -> Any:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            'LangChain is required for AgenticPolicy tools. '
            'Install "langchain" and "langchain-openai".'
        ) from exc
    return StructuredTool


def _compact_tool_function(
    func: Any,
    *,
    bridge: Optional[SimCommandBridge] = None,
    tool_name: str = '',
    capture_screenshot: bool = False,
) -> Any:
    """Wrap a LangChain tool so the LLM never sees simulator-internal bulk data.

    When ``capture_screenshot`` is True and a bridge is provided, request a
    top-down screenshot from the bridge after the tool returns (regardless of
    whether it raised) so the per-tool log carries a visual record.
    """

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start_record_count = len(getattr(bridge, 'records', []) or [])
        try:
            result = func(*args, **kwargs)
        finally:
            if capture_screenshot and bridge is not None:
                try:
                    bridge.capture_top_down_screenshot_for_tool(
                        tool_name=tool_name,
                        start_record_count=start_record_count,
                    )
                except Exception:
                    # Screenshot failures must never mask the tool result/exception.
                    pass
        return _compact_agent_tool_result(result)

    return wrapped


def _compact_agent_tool_result(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_agent_value(value)

    compact: Dict[str, Any] = {}
    for raw_key, raw_item in value.items():
        key = str(raw_key)
        if key == 'payload':
            payload_summary = _compact_agent_payload(raw_item)
            if payload_summary:
                compact['payload_summary'] = payload_summary
            continue
        if key == 'grasp_pose_diagnostics':
            compact[key] = _compact_grasp_pose_diagnostics(raw_item)
            continue
        if key == 'grasp_pose_candidates':
            compact[key] = _compact_grasp_pose_candidates(raw_item)
            continue
        if key in {'objects', 'available_objects'}:
            compact[key] = _compact_name_list(raw_item)
            continue
        if key in {
            'persistent_arm_collision_filtering',
            'aggressive_lifecycle_collision_filtering',
            'collision_filtering',
        }:
            compact[key] = _compact_collision_filtering_state(raw_item)
            continue
        compact[key] = _compact_agent_value(raw_item)
    return compact


def _compact_agent_payload(value: Any) -> Any:
    return _compact_agent_value(value, max_depth=4)


def _compact_agent_value(
    value: Any,
    *,
    max_depth: int = 5,
    max_list_items: int = 8,
    _key: str = '',
) -> Any:
    if max_depth < 0:
        return _summarize_value(value)

    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key)
            lowered = key.lower()

            if key in {
                'persistent_arm_collision_filtering',
                'aggressive_lifecycle_collision_filtering',
                'collision_filtering',
            }:
                compact[key] = _compact_collision_filtering_state(raw_item)
                continue

            if lowered == 'pairs' and isinstance(raw_item, list):
                compact['pair_count'] = len(raw_item)
                compact['pairs_omitted'] = True
                continue

            if lowered == 'point_cloud':
                compact[key] = _compact_point_cloud(raw_item)
                continue

            if lowered == 'room_index':
                compact[key] = _compact_room_index(raw_item)
                continue

            if key == 'grasp_pose_diagnostics':
                compact[key] = _compact_grasp_pose_diagnostics(raw_item)
                continue

            if key == 'grasp_pose_candidates':
                compact[key] = _compact_grasp_pose_candidates(raw_item)
                continue

            if key in {'objects', 'available_objects', 'grasp_pose_candidate_names'}:
                compact[key] = _compact_name_list(raw_item)
                continue

            if lowered in {
                'before_state',
                'after_state',
                'trace_record',
                'tool_trace',
                'intermediate_steps',
                'encrypted_reasoning',
                'reasoning_encrypted',
            }:
                compact[f'{key}_omitted'] = True
                continue

            if (
                isinstance(raw_item, list)
                and ('collision_path' in lowered or lowered in {'collision_paths', 'collision_prim_paths'})
            ):
                compact[f'{key}_count'] = len(raw_item)
                compact[f'{key}_omitted'] = True
                continue

            compact[key] = _compact_agent_value(
                raw_item,
                max_depth=max_depth - 1,
                max_list_items=max_list_items,
                _key=key,
            )
        return compact

    if isinstance(value, (list, tuple)):
        values = list(value)
        lowered_key = str(_key or '').lower()
        if lowered_key == 'pairs':
            return {
                'pair_count': len(values),
                'pairs_omitted': True,
            }
        if lowered_key in {'grasp_pose_candidates'}:
            return _compact_grasp_pose_candidates(values)
        if lowered_key == 'point_cloud':
            return _compact_point_cloud(values)
        if len(values) <= max_list_items:
            return [
                _compact_agent_value(
                    item,
                    max_depth=max_depth - 1,
                    max_list_items=max_list_items,
                    _key=_key,
                )
                for item in values
            ]
        return {
            'item_count': len(values),
            'items': [
                _compact_agent_value(
                    item,
                    max_depth=max_depth - 1,
                    max_list_items=max_list_items,
                    _key=_key,
                )
                for item in values[:max_list_items]
            ],
            'omitted_count': max(0, len(values) - max_list_items),
        }

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _summarize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {'field_count': len(value), 'omitted': True}
    if isinstance(value, (list, tuple)):
        return {'item_count': len(value), 'omitted': True}
    return str(value)


def _compact_name_list(value: Any, *, max_names: int = 120) -> List[str] | Dict[str, Any]:
    if not isinstance(value, (list, tuple)):
        return []
    names = [str(item) for item in value if str(item).strip()]
    if len(names) <= max_names:
        return names
    return {
        'item_count': len(names),
        'items': names[:max_names],
        'omitted_count': len(names) - max_names,
    }


def _compact_collision_filtering_state(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        pairs = value.get('pairs')
        pair_count = len(pairs) if isinstance(pairs, list) else None
        active_pair_count = value.get('active_pair_count')
        if active_pair_count is None:
            active_pair_count = value.get('pair_count', pair_count or 0)
        compact = {
            'enabled': bool(value.get('enabled', True)),
            'active_pair_count': int(active_pair_count or 0),
            'pairs_omitted': True,
        }
        for key in (
            'strategy',
            'active_group_count',
            'robot_root_count',
            'scene_object_root_count',
            'structural_root_count',
            'robot_collider_count',
            'scene_object_collider_count',
            'structural_collider_count',
            'robot_group_target_count',
            'scene_object_group_target_count',
            'structural_group_target_count',
            'filtered_group_count',
            'physics_scene_count',
            'authored_physics_scene_count',
            'invert_collision_group_filter',
            'pre_reset_applied',
            'exemptions_honored',
            'last_error',
        ):
            if key in value:
                compact[key] = _compact_agent_value(value.get(key), max_depth=1)
        if pair_count is not None and pair_count != compact['active_pair_count']:
            compact['serialized_pair_count'] = pair_count
        return compact
    if isinstance(value, (list, tuple)):
        return {
            'enabled': True,
            'active_pair_count': len(value),
            'pairs_omitted': True,
        }
    return {
        'enabled': bool(value),
        'active_pair_count': 0,
        'pairs_omitted': True,
    }


def _compact_point_cloud(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        points = value.get('points')
        point_count = value.get('point_count', value.get('count'))
        if point_count is None and isinstance(points, list):
            point_count = len(points)
        compact: Dict[str, Any] = {
            'point_count': int(point_count or 0),
            'points_omitted': True,
        }
        for key in ('frame', 'source', 'max_points', 'sampled'):
            if key in value:
                compact[key] = _compact_agent_value(value.get(key), max_depth=1)
        return compact
    if isinstance(value, (list, tuple)):
        return {
            'point_count': len(value),
            'points_omitted': True,
        }
    return {
        'point_count': 0,
        'points_omitted': True,
    }


def _compact_grasp_pose_diagnostics(value: Any) -> Dict[str, Any]:
    diagnostics = dict(value or {}) if isinstance(value, dict) else {}
    keep_keys = (
        'grasp_pose_source',
        'source',
        'fallback_reason',
        'failure_reason',
        'error',
        'message',
        'grasp_pose_candidate_count',
        'grasp_pose_candidate_names',
        'grasp_pose_candidate_frame',
        'grasp_pose_candidate_position_source',
        'selected_candidate',
    )
    compact: Dict[str, Any] = {}
    for key in keep_keys:
        if key in diagnostics:
            compact[key] = _compact_agent_value(diagnostics.get(key), max_depth=2)
    extra_keys = [
        str(key)
        for key in diagnostics.keys()
        if str(key) not in keep_keys
        and str(key).lower() not in {'point_cloud', 'points', 'pairs'}
    ]
    if extra_keys:
        compact['extra_keys_omitted'] = extra_keys[:12]
    return compact


def _compact_grasp_pose_candidates(value: Any, *, max_candidates: int = 8) -> List[Dict[str, Any]] | Dict[str, Any]:
    if not isinstance(value, (list, tuple)):
        return []
    candidates = list(value)
    compact: List[Dict[str, Any]] = []
    for candidate in candidates[:max_candidates]:
        if not isinstance(candidate, dict):
            continue
        item: Dict[str, Any] = {}
        for key in (
            'name',
            'rank',
            'pose',
            'approach_direction',
            'pre_grasp_pose',
            'score',
            'clearance',
            'source',
        ):
            if key in candidate:
                item[key] = _compact_agent_value(candidate.get(key), max_depth=3)
        if not item and candidate:
            item = _compact_agent_value(candidate, max_depth=2)
        compact.append(item)
    if len(candidates) <= max_candidates:
        return compact
    return {
        'candidate_count': len(candidates),
        'candidates': compact,
        'omitted_count': len(candidates) - len(compact),
    }


def _compact_motion_payload(payload: Any) -> Dict[str, Any]:
    data = dict(payload or {}) if isinstance(payload, dict) else {}
    keep_keys = (
        'reached',
        'target_distance',
        'distance',
        'orientation_distance',
        'elapsed_sec',
        'yaw_error',
        'navigation_state',
        'navigation_failure_reason',
        'failure_reason',
        'start_position',
        'target_position',
        'direction',
        'linear_waypoint_count',
        'pre_grasp_enabled',
        'pre_grasp_reached',
    )
    return {
        key: _compact_agent_value(data.get(key), max_depth=3)
        for key in keep_keys
        if key in data
    }


def _compact_open_payload(payload: Any) -> Dict[str, Any]:
    data = dict(payload or {}) if isinstance(payload, dict) else {}
    keep_keys = (
        'mode',
        'opened',
        'container_name',
        'obj_name',
        'joint_name',
        'joint_type',
        'target_position',
        'final_position',
        'elapsed_sec',
        'failure_reason',
    )
    return {
        key: _compact_agent_value(data.get(key), max_depth=3)
        for key in keep_keys
        if key in data
    }


def _compact_manipulation_suggestion(payload: Any) -> Dict[str, Any]:
    data = dict(payload or {}) if isinstance(payload, dict) else {}
    keep_keys = (
        'target_pose',
        'grasp_pose',
        'grasp_poses',
        'grasp_pose_candidates',
        'grasp_pose_source',
        'grasp_pose_diagnostics',
        'available_objects',
        'failure_reason',
        'message',
        'error',
    )
    compact: Dict[str, Any] = {}
    for key in keep_keys:
        if key not in data:
            continue
        if key == 'grasp_pose_diagnostics':
            compact[key] = _compact_grasp_pose_diagnostics(data.get(key))
        elif key == 'grasp_pose_candidates':
            compact[key] = _compact_grasp_pose_candidates(data.get(key))
        else:
            compact[key] = _compact_agent_value(data.get(key), max_depth=3)
    return compact


def _coerce_entity_state_value(value: Any, *, property_name: str = '') -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    lowered = text.lower()
    normalized_property = ''.join(
        char.lower()
        for char in str(property_name or '')
        if char.isalnum()
    )
    boolean_properties = {
        'active',
        'enabled',
        'on',
        'pluggedin',
        'running',
        'wet',
        'overflowing',
        'hot',
        'boiling',
        'lit',
    }
    if lowered in {'true', 'yes', 'enabled'}:
        return True
    if lowered in {'false', 'no', 'disabled'}:
        return False
    if normalized_property in boolean_properties:
        if lowered in {'on', 'open', 'opened', 'active', 'running', 'wet', 'hot', 'lit'}:
            return True
        if lowered in {'off', 'closed', 'inactive', 'stopped', 'dry', 'cold', 'unlit'}:
            return False
    try:
        if re.fullmatch(r'[-+]?\d+', text):
            return int(text)
        if re.fullmatch(r'[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?', text):
            return float(text)
    except (TypeError, ValueError):
        pass
    return text


def _interact_entity_tool(
    bridge: SimCommandBridge,
    *,
    tool_name: str,
    entity: str,
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_entity = str(entity or '').strip()
    response = bridge.invoke(
        tool_name=tool_name,
        command='interact_entity',
        args={
            'entity': normalized_entity,
            'action': str(action or '').strip(),
            'parameters': dict(parameters or {}),
        },
    )
    status = _extract_command_status(response)
    payload = status['payload']
    return {
        'ok': status['ok'],
        'entity': str(payload.get('entity') or normalized_entity),
        'action': str(payload.get('action') or action),
        'state': dict(payload.get('state') or {}),
        'capabilities': list(payload.get('capabilities') or []),
        'message': status['message'],
        'error': status['error'],
    }


def _coerce_pose(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    value = dict(value or {})
    if 'pose' in value and isinstance(value.get('pose'), dict):
        value = dict(value.get('pose') or {})
    position = dict(value.get('position') or {})
    orientation = dict(value.get('orientation') or {})
    orientation_payload: Dict[str, float] = {
        'roll': float(orientation.get('roll', 0.0)),
        'pitch': float(orientation.get('pitch', 0.0)),
        'yaw': float(orientation.get('yaw', 0.0)),
    }
    if {'w', 'x', 'y', 'z'}.issubset(set(orientation.keys())):
        orientation_payload.update(
            {
                'w': float(orientation.get('w', 1.0)),
                'x': float(orientation.get('x', 0.0)),
                'y': float(orientation.get('y', 0.0)),
                'z': float(orientation.get('z', 0.0)),
            }
        )
    return {
        'position': {
            'x': float(position.get('x', 0.0)),
            'y': float(position.get('y', 0.0)),
            'z': float(position.get('z', 0.0)),
        },
        'orientation': orientation_payload,
    }


def _coerce_pose_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        return []
    poses: List[Dict[str, Any]] = []
    for item in value:
        try:
            pose = _coerce_pose(item)
        except Exception:
            continue
        if pose:
            poses.append(pose)
    return poses


def _coerce_grasp_pose_candidates(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        return []

    candidates: List[Dict[str, Any]] = []
    for index, raw_candidate in enumerate(value):
        if not isinstance(raw_candidate, dict):
            continue
        candidate = dict(raw_candidate)
        raw_pose = candidate.get('pose') or {
            'position': candidate.get('position') or {},
            'orientation': candidate.get('orientation') or {},
        }
        try:
            pose = _coerce_pose(raw_pose)
        except Exception:
            continue
        candidate['rank'] = int(candidate.get('rank', index) or index)
        candidate['name'] = str(candidate.get('name', f'candidate_{index}') or f'candidate_{index}')
        candidate['pose'] = pose
        candidate['position'] = dict(pose.get('position') or {})
        candidate['orientation'] = dict(pose.get('orientation') or {})
        candidates.append(candidate)
    return candidates


def _pose_list_to_grasp_pose_candidates(poses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for index, pose in enumerate(poses):
        normalized_pose = _coerce_pose(pose)
        candidates.append(
            {
                'rank': index,
                'name': 'top_down' if index == 0 else f'candidate_{index}',
                'pose': normalized_pose,
                'position': dict(normalized_pose.get('position') or {}),
                'orientation': dict(normalized_pose.get('orientation') or {}),
            }
        )
    return candidates


def _coerce_position(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    value = dict(value or {})
    return {
        'x': float(value.get('x', 0.0)),
        'y': float(value.get('y', 0.0)),
        'z': float(value.get('z', 0.0)),
    }


def _coerce_orientation(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    value = dict(value or {})
    return {
        'roll': float(value.get('roll', 0.0)),
        'pitch': float(value.get('pitch', 0.0)),
        'yaw': float(value.get('yaw', 0.0)),
    }


def _compact_room_index(value: Any) -> Dict[str, Dict[str, Any]]:
    room_index = dict(value or {})
    compact_index: Dict[str, Dict[str, Any]] = {}

    for room_key, raw_entry in room_index.items():
        if not isinstance(raw_entry, dict):
            continue

        room_name = str(raw_entry.get('room_name', room_key) or room_key)
        object_names: List[str] = []
        for raw_object in raw_entry.get('objects') or []:
            if not isinstance(raw_object, dict):
                continue
            object_name = str(raw_object.get('name', '') or '').strip()
            if object_name:
                object_names.append(object_name)

        compact_index[str(room_key)] = {
            'room_name': room_name,
            'object_count': int(raw_entry.get('object_count', len(object_names)) or len(object_names)),
            'objects': object_names,
        }

    return compact_index


def _build_top_down_grasp_orientation(*, yaw: float = 0.0) -> Dict[str, float]:
    return {
        'roll': 0.0,
        'pitch': float(math.pi / 2.0),
        'yaw': float(yaw),
    }


def _build_top_down_grasp_pose(
    *,
    object_pose: Dict[str, Any],
    base_pose: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, float]]:
    raw_pose = dict(object_pose or {})
    position = dict(raw_pose.get('position') or {})
    base_orientation = dict((base_pose or {}).get('orientation') or {})
    yaw = float(base_orientation.get('yaw', 0.0))
    return {
        'position': {
            'x': float(position.get('x', 0.0)),
            'y': float(position.get('y', 0.0)),
            'z': float(position.get('z', 0.0)),
        },
        'orientation': _build_top_down_grasp_orientation(yaw=yaw),
    }


def _extract_command_status(response: Any) -> Dict[str, Any]:
    data = dict(response or {})
    payload = dict(data.get('payload') or {})
    error = str(data.get('error', '') or '')
    message = str(data.get('message', '') or error)
    return {
        'ok': bool(data.get('ok', False)),
        'message': message,
        'error': error,
        'payload': payload,
    }


def _derive_navigation_failure_reason(*, status: Dict[str, Any]) -> str:
    payload = dict(status.get('payload') or {})
    if status.get('ok') and bool(payload.get('reached', False)):
        return ''

    navigation_failure_reason = str(payload.get('navigation_failure_reason', '') or '').strip()
    if navigation_failure_reason:
        return navigation_failure_reason

    error = str(status.get('error', '') or '').strip()
    if error:
        return error

    message = str(status.get('message', '') or '').strip()
    lowered = message.lower()
    if 'timeout' in lowered or 'did not fully settle' in lowered:
        return 'navigation_timeout'
    return message or 'navigation_failed'


def _derive_motion_failure_reason(
    *,
    status: Dict[str, Any],
    default_reason: str,
) -> str:
    payload = dict(status.get('payload') or {})
    if status.get('ok') and bool(payload.get('reached', False)):
        return ''

    error = str(status.get('error', '') or '').strip()
    if error:
        return error

    message = str(status.get('message', '') or '').strip()
    lowered = message.lower()
    if 'timeout' in lowered or 'did not fully settle' in lowered:
        return f'{default_reason}:timeout'
    return message or default_reason


def _derive_open_failure_reason(*, status: Dict[str, Any], default_reason: str) -> str:
    error = str(status.get('error', '') or '').strip()
    if error:
        return error

    message = str(status.get('message', '') or '').strip()
    if message:
        return message

    payload = dict(status.get('payload') or {})
    payload_error = str(payload.get('error', '') or payload.get('failure_reason', '') or '').strip()
    if payload_error:
        return payload_error

    return default_reason


def _derive_failure_message(*, status: Dict[str, Any]) -> str:
    payload = dict(status.get('payload') or {})
    payload_message = str(payload.get('failure_message', '') or '').strip()
    if payload_message:
        return payload_message
    return str(status.get('message', '') or '').strip()


def _set_if_present(result: Dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return
        result[key] = normalized
        return
    result[key] = value


def _is_recoverable_failure_reason(reason: Any) -> bool:
    lowered = str(reason or '').strip().lower()
    if not lowered:
        return False
    recoverable_tokens = (
        'navigation_',
        'grid_astar_failed',
        'point_outside_rooms',
        'rooms_not_connected',
        'no_nearby_free_cell',
        'did not fully settle',
        'timeout',
        'goal_in_collision',
        'start_in_collision',
        'not_reached',
        'grasp_not_acquired',
        'pose_unavailable',
        'manipulation_base_pose_not_found',
        'lateral_shift',
        'rotation_failed',
    )
    return any(token in lowered for token in recoverable_tokens)


def _result_requires_recovery(result: Dict[str, Any]) -> bool:
    if result.get('ok') is False:
        return True
    if not bool(result.get('recoverable', False)):
        return False
    if result.get('grasp_success') is False:
        return True
    if result.get('reached') is False:
        return True
    return False


def _resolve_dependency(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str) and '{{' in value and '}}' in value:
        pattern = r'\{\{(step\d+\.result(?:\.\w+)*(?:\s*[+\-*/]\s*[\d.]+)?)\}\}'
        matches = re.findall(pattern, value)
        if matches:
            resolved_value = value
            for match in matches:
                result = _evaluate_reference(match, context)
                placeholder = f'{{{{{match}}}}}'
                if resolved_value == placeholder:
                    return result
                resolved_value = resolved_value.replace(placeholder, str(result))
            return resolved_value

    if isinstance(value, dict):
        return {key: _resolve_dependency(item, context) for key, item in value.items()}

    if isinstance(value, list):
        return [_resolve_dependency(item, context) for item in value]

    return value


def _evaluate_reference(reference: str, context: Dict[str, Any]) -> Any:
    arithmetic_pattern = r'(step\d+\.result(?:\.\w+)*)\s*([+\-*/])\s*([\d.]+)'
    arithmetic_match = re.match(arithmetic_pattern, reference)
    if arithmetic_match:
        base_reference = arithmetic_match.group(1)
        operator = arithmetic_match.group(2)
        operand = float(arithmetic_match.group(3))
        value = float(_evaluate_reference(base_reference, context))
        if operator == '+':
            return value + operand
        if operator == '-':
            return value - operand
        if operator == '*':
            return value * operand
        if operator == '/':
            return value / operand

    parts = reference.split('.')
    step_key = parts[0]
    if step_key not in context:
        raise ValueError(f'Dependency error: {step_key} not found in context.')

    result: Any = context[step_key]['result']
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            pass

    for part in parts[2:]:
        if isinstance(result, dict):
            result = result.get(part)
        else:
            raise ValueError(f"Cannot access field '{part}' in non-dict result.")
    return result

if __name__ == '__main__':
    # TODO 调试suggest_manipulation_base_pose函数
    pass

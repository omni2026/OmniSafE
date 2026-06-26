from __future__ import annotations

import importlib.util
import json
import os
import queue
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from isaacsim import SimulationApp

try:
    from .pipe_communication import PipeCommunicationServer
except ImportError:
    from pipe_communication import PipeCommunicationServer

try:
    from .fetch_nav_planner import ensure_shapely
except ImportError:
    from fetch_nav_planner import ensure_shapely

try:
    from .world_state import WorldStateError, WorldStateStore
except ImportError:
    from world_state import WorldStateError, WorldStateStore

try:
    from .collision_grouping import (
        ROOM_DIRECT_CHILD_STRUCTURAL_KEYWORDS,
        ROOMS_ROOT_PATH,
        classify_room_direct_child,
        room_direct_child_identity,
    )
except ImportError:
    from collision_grouping import (
        ROOM_DIRECT_CHILD_STRUCTURAL_KEYWORDS,
        ROOMS_ROOT_PATH,
        classify_room_direct_child,
        room_direct_child_identity,
    )


def _load_project_dotenv() -> None:
    candidates = [
        Path(__file__).resolve().parents[2] / '.env',
        Path.cwd() / '.env',
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:].strip()
            if '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value
        seen.add(candidate)


_load_project_dotenv()


def _read_env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return bool(default)
    normalized = raw_value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on', 'enable', 'enabled'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off', 'disable', 'disabled'}:
        return False
    return bool(default)


def _isaac_sim_headless_enabled() -> bool:
    # Preserve the previous defaults unless explicitly overridden:
    # Windows used to launch with a visible UI; Linux used to launch headless.
    default_headless = sys.platform == 'linux'
    for env_name in (
        'OMNISAFE_ISAAC_HEADLESS',
        'ISAACSIM_HEADLESS',
        'OMNISAFE_HEADLESS',
    ):
        if os.getenv(env_name) is not None:
            return _read_env_bool(env_name, default_headless)
    return default_headless


def _create_simulation_app():
    headless = _isaac_sim_headless_enabled()
    if sys.platform == 'win32':
        return SimulationApp({'headless': headless})

    if sys.platform == 'linux':
        config = {
            'headless': headless,
            'hide_ui': _read_env_bool('OMNISAFE_ISAAC_HIDE_UI', False),
            'display_options': 3286,
        }
        app = SimulationApp(launch_config=config)
        enable_livestream = _read_env_bool(
            'OMNISAFE_ISAAC_LIVESTREAM',
            True,
        )

        if enable_livestream:
            from isaacsim.core.utils.extensions import enable_extension

            app.set_setting('/app/window/drawMouse', True)
            enable_extension('omni.services.livestream.nvcf')
        return app

    raise OSError(f'Unsupported platform: {sys.platform}')


simulation_app = _create_simulation_app()

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_euler_angles
from omni.isaac.core.utils.stage import is_stage_loading
from omni.isaac.core.utils.xforms import get_world_pose
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdSkel

import omni.client
import omni.usd


class ManipulationBasePosePlanningError(RuntimeError):
    def __init__(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.payload = dict(payload or {})


class RuntimeContactSafetyMonitor:
    """Physics-step runtime monitor for contact/collision assertions."""

    CONTACT_PREDICATES = {'runtime_contact', 'contact', 'collision'}

    def __init__(self, runner: 'IsaacSimAppRunner') -> None:
        self.runner = runner
        self.assertions: list[Dict[str, Any]] = []
        self.events: list[Dict[str, Any]] = []
        self.observations: list[Dict[str, Any]] = []
        self.contact_events: Dict[str, Dict[str, Any]] = {}
        self._contact_start_centers: Dict[str, tuple[list[float], list[float]]] = {}
        self._active_contact_keys: set[str] = set()
        self._physics_step = 0
        self._callback_id: Any = None
        self._callback_registered = False
        self._callback_error = ''

    def start(self) -> None:
        if self._callback_registered:
            return
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            try:
                from isaacsim.core.simulation_manager import SimulationEvent

                post_physics_step_event = SimulationEvent.PHYSICS_POST_STEP
            except ImportError:
                from isaacsim.core.simulation_manager import IsaacEvents

                post_physics_step_event = IsaacEvents.POST_PHYSICS_STEP

            self._callback_id = SimulationManager.register_callback(
                lambda *args, **kwargs: self.on_physics_step(*args, **kwargs),
                event=post_physics_step_event,
            )
            self._callback_registered = True
            self._callback_error = ''
        except Exception as exc:
            self._callback_registered = False
            self._callback_error = f'{exc.__class__.__name__}: {exc}'

    def reset(self) -> None:
        self.assertions = []
        self.events = []
        self.observations = []
        self.contact_events = {}
        self._contact_start_centers = {}
        self._active_contact_keys = set()
        self._physics_step = 0

    def register_assertions(self, assertions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        runtime_assertions = []
        for raw_assertion in assertions or []:
            assertion = dict(raw_assertion or {})
            trigger = dict(assertion.get('trigger') or {})
            if not self._is_contact_trigger(trigger):
                continue
            runtime_assertions.append({
                'assertion_id': str(assertion.get('assertion_id') or assertion.get('id') or f'contact_{len(runtime_assertions) + 1}'),
                'description': str(assertion.get('description') or ''),
                'trigger': trigger,
                'severity': str(assertion.get('severity') or 'HIGH').upper(),
                'unsafe_event_category': str(assertion.get('unsafe_event_category') or assertion.get('category') or 'runtime_contact'),
            })
        self.assertions = runtime_assertions
        self.events = []
        self._active_contact_keys = set()
        self.start()
        return {
            'registered_count': len(self.assertions),
            'callback_registered': bool(self._callback_registered),
            'callback_error': self._callback_error,
            'assertion_ids': [item['assertion_id'] for item in self.assertions],
        }

    def get_events(self, *, clear: bool = False) -> list[Dict[str, Any]]:
        events = [dict(event) for event in self.events]
        if clear:
            self.events = []
        return events

    def register_observations(self, observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        supported = {
            'object_contact',
            'objects_in_contact',
            'object_contact_with_object',
            'object_contact_motion',
        }
        self.observations = [
            dict(item or {})
            for item in observations or []
            if str(dict(item or {}).get('predicate') or '').strip() in supported
        ]
        self.contact_events = {}
        self._contact_start_centers = {}
        self.start()
        return {
            'registered_count': len(self.observations),
            'observation_ids': [
                str(item.get('usage_id') or item.get('id') or '')
                for item in self.observations
            ],
        }

    def get_contact_events(self) -> list[Dict[str, Any]]:
        return [
            dict(event)
            for event in sorted(
                self.contact_events.values(),
                key=lambda item: (
                    int(item.get('physics_step', 0) or 0),
                    str(item.get('contact_key') or ''),
                ),
            )
        ]

    def clear_events(self) -> None:
        self.events = []

    def on_physics_step(self, *args, **kwargs) -> None:
        _ = (args, kwargs)
        if (not self.assertions and not self.observations) or self.runner.stage is None:
            return
        self._physics_step += 1
        current_contact_keys: set[str] = set()
        try:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            )
        except Exception:
            bbox_cache = None

        for assertion in self.assertions:
            contact = self._evaluate_contact_assertion(assertion, bbox_cache)
            if not contact:
                continue
            contact_key = str(contact.get('contact_key') or assertion['assertion_id'])
            current_contact_keys.add(contact_key)
            if contact_key in self._active_contact_keys:
                continue
            self.events.append(self._unsafe_event(assertion, contact))
        self._update_contact_observations(bbox_cache)
        self._active_contact_keys = current_contact_keys

    def _update_contact_observations(self, bbox_cache: Any) -> None:
        current_keys: set[str] = set()
        for observation in self.observations:
            args = dict(observation.get('arguments') or observation.get('args') or {})
            usage_id = str(
                observation.get('usage_id')
                or observation.get('id')
                or observation.get('predicate')
                or 'contact_observation'
            )
            contact = self._evaluate_contact_pair(args, bbox_cache)
            if not contact:
                continue
            key = f"{usage_id}:{contact['source_prim_path']}:{contact['target_prim_path']}"
            current_keys.add(key)
            source_center = self._aabb_center(contact['source_prim_path'], bbox_cache)
            target_center = self._aabb_center(contact['target_prim_path'], bbox_cache)
            if key not in self._contact_start_centers and source_center and target_center:
                self._contact_start_centers[key] = (source_center, target_center)
            relative_displacement = 0.0
            start_centers = self._contact_start_centers.get(key)
            if start_centers and source_center and target_center:
                relative_displacement = sum(
                    (
                        (source_center[index] - start_centers[0][index])
                        - (target_center[index] - start_centers[1][index])
                    ) ** 2
                    for index in range(3)
                ) ** 0.5
            existing = dict(self.contact_events.get(key) or {})
            start_physics_step = int(
                existing.get('start_physics_step', self._physics_step)
                or self._physics_step
            )
            duration_s = max(
                0.0,
                float(self._physics_step - start_physics_step)
                * float(self.runner.DEFAULT_PHYSICS_DT),
            )
            min_motion = float(
                args.get('min_displacement_m')
                or args.get('min_tangential_displacement_m')
                or 0.02
            )
            min_duration = float(args.get('min_contact_duration_s', 0.1) or 0.1)
            classification = ''
            if relative_displacement >= min_motion and duration_s >= min_duration:
                requested_motion = str(args.get('motion') or '').strip().lower()
                classification = requested_motion if requested_motion in {
                    'scraping',
                    'wiping',
                    'pushing',
                } else 'scraping'
            self.contact_events[key] = {
                'contact_key': key,
                'usage_id': usage_id,
                'predicate': str(observation.get('predicate') or ''),
                'source': str(
                    args.get('object')
                    or args.get('object_a')
                    or args.get('source')
                    or args.get('body')
                    or ''
                ),
                'target': str(args.get('target') or args.get('object_b') or ''),
                'source_prim_path': contact['source_prim_path'],
                'target_prim_path': contact['target_prim_path'],
                'phase': 'active',
                'motion': classification or ('contact_motion' if relative_displacement > 0.0 else ''),
                'classification': classification,
                'relative_tangential_displacement_m': float(relative_displacement),
                'contact_duration_s': duration_s,
                'start_physics_step': start_physics_step,
                'step': int(self.runner.command_counter),
                'physics_step': self._physics_step,
                'detection_method': contact['detection_method'],
                'force_available': False,
            }
        for key, event in list(self.contact_events.items()):
            if event.get('phase') == 'active' and key not in current_keys:
                event['phase'] = 'ended'
                event['physics_step'] = self._physics_step
                event['step'] = int(self.runner.command_counter)
                self._contact_start_centers.pop(key, None)

    def _evaluate_contact_assertion(self, assertion: Dict[str, Any], bbox_cache: Any) -> Optional[Dict[str, Any]]:
        trigger = dict(assertion.get('trigger') or {})
        args = dict(trigger.get('arguments') or trigger.get('args') or {})
        return self._evaluate_contact_pair(
            args,
            bbox_cache,
            assertion_id=assertion['assertion_id'],
        )

    def _evaluate_contact_pair(
        self,
        args: Dict[str, Any],
        bbox_cache: Any,
        *,
        assertion_id: str = 'observation',
    ) -> Optional[Dict[str, Any]]:
        margin = float(args.get('margin', args.get('contact_margin', 0.005)) or 0.005)
        source_paths = self._resolve_source_paths(args)
        target_paths = self._resolve_target_paths(args)
        if not source_paths or not target_paths:
            return None
        for source_path in source_paths:
            for target_path in target_paths:
                if self._same_or_nested_path(source_path, target_path):
                    continue
                if self._paths_overlap(source_path, target_path, bbox_cache, margin=margin):
                    return {
                        'contact_key': f"{assertion_id}:{source_path}:{target_path}",
                        'source_prim_path': source_path,
                        'target_prim_path': target_path,
                        'margin': margin,
                        'physics_step': self._physics_step,
                        'detection_method': 'physics_post_step_aabb_overlap',
                    }
        return None

    def _resolve_source_paths(self, args: Dict[str, Any]) -> list[str]:
        explicit_paths = self._normalize_path_list(
            args.get('source_prim_paths') or args.get('source_prim_path') or args.get('body_prim_paths')
        )
        if explicit_paths:
            return explicit_paths

        body = str(
            args.get('body')
            or args.get('source')
            or args.get('object')
            or args.get('object_a')
            or 'robot'
        ).strip()
        body_lower = body.lower()
        if body_lower in {'robot', 'agent', 'fetch'}:
            robot_root = (
                str(self.runner.robot_articulation_path or '').strip()
                or str(self.runner.robot_prim_path or '').strip()
            )
            return self.runner._collect_active_collision_prim_paths(
                robot_root,
                max_count=64,
            )
        if body_lower in {'*', 'any', 'any_object'}:
            return self._active_collision_or_root_paths(
                self.runner.object_prim_paths.values(),
                max_count=64,
            )
        prim_path = self.runner._lookup_object_prim_path(body)
        if not prim_path:
            return []
        return self.runner._collect_active_collision_prim_paths(
            prim_path,
            max_count=64,
        )

    def _resolve_target_paths(self, args: Dict[str, Any]) -> list[str]:
        explicit_paths = self._normalize_path_list(
            args.get('target_prim_paths') or args.get('target_prim_path')
        )
        if explicit_paths:
            return explicit_paths

        target = str(
            args.get('target')
            or args.get('object_b')
            or args.get('object')
            or args.get('object_name')
            or args.get('target_object')
            or ''
        ).strip()
        if target in {'*', 'any', 'any_object'}:
            return self._active_collision_or_root_paths(
                self.runner.object_prim_paths.values(),
                max_count=64,
            )
        if target:
            prim_path = self.runner._lookup_object_prim_path(target)
            if prim_path:
                return self.runner._collect_active_collision_prim_paths(
                    prim_path,
                    max_count=64,
                )
        return []

    def _active_collision_or_root_paths(
        self,
        root_paths: Sequence[str],
        *,
        max_count: Optional[int] = None,
    ) -> list[str]:
        """Return active collider paths for roots, falling back only for non-physics roots.

        A root whose subtree contains CollisionAPI prims but all of those
        colliders are disabled is intentionally skipped.  Without this
        distinction, the AABB-based monitor would keep reporting contacts for
        a latch-disabled robot/object by falling back to the root xform bbox.
        """
        paths: list[str] = []
        for root_path in root_paths or []:
            remaining = None
            if max_count is not None:
                remaining = int(max_count) - len(paths)
                if remaining <= 0:
                    break
            paths.extend(
                self.runner._collect_active_collision_prim_paths(
                    str(root_path or ''),
                    max_count=remaining,
                )
            )
        return paths

    def _paths_overlap(self, lhs_path: str, rhs_path: str, bbox_cache: Any, *, margin: float) -> bool:
        if not self.runner._collision_query_path_enabled(lhs_path):
            return False
        if not self.runner._collision_query_path_enabled(rhs_path):
            return False
        lhs = self._aabb(lhs_path, bbox_cache)
        rhs = self._aabb(rhs_path, bbox_cache)
        if lhs is None or rhs is None:
            return False
        lhs_min, lhs_max = lhs
        rhs_min, rhs_max = rhs
        for axis in range(3):
            if lhs_max[axis] + margin < rhs_min[axis]:
                return False
            if rhs_max[axis] + margin < lhs_min[axis]:
                return False
        return True

    def _aabb(self, prim_path: str, bbox_cache: Any) -> Optional[tuple[list[float], list[float]]]:
        if self.runner.stage is None or bbox_cache is None:
            return None
        prim = self.runner.stage.GetPrimAtPath(str(prim_path or ''))
        if prim is None or not prim.IsValid():
            return None
        try:
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            min_pt = box.GetMin()
            max_pt = box.GetMax()
            return (
                [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
                [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])],
            )
        except Exception:
            return None

    def _unsafe_event(self, assertion: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'event_id': (
                f"runtime_contact:{assertion['assertion_id']}:"
                f"{self.runner.command_counter}:{self._physics_step}:{len(self.events) + 1}"
            ),
            'assertion_id': assertion['assertion_id'],
            'step': int(self.runner.command_counter),
            'severity': assertion['severity'],
            'reason': assertion['description'] or assertion['unsafe_event_category'],
            'evidence': evidence,
            'source': 'runtime_contact_monitor',
        }

    @classmethod
    def _is_contact_trigger(cls, trigger: Dict[str, Any]) -> bool:
        trigger_type = str(trigger.get('type') or '').strip().lower()
        predicate = str(trigger.get('predicate') or '').strip().lower()
        return trigger_type in cls.CONTACT_PREDICATES or predicate in cls.CONTACT_PREDICATES

    @staticmethod
    def _normalize_path_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        else:
            values = list(value or []) if isinstance(value, (list, tuple, set)) else [value]
        return [str(item).strip() for item in values if str(item).strip()]

    @staticmethod
    def _same_or_nested_path(lhs: str, rhs: str) -> bool:
        lhs = str(lhs or '').strip().rstrip('/')
        rhs = str(rhs or '').strip().rstrip('/')
        return bool(lhs and rhs) and (lhs == rhs or lhs.startswith(rhs + '/') or rhs.startswith(lhs + '/'))


class IsaacSimAppRunner:
    ROBOT_REGISTRY = {
        'fetch': {
            'module_path': str((Path(__file__).resolve().parent / 'fetch_robot_control.py').resolve()),
            'class_name': 'FetchRobotController',
        }
    }
    PERSON_ASSET_PRESETS = {
        'male_adult_police_04': (
            'original_male_adult_police_04',
            'male_adult_police_04.usd',
        ),
        'male_adult_medical_01': (
            'original_male_adult_medical_01',
            'male_adult_medical_01.usd',
        ),
        'female_adult_police_02': (
            'original_female_adult_police_02',
            'female_adult_police_02.usd',
        ),
        'male_adult_construction_03': (
            'original_male_adult_construction_03',
            'male_adult_construction_03.usd',
        ),
    }
    PERSON_POSTURE_PRESETS = {
        'standing': {
            'L_Upperarm': [0.0, 0.0, -70.4],
            'R_Upperarm': [0.0, 0.0, 70.4],
        },
        'asset': {},
    }

    # 仿真主循环的默认物理步长，单位为秒。
    # 值越小，控制与碰撞更稳定，但每次动作需要的步数更多、整体执行更慢。
    # 值越大，仿真推进更快，但可能带来控制抖动、穿透或收敛不稳定。
    DEFAULT_PHYSICS_DT = 1.0 / 60.0

    # 末端执行器运动等待超时时间，单位为秒。
    # 控制 move_end_effector_to_pose 最多等待多久判定"已到达或超时"。
    # 调大：更宽容，复杂运动更容易等到收敛；调小：失败更快暴露，但可能误判慢动作。
    DEFAULT_MOVE_TIMEOUT_SEC = 25.0
    # 末端位置到达容差，单位为米。
    # 越小要求越精确，但更容易因控制噪声导致"长时间到不了"；越大则更容易判定成功。 0.015
    DEFAULT_MOVE_POSITION_TOLERANCE = 0.1
    # 末端姿态到达容差，单位为弧度。
    # 越小越严格，越能保证抓取姿态准确；越大则更容易成功，但姿态误差会更明显。 0.18
    DEFAULT_MOVE_ORIENTATION_TOLERANCE = 0.6
    # 设置pre-grasp
    ENABLE_PRE_GRASP = False
    DEFAULT_PRE_GRASP_DISTANCE = 0.12
    DEFAULT_PRE_GRASP_LINEAR_STEP = 0.03

    # 底盘导航等待超时时间，单位为秒。
    # 控制 navigate 最多等待多久判定"已到达或超时"。
    # 场景大、路径长或绕障复杂时通常需要更大值。
    DEFAULT_NAVIGATION_TIMEOUT_SEC = 60.0
    # 导航位置到达容差，单位为米。
    # 越小表示底盘必须更贴近目标点；越大则更容易结束，但停靠精度下降。 0.08
    DEFAULT_NAVIGATION_POSITION_TOLERANCE = 0.2
    # 导航朝向到达容差，单位为弧度。
    # 越小表示底盘必须更正对目标方向；越大则更容易判定成功，但机械臂朝向准备可能变差。 0.12
    DEFAULT_NAVIGATION_YAW_TOLERANCE = 0.3

    # 夹爪开合动作等待超时时间，单位为秒。
    # 调大可减少误判"夹爪未收敛"，调小则能更快发现控制异常。
    DEFAULT_GRIPPER_TIMEOUT_SEC = 2.5
    # 夹爪关节位置容差，单位通常可理解为关节位置量纲。
    # 越小越严格，越大越容易把"接近目标开合位置"判定为成功。
    DEFAULT_GRIPPER_POSITION_TOLERANCE = 0.0105
    DEFAULT_GRIPPER_OBJECT_DETECTION_THRESHOLD = 0.001
    DEFAULT_GRIPPER_DRIVE_STIFFNESS = 5000.0
    DEFAULT_GRIPPER_DRIVE_DAMPING = 500.0
    DEFAULT_GRIPPER_DRIVE_MAX_FORCE = 250.0
    # open operation
    # set baselink mass for stable open
    DEFAULT_CONTAINER_BASE_LINK_MASS = 5.0
    # set open-related settings
    DEFAULT_CONTAINER_DOOR_DRIVE_STIFFNESS = 80.0
    DEFAULT_CONTAINER_DOOR_DRIVE_DAMPING = 20.0
    DEFAULT_CONTAINER_DOOR_DRIVE_MAX_FORCE = 200.0
    DEFAULT_CONTAINER_OPEN_TIMEOUT_SEC = 5.0
    DEFAULT_ROBOT_CONTAINER_OPEN_WAYPOINT_ANGLE_RAD = 1.05
    DEFAULT_ROBOT_CONTAINER_OPEN_MAX_WAYPOINT_DISTANCE = 0.45
    DEFAULT_ROBOT_CONTAINER_HANDLE_PRE_GRASP_DISTANCE = 0.08
    # physical robot open, hard to implement :(
    DEFAULT_ENABLE_PHYSICAL_ROBOT_CONTAINER_OPEN = False
    # mock robot action when open container
    DEFAULT_ENABLE_MOCK_ROBOT_CONTAINER_OPEN = True
    
    # 抓取物体质量上限，单位为千克。当前的抓取会在抓取后调整该质量
    # Grasp target runtime mass cap in kg. This near-zero value approximates a massless object.
    # 这里试过0.05不行,1e-4不行.
    DEFAULT_MAX_GRASPED_OBJECT_MASS = 0.01

    DEFAULT_GRASP_STATIC_FRICTION = 2.5
    DEFAULT_GRASP_DYNAMIC_FRICTION = 2.0
    DEFAULT_GRASP_RESTITUTION = 0.0

    # latch mode: 当启用的时候，进行mock grasp
    ENABLE_GRASP_POSE_LATCH = True

    DEFAULT_TORSO_TIMEOUT_SEC = 4.0
    DEFAULT_TORSO_POSITION_TOLERANCE = 0.01

    # 机器人出生点与障碍物的最小安全半径，单位为米。
    # 越大，初始站位越保守；越小，更容易在拥挤房间里找到出生点，但碰撞风险上升。
    DEFAULT_ROBOT_SPAWN_CLEARANCE_RADIUS = 0.45
    # 出生点与房间墙体/边界的最小保留边距，单位为米。
    # 越大越不容易贴墙出生；越小则更容易在窄房间里找到候选位置。
    DEFAULT_ROBOT_SPAWN_WALL_MARGIN = 0.35
    # 随机采样出生点时的最大尝试次数。
    # 越大越有机会在复杂环境中找到可行位置，但初始化更慢。
    DEFAULT_SPAWN_ATTEMPTS = 80
    # 房间自由空间栅格采样分辨率，单位为米。
    # 越小采样越细、候选点质量可能更好，但搜索更慢；越大则更快但更粗糙。
    DEFAULT_ROBOT_SPAWN_GRID_RESOLUTION = 0.22
    # 机器人出生后用于"静置收敛"的仿真步数。
    # 越大越有助于底盘和关节稳定落地，但初始化耗时更长。
    DEFAULT_ROBOT_SPAWN_SETTLE_STEPS = 24

    # 末端操作前，为 RMPFlow 自动注册附近障碍物时的查询半径，单位为米。
    # 越大考虑的邻近障碍物越多，避障更保守；越小则更激进，可能漏掉侧边障碍物。
    DEFAULT_ARM_OBSTACLE_QUERY_DISTANCE = 1
    # 自动注册到 RMPFlow 的最大障碍物数量。
    # 越大避障更全面，但规划/控制负担更重；越小则更快，但可能忽略重要障碍。
    DEFAULT_ARM_OBSTACLE_MAX_COUNT = 12
    # 是否开启避障？避障：选择目标物体周围的物体，进行Mesh-level的BBOX组合近似然后加入RMPFlow的避障接口
    DEFAULT_ARM_OBSTACLE_AVOIDANCE_ENABLED = False
    # 是否开启机械臂自碰撞过滤？开启后会设置机械臂与目标物品周围的物品不发生碰撞。
    DEFAULT_ARM_COLLISION_FILTERING_ENABLED = True
    DEFAULT_ARM_COLLISION_FILTER_QUERY_PADDING = 1.0
    DEFAULT_ARM_COLLISION_FILTER_MAX_PAIRS = 2048
    # Keep robot-vs-object FilteredPairs authored during manipulation until
    # the runtime is reset/reloaded. This prevents grasp-time collision
    # cancellation from being restored immediately after reaching/grasping.
    DEFAULT_PERSIST_ARM_COLLISION_FILTERING = True
    # Most aggressive collision strategy. When enabled, robot-vs-scene-object
    # contacts are filtered through USD Physics collision groups for the whole
    # robot lifecycle. This intentionally avoids grasp-time FilteredPairsAPI
    # authoring, whose pair count grows with robot collider count * scene
    # object collider count.
    ENABLE_AGGRESSIVE_LIFECYCLE_COLLISION_FILTERING = True
    AGGRESSIVE_LIFECYCLE_COLLISION_GROUP_ROOT_PATH = '/omnisafe_collision_groups'
    AGGRESSIVE_LIFECYCLE_LEGACY_COLLISION_GROUP_ROOT_PATHS = (
        '/World/omnisafe_collision_groups',
    )
    AGGRESSIVE_LIFECYCLE_ROBOT_COLLISION_GROUP_PATH = (
        '/omnisafe_collision_groups/robotCollisionGroup'
    )
    AGGRESSIVE_LIFECYCLE_SCENE_OBJECT_COLLISION_GROUP_PATH = (
        '/omnisafe_collision_groups/sceneObjectCollisionGroup'
    )
    AGGRESSIVE_LIFECYCLE_STRUCTURAL_COLLISION_GROUP_PATH = (
        '/omnisafe_collision_groups/structuralCollisionGroup'
    )
    # 只有高度超过该阈值的物体才会被当作机械臂避障障碍物，单位为米。
    # 主要用来过滤地面薄物体、装饰面等对机械臂意义不大的障碍。
    DEFAULT_ARM_OBSTACLE_MIN_HEIGHT = 0.05

    # 以下参数用于 suggest_manipulation_base_pose 的启发式求解。
    # 除特别说明外，单位均为米。

    # 底盘中心到目标物品抓取点的期望平面距离。
    # 这是机械臂可操作性的核心参数：机器人 base 到 grasp point 的平面距离应尽量固定，
    # 以保证机械臂工作空间的一致性。
    DEFAULT_MANIPULATION_BASE_REACH = 0.65
    # Legacy compatibility knobs from the old support-edge sampler.  The current
    # suggest_manipulation_base_pose nav-target sampler no longer depends on
    # these values; it samples a ring controlled by DEFAULT_MANIPULATION_BASE_REACH.
    # 底盘中心相对支撑物边缘的最小安全间距（防碰撞下限）。
    # 当候选点按 reach 生成后，还需确保不进入支撑物内部；此值仅作为安全兜底。
    DEFAULT_MANIPULATION_BASE_STANDOFF = 0.15
    # 候选底盘点到目标物体的最大允许平面距离。
    # 与 BASE_REACH 配合使用：snap/navigation 微调后允许偏离 reach 的上限。
    DEFAULT_MANIPULATION_MAX_TARGET_DISTANCE = 1.4
    # 候选底盘点到目标物体的最小允许平面距离。
    # 与 BASE_REACH 配合使用：snap/navigation 微调后允许偏离 reach 的下限。
    DEFAULT_MANIPULATION_MIN_TARGET_DISTANCE = 0.80
    # 候选点吸附到最近 free-space 后允许的最大位移。
    # 越大越容易在拥挤环境里找到可行点，但返回点可能不再真正"在目标支撑面一侧"。
    DEFAULT_MANIPULATION_MAX_SNAP_DISTANCE = 0.35
    # 沿候选支撑面边缘做横向偏移时的尝试序列，按顺序依次测试。
    # 该顺序会直接影响"第一个可行站位"最终落在哪个侧向位置。
    DEFAULT_MANIPULATION_LATERAL_OFFSETS = (0.0, -0.18, 0.18, -0.32, 0.32)
    # 目标物体 bbox 底面允许高于支撑物顶面的最大垂直间隙。
    # 用于判断"这个桌面/台面是否真的在支撑该物体"。
    DEFAULT_SUPPORT_VERTICAL_GAP_TOLERANCE = 0.5
    DEFAULT_MANIPULATION_TORSO_HEIGHT_OFFSET = 1.3113
    # Alternative deterministic grasp pose used for retries. The tilt is
    # measured away from vertical top-down, inside the vertical plane spanned by
    # the object and the robot/base center.
    DEFAULT_MANIPULATION_GRASP_TILT_DEGREES = 45.0
    DEBUG_VISUALIZE_MANIPULATION_SUGGESTION = True
    # 蓝色：机器人当前底盘起点
    # 白色：物体中心
    # 黄色：实际用于建议的抓取目标点
    # 紫色：support 候选物体中心
    # 红色：原始候选 base 点
    # 橙色：snap 到 free space 后但被淘汰的点
    # 绿色：最终选中的点
    MANIPULATION_SUGGESTION_DEBUG_ROOT = '/World/debug/manipulation_base_pose'
    # Keep this off by default so the runtime matches tests/debug_point_cloud.py,
    # which sends world-frame point clouds without an extra base transform.
    GRASP_SERVER_USE_WORLD_TO_ROBOT_BASE = False
    # 支撑物名字先验：命中这些关键词会提升它被识别为桌面/台面/架子的概率。
    # 如果你们数据集里的家具命名有明显规律，这里通常是最值得优先补充的地方。
    DEFAULT_SUPPORT_SURFACE_KEYWORDS = (
        'table',
        'desk',
        'counter',
        'countertop',
        'shelf',
        'cabinet',
        'drawer',
        'stand',
        'bench',
        'cart',
        'island',
    )
    ARCHITECTURAL_OBJECT_KEYWORDS = (
        'floor',
        'ground',
        'wall',
        'door',
        'ceiling',
        'room',
        'light',
        'camera',
        'window',
    )
    AGGRESSIVE_LIFECYCLE_COLLISION_FILTER_STRUCTURAL_KEYWORDS = tuple(
        sorted(ROOM_DIRECT_CHILD_STRUCTURAL_KEYWORDS)
    )
    GRIPPER_HIGH_FRICTION_LINK_NAMES = (
        'l_gripper_finger_link',
        'r_gripper_finger_link',
    )
    ARM_COLLISION_FILTER_LINK_NAMES = (
        'shoulder_pan_link',
        'shoulder_lift_link',
        'upperarm_roll_link',
        'elbow_flex_link',
        'forearm_roll_link',
        'wrist_flex_link',
        'wrist_roll_link',
        'gripper_link',
        'l_gripper_finger_link',
        'r_gripper_finger_link',
    )

    def __init__(self, sim_app: SimulationApp):
        self.simulation_app = sim_app
        self.usd_file_path = ''

        self.usd_context = omni.usd.get_context()
        self.stage = None
        self.my_world: Optional[World] = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.pipe_server = None
        self.command_counter = 0
        self.last_command = None
        self.robot_pose = self._default_robot_pose()
        self.object_prim_paths: Dict[str, str] = {}
        self.object_poses: Dict[str, Dict[str, Any]] = {}
        self.object_bounds: Dict[str, Dict[str, Any]] = {}
        self.room_index: Dict[str, Dict[str, Any]] = {}
        self.gripper_state = 'open'
        self.object_in_gripper = False
        self.grasped_object_name = ''
        self.grasped_object_mass: Optional[float] = None
        self.robot_name = ''
        self.requested_room_name = ''
        self.current_room_name = ''
        self.spawn_room_name = ''
        self.spawn_pose = self._default_robot_pose()

        self.robot_controller: Any = None
        self.robot_prim_path = ''
        self.robot_articulation_path = ''
        self.control_module_path = ''
        self.controller_class_name = ''

        self._active_arm_target: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._active_gripper_action: Optional[str] = None
        self._active_navigation_target: Optional[Dict[str, Any]] = None
        self._grasp_mass_override_state: Optional[Dict[str, Any]] = None
        self._pending_grasp_pose_latch_state: Optional[Dict[str, Any]] = None
        self._grasp_pose_latch_state: Optional[Dict[str, Any]] = None
        self._preferred_grasp_cache: Dict[str, Dict[str, Any]] = {}
        self.persist_arm_collision_filtering = self._read_env_bool(
            'OMNISAFE_PERSIST_ARM_COLLISION_FILTERING',
            bool(self.DEFAULT_PERSIST_ARM_COLLISION_FILTERING),
        )
        self._persistent_arm_collision_filter_pairs: set[tuple[str, str]] = set()
        self.enable_aggressive_lifecycle_collision_filtering = self._read_env_bool(
            'OMNISAFE_AGGRESSIVE_LIFECYCLE_COLLISION_FILTERING',
            bool(self.ENABLE_AGGRESSIVE_LIFECYCLE_COLLISION_FILTERING),
        )
        # Legacy pair state is kept only so older pair-authored sessions can be
        # cleaned up. The active aggressive strategy below is collision groups.
        self._aggressive_lifecycle_collision_filter_pairs: set[tuple[str, str]] = set()
        self._aggressive_lifecycle_collision_exempt_root_paths: set[str] = set()
        self._aggressive_lifecycle_collision_group_robot_root_paths: set[str] = set()
        self._aggressive_lifecycle_collision_group_object_root_paths: set[str] = set()
        self._aggressive_lifecycle_collision_group_structural_root_paths: set[str] = set()
        self._aggressive_lifecycle_collision_group_filtered_targets: set[str] = set()
        self._aggressive_lifecycle_collision_group_robot_collider_count = 0
        self._aggressive_lifecycle_collision_group_object_collider_count = 0
        self._aggressive_lifecycle_collision_group_structural_collider_count = 0
        self._aggressive_lifecycle_collision_group_robot_target_count = 0
        self._aggressive_lifecycle_collision_group_object_target_count = 0
        self._aggressive_lifecycle_collision_group_structural_target_count = 0
        self._aggressive_lifecycle_collision_group_physics_scene_count = 0
        self._aggressive_lifecycle_collision_filter_pre_reset_applied = False
        self._aggressive_lifecycle_collision_filter_last_error = ''
        self._rng = np.random.default_rng()
        self.enable_arm_obstacle_avoidance = bool(self.DEFAULT_ARM_OBSTACLE_AVOIDANCE_ENABLED)
        self.enable_arm_collision_filtering = bool(self.DEFAULT_ARM_COLLISION_FILTERING_ENABLED)
        self.runtime_safety_monitor = RuntimeContactSafetyMonitor(self)
        self.runtime_observation_specs: list[Dict[str, Any]] = []
        self.world_state = WorldStateStore()
        self._world_state_wall_start = time.monotonic()

        # Top-down screenshot capture pipeline (lazily created on first capture).
        self._top_down_render_product: Any = None
        self._top_down_rgb_annotator: Any = None
        self._top_down_resolution: Optional[tuple[int, int]] = None
        self._top_down_capture_error: str = ''

    @staticmethod
    def _default_robot_pose():
        return {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        }

    @staticmethod
    def _sanitize_prim_identifier(value: str, *, fallback: str) -> str:
        normalized = ''.join(
            char if char.isalnum() or char == '_' else '_'
            for char in str(value or '').strip()
        ).strip('_')
        if not normalized:
            normalized = fallback
        if normalized[0].isdigit():
            normalized = f'_{normalized}'
        return normalized

    @staticmethod
    def _is_remote_asset_path(asset_path: str) -> bool:
        lowered = str(asset_path or '').strip().lower()
        return '://' in lowered

    @staticmethod
    def _join_asset_path(root: str, *parts: str) -> str:
        normalized_root = str(root or '').strip().rstrip('/\\')
        normalized_parts = [
            str(part or '').strip().strip('/\\')
            for part in parts
            if str(part or '').strip()
        ]
        return '/'.join([normalized_root, *normalized_parts])

    def _resolve_local_asset_path(self, asset_path: str) -> str:
        requested = str(asset_path or '').strip()
        if not requested or self._is_remote_asset_path(requested):
            return requested

        path = Path(requested).expanduser()
        if path.is_absolute():
            return str(path.resolve())

        candidates = []
        if self.usd_file_path:
            candidates.append(Path(self.usd_file_path).resolve().parent / path)
        candidates.append(Path.cwd() / path)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str(path)

    def _person_asset_candidates(
        self,
        *,
        asset_path: str,
        asset_preset: str,
    ) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        explicit_path = self._resolve_local_asset_path(asset_path)
        if explicit_path:
            return [('explicit', explicit_path)]

        env_path = self._resolve_local_asset_path(os.getenv('ISAACSIM_PERSON_ASSET', ''))
        if env_path:
            candidates.append(('environment', env_path))

        normalized_preset = str(asset_preset or 'male_adult_police_04').strip().lower()
        preset = self.PERSON_ASSET_PRESETS.get(normalized_preset)
        if preset is None:
            supported = ','.join(sorted(self.PERSON_ASSET_PRESETS))
            raise ValueError(f'unsupported_person_asset_preset:{normalized_preset}; supported=[{supported}]')
        asset_folder, asset_file = preset

        try:
            from isaacsim.core.utils.nucleus import get_assets_root_path

            assets_root = str(get_assets_root_path() or '').strip()
        except Exception:
            assets_root = ''
        if assets_root:
            for folder_name in dict.fromkeys((asset_folder, normalized_preset)):
                candidates.append((
                    'isaac_assets',
                    self._join_asset_path(
                        assets_root,
                        'Isaac',
                        'People',
                        'Characters',
                        folder_name,
                        asset_file,
                    ),
                ))

        isaac_path = str(os.getenv('ISAAC_PATH', '') or '').strip()
        if isaac_path:
            collected_fallback = (
                Path(isaac_path)
                / 'emAgent'
                / 'scenes'
                / 'desktop_demo'
                / 'Collected_demo'
                / 'Collected_demo'
                / 'SubUSDs'
                / 'adjust.usd'
            )
            if collected_fallback.exists():
                candidates.append(('local_collected_fallback', str(collected_fallback.resolve())))

        for folder_name in dict.fromkeys((asset_folder, normalized_preset)):
            candidates.append((
                'nvidia_fallback',
                self._join_asset_path(
                    'http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5',
                    'Isaac',
                    'People',
                    'Characters',
                    folder_name,
                    asset_file,
                ),
            ))
        return candidates

    def _resolve_person_asset_path(
        self,
        *,
        asset_path: str,
        asset_preset: str,
    ) -> tuple[str, str, list[Dict[str, Any]]]:
        attempts: list[Dict[str, Any]] = []
        remote_fallback: Optional[tuple[str, str]] = None
        for source, candidate in self._person_asset_candidates(
            asset_path=asset_path,
            asset_preset=asset_preset,
        ):
            normalized_candidate = str(candidate or '').strip()
            if not normalized_candidate:
                continue
            if not self._is_remote_asset_path(normalized_candidate):
                exists = Path(normalized_candidate).exists()
                attempts.append({
                    'source': source,
                    'asset_path': normalized_candidate,
                    'available': bool(exists),
                })
                if exists:
                    return normalized_candidate.replace('\\', '/'), source, attempts
                continue

            if remote_fallback is None:
                remote_fallback = (source, normalized_candidate)
            try:
                result, _ = omni.client.stat(normalized_candidate)
                available = result == omni.client.Result.OK
            except Exception:
                available = False
            attempts.append({
                'source': source,
                'asset_path': normalized_candidate,
                'available': bool(available),
            })
            if available:
                return normalized_candidate, source, attempts

        if remote_fallback is not None:
            source, candidate = remote_fallback
            return candidate, source, attempts
        raise FileNotFoundError(f'person_asset_not_found:{attempts}')

    def _resolve_person_parent(
        self,
        room_name: str,
    ) -> tuple[str, str]:
        requested_room = str(room_name or '').strip()
        if not requested_room:
            requested_room = str(
                self._detect_current_room_name()
                or self.spawn_room_name
                or ''
            ).strip()

        if requested_room:
            room_entry = self._resolve_room_entry(requested_room)
            if room_entry is None:
                available = ','.join(sorted(self.room_index))
                raise ValueError(f'room_not_found:{requested_room}; available=[{available}]')
            resolved_room_name = str(room_entry.get('room_name', '') or requested_room)
            return str(
                room_entry.get('prim_path') or f'{ROOMS_ROOT_PATH}/{resolved_room_name}'
            ), resolved_room_name

        people_root = '/World/People'
        UsdGeom.Xform.Define(self.stage, people_root)
        return people_root, ''

    def _default_person_pose(self, room_name: str) -> Dict[str, Any]:
        raw_base_pose = self._get_robot_base_pose_raw()
        if raw_base_pose is not None:
            base_position = np.array(raw_base_pose[0], dtype=float)
            base_orientation = self._normalize_quaternion(np.array(raw_base_pose[1], dtype=float))
            forward = self._rotate_vector_by_quaternion(
                np.array([1.0, 0.0, 0.0], dtype=float),
                base_orientation,
            )
            person_position = base_position + forward * 1.5
            person_yaw = self._yaw_from_quaternion(base_orientation) + float(np.pi)
            return {
                'position': {
                    'x': float(person_position[0]),
                    'y': float(person_position[1]),
                    'z': float(person_position[2]),
                },
                'orientation': {
                    'roll': 0.0,
                    'pitch': 0.0,
                    'yaw': float(person_yaw),
                },
            }

        room_entry = self._resolve_room_entry(room_name) if room_name else None
        bounds = dict((room_entry or {}).get('bounds') or {})
        minimum = dict(bounds.get('min') or {})
        maximum = dict(bounds.get('max') or {})
        return {
            'position': {
                'x': (
                    float(minimum.get('x', 0.0))
                    + float(maximum.get('x', 0.0))
                ) / 2.0,
                'y': (
                    float(minimum.get('y', 0.0))
                    + float(maximum.get('y', 0.0))
                ) / 2.0,
                'z': float(minimum.get('z', 0.0)),
            },
            'orientation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        }

    @staticmethod
    def _coerce_person_scale(value: Any) -> np.ndarray:
        if isinstance(value, dict):
            scale = np.array(
                [
                    float(value.get('x', 1.0)),
                    float(value.get('y', 1.0)),
                    float(value.get('z', 1.0)),
                ],
                dtype=float,
            )
        elif isinstance(value, (list, tuple, np.ndarray)):
            raw = np.array(value, dtype=float).reshape(-1)
            if raw.size == 1:
                scale = np.repeat(raw[0], 3)
            elif raw.size >= 3:
                scale = raw[:3]
            else:
                raise ValueError('invalid_person_scale')
        else:
            scale = np.repeat(float(value), 3)

        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError('person_scale_must_be_positive')
        return scale

    @staticmethod
    def _coerce_joint_rotation_degrees(value: Any) -> np.ndarray:
        if isinstance(value, dict):
            rotation = np.array(
                [
                    float(value.get('x', value.get('roll', 0.0))),
                    float(value.get('y', value.get('pitch', 0.0))),
                    float(value.get('z', value.get('yaw', 0.0))),
                ],
                dtype=float,
            )
        elif isinstance(value, (list, tuple, np.ndarray)):
            raw = np.array(value, dtype=float).reshape(-1)
            if raw.size < 3:
                raise ValueError('joint_rotation_requires_three_values')
            rotation = raw[:3]
        else:
            raise ValueError('invalid_joint_rotation')
        if np.any(~np.isfinite(rotation)):
            raise ValueError('joint_rotation_must_be_finite')
        return rotation

    @staticmethod
    def _normalized_joint_name(value: str) -> str:
        return ''.join(
            char.lower()
            for char in str(value or '')
            if char.isalnum()
        )

    def _person_posture_joint_rotations(
        self,
        posture: Any,
    ) -> tuple[str, Dict[str, Any]]:
        requested = str(posture or 'standing').strip().lower().replace('-', '_')
        aliases = {
            'natural': 'standing',
            'natural_standing': 'standing',
            'default': 'standing',
            'source': 'asset',
            'original': 'asset',
        }
        normalized = aliases.get(requested, requested)
        preset = self.PERSON_POSTURE_PRESETS.get(normalized)
        if preset is None:
            supported = ','.join(sorted(self.PERSON_POSTURE_PRESETS))
            raise ValueError(
                f'unsupported_person_posture:{requested}; supported=[{supported}]'
            )
        return normalized, {
            joint_name: list(rotation)
            for joint_name, rotation in preset.items()
        }

    def _match_skeleton_joint_index(
        self,
        requested_joint: str,
        joints: Sequence[Any],
    ) -> Optional[int]:
        requested = self._normalized_joint_name(requested_joint)
        if not requested:
            return None

        exact_matches = []
        suffix_matches = []
        for index, joint in enumerate(joints):
            joint_path = str(joint or '')
            full_name = self._normalized_joint_name(joint_path)
            leaf_name = self._normalized_joint_name(joint_path.rsplit('/', 1)[-1])
            if requested in {full_name, leaf_name}:
                exact_matches.append(index)
            elif full_name.endswith(requested) or leaf_name.endswith(requested):
                suffix_matches.append(index)
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return exact_matches[0]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return None

    @staticmethod
    def _euler_degrees_rotation_matrix(rotation_degrees: np.ndarray) -> Gf.Matrix4d:
        rx, ry, rz = [float(value) for value in np.array(rotation_degrees, dtype=float)]
        rotation = (
            Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), rx)
            * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), ry)
            * Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rz)
        )
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(rotation)
        return matrix

    def _apply_usd_skel_static_pose(
        self,
        person_prim,
        joint_rotations: Dict[str, Any],
    ) -> Dict[str, Any]:
        skeleton_prim = None
        for prim in Usd.PrimRange(person_prim):
            if prim.IsA(UsdSkel.Skeleton):
                skeleton_prim = prim
                break
        if skeleton_prim is None:
            return {
                'method': 'usd_skel',
                'supported': False,
                'available_joints': [],
                'applied_joints': [],
                'missing_joints': list(joint_rotations),
            }

        skeleton = UsdSkel.Skeleton(skeleton_prim)
        joints = list(skeleton.GetJointsAttr().Get() or [])
        rest_transforms = list(skeleton.GetRestTransformsAttr().Get() or [])
        if not joints or len(rest_transforms) != len(joints):
            return {
                'method': 'usd_skel',
                'supported': False,
                'skeleton_prim_path': str(skeleton_prim.GetPath()),
                'available_joints': [str(joint) for joint in joints],
                'applied_joints': [],
                'missing_joints': list(joint_rotations),
                'error': 'skeleton_rest_transforms_unavailable',
            }

        posed_transforms = [Gf.Matrix4d(transform) for transform in rest_transforms]
        applied_joints = []
        missing_joints = []
        for requested_joint, raw_rotation in joint_rotations.items():
            joint_index = self._match_skeleton_joint_index(requested_joint, joints)
            if joint_index is None:
                missing_joints.append(str(requested_joint))
                continue

            rotation_degrees = self._coerce_joint_rotation_degrees(raw_rotation)
            rest_transform = Gf.Matrix4d(posed_transforms[joint_index])
            translation = rest_transform.ExtractTranslation()
            rest_transform.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, 0.0))
            posed_transform = rest_transform * self._euler_degrees_rotation_matrix(rotation_degrees)
            posed_transform.SetTranslateOnly(translation)
            posed_transforms[joint_index] = posed_transform
            applied_joints.append({
                'requested_joint': str(requested_joint),
                'joint': str(joints[joint_index]),
                'rotation_degrees': rotation_degrees.tolist(),
            })

        if applied_joints:
            animation_name = self._sanitize_prim_identifier(
                f'{person_prim.GetName()}_StaticPose',
                fallback='StaticPose',
            )
            animation_path = person_prim.GetPath().AppendChild(animation_name)
            animation = UsdSkel.Animation.Define(self.stage, animation_path)
            animation.CreateJointsAttr().Set(joints)
            animation.SetTransforms(posed_transforms, Usd.TimeCode.Default())
            binding_api = UsdSkel.BindingAPI.Apply(skeleton_prim)
            binding_api.CreateAnimationSourceRel().SetTargets([animation_path])
        else:
            animation_path = None

        return {
            'method': 'usd_skel',
            'supported': True,
            'skeleton_prim_path': str(skeleton_prim.GetPath()),
            'animation_prim_path': str(animation_path or ''),
            'available_joint_count': len(joints),
            'available_joints': [str(joint) for joint in joints],
            'applied_joints': applied_joints,
            'missing_joints': missing_joints,
            'rotation_unit': 'degrees',
        }

    def _apply_xform_joint_pose(
        self,
        person_prim,
        joint_rotations: Dict[str, Any],
    ) -> Dict[str, Any]:
        descendants = [
            prim
            for prim in Usd.PrimRange(person_prim)
            if prim != person_prim and prim.IsA(UsdGeom.Xform)
        ]
        applied_joints = []
        missing_joints = []
        for requested_joint, raw_rotation in joint_rotations.items():
            requested = self._normalized_joint_name(requested_joint)
            candidates = [
                prim
                for prim in descendants
                if self._normalized_joint_name(prim.GetName()) == requested
            ]
            if len(candidates) != 1:
                missing_joints.append(str(requested_joint))
                continue

            rotation_degrees = self._coerce_joint_rotation_degrees(raw_rotation)
            prim = candidates[0]
            attr = prim.GetAttribute('xformOp:rotateXYZ')
            if attr is not None and attr.IsValid():
                type_name = str(attr.GetTypeName())
                value = (
                    Gf.Vec3d(*rotation_degrees.tolist())
                    if type_name == 'double3'
                    else Gf.Vec3f(*rotation_degrees.tolist())
                )
                attr.Set(value)
            else:
                rotate_op = UsdGeom.Xformable(prim).AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3f(*rotation_degrees.tolist()))
            applied_joints.append({
                'requested_joint': str(requested_joint),
                'prim_path': str(prim.GetPath()),
                'rotation_degrees': rotation_degrees.tolist(),
            })

        return {
            'method': 'xform',
            'supported': bool(descendants),
            'available_joints': sorted({
                str(prim.GetName())
                for prim in descendants
                if str(prim.GetName() or '').strip()
            }),
            'applied_joints': applied_joints,
            'missing_joints': missing_joints,
            'rotation_unit': 'degrees',
        }

    def _apply_person_joint_pose(
        self,
        person_prim,
        joint_rotations: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            skel_result = self._apply_usd_skel_static_pose(person_prim, joint_rotations)
        except Exception as exc:
            skel_result = {
                'method': 'usd_skel',
                'supported': False,
                'applied_joints': [],
                'missing_joints': list(joint_rotations),
                'error': f'{exc.__class__.__name__}: {exc}',
            }
        if skel_result.get('supported'):
            return skel_result

        try:
            xform_result = self._apply_xform_joint_pose(person_prim, joint_rotations)
        except Exception as exc:
            xform_result = {
                'method': 'xform',
                'supported': False,
                'applied_joints': [],
                'missing_joints': list(joint_rotations),
                'error': f'{exc.__class__.__name__}: {exc}',
            }
        xform_result['usd_skel_attempt'] = skel_result
        return xform_result

    def _add_person(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self.stage is None:
            raise RuntimeError('scene_not_loaded')

        raw_pose = dict(args.get('pose') or {})
        posture, joint_rotations = self._person_posture_joint_rotations(
            args.get('posture', raw_pose.get('posture', 'standing'))
        )
        joint_rotations.update(dict(
            args.get('joint_rotations')
            or raw_pose.get('joint_rotations')
            or {}
        ))

        self._refresh_room_index()
        person_name = self._sanitize_prim_identifier(
            str(args.get('person_name') or args.get('name') or 'person_01'),
            fallback='person_01',
        )
        parent_path, resolved_room_name = self._resolve_person_parent(
            str(args.get('room_name') or ''),
        )
        person_prim_path = f'{parent_path}/{person_name}'
        existing_prim = self.stage.GetPrimAtPath(person_prim_path)
        replace_existing = self._coerce_bool(args.get('replace_existing', True), default=True)
        if existing_prim is not None and existing_prim.IsValid():
            if not replace_existing:
                raise ValueError(f'person_already_exists:{person_prim_path}')
            self.stage.RemovePrim(person_prim_path)

        asset_path, asset_source, asset_attempts = self._resolve_person_asset_path(
            asset_path=str(args.get('asset_path') or ''),
            asset_preset=str(args.get('asset_preset') or 'male_adult_police_04'),
        )
        person_prim = UsdGeom.Xform.Define(self.stage, person_prim_path).GetPrim()
        visual_prim = UsdGeom.Xform.Define(self.stage, f'{person_prim_path}/Visual').GetPrim()
        model_prim = UsdGeom.Xform.Define(self.stage, f'{person_prim_path}/Visual/Model').GetPrim()

        reference_added = model_prim.GetReferences().AddReference(asset_path)
        if not reference_added:
            self.stage.RemovePrim(person_prim_path)
            raise RuntimeError(f'person_asset_reference_failed:{asset_path}')

        deadline = time.monotonic() + float(args.get('asset_load_timeout_sec', 60.0) or 60.0)
        while is_stage_loading() and time.monotonic() < deadline:
            self.simulation_app.update()
        self.simulation_app.update()
        asset_load_timed_out = bool(is_stage_loading())
        composed_prim_count = sum(1 for _ in Usd.PrimRange(model_prim))

        scale = self._coerce_person_scale(args.get('scale', 1.0))
        UsdGeom.XformCommonAPI(visual_prim).SetScale(
            Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2]))
        )

        default_pose = self._default_person_pose(resolved_room_name)
        position = dict(default_pose.get('position') or {})
        position.update(dict(raw_pose.get('position') or {}))
        orientation = dict(default_pose.get('orientation') or {})
        orientation.update(dict(raw_pose.get('orientation') or {}))
        normalized_pose_input = {
            'position': position,
            'orientation': orientation,
        }
        target_position, target_orientation, normalized_pose = self._pose_command_to_raw(
            normalized_pose_input
        )
        if not self._set_prim_world_pose_raw(
            person_prim_path,
            target_position,
            target_orientation,
        ):
            self.stage.RemovePrim(person_prim_path)
            raise RuntimeError(f'person_pose_apply_failed:{person_prim_path}')

        joint_pose_result = self._apply_person_joint_pose(
            person_prim,
            joint_rotations,
        )
        joint_pose_result['posture'] = posture

        person_prim.SetCustomDataByKey('omnisafe_entity_type', 'person')
        person_prim.SetCustomDataByKey('omnisafe_asset_path', asset_path)
        person_prim.SetCustomDataByKey('omnisafe_posture', posture)
        person_prim.SetCustomDataByKey('omnisafe_static', True)
        self._refresh_scene_index()
        bbox = self._compute_world_bbox(person_prim)
        return {
            'person_name': person_name,
            'prim_path': person_prim_path,
            'model_prim_path': str(model_prim.GetPath()),
            'room_name': resolved_room_name,
            'asset_path': asset_path,
            'asset_source': asset_source,
            'asset_resolution_attempts': asset_attempts,
            'asset_load_timed_out': asset_load_timed_out,
            'composed_prim_count': composed_prim_count,
            'pose': normalized_pose,
            'posture': posture,
            'scale': scale.tolist(),
            'joint_pose': joint_pose_result,
            'bounds': self._bbox_to_dict(*bbox) if bbox is not None else {},
            'replace_existing': replace_existing,
        }

    def _load_scene(self, usd_file_path: str) -> bool:
        if not usd_file_path:
            print('ERROR: Empty USD file path.')
            return False

        self.usd_file_path = usd_file_path
        if not os.path.exists(self.usd_file_path):
            print(f'ERROR: USD file not found at: {self.usd_file_path}')
            return False

        # Render products are bound to prims on the current stage. Reusing one
        # after open_stage() points the annotator at stale stage state.
        self._destroy_top_down_capture_pipeline()

        print(f'Loading USD file from: {self.usd_file_path}')
        try:
            success = self.usd_context.open_stage(self.usd_file_path)
            if not success:
                print(f'ERROR: Failed to open USD file: {self.usd_file_path}')
                return False
        except Exception as exc:
            print(f'ERROR: Exception while opening USD file: {exc}')
            return False

        print('Waiting for stage to load...')
        while is_stage_loading():
            self.simulation_app.update()

        self.stage = self.usd_context.get_stage()
        if self.stage is None:
            print('ERROR: Stage is None after load.')
            return False

        self._setup_world_lights_and_cameras()
        self._setup_runtime_for_stage()
        self._clear_robot_runtime(reset_config=False)
        self._refresh_scene_index()
        self._initialize_world_state()
        return True

    def _setup_world_lights_and_cameras(self):
        default_prim = UsdGeom.Xform.Define(self.stage, Sdf.Path('/World'))
        self.stage.SetDefaultPrim(default_prim.GetPrim())

        dome_light = UsdLux.DomeLight.Define(self.stage, '/World/DomeLight')
        dome_light.CreateIntensityAttr().Set(2000.0)
        dome_light.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        if not self.stage.GetPrimAtPath('/World/GroundPlane'):
            GroundPlane(prim_path='/World/GroundPlane', z_position=0)

        top_camera_prim = self.stage.GetPrimAtPath(self.TOP_DOWN_CAMERA_PRIM_PATH)
        if (
            top_camera_prim
            and top_camera_prim.IsValid()
            and not top_camera_prim.IsA(UsdGeom.Camera)
        ):
            self.stage.RemovePrim(self.TOP_DOWN_CAMERA_PRIM_PATH)
            top_camera_prim = None
        if not top_camera_prim or not top_camera_prim.IsValid():
            top_camera = UsdGeom.Camera.Define(self.stage, '/World/TopCamera')
            top_camera_prim = top_camera.GetPrim()
        else:
            top_camera = UsdGeom.Camera(top_camera_prim)
        self._set_top_down_camera_transform(top_camera_prim, 0.0, 0.0, 20.0)
        top_camera.GetFocalLengthAttr().Set(self.TOP_DOWN_FOCAL_LENGTH_MM)
        top_camera.GetClippingRangeAttr().Set(self.TOP_DOWN_CLIPPING_RANGE)
        top_camera.GetHorizontalApertureAttr().Set(20)
        top_camera.GetVerticalApertureAttr().Set(20)

        if not self.stage.GetPrimAtPath('/World/SideCamera'):
            side_camera = UsdGeom.Camera.Define(self.stage, '/World/SideCamera')
            side_camera_xform = UsdGeom.Xformable(side_camera.GetPrim())
            side_camera_xform.AddTranslateOp().Set(Gf.Vec3d(20, 0, 0))
            side_camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, -90, 0))
            side_camera.GetHorizontalApertureAttr().Set(20)
            side_camera.GetVerticalApertureAttr().Set(20)

        if not self.stage.GetPrimAtPath('/World/MainCamera'):
            main_camera = UsdGeom.Camera.Define(self.stage, '/World/MainCamera')
            main_camera_xform = UsdGeom.Xformable(main_camera.GetPrim())
            main_camera_xform.AddTranslateOp().Set(Gf.Vec3d(0, 20, 0))
            main_camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
            main_camera.GetHorizontalApertureAttr().Set(20)
            main_camera.GetVerticalApertureAttr().Set(20)

    # ------------------------------------------------------------------
    # Top-down screenshot capture
    # ------------------------------------------------------------------
    TOP_DOWN_CAMERA_PRIM_PATH = '/World/TopCamera'
    DEFAULT_TOP_DOWN_RESOLUTION = (1024, 1024)
    # Focal length (mm) for the top-down camera. Matches the scene-generator
    # eval runtime's overhead-view focal length so screenshots share the same
    # framing conventions as the dataset preview renders.
    TOP_DOWN_FOCAL_LENGTH_MM = 16.0
    # Near/far clipping range (meters). A wide far clip keeps large rooms
    # visible; a tiny near clip avoids cutting off nearby furniture. Mirrors
    # ``_define_eval_camera`` in scene_genetator/eval/isaac_eval_runtime.py.
    TOP_DOWN_CLIPPING_RANGE = (0.01, 10000.0)
    # Overhead framing constants, mirroring
    # scene_genetator/eval/isaac_eval_runtime.py::_plan_per_room_views. The
    # camera height is derived from the room footprint instead of a fixed
    # constant: camera_z = target_z + overhead_height, where
    #   target_z       = floor_z + min(room_height * 0.18, 0.55)
    #   overhead_height = max(footprint_span * 1.40, room_height + 1.5)
    TOP_DOWN_HEIGHT_SPAN_FACTOR = 1.40
    TOP_DOWN_HEIGHT_ROOM_FACTOR = 1.5
    TOP_DOWN_TARGET_Z_FACTOR = 0.18
    TOP_DOWN_TARGET_Z_MAX_M = 0.55

    @classmethod
    def _normalize_top_down_resolution(cls, resolution: Any) -> tuple[int, int]:
        try:
            seq = list(resolution or [])
            if len(seq) >= 2:
                width = max(16, int(seq[0]))
                height = max(16, int(seq[1]))
                return (width, height)
        except Exception:
            pass
        return cls.DEFAULT_TOP_DOWN_RESOLUTION

    def _resolve_top_down_camera_target(self) -> tuple[float, float, float, float, float, float]:
        """Return (center_x, center_y, span_x, span_y, floor_z, room_height_z).

        Prefers the current room's bounding box; falls back to the robot base
        position; finally falls back to the world origin. ``span_*`` are the
        X/Y extents (meters); ``floor_z`` is the room floor height and
        ``room_height_z`` is the room's Z extent, both used to size the
        adaptive top-down camera height.
        """
        # 1) Current room bounding box (most informative framing).
        try:
            room_name = str(self._detect_current_room_name() or '').strip()
        except Exception:
            room_name = ''
        if room_name and room_name in self.room_index:
            room_bounds = self._bbox_dict_to_arrays(
                self.room_index[room_name].get('bounds') or {}
            )
            if room_bounds is not None:
                room_min, room_max = room_bounds
                # _plan_per_room_views clamps each room dimension to 0.5 m.
                span_x = max(0.5, float(room_max[0]) - float(room_min[0]))
                span_y = max(0.5, float(room_max[1]) - float(room_min[1]))
                center_x = float((room_max[0] + room_min[0]) * 0.5)
                center_y = float((room_max[1] + room_min[1]) * 0.5)
                floor_z = float(room_min[2])
                room_height_z = max(0.0, float(room_max[2]) - float(room_min[2]))
                return center_x, center_y, span_x, span_y, floor_z, room_height_z

        # 2) Robot base link position.
        try:
            base_pose = self._get_robot_base_pose_raw()
        except Exception:
            base_pose = None
        if base_pose is not None:
            position = np.asarray(base_pose[0], dtype=float).reshape(-1)
            if position.size >= 2:
                floor_z = float(position[2]) if position.size >= 3 else 0.0
                return float(position[0]), float(position[1]), 6.0, 6.0, floor_z, 0.0

        # 3) World origin fallback.
        return 0.0, 0.0, 6.0, 6.0, 0.0, 0.0

    @staticmethod
    def _set_top_down_camera_transform(
        camera_prim: Any,
        center_x: float,
        center_y: float,
        camera_z: float,
    ) -> None:
        """Set the camera translation and make every authored rotation identity."""
        xform = UsdGeom.Xformable(camera_prim)
        ordered_ops = list(xform.GetOrderedXformOps())

        translate_op = next(
            (op for op in ordered_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(center_x, center_y, camera_z))

        vector_rotation_types = {
            UsdGeom.XformOp.TypeRotateXYZ,
            UsdGeom.XformOp.TypeRotateXZY,
            UsdGeom.XformOp.TypeRotateYXZ,
            UsdGeom.XformOp.TypeRotateYZX,
            UsdGeom.XformOp.TypeRotateZXY,
            UsdGeom.XformOp.TypeRotateZYX,
        }
        scalar_rotation_types = {
            UsdGeom.XformOp.TypeRotateX,
            UsdGeom.XformOp.TypeRotateY,
            UsdGeom.XformOp.TypeRotateZ,
        }
        authored_rotation = False
        for op in ordered_ops:
            if op.IsInverseOp():
                continue
            op_type = op.GetOpType()
            if op_type in vector_rotation_types:
                op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
                authored_rotation = True
            elif op_type in scalar_rotation_types:
                op.Set(0.0)
                authored_rotation = True
            elif op_type == UsdGeom.XformOp.TypeOrient:
                op.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
                authored_rotation = True

        if not authored_rotation:
            xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    def _position_top_down_camera(
        self,
        center_x: float,
        center_y: float,
        span_x: float,
        span_y: float,
        floor_z: float,
        room_height_z: float,
        resolution: tuple[int, int],
    ) -> float:
        """Move /World/TopCamera above (center_x, center_y) and configure focal
        length, clipping range and an adaptive height that frames the room.

        Mirrors the overhead-view camera setup in
        scene_genetator/eval/isaac_eval_runtime.py::_plan_per_room_views: a
        fixed 16 mm focal length, the USD default aperture, an explicit
        clipping range, and a height derived from the room footprint rather
        than a constant. The camera sits directly above the room center with
        no rotation, so it looks straight down (USD Camera default -Z forward).
        """
        if self.stage is None:
            raise RuntimeError('cannot position top-down camera without a loaded stage')

        camera_prim = self.stage.GetPrimAtPath(self.TOP_DOWN_CAMERA_PRIM_PATH)
        if camera_prim is None or not camera_prim.IsValid():
            raise RuntimeError(f'top-down camera is missing: {self.TOP_DOWN_CAMERA_PRIM_PATH}')

        camera = UsdGeom.Camera(camera_prim)
        # Adaptive overhead height, matching the scene-generator eval runtime:
        #   target_z        = floor_z + min(room_height * 0.18, 0.55)
        #   overhead_height = max(footprint_span * 1.40, room_height + 1.5)
        #   camera_z        = target_z + overhead_height
        footprint_span = max(span_x, span_y, 0.5)
        room_height = max(room_height_z, 0.5)
        target_z = float(floor_z) + min(
            room_height * self.TOP_DOWN_TARGET_Z_FACTOR,
            self.TOP_DOWN_TARGET_Z_MAX_M,
        )
        overhead_height = max(
            footprint_span * self.TOP_DOWN_HEIGHT_SPAN_FACTOR,
            room_height + self.TOP_DOWN_HEIGHT_ROOM_FACTOR,
        )
        camera_z = target_z + overhead_height

        # USD Camera looks down local -Z. With identity rotation this is world
        # -Z, i.e. the requested (0, 0, 0) true top-down orientation.
        self._set_top_down_camera_transform(
            camera_prim,
            center_x,
            center_y,
            camera_z,
        )

        camera.GetFocalLengthAttr().Set(self.TOP_DOWN_FOCAL_LENGTH_MM)
        camera.GetClippingRangeAttr().Set(self.TOP_DOWN_CLIPPING_RANGE)
        return float(camera_z)

    def _destroy_top_down_capture_pipeline(self) -> None:
        annotator = self._top_down_rgb_annotator
        render_product = self._top_down_render_product
        if annotator is not None and render_product is not None:
            try:
                annotator.detach([render_product])
            except Exception:
                pass
        if render_product is not None:
            try:
                destroy_fn = getattr(render_product, 'destroy', None)
                if callable(destroy_fn):
                    destroy_fn()
            except Exception:
                pass
        self._top_down_render_product = None
        self._top_down_rgb_annotator = None
        self._top_down_resolution = None

    def _ensure_top_down_capture_pipeline(self, resolution: tuple[int, int]) -> tuple[Any, Any]:
        """Lazily create a Replicator render product + RGB annotator for the top-down camera."""
        if (
            self._top_down_render_product is not None
            and self._top_down_rgb_annotator is not None
            and self._top_down_resolution == resolution
        ):
            return self._top_down_render_product, self._top_down_rgb_annotator

        # Drop any stale render product before creating a new one (e.g. after a resolution change).
        if self._top_down_render_product is not None:
            self._destroy_top_down_capture_pipeline()

        import omni.replicator.core as rep

        render_product = rep.create.render_product(
            self.TOP_DOWN_CAMERA_PRIM_PATH,
            resolution=resolution,
        )
        annotator = rep.AnnotatorRegistry.get_annotator('rgb')
        annotator.attach([render_product])

        self._top_down_render_product = render_product
        self._top_down_rgb_annotator = annotator
        self._top_down_resolution = resolution
        return render_product, annotator

    def _capture_top_down_screenshot(
        self,
        path: str,
        resolution: tuple[int, int],
    ) -> Dict[str, Any]:
        """Render from /World/TopCamera and write a PNG to the requested absolute path."""
        if not str(path or '').strip():
            raise ValueError('screenshot path must be non-empty')

        out_path = Path(str(path)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Reposition/orient the camera onto the current room so the frame always
        # has content regardless of where the scene lives in world space. Do not
        # silently capture from an invalid pose if transform authoring fails.
        center_x, center_y, span_x, span_y, floor_z, room_height_z = self._resolve_top_down_camera_target()
        camera_z = self._position_top_down_camera(
            center_x, center_y, span_x, span_y, floor_z, room_height_z, resolution,
        )
        camera_target = {
            'center': [center_x, center_y],
            'span': [span_x, span_y],
            'floor_z': floor_z,
            'room_height_z': room_height_z,
            'position': [center_x, center_y, camera_z],
            'orientation_xyz_degrees': [0.0, 0.0, 0.0],
        }

        _, annotator = self._ensure_top_down_capture_pipeline(resolution)

        # A newly attached annotator commonly returns a 0x0 first frame. Wait
        # for a complete RGB frame, following the reference runtime's warm-up
        # and retry behavior instead of passing an empty array to Pillow.
        rgb = None
        observed_shapes: list[tuple[int, ...]] = []
        for _ in range(30):
            if self.my_world is not None:
                try:
                    self.my_world.step(render=True)
                except Exception:
                    import omni.replicator.core as rep_step
                    rep_step.orchestrator.step()
                self.simulation_app.update()
            else:
                import omni.replicator.core as rep_step
                rep_step.orchestrator.step()
                self.simulation_app.update()

            data = annotator.get_data()
            if data is None:
                continue
            candidate = np.asarray(data)
            observed_shapes.append(tuple(int(value) for value in candidate.shape))
            if (
                candidate.ndim == 3
                and candidate.shape[0] == int(resolution[1])
                and candidate.shape[1] == int(resolution[0])
                and candidate.shape[2] >= 3
            ):
                rgb = candidate
                break

        if rgb is None:
            raise RuntimeError(
                'top-down annotator did not return a non-empty RGB frame; '
                f'observed_shapes={observed_shapes[-5:]}'
            )

        if rgb.dtype != np.uint8:
            # Some annotator versions return normalized floats in [0, 1].
            finite = rgb[np.isfinite(rgb)]
            if finite.size == 0:
                raise RuntimeError('top-down annotator returned no finite pixels')
            if float(np.max(finite)) <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.shape[2] >= 4:
            rgb = rgb[:, :, :3]
        rgb = np.ascontiguousarray(rgb)

        # Prefer Pillow; fall back to imageio if Pillow is unavailable.
        try:
            from PIL import Image as _PILImage
            _PILImage.fromarray(rgb).save(str(out_path))
        except ImportError:
            import imageio.v2 as _imageio
            _imageio.imwrite(str(out_path), rgb)

        return {
            'path': str(out_path),
            'resolution': [int(resolution[0]), int(resolution[1])],
            'camera_prim_path': self.TOP_DOWN_CAMERA_PRIM_PATH,
            'camera_target': camera_target,
        }

    def _setup_runtime_for_stage(self):
        if self.stage is None:
            print('Cannot setup stage runtime: no stage is currently loaded.')
            return

        self.my_world = World(stage_units_in_meters=1.0)
        self.runtime_safety_monitor.start()

        camera_prim = self.stage.GetPrimAtPath('/OmniverseKit_Persp')
        if camera_prim and camera_prim.IsValid():
            camera = UsdGeom.Camera(camera_prim)
            camera.GetHorizontalApertureAttr().Set(20)
            camera.GetVerticalApertureAttr().Set(20)

    def _wait_for_stage_load(self, timeout_sec: float = 60.0) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while is_stage_loading() and time.monotonic() < deadline:
            self.simulation_app.update()
        try:
            self.simulation_app.update()
        except Exception:
            pass
        return not bool(is_stage_loading())

    def _clear_robot_runtime(self, *, reset_config: bool) -> None:
        self._clear_aggressive_lifecycle_collision_filtering(remove_pairs=True)
        self._release_grasp_pose_latch(force_remove_collision_filters=True)
        self._clear_persistent_arm_collision_filtering(remove_pairs=True)
        self._grasp_mass_override_state = None
        self._pending_grasp_pose_latch_state = None
        self._grasp_pose_latch_state = None
        self._preferred_grasp_cache = {}
        self.robot_controller = None
        self.robot_prim_path = ''
        self.robot_articulation_path = ''
        self.control_module_path = ''
        self.controller_class_name = ''
        self.robot_name = ''
        self.requested_room_name = ''
        self.current_room_name = ''
        self.spawn_room_name = ''
        self.spawn_pose = self._default_robot_pose()
        self._active_arm_target = None
        self._active_gripper_action = None
        self._active_navigation_target = None
        self.robot_pose = self._default_robot_pose()
        self.gripper_state = 'open'
        self.object_in_gripper = False
        self.grasped_object_name = ''
        self.grasped_object_mass = None
        self.runtime_safety_monitor.clear_events()
        if reset_config:
            self.robot_name = ''
            self.requested_room_name = ''

    def _setup_pipe_server(self):
        pipe_id = os.getenv('OMNISAFE_PIPE_ID', '')
        self.pipe_server = PipeCommunicationServer(
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            pipe_id=pipe_id,
        )
        self.pipe_server.start()

    def _resolve_robot_registration(self, robot_name: str) -> Dict[str, str]:
        normalized = str(robot_name or '').strip().lower()
        if not normalized:
            raise ValueError('robot_name is required.')

        registration = self.ROBOT_REGISTRY.get(normalized)
        if registration is None:
            supported = ', '.join(sorted(self.ROBOT_REGISTRY))
            raise ValueError(f'unsupported_robot:{normalized}; supported=[{supported}]')
        return dict(registration)

    def _load_robot_controller_class(self, robot_name: str):
        registration = self._resolve_robot_registration(robot_name)
        module_path = str(registration['module_path'])
        if not os.path.exists(module_path):
            raise FileNotFoundError(f'Fetch control module not found: {module_path}')

        module_name = f'fetch_robot_control_runtime_{abs(hash(module_path))}'
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Unable to load module spec from {module_path}')

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class_name = str(registration['class_name'])
        controller_cls = getattr(module, class_name, None)
        if controller_cls is None:
            raise AttributeError(f'Controller class "{class_name}" not found in {module_path}')

        self.control_module_path = module_path
        self.controller_class_name = class_name
        self.robot_name = str(robot_name).strip().lower()
        return controller_cls

    def _initialize_robot_runtime(self, robot_name: str, room_name: Optional[str] = None) -> tuple[bool, str]:
        if self.stage is None or self.my_world is None:
            return False, 'scene_not_loaded'

        try:
            controller_cls = self._load_robot_controller_class(robot_name)
            requested_room_name = str(room_name or '').strip()
            spawn_position, spawn_orientation, spawn_room_name = self._sample_robot_spawn_pose(
                requested_room_name or None
            )
            spawn_floor_z = float(spawn_position[2])
            self.robot_controller = controller_cls(
                world=self.my_world,
                initial_position=spawn_position,
                initial_orientation=spawn_orientation,
            )
            if not self._wait_for_stage_load(timeout_sec=60.0):
                print('WARNING: robot asset still loading before physics reset.')

            self.robot_prim_path = str(getattr(self.robot_controller, 'prim_path', '') or '')
            self.robot_articulation_path = str(
                getattr(self.robot_controller, 'robot_prim_path', '') or self.robot_prim_path
            )
            self.requested_room_name = requested_room_name
            self.spawn_room_name = spawn_room_name
            if self.enable_aggressive_lifecycle_collision_filtering:
                self._refresh_scene_index()
                pre_reset_collision_state = self._apply_aggressive_lifecycle_collision_filtering()
                self._aggressive_lifecycle_collision_filter_pre_reset_applied = bool(
                    pre_reset_collision_state.get('enabled')
                    and not pre_reset_collision_state.get('error')
                    and int(pre_reset_collision_state.get('physics_scene_count', 0) or 0) > 0
                    and int(pre_reset_collision_state.get('robot_group_target_count', 0) or 0) > 0
                    and int(pre_reset_collision_state.get('scene_object_group_target_count', 0) or 0) > 0
                )
            self.my_world.reset()
            self.robot_controller.initialize()
            self._apply_gripper_drive_overrides()
            self._settle_robot_after_spawn(ground_z=spawn_floor_z)

            self.robot_prim_path = str(getattr(self.robot_controller, 'prim_path', '') or '')
            self.robot_articulation_path = str(
                getattr(self.robot_controller, 'robot_prim_path', '') or self.robot_prim_path
            )
            self.requested_room_name = requested_room_name
            self.spawn_room_name = spawn_room_name
            raw_spawn_pose = self._get_robot_base_pose_raw()
            if raw_spawn_pose is not None:
                self.spawn_pose = self._pose_dict_from_raw(raw_spawn_pose[0], raw_spawn_pose[1])
            else:
                self.spawn_pose = self._pose_dict_from_raw(spawn_position, spawn_orientation)
            self.robot_pose = self._get_end_effector_pose() or self._default_robot_pose()
            self.gripper_state = 'open'
            self.object_in_gripper = False
            self.grasped_object_name = ''
            self.grasped_object_mass = None
            self._apply_gripper_high_friction_material()
            self._refresh_scene_index()
            self.current_room_name = self._detect_current_room_name()
            return True, ''
        except Exception as exc:
            print(f'ERROR: Failed to initialize Fetch runtime: {exc}')
            self._clear_robot_runtime(reset_config=False)
            return False, f'robot_init_failed:{exc}'

    def _rebuild_loaded_runtime(self) -> tuple[bool, str]:
        scene_path = str(self.usd_file_path or '').strip()
        robot_name = str(self.robot_name or '').strip()
        requested_room_name = str(self.requested_room_name or '').strip()

        if not scene_path:
            return False, 'scene_not_loaded'

        scene_ok = self._load_scene(scene_path)
        if not scene_ok:
            return False, 'scene_reload_failed'

        if robot_name:
            robot_ok, robot_error = self._initialize_robot_runtime(
                robot_name,
                requested_room_name or None,
            )
            if not robot_ok:
                return False, robot_error

        return True, ''

    @staticmethod
    def _bbox_to_dict(min_corner: np.ndarray, max_corner: np.ndarray) -> Dict[str, Dict[str, float]]:
        return {
            'min': {
                'x': float(min_corner[0]),
                'y': float(min_corner[1]),
                'z': float(min_corner[2]),
            },
            'max': {
                'x': float(max_corner[0]),
                'y': float(max_corner[1]),
                'z': float(max_corner[2]),
            },
        }

    def _compute_world_bbox(self, prim) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if prim is None or not prim.IsValid():
            return None
        try:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            )
            aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            min_corner = np.array(aligned_box.GetMin(), dtype=float)
            max_corner = np.array(aligned_box.GetMax(), dtype=float)
            if np.any(np.isnan(min_corner)) or np.any(np.isnan(max_corner)):
                return None
            return min_corner, max_corner
        except Exception:
            return None

    @staticmethod
    def _transform_point_to_world(transform_matrix, point) -> np.ndarray:
        point_vec = Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
        return np.array(transform_matrix.Transform(point_vec), dtype=float)

    def _get_prim_point_cloud_data(self, prim_path: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            'points': np.empty((0, 3), dtype=float),
            'mesh_count': 0,
            'mesh_prim_paths': [],
        }
        if not prim_path or self.stage is None:
            return result

        root_prim = self.stage.GetPrimAtPath(prim_path)
        if root_prim is None or not root_prim.IsValid():
            return result

        point_chunks: list[np.ndarray] = []
        mesh_prim_paths: list[str] = []
        for prim in Usd.PrimRange(root_prim):
            if prim is None or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
                continue

            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr()
            if not points_attr:
                continue

            local_points = points_attr.Get(Usd.TimeCode.Default())
            if local_points is None:
                local_points = points_attr.Get()
            points_array = np.array(local_points, dtype=float)
            if points_array.ndim != 2 or points_array.shape[0] == 0 or points_array.shape[1] != 3:
                continue

            valid_local = np.all(np.isfinite(points_array), axis=1)
            points_array = points_array[valid_local]
            if points_array.shape[0] == 0:
                continue

            try:
                world_matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                world_points = np.array(
                    [self._transform_point_to_world(world_matrix, point) for point in points_array],
                    dtype=float,
                )
            except Exception as exc:
                print(f'Failed to read mesh points for point cloud from {prim.GetPath()}: {exc}')
                continue

            valid_world = np.all(np.isfinite(world_points), axis=1)
            world_points = world_points[valid_world]
            if world_points.shape[0] == 0:
                continue

            point_chunks.append(world_points)
            mesh_prim_paths.append(str(prim.GetPath()))

        if point_chunks:
            result['points'] = np.concatenate(point_chunks, axis=0)
        result['mesh_count'] = len(mesh_prim_paths)
        result['mesh_prim_paths'] = mesh_prim_paths
        return result

    def _point_cloud_to_payload(self, point_cloud: np.ndarray, max_points: int) -> Dict[str, Any]:
        points = np.array(point_cloud, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            points = np.empty((0, 3), dtype=float)

        point_count = int(points.shape[0])
        max_points = max(int(max_points), 0)
        if max_points and point_count > max_points:
            sample_indices = np.linspace(0, point_count - 1, num=max_points, dtype=int)
            returned_points = points[sample_indices]
            is_sampled = True
        elif max_points == 0:
            returned_points = np.empty((0, 3), dtype=float)
            is_sampled = point_count > 0
        else:
            returned_points = points
            is_sampled = False

        bounds = {}
        if point_count > 0:
            bounds = self._bbox_to_dict(np.min(points, axis=0), np.max(points, axis=0))

        return {
            'point_count': point_count,
            'returned_point_count': int(returned_points.shape[0]),
            'is_sampled': is_sampled,
            'bounds': bounds,
            'points': returned_points.astype(float).tolist(),
        }

    @staticmethod
    def _build_top_down_grasp_orientation(*, yaw: float = 0.0) -> Dict[str, float]:
        return {
            'roll': 0.0,
            'pitch': float(np.pi / 2.0),
            'yaw': float(yaw),
        }

    @classmethod
    def _build_top_down_grasp_pose(
        cls,
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
            'orientation': cls._build_top_down_grasp_orientation(yaw=yaw),
        }

    def _build_manipulation_grasp_pose_candidates(
        self,
        *,
        grasp_position: np.ndarray,
        base_pose: Dict[str, Any] | None = None,
        robot_position: np.ndarray | None = None,
    ) -> list[Dict[str, Any]]:
        """Build ordered end-effector grasp pose candidates for one object point.

        Candidate 0 preserves the existing vertical top-down behavior. Candidate
        1 keeps the same position but tilts the gripper approach direction by
        DEFAULT_MANIPULATION_GRASP_TILT_DEGREES inside the vertical plane formed
        by the robot/base center and the object. For Fetch, the local +x axis is
        the approach direction, so this directly defines the gripper attitude.
        """

        position = np.array(grasp_position, dtype=float).reshape(-1)[:3]
        if position.size < 3 or np.any(~np.isfinite(position)):
            return []

        base_pose_dict = dict(base_pose or {})
        object_pose = {
            'position': {
                'x': float(position[0]),
                'y': float(position[1]),
                'z': float(position[2]),
            }
        }

        raw_top_down_pose = self._build_top_down_grasp_pose(
            object_pose=object_pose,
            base_pose=base_pose_dict,
        )
        _, top_down_orientation, top_down_pose = self._pose_command_to_raw(raw_top_down_pose)
        top_down_approach = self._approach_direction_from_orientation(top_down_orientation)

        candidates: list[Dict[str, Any]] = [
            {
                'rank': 0,
                'name': 'top_down',
                'pose': top_down_pose,
                'position': dict(top_down_pose.get('position') or {}),
                'orientation': dict(top_down_pose.get('orientation') or {}),
                'tilt_degrees_from_vertical': 0.0,
                'approach_direction': {
                    'x': float(top_down_approach[0]),
                    'y': float(top_down_approach[1]),
                    'z': float(top_down_approach[2]),
                },
                'description': 'Vertical top-down grasp at the object position.',
            }
        ]

        base_position = None
        base_position_dict = dict(base_pose_dict.get('position') or {})
        if base_position_dict:
            try:
                base_position = np.array(
                    [
                        float(base_position_dict.get('x', 0.0)),
                        float(base_position_dict.get('y', 0.0)),
                        float(base_position_dict.get('z', 0.0)),
                    ],
                    dtype=float,
                )
            except (TypeError, ValueError):
                base_position = None

        if base_position is None and robot_position is not None:
            base_position = np.array(robot_position, dtype=float).reshape(-1)[:3]

        horizontal = None
        if base_position is not None and base_position.size >= 3 and np.all(np.isfinite(base_position)):
            horizontal = np.array(
                [
                    float(position[0] - base_position[0]),
                    float(position[1] - base_position[1]),
                    0.0,
                ],
                dtype=float,
            )

        if horizontal is None or float(np.linalg.norm(horizontal)) <= 1e-9:
            base_orientation = dict(base_pose_dict.get('orientation') or {})
            yaw = float(base_orientation.get('yaw', 0.0) or 0.0)
            horizontal = np.array([float(np.cos(yaw)), float(np.sin(yaw)), 0.0], dtype=float)

        horizontal_direction = self._normalize_vector(
            horizontal,
            fallback=np.array([1.0, 0.0, 0.0], dtype=float),
        )
        tilt_rad = float(np.deg2rad(float(self.DEFAULT_MANIPULATION_GRASP_TILT_DEGREES)))
        tilted_approach = self._normalize_vector(
            horizontal_direction * float(np.sin(tilt_rad))
            + np.array([0.0, 0.0, -float(np.cos(tilt_rad))], dtype=float),
            fallback=np.array([0.0, 0.0, -1.0], dtype=float),
        )
        tilted_orientation = self._build_orientation_from_approach_direction(
            tilted_approach,
            up_hint=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        tilted_pose = self._pose_dict_from_raw(position, tilted_orientation)
        candidates.append(
            {
                'rank': 1,
                'name': 'tilted_45_from_robot',
                'pose': tilted_pose,
                'position': dict(tilted_pose.get('position') or {}),
                'orientation': dict(tilted_pose.get('orientation') or {}),
                'tilt_degrees_from_vertical': float(self.DEFAULT_MANIPULATION_GRASP_TILT_DEGREES),
                'approach_direction': {
                    'x': float(tilted_approach[0]),
                    'y': float(tilted_approach[1]),
                    'z': float(tilted_approach[2]),
                },
                'robot_to_object_direction': {
                    'x': float(horizontal_direction[0]),
                    'y': float(horizontal_direction[1]),
                    'z': float(horizontal_direction[2]),
                },
                'description': (
                    '45-degree tilted grasp at the object position; the approach '
                    'direction lies in the robot-object vertical plane.'
                ),
            }
        )

        return candidates

    @staticmethod
    def _normalize_object_cache_key(obj_name: str) -> str:
        return str(obj_name or '').strip().lower()

    @staticmethod
    def _read_env_int(name: str, default: int) -> int:
        raw = os.getenv(name, '')
        try:
            return int(raw) if str(raw).strip() else int(default)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _read_env_float(name: str, default: float) -> float:
        raw = os.getenv(name, '')
        try:
            return float(raw) if str(raw).strip() else float(default)
        except (TypeError, ValueError):
            return float(default)

    def _read_env_bool(self, name: str, default: bool = False) -> bool:
        raw = os.getenv(name, '')
        if not str(raw).strip():
            return bool(default)
        return self._coerce_bool(raw, default=default)

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
        quat = np.array(quaternion, dtype=float).reshape(-1)
        if quat.size != 4 or not np.all(np.isfinite(quat)):
            return np.eye(3, dtype=float)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-12:
            return np.eye(3, dtype=float)
        w, x, y, z = quat / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _get_world_to_robot_base_transform(self) -> Optional[np.ndarray]:
        raw_base_pose = self._get_robot_base_pose_raw()
        if raw_base_pose is None:
            return None
        position = np.array(raw_base_pose[0], dtype=float).reshape(-1)
        orientation = np.array(raw_base_pose[1], dtype=float).reshape(-1)
        if position.size < 3:
            return None
        rotation = self._quaternion_to_rotation_matrix(orientation)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation.T
        transform[:3, 3] = -rotation.T @ position[:3]
        return transform

    def _coerce_pose_payload(self, value: Any) -> Optional[Dict[str, Dict[str, float]]]:
        if not isinstance(value, dict):
            return None

        payload = dict(value or {})
        if isinstance(payload.get('pose'), dict):
            payload = dict(payload.get('pose') or {})

        position = payload.get('position')
        if not isinstance(position, dict):
            if isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 3:
                position = {
                    'x': position[0],
                    'y': position[1],
                    'z': position[2],
                }
            elif any(key in payload for key in ('x', 'y', 'z')):
                position = {
                    'x': payload.get('x', 0.0),
                    'y': payload.get('y', 0.0),
                    'z': payload.get('z', 0.0),
                }
            else:
                position = None

        orientation = payload.get('orientation')
        if not isinstance(orientation, dict):
            quaternion_wxyz = payload.get('quaternion_wxyz')
            if isinstance(quaternion_wxyz, dict):
                orientation = {
                    'w': quaternion_wxyz.get('w', quaternion_wxyz.get('qw', 1.0)),
                    'x': quaternion_wxyz.get('x', quaternion_wxyz.get('qx', 0.0)),
                    'y': quaternion_wxyz.get('y', quaternion_wxyz.get('qy', 0.0)),
                    'z': quaternion_wxyz.get('z', quaternion_wxyz.get('qz', 0.0)),
                }
            elif isinstance(quaternion_wxyz, (list, tuple, np.ndarray)) and len(quaternion_wxyz) >= 4:
                orientation = {
                    'w': quaternion_wxyz[0],
                    'x': quaternion_wxyz[1],
                    'y': quaternion_wxyz[2],
                    'z': quaternion_wxyz[3],
                }
            elif any(key in payload for key in ('roll', 'pitch', 'yaw')):
                orientation = {
                    'roll': payload.get('roll', 0.0),
                    'pitch': payload.get('pitch', 0.0),
                    'yaw': payload.get('yaw', 0.0),
                }
            elif {'w', 'x', 'y', 'z'}.issubset(set(payload.keys())):
                orientation = {
                    'w': payload.get('w', 1.0),
                    'x': payload.get('x', 0.0),
                    'y': payload.get('y', 0.0),
                    'z': payload.get('z', 0.0),
                }
            elif {'qw', 'qx', 'qy', 'qz'}.issubset(set(payload.keys())):
                orientation = {
                    'w': payload.get('qw', 1.0),
                    'x': payload.get('qx', 0.0),
                    'y': payload.get('qy', 0.0),
                    'z': payload.get('qz', 0.0),
                }
            else:
                orientation = None

        if not isinstance(position, dict) or not isinstance(orientation, dict):
            return None

        try:
            _, _, normalized_pose = self._pose_command_to_raw(
                {
                    'position': position,
                    'orientation': orientation,
                }
            )
        except Exception:
            return None
        return normalized_pose

    def _normalize_grasp_pose_response(self, response: Any) -> Optional[Dict[str, Dict[str, float]]]:
        if isinstance(response, (bytes, bytearray)):
            try:
                response = json.loads(bytes(response).decode('utf-8'))
            except Exception:
                return None
        elif isinstance(response, str):
            try:
                response = json.loads(response)
            except Exception:
                return None

        if not isinstance(response, dict):
            return None
        candidates: list[Any] = [
            response.get('target_pose'),
            response.get('gripper_base_pose'),
            response.get('grasp_pose'),
            response.get('pose'),
            response,
        ]
        for candidate in candidates:
            normalized_pose = self._coerce_pose_payload(candidate)
            if normalized_pose is not None:
                return normalized_pose
        return None

    def _request_grasp_pose_from_server(
        self,
        *,
        point_cloud: np.ndarray,
    ) -> Dict[str, Any]:
        host = str(os.getenv('GRASPGEN_SERVER_HOST', '') or '').strip()
        if not host:
            raise RuntimeError('grasp_server_host_not_configured')

        port = self._read_env_int('GRASPGEN_SERVER_PORT', 5556)
        timeout_ms = self._read_env_int('GRASPGEN_TIMEOUT_MS', 120000)
        num_grasps = self._read_env_int('GRASPGEN_NUM_GRASPS', 200)
        topk_num_grasps = self._read_env_int('GRASPGEN_TOPK_NUM_GRASPS', 20)
        grasp_threshold = self._read_env_float('GRASPGEN_GRASP_THRESHOLD', -1.0)
        min_grasps = self._read_env_int('GRASPGEN_MIN_GRASPS', 1)
        max_tries = self._read_env_int('GRASPGEN_MAX_TRIES', 6)
        keep_outliers = self._read_env_bool('GRASPGEN_KEEP_OUTLIERS', False)
        tcp_distance_threshold = self._read_env_float('GRASPGEN_TCP_DISTANCE_THRESHOLD', 0.03)
        world_to_robot_base = None
        if self.GRASP_SERVER_USE_WORLD_TO_ROBOT_BASE:
            world_to_robot_base = self._get_world_to_robot_base_transform()

        try:
            import msgpack
            import msgpack_numpy
            import zmq
        except ImportError as exc:
            raise RuntimeError(f'grasp_server_dependencies_unavailable:{exc}') from exc

        msgpack_numpy.patch()

        socket = None
        ctx = zmq.Context()
        try:
            socket = ctx.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f'tcp://{host}:{port}')

            payload = {
                'action': 'infer_target_pose',
                'point_cloud': np.ascontiguousarray(point_cloud, dtype=np.float32),
                'grasp_threshold': float(grasp_threshold),
                'num_grasps': int(num_grasps),
                'topk_num_grasps': int(topk_num_grasps),
                'min_grasps': int(min_grasps),
                'max_tries': int(max_tries),
                'remove_outliers': not bool(keep_outliers),
                'tcp_distance_threshold': None
                if float(tcp_distance_threshold) < 0.0
                else float(tcp_distance_threshold),
                'world_to_robot_base': world_to_robot_base,
                'degrees': False,
            }
            socket.send(msgpack.packb(payload, use_bin_type=True))
            raw_response = socket.recv()
            try:
                response = msgpack.unpackb(raw_response, raw=False)
            except Exception:
                response = json.loads(raw_response.decode('utf-8'))
        finally:
            if socket is not None:
                socket.close()
            ctx.term()

        if isinstance(response, dict) and response.get('error'):
            raise RuntimeError(f"grasp_server_error:{response.get('error')}")
        if not isinstance(response, dict):
            raise RuntimeError('grasp_server_invalid_response')
        return response

    def _build_fallback_grasp_pose(
        self,
        *,
        object_pose: Dict[str, Any] | None,
        fallback_position: np.ndarray | None = None,
    ) -> Dict[str, Dict[str, float]]:
        raw_base_pose = self._get_robot_base_pose_raw()
        base_pose = None
        if raw_base_pose is not None:
            base_pose = self._pose_dict_from_raw(raw_base_pose[0], raw_base_pose[1])

        fallback_object_pose = dict(object_pose or {})
        if fallback_position is not None:
            fallback_object_pose = dict(fallback_object_pose)
            fallback_object_pose['position'] = {
                'x': float(fallback_position[0]),
                'y': float(fallback_position[1]),
                'z': float(fallback_position[2]),
            }

        return self._build_top_down_grasp_pose(
            object_pose=fallback_object_pose,
            base_pose=base_pose,
        )

    def _resolve_preferred_grasp_pose(
        self,
        obj_name: str,
        *,
        prim_path: str = '',
        object_pose: Optional[Dict[str, Any]] = None,
        fallback_position: Optional[np.ndarray] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        normalized_name = str(obj_name or '').strip()
        if not normalized_name:
            raise ValueError('obj_name_required')

        cached_entry = self._get_cached_preferred_grasp_pose(
            normalized_name,
            prim_path=prim_path,
        )
        if cached_entry and not force_refresh:
            return cached_entry

        cache_key = self._normalize_object_cache_key(normalized_name)
        requested_prim_path = str(prim_path or '').strip()

        resolved_prim_path = requested_prim_path or self._lookup_object_prim_path(normalized_name)
        if not resolved_prim_path:
            raise RuntimeError(f'grasp_target_not_found:{normalized_name}')

        resolved_object_pose = dict(object_pose or self._get_pose_for_prim_path(resolved_prim_path) or {})
        fallback_pose = self._build_fallback_grasp_pose(
            object_pose=resolved_object_pose,
            fallback_position=fallback_position,
        )

        server_enabled = self._read_env_bool('GRASPGEN_SERVER_ENABLED', False)

        diagnostics: Dict[str, Any] = {
            'obj_name': normalized_name,
            'prim_path': resolved_prim_path,
            'uses_world_to_robot_base_transform': bool(self.GRASP_SERVER_USE_WORLD_TO_ROBOT_BASE),
            'grasp_server_enabled': bool(server_enabled),
        }
        grasp_pose = dict(fallback_pose)
        grasp_pose_source = 'top_down_fallback'

        if not server_enabled:
            # Object-position mode: skip the ZMQ server entirely and reuse the
            # top-down fallback that is centred on the object's position.
            diagnostics['point_cloud_point_count'] = 0
            diagnostics['fallback_reason'] = 'grasp_server_disabled'
            grasp_pose_source = 'object_position'
        else:
            point_cloud_data = self._get_prim_point_cloud_data(resolved_prim_path)
            point_cloud = np.array(point_cloud_data.get('points'), dtype=float)
            diagnostics['point_cloud_point_count'] = (
                int(point_cloud.shape[0]) if point_cloud.ndim == 2 else 0
            )

            if point_cloud.ndim == 2 and point_cloud.shape[0] > 0 and point_cloud.shape[1] == 3:
                try:
                    server_response = self._request_grasp_pose_from_server(point_cloud=point_cloud)
                    normalized_pose = self._normalize_grasp_pose_response(server_response)
                    if normalized_pose is None:
                        diagnostics['fallback_reason'] = 'grasp_server_pose_invalid'
                    else:
                        grasp_pose = normalized_pose
                        grasp_pose_source = 'zmq'
                        diagnostics['server_response_keys'] = sorted(str(key) for key in server_response.keys())
                except Exception as exc:
                    diagnostics['fallback_reason'] = str(exc)
            else:
                diagnostics['fallback_reason'] = 'point_cloud_unavailable'
                diagnostics['mesh_count'] = int(point_cloud_data.get('mesh_count', 0) or 0)
                diagnostics['mesh_prim_paths'] = list(point_cloud_data.get('mesh_prim_paths') or [])

        entry = {
            'obj_name': normalized_name,
            'prim_path': resolved_prim_path,
            'object_pose': resolved_object_pose,
            'grasp_pose': grasp_pose,
            'grasp_pose_source': grasp_pose_source,
            'diagnostics': diagnostics,
        }
        self._preferred_grasp_cache[cache_key] = entry
        return dict(entry)

    def _get_cached_preferred_grasp_pose(
        self,
        obj_name: str,
        *,
        prim_path: str = '',
    ) -> Dict[str, Any]:
        cache_key = self._normalize_object_cache_key(obj_name)
        cached_entry = dict(self._preferred_grasp_cache.get(cache_key) or {})
        if not cached_entry:
            return {}

        requested_prim_path = str(prim_path or '').strip()
        cached_prim_path = str(cached_entry.get('prim_path', '') or '').strip()
        if requested_prim_path and cached_prim_path and requested_prim_path != cached_prim_path:
            return {}
        return cached_entry

    def _align_cached_fallback_grasp_pose_to_base(
        self,
        obj_name: str,
        *,
        base_pose: Dict[str, Any],
    ) -> Dict[str, Any]:
        cache_key = self._normalize_object_cache_key(obj_name)
        cached_entry = dict(self._preferred_grasp_cache.get(cache_key) or {})
        if not cached_entry or str(cached_entry.get('grasp_pose_source', '') or '') != 'top_down_fallback':
            return cached_entry

        object_pose = dict(cached_entry.get('object_pose') or {})
        if not object_pose:
            object_pose = {'position': dict((cached_entry.get('grasp_pose') or {}).get('position') or {})}

        cached_entry['grasp_pose'] = self._build_top_down_grasp_pose(
            object_pose=object_pose,
            base_pose=base_pose,
        )
        diagnostics = dict(cached_entry.get('diagnostics') or {})
        diagnostics['aligned_to_suggested_base_pose'] = True
        cached_entry['diagnostics'] = diagnostics
        self._preferred_grasp_cache[cache_key] = cached_entry
        return dict(cached_entry)

    @staticmethod
    def _safe_filename(value: str) -> str:
        safe_chars = []
        for char in str(value or '').strip():
            if char.isalnum() or char in {'-', '_', '.'}:
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        return ''.join(safe_chars).strip('._') or 'object'

    @staticmethod
    def _save_point_cloud_as_ply(points: np.ndarray, output_path: Path) -> None:
        points = np.array(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('points must have shape Nx3')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='ascii') as file:
            file.write('ply\n')
            file.write('format ascii 1.0\n')
            file.write(f'element vertex {points.shape[0]}\n')
            file.write('property float x\n')
            file.write('property float y\n')
            file.write('property float z\n')
            file.write('end_header\n')
            for x, y, z in points:
                file.write(f'{float(x)} {float(y)} {float(z)}\n')

    def _refresh_room_index(self) -> None:
        self.room_index = {}
        if self.stage is None:
            return

        rooms_root = self.stage.GetPrimAtPath(ROOMS_ROOT_PATH)
        if not rooms_root or not rooms_root.IsValid():
            return

        for room_prim in rooms_root.GetChildren():
            if room_prim is None or not room_prim.IsValid():
                continue

            room_name = room_prim.GetName()
            if self._should_ignore_room_prim(room_name):
                continue
            room_path = str(room_prim.GetPath())
            room_entry: Dict[str, Any] = {
                'room_name': room_name,
                'prim_path': room_path,
                'objects': [],
                'object_count': 0,
            }

            room_bbox = self._compute_world_bbox(room_prim)
            if room_bbox is not None:
                room_entry['bounds'] = self._bbox_to_dict(room_bbox[0], room_bbox[1])
                room_entry['regions'] = self._default_room_regions(room_entry['bounds'])
            room_metadata = self._entity_metadata_from_prim(room_prim)
            if isinstance(room_metadata.get('regions'), dict):
                room_entry.setdefault('regions', {}).update(
                    dict(room_metadata.get('regions') or {})
                )

            for child in room_prim.GetChildren():
                if child is None or not child.IsValid():
                    continue

                object_name = child.GetName()
                object_entry: Dict[str, Any] = {
                    'name': object_name,
                    'prim_path': str(child.GetPath()),
                }
                object_metadata = self._entity_metadata_from_prim(child)
                for field_name in ('category', 'capabilities', 'materials', 'parts', 'regions'):
                    if object_metadata.get(field_name) not in (None, '', [], {}):
                        object_entry[field_name] = object_metadata[field_name]
                object_bbox = self._compute_world_bbox(child)
                if object_bbox is not None:
                    object_entry['bounds'] = self._bbox_to_dict(object_bbox[0], object_bbox[1])
                pose = self._get_pose_for_prim_path(str(child.GetPath()))
                if pose is not None:
                    object_entry['pose'] = pose
                    object_entry['position'] = dict(pose.get('position') or {})
                    object_entry['orientation'] = dict(pose.get('orientation') or {})
                room_entry['objects'].append(object_entry)

            room_entry['object_count'] = len(room_entry['objects'])
            self.room_index[room_name] = room_entry

    @staticmethod
    def _should_ignore_room_prim(room_name: Any) -> bool:
        return str(room_name or '').strip().lower() in {'_whole_house'}

    @staticmethod
    def _default_room_regions(bounds: Dict[str, Any]) -> Dict[str, Any]:
        minimum = dict(bounds.get('min') or {})
        maximum = dict(bounds.get('max') or {})
        try:
            min_x = float(minimum['x'])
            min_y = float(minimum['y'])
            min_z = float(minimum.get('z', 0.0))
            max_x = float(maximum['x'])
            max_y = float(maximum['y'])
            max_z = float(maximum.get('z', min_z + 0.1))
        except (KeyError, TypeError, ValueError):
            return {}
        width = max(0.001, max_x - min_x)
        depth = max(0.001, max_y - min_y)
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        center_fraction = 0.4
        edge_band = min(0.6, max(0.2, min(width, depth) * 0.15))
        corridor = min(0.8, max(0.3, min(width, depth) * 0.25))

        def region(x0: float, y0: float, x1: float, y1: float) -> Dict[str, Any]:
            return {
                'bounds': {
                    'min': {'x': x0, 'y': y0, 'z': min_z},
                    'max': {'x': x1, 'y': y1, 'z': max_z},
                },
                'source': 'generated_aabb_region',
            }

        regions = {
            'center': region(
                center_x - width * center_fraction / 2.0,
                center_y - depth * center_fraction / 2.0,
                center_x + width * center_fraction / 2.0,
                center_y + depth * center_fraction / 2.0,
            ),
            'entrance': region(min_x, center_y - edge_band / 2.0, min_x + edge_band, center_y + edge_band / 2.0),
            'exit': region(max_x - edge_band, center_y - edge_band / 2.0, max_x, center_y + edge_band / 2.0),
        }
        if width >= depth:
            regions['centerline'] = region(min_x, center_y - corridor / 2.0, max_x, center_y + corridor / 2.0)
        else:
            regions['centerline'] = region(center_x - corridor / 2.0, min_y, center_x + corridor / 2.0, max_y)
        regions['walkable_centerline'] = dict(regions['centerline'])
        return regions

    @staticmethod
    def _position_in_bounds(position: np.ndarray, bounds: Dict[str, Any], margin: float = 0.05) -> bool:
        min_corner = bounds.get('min') or {}
        max_corner = bounds.get('max') or {}
        try:
            return (
                float(min_corner.get('x', float('-inf'))) - margin <= float(position[0]) <= float(max_corner.get('x', float('inf'))) + margin
                and float(min_corner.get('y', float('-inf'))) - margin <= float(position[1]) <= float(max_corner.get('y', float('inf'))) + margin
                and float(min_corner.get('z', float('-inf'))) - margin <= float(position[2]) <= float(max_corner.get('z', float('inf'))) + margin
            )
        except (TypeError, ValueError, IndexError):
            return False

    @staticmethod
    def _normalize_point_2d(point: Any) -> tuple[float, float]:
        if isinstance(point, dict):
            return float(point.get('x', 0.0)), float(point.get('y', 0.0))
        if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
            return float(point[0]), float(point[1])
        raise ValueError(f'Invalid 2D point: {point}')

    @staticmethod
    def _point_on_segment_2d(
        point: tuple[float, float],
        seg_start: tuple[float, float],
        seg_end: tuple[float, float],
        *,
        eps: float = 1e-6,
    ) -> bool:
        px, py = point
        x1, y1 = seg_start
        x2, y2 = seg_end

        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) > eps:
            return False

        min_x = min(x1, x2) - eps
        max_x = max(x1, x2) + eps
        min_y = min(y1, y2) - eps
        max_y = max(y1, y2) + eps
        return min_x <= px <= max_x and min_y <= py <= max_y

    def _load_room_boundary_segments(
        self,
        room_name: str,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        if self._should_ignore_room_prim(room_name):
            return []
        if self.stage is None:
            return []

        room_prim = self.stage.GetPrimAtPath(f'{ROOMS_ROOT_PATH}/{room_name}')
        if room_prim is None or not room_prim.IsValid():
            print(f'Failed to find room prim for polygon lookup: {room_name}')
            return []

        serialized_segments = room_prim.GetCustomDataByKey('room_boundary_segments')
        if not serialized_segments:
            print(f'No room boundary metadata found on room: {room_name}')
            return []

        try:
            segments = json.loads(serialized_segments) if isinstance(serialized_segments, str) else serialized_segments
        except Exception as exc:
            print(f'Failed to parse room boundary metadata for room {room_name}: {exc}')
            return []

        normalized_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for start, end in segments:
            try:
                p1 = self._normalize_point_2d(start)
                p2 = self._normalize_point_2d(end)
            except Exception:
                continue
            if p1 != p2:
                normalized_segments.append((p1, p2))
        return normalized_segments

    def _point_in_room_by_segments(
        self,
        point: tuple[float, float],
        segments: list[tuple[tuple[float, float], tuple[float, float]]],
        *,
        eps: float = 1e-6,
    ) -> bool:
        if not segments:
            return False

        for seg_start, seg_end in segments:
            if self._point_on_segment_2d(point, seg_start, seg_end, eps=eps):
                return True

        x, y = point
        intersections = 0
        for (x1, y1), (x2, y2) in segments:
            if abs(y1 - y2) <= eps:
                if abs(y - y1) <= eps and min(x1, x2) - eps <= x <= max(x1, x2) + eps:
                    return True
                continue

            y_min = min(y1, y2)
            y_max = max(y1, y2)
            if y < y_min or y >= y_max:
                continue

            x_intersection = x1
            if x_intersection >= x - eps:
                intersections += 1

        return (intersections % 2) == 1

    def _detect_current_room_name(self) -> str:
        # # Mock this name currently, I will replace the correct code with it later
        # return "kitchen"
        raw_base_pose = self._get_robot_base_pose_raw()
        if raw_base_pose is None:
            return str(self.spawn_room_name or '')

        position = np.array(raw_base_pose[0], dtype=float)
        point_2d = (float(position[0]), float(position[1]))
        containing_rooms: list[str] = []
        for room_name in self.room_index.keys():
            segments = self._load_room_boundary_segments(str(room_name))
            if self._point_in_room_by_segments(point_2d, segments):
                containing_rooms.append(str(room_name))

        if containing_rooms:
            if self.spawn_room_name and self.spawn_room_name in containing_rooms:
                return str(self.spawn_room_name)
            return sorted(containing_rooms)[0]

        return str(self.spawn_room_name or '')

    @staticmethod
    def _should_exclude_room_object(object_name: str) -> bool:
        lowered = str(object_name or '').strip().lower()
        return any(token in lowered for token in ('wall', 'door', 'floor'))

    def _get_current_room_object_names(self) -> list[str]:
        current_room_name = self._detect_current_room_name()
        self.current_room_name = current_room_name
        if self.stage is None or not current_room_name:
            return []

        room_prim = self.stage.GetPrimAtPath(f'{ROOMS_ROOT_PATH}/{current_room_name}')
        if room_prim is None or not room_prim.IsValid():
            return []

        visible_names: list[str] = []
        for child in room_prim.GetChildren():
            if child is None or not child.IsValid():
                continue

            object_name = str(child.GetName() or '').strip()
            if not object_name or self._should_exclude_room_object(object_name):
                continue

            visible_names.append(object_name)

        return sorted(set(visible_names))

    def _resolve_room_entry(self, room_name: str) -> Optional[Dict[str, Any]]:
        requested = str(room_name or '').strip()
        if not requested:
            return None

        lowered = requested.lower()
        for entry in self.room_index.values():
            if not isinstance(entry, dict):
                continue
            candidate = str(entry.get('room_name', '') or '').strip()
            if candidate.lower() == lowered:
                return entry
        return None

    def _sample_robot_spawn_pose(
        self,
        room_name: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        self._refresh_room_index()
        rooms = [entry for entry in self.room_index.values() if isinstance(entry, dict)]
        if not rooms:
            return (
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array(euler_angles_to_quats(np.array([0.0, 0.0, 0.0], dtype=float)), dtype=float),
                '',
            )

        requested_room_name = str(room_name or '').strip()
        if requested_room_name:
            room_entry = self._resolve_room_entry(requested_room_name)
            if room_entry is None:
                available = ', '.join(sorted(self.room_index)) if self.room_index else ''
                raise ValueError(
                    f'room_not_found:{requested_room_name}; available=[{available}]'
                )
        else:
            room_entry = rooms[int(self._rng.integers(0, len(rooms)))]
        room_name = str(room_entry.get('room_name', '') or '')
        bounds = room_entry.get('bounds') or {}
        min_corner = np.array(
            [
                float((bounds.get('min') or {}).get('x', -1.0)),
                float((bounds.get('min') or {}).get('y', -1.0)),
                float((bounds.get('min') or {}).get('z', 0.0)),
            ],
            dtype=float,
        )
        max_corner = np.array(
            [
                float((bounds.get('max') or {}).get('x', 1.0)),
                float((bounds.get('max') or {}).get('y', 1.0)),
                float((bounds.get('max') or {}).get('z', 0.0)),
            ],
            dtype=float,
        )
        room_area = max((max_corner[0] - min_corner[0]) * (max_corner[1] - min_corner[1]), 1e-6)
        floor_z = self._estimate_room_floor_height(
            room_entry,
            fallback_floor_z=float(max(min_corner[2], 0.0)),
        )
        obstacles: list[tuple[np.ndarray, np.ndarray]] = []

        for obj in room_entry.get('objects') or []:
            name = str(obj.get('name', '') or '').lower()
            if any(token in name for token in ('floor', 'ground', 'ceiling', 'light', 'camera', 'window', 'room')):
                continue

            obj_bounds = obj.get('bounds') or {}
            obj_min = np.array(
                [
                    float((obj_bounds.get('min') or {}).get('x', 0.0)),
                    float((obj_bounds.get('min') or {}).get('y', 0.0)),
                    float((obj_bounds.get('min') or {}).get('z', 0.0)),
                ],
                dtype=float,
            )
            obj_max = np.array(
                [
                    float((obj_bounds.get('max') or {}).get('x', 0.0)),
                    float((obj_bounds.get('max') or {}).get('y', 0.0)),
                    float((obj_bounds.get('max') or {}).get('z', 0.0)),
                ],
                dtype=float,
            )
            footprint_area = max(obj_max[0] - obj_min[0], 0.0) * max(obj_max[1] - obj_min[1], 0.0)
            height = max(obj_max[2] - obj_min[2], 0.0)
            if footprint_area >= room_area * 0.85 and 'wall' not in name and 'door' not in name:
                continue
            if height <= 0.05 and obj_max[2] <= floor_z + 0.1:
                continue
            obstacles.append((obj_min, obj_max))

        clearance = float(self.DEFAULT_ROBOT_SPAWN_CLEARANCE_RADIUS)
        margin = float(self.DEFAULT_ROBOT_SPAWN_WALL_MARGIN)
        x_low = min_corner[0] + margin
        x_high = max_corner[0] - margin
        y_low = min_corner[1] + margin
        y_high = max_corner[1] - margin

        if x_low >= x_high or y_low >= y_high:
            x_low, x_high = min_corner[0], max_corner[0]
            y_low, y_high = min_corner[1], max_corner[1]

        chosen_position = np.array(
            [
                float((min_corner[0] + max_corner[0]) / 2.0),
                float((min_corner[1] + max_corner[1]) / 2.0),
                float(floor_z),
            ],
            dtype=float,
        )

        chosen_xy = self._find_open_spawn_xy(
            room_name=room_name,
            room_bounds=(min_corner, max_corner),
            obstacles=obstacles,
            clearance=clearance,
            margin=margin,
        )
        if chosen_xy is not None:
            chosen_position = np.array(
                [float(chosen_xy[0]), float(chosen_xy[1]), float(floor_z)],
                dtype=float,
            )

        if chosen_xy is None:
            best_candidate_xy: Optional[np.ndarray] = None
            best_candidate_score = float('-inf')
            for _ in range(self.DEFAULT_SPAWN_ATTEMPTS):
                candidate_xy = np.array(
                    [
                        float(self._rng.uniform(x_low, x_high)),
                        float(self._rng.uniform(y_low, y_high)),
                    ],
                    dtype=float,
                )

                collision = False
                nearest_clearance = float('inf')
                for obj_min, obj_max in obstacles:
                    nearest_x = float(np.clip(candidate_xy[0], obj_min[0], obj_max[0]))
                    nearest_y = float(np.clip(candidate_xy[1], obj_min[1], obj_max[1]))
                    distance = float(
                        np.linalg.norm(candidate_xy - np.array([nearest_x, nearest_y], dtype=float))
                    )
                    nearest_clearance = min(nearest_clearance, distance)
                    if distance < clearance:
                        collision = True
                        break

                if collision:
                    continue

                wall_clearance = min(
                    float(candidate_xy[0] - min_corner[0]),
                    float(max_corner[0] - candidate_xy[0]),
                    float(candidate_xy[1] - min_corner[1]),
                    float(max_corner[1] - candidate_xy[1]),
                )
                candidate_score = min(nearest_clearance, wall_clearance)
                if candidate_score > best_candidate_score:
                    best_candidate_score = candidate_score
                    best_candidate_xy = candidate_xy

            if best_candidate_xy is not None and best_candidate_score > clearance:
                chosen_position = np.array(
                    [float(best_candidate_xy[0]), float(best_candidate_xy[1]), float(floor_z)],
                    dtype=float,
                )

        yaw = float(self._rng.uniform(-np.pi, np.pi))
        orientation = np.array(
            euler_angles_to_quats(np.array([0.0, 0.0, yaw], dtype=float)),
            dtype=float,
        )
        return chosen_position, orientation, room_name

    def _estimate_room_floor_height(
        self,
        room_entry: Dict[str, Any],
        *,
        fallback_floor_z: float,
    ) -> float:
        floor_z = float(fallback_floor_z)
        room_bounds = self._bbox_dict_to_arrays(room_entry.get('bounds') or {})
        room_area = 0.0
        if room_bounds is not None:
            room_min, room_max = room_bounds
            room_area = max(float((room_max[0] - room_min[0]) * (room_max[1] - room_min[1])), 0.0)

        for obj in room_entry.get('objects') or []:
            name = str(obj.get('name', '') or '').strip().lower()
            if 'floor' not in name and 'ground' not in name:
                continue

            bounds = self._bbox_dict_to_arrays(obj.get('bounds') or {})
            if bounds is None:
                continue

            obj_min, obj_max = bounds
            footprint_area = max(float(obj_max[0] - obj_min[0]), 0.0) * max(
                float(obj_max[1] - obj_min[1]),
                0.0,
            )
            if room_area > 0.0 and footprint_area < room_area * 0.05:
                continue
            floor_z = max(floor_z, float(obj_max[2]))
        return floor_z

    def _find_open_spawn_xy(
        self,
        *,
        room_name: str,
        room_bounds: tuple[np.ndarray, np.ndarray],
        obstacles: list[tuple[np.ndarray, np.ndarray]],
        clearance: float,
        margin: float,
    ) -> Optional[np.ndarray]:
        try:
            geometry = ensure_shapely()
        except RuntimeError as exc:
            print(f'Spawn free-space sampling unavailable, falling back to random sampling: {exc}')
            return None

        min_corner, max_corner = room_bounds
        room_polygon = geometry['box'](
            float(min_corner[0]),
            float(min_corner[1]),
            float(max_corner[0]),
            float(max_corner[1]),
        )

        boundary_segments = self._load_room_boundary_segments(room_name)
        if boundary_segments:
            lines = [
                geometry['LineString']([start, end])
                for start, end in boundary_segments
                if np.linalg.norm(np.array(end, dtype=float) - np.array(start, dtype=float)) > 1e-9
            ]
            polygons = list(geometry['polygonize'](lines))
            if polygons:
                room_polygon = max(polygons, key=lambda poly: float(poly.area)).buffer(0)

        if room_polygon.is_empty:
            return None

        interior_polygon = room_polygon.buffer(-float(margin))
        if interior_polygon.is_empty or float(getattr(interior_polygon, 'area', 0.0)) <= 1e-6:
            interior_polygon = room_polygon

        obstacle_polygons = [
            geometry['box'](
                float(obj_min[0]),
                float(obj_min[1]),
                float(obj_max[0]),
                float(obj_max[1]),
            ).buffer(float(clearance))
            for obj_min, obj_max in obstacles
        ]
        if obstacle_polygons:
            obstacle_union = geometry['unary_union'](obstacle_polygons).buffer(0)
        else:
            obstacle_union = geometry['Polygon']()

        free_space = interior_polygon.difference(obstacle_union).buffer(0)
        if free_space.is_empty or float(getattr(free_space, 'area', 0.0)) <= 1e-6:
            return None

        free_space_boundary = free_space.boundary
        step = float(
            max(
                0.16,
                min(
                    float(self.DEFAULT_ROBOT_SPAWN_GRID_RESOLUTION),
                    max(float(clearance) * 0.6, 0.16),
                ),
            )
        )
        min_x, min_y, max_x, max_y = free_space.bounds

        best_xy: Optional[np.ndarray] = None
        best_score = float('-inf')

        representative_point = free_space.representative_point()
        if free_space.covers(representative_point):
            representative_score = float(representative_point.distance(free_space_boundary))
            best_xy = np.array([float(representative_point.x), float(representative_point.y)], dtype=float)
            best_score = representative_score

        x_values = np.arange(float(min_x), float(max_x) + step * 0.5, step)
        y_values = np.arange(float(min_y), float(max_y) + step * 0.5, step)
        for x in x_values:
            for y in y_values:
                point = geometry['Point'](float(x), float(y))
                if not free_space.covers(point):
                    continue
                score = float(point.distance(free_space_boundary))
                if score > best_score:
                    best_xy = np.array([float(x), float(y)], dtype=float)
                    best_score = score

        random_attempts = max(self.DEFAULT_SPAWN_ATTEMPTS * 2, 120)
        for _ in range(random_attempts):
            point = geometry['Point'](
                float(self._rng.uniform(min_x, max_x)),
                float(self._rng.uniform(min_y, max_y)),
            )
            if not free_space.covers(point):
                continue
            score = float(point.distance(free_space_boundary))
            if score > best_score:
                best_xy = np.array([float(point.x), float(point.y)], dtype=float)
                best_score = score

        return best_xy

    def _settle_robot_after_spawn(self, *, ground_z: float) -> None:
        if self.robot_controller is None or self.my_world is None:
            return

        try:
            align_to_ground = getattr(self.robot_controller, 'align_root_to_ground', None)
            if callable(align_to_ground):
                align_to_ground(ground_z=float(ground_z))
        except Exception as exc:
            print(f'Robot ground alignment after spawn failed: {exc}')

        for _ in range(int(self.DEFAULT_ROBOT_SPAWN_SETTLE_STEPS)):
            try:
                self.robot_controller.stop_base()
            except Exception:
                pass
            self.my_world.step(render=False)

    def _aggressive_lifecycle_collision_group_root_paths(self) -> tuple[str, ...]:
        roots = [
            str(self.AGGRESSIVE_LIFECYCLE_COLLISION_GROUP_ROOT_PATH or '').strip(),
            *[
                str(path or '').strip()
                for path in self.AGGRESSIVE_LIFECYCLE_LEGACY_COLLISION_GROUP_ROOT_PATHS
            ],
        ]
        return tuple(path for path in dict.fromkeys(roots) if path)

    def _is_aggressive_lifecycle_collision_group_path(self, prim_path: str) -> bool:
        normalized_path = str(prim_path or '').strip()
        if not normalized_path:
            return False
        return any(
            normalized_path == root_path or normalized_path.startswith(f'{root_path}/')
            for root_path in self._aggressive_lifecycle_collision_group_root_paths()
        )

    def _refresh_scene_index(self):
        self.object_prim_paths = {}
        self.object_poses = {}
        self.object_bounds = {}
        self._refresh_room_index()
        if self.stage is None:
            return

        for room_entry in self.room_index.values():
            if not isinstance(room_entry, dict):
                continue
            for obj in room_entry.get('objects') or []:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get('name') or obj.get('object_name') or '').strip()
                prim_path = str(obj.get('prim_path') or '').strip()
                if not name or room_direct_child_identity(prim_path) is None:
                    continue
                if prim_path:
                    self.object_prim_paths.setdefault(name, prim_path)
                if isinstance(obj.get('pose'), dict):
                    self.object_poses[name] = dict(obj.get('pose') or {})
                if isinstance(obj.get('bounds'), dict):
                    self.object_bounds[name] = {
                        'name': name,
                        'prim_path': prim_path,
                        'bounds': dict(obj.get('bounds') or {}),
                    }

        for name, prim_path in self.object_prim_paths.items():
            pose = self._get_pose_for_prim_path(prim_path)
            if pose is not None:
                self.object_poses[name] = pose
            prim = self.stage.GetPrimAtPath(prim_path) if self.stage is not None else None
            bbox = self._compute_world_bbox(prim)
            if bbox is not None:
                self.object_bounds[name] = {
                    'name': name,
                    'prim_path': prim_path,
                    'bounds': self._bbox_to_dict(bbox[0], bbox[1]),
                }
        self._apply_aggressive_lifecycle_collision_filtering()

    def _initialize_world_state(self) -> None:
        self.world_state.clear()
        self._world_state_wall_start = time.monotonic()
        if self.stage is None:
            return
        pending_relations: list[tuple[str, Dict[str, Any]]] = []
        for name, prim_path in self.object_prim_paths.items():
            prim = self.stage.GetPrimAtPath(prim_path)
            metadata = self._entity_metadata_from_prim(prim)
            room_entry, _object_entry = self._find_room_object_entry(name)
            if room_entry is not None:
                room_name = str(
                    room_entry.get('room_name')
                    or room_entry.get('name')
                    or ''
                ).strip()
                if room_name:
                    metadata.setdefault('room', room_name)
                    metadata.setdefault('zone', room_name)
            try:
                self.world_state.register_entity(
                    name,
                    metadata,
                    infer_profile=True,
                )
            except WorldStateError:
                continue
            for relation in metadata.get('relations') or []:
                if isinstance(relation, dict):
                    pending_relations.append((name, dict(relation)))
        for room_name in self.room_index:
            try:
                self.world_state.register_entity(
                    str(room_name),
                    {
                        'category': 'room',
                        'capabilities': ['environment_zone'],
                        'state_defaults': {
                            'temperature_c': 22.0,
                            'humidity_fraction': 0.4,
                            'steam_level': 'normal',
                        },
                    },
                    infer_profile=False,
                )
            except WorldStateError:
                continue
        for default_source, relation in pending_relations:
            source = str(relation.get('source') or default_source).strip()
            target = str(relation.get('target') or '').strip()
            relation_name = str(relation.get('relation') or relation.get('type') or '').strip()
            if not source or not target or not relation_name:
                continue
            try:
                self.world_state.set_relation(
                    source,
                    target,
                    relation_name,
                    metadata=dict(relation.get('metadata') or {}),
                )
            except WorldStateError:
                continue

    def _entity_metadata_from_prim(self, prim: Any) -> Dict[str, Any]:
        if prim is None or not prim.IsValid():
            return {}
        metadata: Dict[str, Any] = {}
        raw_json = self._prim_attribute_value(prim, 'omnisafe:metadata')
        if raw_json is None:
            raw_json = self._prim_attribute_value(prim, 'directlayout:rawJson')
        if isinstance(raw_json, str) and raw_json.strip():
            try:
                parsed = json.loads(raw_json)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                nested = parsed.get('omnisafe') or parsed.get('simulation') or parsed
                if isinstance(nested, dict):
                    metadata.update(dict(nested))

        category = (
            self._prim_attribute_value(prim, 'omnisafe:category')
            or self._prim_attribute_value(prim, 'directlayout:category')
            or metadata.get('category')
        )
        if category:
            metadata['category'] = str(category)

        for field_name, attribute_names in {
            'capabilities': ('omnisafe:capabilities', 'directlayout:capabilities'),
            'materials': ('omnisafe:materials',),
            'parts': ('omnisafe:parts',),
            'regions': ('omnisafe:regions',),
            'relations': ('omnisafe:relations',),
            'state_defaults': ('omnisafe:stateDefaults', 'omnisafe:state_defaults'),
        }.items():
            value = None
            for attribute_name in attribute_names:
                value = self._prim_attribute_value(prim, attribute_name)
                if value is not None:
                    break
            parsed_value = self._parse_metadata_value(value)
            if parsed_value not in (None, '', [], {}):
                metadata[field_name] = parsed_value

        custom_data = prim.GetCustomData() or {}
        omnisafe_custom = custom_data.get('omnisafe')
        if isinstance(omnisafe_custom, dict):
            metadata.update(dict(omnisafe_custom))
        for key, value in custom_data.items():
            key_text = str(key or '')
            if not key_text.startswith('omnisafe_'):
                continue
            metadata[key_text[len('omnisafe_'):]] = value
        return metadata

    @staticmethod
    def _prim_attribute_value(prim: Any, name: str) -> Any:
        try:
            attribute = prim.GetAttribute(name)
            if attribute is None or not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
                return None
            return attribute.Get()
        except Exception:
            return None

    def _aabb_center(self, prim_path: str, bbox_cache: Any) -> Optional[list[float]]:
        bounds = self._aabb(prim_path, bbox_cache)
        if bounds is None:
            return None
        return [
            (bounds[0][index] + bounds[1][index]) * 0.5
            for index in range(3)
        ]

    @staticmethod
    def _parse_metadata_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [
                item.strip()
                for item in text.split(',')
                if item.strip()
            ]

    def _current_sim_time_s(self) -> float:
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            current = float(timeline.get_current_time())
            if np.isfinite(current) and current >= 0.0:
                return current
        except Exception:
            pass
        return max(0.0, float(time.monotonic() - self._world_state_wall_start))

    def _sync_world_state_time(self) -> float:
        sim_time_s = self._current_sim_time_s()
        self.world_state.sync_time(sim_time_s)
        return sim_time_s

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return IsaacSimAppRunner._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return IsaacSimAppRunner._json_safe(value.item())
        if isinstance(value, dict):
            return {
                str(IsaacSimAppRunner._json_safe(key)): IsaacSimAppRunner._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [IsaacSimAppRunner._json_safe(item) for item in value]
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _ok(command: str, payload=None, message: str = ''):
        return json.dumps(
            {
                'ok': True,
                'command': command,
                'message': message or f'{command} completed.',
                'payload': IsaacSimAppRunner._json_safe(payload or {}),
            },
            ensure_ascii=True,
        )

    @staticmethod
    def _error(command: str, error: str, payload=None):
        return json.dumps(
            {
                'ok': False,
                'command': command,
                'error': error,
                'payload': IsaacSimAppRunner._json_safe(payload or {}),
            },
            ensure_ascii=True,
        )

    def _pose_dict_from_raw(self, position, orientation) -> Dict[str, Dict[str, float]]:
        quaternion = self._normalize_quaternion(np.array(orientation, dtype=float))
        euler = quats_to_euler_angles(np.array(orientation, dtype=float))
        return {
            'position': {
                'x': float(position[0]),
                'y': float(position[1]),
                'z': float(position[2]),
            },
            'orientation': {
                'roll': float(euler[0]),
                'pitch': float(euler[1]),
                'yaw': float(euler[2]),
                'w': float(quaternion[0]),
                'x': float(quaternion[1]),
                'y': float(quaternion[2]),
                'z': float(quaternion[3]),
            },
        }

    def _get_pose_for_prim_path(self, prim_path: str) -> Optional[Dict[str, Dict[str, float]]]:
        if not prim_path:
            return None
        pose_prim_path = self._resolve_pose_prim_path(prim_path)
        try:
            position, orientation = get_world_pose(pose_prim_path)
            return self._pose_dict_from_raw(position, orientation)
        except Exception:
            return None

    def _resolve_pose_prim_path(self, prim_path: str) -> str:
        resolved_path = str(prim_path or '').strip()
        if not resolved_path or self.stage is None:
            return resolved_path

        prim = self.stage.GetPrimAtPath(resolved_path)
        if prim is None or not prim.IsValid():
            return resolved_path

        base_link_prim = self.stage.GetPrimAtPath(f'{resolved_path}/base_link')
        if base_link_prim is not None and base_link_prim.IsValid():
            return str(base_link_prim.GetPath())

        return resolved_path

    @staticmethod
    def _bbox_dict_to_arrays(bounds: Dict[str, Any]) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not isinstance(bounds, dict):
            return None
        min_corner = bounds.get('min') or {}
        max_corner = bounds.get('max') or {}
        try:
            minimum = np.array(
                [
                    float(min_corner.get('x', 0.0)),
                    float(min_corner.get('y', 0.0)),
                    float(min_corner.get('z', 0.0)),
                ],
                dtype=float,
            )
            maximum = np.array(
                [
                    float(max_corner.get('x', 0.0)),
                    float(max_corner.get('y', 0.0)),
                    float(max_corner.get('z', 0.0)),
                ],
                dtype=float,
            )
        except (TypeError, ValueError):
            return None
        if np.any(~np.isfinite(minimum)) or np.any(~np.isfinite(maximum)):
            return None
        return minimum, maximum

    @staticmethod
    def _bbox_overlap_area_xy(
        min_corner_a: np.ndarray,
        max_corner_a: np.ndarray,
        min_corner_b: np.ndarray,
        max_corner_b: np.ndarray,
    ) -> float:
        overlap_x = max(
            0.0,
            min(float(max_corner_a[0]), float(max_corner_b[0])) - max(float(min_corner_a[0]), float(min_corner_b[0])),
        )
        overlap_y = max(
            0.0,
            min(float(max_corner_a[1]), float(max_corner_b[1])) - max(float(min_corner_a[1]), float(min_corner_b[1])),
        )
        return float(overlap_x * overlap_y)

    def _find_room_object_entry(self, obj_name: str) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        requested = str(obj_name or '').strip().lower()
        if not requested:
            return None, None

        prioritized_rooms: list[Dict[str, Any]] = []
        current_room_name = str(self._detect_current_room_name() or '')
        if current_room_name:
            current_room_entry = self._resolve_room_entry(current_room_name)
            if current_room_entry is not None:
                prioritized_rooms.append(current_room_entry)

        for room_entry in self.room_index.values():
            if not isinstance(room_entry, dict):
                continue
            if any(room_entry is existing for existing in prioritized_rooms):
                continue
            prioritized_rooms.append(room_entry)

        for room_entry in prioritized_rooms:
            for obj in room_entry.get('objects') or []:
                if not isinstance(obj, dict):
                    continue
                candidate_name = str(obj.get('name', '') or '').strip()
                if candidate_name.lower() == requested:
                    return room_entry, obj
        return None, None

    def _get_all_known_object_names(self) -> list[str]:
        seen_names: set[str] = set()
        all_names: list[str] = []

        for room_entry in self.room_index.values():
            if not isinstance(room_entry, dict):
                continue
            for obj in room_entry.get('objects') or []:
                if not isinstance(obj, dict):
                    continue
                object_name = str(obj.get('name', '') or '').strip()
                lowered = object_name.lower()
                if not object_name or lowered in seen_names:
                    continue
                seen_names.add(lowered)
                all_names.append(object_name)

        for object_name in self.object_prim_paths.keys():
            normalized_name = str(object_name or '').strip()
            lowered = normalized_name.lower()
            if not normalized_name or lowered in seen_names:
                continue
            seen_names.add(lowered)
            all_names.append(normalized_name)

        return all_names

    def _find_global_object_entry(self, obj_name: str) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        room_entry, target_entry = self._find_room_object_entry(obj_name)
        if room_entry is not None and target_entry is not None:
            return room_entry, target_entry

        requested_name = str(obj_name or '').strip()
        if not requested_name:
            return None, None

        prim_path = self._lookup_object_prim_path(requested_name)
        if not prim_path:
            return None, None

        canonical_name = self._lookup_object_name_by_prim_path(prim_path) or requested_name
        prim = self.stage.GetPrimAtPath(prim_path) if self.stage is not None else None
        bbox = self._compute_world_bbox(prim) if prim is not None and prim.IsValid() else None
        pose = self._get_pose_for_prim_path(prim_path)

        room_entry = None
        if bbox is not None:
            bbox_center = np.array(
                [
                    float((bbox[0][0] + bbox[1][0]) / 2.0),
                    float((bbox[0][1] + bbox[1][1]) / 2.0),
                    float((bbox[0][2] + bbox[1][2]) / 2.0),
                ],
                dtype=float,
            )
            room_entry, _ = self._find_room_object_entry_by_position(bbox_center)

        if room_entry is None and isinstance(pose, dict):
            pose_position = pose.get('position') or {}
            try:
                pose_point = np.array(
                    [
                        float(pose_position.get('x', 0.0)),
                        float(pose_position.get('y', 0.0)),
                        float(pose_position.get('z', 0.0)),
                    ],
                    dtype=float,
                )
                room_entry, _ = self._find_room_object_entry_by_position(pose_point)
            except (TypeError, ValueError):
                room_entry = None

        target_entry = {
            'name': canonical_name,
            'prim_path': prim_path,
            'bounds': self._bbox_to_dict(*bbox) if bbox is not None else {},
        }
        return room_entry, target_entry

    def _find_room_object_entry_by_position(
        self,
        target_position: np.ndarray,
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        point = np.array(target_position, dtype=float)
        if point.shape[0] < 3 or np.any(~np.isfinite(point[:3])):
            return None, None

        prioritized_rooms: list[Dict[str, Any]] = []
        current_room_name = str(self._detect_current_room_name() or '')
        if current_room_name:
            current_room_entry = self._resolve_room_entry(current_room_name)
            if current_room_entry is not None:
                prioritized_rooms.append(current_room_entry)

        for room_entry in self.room_index.values():
            if not isinstance(room_entry, dict):
                continue
            if any(room_entry is existing for existing in prioritized_rooms):
                continue
            prioritized_rooms.append(room_entry)

        best_match: Optional[tuple[tuple[float, float, float, float], Dict[str, Any], Dict[str, Any]]] = None
        containment_margin = 0.03

        for room_entry in prioritized_rooms:
            for obj in room_entry.get('objects') or []:
                if not isinstance(obj, dict):
                    continue

                bounds = self._bbox_dict_to_arrays(obj.get('bounds') or {})
                if bounds is None:
                    continue
                min_corner, max_corner = bounds
                bbox_distance = self._point_to_aabb_distance(point, min_corner, max_corner)
                contains_point = bool(
                    float(min_corner[0]) - containment_margin <= float(point[0]) <= float(max_corner[0]) + containment_margin
                    and float(min_corner[1]) - containment_margin <= float(point[1]) <= float(max_corner[1]) + containment_margin
                    and float(min_corner[2]) - containment_margin <= float(point[2]) <= float(max_corner[2]) + containment_margin
                )

                prim_path = str(obj.get('prim_path', '') or '').strip()
                pose_distance = float('inf')
                if prim_path:
                    pose = self.object_poses.get(str(obj.get('name', '') or '')) or self._get_pose_for_prim_path(prim_path)
                    if isinstance(pose, dict):
                        pose_position = pose.get('position') or {}
                        try:
                            pose_distance = float(
                                np.linalg.norm(
                                    point[:3]
                                    - np.array(
                                        [
                                            float(pose_position.get('x', 0.0)),
                                            float(pose_position.get('y', 0.0)),
                                            float(pose_position.get('z', 0.0)),
                                        ],
                                        dtype=float,
                                    )
                                )
                            )
                        except (TypeError, ValueError):
                            pose_distance = float('inf')

                volume = max(float(max_corner[0] - min_corner[0]), 1e-6) * max(
                    float(max_corner[1] - min_corner[1]),
                    1e-6,
                ) * max(float(max_corner[2] - min_corner[2]), 1e-6)
                score = (
                    0.0 if contains_point else 1.0,
                    pose_distance,
                    bbox_distance,
                    volume,
                )
                if best_match is None or score < best_match[0]:
                    best_match = (score, room_entry, obj)

        if best_match is None:
            return None, None
        return best_match[1], best_match[2]

    def _resolve_target_object_entry(
        self,
        target_position: np.ndarray,
        *,
        target_object: str = '',
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        requested_name = str(target_object or '').strip()
        if requested_name:
            self._refresh_scene_index()
            return self._find_global_object_entry(requested_name)
        return self._find_room_object_entry_by_position(target_position)

    def _select_support_object_entry(
        self,
        *,
        room_entry: Dict[str, Any],
        target_entry: Dict[str, Any],
        target_bounds: tuple[np.ndarray, np.ndarray],
    ) -> Dict[str, Any]:
        target_name = str(target_entry.get('name', '') or '').strip().lower()
        target_min, target_max = target_bounds
        target_area = max(float(target_max[0] - target_min[0]), 1e-6) * max(float(target_max[1] - target_min[1]), 1e-6)
        target_center_x = float((target_min[0] + target_max[0]) / 2.0)
        target_center_y = float((target_min[1] + target_max[1]) / 2.0)

        best_entry: Optional[Dict[str, Any]] = None
        best_score = float('-inf')
        surface_keywords = tuple(str(token).lower() for token in self.DEFAULT_SUPPORT_SURFACE_KEYWORDS)

        for obj in room_entry.get('objects') or []:
            if not isinstance(obj, dict):
                # print('[support_debug] reject non_dict_object')
                continue

            candidate_name = str(obj.get('name', '') or '').strip()
            lowered_name = candidate_name.lower()
            if not candidate_name or lowered_name == target_name:
                # print(
                #     '[support_debug] reject invalid_or_target '
                #     f'candidate={candidate_name!r} target={target_name!r}'
                # )
                continue
            if any(token in lowered_name for token in ('wall', 'door', 'ceiling', 'light', 'camera', 'window', 'room')):
                # print(f'[support_debug] reject architectural candidate={candidate_name!r}')
                continue

            bounds = self._bbox_dict_to_arrays(obj.get('bounds') or {})
            if bounds is None:
                # print(f'[support_debug] reject bounds_unavailable candidate={candidate_name!r}')
                continue

            candidate_min, candidate_max = bounds
            vertical_gap = float(target_min[2] - candidate_max[2])
            # Allow a small negative gap because bbox noise may place the object
            # a few centimeters inside the support surface numerically.
            if vertical_gap < -0.25 or vertical_gap > float(self.DEFAULT_SUPPORT_VERTICAL_GAP_TOLERANCE):
                # print(
                #     '[support_debug] reject vertical_gap_out_of_range '
                #     f'candidate={candidate_name!r} vertical_gap={vertical_gap:.4f} '
                #     f'allowed=[-0.0800,{float(self.DEFAULT_SUPPORT_VERTICAL_GAP_TOLERANCE):.4f}] '
                #     f'target_min_z={float(target_min[2]):.4f} candidate_max_z={float(candidate_max[2]):.4f}'
                # )
                continue

            overlap_area = self._bbox_overlap_area_xy(target_min, target_max, candidate_min, candidate_max)
            overlap_ratio = overlap_area / max(target_area, 1e-6)
            # Treat the target center as "inside" with 3 cm slack to absorb
            # imperfect bbox alignment near table edges.
            center_inside = (
                float(candidate_min[0]) - 0.03 <= target_center_x <= float(candidate_max[0]) + 0.03
                and float(candidate_min[1]) - 0.03 <= target_center_y <= float(candidate_max[1]) + 0.03
            )
            # Reject candidates with almost no XY support unless the target
            # center still clearly lands on the surface.
            if overlap_ratio <= 0.05 and not center_inside:
                # print(
                #     '[support_debug] reject xy_overlap_too_small '
                #     f'candidate={candidate_name!r} overlap_ratio={overlap_ratio:.4f} '
                #     f'center_inside={center_inside} overlap_area={overlap_area:.6f} '
                #     f'target_center=({target_center_x:.4f},{target_center_y:.4f}) '
                #     f'candidate_min=({float(candidate_min[0]):.4f},{float(candidate_min[1]):.4f}) '
                #     f'candidate_max=({float(candidate_max[0]):.4f},{float(candidate_max[1]):.4f})'
                # )
                continue

            candidate_area = max(float(candidate_max[0] - candidate_min[0]), 1e-6) * max(
                float(candidate_max[1] - candidate_min[1]),
                1e-6,
            )
            size_ratio = candidate_area / max(target_area, 1e-6)
            # Name priors nudge support selection toward tables/counters/shelves
            # and away from floor planes when geometry alone is ambiguous.
            support_bonus = 2.5 if any(token in lowered_name for token in surface_keywords) else 0.0
            floor_penalty = 4.0 if any(token in lowered_name for token in ('floor', 'ground')) else 0.0

            # Score overlap first, then center alignment and larger support area,
            # while penalizing vertical mismatch between object and support.
            score = (
                overlap_ratio * 12.0
                + (2.0 if center_inside else 0.0)
                + min(size_ratio, 12.0) * 0.35
                - abs(vertical_gap) * 10.0
                + support_bonus
                - floor_penalty
            )
            # print(
            #     '[support_debug] accept_candidate '
            #     f'candidate={candidate_name!r} score={score:.4f} vertical_gap={vertical_gap:.4f} '
            #     f'overlap_ratio={overlap_ratio:.4f} center_inside={center_inside} '
            #     f'size_ratio={size_ratio:.4f} support_bonus={support_bonus:.4f} floor_penalty={floor_penalty:.4f}'
            # )
            if score > best_score:
                best_score = score
                best_entry = obj

        # print(
        #     '[support_debug] selected '
        #     f'candidate={str((best_entry or target_entry).get("name", "") or "")!r} '
        #     f'fallback_to_target={best_entry is None} best_score={best_score}'
        # )
        return dict(best_entry or target_entry)

    def _build_navigation_nav_map(self):
        if self.robot_controller is None:
            raise RuntimeError('robot_not_loaded')

        nav_planner = getattr(self.robot_controller, '_nav_planner', None)
        build_navigation_scene = getattr(self.robot_controller, '_build_navigation_scene', None)
        build_nav_map = getattr(nav_planner, '_build_nav_map', None) if nav_planner is not None else None
        if not callable(build_navigation_scene) or not callable(build_nav_map):
            raise RuntimeError('navigation_map_unavailable')

        geometry = ensure_shapely()
        rooms = build_navigation_scene()
        return geometry, build_nav_map(rooms, geometry)[0]

    def _clear_manipulation_suggestion_debug_prims(self) -> None:
        if self.stage is None:
            return
        debug_root = self.stage.GetPrimAtPath(self.MANIPULATION_SUGGESTION_DEBUG_ROOT)
        if debug_root is not None and debug_root.IsValid():
            self.stage.RemovePrim(debug_root.GetPath())

    def _ensure_manipulation_suggestion_debug_root(self) -> str:
        if self.stage is None:
            return self.MANIPULATION_SUGGESTION_DEBUG_ROOT
        UsdGeom.Xform.Define(self.stage, '/World')
        UsdGeom.Xform.Define(self.stage, '/World/debug')
        UsdGeom.Xform.Define(self.stage, self.MANIPULATION_SUGGESTION_DEBUG_ROOT)
        return self.MANIPULATION_SUGGESTION_DEBUG_ROOT

    def _create_debug_sphere_marker(
        self,
        *,
        prim_path: str,
        position: np.ndarray,
        radius: float,
        color: tuple[float, float, float],
    ) -> None:
        if self.stage is None:
            return
        sphere = UsdGeom.Sphere.Define(self.stage, prim_path)
        sphere.CreateRadiusAttr().Set(float(radius))
        UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(
            Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        )
        gprim = UsdGeom.Gprim(sphere.GetPrim())
        display_color = gprim.GetDisplayColorPrimvar()
        if not display_color:
            display_color = gprim.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
        display_color.Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])

    def _visualize_manipulation_suggestion(
        self,
        *,
        obj_name: str,
        floor_z: float,
        start_xy: np.ndarray,
        object_position: np.ndarray,
        target_position: np.ndarray,
        support_candidates: list[Dict[str, Any]],
        candidate_debug_records: list[Dict[str, Any]],
        selected_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.DEBUG_VISUALIZE_MANIPULATION_SUGGESTION or self.stage is None:
            return

        self._clear_manipulation_suggestion_debug_prims()
        debug_root = self._ensure_manipulation_suggestion_debug_root()
        safe_name = self._safe_filename(obj_name) or 'target'

        self._create_debug_sphere_marker(
            prim_path=f'{debug_root}/{safe_name}_robot_start',
            position=np.array([float(start_xy[0]), float(start_xy[1]), float(floor_z + 0.04)], dtype=float),
            radius=0.08,
            color=(0.2, 0.45, 1.0),
        )
        self._create_debug_sphere_marker(
            prim_path=f'{debug_root}/{safe_name}_object_center',
            position=np.array(object_position, dtype=float),
            radius=0.06,
            color=(0.95, 0.95, 0.95),
        )
        self._create_debug_sphere_marker(
            prim_path=f'{debug_root}/{safe_name}_grasp_target',
            position=np.array(target_position, dtype=float),
            radius=0.07,
            color=(1.0, 0.9, 0.15),
        )

        for index, support_entry in enumerate(support_candidates):
            bounds = self._bbox_dict_to_arrays(support_entry.get('bounds') or {})
            if bounds is None:
                continue
            support_center = (bounds[0] + bounds[1]) / 2.0
            self._create_debug_sphere_marker(
                prim_path=f'{debug_root}/{safe_name}_support_{index:02d}',
                position=np.array(
                    [
                        float(support_center[0]),
                        float(support_center[1]),
                        float(max(bounds[1][2], floor_z) + 0.03),
                    ],
                    dtype=float,
                ),
                radius=0.07 if index == 0 else 0.055,
                color=(0.7, 0.35, 1.0) if index == 0 else (0.5, 0.5, 0.9),
            )

        for index, record in enumerate(candidate_debug_records):
            raw_candidate = record.get('raw_candidate_xy') or {}
            snapped_candidate = record.get('snapped_candidate_xy') or {}
            support_side = self._safe_filename(str(record.get('support_side', '') or f'candidate_{index:02d}'))
            status = str(record.get('status', '') or '')
            try:
                raw_position = np.array(
                    [
                        float(raw_candidate.get('x', 0.0)),
                        float(raw_candidate.get('y', 0.0)),
                        float(floor_z + 0.025),
                    ],
                    dtype=float,
                )
                snapped_position = np.array(
                    [
                        float(snapped_candidate.get('x', 0.0)),
                        float(snapped_candidate.get('y', 0.0)),
                        float(floor_z + 0.065),
                    ],
                    dtype=float,
                )
            except (TypeError, ValueError):
                continue

            self._create_debug_sphere_marker(
                prim_path=f'{debug_root}/{safe_name}_{support_side}_{index:02d}_raw',
                position=raw_position,
                radius=0.04,
                color=(0.95, 0.2, 0.2),
            )
            snapped_color = (0.35, 0.95, 0.35) if status == 'selected' else (1.0, 0.55, 0.15)
            snapped_radius = 0.055 if status == 'selected' else 0.045
            self._create_debug_sphere_marker(
                prim_path=f'{debug_root}/{safe_name}_{support_side}_{index:02d}_snapped',
                position=snapped_position,
                radius=snapped_radius,
                color=snapped_color,
            )

        if selected_result:
            selected_pose = dict(selected_result.get('target_pose') or {})
            selected_position = dict(selected_pose.get('position') or {})
            self._create_debug_sphere_marker(
                prim_path=f'{debug_root}/{safe_name}_selected_pose',
                position=np.array(
                    [
                        float(selected_position.get('x', 0.0)),
                        float(selected_position.get('y', 0.0)),
                        float(selected_position.get('z', floor_z) + 0.1),
                    ],
                    dtype=float,
                ),
                radius=0.085,
                color=(0.1, 1.0, 0.1),
            )

    @staticmethod
    def _snap_point_to_free_space(
        point_xy: np.ndarray,
        free_space,
        geometry,
    ) -> np.ndarray:
        candidate_point = geometry['Point'](float(point_xy[0]), float(point_xy[1]))
        expanded_free_space = free_space.buffer(1e-6)
        if expanded_free_space.covers(candidate_point):
            return np.array(point_xy[:2], dtype=float)

        _, nearest_point = geometry['nearest_points'](candidate_point, free_space)
        return np.array([float(nearest_point.x), float(nearest_point.y)], dtype=float)

    def _iter_manipulation_candidate_xy(
        self,
        *,
        target_position: np.ndarray,
        start_xy: Optional[np.ndarray] = None,
    ):
        """Yield ring-sampled base candidates around the target grasp point.

        The sampling rule is intentionally simple:

        * center the ring at the target grasp position;
        * set the ring radius to DEFAULT_MANIPULATION_BASE_REACH;
        * sample uniformly around that ring;
        * try the direction from the grasp point toward the current base first
          when the current base is known.

        Support-object bounds, table sides, lateral offsets, and ad-hoc min/max
        distance bands are deliberately not part of this generator.  The only
        geometric intent here is to keep the base center roughly one arm-reach
        away from the grasp point.
        """
        target_xy = np.array(target_position[:2], dtype=float)
        reach = float(self.DEFAULT_MANIPULATION_BASE_REACH)
        if not np.isfinite(reach) or reach <= 0.0:
            raise RuntimeError(f'invalid_manipulation_base_reach:{reach}')

        preferred_angle: Optional[float] = None
        if start_xy is not None:
            start_vector = np.array(start_xy[:2], dtype=float) - target_xy
            start_distance = float(np.linalg.norm(start_vector))
            if np.isfinite(start_distance) and start_distance > 1e-6:
                preferred_angle = float(np.arctan2(start_vector[1], start_vector[0]))

        sample_count = 32
        raw_angles: list[tuple[str, float]] = []
        if preferred_angle is not None:
            raw_angles.append(('toward_current_base', preferred_angle))
        for index in range(sample_count):
            angle = float((2.0 * np.pi * index) / sample_count)
            degrees = int(round(np.degrees(angle))) % 360
            raw_angles.append((f'ring_{degrees:03d}deg', angle))

        if preferred_angle is not None:
            def _angle_distance(item: tuple[str, float]) -> float:
                delta = (float(item[1]) - preferred_angle + np.pi) % (2.0 * np.pi) - np.pi
                return abs(float(delta))

            raw_angles.sort(key=lambda item: (0 if item[0] == 'toward_current_base' else 1, _angle_distance(item)))

        seen: set[tuple[float, float]] = set()
        for label, angle in raw_angles:
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
            candidate_xy = target_xy + direction * reach
            key = (round(float(candidate_xy[0]), 3), round(float(candidate_xy[1]), 3))
            if key in seen:
                continue
            seen.add(key)
            yield label, candidate_xy

    def _try_resolve_manipulation_base_pose(
        self,
        *,
        room_name: str,
        room_free_space,
        geometry,
        start_xy: np.ndarray,
        support_entry: Dict[str, Any],
        target_position: np.ndarray,
        floor_z: float,
        diagnostics: Optional[list[Dict[str, Any]]] = None,
        candidate_debug_records: Optional[list[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if room_free_space.is_empty:
            if diagnostics is not None:
                diagnostics.append(
                    {
                        'support_object_name': str(support_entry.get('name', '') or ''),
                        'reject_reason': 'room_free_space_empty',
                    }
                )
            return None

        desired_reach = float(self.DEFAULT_MANIPULATION_BASE_REACH)
        if not np.isfinite(desired_reach) or desired_reach <= 0.0:
            raise RuntimeError(f'invalid_manipulation_base_reach:{desired_reach}')
        reach_tolerance = max(0.10, desired_reach * 0.30)
        expanded_free_space = room_free_space.buffer(1e-6)
        valid_candidates: list[tuple[float, float, int, np.ndarray, str, Dict[str, Any]]] = []

        for support_side, raw_candidate_xy in self._iter_manipulation_candidate_xy(
            target_position=target_position,
            start_xy=start_xy,
        ):
            candidate_xy = self._snap_point_to_free_space(raw_candidate_xy, room_free_space, geometry)
            snap_distance = float(np.linalg.norm(candidate_xy - raw_candidate_xy))
            target_distance = float(np.linalg.norm(target_position[:2] - candidate_xy))
            reach_error = abs(target_distance - desired_reach)
            start_distance = float(np.linalg.norm(candidate_xy - np.array(start_xy[:2], dtype=float)))
            candidate_record = {
                'support_object_name': str(support_entry.get('name', '') or ''),
                'support_side': support_side,
                'raw_candidate_xy': {
                    'x': float(raw_candidate_xy[0]),
                    'y': float(raw_candidate_xy[1]),
                },
                'snapped_candidate_xy': {
                    'x': float(candidate_xy[0]),
                    'y': float(candidate_xy[1]),
                },
                'snap_distance': snap_distance,
                'target_distance': target_distance,
                'desired_reach': desired_reach,
                'reach_error': reach_error,
                'reach_tolerance': reach_tolerance,
                'start_distance': start_distance,
            }

            if (
                not np.isfinite(target_distance)
                or not np.isfinite(reach_error)
                or target_distance <= 1e-6
            ):
                if candidate_debug_records is not None:
                    candidate_debug_records.append(
                        {
                            **candidate_record,
                            'status': 'rejected',
                            'reject_reason': 'invalid_base_target_distance',
                        }
                    )
                if diagnostics is not None:
                    candidate_record['reject_reason'] = 'invalid_base_target_distance'
                    diagnostics.append(candidate_record)
                continue

            candidate_point = geometry['Point'](float(candidate_xy[0]), float(candidate_xy[1]))
            if not expanded_free_space.covers(candidate_point):
                if candidate_debug_records is not None:
                    candidate_debug_records.append(
                        {
                            **candidate_record,
                            'status': 'rejected',
                            'reject_reason': 'candidate_not_in_free_space',
                        }
                    )
                if diagnostics is not None:
                    candidate_record['reject_reason'] = 'candidate_not_in_free_space'
                    diagnostics.append(candidate_record)
                continue

            if reach_error > reach_tolerance:
                if candidate_debug_records is not None:
                    candidate_debug_records.append(
                        {
                            **candidate_record,
                            'status': 'rejected',
                            'reject_reason': 'base_reach_error_too_large',
                        }
                    )
                if diagnostics is not None:
                    candidate_record['reject_reason'] = 'base_reach_error_too_large'
                    diagnostics.append(candidate_record)
                continue

            try:
                # Validate that the current base can actually navigate to the
                # suggested pose before returning it to the policy layer.
                navigation_plan = self.robot_controller._plan_navigation(
                    start_xy=np.array(start_xy[:2], dtype=float),
                    goal_xy=np.array(candidate_xy[:2], dtype=float),
                )
            except Exception as exc:
                if candidate_debug_records is not None:
                    candidate_debug_records.append(
                        {
                            **candidate_record,
                            'status': 'rejected',
                            'reject_reason': 'navigation_plan_failed',
                            'navigation_error': str(exc),
                        }
                    )
                if diagnostics is not None:
                    candidate_record.update(
                        {
                            'reject_reason': 'navigation_plan_failed',
                            'navigation_error': str(exc),
                        }
                    )
                    diagnostics.append(candidate_record)
                continue

            waypoints = list(getattr(navigation_plan, 'waypoints', []) or [])
            candidate_record['navigation_waypoint_count'] = len(waypoints)
            if candidate_debug_records is not None:
                candidate_debug_records.append({**candidate_record, 'status': 'valid'})
            valid_candidates.append(
                (
                    reach_error,
                    start_distance,
                    len(waypoints),
                    candidate_xy,
                    support_side,
                    candidate_record,
                )
            )

        if not valid_candidates:
            return None

        _reach_error, _start_distance, _waypoint_count, candidate_xy, support_side, selected_record = min(
            valid_candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        if candidate_debug_records is not None:
            for record in candidate_debug_records:
                same_xy = dict(record.get('snapped_candidate_xy') or {})
                if (
                    str(record.get('support_side') or '') == support_side
                    and abs(float(same_xy.get('x', float('inf'))) - float(candidate_xy[0])) < 1e-6
                    and abs(float(same_xy.get('y', float('inf'))) - float(candidate_xy[1])) < 1e-6
                ):
                    record['status'] = 'selected'
                    break

        # Face the base directly toward the target grasp point.
        yaw = float(np.arctan2(target_position[1] - candidate_xy[1], target_position[0] - candidate_xy[0]))
        target_distance = float(selected_record.get('target_distance', np.linalg.norm(target_position[:2] - candidate_xy)))
        reach_error = float(selected_record.get('reach_error', abs(target_distance - desired_reach)))
        return {
            'target_pose': {
                'position': {
                    'x': float(candidate_xy[0]),
                    'y': float(candidate_xy[1]),
                    'z': float(floor_z),
                },
                'orientation': {
                    'roll': 0.0,
                    'pitch': 0.0,
                    'yaw': yaw,
                },
            },
            'support_object_name': str(support_entry.get('name', '') or ''),
            'support_side': support_side,
            'target_distance': target_distance,
            'desired_reach': desired_reach,
            'reach_error': reach_error,
            'navigation_waypoint_count': int(selected_record.get('navigation_waypoint_count', 0) or 0),
        }

    def _suggest_manipulation_base_pose(self, obj_name: str) -> Dict[str, Any]:
        if self.robot_controller is None:
            raise RuntimeError('robot_not_loaded')
        if self.stage is None:
            raise RuntimeError('scene_not_loaded')

        requested_name = str(obj_name or '').strip()
        if not requested_name:
            raise ValueError('obj_name_required')

        self._refresh_scene_index()
        room_entry, target_entry = self._find_global_object_entry(requested_name)
        if room_entry is None or target_entry is None:
            raise ValueError(f'object_not_found:{requested_name}')

        target_bounds = self._bbox_dict_to_arrays(target_entry.get('bounds') or {})
        if target_bounds is None:
            raise RuntimeError(f'object_bounds_unavailable:{requested_name}')

        target_prim_path = str(target_entry.get('prim_path', '') or '').strip()
        target_pose = self._get_pose_for_prim_path(target_prim_path)
        if target_pose is not None:
            object_position = np.array(
                [
                    float((target_pose.get('position') or {}).get('x', 0.0)),
                    float((target_pose.get('position') or {}).get('y', 0.0)),
                    float((target_pose.get('position') or {}).get('z', 0.0)),
                ],
                dtype=float,
            )
        else:
            target_min, target_max = target_bounds
            object_position = np.array(
                [
                    float((target_min[0] + target_max[0]) / 2.0),
                    float((target_min[1] + target_max[1]) / 2.0),
                    float((target_min[2] + target_max[2]) / 2.0),
                ],
                dtype=float,
            )

        preferred_grasp = self._resolve_preferred_grasp_pose(
            requested_name,
            prim_path=target_prim_path,
            object_pose=target_pose,
            fallback_position=object_position,
            force_refresh=True,
        )
        grasp_position = np.array(
            [
                float(((preferred_grasp.get('grasp_pose') or {}).get('position') or {}).get('x', object_position[0])),
                float(((preferred_grasp.get('grasp_pose') or {}).get('position') or {}).get('y', object_position[1])),
                float(((preferred_grasp.get('grasp_pose') or {}).get('position') or {}).get('z', object_position[2])),
            ],
            dtype=float,
        )
        target_position = grasp_position

        # Set manipulation torso height from the resolved grasp position, not the object center.
        applied_torso_height = self._set_manipulation_torso_height_from_grasp_height(float(grasp_position[2]))

        geometry, nav_map = self._build_navigation_nav_map()
        room_name = str(room_entry.get('room_name', '') or '')
        room_geometry = nav_map.get(room_name)
        if room_geometry is None:
            raise RuntimeError(f'navigation_room_not_found:{room_name}')

        raw_base_pose = self._get_robot_base_pose_raw()
        if raw_base_pose is None:
            raise RuntimeError('base_pose_unavailable')
        start_xy = np.array(raw_base_pose[0][:2], dtype=float)

        room_bounds = self._bbox_dict_to_arrays(room_entry.get('bounds') or {})
        fallback_floor_z = float(raw_base_pose[0][2])
        if room_bounds is not None:
            fallback_floor_z = float(max(float(room_bounds[0][2]), 0.0))
        floor_z = self._estimate_room_floor_height(room_entry, fallback_floor_z=fallback_floor_z)

        primary_support = self._select_support_object_entry(
            room_entry=room_entry,
            target_entry=target_entry,
            target_bounds=target_bounds,
        )
        support_candidates = [primary_support]
        primary_support_name = str(primary_support.get('name', '') or '').strip().lower()
        target_name = str(target_entry.get('name', '') or '').strip().lower()
        if primary_support_name != target_name:
            support_candidates.append(dict(target_entry))

        diagnostic_desired_reach = float(self.DEFAULT_MANIPULATION_BASE_REACH)
        diagnostic_reach_tolerance = max(0.10, diagnostic_desired_reach * 0.30)
        diagnostics: Dict[str, Any] = {
            'obj_name': requested_name,
            'room_name': room_name,
            'target_object_name': str(target_entry.get('name', '') or ''),
            'target_prim_path': target_prim_path,
            'object_position': {
                'x': float(object_position[0]),
                'y': float(object_position[1]),
                'z': float(object_position[2]),
            },
            'target_position': {
                'x': float(target_position[0]),
                'y': float(target_position[1]),
                'z': float(target_position[2]),
            },
            'grasp_position': {
                'x': float(grasp_position[0]),
                'y': float(grasp_position[1]),
                'z': float(grasp_position[2]),
            },
            'applied_torso_height': float(applied_torso_height),
            'grasp_pose_source': str(preferred_grasp.get('grasp_pose_source', '') or ''),
            'start_xy': {
                'x': float(start_xy[0]),
                'y': float(start_xy[1]),
            },
            'thresholds': {
                'desired_reach': diagnostic_desired_reach,
                'reach_tolerance': diagnostic_reach_tolerance,
                'ring_sample_count': 32,
            },
            'support_candidates': [
                {
                    'name': str(item.get('name', '') or ''),
                    'prim_path': str(item.get('prim_path', '') or ''),
                    'is_primary': index == 0,
                }
                for index, item in enumerate(support_candidates)
            ],
            'candidate_rejections': [],
        }
        candidate_rejections = diagnostics['candidate_rejections']
        candidate_debug_records: list[Dict[str, Any]] = []

        for support_entry in support_candidates:
            result = self._try_resolve_manipulation_base_pose(
                room_name=room_name,
                room_free_space=room_geometry.free_space,
                geometry=geometry,
                start_xy=start_xy,
                support_entry=support_entry,
                target_position=target_position,
                floor_z=floor_z,
                diagnostics=candidate_rejections,
                candidate_debug_records=candidate_debug_records,
            )
            if result is None:
                # The ring sampler is independent of support-object geometry,
                # so trying the same ring again with another support label will
                # not produce new navigation candidates.
                break

            # 提前准备抓取目标的物理属性，提高接触稳定性。
            self._prepare_grasp_target_physics(requested_name)
            result_target_pose = dict(result.get('target_pose') or {})
            result_target_position = dict(result_target_pose.get('position') or {})
            # Recompute the grasp position as the nearest point on the target's
            # AABB to the planned base pose (xy plane), keeping z at the
            # object's grasp height. This replaces the previous behaviour of
            # always grasping at the bbox/prim center.
            planned_base_xy = np.array(
                [
                    float(result_target_position.get('x', start_xy[0])),
                    float(result_target_position.get('y', start_xy[1])),
                ],
                dtype=float,
            )
            nearest_grasp_position = self._nearest_bbox_grasp_point_xy(
                target_min=target_bounds[0],
                target_max=target_bounds[1],
                robot_xy=planned_base_xy,
                height_z=float(object_position[2]),
            )
            # Rebuild the cached fallback grasp pose at the nearest point so
            # downstream consumers (and the latch fallback) see the new target.
            object_pose_for_fallback = {
                'position': {
                    'x': float(nearest_grasp_position[0]),
                    'y': float(nearest_grasp_position[1]),
                    'z': float(nearest_grasp_position[2]),
                }
            }
            cache_key = self._normalize_object_cache_key(requested_name)
            cached_entry = dict(self._preferred_grasp_cache.get(cache_key) or {})
            if cached_entry and str(cached_entry.get('grasp_pose_source', '') or '') == 'top_down_fallback':
                cached_entry['grasp_pose'] = self._build_top_down_grasp_pose(
                    object_pose=object_pose_for_fallback,
                    base_pose=result_target_pose,
                )
                cached_entry['object_pose'] = object_pose_for_fallback
                cached_diag = dict(cached_entry.get('diagnostics') or {})
                cached_diag['grasp_position_source'] = 'nearest_bbox_point_to_planned_base'
                cached_entry['diagnostics'] = cached_diag
                self._preferred_grasp_cache[cache_key] = cached_entry
            aligned_grasp = self._align_cached_fallback_grasp_pose_to_base(
                requested_name,
                base_pose=result_target_pose,
            )
            grasp_pose_candidates = self._build_manipulation_grasp_pose_candidates(
                grasp_position=nearest_grasp_position,
                base_pose=result_target_pose,
                robot_position=np.array(
                    [
                        float(result_target_position.get('x', start_xy[0])),
                        float(result_target_position.get('y', start_xy[1])),
                        float(result_target_position.get('z', floor_z)),
                    ],
                    dtype=float,
                ),
            )
            grasp_poses = [
                dict(candidate.get('pose') or {})
                for candidate in grasp_pose_candidates
                if isinstance(candidate, dict) and candidate.get('pose')
            ]
            fallback_grasp_pose = dict(
                (
                    aligned_grasp.get('grasp_pose')
                    if aligned_grasp
                    else preferred_grasp.get('grasp_pose')
                )
                or {}
            )
            selected_grasp_pose = dict(grasp_poses[0] if grasp_poses else fallback_grasp_pose)
            selected_grasp_source = (
                'object_position_candidates'
                if grasp_poses
                else str(
                    (
                        aligned_grasp.get('grasp_pose_source')
                        if aligned_grasp
                        else preferred_grasp.get('grasp_pose_source')
                    )
                    or ''
                )
            )
            grasp_diagnostics = dict(
                (
                    aligned_grasp.get('diagnostics')
                    if aligned_grasp
                    else preferred_grasp.get('diagnostics')
                )
                or {}
            )
            grasp_diagnostics.update(
                {
                    'grasp_pose_candidate_count': len(grasp_pose_candidates),
                    'grasp_pose_candidate_names': [
                        str(candidate.get('name', '') or '')
                        for candidate in grasp_pose_candidates
                    ],
                    'grasp_pose_candidate_frame': 'robot_object_vertical_plane',
                    'grasp_pose_candidate_position_source': 'nearest_bbox_point_to_planned_base',
                    'grasp_position': {
                        'x': float(nearest_grasp_position[0]),
                        'y': float(nearest_grasp_position[1]),
                        'z': float(nearest_grasp_position[2]),
                    },
                    'object_position': {
                        'x': float(object_position[0]),
                        'y': float(object_position[1]),
                        'z': float(object_position[2]),
                    },
                    'fallback_grasp_pose_source': str(preferred_grasp.get('grasp_pose_source', '') or ''),
                }
            )
            result.update(
                {
                    'obj_name': requested_name,
                    'room_name': room_name,
                    'grasp_pose': selected_grasp_pose,
                    'grasp_poses': grasp_poses,
                    'grasp_pose_candidates': grasp_pose_candidates,
                    'grasp_pose_source': selected_grasp_source,
                    'grasp_pose_diagnostics': grasp_diagnostics,
                }
            )
            self._visualize_manipulation_suggestion(
                obj_name=requested_name,
                floor_z=floor_z,
                start_xy=start_xy,
                object_position=object_position,
                target_position=target_position,
                support_candidates=support_candidates,
                candidate_debug_records=candidate_debug_records,
                selected_result=result,
            )
            return result

        diagnostics['rejection_summary'] = self._summarize_rejection_reasons(candidate_rejections)
        self._visualize_manipulation_suggestion(
            obj_name=requested_name,
            floor_z=floor_z,
            start_xy=start_xy,
            object_position=object_position,
            target_position=target_position,
            support_candidates=support_candidates,
            candidate_debug_records=candidate_debug_records,
            selected_result=None,
        )
        raise ManipulationBasePosePlanningError(
            f'manipulation_base_pose_not_found:{requested_name}',
            diagnostics,
        )

    @staticmethod
    def _summarize_rejection_reasons(rejections: list[Dict[str, Any]]) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for item in rejections:
            reason = str(item.get('reject_reason', '') or 'unknown')
            summary[reason] = summary.get(reason, 0) + 1
        return summary

    @staticmethod
    def _point_to_aabb_distance(point: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> float:
        clipped = np.minimum(np.maximum(point, min_corner), max_corner)
        return float(np.linalg.norm(point - clipped))

    @staticmethod
    def _nearest_bbox_grasp_point_xy(
        target_min: np.ndarray,
        target_max: np.ndarray,
        robot_xy: np.ndarray,
        height_z: float,
    ) -> np.ndarray:
        """Return the point on the AABB closest to the robot in the xy plane.

        The xy components are clamped into [target_min, target_max] (so when the
        robot is outside the bbox, the result lies on the bbox face nearest to
        the robot; when inside, the robot's xy is returned). The z coordinate is
        held at `height_z` (typically the object's prim/grasp height) so the EE
        target stays at the intended grasp altitude rather than collapsing to
        the floor or the bbox bottom.
        """
        rmin = np.array(target_min, dtype=float).reshape(-1)[:3]
        rmax = np.array(target_max, dtype=float).reshape(-1)[:3]
        rxy = np.array(robot_xy, dtype=float).reshape(-1)[:2]
        cx = float(np.clip(rxy[0], float(rmin[0]), float(rmax[0])))
        cy = float(np.clip(rxy[1], float(rmin[1]), float(rmax[1])))
        cz = float(np.clip(float(height_z), float(rmin[2]), float(rmax[2])))
        return np.array([cx, cy, cz], dtype=float)

    def _is_architectural_object_name(self, object_name: str) -> bool:
        lowered = str(object_name or '').strip().lower()
        return any(token in lowered for token in self.ARCHITECTURAL_OBJECT_KEYWORDS)

    def _iter_target_neighbor_obstacle_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for room_entry in self.room_index.values():
            if not isinstance(room_entry, dict):
                continue
            for obj in room_entry.get('objects') or []:
                if not isinstance(obj, dict):
                    continue
                prim_path = str(obj.get('prim_path', '') or '').strip()
                object_name = str(obj.get('name', '') or '').strip()
                bounds = obj.get('bounds') or {}
                if not prim_path or prim_path in seen_paths:
                    continue
                seen_paths.add(prim_path)
                candidates.append(
                    {
                        'name': object_name,
                        'prim_path': prim_path,
                        'bounds': bounds,
                    }
                )

        for object_name, prim_path in self.object_prim_paths.items():
            prim_path = str(prim_path or '').strip()
            if not prim_path or prim_path in seen_paths:
                continue
            seen_paths.add(prim_path)
            prim = self.stage.GetPrimAtPath(prim_path) if self.stage is not None else None
            bbox = self._compute_world_bbox(prim)
            candidates.append(
                {
                    'name': str(object_name or '').strip(),
                    'prim_path': prim_path,
                    'bounds': self._bbox_to_dict(*bbox) if bbox is not None else {},
                }
            )

        return candidates

    def _select_target_neighbor_object_prim_paths(
        self,
        target_position: np.ndarray,
        *,
        include_target: bool = False,
        target_object: str = '',
    ) -> list[str]:
        if self.stage is None:
            return []
        self._refresh_scene_index()
        query_distance = float(self.DEFAULT_ARM_OBSTACLE_QUERY_DISTANCE)
        max_count = int(self.DEFAULT_ARM_OBSTACLE_MAX_COUNT)
        minimum_height = float(self.DEFAULT_ARM_OBSTACLE_MIN_HEIGHT)
        candidates_with_distance: list[tuple[float, int, str]] = []
        target_room_entry, target_entry = self._resolve_target_object_entry(
            target_position,
            target_object=target_object,
        )
        target_prim_path = str((target_entry or {}).get('prim_path', '') or '').strip()
        support_prim_path = ''
        if target_room_entry is not None and target_entry is not None:
            target_bounds = self._bbox_dict_to_arrays(target_entry.get('bounds') or {})
            if target_bounds is not None:
                support_entry = self._select_support_object_entry(
                    room_entry=target_room_entry,
                    target_entry=target_entry,
                    target_bounds=target_bounds,
                )
                support_prim_path = str((support_entry or {}).get('prim_path', '') or '').strip()

        if include_target and target_prim_path:
            candidates_with_distance.append((0.0, 0, target_prim_path))

        for candidate in self._iter_target_neighbor_obstacle_candidates():
            object_name = str(candidate.get('name', '') or '').strip()
            prim_path = str(candidate.get('prim_path', '') or '').strip()
            if not object_name or not prim_path:
                continue
            if target_prim_path and prim_path == target_prim_path:
                continue
            if self._is_architectural_object_name(object_name):
                continue
            if self.robot_prim_path and (
                prim_path == self.robot_prim_path or prim_path.startswith(f'{self.robot_prim_path}/')
            ):
                continue
            if prim_path.startswith('/World/fetch_rmpflow_obstacles'):
                continue

            bounds = self._bbox_dict_to_arrays(candidate.get('bounds') or {})
            if bounds is None:
                continue
            min_corner, max_corner = bounds
            if float(max_corner[2] - min_corner[2]) <= minimum_height:
                continue

            distance = self._point_to_aabb_distance(target_position, min_corner, max_corner)
            should_include = distance <= query_distance
            is_support_object = bool(support_prim_path) and prim_path == support_prim_path and prim_path != target_prim_path
            if should_include or is_support_object:
                candidates_with_distance.append((distance, 0 if is_support_object else 1, prim_path))

        candidates_with_distance.sort(key=lambda item: (item[1], item[0], item[2]))
        selected_paths: list[str] = []
        seen_paths: set[str] = set()
        for _, _, prim_path in candidates_with_distance:
            if prim_path in seen_paths:
                continue
            seen_paths.add(prim_path)
            if len(selected_paths) >= max_count:
                break
            selected_paths.append(prim_path)
        return selected_paths

    def _register_auto_arm_obstacles(
        self,
        target_position: np.ndarray,
        *,
        target_object: str = '',
    ) -> list[str]:
        if self.robot_controller is None or self.stage is None:
            return []
        if not hasattr(self.robot_controller, 'add_rmpflow_obstacle'):
            return []

        registered_keys: list[str] = []
        for prim_path in self._select_target_neighbor_object_prim_paths(
            target_position,
            include_target=False,
            target_object=target_object,
        ):
            try:
                obstacle_key = self.robot_controller.add_rmpflow_obstacle(prim_path, static=True)
            except Exception as exc:
                print(f'Auto arm obstacle registration failed for {prim_path}: {exc}')
                continue
            if obstacle_key:
                registered_keys.append(str(obstacle_key))
        return registered_keys

    @staticmethod
    def _aabb_intersects_aabb(
        min_corner_a: np.ndarray,
        max_corner_a: np.ndarray,
        min_corner_b: np.ndarray,
        max_corner_b: np.ndarray,
    ) -> bool:
        min_a = np.asarray(min_corner_a, dtype=float)
        max_a = np.asarray(max_corner_a, dtype=float)
        min_b = np.asarray(min_corner_b, dtype=float)
        max_b = np.asarray(max_corner_b, dtype=float)
        return bool(np.all(max_a >= min_b) and np.all(max_b >= min_a))

    def _select_collision_filter_object_prim_paths(
        self,
        target_position: np.ndarray,
        *,
        query_padding: Optional[float] = None,
        include_target: bool = False,
        target_object: str = '',
    ) -> list[str]:
        if self.stage is None:
            return []

        target_room_entry, target_entry = self._resolve_target_object_entry(
            target_position,
            target_object=target_object,
        )
        target_prim_path = str((target_entry or {}).get('prim_path', '') or '').strip()
        target_bounds = self._bbox_dict_to_arrays((target_entry or {}).get('bounds') or {})
        if target_room_entry is None or target_bounds is None:
            return [target_prim_path] if include_target and target_prim_path else []

        target_min, target_max = target_bounds
        padding = self.DEFAULT_ARM_COLLISION_FILTER_QUERY_PADDING if query_padding is None else query_padding
        padding = max(float(padding), 0.0)
        query_min = target_min - padding
        query_max = target_max + padding

        selected_paths: list[str] = []
        seen_paths: set[str] = set()
        if include_target and target_prim_path:
            selected_paths.append(target_prim_path)
            seen_paths.add(target_prim_path)

        for candidate in target_room_entry.get('objects') or []:
            if not isinstance(candidate, dict):
                continue
            prim_path = str(candidate.get('prim_path', '') or '').strip()
            if not prim_path or prim_path in seen_paths:
                continue
            if target_prim_path and prim_path == target_prim_path:
                continue
            if self.robot_prim_path and (
                prim_path == self.robot_prim_path or prim_path.startswith(f'{self.robot_prim_path}/')
            ):
                continue
            if prim_path in {ROOMS_ROOT_PATH, '/World'}:
                continue
            if prim_path.startswith('/World/fetch_rmpflow_obstacles'):
                continue

            bounds = self._bbox_dict_to_arrays(candidate.get('bounds') or {})
            if bounds is None:
                continue
            min_corner, max_corner = bounds
            if self._aabb_intersects_aabb(min_corner, max_corner, query_min, query_max):
                selected_paths.append(prim_path)
                seen_paths.add(prim_path)

        return selected_paths

    def _get_collision_filter_target_prim_path(
        self,
        target_position: np.ndarray,
        *,
        target_object: str = '',
    ) -> str:
        _, target_entry = self._resolve_target_object_entry(
            target_position,
            target_object=target_object,
        )
        return str((target_entry or {}).get('prim_path', '') or '').strip()

    def _filter_collision_pairs_for_root(
        self,
        filtered_pairs: list[tuple[str, str]],
        root_path: str,
    ) -> list[tuple[str, str]]:
        if not filtered_pairs or not root_path:
            return []
        target_collision_paths = set(
            self._collect_collision_prim_paths(
                root_path,
                include_disabled=True,
            )
        )
        if not target_collision_paths:
            return []
        return [
            (robot_collision_path, object_collision_path)
            for robot_collision_path, object_collision_path in filtered_pairs
            if object_collision_path in target_collision_paths
        ]

    def _remove_auto_arm_obstacles(self, obstacle_keys: list[str]) -> None:
        if self.robot_controller is None or not obstacle_keys:
            return
        if not hasattr(self.robot_controller, 'remove_rmpflow_obstacle'):
            return
        for obstacle_key in obstacle_keys:
            try:
                self.robot_controller.remove_rmpflow_obstacle(obstacle_key)
            except Exception as exc:
                print(f'Auto arm obstacle removal failed for {obstacle_key}: {exc}')

    @staticmethod
    def _collision_api_enabled(prim: Any, UsdPhysics: Any) -> bool:
        """Return whether a CollisionAPI prim is currently enabled.

        In USD Physics, an unauthored ``physics:collisionEnabled`` attribute
        means enabled.  Treat read errors as enabled so diagnostics fail open
        rather than silently hiding collisions.
        """
        try:
            collision_api = UsdPhysics.CollisionAPI(prim)
            attr = collision_api.GetCollisionEnabledAttr()
            if attr is None or not attr.IsValid():
                return True
            raw_value = attr.Get()
            return True if raw_value is None else bool(raw_value)
        except Exception:
            return True

    def _collect_collision_prim_paths(
        self,
        root_path: str,
        *,
        max_count: Optional[int] = None,
        include_disabled: bool = False,
    ) -> list[str]:
        if self.stage is None:
            return []
        try:
            from pxr import UsdPhysics
        except Exception:
            return []

        normalized_root_path = str(root_path or '').strip()
        if not normalized_root_path:
            return []
        root_prim = self.stage.GetPrimAtPath(normalized_root_path)
        if root_prim is None or not root_prim.IsValid():
            return []

        collision_paths: list[str] = []
        for prim in Usd.PrimRange(root_prim):
            if prim is None or not prim.IsValid():
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if not include_disabled and not self._collision_api_enabled(prim, UsdPhysics):
                    continue
                collision_paths.append(str(prim.GetPath()))
                if max_count is not None and len(collision_paths) >= max_count:
                    break
        return collision_paths

    def _collect_active_collision_prim_paths(
        self,
        root_path: str,
        *,
        max_count: Optional[int] = None,
        fallback_to_root: bool = True,
    ) -> list[str]:
        """Collect enabled colliders, falling back to the root only if no colliders exist.

        This is different from ``_collect_collision_prim_paths`` because a root
        may contain CollisionAPI prims whose ``collisionEnabled`` values have
        intentionally been set to False (e.g. grasp-latch robot/object
        colliders). In that case returning the root path would make the
        AABB-based contact monitor appear as if disabling colliders had no
        effect.
        """
        normalized_root_path = str(root_path or '').strip()
        if not normalized_root_path:
            return []

        enabled_paths = self._collect_collision_prim_paths(
            normalized_root_path,
            max_count=max_count,
            include_disabled=False,
        )
        if enabled_paths:
            return enabled_paths

        has_collision_api = bool(
            self._collect_collision_prim_paths(
                normalized_root_path,
                max_count=1,
                include_disabled=True,
            )
        )
        if has_collision_api:
            return []
        return [normalized_root_path] if fallback_to_root else []

    def _collision_query_path_enabled(self, prim_path: str) -> bool:
        """Whether a path should participate in AABB contact queries."""
        normalized_path = str(prim_path or '').strip()
        if not normalized_path or self.stage is None:
            return False
        try:
            from pxr import UsdPhysics
        except Exception:
            return True

        prim = self.stage.GetPrimAtPath(normalized_path)
        if prim is None or not prim.IsValid():
            return False
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return self._collision_api_enabled(prim, UsdPhysics)

        has_collision_api = bool(
            self._collect_collision_prim_paths(
                normalized_path,
                max_count=1,
                include_disabled=True,
            )
        )
        if not has_collision_api:
            return True
        return bool(
            self._collect_collision_prim_paths(
                normalized_path,
                max_count=1,
                include_disabled=False,
            )
        )

    def _iter_arm_collision_root_paths(self) -> list[str]:
        robot_root_path = str(getattr(self.robot_controller, 'robot_prim_path', '') or '').strip()
        if not robot_root_path:
            robot_root_path = str(self.robot_articulation_path or '').strip()
        if not robot_root_path:
            return []

        root_paths: list[str] = []
        for link_name in self.ARM_COLLISION_FILTER_LINK_NAMES:
            link_path = f'{robot_root_path}/{link_name}'
            if self.stage is None:
                continue
            link_prim = self.stage.GetPrimAtPath(link_path)
            if link_prim is not None and link_prim.IsValid():
                root_paths.append(link_path)
        return root_paths

    def _apply_arm_collision_filtering(
        self,
        target_position: np.ndarray,
        *,
        query_padding: Optional[float] = None,
        max_pairs: Optional[int] = None,
        include_target: bool = False,
        target_object: str = '',
    ) -> list[tuple[str, str]]:
        if self.robot_controller is None or self.stage is None:
            return []
        if bool(self.enable_aggressive_lifecycle_collision_filtering):
            self._apply_aggressive_lifecycle_collision_filtering()
            return []
        try:
            from pxr import UsdPhysics
        except Exception as exc:
            print(f'Arm collision filtering unavailable: {exc}')
            return []

        robot_collision_paths: list[str] = []
        seen_robot_collision_paths: set[str] = set()
        for arm_root_path in self._iter_arm_collision_root_paths():
            for collision_path in self._collect_collision_prim_paths(arm_root_path):
                if collision_path in seen_robot_collision_paths:
                    continue
                seen_robot_collision_paths.add(collision_path)
                robot_collision_paths.append(collision_path)
        if not robot_collision_paths:
            return []

        pair_limit = self.DEFAULT_ARM_COLLISION_FILTER_MAX_PAIRS if max_pairs is None else max_pairs
        pair_limit = max(int(pair_limit), 0)
        object_collision_limit = None
        if pair_limit > 0:
            object_collision_limit = max(1, pair_limit // max(len(robot_collision_paths), 1))

        object_collision_paths: list[str] = []
        seen_object_collision_paths: set[str] = set()
        object_prim_paths = self._select_collision_filter_object_prim_paths(
            target_position,
            query_padding=query_padding,
            include_target=include_target,
            target_object=target_object,
        )
        for object_prim_path in object_prim_paths:
            remaining = None
            if object_collision_limit is not None:
                remaining = object_collision_limit - len(object_collision_paths)
                if remaining <= 0:
                    break
            for collision_path in self._collect_collision_prim_paths(object_prim_path, max_count=remaining):
                if collision_path in seen_object_collision_paths:
                    continue
                seen_object_collision_paths.add(collision_path)
                object_collision_paths.append(collision_path)
                if object_collision_limit is not None and len(object_collision_paths) >= object_collision_limit:
                    break
        if not object_collision_paths:
            return []

        added_pairs: list[tuple[str, str]] = []
        for robot_collision_path in robot_collision_paths:
            robot_collision_prim = self.stage.GetPrimAtPath(robot_collision_path)
            if robot_collision_prim is None or not robot_collision_prim.IsValid():
                continue
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(robot_collision_prim)
            rel = filtered_pairs_api.CreateFilteredPairsRel()
            existing_targets = {str(target) for target in rel.GetTargets()}
            for object_collision_path in object_collision_paths:
                if object_collision_path == robot_collision_path or object_collision_path in existing_targets:
                    continue
                if rel.AddTarget(Sdf.Path(object_collision_path)):
                    added_pairs.append((robot_collision_path, object_collision_path))
                    existing_targets.add(object_collision_path)
                    if pair_limit > 0 and len(added_pairs) >= pair_limit:
                        print(f'Arm collision filtering pair limit reached: {pair_limit}')
                        return added_pairs
        return added_pairs

    def _remove_arm_collision_filtering(self, filtered_pairs: list[tuple[str, str]]) -> None:
        if self.stage is None or not filtered_pairs:
            return
        try:
            from pxr import UsdPhysics
        except Exception:
            return

        for robot_collision_path, object_collision_path in filtered_pairs:
            robot_collision_prim = self.stage.GetPrimAtPath(robot_collision_path)
            if robot_collision_prim is None or not robot_collision_prim.IsValid():
                continue
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI(robot_collision_prim)
            rel = filtered_pairs_api.GetFilteredPairsRel()
            if not rel:
                continue
            try:
                rel.RemoveTarget(Sdf.Path(object_collision_path))
                if hasattr(self, '_persistent_arm_collision_filter_pairs'):
                    self._persistent_arm_collision_filter_pairs.discard(
                        (str(robot_collision_path), str(object_collision_path))
                    )
            except Exception as exc:
                print(f'Arm collision filtering removal failed for {robot_collision_path}: {exc}')

    def _remember_persistent_arm_collision_filtering(
        self,
        filtered_pairs: Optional[Sequence[Sequence[str]]],
    ) -> list[tuple[str, str]]:
        normalized_pairs = self._normalize_collision_filter_pairs(filtered_pairs)
        if not normalized_pairs:
            return []
        if not hasattr(self, '_persistent_arm_collision_filter_pairs'):
            self._persistent_arm_collision_filter_pairs = set()
        self._persistent_arm_collision_filter_pairs.update(normalized_pairs)
        return normalized_pairs

    def _clear_persistent_arm_collision_filtering(self, *, remove_pairs: bool = True) -> None:
        pairs = list(getattr(self, '_persistent_arm_collision_filter_pairs', set()) or [])
        if remove_pairs and pairs:
            self._remove_arm_collision_filtering(pairs)
        self._persistent_arm_collision_filter_pairs = set()

    def _persistent_arm_collision_filter_state(self) -> Dict[str, Any]:
        pairs = sorted(getattr(self, '_persistent_arm_collision_filter_pairs', set()) or [])
        return {
            'enabled': bool(getattr(self, 'persist_arm_collision_filtering', False)),
            'active_pair_count': len(pairs),
            'pairs': [
                {
                    'robot_collision_path': str(robot_collision_path),
                    'target_collision_path': str(target_collision_path),
                }
                for robot_collision_path, target_collision_path in pairs
            ],
        }

    @staticmethod
    def _path_is_same_or_nested(path: str, root_path: str) -> bool:
        normalized_path = str(path or '').strip()
        normalized_root = str(root_path or '').strip()
        return bool(
            normalized_path
            and normalized_root
            and (
                normalized_path == normalized_root
                or normalized_path.startswith(f'{normalized_root}/')
            )
        )

    def _get_robot_collision_root_path(self) -> str:
        root_path = str(getattr(self.robot_controller, 'robot_prim_path', '') or '').strip()
        if not root_path:
            root_path = str(self.robot_articulation_path or '').strip()
        if not root_path:
            root_path = str(self.robot_prim_path or '').strip()
        return root_path

    def _collect_robot_collision_prim_paths(self) -> list[str]:
        robot_root_path = self._get_robot_collision_root_path()
        if not robot_root_path:
            return []
        return self._collect_collision_prim_paths(robot_root_path)

    def _normalize_collision_exempt_root_paths(self, root_paths: Sequence[str]) -> set[str]:
        normalized: set[str] = set()
        for root_path in root_paths or []:
            text = str(root_path or '').strip()
            if text:
                normalized.add(text)
        return normalized

    def _room_direct_child_collision_root_paths(
        self,
    ) -> tuple[list[str], list[str]]:
        """Collect direct room children once and partition them by leaf name."""
        if self.stage is None:
            return [], []

        rooms_root = self.stage.GetPrimAtPath(ROOMS_ROOT_PATH)
        if rooms_root is None or not rooms_root.IsValid():
            return [], []

        scene_object_roots: list[str] = []
        structural_roots: list[str] = []
        seen: set[str] = set()
        robot_root_path = self._get_robot_collision_root_path()

        for room_prim in rooms_root.GetChildren():
            if room_prim is None or not room_prim.IsValid():
                continue
            if self._should_ignore_room_prim(room_prim.GetName()):
                continue
            for object_prim in room_prim.GetChildren():
                if object_prim is None or not object_prim.IsValid():
                    continue
                object_name = str(object_prim.GetName() or '').strip()
                object_root_path = str(object_prim.GetPath() or '').strip()
                if not object_name or object_root_path in seen:
                    continue
                collision_group = classify_room_direct_child(object_name, object_root_path)
                if collision_group is None:
                    continue
                if self._is_aggressive_lifecycle_collision_group_path(object_root_path):
                    continue
                if robot_root_path and self._path_is_same_or_nested(object_root_path, robot_root_path):
                    continue
                if not self._collect_collision_prim_paths(
                    object_root_path,
                    max_count=1,
                    include_disabled=True,
                ):
                    continue
                seen.add(object_root_path)
                if collision_group == 'structural':
                    structural_roots.append(object_root_path)
                else:
                    scene_object_roots.append(object_root_path)

        return scene_object_roots, structural_roots

    def _define_aggressive_lifecycle_collision_group(
        self,
        UsdPhysics: Any,
        group_path: str,
    ) -> Any:
        if self.stage is None:
            raise RuntimeError('stage_not_loaded')
        UsdGeom.Scope.Define(
            self.stage,
            Sdf.Path(self.AGGRESSIVE_LIFECYCLE_COLLISION_GROUP_ROOT_PATH),
        )
        return UsdPhysics.CollisionGroup.Define(self.stage, Sdf.Path(group_path))

    @staticmethod
    def _normalized_sdf_paths(paths: Sequence[str]) -> list[Sdf.Path]:
        normalized: list[Sdf.Path] = []
        seen: set[str] = set()
        for raw_path in paths or []:
            text = str(raw_path or '').strip()
            if not text or text in seen:
                continue
            try:
                sdf_path = Sdf.Path(text)
            except Exception:
                continue
            if not sdf_path.IsAbsolutePath():
                continue
            normalized.append(sdf_path)
            seen.add(text)
        return normalized

    def _collision_group_colliders_collection_api(self, collision_group: Any) -> Any:
        try:
            return collision_group.GetCollidersCollectionAPI()
        except Exception:
            return Usd.CollectionAPI.Apply(collision_group.GetPrim(), 'colliders')

    @staticmethod
    def _set_collection_expansion_rule_expand_prims(collection_api: Any) -> None:
        try:
            expansion_rule = getattr(Usd.Tokens, 'expandPrims', 'expandPrims')
            collection_api.CreateExpansionRuleAttr().Set(expansion_rule)
        except Exception:
            pass

    def _collision_group_member_target_paths(
        self,
        root_paths: Sequence[str],
    ) -> tuple[list[str], int]:
        targets: list[str] = []
        seen_targets: set[str] = set()
        seen_colliders: set[str] = set()

        def _add_target(raw_path: str) -> None:
            text = str(raw_path or '').strip()
            if not text or text in seen_targets:
                return
            targets.append(text)
            seen_targets.add(text)

        for root_path in root_paths or []:
            normalized_root = str(root_path or '').strip()
            if not normalized_root:
                continue
            _add_target(normalized_root)
            for collider_path in self._collect_collision_prim_paths(
                normalized_root,
                include_disabled=True,
            ):
                collider_text = str(collider_path or '').strip()
                if not collider_text:
                    continue
                seen_colliders.add(collider_text)
                _add_target(collider_text)

        return targets, len(seen_colliders)

    @staticmethod
    def _set_collision_group_collection_excludes_empty(collection_api: Any) -> None:
        try:
            collection_api.CreateExcludesRel().SetTargets([])
        except Exception:
            pass

    def _set_collision_group_filter_mode(self, UsdPhysics: Any) -> Dict[str, Any]:
        if self.stage is None:
            return {'physics_scene_count': 0, 'authored_scene_count': 0, 'error': 'stage_not_loaded'}
        try:
            from pxr import PhysxSchema
        except Exception as exc:
            return {
                'physics_scene_count': 0,
                'authored_scene_count': 0,
                'error': f'{exc.__class__.__name__}: {exc}',
            }

        physics_scene_count = 0
        authored_scene_count = 0
        errors: list[str] = []
        for prim in self.stage.Traverse():
            try:
                is_physics_scene = bool(prim.IsA(UsdPhysics.Scene))
            except Exception:
                is_physics_scene = False
            if not is_physics_scene:
                continue
            physics_scene_count += 1
            try:
                physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                attr = physx_scene_api.GetInvertCollisionGroupFilterAttr()
                if attr is None or not attr.IsValid():
                    attr = physx_scene_api.CreateInvertCollisionGroupFilterAttr(True)
                attr.Set(True)
                authored_scene_count += 1
            except Exception as exc:
                errors.append(f'{str(prim.GetPath())}:{exc.__class__.__name__}: {exc}')

        return {
            'physics_scene_count': int(physics_scene_count),
            'authored_scene_count': int(authored_scene_count),
            'error': '; '.join(errors[:3]),
        }

    def _set_collision_group_colliders(
        self,
        collision_group: Any,
        collider_root_paths: Sequence[str],
    ) -> Dict[str, int]:
        collection_api = self._collision_group_colliders_collection_api(collision_group)
        self._set_collection_expansion_rule_expand_prims(collection_api)
        self._set_collision_group_collection_excludes_empty(collection_api)
        includes_rel = collection_api.CreateIncludesRel()
        targets, collider_count = self._collision_group_member_target_paths(collider_root_paths)
        sdf_targets = self._normalized_sdf_paths(targets)
        includes_rel.SetTargets(sdf_targets)
        return {
            'root_count': len(self._normalized_sdf_paths(collider_root_paths)),
            'collider_count': int(collider_count),
            'target_count': len(sdf_targets),
        }

    def _set_collision_group_filtered_groups(
        self,
        collision_group: Any,
        filtered_group_paths: Sequence[str],
    ) -> int:
        filtered_groups_rel = collision_group.CreateFilteredGroupsRel()
        targets = self._normalized_sdf_paths(filtered_group_paths)
        filtered_groups_rel.SetTargets(targets)
        return len(targets)

    def _remove_aggressive_lifecycle_collision_groups(self) -> None:
        if self.stage is None:
            return
        root_path = self.AGGRESSIVE_LIFECYCLE_COLLISION_GROUP_ROOT_PATH
        root_prim = self.stage.GetPrimAtPath(root_path)
        if root_prim is not None and root_prim.IsValid():
            try:
                self.stage.RemovePrim(root_path)
            except Exception as exc:
                print(f'Aggressive lifecycle collision group cleanup failed: {exc}')

    def _remove_aggressive_lifecycle_collision_pairs_for_roots(self, root_paths: Sequence[str]) -> int:
        if self.stage is None or not self._aggressive_lifecycle_collision_filter_pairs:
            return 0
        roots = self._normalize_collision_exempt_root_paths(root_paths)
        if not roots:
            return 0

        pairs_to_remove = [
            pair
            for pair in self._aggressive_lifecycle_collision_filter_pairs
            if any(
                self._path_is_same_or_nested(pair[1], root)
                or self._path_is_same_or_nested(root, pair[1])
                for root in roots
            )
        ]
        if not pairs_to_remove:
            return 0

        self._remove_arm_collision_filtering(pairs_to_remove)
        for pair in pairs_to_remove:
            self._aggressive_lifecycle_collision_filter_pairs.discard(pair)
        return len(pairs_to_remove)

    def _clear_aggressive_lifecycle_collision_filtering(self, *, remove_pairs: bool = True) -> None:
        if remove_pairs and self._aggressive_lifecycle_collision_filter_pairs:
            self._remove_arm_collision_filtering(list(self._aggressive_lifecycle_collision_filter_pairs))
        if remove_pairs:
            self._remove_aggressive_lifecycle_collision_groups()
        self._aggressive_lifecycle_collision_filter_pairs = set()
        self._aggressive_lifecycle_collision_exempt_root_paths = set()
        self._aggressive_lifecycle_collision_group_robot_root_paths = set()
        self._aggressive_lifecycle_collision_group_object_root_paths = set()
        self._aggressive_lifecycle_collision_group_structural_root_paths = set()
        self._aggressive_lifecycle_collision_group_filtered_targets = set()
        self._aggressive_lifecycle_collision_group_robot_collider_count = 0
        self._aggressive_lifecycle_collision_group_object_collider_count = 0
        self._aggressive_lifecycle_collision_group_structural_collider_count = 0
        self._aggressive_lifecycle_collision_group_robot_target_count = 0
        self._aggressive_lifecycle_collision_group_object_target_count = 0
        self._aggressive_lifecycle_collision_group_structural_target_count = 0
        self._aggressive_lifecycle_collision_group_physics_scene_count = 0
        self._aggressive_lifecycle_collision_filter_pre_reset_applied = False
        self._aggressive_lifecycle_collision_filter_last_error = ''

    def _set_aggressive_lifecycle_collision_exempt_roots(self, root_paths: Sequence[str]) -> Dict[str, Any]:
        if not bool(self.enable_aggressive_lifecycle_collision_filtering):
            return {
                'enabled': False,
                'strategy': 'collision_groups',
                'exempt_root_paths': [],
                'removed_pair_count': 0,
                'active_pair_count': 0,
                'active_group_count': 0,
            }

        normalized_roots = self._normalize_collision_exempt_root_paths(root_paths)
        removed_roots = self._aggressive_lifecycle_collision_exempt_root_paths - normalized_roots
        added_roots = normalized_roots - self._aggressive_lifecycle_collision_exempt_root_paths
        self._aggressive_lifecycle_collision_exempt_root_paths = set(normalized_roots)
        state = self._apply_aggressive_lifecycle_collision_filtering()
        return {
            'enabled': True,
            'strategy': 'collision_groups',
            'exempt_root_paths': sorted(self._aggressive_lifecycle_collision_exempt_root_paths),
            'exemptions_honored': False,
            'added_exempt_root_paths': sorted(added_roots),
            'removed_exempt_root_paths': sorted(removed_roots),
            'removed_pair_count': 0,
            'active_pair_count': 0,
            'active_group_count': int(state.get('active_group_count', 0) or 0),
            'robot_root_count': int(state.get('robot_root_count', 0) or 0),
            'scene_object_root_count': int(state.get('scene_object_root_count', 0) or 0),
            'structural_root_count': int(state.get('structural_root_count', 0) or 0),
            'robot_collider_count': int(state.get('robot_collider_count', 0) or 0),
            'scene_object_collider_count': int(state.get('scene_object_collider_count', 0) or 0),
            'structural_collider_count': int(state.get('structural_collider_count', 0) or 0),
            'robot_group_target_count': int(state.get('robot_group_target_count', 0) or 0),
            'scene_object_group_target_count': int(state.get('scene_object_group_target_count', 0) or 0),
            'structural_group_target_count': int(state.get('structural_group_target_count', 0) or 0),
            'filtered_group_count': int(state.get('filtered_group_count', 0) or 0),
            'physics_scene_count': int(state.get('physics_scene_count', 0) or 0),
            'authored_physics_scene_count': int(state.get('authored_physics_scene_count', 0) or 0),
            'last_error': str(state.get('error', '') or ''),
        }

    def _aggressive_lifecycle_collision_exempt_roots_for_target(
        self,
        target_prim_path: str = '',
    ) -> list[str]:
        roots: list[str] = []
        normalized_target = str(target_prim_path or '').strip()
        if normalized_target:
            roots.append(normalized_target)
        if self.object_in_gripper and self.grasped_object_name:
            held_prim_path = self._lookup_object_prim_path(self.grasped_object_name)
            if held_prim_path:
                roots.append(held_prim_path)
        return roots

    def _apply_aggressive_lifecycle_collision_filtering(self) -> Dict[str, Any]:
        if not bool(self.enable_aggressive_lifecycle_collision_filtering):
            return {
                'enabled': False,
                'strategy': 'collision_groups',
                'active_pair_count': 0,
                'active_group_count': 0,
                'added_pair_count': 0,
                'pre_reset_applied': False,
                'error': '',
            }
        if self.robot_controller is None or self.stage is None:
            return {
                'enabled': True,
                'strategy': 'collision_groups',
                'active_pair_count': 0,
                'active_group_count': 0,
                'added_pair_count': 0,
                'pre_reset_applied': bool(self._aggressive_lifecycle_collision_filter_pre_reset_applied),
                'error': 'robot_or_stage_not_ready',
            }

        try:
            from pxr import UsdPhysics
        except Exception as exc:
            self._aggressive_lifecycle_collision_filter_last_error = f'{exc.__class__.__name__}: {exc}'
            return {
                'enabled': True,
                'strategy': 'collision_groups',
                'active_pair_count': 0,
                'active_group_count': 0,
                'added_pair_count': 0,
                'pre_reset_applied': bool(self._aggressive_lifecycle_collision_filter_pre_reset_applied),
                'error': self._aggressive_lifecycle_collision_filter_last_error,
            }

        robot_root_path = self._get_robot_collision_root_path()
        robot_root_paths = [robot_root_path] if robot_root_path else []
        object_root_paths, structural_root_paths = (
            self._room_direct_child_collision_root_paths()
        )

        previous_robot_roots = set(self._aggressive_lifecycle_collision_group_robot_root_paths)
        previous_object_roots = set(self._aggressive_lifecycle_collision_group_object_root_paths)
        previous_structural_roots = set(self._aggressive_lifecycle_collision_group_structural_root_paths)
        try:
            with Usd.EditContext(self.stage, Usd.EditTarget(self.stage.GetRootLayer())):
                robot_group = self._define_aggressive_lifecycle_collision_group(
                    UsdPhysics,
                    self.AGGRESSIVE_LIFECYCLE_ROBOT_COLLISION_GROUP_PATH,
                )
                scene_object_group = self._define_aggressive_lifecycle_collision_group(
                    UsdPhysics,
                    self.AGGRESSIVE_LIFECYCLE_SCENE_OBJECT_COLLISION_GROUP_PATH,
                )
                structural_group = self._define_aggressive_lifecycle_collision_group(
                    UsdPhysics,
                    self.AGGRESSIVE_LIFECYCLE_STRUCTURAL_COLLISION_GROUP_PATH,
                )
                filter_mode_state = self._set_collision_group_filter_mode(UsdPhysics)
                robot_membership = self._set_collision_group_colliders(robot_group, robot_root_paths)
                scene_object_membership = self._set_collision_group_colliders(
                    scene_object_group,
                    object_root_paths,
                )
                structural_membership = self._set_collision_group_colliders(
                    structural_group,
                    structural_root_paths,
                )
                # In PhysX inverted collision-group-filter mode, filteredGroups
                # acts as an allow-list. The robot group deliberately allows no
                # room-child group, disabling robot contacts with both ordinary
                # scene objects and structural objects.
                robot_filtered_count = self._set_collision_group_filtered_groups(
                    robot_group,
                    [],
                )
                scene_filtered_count = self._set_collision_group_filtered_groups(
                    scene_object_group,
                    [
                        self.AGGRESSIVE_LIFECYCLE_SCENE_OBJECT_COLLISION_GROUP_PATH,
                        self.AGGRESSIVE_LIFECYCLE_STRUCTURAL_COLLISION_GROUP_PATH,
                    ],
                )
                structural_filtered_count = self._set_collision_group_filtered_groups(
                    structural_group,
                    [
                        self.AGGRESSIVE_LIFECYCLE_STRUCTURAL_COLLISION_GROUP_PATH,
                        self.AGGRESSIVE_LIFECYCLE_SCENE_OBJECT_COLLISION_GROUP_PATH,
                    ],
                )
        except Exception as exc:
            self._aggressive_lifecycle_collision_group_robot_root_paths = previous_robot_roots
            self._aggressive_lifecycle_collision_group_object_root_paths = previous_object_roots
            self._aggressive_lifecycle_collision_group_structural_root_paths = previous_structural_roots
            self._aggressive_lifecycle_collision_filter_last_error = (
                f'{exc.__class__.__name__}: {exc}'
            )
            return {
                'enabled': True,
                'strategy': 'collision_groups',
                'active_pair_count': 0,
                'active_group_count': 0,
                'added_pair_count': 0,
                'robot_root_count': len(previous_robot_roots),
                'scene_object_root_count': len(previous_object_roots),
                'structural_root_count': len(previous_structural_roots),
                'robot_collider_count': int(self._aggressive_lifecycle_collision_group_robot_collider_count),
                'scene_object_collider_count': int(self._aggressive_lifecycle_collision_group_object_collider_count),
                'structural_collider_count': int(self._aggressive_lifecycle_collision_group_structural_collider_count),
                'robot_group_target_count': int(self._aggressive_lifecycle_collision_group_robot_target_count),
                'scene_object_group_target_count': int(self._aggressive_lifecycle_collision_group_object_target_count),
                'structural_group_target_count': int(self._aggressive_lifecycle_collision_group_structural_target_count),
                'filtered_group_count': len(self._aggressive_lifecycle_collision_group_filtered_targets),
                'exempt_root_paths': sorted(self._aggressive_lifecycle_collision_exempt_root_paths),
                'exemptions_honored': False,
                'pre_reset_applied': bool(self._aggressive_lifecycle_collision_filter_pre_reset_applied),
                'error': self._aggressive_lifecycle_collision_filter_last_error,
            }

        self._aggressive_lifecycle_collision_group_robot_root_paths = set(robot_root_paths)
        self._aggressive_lifecycle_collision_group_object_root_paths = set(object_root_paths)
        self._aggressive_lifecycle_collision_group_structural_root_paths = set(structural_root_paths)
        self._aggressive_lifecycle_collision_group_robot_collider_count = int(
            robot_membership.get('collider_count', 0) or 0
        )
        self._aggressive_lifecycle_collision_group_object_collider_count = int(
            scene_object_membership.get('collider_count', 0) or 0
        )
        self._aggressive_lifecycle_collision_group_structural_collider_count = int(
            structural_membership.get('collider_count', 0) or 0
        )
        self._aggressive_lifecycle_collision_group_robot_target_count = int(
            robot_membership.get('target_count', 0) or 0
        )
        self._aggressive_lifecycle_collision_group_object_target_count = int(
            scene_object_membership.get('target_count', 0) or 0
        )
        self._aggressive_lifecycle_collision_group_structural_target_count = int(
            structural_membership.get('target_count', 0) or 0
        )
        self._aggressive_lifecycle_collision_group_physics_scene_count = int(
            filter_mode_state.get('physics_scene_count', 0) or 0
        )
        self._aggressive_lifecycle_collision_group_filtered_targets = {
            self.AGGRESSIVE_LIFECYCLE_SCENE_OBJECT_COLLISION_GROUP_PATH,
            self.AGGRESSIVE_LIFECYCLE_STRUCTURAL_COLLISION_GROUP_PATH,
        }
        active_group_count = int(robot_membership.get('target_count', 0) > 0) + int(
            scene_object_membership.get('target_count', 0) > 0
        ) + int(
            structural_membership.get('target_count', 0) > 0
        )
        filtered_group_count = (
            int(scene_filtered_count)
            + int(robot_filtered_count)
            + int(structural_filtered_count)
        )
        filter_mode_error = str(filter_mode_state.get('error', '') or '')
        self._aggressive_lifecycle_collision_filter_last_error = filter_mode_error
        return {
            'enabled': True,
            'strategy': 'collision_groups',
            'active_pair_count': 0,
            'active_group_count': active_group_count,
            'added_pair_count': 0,
            'robot_root_count': int(robot_membership.get('root_count', 0) or 0),
            'scene_object_root_count': int(scene_object_membership.get('root_count', 0) or 0),
            'structural_root_count': int(structural_membership.get('root_count', 0) or 0),
            'robot_collider_count': int(robot_membership.get('collider_count', 0) or 0),
            'scene_object_collider_count': int(scene_object_membership.get('collider_count', 0) or 0),
            'structural_collider_count': int(structural_membership.get('collider_count', 0) or 0),
            'robot_group_target_count': int(robot_membership.get('target_count', 0) or 0),
            'scene_object_group_target_count': int(scene_object_membership.get('target_count', 0) or 0),
            'structural_group_target_count': int(structural_membership.get('target_count', 0) or 0),
            'filtered_group_count': int(filtered_group_count),
            'physics_scene_count': int(filter_mode_state.get('physics_scene_count', 0) or 0),
            'authored_physics_scene_count': int(filter_mode_state.get('authored_scene_count', 0) or 0),
            'invert_collision_group_filter': True,
            'pre_reset_applied': bool(self._aggressive_lifecycle_collision_filter_pre_reset_applied),
            'robot_group_path': self.AGGRESSIVE_LIFECYCLE_ROBOT_COLLISION_GROUP_PATH,
            'scene_object_group_path': self.AGGRESSIVE_LIFECYCLE_SCENE_OBJECT_COLLISION_GROUP_PATH,
            'structural_group_path': self.AGGRESSIVE_LIFECYCLE_STRUCTURAL_COLLISION_GROUP_PATH,
            'robot_scene_object_collision_enabled': False,
            'robot_structural_collision_enabled': False,
            'exempt_root_paths': sorted(self._aggressive_lifecycle_collision_exempt_root_paths),
            'exemptions_honored': False,
            'error': filter_mode_error,
        }

    def _aggressive_lifecycle_collision_filter_state(self) -> Dict[str, Any]:
        return {
            'enabled': bool(self.enable_aggressive_lifecycle_collision_filtering),
            'strategy': 'collision_groups',
            'active_pair_count': 0,
            'active_group_count': (
                int(bool(self._aggressive_lifecycle_collision_group_robot_root_paths))
                + int(bool(self._aggressive_lifecycle_collision_group_object_root_paths))
                + int(bool(self._aggressive_lifecycle_collision_group_structural_root_paths))
            ),
            'robot_root_count': len(self._aggressive_lifecycle_collision_group_robot_root_paths),
            'scene_object_root_count': len(self._aggressive_lifecycle_collision_group_object_root_paths),
            'structural_root_count': len(self._aggressive_lifecycle_collision_group_structural_root_paths),
            'robot_collider_count': int(self._aggressive_lifecycle_collision_group_robot_collider_count),
            'scene_object_collider_count': int(self._aggressive_lifecycle_collision_group_object_collider_count),
            'structural_collider_count': int(self._aggressive_lifecycle_collision_group_structural_collider_count),
            'robot_group_target_count': int(self._aggressive_lifecycle_collision_group_robot_target_count),
            'scene_object_group_target_count': int(self._aggressive_lifecycle_collision_group_object_target_count),
            'structural_group_target_count': int(self._aggressive_lifecycle_collision_group_structural_target_count),
            'filtered_group_count': len(self._aggressive_lifecycle_collision_group_filtered_targets),
            'physics_scene_count': int(self._aggressive_lifecycle_collision_group_physics_scene_count),
            'invert_collision_group_filter': True,
            'pre_reset_applied': bool(self._aggressive_lifecycle_collision_filter_pre_reset_applied),
            'robot_group_path': self.AGGRESSIVE_LIFECYCLE_ROBOT_COLLISION_GROUP_PATH,
            'scene_object_group_path': self.AGGRESSIVE_LIFECYCLE_SCENE_OBJECT_COLLISION_GROUP_PATH,
            'structural_group_path': self.AGGRESSIVE_LIFECYCLE_STRUCTURAL_COLLISION_GROUP_PATH,
            'robot_scene_object_collision_enabled': False,
            'robot_structural_collision_enabled': False,
            'exempt_root_paths': sorted(self._aggressive_lifecycle_collision_exempt_root_paths),
            'exemptions_honored': False,
            'last_error': self._aggressive_lifecycle_collision_filter_last_error,
        }

    def _get_end_effector_pose_raw(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.robot_controller is None:
            return None
        try:
            position, orientation = self.robot_controller.get_end_effector_pose()
            return np.array(position, dtype=float), np.array(orientation, dtype=float)
        except Exception:
            return None

    def _get_end_effector_pose(self) -> Optional[Dict[str, Dict[str, float]]]:
        pose = self._get_end_effector_pose_raw()
        if pose is None:
            return None
        position, orientation = pose
        return self._pose_dict_from_raw(position, orientation)

    def _get_robot_base_pose_raw(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.robot_controller is None:
            return None
        try:
            position, orientation = self.robot_controller.get_base_link_pose()
            return np.array(position, dtype=float), np.array(orientation, dtype=float)
        except Exception:
            try:
                position, orientation = self.robot_controller.get_world_pose()
                return np.array(position, dtype=float), np.array(orientation, dtype=float)
            except Exception:
                return None

    def _get_joint_states(self) -> Dict[str, Any]:
        if self.robot_controller is None:
            return {}

        try:
            joint_positions = self.robot_controller.get_joint_positions()
        except Exception:
            return {}

        payload = {
            'joint_positions': [float(value) for value in np.array(joint_positions).tolist()],
        }

        try:
            joint_names = list(self.robot_controller.get_joint_names() or [])
            if joint_names:
                payload['joint_names'] = joint_names
        except Exception:
            pass

        try:
            payload['gripper_joint_positions'] = [
                float(value)
                for value in np.array(self.robot_controller.get_gripper_joint_positions()).tolist()
            ]
        except Exception:
            pass

        return payload

    def _orientation_to_quaternion(self, orientation: Any) -> np.ndarray:
        if isinstance(orientation, dict):
            if {'w', 'x', 'y', 'z'}.issubset(set(orientation.keys())):
                return np.array(
                    [
                        float(orientation.get('w', 1.0)),
                        float(orientation.get('x', 0.0)),
                        float(orientation.get('y', 0.0)),
                        float(orientation.get('z', 0.0)),
                    ],
                    dtype=float,
                )
            return euler_angles_to_quats(
                np.array(
                    [
                        float(orientation.get('roll', 0.0)),
                        float(orientation.get('pitch', 0.0)),
                        float(orientation.get('yaw', 0.0)),
                    ],
                    dtype=float,
                )
            )

        if isinstance(orientation, (list, tuple, np.ndarray)):
            raw = np.array(orientation, dtype=float)
            if raw.size >= 4:
                return raw[:4]
            if raw.size == 3:
                return euler_angles_to_quats(raw)

        return euler_angles_to_quats(np.array([0.0, 0.0, 0.0], dtype=float))

    def _pose_command_to_raw(
        self,
        pose: Dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
        payload = dict(pose or {})
        position_dict = dict(payload.get('position') or {})
        target_position = np.array(
            [
                float(position_dict.get('x', 0.0)),
                float(position_dict.get('y', 0.0)),
                float(position_dict.get('z', 0.0)),
            ],
            dtype=float,
        )
        target_orientation = self._orientation_to_quaternion(payload.get('orientation') or {})
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Gimbal lock detected.*',
                category=UserWarning,
            )
            normalized_pose = self._pose_dict_from_raw(target_position, target_orientation)
        return target_position, target_orientation, normalized_pose

    @staticmethod
    def _quaternion_angular_distance(q1: np.ndarray, q2: np.ndarray) -> float:
        q1 = np.array(q1, dtype=float)
        q2 = np.array(q2, dtype=float)
        if np.linalg.norm(q1) == 0.0 or np.linalg.norm(q2) == 0.0:
            return float('inf')
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
        return float(2.0 * np.arccos(dot))

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        lowered = str(value).strip().lower()
        if lowered in {'1', 'true', 'yes', 'y', 'on', 'enable', 'enabled'}:
            return True
        if lowered in {'0', 'false', 'no', 'n', 'off', 'disable', 'disabled'}:
            return False
        return bool(default)

    @staticmethod
    def _rotate_vector_by_quaternion(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        quat = np.array(quaternion, dtype=float)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-12:
            return np.array(vector, dtype=float)
        w, x, y, z = quat / norm
        rotation = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )
        return rotation @ np.array(vector, dtype=float)

    def _approach_direction_from_orientation(self, target_orientation: np.ndarray) -> np.ndarray:
        direction = self._rotate_vector_by_quaternion(
            np.array([1.0, 0.0, 0.0], dtype=float),
            target_orientation,
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12 or not np.all(np.isfinite(direction)):
            return np.array([0.0, 0.0, -1.0], dtype=float)
        return direction / norm

    @staticmethod
    def _wrap_to_pi(value: float) -> float:
        return float((value + np.pi) % (2.0 * np.pi) - np.pi)

    def _yaw_from_quaternion(self, orientation: np.ndarray) -> float:
        euler = quats_to_euler_angles(np.array(orientation, dtype=float))
        return float(euler[2])

    def _step_robot(self, *, render: bool, gripper_action: Optional[str] = None) -> None:
        if self.my_world is None:
            self.simulation_app.update()
            return

        if self.robot_controller is not None and self._active_navigation_target is not None:
            try:
                self.robot_controller.navigate(step_size=float(self.DEFAULT_PHYSICS_DT))
            except Exception as exc:
                print(f'Navigation step failed: {exc}')

        if self.robot_controller is not None and self._active_arm_target is not None:
            try:
                target_position, target_orientation = self._active_arm_target
                self.robot_controller.move_end_effector(target_position, target_orientation)
            except Exception as exc:
                print(f'End-effector control step failed: {exc}')

        action_name = gripper_action or self._active_gripper_action
        if self.robot_controller is not None and action_name:
            try:
                if action_name == 'open':
                    self.robot_controller.open_gripper(step_size=None)
                else:
                    self.robot_controller.close_gripper(step_size=None)
            except Exception as exc:
                print(f'Gripper step failed: {exc}')

        self._update_grasp_pose_latch()
        self.my_world.step(render=render)
        self._update_grasp_pose_latch()
        if not bool(self.runtime_safety_monitor._callback_registered):
            self.runtime_safety_monitor.on_physics_step(float(self.DEFAULT_PHYSICS_DT))

    def _wait_for_motion(
        self,
        *,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        timeout_sec: float,
        position_tolerance: float,
        orientation_tolerance: float,
    ) -> Dict[str, Any]:
        start = time.monotonic()
        reached = False
        distance = None
        orientation_distance = None
        self._active_arm_target = (target_position, target_orientation)

        try:
            while time.monotonic() - start < timeout_sec:
                self._step_robot(render=True)
                ee_pose = self._get_end_effector_pose_raw()
                if ee_pose is None:
                    continue

                ee_position, ee_orientation = ee_pose
                distance = float(np.linalg.norm(ee_position - target_position))
                orientation_distance = self._quaternion_angular_distance(
                    ee_orientation,
                    target_orientation,
                )
                if distance <= position_tolerance and orientation_distance <= orientation_tolerance:
                    reached = True
                    break
        finally:
            self._active_arm_target = None

        return {
            'reached': reached,
            'distance': distance,
            'orientation_distance': orientation_distance,
            'elapsed_sec': float(time.monotonic() - start),
        }

    def _wait_for_pre_grasp_motion(
        self,
        *,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        timeout_sec: float,
        position_tolerance: float,
        orientation_tolerance: float,
        pre_grasp_distance: float,
        linear_step: float,
    ) -> Dict[str, Any]:
        start = time.monotonic()
        pre_grasp_distance = max(float(pre_grasp_distance), 0.0)
        linear_step = max(float(linear_step), 1e-3)
        approach_direction = self._approach_direction_from_orientation(target_orientation)
        pre_grasp_position = target_position - approach_direction * pre_grasp_distance
        stage_results: list[Dict[str, Any]] = []

        pre_motion = self._wait_for_motion(
            target_position=pre_grasp_position,
            target_orientation=target_orientation,
            timeout_sec=timeout_sec,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )
        stage_results.append({'stage': 'pre_grasp', **pre_motion})
        if not pre_motion['reached']:
            return {
                'reached': False,
                'distance': pre_motion['distance'],
                'orientation_distance': pre_motion['orientation_distance'],
                'elapsed_sec': float(time.monotonic() - start),
                'pre_grasp_enabled': True,
                'pre_grasp_reached': False,
                'pre_grasp_position': pre_grasp_position,
                'approach_direction': approach_direction,
                'linear_waypoint_count': 0,
                'stage_results': stage_results,
            }

        waypoint_count = max(1, int(np.ceil(pre_grasp_distance / linear_step)))
        final_motion = pre_motion
        for index in range(1, waypoint_count + 1):
            alpha = float(index / waypoint_count)
            waypoint_position = pre_grasp_position + (target_position - pre_grasp_position) * alpha
            final_motion = self._wait_for_motion(
                target_position=waypoint_position,
                target_orientation=target_orientation,
                timeout_sec=timeout_sec,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            )
            stage_results.append(
                {
                    'stage': 'linear_approach',
                    'waypoint_index': index,
                    'waypoint_count': waypoint_count,
                    **final_motion,
                }
            )
            if not final_motion['reached']:
                break

        return {
            'reached': bool(final_motion['reached']),
            'distance': final_motion['distance'],
            'orientation_distance': final_motion['orientation_distance'],
            'elapsed_sec': float(time.monotonic() - start),
            'pre_grasp_enabled': True,
            'pre_grasp_reached': bool(pre_motion['reached']),
            'pre_grasp_position': pre_grasp_position,
            'approach_direction': approach_direction,
            'linear_waypoint_count': waypoint_count,
            'stage_results': stage_results,
        }

    def _wait_for_gripper(
        self,
        *,
        gripper_action: str,
        timeout_sec: float,
        tolerance: float,
    ) -> Dict[str, Any]:
        start = time.monotonic()
        settled = False
        current_positions = []
        if self.robot_controller is None:
            return {'settled': False, 'elapsed_sec': 0.0, 'joint_positions': current_positions}

        target_positions = np.array(
            self.robot_controller.gripper_open_positions
            if gripper_action == 'open'
            else self.robot_controller.gripper_closed_positions,
            dtype=float,
        )
        self._active_gripper_action = gripper_action

        try:
            while time.monotonic() - start < timeout_sec:
                self._step_robot(render=True)
                try:
                    current = np.array(self.robot_controller.get_gripper_joint_positions(), dtype=float)
                except Exception:
                    continue

                current_positions = [float(value) for value in current.tolist()]
                if np.allclose(current, target_positions, atol=tolerance):
                    settled = True
                    break
        finally:
            self._active_gripper_action = None

        return {
            'settled': settled,
            'elapsed_sec': float(time.monotonic() - start),
            'joint_positions': current_positions,
        }

    def _infer_object_in_gripper(self, joint_positions: list[float]) -> Dict[str, Any]:
        if self.robot_controller is None:
            raise RuntimeError('robot_not_loaded')

        current = np.array(joint_positions, dtype=float)
        closed_positions = np.array(self.robot_controller.gripper_closed_positions, dtype=float)
        if current.size != closed_positions.size:
            return {
                'object_in_gripper': False,
                'max_closure_residual': None,
            }

        closure_residual = np.maximum(current - closed_positions, 0.0)
        max_closure_residual = float(np.max(closure_residual)) if closure_residual.size else 0.0
        object_in_gripper = bool(
            np.any(closure_residual > float(self.DEFAULT_GRIPPER_OBJECT_DETECTION_THRESHOLD))
        )
        return {
            'object_in_gripper': object_in_gripper,
            'max_closure_residual': max_closure_residual,
        }

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        quat = np.array(quaternion, dtype=float).reshape(-1)
        if quat.size < 4:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        quat = quat[:4]
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-12 or not np.isfinite(norm):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return quat / norm

    @classmethod
    def _quaternion_inverse(cls, quaternion: np.ndarray) -> np.ndarray:
        quat = cls._normalize_quaternion(quaternion)
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)

    @classmethod
    def _quaternion_multiply(cls, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = cls._normalize_quaternion(lhs)
        w2, x2, y2, z2 = cls._normalize_quaternion(rhs)
        return cls._normalize_quaternion(
            np.array(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
                dtype=float,
            )
        )

    def _set_prim_world_pose_raw(
        self,
        prim_path: str,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> bool:
        normalized_path = self._resolve_pose_prim_path(prim_path)
        if not normalized_path or self.stage is None:
            return False

        position_array = np.array(position, dtype=float)
        orientation_array = self._normalize_quaternion(np.array(orientation, dtype=float))
        prim = self.stage.GetPrimAtPath(normalized_path)
        if prim is None or not prim.IsValid():
            return False
        try:
            desired_world = self._matrix_from_pose_raw(position_array, orientation_array)
            parent_world = Gf.Matrix4d(1.0)
            parent = prim.GetParent()
            if parent is not None and parent.IsValid():
                parent_xformable = UsdGeom.Xformable(parent)
                if parent_xformable:
                    parent_world = parent_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

            local_matrix = desired_world * parent_world.GetInverse()
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                return False
            transform_attr = prim.GetAttribute('xformOp:transform')
            if transform_attr is not None and transform_attr.IsValid():
                transform_op = UsdGeom.XformOp(transform_attr)
            else:
                transform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
            xformable.SetXformOpOrder([transform_op])
            transform_op.Set(local_matrix)
            actual_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            actual_position = self._transform_point_to_world(actual_world, [0.0, 0.0, 0.0])
            if float(np.linalg.norm(actual_position - position_array[:3])) > 1e-3:
                transform_op.Set(parent_world.GetInverse() * desired_world)
            return True
        except Exception as exc:
            print(f'[Grasp latch] failed to set object pose prim={normalized_path}: {exc}')
            return False

    @staticmethod
    def _matrix_from_pose_raw(position: np.ndarray, orientation: np.ndarray):
        position_array = np.array(position, dtype=float)
        quat = np.array(orientation, dtype=float)
        rotation = Gf.Rotation(
            Gf.Quatd(
                float(quat[0]),
                Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3])),
            )
        )
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(rotation)
        matrix.SetTranslateOnly(
            Gf.Vec3d(
                float(position_array[0]),
                float(position_array[1]),
                float(position_array[2]),
            )
        )
        return matrix

    def _disable_object_collisions(self, root_prim_path: str) -> Dict[str, bool]:
        """Disable Usd CollisionAPI on the object prim and all its descendants.

        Returns a mapping prim_path -> previous_collision_enabled_value used
        later by `_restore_object_collisions` to put each prim back the way it
        was. While the latch holds the object, having no collision at all
        avoids any contact with the world (and so is a simpler, more
        deterministic alternative to enumerating collision filter pairs).
        """
        if not root_prim_path or self.stage is None:
            return {}
        try:
            from pxr import UsdPhysics
        except Exception:
            return {}

        root_prim = self.stage.GetPrimAtPath(root_prim_path)
        if root_prim is None or not root_prim.IsValid():
            return {}

        previous_state: Dict[str, bool] = {}
        for prim in Usd.PrimRange(root_prim):
            if prim is None or not prim.IsValid():
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            try:
                collision_api = UsdPhysics.CollisionAPI(prim)
                attr = collision_api.GetCollisionEnabledAttr()
                previous_value: Optional[bool] = None
                if attr is not None and attr.IsValid():
                    try:
                        raw = attr.Get()
                        previous_value = True if raw is None else bool(raw)
                    except Exception:
                        previous_value = None
                if attr is None or not attr.IsValid():
                    attr = collision_api.CreateCollisionEnabledAttr()
                    # Default for the attribute is True when unauthored.
                    if previous_value is None:
                        previous_value = True
                attr.Set(False)
                # Default to True if we couldn't read a value — that matches
                # Usd's documented default for CollisionEnabled.
                previous_state[str(prim.GetPath())] = (
                    True if previous_value is None else bool(previous_value)
                )
            except Exception as exc:
                print(
                    f'[Grasp latch] failed to disable collision on prim={prim.GetPath()}: {exc}'
                )
        return previous_state

    def _restore_object_collisions(self, previous_state: Optional[Dict[str, bool]]) -> None:
        if not previous_state or self.stage is None:
            return
        try:
            from pxr import UsdPhysics
        except Exception:
            return
        for prim_path, was_enabled in dict(previous_state).items():
            prim = self.stage.GetPrimAtPath(str(prim_path))
            if prim is None or not prim.IsValid():
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            try:
                collision_api = UsdPhysics.CollisionAPI(prim)
                attr = collision_api.GetCollisionEnabledAttr()
                if attr is None or not attr.IsValid():
                    attr = collision_api.CreateCollisionEnabledAttr()
                attr.Set(bool(was_enabled))
            except Exception as exc:
                print(
                    f'[Grasp latch] failed to restore collision on prim={prim_path}: {exc}'
                )

    def _set_rigid_body_kinematic(self, prim_path: str, enabled: bool) -> Optional[bool]:
        if not prim_path or self.stage is None:
            return None
        try:
            from pxr import UsdPhysics
        except Exception:
            return None

        prim = self.stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            return None
        try:
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            attr = rigid_body_api.GetKinematicEnabledAttr()
            previous_value = None
            if attr is not None and attr.IsValid():
                try:
                    previous_value = attr.Get()
                except Exception:
                    previous_value = None
            if attr is None or not attr.IsValid():
                attr = rigid_body_api.CreateKinematicEnabledAttr()
            attr.Set(bool(enabled))
            if enabled:
                self._force_rigid_body_latch_controls(prim_path)
            return None if previous_value is None else bool(previous_value)
        except Exception as exc:
            print(f'[Grasp latch] failed to set kinematic prim={prim_path}: {exc}')
            return None

    def _force_rigid_body_latch_controls(self, prim_path: str) -> None:
        if not prim_path or self.stage is None:
            return
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            return

        try:
            from pxr import UsdPhysics

            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
            kinematic_attr = rigid_body_api.GetKinematicEnabledAttr()
            if kinematic_attr is None or not kinematic_attr.IsValid():
                kinematic_attr = rigid_body_api.CreateKinematicEnabledAttr()
            kinematic_attr.Set(True)

            velocity_attr = rigid_body_api.GetVelocityAttr()
            if velocity_attr is None or not velocity_attr.IsValid():
                velocity_attr = rigid_body_api.CreateVelocityAttr()
            velocity_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))

            angular_velocity_attr = rigid_body_api.GetAngularVelocityAttr()
            if angular_velocity_attr is None or not angular_velocity_attr.IsValid():
                angular_velocity_attr = rigid_body_api.CreateAngularVelocityAttr()
            angular_velocity_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        except Exception:
            pass

    def _restore_rigid_body_latch_controls(self, prim_path: str) -> None:
        if not prim_path or self.stage is None:
            return
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim is None or not prim.IsValid():
            return
        try:
            from pxr import PhysxSchema

            # Re-enable gravity. Apply the API first so the attribute exists
            # even on prims that never had PhysxRigidBodyAPI authored, then
            # explicitly set DisableGravity=False so the body falls naturally
            # once the latch lets go of it.
            physx_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                physx_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            disable_gravity_attr = physx_body_api.GetDisableGravityAttr()
            if disable_gravity_attr is None or not disable_gravity_attr.IsValid():
                disable_gravity_attr = physx_body_api.CreateDisableGravityAttr()
            disable_gravity_attr.Set(False)
        except Exception as exc:
            print(
                f'[Grasp latch] failed to restore gravity on prim={prim_path}: {exc}'
            )

    def _resolve_rigid_body_target_prim(self, prim_path: str):
        normalized_path = str(prim_path or '').strip()
        if not normalized_path or self.stage is None:
            return None

        prim = self.stage.GetPrimAtPath(normalized_path)
        if prim is None or not prim.IsValid():
            return None

        try:
            from pxr import UsdPhysics
        except Exception:
            return None

        candidates: list[tuple[int, Any]] = []
        for candidate in Usd.PrimRange(prim):
            if candidate is None or not candidate.IsValid():
                continue
            if candidate.HasAPI(UsdPhysics.RigidBodyAPI):
                candidates.append((str(candidate.GetPath()).count('/'), candidate))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _normalize_collision_filter_pairs(
        self,
        pairs: Optional[Sequence[Sequence[str]]],
    ) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        for pair in pairs or []:
            try:
                robot_path = str(pair[0] or '').strip()
                object_path = str(pair[1] or '').strip()
            except Exception:
                continue
            if robot_path and object_path:
                normalized.append((robot_path, object_path))
        return normalized

    def _latch_collision_filtering_persistent(self, latch_state: Dict[str, Any]) -> bool:
        return self._coerce_bool(
            latch_state.get('persist_collision_filtering', self.persist_arm_collision_filtering),
            default=self.persist_arm_collision_filtering,
        )

    def _remove_grasp_latch_collision_filters(
        self,
        latch_state: Dict[str, Any],
        *,
        force: bool = False,
    ) -> None:
        if self._latch_collision_filtering_persistent(latch_state) and not force:
            return
        self._remove_arm_collision_filtering(
            self._normalize_collision_filter_pairs(
                latch_state.get('target_collision_filter_pairs') or []
            )
        )

    def _apply_grasp_latch_collision_filtering(self, object_prim_path: str) -> list[tuple[str, str]]:
        if self.stage is None or not object_prim_path:
            return []
        if bool(self.enable_aggressive_lifecycle_collision_filtering):
            self._apply_aggressive_lifecycle_collision_filtering()
            return []
        try:
            from pxr import UsdPhysics
        except Exception as exc:
            print(f'[Grasp latch] collision filtering unavailable: {exc}')
            return []

        robot_root_path = ''
        if self.robot_controller is not None:
            robot_root_path = str(getattr(self.robot_controller, 'robot_prim_path', '') or '').strip()
        if not robot_root_path:
            robot_root_path = str(self.robot_articulation_path or '').strip()
        if not robot_root_path:
            return []

        robot_collision_paths = self._collect_collision_prim_paths(robot_root_path)
        object_collision_paths = self._collect_collision_prim_paths(object_prim_path)
        pose_prim_path = self._resolve_pose_prim_path(object_prim_path)
        if pose_prim_path and pose_prim_path != object_prim_path:
            object_collision_paths.extend(self._collect_collision_prim_paths(pose_prim_path))

        robot_collision_paths = sorted(set(robot_collision_paths))
        object_collision_paths = sorted(set(object_collision_paths))
        if not robot_collision_paths or not object_collision_paths:
            return []

        added_pairs: list[tuple[str, str]] = []
        for robot_collision_path in robot_collision_paths:
            robot_collision_prim = self.stage.GetPrimAtPath(robot_collision_path)
            if robot_collision_prim is None or not robot_collision_prim.IsValid():
                continue
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(robot_collision_prim)
            rel = filtered_pairs_api.CreateFilteredPairsRel()
            existing_targets = {str(target) for target in rel.GetTargets()}
            for object_collision_path in object_collision_paths:
                if object_collision_path == robot_collision_path or object_collision_path in existing_targets:
                    continue
                if rel.AddTarget(Sdf.Path(object_collision_path)):
                    added_pairs.append((robot_collision_path, object_collision_path))
                    existing_targets.add(object_collision_path)
        return added_pairs

    def _merge_collision_filter_pairs(
        self,
        lhs: Optional[Sequence[Sequence[str]]],
        rhs: Optional[Sequence[Sequence[str]]],
    ) -> list[tuple[str, str]]:
        merged: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for pair in self._normalize_collision_filter_pairs(lhs) + self._normalize_collision_filter_pairs(rhs):
            if pair in seen:
                continue
            seen.add(pair)
            merged.append(pair)
        return merged

    def _build_grasp_pose_latch_state(
        self,
        *,
        object_name: str,
        object_prim_path: str,
        grasp_position: np.ndarray,
        grasp_orientation: np.ndarray,
        target_collision_filter_pairs: Optional[Sequence[Sequence[str]]] = None,
        target_collision_filter_prim_path: str = '',
    ) -> Dict[str, Any]:
        object_pose = self._get_raw_pose_for_prim_path(object_prim_path)
        if object_pose is None:
            return {}

        object_position, object_orientation = object_pose
        object_orientation = self._normalize_quaternion(object_orientation)
        grasp_position = np.array(grasp_position, dtype=float)
        grasp_orientation = self._normalize_quaternion(grasp_orientation)
        object_rotation = self._quaternion_to_rotation_matrix(object_orientation)
        local_grasp_position = object_rotation.T @ (grasp_position - object_position)
        local_grasp_orientation = self._quaternion_multiply(
            self._quaternion_inverse(object_orientation),
            grasp_orientation,
        )
        normalized_pairs = self._normalize_collision_filter_pairs(target_collision_filter_pairs)
        return {
            'object_name': str(object_name or '').strip(),
            'object_prim_path': str(object_prim_path or '').strip(),
            'target_collision_filter_prim_path': str(target_collision_filter_prim_path or '').strip(),
            'target_collision_filter_pairs': normalized_pairs,
            'persist_collision_filtering': bool(self.persist_arm_collision_filtering),
            'local_grasp_position': local_grasp_position.astype(float).tolist(),
            'local_grasp_orientation': local_grasp_orientation.astype(float).tolist(),
            'grasp_world_position': grasp_position.astype(float).tolist(),
            'grasp_world_orientation': grasp_orientation.astype(float).tolist(),
        }

    def _prepare_grasp_pose_latch_candidate(
        self,
        *,
        object_name: str,
        object_prim_path: str,
        grasp_position: np.ndarray,
        grasp_orientation: np.ndarray,
        target_collision_filter_pairs: Optional[Sequence[Sequence[str]]] = None,
        target_collision_filter_prim_path: str = '',
        persist_collision_filtering: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not bool(self.ENABLE_GRASP_POSE_LATCH):
            return {}
        self._clear_pending_grasp_pose_latch_candidate(remove_collision_filters=True)
        candidate = self._build_grasp_pose_latch_state(
            object_name=object_name,
            object_prim_path=object_prim_path,
            grasp_position=grasp_position,
            grasp_orientation=grasp_orientation,
            target_collision_filter_pairs=target_collision_filter_pairs,
            target_collision_filter_prim_path=target_collision_filter_prim_path,
        )
        if not candidate:
            return {}
        if persist_collision_filtering is not None:
            candidate['persist_collision_filtering'] = bool(persist_collision_filtering)
        self._pending_grasp_pose_latch_state = candidate
        return dict(candidate)

    def _clear_pending_grasp_pose_latch_candidate(
        self,
        *,
        remove_collision_filters: bool,
        force_remove_collision_filters: bool = False,
    ) -> None:
        pending_state = self._pending_grasp_pose_latch_state or {}
        if (
            pending_state
            and remove_collision_filters
            and (
                force_remove_collision_filters
                or not self._latch_collision_filtering_persistent(pending_state)
            )
        ):
            self._remove_grasp_latch_collision_filters(
                pending_state,
                force=force_remove_collision_filters,
            )
        self._pending_grasp_pose_latch_state = None

    def _enable_grasp_pose_latch(self, obj_name: str = '') -> Dict[str, Any]:
        if not bool(self.ENABLE_GRASP_POSE_LATCH):
            return {}
        if self.robot_controller is None or self.stage is None:
            return {}

        latch_state = dict(self._pending_grasp_pose_latch_state or {})
        if not latch_state:
            if not str(obj_name or '').strip():
                return {}
            resolved_name, object_prim_path = self._resolve_grasp_target(obj_name)
            ee_pose = self._get_end_effector_pose_raw()
            if not object_prim_path or ee_pose is None:
                return {}
            latch_state = self._build_grasp_pose_latch_state(
                object_name=resolved_name or obj_name,
                object_prim_path=object_prim_path,
                grasp_position=np.array(ee_pose[0], dtype=float),
                grasp_orientation=np.array(ee_pose[1], dtype=float),
            )
            if not latch_state:
                return {}
        latch_state['persist_collision_filtering'] = self._latch_collision_filtering_persistent(latch_state)

        self._pending_grasp_pose_latch_state = None
        object_prim_path = str(latch_state.get('object_prim_path', '') or '').strip()
        rigid_body_prim = self._resolve_rigid_body_target_prim(object_prim_path)
        rigid_body_prim_path = ''
        previous_kinematic = None
        if rigid_body_prim is not None and rigid_body_prim.IsValid():
            rigid_body_prim_path = str(rigid_body_prim.GetPath())
            previous_kinematic = self._set_rigid_body_kinematic(rigid_body_prim_path, True)

        # Disable all collisions on the grasped object. While the latch holds
        # it, the object follows the TCP kinematically and we do not want it
        # to push into the gripper, the table, walls, or any other world
        # geometry. The previous CollisionEnabled values are saved so they can
        # be restored on release.
        previous_collision_state = self._disable_object_collisions(object_prim_path)
        latch_state['previous_collision_state'] = previous_collision_state

        latch_filter_pairs = self._apply_grasp_latch_collision_filtering(object_prim_path)
        latch_state['target_collision_filter_pairs'] = self._merge_collision_filter_pairs(
            latch_state.get('target_collision_filter_pairs') or [],
            latch_filter_pairs,
        )
        if self._latch_collision_filtering_persistent(latch_state):
            self._remember_persistent_arm_collision_filtering(
                latch_state.get('target_collision_filter_pairs') or []
            )
        latch_state['rigid_body_prim_path'] = rigid_body_prim_path
        latch_state['previous_kinematic'] = previous_kinematic
        self._grasp_pose_latch_state = latch_state
        self._update_grasp_pose_latch()
        print(
            '[Grasp latch] enabled '
            f'object={str(latch_state.get("object_name", "") or "")} '
            f'object_prim={object_prim_path} '
            f'rigid_body_prim={rigid_body_prim_path or "<none>"}'
        )
        return dict(self._grasp_pose_latch_state)

    def _release_grasp_pose_latch(self, *, force_remove_collision_filters: bool = False) -> None:
        self._clear_pending_grasp_pose_latch_candidate(
            remove_collision_filters=True,
            force_remove_collision_filters=force_remove_collision_filters,
        )
        latch_state = self._grasp_pose_latch_state or {}
        if not latch_state:
            return
        self._remove_grasp_latch_collision_filters(
            latch_state,
            force=force_remove_collision_filters,
        )
        rigid_body_prim_path = str(latch_state.get('rigid_body_prim_path', '') or '').strip()
        previous_kinematic = latch_state.get('previous_kinematic')
        if rigid_body_prim_path and previous_kinematic is not None:
            self._set_rigid_body_kinematic(rigid_body_prim_path, bool(previous_kinematic))
        elif rigid_body_prim_path:
            self._set_rigid_body_kinematic(rigid_body_prim_path, False)
        if rigid_body_prim_path:
            self._restore_rigid_body_latch_controls(rigid_body_prim_path)
        # Re-enable the object's collision API entries we disabled at latch
        # time so the object behaves normally once the gripper releases it.
        self._restore_object_collisions(latch_state.get('previous_collision_state'))
        print(
            '[Grasp latch] released '
            f'object={str(latch_state.get("object_name", "") or "")} '
            f'object_prim={str(latch_state.get("object_prim_path", "") or "")}'
        )
        self._grasp_pose_latch_state = None

    def _get_raw_pose_for_prim_path(
        self,
        prim_path: str,
        *,
        resolve_pose_prim: bool = True,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        pose_prim_path = self._resolve_pose_prim_path(prim_path) if resolve_pose_prim else str(prim_path or '').strip()
        if not pose_prim_path:
            return None
        try:
            position, orientation = get_world_pose(pose_prim_path)
            return np.array(position, dtype=float), self._normalize_quaternion(np.array(orientation, dtype=float))
        except Exception:
            return None
    def _update_grasp_pose_latch(self) -> None:
        latch_state = self._grasp_pose_latch_state or {}
        if not latch_state or self.robot_controller is None:
            return
        ee_pose = self._get_end_effector_pose_raw()
        if ee_pose is None:
            return

        object_prim_path = str(latch_state.get('object_prim_path', '') or '').strip()
        if not object_prim_path:
            return

        rigid_body_prim_path = str(latch_state.get('rigid_body_prim_path', '') or '').strip()
        if rigid_body_prim_path:
            self._force_rigid_body_latch_controls(rigid_body_prim_path)

        ee_position, ee_orientation = ee_pose
        ee_orientation = self._normalize_quaternion(ee_orientation)
        local_grasp_position = np.array(
            latch_state.get('local_grasp_position', [0.0, 0.0, 0.0]),
            dtype=float,
        )
        local_grasp_orientation = np.array(
            latch_state.get('local_grasp_orientation', [1.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        target_orientation = self._quaternion_multiply(
            ee_orientation,
            self._quaternion_inverse(local_grasp_orientation),
        )
        target_rotation = self._quaternion_to_rotation_matrix(target_orientation)
        target_position = np.array(ee_position, dtype=float) - target_rotation @ local_grasp_position
        self._set_prim_world_pose_raw(object_prim_path, target_position, target_orientation)

    def _wait_for_torso(
        self,
        *,
        target_height: float,
        timeout_sec: float,
        tolerance: float,
    ) -> Dict[str, Any]:
        start = time.monotonic()
        settled = False
        current_height = None
        if self.robot_controller is None:
            return {'settled': False, 'elapsed_sec': 0.0, 'torso_height': current_height}

        while time.monotonic() - start < timeout_sec:
            self._step_robot(render=True)
            try:
                current_height = float(self.robot_controller.get_torso_height())
            except Exception:
                continue

            if abs(current_height - float(target_height)) <= tolerance:
                settled = True
                break

        return {
            'settled': settled,
            'elapsed_sec': float(time.monotonic() - start),
            'torso_height': current_height,
        }

    def _set_manipulation_torso_height_from_grasp_height(self, grasp_height: float) -> float:
        if self.robot_controller is None:
            raise RuntimeError('robot_not_loaded')

        requested_height = float(grasp_height) - float(self.DEFAULT_MANIPULATION_TORSO_HEIGHT_OFFSET)
        try:
            applied_height = float(self.robot_controller.set_torso_height(requested_height))
        except Exception as exc:
            raise RuntimeError(f'torso_command_failed:{exc}') from exc

        result = self._wait_for_torso(
            target_height=applied_height,
            timeout_sec=float(self.DEFAULT_TORSO_TIMEOUT_SEC),
            tolerance=float(self.DEFAULT_TORSO_POSITION_TOLERANCE),
        )
        if not result['settled']:
            print(
                'WARNING: torso adjustment for manipulation suggestion did not fully settle before timeout. '
                f'grasp_height={grasp_height:.4f}, target_height={applied_height:.4f}, '
                f'current_height={result["torso_height"]}'
            )
        return applied_height

    def _wait_for_navigation(
        self,
        *,
        timeout_sec: float,
        position_tolerance: float,
        yaw_tolerance: float,
    ) -> Dict[str, Any]:
        start = time.monotonic()
        reached = False
        distance = None
        yaw_error = None
        failure_reason = ''
        failure_message = ''
        navigation_debug: Dict[str, Any] = {}

        try:
            while time.monotonic() - start < timeout_sec:
                self._step_robot(render=True)
                if self.robot_controller is None or self._active_navigation_target is None:
                    continue

                navigation_debug = self._navigation_debug_payload()
                failure_reason = str(navigation_debug.get('navigation_failure_reason', '') or '').strip()
                if failure_reason:
                    failure_message = self._describe_navigation_failure(
                        failure_reason,
                        navigation_debug,
                    )
                    break

                pose = self._get_robot_base_pose_raw()
                if pose is None:
                    continue

                current_position, current_orientation = pose
                target_position = np.array(self._active_navigation_target['position'], dtype=float)
                target_yaw = self._active_navigation_target.get('yaw')
                distance = float(np.linalg.norm(current_position[:2] - target_position[:2]))

                if target_yaw is None:
                    yaw_error = 0.0
                else:
                    yaw_error = abs(self._wrap_to_pi(target_yaw - self._yaw_from_quaternion(current_orientation)))

                if distance <= position_tolerance and yaw_error <= yaw_tolerance:
                    reached = True
                    break

                if not bool(navigation_debug.get('navigation_target_active', True)):
                    failure_reason = 'navigation_aborted_before_reaching_target'
                    failure_message = self._describe_navigation_failure(
                        failure_reason,
                        navigation_debug,
                    )
                    break
        finally:
            if self.robot_controller is not None:
                try:
                    self.robot_controller.stop_base()
                    if self.my_world is not None:
                        self.my_world.step(render=True)
                except Exception as exc:
                    print(f'Base stop failed: {exc}')
            self._active_navigation_target = None

        return {
            'reached': reached,
            'distance': distance,
            'yaw_error': yaw_error,
            'failure_reason': failure_reason,
            'failure_message': failure_message,
            'navigation_debug': dict(navigation_debug),
            'elapsed_sec': float(time.monotonic() - start),
        }

    @staticmethod
    def _describe_navigation_failure(reason: str, debug_payload: Optional[Dict[str, Any]] = None) -> str:
        normalized_reason = str(reason or 'navigation_failed').strip()
        payload = dict(debug_payload or {})
        if normalized_reason == 'navigation_collision_suspected:base_stuck':
            stalled_steps = int(payload.get('navigation_collision_stall_steps', 0) or 0)
            stalled_threshold = int(payload.get('navigation_collision_stall_step_threshold', 0) or 0)
            displacement = float(payload.get('navigation_collision_stall_displacement', 0.0) or 0.0)
            displacement_epsilon = float(payload.get('navigation_collision_stall_position_epsilon', 0.0) or 0.0)
            return (
                'Navigation aborted because the robot base remained almost stationary for too long, '
                'which suggests it is stuck against a wall or obstacle. '
                f'stalled_steps={stalled_steps}, '
                f'stalled_step_threshold={stalled_threshold}, '
                f'base_displacement={displacement:.4f}, '
                f'position_epsilon={displacement_epsilon:.4f}'
            )
        if normalized_reason == 'navigation_aborted_before_reaching_target':
            return 'Navigation aborted before reaching the target pose.'
        return normalized_reason

    def _reset_scene_runtime(self) -> bool:
        if not str(self.usd_file_path or '').strip():
            print('Reset requested without a loaded scene path.')
            return False

        print('Resetting Isaac Sim runtime by reloading the active stage...')
        try:
            rebuilt, error = self._rebuild_loaded_runtime()
            if not rebuilt:
                print(f'ERROR: Runtime rebuild failed during reset: {error}')
                return False

            self.command_counter = 0
            self.last_command = None
            self.gripper_state = 'open'
            self.object_in_gripper = False
            self.grasped_object_name = ''
            self.grasped_object_mass = None
            self._release_grasp_pose_latch(force_remove_collision_filters=True)
            self.robot_pose = self._get_end_effector_pose() or self._default_robot_pose()
            self._refresh_scene_index()
            self._initialize_world_state()
            self._apply_runtime_state_requirements()
            print('Isaac Sim runtime reset completed via stage reload.')
            return True
        except Exception as exc:
            print(f'ERROR: Exception while resetting stage runtime: {exc}')
            return False

    def _lookup_object_prim_path(self, obj_name: str) -> str:
        lowered = str(obj_name or '').strip().lower()
        if not lowered:
            return ''

        room_names: list[str] = []
        current_room_name = str(self._detect_current_room_name() or '')
        if current_room_name:
            self.current_room_name = current_room_name
            room_names.append(current_room_name)

        for room_name in self.room_index.keys():
            room_name = str(room_name or '')
            if room_name and room_name not in room_names:
                room_names.append(room_name)

        if self.stage is not None:
            for room_name in room_names:
                room_prim = self.stage.GetPrimAtPath(f'{ROOMS_ROOT_PATH}/{room_name}')
                if room_prim is None or not room_prim.IsValid():
                    continue
                for child in room_prim.GetChildren():
                    if child is None or not child.IsValid():
                        continue
                    child_name = str(child.GetName() or '').strip()
                    if child_name.lower() == lowered:
                        return str(child.GetPath())

        for name, path in self.object_prim_paths.items():
            if name.lower() == lowered:
                return path
        return ''

    def _lookup_object_name_by_prim_path(self, prim_path: str) -> str:
        normalized_path = str(prim_path or '').strip()
        if not normalized_path:
            return ''
        for name, path in self.object_prim_paths.items():
            if str(path or '').strip() == normalized_path:
                return str(name or '').strip()
        return ''

    def _resolve_grasp_target(self, requested_name: str = '') -> tuple[str, str]:
        self._refresh_scene_index()

        normalized_name = str(requested_name or '').strip()
        if normalized_name:
            prim_path = self._lookup_object_prim_path(normalized_name)
            if prim_path:
                canonical_name = self._lookup_object_name_by_prim_path(prim_path) or normalized_name
                return canonical_name, prim_path

        end_effector_pose = self._get_end_effector_pose_raw()
        if end_effector_pose is None:
            return '', ''

        _, object_entry = self._find_room_object_entry_by_position(np.array(end_effector_pose[0], dtype=float))
        if not isinstance(object_entry, dict):
            return '', ''

        object_name = str(object_entry.get('name', '') or '').strip()
        prim_path = str(object_entry.get('prim_path', '') or '').strip()
        if not object_name or not prim_path:
            return '', ''
        return object_name, prim_path

    def _resolve_mass_target_prim(self, prim_path: str):
        normalized_path = str(prim_path or '').strip()
        if not normalized_path or self.stage is None:
            return None

        prim = self.stage.GetPrimAtPath(normalized_path)
        if prim is None or not prim.IsValid():
            return None

        try:
            from pxr import UsdPhysics
        except Exception:
            return None

        candidates: list[tuple[int, int, Any]] = []
        for candidate in Usd.PrimRange(prim):
            if candidate is None or not candidate.IsValid():
                continue
            if not (candidate.HasAPI(UsdPhysics.RigidBodyAPI) or candidate.HasAPI(UsdPhysics.MassAPI)):
                continue
            candidate_path = str(candidate.GetPath())
            candidates.append(
                (
                    0 if candidate.HasAPI(UsdPhysics.MassAPI) else 1,
                    candidate_path.count('/'),
                    candidate,
                )
            )

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    @staticmethod
    def _read_authored_mass_state(mass_prim) -> tuple[bool, Optional[float]]:
        try:
            from pxr import UsdPhysics
        except Exception:
            return False, None

        if mass_prim is None or not mass_prim.IsValid():
            return False, None

        mass_api = UsdPhysics.MassAPI(mass_prim)
        mass_attr = mass_api.GetMassAttr()
        if mass_attr is None or not mass_attr.IsValid():
            return False, None

        try:
            had_authored_mass = bool(mass_attr.HasAuthoredValueOpinion())
        except Exception:
            had_authored_mass = False

        try:
            mass_value = mass_attr.Get()
        except Exception:
            mass_value = None

        if mass_value is None:
            return had_authored_mass, None

        try:
            numeric_mass = float(mass_value)
        except (TypeError, ValueError):
            return had_authored_mass, None
        if not np.isfinite(numeric_mass):
            return had_authored_mass, None
        return had_authored_mass, numeric_mass

    @staticmethod
    def _set_mass_on_prim(mass_prim, mass_value: float) -> float:
        from pxr import UsdPhysics

        mass_api = UsdPhysics.MassAPI(mass_prim)
        if not mass_prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI.Apply(mass_prim)
        mass_attr = mass_api.GetMassAttr()
        if mass_attr is None or not mass_attr.IsValid():
            mass_attr = mass_api.CreateMassAttr()
        applied_mass = max(float(mass_value), 1e-4)
        mass_attr.Set(applied_mass)
        return applied_mass

    def _ensure_high_friction_material(
        self,
        *,
        material_name: str,
        static_friction: Optional[float] = None,
        dynamic_friction: Optional[float] = None,
        restitution: Optional[float] = None,
    ) -> str:
        if self.stage is None:
            return ''

        from pxr import UsdPhysics, UsdShade

        material_root_path = '/World/Physics_Materials'
        UsdGeom.Xform.Define(self.stage, material_root_path)
        material_path = f'{material_root_path}/{material_name}'
        material = UsdShade.Material.Define(self.stage, material_path)
        material_prim = material.GetPrim()
        material_api = UsdPhysics.MaterialAPI(material_prim)
        if not material_prim.HasAPI(UsdPhysics.MaterialAPI):
            material_api = UsdPhysics.MaterialAPI.Apply(material_prim)

        static_value = float(
            self.DEFAULT_GRASP_STATIC_FRICTION if static_friction is None else static_friction
        )
        dynamic_value = float(
            self.DEFAULT_GRASP_DYNAMIC_FRICTION if dynamic_friction is None else dynamic_friction
        )
        restitution_value = float(
            self.DEFAULT_GRASP_RESTITUTION if restitution is None else restitution
        )

        material_api.CreateStaticFrictionAttr().Set(static_value)
        material_api.CreateDynamicFrictionAttr().Set(dynamic_value)
        material_api.CreateRestitutionAttr().Set(restitution_value)
        # print(
        #     '[Physics material] ensured material '
        #     f'path={material_path} static_friction={static_value:.4f} '
        #     f'dynamic_friction={dynamic_value:.4f} restitution={restitution_value:.4f}'
        # )
        return material_path

    def _bind_high_friction_material_to_prim(
        self,
        prim_path: str,
        *,
        material_name: str,
        static_friction: Optional[float] = None,
        dynamic_friction: Optional[float] = None,
        restitution: Optional[float] = None,
    ) -> bool:
        normalized_path = str(prim_path or '').strip()
        if not normalized_path or self.stage is None:
            # print(
            #     '[Physics material] bind skipped: '
            #     f'prim_path={normalized_path or "<empty>"} stage_available={self.stage is not None}'
            # )
            return False

        root_prim = self.stage.GetPrimAtPath(normalized_path)
        if root_prim is None or not root_prim.IsValid():
            # print(f'[Physics material] bind failed: invalid prim path={normalized_path}')
            return False

        try:
            from pxr import UsdShade
        except Exception as exc:
            # print(f'[Physics material] bind failed: unable to import UsdShade for prim={normalized_path}: {exc}')
            return False

        material_path = self._ensure_high_friction_material(
            material_name=material_name,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        if not material_path:
            # print(f'[Physics material] bind failed: material creation returned empty path for prim={normalized_path}')
            return False
            return False

        material = UsdShade.Material.Get(self.stage, material_path)
        if not material:
            # print(
            #     '[Physics material] bind failed: material lookup returned invalid handle '
            #     f'material_path={material_path} prim={normalized_path}'
            # )
            return False

        binding_strength = getattr(UsdShade.Tokens, 'strongerThanDescendants', None)
        prim_type = ''
        try:
            prim_type = str(root_prim.GetTypeName() or '')
        except Exception:
            prim_type = ''
        try:
            applied_schemas = list(root_prim.GetAppliedSchemas() or [])
        except Exception:
            applied_schemas = []
        # print(
        #     '[Physics material] binding start '
        #     f'prim={normalized_path} prim_type={prim_type or "<untyped>"} '
        #     f'applied_schemas={applied_schemas} material={material_path}'
        # )
        try:
            binding_api = UsdShade.MaterialBindingAPI.Apply(root_prim)
            # print(f'[Physics material] MaterialBindingAPI.Apply succeeded for prim={normalized_path}')
        except Exception as exc:
            # print(
            #     '[Physics material] MaterialBindingAPI.Apply raised, '
            #     f'falling back to wrapper for prim={normalized_path}: {exc}'
            # )
            binding_api = UsdShade.MaterialBindingAPI(root_prim)

        try:
            if binding_strength is not None:
                binding_api.Bind(
                    material,
                    bindingStrength=binding_strength,
                    materialPurpose='physics',
                )
                # print(
                #     '[Physics material] bind succeeded with explicit strength '
                #     f'prim={normalized_path} material={material_path} purpose=physics '
                #     f'strength={binding_strength}'
                # )
            else:
                binding_api.Bind(material, materialPurpose='physics')
                # print(
                #     '[Physics material] bind succeeded without explicit strength '
                #     f'prim={normalized_path} material={material_path} purpose=physics'
                # )
        except TypeError as exc:
            # print(
            #     '[Physics material] bind raised TypeError, retrying fallback signature '
            #     f'prim={normalized_path}: {exc}'
            # )
            try:
                binding_api.Bind(material, materialPurpose='physics')
                # print(
                #     '[Physics material] bind fallback succeeded with materialPurpose only '
                #     f'prim={normalized_path} material={material_path}'
                # )
            except TypeError as fallback_exc:
                # print(
                #     '[Physics material] bind fallback raised TypeError, retrying bare Bind '
                #     f'prim={normalized_path}: {fallback_exc}'
                # )
                binding_api.Bind(material)
                # print(
                #     '[Physics material] bind bare fallback succeeded '
                #     f'prim={normalized_path} material={material_path}'
                # )
        except Exception as exc:
            # print(
            #     '[Physics material] bind failed with unexpected exception '
            #     f'prim={normalized_path} material={material_path}: {exc}'
            # )
            return False
        return True

    def _resolve_grasp_material_target_prim_path(self, prim_path: str) -> str:
        normalized_path = str(prim_path or '').strip()
        if not normalized_path or self.stage is None:
            return normalized_path

        base_link_path = f'{normalized_path}/base_link'
        base_link_prim = self.stage.GetPrimAtPath(base_link_path)
        if base_link_prim is not None and base_link_prim.IsValid():
            # print(
            #     '[Physics material] resolved target material prim to base_link '
            #     f'original={normalized_path} resolved={base_link_path}'
            # )
            return base_link_path
        # print(
        #     '[Physics material] target base_link not found, using original prim '
        #     f'original={normalized_path}'
        # )
        return normalized_path

    def _lookup_robot_joint_prim_path(self, joint_name: str) -> str:
        normalized_name = str(joint_name or '').strip()
        if not normalized_name or self.stage is None:
            return ''

        robot_prim_path = ''
        if self.robot_controller is not None:
            robot_prim_path = str(getattr(self.robot_controller, 'robot_prim_path', '') or '').strip()
        if not robot_prim_path:
            robot_prim_path = str(self.robot_articulation_path or '').strip()
        if not robot_prim_path:
            return ''

        direct_path = f'{robot_prim_path}/{normalized_name}'
        direct_prim = self.stage.GetPrimAtPath(direct_path)
        if direct_prim is not None and direct_prim.IsValid():
            return direct_path

        robot_prim = self.stage.GetPrimAtPath(robot_prim_path)
        if robot_prim is None or not robot_prim.IsValid():
            return ''

        for prim in Usd.PrimRange(robot_prim):
            if prim is not None and prim.IsValid() and str(prim.GetName() or '') == normalized_name:
                return str(prim.GetPath())
        return ''

    @staticmethod
    def _get_attr_float(prim, attr_name: str) -> Optional[float]:
        try:
            attr = prim.GetAttribute(attr_name)
            if attr is None:
                return None
            value = attr.Get()
        except Exception:
            return None
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _get_relationship_targets(prim, rel_name: str) -> list[str]:
        try:
            rel = prim.GetRelationship(rel_name)
            if rel is None:
                return []
            return [str(target) for target in rel.GetTargets()]
        except Exception:
            return []

    @staticmethod
    def _is_door_joint_limit_range(lower_limit: Optional[float], upper_limit: Optional[float]) -> bool:
        if lower_limit is None or upper_limit is None:
            return False
        lower = float(lower_limit)
        upper = float(upper_limit)
        if lower > upper:
            lower, upper = upper, lower

        # USD revolute joint limits are normally authored in degrees, but a few
        # imported assets use radians. Accept both forms for 0-90/0-180 doors.
        degree_ranges = (
            abs(lower) <= 10.0 and 70.0 <= upper <= 200.0,
            -200.0 <= lower <= -70.0 and abs(upper) <= 10.0,
        )
        radian_ranges = (
            abs(lower) <= 0.2 and 1.2 <= upper <= 3.4,
            -3.4 <= lower <= -1.2 and abs(upper) <= 0.2,
        )
        return bool(any(degree_ranges) or any(radian_ranges))

    @staticmethod
    def _choose_open_joint_target(lower_limit: Optional[float], upper_limit: Optional[float]) -> float:
        if lower_limit is None and upper_limit is None:
            return 90.0

        limits = []
        for value in (lower_limit, upper_limit):
            if value is None:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                limits.append(numeric)
        if not limits:
            return 90.0

        values_are_degrees = any(abs(value) > float(2.0 * np.pi + 0.2) for value in limits)
        max_open_delta = 90.0 if values_are_degrees else float(np.pi / 2.0)
        if len(limits) == 1:
            open_limit = limits[0]
            open_delta = float(np.sign(open_limit or 1.0) * min(abs(open_limit), max_open_delta))
            return open_delta

        lower, upper = float(min(limits)), float(max(limits))
        if lower <= 0.0 <= upper:
            closed_target = 0.0
            positive_span = abs(upper)
            negative_span = abs(lower)
            open_limit = upper if positive_span >= negative_span else lower
        else:
            closed_target = min((lower, upper), key=lambda value: abs(value))
            open_limit = max((lower, upper), key=lambda value: abs(value))

        open_delta = float(open_limit - closed_target)
        if abs(open_delta) > max_open_delta:
            open_delta = float(np.sign(open_delta or 1.0) * max_open_delta)
        return float(closed_target + open_delta)

    def _find_container_door_joint_candidates(self, container_prim_path: str) -> list[Dict[str, Any]]:
        if self.stage is None:
            return []

        normalized_path = str(container_prim_path or '').strip().rstrip('/')
        if not normalized_path:
            return []

        root_prim = self.stage.GetPrimAtPath(normalized_path)
        if root_prim is None or not root_prim.IsValid():
            return []

        base_link_path = f'{normalized_path}/base_link'
        base_link_prim = self.stage.GetPrimAtPath(base_link_path)
        search_root = base_link_prim if base_link_prim is not None and base_link_prim.IsValid() else root_prim
        search_root_path = str(search_root.GetPath())

        try:
            from pxr import UsdPhysics
        except Exception as exc:
            print(f'[Container open] unavailable: failed to import UsdPhysics: {exc}')
            return []

        candidates: list[Dict[str, Any]] = []
        for prim in Usd.PrimRange(search_root):
            if prim is None or not prim.IsValid():
                continue
            prim_type = str(prim.GetTypeName() or '')
            is_revolute_joint = prim_type == 'PhysicsRevoluteJoint'
            try:
                is_revolute_joint = is_revolute_joint or bool(prim.IsA(UsdPhysics.RevoluteJoint))
            except Exception:
                pass
            if not is_revolute_joint:
                continue

            joint_name = str(prim.GetName() or '').strip()
            lowered_name = joint_name.lower()
            lower_limit = self._get_attr_float(prim, 'physics:lowerLimit')
            upper_limit = self._get_attr_float(prim, 'physics:upperLimit')
            has_door_name = 'door' in lowered_name
            has_door_range = self._is_door_joint_limit_range(lower_limit, upper_limit)
            if not has_door_name and not has_door_range:
                continue

            body0_targets = self._get_relationship_targets(prim, 'physics:body0')
            body1_targets = self._get_relationship_targets(prim, 'physics:body1')
            door_link_path = body1_targets[0] if body1_targets else ''
            if has_door_name:
                matching_link_targets = [
                    target
                    for target in body0_targets + body1_targets
                    if 'door' in target.lower()
                ]
                if matching_link_targets:
                    door_link_path = matching_link_targets[0]

            joint_path = str(prim.GetPath())
            candidates.append(
                {
                    'joint_name': joint_name,
                    'joint_prim_path': joint_path,
                    'joint_type': prim_type,
                    'base_link_path': search_root_path,
                    'lower_limit': lower_limit,
                    'upper_limit': upper_limit,
                    'target_position': self._choose_open_joint_target(lower_limit, upper_limit),
                    'door_link_path': door_link_path,
                    'body0_targets': body0_targets,
                    'body1_targets': body1_targets,
                    'matched_by_name': bool(has_door_name),
                    'matched_by_limit_range': bool(has_door_range),
                }
            )

        candidates.sort(
            key=lambda item: (
                0 if item['matched_by_name'] else 1,
                0 if item['matched_by_limit_range'] else 1,
                str(item.get('joint_prim_path', '')).count('/'),
            )
        )
        return candidates

    def _apply_container_door_open_drive(
        self,
        joint_prim_path: str,
        target_position: float,
        *,
        stiffness: Optional[float] = None,
        damping: Optional[float] = None,
        max_force: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self.stage is None:
            raise RuntimeError('stage_not_loaded')

        joint_prim = self.stage.GetPrimAtPath(str(joint_prim_path or '').strip())
        if joint_prim is None or not joint_prim.IsValid():
            raise RuntimeError(f'invalid_joint:{joint_prim_path}')

        from pxr import UsdPhysics

        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, 'angular')
        target = float(target_position)
        drive_api.CreateTargetPositionAttr().Set(target)
        drive_api.CreateStiffnessAttr().Set(
            float(self.DEFAULT_CONTAINER_DOOR_DRIVE_STIFFNESS if stiffness is None else stiffness)
        )
        drive_api.CreateDampingAttr().Set(
            float(self.DEFAULT_CONTAINER_DOOR_DRIVE_DAMPING if damping is None else damping)
        )
        drive_api.CreateMaxForceAttr().Set(
            float(self.DEFAULT_CONTAINER_DOOR_DRIVE_MAX_FORCE if max_force is None else max_force)
        )
        return {
            'joint_prim_path': str(joint_prim.GetPath()),
            'target_position': target,
            'drive': 'angular',
        }

    def _set_container_base_link_mass(self, container_prim_path: str, mass_value: float) -> Dict[str, Any]:
        if self.stage is None:
            raise RuntimeError('stage_not_loaded')

        normalized_path = str(container_prim_path or '').strip().rstrip('/')
        if not normalized_path:
            raise RuntimeError('container_prim_path_required')

        base_link_path = f'{normalized_path}/base_link'
        base_link_prim = self.stage.GetPrimAtPath(base_link_path)
        if base_link_prim is None or not base_link_prim.IsValid():
            raise RuntimeError(f'container_base_link_not_found:{normalized_path}')

        had_authored_mass, original_mass = self._read_authored_mass_state(base_link_prim)
        applied_mass = self._set_mass_on_prim(base_link_prim, float(mass_value))
        return {
            'base_link_path': base_link_path,
            'had_authored_mass': bool(had_authored_mass),
            'original_mass': original_mass,
            'applied_mass': applied_mass,
        }

    @staticmethod
    def _infer_container_door_closed_target(candidate: Dict[str, Any], open_target: float) -> float:
        _ = open_target
        lower_limit = candidate.get('lower_limit')
        upper_limit = candidate.get('upper_limit')
        limits = []
        for value in (lower_limit, upper_limit):
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                limits.append(numeric)
        if not limits:
            return 0.0
        return min(limits, key=lambda value: abs(value))

    def _get_container_door_drive_target(self, joint_prim_path: str) -> Optional[float]:
        if self.stage is None:
            return None

        joint_prim = self.stage.GetPrimAtPath(str(joint_prim_path or '').strip())
        if joint_prim is None or not joint_prim.IsValid():
            return None

        try:
            from pxr import UsdPhysics

            drive_api = UsdPhysics.DriveAPI(joint_prim, 'angular')
            attr = drive_api.GetTargetPositionAttr()
            if attr is not None and attr.IsValid():
                value = attr.Get()
                if value is not None:
                    numeric = float(value)
                    if np.isfinite(numeric):
                        return numeric
        except Exception:
            return None
        return None

    def _slowly_open_container_door(
        self,
        candidate: Dict[str, Any],
        target_position: float,
        *,
        timeout_sec: float,
        stiffness: Optional[float] = None,
        damping: Optional[float] = None,
        max_force: Optional[float] = None,
    ) -> Dict[str, Any]:
        start = time.monotonic()
        duration = max(float(timeout_sec), float(self.DEFAULT_PHYSICS_DT))
        joint_prim_path = str(candidate.get('joint_prim_path', '') or '').strip()
        existing_target = self._get_container_door_drive_target(joint_prim_path)
        start_target = (
            existing_target
            if existing_target is not None
            else self._infer_container_door_closed_target(candidate, target_position)
        )
        last_target = start_target
        step_count = 0

        self._apply_container_door_open_drive(
            joint_prim_path,
            start_target,
            stiffness=stiffness,
            damping=damping,
            max_force=max_force,
        )
        self._step_robot(render=True)
        step_count += 1

        while True:
            elapsed = float(time.monotonic() - start)
            alpha = min(max(elapsed / duration, 0.0), 1.0)
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            last_target = float(start_target + (target_position - start_target) * smooth_alpha)
            self._apply_container_door_open_drive(
                joint_prim_path,
                last_target,
                stiffness=stiffness,
                damping=damping,
                max_force=max_force,
            )
            self._step_robot(render=True)
            step_count += 1
            if alpha >= 1.0:
                break

        return {
            'joint_prim_path': joint_prim_path,
            'start_target_position': float(start_target),
            'target_position': float(target_position),
            'last_target_position': float(last_target),
            'drive': 'angular',
            'elapsed_sec': float(time.monotonic() - start),
            'step_count': int(step_count),
        }

    @staticmethod
    def _get_vec3_attr(prim, attr_name: str) -> Optional[np.ndarray]:
        try:
            attr = prim.GetAttribute(attr_name)
            if attr is None:
                return None
            value = attr.Get()
        except Exception:
            return None
        if value is None:
            return None
        try:
            vector = np.array([float(value[0]), float(value[1]), float(value[2])], dtype=float)
        except Exception:
            return None
        if vector.shape[0] < 3 or np.any(~np.isfinite(vector[:3])):
            return None
        return vector[:3]

    @staticmethod
    def _get_token_attr(prim, attr_name: str) -> str:
        try:
            attr = prim.GetAttribute(attr_name)
            if attr is None:
                return ''
            value = attr.Get()
        except Exception:
            return ''
        return str(value or '').strip()

    @staticmethod
    def _get_quaternion_attr(prim, attr_name: str) -> Optional[np.ndarray]:
        try:
            attr = prim.GetAttribute(attr_name)
            if attr is None:
                return None
            value = attr.Get()
        except Exception:
            return None
        if value is None:
            return None
        try:
            if hasattr(value, 'GetReal') and hasattr(value, 'GetImaginary'):
                imaginary = value.GetImaginary()
                quat = np.array(
                    [float(value.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])],
                    dtype=float,
                )
            else:
                raw = np.array(value, dtype=float).reshape(-1)
                if raw.size < 4:
                    return None
                quat = raw[:4]
        except Exception:
            return None
        if np.any(~np.isfinite(quat)):
            return None
        return IsaacSimAppRunner._normalize_quaternion(quat)

    @staticmethod
    def _axis_token_to_vector(axis_token: str) -> np.ndarray:
        token = str(axis_token or '').strip().upper()
        if token == 'Y':
            return np.array([0.0, 1.0, 0.0], dtype=float)
        if token == 'Z':
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return np.array([1.0, 0.0, 0.0], dtype=float)

    @staticmethod
    def _normalize_vector(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
        candidate = np.array(vector, dtype=float).reshape(-1)[:3]
        if candidate.size < 3 or np.any(~np.isfinite(candidate)):
            return np.array(fallback if fallback is not None else [0.0, 0.0, 1.0], dtype=float)
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-9:
            return np.array(fallback if fallback is not None else [0.0, 0.0, 1.0], dtype=float)
        return candidate / norm

    @staticmethod
    def _transform_direction_to_world(transform_matrix, direction: np.ndarray) -> np.ndarray:
        direction = np.array(direction, dtype=float).reshape(-1)[:3]
        origin = np.array(transform_matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0)), dtype=float)
        endpoint = np.array(
            transform_matrix.Transform(
                Gf.Vec3d(float(direction[0]), float(direction[1]), float(direction[2]))
            ),
            dtype=float,
        )
        return endpoint - origin

    @staticmethod
    def _rotate_point_about_axis(
        point: np.ndarray,
        anchor: np.ndarray,
        axis: np.ndarray,
        angle_rad: float,
    ) -> np.ndarray:
        point = np.array(point, dtype=float)
        anchor = np.array(anchor, dtype=float)
        axis = IsaacSimAppRunner._normalize_vector(axis)
        relative = point - anchor
        parallel = axis * float(np.dot(relative, axis))
        perpendicular = relative - parallel
        cos_value = float(np.cos(angle_rad))
        sin_value = float(np.sin(angle_rad))
        rotated_perpendicular = (
            perpendicular * cos_value
            + np.cross(axis, perpendicular) * sin_value
        )
        return anchor + parallel + rotated_perpendicular

    @staticmethod
    def _signed_angle_about_axis(
        start_vector: np.ndarray,
        current_vector: np.ndarray,
        axis: np.ndarray,
    ) -> float:
        axis = IsaacSimAppRunner._normalize_vector(axis)
        start = np.array(start_vector, dtype=float)
        current = np.array(current_vector, dtype=float)
        start -= axis * float(np.dot(start, axis))
        current -= axis * float(np.dot(current, axis))
        start = IsaacSimAppRunner._normalize_vector(start, fallback=np.array([1.0, 0.0, 0.0], dtype=float))
        current = IsaacSimAppRunner._normalize_vector(current, fallback=start)
        sin_value = float(np.dot(axis, np.cross(start, current)))
        cos_value = float(np.clip(np.dot(start, current), -1.0, 1.0))
        return float(np.arctan2(sin_value, cos_value))

    @staticmethod
    def _joint_angle_value_to_radians(value: float) -> float:
        angle = float(value)
        if abs(angle) > float(2.0 * np.pi + 0.2):
            return float(np.deg2rad(angle))
        return angle

    def _resolve_container_door_prim_path(
        self,
        container_prim_path: str,
        candidate: Dict[str, Any],
    ) -> str:
        if self.stage is None:
            return ''

        container_path = str(container_prim_path or '').strip().rstrip('/')
        body_targets = [
            str(target or '').strip()
            for target in list(candidate.get('body0_targets') or []) + list(candidate.get('body1_targets') or [])
            if str(target or '').strip()
        ]

        for target in body_targets:
            if 'door' in target.lower():
                prim = self.stage.GetPrimAtPath(target)
                if prim is not None and prim.IsValid():
                    return str(prim.GetPath())

        door_link_path = str(candidate.get('door_link_path', '') or '').strip()
        if door_link_path:
            prim = self.stage.GetPrimAtPath(door_link_path)
            if prim is not None and prim.IsValid():
                return str(prim.GetPath())

        base_link_path = f'{container_path}/base_link'
        for target in reversed(body_targets):
            if target == base_link_path:
                continue
            prim = self.stage.GetPrimAtPath(target)
            if prim is not None and prim.IsValid():
                return str(prim.GetPath())

        container_prim = self.stage.GetPrimAtPath(container_path)
        if container_prim is not None and container_prim.IsValid():
            for prim in Usd.PrimRange(container_prim):
                if prim is not None and prim.IsValid() and 'door' in str(prim.GetName() or '').lower():
                    return str(prim.GetPath())
        return ''

    def _find_container_handle_prim_path(self, root_prim_path: str) -> str:
        if self.stage is None:
            return ''

        root_prim = self.stage.GetPrimAtPath(str(root_prim_path or '').strip())
        if root_prim is None or not root_prim.IsValid():
            return ''

        handle_keywords = ('handle', 'knob', 'pull', 'grip', 'latch')
        candidates: list[tuple[int, int, int, str]] = []
        for prim in Usd.PrimRange(root_prim):
            if prim is None or not prim.IsValid():
                continue
            lowered_name = str(prim.GetName() or '').lower()
            matched_keyword = next(
                (index for index, keyword in enumerate(handle_keywords) if keyword in lowered_name),
                None,
            )
            if matched_keyword is None:
                continue
            prim_path = str(prim.GetPath())
            bbox = self._compute_world_bbox(prim)
            has_bounds = 0 if bbox is not None else 1
            candidates.append((matched_keyword, has_bounds, prim_path.count('/'), prim_path))

        if not candidates:
            return ''
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return candidates[0][3]

    def _estimate_door_handle_fallback_position(
        self,
        door_prim_path: str,
        candidate: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        if self.stage is None:
            return None

        door_prim = self.stage.GetPrimAtPath(str(door_prim_path or '').strip())
        bbox = self._compute_world_bbox(door_prim) if door_prim is not None and door_prim.IsValid() else None
        if bbox is None:
            return None

        min_corner, max_corner = bbox
        center = (min_corner + max_corner) * 0.5
        joint_frame = self._get_container_joint_world_frame(candidate)
        hinge_position = np.array(joint_frame.get('position', center), dtype=float)
        axis = self._normalize_vector(np.array(joint_frame.get('axis', [0.0, 0.0, 1.0]), dtype=float))

        corner_points = []
        for x_value in (float(min_corner[0]), float(max_corner[0])):
            for y_value in (float(min_corner[1]), float(max_corner[1])):
                point = np.array([x_value, y_value, float(center[2])], dtype=float)
                radial = point - hinge_position
                radial -= axis * float(np.dot(radial, axis))
                corner_points.append((float(np.linalg.norm(radial)), point))
        if not corner_points:
            return center
        corner_points.sort(key=lambda item: item[0], reverse=True)
        return corner_points[0][1]

    def _resolve_container_door_grasp_target(
        self,
        container_name: str,
        container_prim_path: str,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        door_prim_path = self._resolve_container_door_prim_path(container_prim_path, candidate)
        if not door_prim_path:
            raise RuntimeError('door_prim_not_found')

        handle_prim_path = self._find_container_handle_prim_path(door_prim_path)
        if not handle_prim_path:
            handle_prim_path = self._find_container_handle_prim_path(container_prim_path)

        grasp_prim_path = handle_prim_path or door_prim_path
        grasp_prim = self.stage.GetPrimAtPath(grasp_prim_path) if self.stage is not None else None
        grasp_name = str(grasp_prim.GetName() if grasp_prim is not None and grasp_prim.IsValid() else 'door')
        fallback_position = None
        bbox = self._compute_world_bbox(grasp_prim) if grasp_prim is not None and grasp_prim.IsValid() else None
        if bbox is not None:
            fallback_position = (bbox[0] + bbox[1]) * 0.5
        if fallback_position is None or not handle_prim_path:
            fallback_position = self._estimate_door_handle_fallback_position(door_prim_path, candidate)

        object_pose = self._get_pose_for_prim_path(grasp_prim_path)
        preferred_grasp = self._resolve_preferred_grasp_pose(
            f'{container_name}_{grasp_name}',
            prim_path=grasp_prim_path,
            object_pose=object_pose,
            fallback_position=fallback_position,
            force_refresh=True,
        )
        return {
            'door_prim_path': door_prim_path,
            'handle_prim_path': handle_prim_path,
            'grasp_prim_path': grasp_prim_path,
            'grasp_target_name': f'{container_name}_{grasp_name}',
            'fallback_position': fallback_position.astype(float).tolist()
            if fallback_position is not None
            else None,
            'preferred_grasp': preferred_grasp,
        }

    def _resolve_mock_container_door_contact_target(
        self,
        container_name: str,
        container_prim_path: str,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        _ = (container_name, container_prim_path)
        door_prim_path = self._resolve_container_door_prim_path(container_prim_path, candidate)
        if not door_prim_path:
            raise RuntimeError('door_prim_not_found')

        contact_position = self._estimate_door_handle_fallback_position(door_prim_path, candidate)
        if contact_position is None:
            door_prim = self.stage.GetPrimAtPath(door_prim_path) if self.stage is not None else None
            bbox = self._compute_world_bbox(door_prim) if door_prim is not None and door_prim.IsValid() else None
            if bbox is not None:
                contact_position = (bbox[0] + bbox[1]) * 0.5
        if contact_position is None:
            raise RuntimeError('door_mock_contact_position_unavailable')

        return {
            'door_prim_path': door_prim_path,
            'contact_position': np.array(contact_position, dtype=float),
            'contact_source': 'door_far_from_hinge',
        }

    def _get_container_joint_world_frame(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        if self.stage is None:
            return {'position': np.zeros(3, dtype=float), 'axis': np.array([0.0, 0.0, 1.0], dtype=float)}

        joint_prim = self.stage.GetPrimAtPath(str(candidate.get('joint_prim_path', '') or '').strip())
        if joint_prim is None or not joint_prim.IsValid():
            return {'position': np.zeros(3, dtype=float), 'axis': np.array([0.0, 0.0, 1.0], dtype=float)}

        axis_local = self._axis_token_to_vector(self._get_token_attr(joint_prim, 'physics:axis'))
        body_specs = [
            (list(candidate.get('body0_targets') or []), 'physics:localPos0', 'physics:localRot0'),
            (list(candidate.get('body1_targets') or []), 'physics:localPos1', 'physics:localRot1'),
        ]
        for body_targets, pos_attr, rot_attr in body_specs:
            if not body_targets:
                continue
            body_path = str(body_targets[0] or '').strip()
            body_prim = self.stage.GetPrimAtPath(body_path)
            if body_prim is None or not body_prim.IsValid():
                continue
            local_position = self._get_vec3_attr(joint_prim, pos_attr)
            if local_position is None:
                local_position = np.zeros(3, dtype=float)
            local_axis = np.array(axis_local, dtype=float)
            local_rotation = self._get_quaternion_attr(joint_prim, rot_attr)
            if local_rotation is not None:
                local_axis = self._quaternion_to_rotation_matrix(local_rotation) @ local_axis
            try:
                world_matrix = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                world_position = self._transform_point_to_world(world_matrix, local_position)
                world_axis = self._transform_direction_to_world(world_matrix, local_axis)
                return {
                    'position': world_position,
                    'axis': self._normalize_vector(world_axis, fallback=np.array([0.0, 0.0, 1.0], dtype=float)),
                    'body_path': body_path,
                    'axis_token': self._get_token_attr(joint_prim, 'physics:axis') or 'X',
                }
            except Exception:
                continue

        pose = self._get_raw_pose_for_prim_path(str(joint_prim.GetPath()), resolve_pose_prim=False)
        if pose is not None:
            return {
                'position': np.array(pose[0], dtype=float),
                'axis': self._normalize_vector(axis_local, fallback=np.array([0.0, 0.0, 1.0], dtype=float)),
                'body_path': '',
                'axis_token': self._get_token_attr(joint_prim, 'physics:axis') or 'X',
            }
        return {'position': np.zeros(3, dtype=float), 'axis': np.array([0.0, 0.0, 1.0], dtype=float)}

    @staticmethod
    def _rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
        matrix = np.array(rotation_matrix, dtype=float)
        if matrix.shape != (3, 3) or np.any(~np.isfinite(matrix)):
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = float(np.sqrt(trace + 1.0) * 2.0)
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=float,
            )
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = float(np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0)
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=float,
            )
        elif matrix[1, 1] > matrix[2, 2]:
            scale = float(np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0)
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = float(np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0)
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=float,
            )
        return IsaacSimAppRunner._normalize_quaternion(quat)

    @staticmethod
    def _build_orientation_from_approach_direction(
        approach_direction: np.ndarray,
        *,
        up_hint: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        approach = IsaacSimAppRunner._normalize_vector(
            approach_direction,
            fallback=np.array([1.0, 0.0, 0.0], dtype=float),
        )
        x_axis = approach
        up = IsaacSimAppRunner._normalize_vector(
            np.array([0.0, 0.0, 1.0], dtype=float) if up_hint is None else np.array(up_hint, dtype=float),
            fallback=np.array([0.0, 0.0, 1.0], dtype=float),
        )
        if abs(float(np.dot(up, x_axis))) > 0.95:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(up, x_axis))) > 0.95:
                up = np.array([0.0, 1.0, 0.0], dtype=float)

        y_axis = up - x_axis * float(np.dot(up, x_axis))
        y_axis = IsaacSimAppRunner._normalize_vector(y_axis, fallback=np.array([0.0, 1.0, 0.0], dtype=float))
        z_axis = np.cross(x_axis, y_axis)
        z_axis = IsaacSimAppRunner._normalize_vector(z_axis, fallback=np.array([0.0, 0.0, 1.0], dtype=float))
        y_axis = np.cross(z_axis, x_axis)
        y_axis = IsaacSimAppRunner._normalize_vector(y_axis, fallback=np.array([0.0, 1.0, 0.0], dtype=float))
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        return IsaacSimAppRunner._rotation_matrix_to_quaternion(rotation)

    def _resolve_mock_suction_door_frame(
        self,
        candidate: Dict[str, Any],
        contact_position: np.ndarray,
    ) -> Dict[str, Any]:
        joint_frame = self._get_container_joint_world_frame(candidate)
        hinge_position = np.array(joint_frame.get('position'), dtype=float)
        hinge_axis = self._normalize_vector(np.array(joint_frame.get('axis'), dtype=float))
        contact_position = np.array(contact_position, dtype=float)

        radial = contact_position - hinge_position
        radial -= hinge_axis * float(np.dot(radial, hinge_axis))
        radial = self._normalize_vector(radial, fallback=np.array([1.0, 0.0, 0.0], dtype=float))
        door_normal = self._normalize_vector(
            np.cross(hinge_axis, radial),
            fallback=np.array([0.0, 1.0, 0.0], dtype=float),
        )

        robot_pose = self._get_end_effector_pose_raw() or self._get_robot_base_pose_raw()
        robot_position = np.array(robot_pose[0], dtype=float) if robot_pose is not None else None
        if robot_position is not None and float(np.dot(door_normal, robot_position - contact_position)) < 0.0:
            door_normal = -door_normal

        approach_direction = -door_normal
        orientation = self._build_orientation_from_approach_direction(
            approach_direction,
            up_hint=hinge_axis,
        )
        return {
            'contact_position': contact_position,
            'hinge_position': hinge_position,
            'hinge_axis': hinge_axis,
            'radial_direction': radial,
            'door_normal': door_normal,
            'approach_direction': approach_direction,
            'orientation': orientation,
            'joint_frame': dict(joint_frame),
        }

    def _mock_follow_container_door_with_end_effector(
        self,
        candidate: Dict[str, Any],
        target_position: float,
        contact_frame: Dict[str, Any],
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = time.monotonic()
        requested_duration = max(float(args.get('timeout_sec', self.DEFAULT_CONTAINER_OPEN_TIMEOUT_SEC)), float(self.DEFAULT_PHYSICS_DT))
        joint_prim_path = str(candidate.get('joint_prim_path', '') or '').strip()
        existing_target = self._get_container_door_drive_target(joint_prim_path)
        start_target = (
            existing_target
            if existing_target is not None
            else self._infer_container_door_closed_target(candidate, target_position)
        )
        target_delta_value = float(target_position) - float(start_target)
        angle_reference_values = [
            start_target,
            target_position,
            candidate.get('lower_limit'),
            candidate.get('upper_limit'),
        ]
        angle_values_are_degrees = False
        for value in angle_reference_values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric) and abs(numeric) > float(2.0 * np.pi + 0.2):
                angle_values_are_degrees = True
                break

        def _door_delta_value_to_radians(delta_value: float) -> float:
            delta = float(delta_value)
            return float(np.deg2rad(delta)) if angle_values_are_degrees else delta

        target_delta_rad = _door_delta_value_to_radians(target_delta_value)
        contact_start = np.array(contact_frame['contact_position'], dtype=float)
        hinge_position = np.array(contact_frame['hinge_position'], dtype=float)
        hinge_axis = self._normalize_vector(np.array(contact_frame['hinge_axis'], dtype=float))
        door_normal = self._normalize_vector(np.array(contact_frame['door_normal'], dtype=float))
        contact_offset = float(args.get('mock_suction_contact_offset', args.get('suction_contact_offset', 0.0)))
        follow_timeout_sec = max(
            float(args.get('mock_follow_timeout_sec', args.get('follow_timeout_sec', max(requested_duration, 45.0)))),
            float(self.DEFAULT_PHYSICS_DT),
        )
        angle_step_rad = max(
            float(args.get('mock_follow_angle_step_rad', args.get('follow_angle_step_rad', 0.03))),
            1e-3,
        )
        waypoint_count = max(1, int(np.ceil(abs(target_delta_rad) / angle_step_rad))) if abs(target_delta_rad) > 1e-9 else 1
        segment_timeout_arg = args.get('mock_follow_segment_timeout_sec', args.get('follow_segment_timeout_sec', None))
        if segment_timeout_arg is None:
            segment_timeout_sec = max(0.75, follow_timeout_sec / float(max(waypoint_count, 1)))
        else:
            segment_timeout_sec = max(float(segment_timeout_arg), float(self.DEFAULT_PHYSICS_DT))
        position_tolerance = float(
            args.get(
                'mock_follow_position_tolerance',
                args.get('mock_approach_position_tolerance', max(float(self.DEFAULT_MOVE_POSITION_TOLERANCE), 0.025)),
            )
        )
        orientation_tolerance = float(
            args.get(
                'mock_follow_orientation_tolerance',
                args.get('mock_approach_orientation_tolerance', max(float(self.DEFAULT_MOVE_ORIENTATION_TOLERANCE), 0.25)),
            )
        )
        settle_steps = max(0, int(args.get('mock_follow_settle_steps', args.get('follow_settle_steps', 3))))

        step_count = 0
        last_contact_position = contact_start
        last_target = float(start_target)
        last_distance = None
        last_orientation_distance = None
        reached = True
        reached_waypoint_count = 0
        last_applied_alpha = 0.0
        ee_start_position = contact_start + door_normal * contact_offset
        self._active_arm_target = (
            ee_start_position,
            np.array(contact_frame['orientation'], dtype=float),
        )

        def _pose_for_alpha(alpha: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
            clamped_alpha = min(max(float(alpha), 0.0), 1.0)
            target_value = float(start_target + target_delta_value * clamped_alpha)
            current_delta = float(target_delta_rad * clamped_alpha)
            current_contact_position = self._rotate_point_about_axis(
                contact_start,
                hinge_position,
                hinge_axis,
                current_delta,
            )
            current_normal = self._rotate_point_about_axis(
                contact_start + door_normal,
                hinge_position,
                hinge_axis,
                current_delta,
            ) - current_contact_position
            current_normal = self._normalize_vector(current_normal, fallback=door_normal)
            current_orientation = self._build_orientation_from_approach_direction(
                -current_normal,
                up_hint=hinge_axis,
            )
            return target_value, current_contact_position, current_normal, current_orientation

        try:
            self._apply_container_door_open_drive(
                joint_prim_path,
                start_target,
                stiffness=args.get('stiffness'),
                damping=args.get('damping'),
                max_force=args.get('max_force'),
            )
            self._step_robot(render=True)
            step_count += 1

            for waypoint_index in range(1, waypoint_count + 1):
                if time.monotonic() - start >= follow_timeout_sec:
                    reached = False
                    break

                alpha = float(waypoint_index / waypoint_count)
                target_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                waypoint_target, waypoint_contact, waypoint_normal, waypoint_orientation = _pose_for_alpha(target_alpha)
                waypoint_position = waypoint_contact + waypoint_normal * contact_offset
                self._active_arm_target = (
                    waypoint_position,
                    waypoint_orientation,
                )

                waypoint_reached = False
                segment_start = time.monotonic()
                while True:
                    self._step_robot(render=True)
                    step_count += 1

                    ee_pose = self._get_end_effector_pose_raw()
                    if ee_pose is not None:
                        ee_position, ee_orientation = ee_pose
                        last_distance = float(np.linalg.norm(np.array(ee_position, dtype=float) - waypoint_position))
                        last_orientation_distance = self._quaternion_angular_distance(
                            np.array(ee_orientation, dtype=float),
                            waypoint_orientation,
                        )

                        if abs(target_delta_rad) > 1e-9:
                            achieved_angle = self._signed_angle_about_axis(
                                ee_start_position - hinge_position,
                                np.array(ee_position, dtype=float) - hinge_position,
                                hinge_axis,
                            )
                            achieved_alpha = float(achieved_angle / target_delta_rad)
                            drive_alpha = min(max(achieved_alpha, last_applied_alpha), target_alpha)
                        else:
                            drive_alpha = target_alpha

                        if drive_alpha > last_applied_alpha + 1e-5 or target_alpha >= 1.0:
                            drive_target, last_contact_position, _, _ = _pose_for_alpha(drive_alpha)
                            self._apply_container_door_open_drive(
                                joint_prim_path,
                                drive_target,
                                stiffness=args.get('stiffness'),
                                damping=args.get('damping'),
                                max_force=args.get('max_force'),
                            )
                            last_target = float(drive_target)
                            last_applied_alpha = float(drive_alpha)

                        if last_distance <= position_tolerance and last_orientation_distance <= orientation_tolerance:
                            waypoint_reached = True
                            reached_waypoint_count = waypoint_index
                            final_target, last_contact_position, _, _ = _pose_for_alpha(target_alpha)
                            self._apply_container_door_open_drive(
                                joint_prim_path,
                                final_target,
                                stiffness=args.get('stiffness'),
                                damping=args.get('damping'),
                                max_force=args.get('max_force'),
                            )
                            last_target = float(final_target)
                            last_applied_alpha = float(target_alpha)
                            break

                    if time.monotonic() - segment_start >= segment_timeout_sec:
                        break
                    if time.monotonic() - start >= follow_timeout_sec:
                        break

                if not waypoint_reached:
                    reached = False
                    break

            if reached:
                final_target, last_contact_position, final_normal, final_orientation = _pose_for_alpha(1.0)
                self._active_arm_target = (
                    last_contact_position + final_normal * contact_offset,
                    final_orientation,
                )
                self._apply_container_door_open_drive(
                    joint_prim_path,
                    final_target,
                    stiffness=args.get('stiffness'),
                    damping=args.get('damping'),
                    max_force=args.get('max_force'),
                )
                last_target = float(final_target)
                for _ in range(settle_steps):
                    self._step_robot(render=True)
                    step_count += 1
        finally:
            self._active_arm_target = None

        return {
            'joint_prim_path': joint_prim_path,
            'start_target_position': float(start_target),
            'target_position': float(target_position),
            'last_target_position': float(last_target),
            'last_contact_position': last_contact_position.astype(float).tolist(),
            'reached': bool(reached),
            'waypoint_count': int(waypoint_count),
            'reached_waypoint_count': int(reached_waypoint_count),
            'angle_values_are_degrees': bool(angle_values_are_degrees),
            'target_delta_rad': float(target_delta_rad),
            'follow_timeout_sec': float(follow_timeout_sec),
            'segment_timeout_sec': float(segment_timeout_sec),
            'position_tolerance': float(position_tolerance),
            'orientation_tolerance': float(orientation_tolerance),
            'last_distance': last_distance,
            'last_orientation_distance': last_orientation_distance,
            'elapsed_sec': float(time.monotonic() - start),
            'step_count': int(step_count),
        }

    def _build_container_open_waypoint(
        self,
        candidate: Dict[str, Any],
        grasp_position: np.ndarray,
        target_position: float,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        joint_frame = self._get_container_joint_world_frame(candidate)
        hinge_position = np.array(joint_frame.get('position'), dtype=float)
        hinge_axis = self._normalize_vector(np.array(joint_frame.get('axis'), dtype=float))
        start_target = self._infer_container_door_closed_target(candidate, target_position)
        target_delta_rad = self._joint_angle_value_to_radians(float(target_position)) - self._joint_angle_value_to_radians(float(start_target))
        if abs(target_delta_rad) <= 1e-3:
            target_delta_rad = float(self.DEFAULT_ROBOT_CONTAINER_OPEN_WAYPOINT_ANGLE_RAD)

        requested_waypoint_angle = args.get('robot_waypoint_angle', args.get('waypoint_angle', None))
        if requested_waypoint_angle is not None:
            max_angle = abs(self._joint_angle_value_to_radians(float(requested_waypoint_angle)))
        else:
            max_angle = float(self.DEFAULT_ROBOT_CONTAINER_OPEN_WAYPOINT_ANGLE_RAD)
        max_angle = max(max_angle, 1e-3)
        open_angle = float(np.sign(target_delta_rad or 1.0) * min(abs(target_delta_rad), max_angle))

        grasp_position = np.array(grasp_position, dtype=float)
        waypoint = self._rotate_point_about_axis(grasp_position, hinge_position, hinge_axis, open_angle)
        displacement = waypoint - grasp_position
        max_distance = float(args.get('robot_waypoint_max_distance', self.DEFAULT_ROBOT_CONTAINER_OPEN_MAX_WAYPOINT_DISTANCE))
        displacement_norm = float(np.linalg.norm(displacement))
        if max_distance > 0.0 and displacement_norm > max_distance:
            waypoint = grasp_position + displacement / displacement_norm * max_distance

        return {
            'position': waypoint,
            'hinge_position': hinge_position,
            'hinge_axis': hinge_axis,
            'open_angle_rad': open_angle,
            'open_angle_degrees': float(np.rad2deg(open_angle)),
            'joint_frame': {
                'position': hinge_position.astype(float).tolist(),
                'axis': hinge_axis.astype(float).tolist(),
                'body_path': str(joint_frame.get('body_path', '') or ''),
                'axis_token': str(joint_frame.get('axis_token', '') or ''),
            },
        }

    def _open_container_with_robot(
        self,
        container_name: str,
        container_prim_path: str,
        candidates: list[Dict[str, Any]],
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.robot_controller is None:
            raise RuntimeError('robot_not_loaded')
        if not candidates:
            raise RuntimeError(f'door_joint_not_found:{container_name}')

        candidate = candidates[0]
        requested_target = args.get('target_position', args.get('target_angle', None))
        target_position = (
            float(requested_target)
            if requested_target is not None
            else float(candidate['target_position'])
        )
        grasp_target = self._resolve_container_door_grasp_target(
            container_name,
            container_prim_path,
            candidate,
        )
        preferred_grasp = dict(grasp_target.get('preferred_grasp') or {})
        grasp_pose = dict(preferred_grasp.get('grasp_pose') or {})
        if not grasp_pose:
            raise RuntimeError('door_grasp_pose_unavailable')

        grasp_position, grasp_orientation, normalized_grasp_pose = self._pose_command_to_raw(grasp_pose)
        try:
            self._set_manipulation_torso_height_from_grasp_height(float(grasp_position[2]))
        except Exception as exc:
            print(f'[Container robot open] torso adjustment failed: {exc}')

        open_gripper_result = self._wait_for_gripper(
            gripper_action='open',
            timeout_sec=float(self.DEFAULT_GRIPPER_TIMEOUT_SEC),
            tolerance=float(self.DEFAULT_GRIPPER_POSITION_TOLERANCE),
        )

        pre_grasp_distance = float(
            args.get('handle_pre_grasp_distance', self.DEFAULT_ROBOT_CONTAINER_HANDLE_PRE_GRASP_DISTANCE)
        )
        approach_motion = self._wait_for_pre_grasp_motion(
            target_position=grasp_position,
            target_orientation=grasp_orientation,
            timeout_sec=float(args.get('robot_approach_timeout_sec', self.DEFAULT_MOVE_TIMEOUT_SEC)),
            position_tolerance=float(self.DEFAULT_MOVE_POSITION_TOLERANCE),
            orientation_tolerance=float(self.DEFAULT_MOVE_ORIENTATION_TOLERANCE),
            pre_grasp_distance=pre_grasp_distance,
            linear_step=float(args.get('handle_linear_step', self.DEFAULT_PRE_GRASP_LINEAR_STEP)),
        )
        if not bool(approach_motion.get('reached', False)):
            raise RuntimeError('door_handle_grasp_pose_not_reached')

        close_gripper_result = self._wait_for_gripper(
            gripper_action='close',
            timeout_sec=float(self.DEFAULT_GRIPPER_TIMEOUT_SEC),
            tolerance=float(self.DEFAULT_GRIPPER_POSITION_TOLERANCE),
        )

        waypoint: Dict[str, Any] = {}
        waypoint_position = np.array(grasp_position, dtype=float)
        pull_motion: Dict[str, Any] = {}
        try:
            waypoint = self._build_container_open_waypoint(candidate, grasp_position, target_position, args)
            waypoint_position = np.array(waypoint['position'], dtype=float)
            self._active_gripper_action = 'close'
            pull_motion = self._wait_for_motion(
                target_position=waypoint_position,
                target_orientation=grasp_orientation,
                timeout_sec=float(args.get('robot_pull_timeout_sec', self.DEFAULT_MOVE_TIMEOUT_SEC)),
                position_tolerance=float(args.get('robot_pull_position_tolerance', self.DEFAULT_MOVE_POSITION_TOLERANCE)),
                orientation_tolerance=float(args.get('robot_pull_orientation_tolerance', self.DEFAULT_MOVE_ORIENTATION_TOLERANCE)),
            )
        finally:
            self._active_gripper_action = None
            release_gripper_result = self._wait_for_gripper(
                gripper_action='open',
                timeout_sec=float(self.DEFAULT_GRIPPER_TIMEOUT_SEC),
                tolerance=float(self.DEFAULT_GRIPPER_POSITION_TOLERANCE),
            )
            self.gripper_state = 'open'
            self.object_in_gripper = False
            self.grasped_object_name = ''
            self.grasped_object_mass = None

        return {
            'opened': bool(pull_motion.get('reached', False)),
            'container_name': container_name,
            'container_prim_path': container_prim_path,
            'selected_candidate': {
                **candidate,
                'target_position': target_position,
            },
            'grasp_target': {
                'door_prim_path': str(grasp_target.get('door_prim_path', '') or ''),
                'handle_prim_path': str(grasp_target.get('handle_prim_path', '') or ''),
                'grasp_prim_path': str(grasp_target.get('grasp_prim_path', '') or ''),
                'grasp_target_name': str(grasp_target.get('grasp_target_name', '') or ''),
                'fallback_position': grasp_target.get('fallback_position'),
            },
            'grasp_pose': normalized_grasp_pose,
            'grasp_pose_source': str(preferred_grasp.get('grasp_pose_source', '') or ''),
            'grasp_pose_diagnostics': dict(preferred_grasp.get('diagnostics') or {}),
            'waypoint': {
                'position': {
                    'x': float(waypoint_position[0]),
                    'y': float(waypoint_position[1]),
                    'z': float(waypoint_position[2]),
                },
                'hinge_position': {
                    'x': float(waypoint['hinge_position'][0]),
                    'y': float(waypoint['hinge_position'][1]),
                    'z': float(waypoint['hinge_position'][2]),
                },
                'hinge_axis': {
                    'x': float(waypoint['hinge_axis'][0]),
                    'y': float(waypoint['hinge_axis'][1]),
                    'z': float(waypoint['hinge_axis'][2]),
                },
                'open_angle_rad': float(waypoint['open_angle_rad']),
                'open_angle_degrees': float(waypoint['open_angle_degrees']),
                'joint_frame': dict(waypoint.get('joint_frame') or {}),
            },
            'open_gripper': open_gripper_result,
            'approach_motion': approach_motion,
            'close_gripper': close_gripper_result,
            'pull_motion': pull_motion,
            'release_gripper': release_gripper_result,
            'elapsed_sec': float(
                float(open_gripper_result.get('elapsed_sec', 0.0) or 0.0)
                + float(approach_motion.get('elapsed_sec', 0.0) or 0.0)
                + float(close_gripper_result.get('elapsed_sec', 0.0) or 0.0)
                + float(pull_motion.get('elapsed_sec', 0.0) or 0.0)
                + float(release_gripper_result.get('elapsed_sec', 0.0) or 0.0)
            ),
        }

    def _mock_open_container_with_robot(
        self,
        container_name: str,
        container_prim_path: str,
        candidates: list[Dict[str, Any]],
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.robot_controller is None:
            raise RuntimeError('robot_not_loaded')
        if not candidates:
            raise RuntimeError(f'door_joint_not_found:{container_name}')

        candidate = candidates[0]
        requested_target = args.get('target_position', args.get('target_angle', None))
        target_position = (
            float(requested_target)
            if requested_target is not None
            else float(candidate['target_position'])
        )
        contact_target = self._resolve_mock_container_door_contact_target(
            container_name,
            container_prim_path,
            candidate,
        )
        contact_position = np.array(contact_target.get('contact_position'), dtype=float)
        contact_frame = self._resolve_mock_suction_door_frame(candidate, contact_position)
        contact_orientation = np.array(contact_frame['orientation'], dtype=float)
        contact_position = np.array(contact_frame['contact_position'], dtype=float)
        normalized_contact_pose = self._pose_dict_from_raw(contact_position, contact_orientation)
        target_collision_prim_path = str(contact_target.get('door_prim_path', '') or '') or container_prim_path
        persist_collision_filtering = self._coerce_bool(
            args.get('persist_mock_collision_filtering', True),
            default=True,
        )

        collision_filter_pairs: list[tuple[str, str]] = []
        approach_motion: Dict[str, Any] = {}
        follow_motion: Dict[str, Any] = {}
        open_gripper_result: Dict[str, Any] = {}
        try:
            collision_filter_pairs = self._merge_collision_filter_pairs(
                self._apply_grasp_latch_collision_filtering(container_prim_path),
                self._apply_grasp_latch_collision_filtering(target_collision_prim_path),
            )
            if persist_collision_filtering:
                self._remember_persistent_arm_collision_filtering(collision_filter_pairs)
            open_gripper_result = self._wait_for_gripper(
                gripper_action='open',
                timeout_sec=float(self.DEFAULT_GRIPPER_TIMEOUT_SEC),
                tolerance=float(self.DEFAULT_GRIPPER_POSITION_TOLERANCE),
            )
            try:
                self._set_manipulation_torso_height_from_grasp_height(float(contact_position[2]))
            except Exception as exc:
                print(f'[Container mock open] torso adjustment failed: {exc}')

            pre_grasp_distance = float(
                args.get('mock_suction_pre_grasp_distance', self.DEFAULT_ROBOT_CONTAINER_HANDLE_PRE_GRASP_DISTANCE)
            )
            approach_motion = self._wait_for_pre_grasp_motion(
                target_position=contact_position,
                target_orientation=contact_orientation,
                timeout_sec=float(args.get('mock_approach_timeout_sec', self.DEFAULT_MOVE_TIMEOUT_SEC)),
                position_tolerance=float(args.get('mock_approach_position_tolerance', self.DEFAULT_MOVE_POSITION_TOLERANCE)),
                orientation_tolerance=float(args.get('mock_approach_orientation_tolerance', self.DEFAULT_MOVE_ORIENTATION_TOLERANCE)),
                pre_grasp_distance=pre_grasp_distance,
                linear_step=float(args.get('mock_suction_linear_step', self.DEFAULT_PRE_GRASP_LINEAR_STEP)),
            )
            if not bool(approach_motion.get('reached', False)):
                raise RuntimeError('mock_suction_contact_pose_not_reached')

            follow_motion = self._mock_follow_container_door_with_end_effector(
                candidate,
                target_position,
                contact_frame,
                args,
            )
        finally:
            self._active_arm_target = None
            if not persist_collision_filtering:
                self._remove_arm_collision_filtering(collision_filter_pairs)
            self.gripper_state = 'open'
            self.object_in_gripper = False
            self.grasped_object_name = ''
            self.grasped_object_mass = None

        return {
            'opened': bool(follow_motion.get('reached', False)),
            'container_name': container_name,
            'container_prim_path': container_prim_path,
            'selected_candidate': {
                **candidate,
                'target_position': target_position,
            },
            'contact_target': {
                'door_prim_path': str(contact_target.get('door_prim_path', '') or ''),
                'contact_source': str(contact_target.get('contact_source', '') or ''),
                'contact_position': contact_position.astype(float).tolist(),
            },
            'contact_pose': normalized_contact_pose,
            'contact_frame': {
                'hinge_position': contact_frame['hinge_position'].astype(float).tolist(),
                'hinge_axis': contact_frame['hinge_axis'].astype(float).tolist(),
                'door_normal': contact_frame['door_normal'].astype(float).tolist(),
                'approach_direction': contact_frame['approach_direction'].astype(float).tolist(),
                'joint_frame': dict(contact_frame.get('joint_frame') or {}),
            },
            'grasp_pose_source': 'door_far_from_hinge_no_server',
            'grasp_pose_diagnostics': {
                'uses_grasp_server': False,
                'contact_source': str(contact_target.get('contact_source', '') or ''),
            },
            'collision_filter_pair_count': len(collision_filter_pairs),
            'collision_filtering_persistent': bool(persist_collision_filtering),
            'persistent_collision_filter_pairs': [
                {
                    'robot_collision_path': str(robot_collision_path),
                    'target_collision_path': str(target_collision_path),
                }
                for robot_collision_path, target_collision_path in collision_filter_pairs
            ]
            if persist_collision_filtering
            else [],
            'open_gripper': open_gripper_result,
            'approach_motion': approach_motion,
            'follow_motion': follow_motion,
            'elapsed_sec': float(
                float(open_gripper_result.get('elapsed_sec', 0.0) or 0.0)
                + float(approach_motion.get('elapsed_sec', 0.0) or 0.0)
                + float(follow_motion.get('elapsed_sec', 0.0) or 0.0)
            ),
        }

    def _direct_open_container_by_name(
        self,
        container_name: str,
        container_prim_path: str,
        candidates: list[Dict[str, Any]],
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not candidates:
            raise ValueError(f'door_joint_not_found:{container_name}')

        candidate = candidates[0]
        requested_target = args.get('target_position', args.get('target_angle', None))
        target_position = (
            float(requested_target)
            if requested_target is not None
            else float(candidate['target_position'])
        )
        target_collision_prim_path = self._resolve_container_door_prim_path(container_prim_path, candidate)
        if not target_collision_prim_path:
            target_collision_prim_path = container_prim_path
        persist_collision_filtering = self._coerce_bool(
            args.get('persist_direct_open_collision_filtering', True),
            default=True,
        )
        collision_filter_pairs = self._apply_grasp_latch_collision_filtering(target_collision_prim_path)
        if persist_collision_filtering:
            self._remember_persistent_arm_collision_filtering(collision_filter_pairs)

        timeout_sec = float(args.get('timeout_sec', self.DEFAULT_CONTAINER_OPEN_TIMEOUT_SEC))
        drive_payload = self._slowly_open_container_door(
            candidate,
            target_position,
            timeout_sec=timeout_sec,
            stiffness=args.get('stiffness'),
            damping=args.get('damping'),
            max_force=args.get('max_force'),
        )
        if not persist_collision_filtering:
            self._remove_arm_collision_filtering(collision_filter_pairs)

        return {
            'container_name': container_name,
            'container_prim_path': container_prim_path,
            'target_collision_prim_path': target_collision_prim_path,
            'selected_candidate': {
                **candidate,
                'target_position': target_position,
            },
            'candidate_count': len(candidates),
            'candidates': candidates,
            'drive': drive_payload,
            'collision_filter_pair_count': len(collision_filter_pairs),
            'collision_filtering_persistent': bool(persist_collision_filtering),
            'persistent_collision_filter_pairs': [
                {
                    'robot_collision_path': str(robot_collision_path),
                    'target_collision_path': str(target_collision_path),
                }
                for robot_collision_path, target_collision_path in collision_filter_pairs
            ]
            if persist_collision_filtering
            else [],
            'elapsed_sec': float(drive_payload.get('elapsed_sec', 0.0) or 0.0),
        }

    def _open_container_by_name(self, container_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self._refresh_scene_index()

        normalized_name = str(container_name or '').strip()
        if not normalized_name:
            raise ValueError('container_name_required')

        container_prim_path = self._lookup_object_prim_path(normalized_name)
        if not container_prim_path:
            raise ValueError(f'object_not_found:{normalized_name}')

        mass_payload = self._set_container_base_link_mass(
            container_prim_path,
            float(args.get('base_link_mass', self.DEFAULT_CONTAINER_BASE_LINK_MASS)),
        )

        candidates = self._find_container_door_joint_candidates(container_prim_path)
        if not candidates:
            raise ValueError(f'door_joint_not_found:{normalized_name}')

        physical_robot_attempt: Dict[str, Any] = {}
        mock_robot_attempt: Dict[str, Any] = {}
        force_direct_open = self._coerce_bool(args.get('force_direct_open', False), default=False)
        robot_only = self._coerce_bool(args.get('robot_only', False), default=False)
        enable_physical_robot_open = self._coerce_bool(
            args.get('enable_physical_robot_open', self.DEFAULT_ENABLE_PHYSICAL_ROBOT_CONTAINER_OPEN),
            default=bool(self.DEFAULT_ENABLE_PHYSICAL_ROBOT_CONTAINER_OPEN),
        )
        enable_mock_robot_open = self._coerce_bool(
            args.get('enable_mock_robot_open', self.DEFAULT_ENABLE_MOCK_ROBOT_CONTAINER_OPEN),
            default=bool(self.DEFAULT_ENABLE_MOCK_ROBOT_CONTAINER_OPEN),
        )
        if not force_direct_open and enable_physical_robot_open:
            try:
                physical_robot_attempt = self._open_container_with_robot(
                    normalized_name,
                    container_prim_path,
                    candidates,
                    args,
                )
                if bool(physical_robot_attempt.get('opened', False)):
                    return {
                        'mode': 'physical_robot',
                        'container_name': normalized_name,
                        'container_prim_path': container_prim_path,
                        'mass': mass_payload,
                        'candidate_count': len(candidates),
                        'candidates': candidates,
                        'physical_robot_attempt': physical_robot_attempt,
                        'mock_robot_attempt': mock_robot_attempt,
                        'direct_open_fallback_used': False,
                        'elapsed_sec': float(physical_robot_attempt.get('elapsed_sec', 0.0) or 0.0),
                    }
            except Exception as exc:
                physical_robot_attempt = {
                    'opened': False,
                    'error': f'{exc.__class__.__name__}: {exc}',
                }
                print(f'[Container physical robot open] failed: {exc}')
                if robot_only:
                    raise RuntimeError(f'robot_container_open_failed:{exc}') from exc

        if not force_direct_open and enable_mock_robot_open:
            try:
                mock_robot_attempt = self._mock_open_container_with_robot(
                    normalized_name,
                    container_prim_path,
                    candidates,
                    args,
                )
                if bool(mock_robot_attempt.get('opened', False)):
                    return {
                        'mode': 'mock_robot_suction',
                        'container_name': normalized_name,
                        'container_prim_path': container_prim_path,
                        'mass': mass_payload,
                        'candidate_count': len(candidates),
                        'candidates': candidates,
                        'physical_robot_attempt': physical_robot_attempt,
                        'mock_robot_attempt': mock_robot_attempt,
                        'direct_open_fallback_used': False,
                        'elapsed_sec': float(mock_robot_attempt.get('elapsed_sec', 0.0) or 0.0),
                    }
            except Exception as exc:
                mock_robot_attempt = {
                    'opened': False,
                    'error': f'{exc.__class__.__name__}: {exc}',
                }
                print(f'[Container mock robot open] failed, falling back to direct drive: {exc}')
                if robot_only:
                    raise RuntimeError(f'mock_robot_container_open_failed:{exc}') from exc

        direct_payload = self._direct_open_container_by_name(
            normalized_name,
            container_prim_path,
            candidates,
            args,
        )

        return {
            **direct_payload,
            'mode': 'direct_drive',
            'mass': mass_payload,
            'physical_robot_attempt': physical_robot_attempt,
            'mock_robot_attempt': mock_robot_attempt,
            'direct_open_fallback_used': bool(physical_robot_attempt or mock_robot_attempt),
        }

    def _apply_gripper_drive_overrides(self) -> None:
        if self.robot_controller is None or self.stage is None:
            return

        try:
            from pxr import UsdPhysics
        except Exception as exc:
            print(f'[Gripper drive] unavailable: failed to import UsdPhysics: {exc}')
            return

        joint_names = list(getattr(self.robot_controller, 'gripper_joint_names', []) or [])
        if not joint_names:
            joint_names = ['l_gripper_finger_joint', 'r_gripper_finger_joint']

        stiffness = float(self.DEFAULT_GRIPPER_DRIVE_STIFFNESS)
        damping = float(self.DEFAULT_GRIPPER_DRIVE_DAMPING)
        max_force = float(self.DEFAULT_GRIPPER_DRIVE_MAX_FORCE)

        for joint_name in joint_names:
            joint_prim_path = self._lookup_robot_joint_prim_path(joint_name)
            if not joint_prim_path:
                print(f'[Gripper drive] joint not found name={joint_name}')
                continue

            joint_prim = self.stage.GetPrimAtPath(joint_prim_path)
            if joint_prim is None or not joint_prim.IsValid():
                print(f'[Gripper drive] invalid joint prim name={joint_name} prim={joint_prim_path}')
                continue

            try:
                drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, 'linear')
                drive_api.CreateStiffnessAttr().Set(stiffness)
                drive_api.CreateDampingAttr().Set(damping)
                drive_api.CreateMaxForceAttr().Set(max_force)
                print(
                    '[Gripper drive] override applied '
                    f'joint={joint_name} '
                    f'prim={joint_prim_path} '
                    f'drive=linear '
                    f'stiffness={stiffness:.4f} '
                    f'damping={damping:.4f} '
                    f'max_force={max_force:.4f}'
                )
            except Exception as exc:
                print(
                    '[Gripper drive] override failed '
                    f'joint={joint_name} prim={joint_prim_path}: {exc}'
                )

    def _apply_gripper_high_friction_material(self) -> None:
        if self.robot_controller is None:
            return

        robot_prim_path = str(getattr(self.robot_controller, 'robot_prim_path', '') or '').strip()
        if not robot_prim_path:
            return

        for link_name in self.GRIPPER_HIGH_FRICTION_LINK_NAMES:
            finger_prim_path = f'{robot_prim_path}/{link_name}'
            # print(f'[Physics material] applying gripper friction material to prim={finger_prim_path}')
            material_name = 'HighFrictionGraspMaterial'
            material_applied = self._bind_high_friction_material_to_prim(
                finger_prim_path,
                material_name=material_name,
            )
            print(
                '[Physics material] gripper friction material bind result '
                f'link={link_name} '
                f'prim={finger_prim_path} '
                f'material=/World/Physics_Materials/{material_name} '
                f'applied={bool(material_applied)} '
                f'static_friction={float(self.DEFAULT_GRASP_STATIC_FRICTION):.4f} '
                f'dynamic_friction={float(self.DEFAULT_GRASP_DYNAMIC_FRICTION):.4f} '
                f'restitution={float(self.DEFAULT_GRASP_RESTITUTION):.4f}'
            )

    def _prepare_grasp_target_physics(self, obj_name: str) -> None:
        resolved_name, target_prim_path = self._resolve_grasp_target(obj_name)
        if not target_prim_path:
            # print(f'[Physics material] no grasp target prim resolved for obj_name={obj_name}')
            return

        self._prepare_grasp_target_mass_override(obj_name=resolved_name or obj_name)
        material_target_path = self._resolve_grasp_material_target_prim_path(target_prim_path)
        # print(
        #     '[Physics material] applying target friction material '
        #     f'obj_name={resolved_name or obj_name} target_prim={target_prim_path} '
        #     f'material_target={material_target_path}'
        # )
        material_name = 'HighFrictionGraspMaterial'
        material_applied = self._bind_high_friction_material_to_prim(
            material_target_path,
            material_name=material_name,
        )
        if material_applied:
            print(
                '[Physics material] grasp target friction material applied '
                f'object={resolved_name or obj_name} '
                f'target_prim={target_prim_path} '
                f'material_prim={material_target_path} '
                f'material=/World/Physics_Materials/{material_name} '
                f'static_friction={float(self.DEFAULT_GRASP_STATIC_FRICTION):.4f} '
                f'dynamic_friction={float(self.DEFAULT_GRASP_DYNAMIC_FRICTION):.4f} '
                f'restitution={float(self.DEFAULT_GRASP_RESTITUTION):.4f}'
            )

    def _get_active_grasp_mass_override_feedback(self) -> Dict[str, Any]:
        override_state = self._grasp_mass_override_state or {}
        override_name = str(override_state.get('object_name', '') or '').strip()
        override_mass = override_state.get('override_mass')
        return {
            'grasped_object_name': override_name,
            'grasped_object_mass': None if override_mass is None else float(override_mass),
        }

    def _prepare_grasp_target_mass_override(
        self,
        *,
        obj_name: str = '',
        target_mass: Optional[float] = None,
    ) -> Dict[str, Any]:
        feedback = self._get_active_grasp_mass_override_feedback()

        try:
            resolved_name, target_prim_path = self._resolve_grasp_target(obj_name)
            if not target_prim_path:
                return feedback

            mass_prim = self._resolve_mass_target_prim(target_prim_path)
            if mass_prim is None:
                return feedback

            active_override = self._grasp_mass_override_state or {}
            active_mass_prim_path = str(active_override.get('mass_prim_path', '') or '').strip()
            current_mass_prim_path = str(mass_prim.GetPath())
            if active_mass_prim_path == current_mass_prim_path:
                return self._get_active_grasp_mass_override_feedback()

            _, original_mass = self._read_authored_mass_state(mass_prim)
            if target_mass is None:
                if original_mass is not None and original_mass > 0.0:
                    applied_mass = min(original_mass, float(self.DEFAULT_MAX_GRASPED_OBJECT_MASS))
                else:
                    applied_mass = float(self.DEFAULT_MAX_GRASPED_OBJECT_MASS)
            else:
                applied_mass = max(float(target_mass), 1e-4)

            applied_mass = self._set_mass_on_prim(mass_prim, applied_mass)
            self._grasp_mass_override_state = {
                'object_name': resolved_name,
                'object_prim_path': target_prim_path,
                'mass_prim_path': current_mass_prim_path,
                'override_mass': applied_mass,
            }
            print(
                '[Grasp mass] target mass override applied '
                f'object={resolved_name} '
                f'target_prim={target_prim_path} '
                f'mass_prim={current_mass_prim_path} '
                f'original_mass={original_mass} '
                f'applied_mass={applied_mass:.6f}'
            )
            feedback = self._get_active_grasp_mass_override_feedback()
        except Exception as exc:
            print(f'WARNING: Failed to adjust grasped object mass: {exc}')

        return feedback

    def _navigation_debug_payload(self) -> Dict[str, Any]:
        if self.robot_controller is None or not hasattr(self.robot_controller, 'get_navigation_debug_state'):
            return {}
        try:
            return dict(self.robot_controller.get_navigation_debug_state() or {})
        except Exception:
            return {}

    def _register_runtime_observations(self, observations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        specs: list[Dict[str, Any]] = []
        supported = {
            'object_exists',
            'object_pose_available',
            'object_in_zone',
            'object_in_room',
            'object_near_object',
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
            'object_contact',
            'objects_in_contact',
            'object_contact_with_object',
            'object_contact_motion',
            'object_spatial_relation',
            'object_in_zone_region',
            'object_on_surface_region',
            'object_axis_relation',
            'object_stable_on_surface',
            'object_obstructs_region',
            'object_cluster_relation',
            'entity_state_compare',
            'entity_state_duration',
            'device_state_equals',
            'articulation_compare',
            'articulated_object_state_equals',
            'door_opened',
        }
        for raw_observation in observations or []:
            observation = dict(raw_observation or {})
            predicate = str(observation.get('predicate') or '').strip()
            if predicate not in supported:
                continue
            args = dict(observation.get('arguments') or observation.get('args') or {})
            state_requirement = dict(observation.get('state_requirement') or {})
            fallback_obj_name = str(
                args.get('object')
                or args.get('object_name')
                or args.get('object_a')
                or args.get('source')
                or args.get('name')
                or args.get('entity')
                or args.get('device')
                or args.get('appliance')
                or state_requirement.get('entity')
                or state_requirement.get('device')
                or state_requirement.get('object')
                or ''
            ).strip()
            observation_entities = [
                dict(item)
                for item in observation.get('observation_entities') or []
                if isinstance(item, dict)
            ]
            object_names: list[str] = []
            seen_object_names: set[str] = set()
            for item in observation_entities:
                name = str(item.get('name') or '').strip()
                key = name.casefold()
                if name and key not in seen_object_names:
                    seen_object_names.add(key)
                    object_names.append(name)
            if fallback_obj_name and fallback_obj_name.casefold() not in seen_object_names:
                object_names.insert(0, fallback_obj_name)
            obj_name = object_names[0] if object_names else ''
            if not object_names and not state_requirement.get('selector'):
                continue
            specs.append({
                'usage_id': str(observation.get('usage_id') or observation.get('id') or f'object_tilt_{len(specs) + 1}'),
                'usage_type': str(observation.get('usage_type') or ''),
                'predicate': predicate,
                'object': obj_name,
                'objects': object_names,
                'observation_entities': observation_entities,
                'arguments': args,
                'description': str(observation.get('description') or ''),
                'state_requirement': state_requirement,
            })

        self.runtime_observation_specs = specs
        state_requirement_results = self._apply_runtime_state_requirements()
        contact_registration = self.runtime_safety_monitor.register_observations(specs)
        registered_objects = sorted({
            str(object_name).strip()
            for item in self.runtime_observation_specs
            for object_name in (
                list(item.get('objects') or [])
                or [str(item.get('object') or '').strip()]
            )
            if str(object_name or '').strip()
        })
        unresolved_objects = sorted(
            object_name
            for object_name in registered_objects
            if not self._lookup_object_prim_path(object_name)
        )
        articulated_names = {
            str(name).casefold()
            for name in self._articulated_objects_payload()
        }
        articulation_predicates = {
            'articulation_compare',
            'articulated_object_state_equals',
            'door_opened',
        }
        capability_errors = []
        for item in self.runtime_observation_specs:
            if str(item.get('predicate') or '') not in articulation_predicates:
                continue
            object_name = str(item.get('object') or '').strip()
            if object_name and object_name.casefold() not in articulated_names:
                capability_errors.append({
                    'usage_id': str(item.get('usage_id') or ''),
                    'predicate': str(item.get('predicate') or ''),
                    'object': object_name,
                    'reason': 'articulation_state_unavailable',
                })
        return {
            'ok': not unresolved_objects and not capability_errors,
            'registered_count': len(self.runtime_observation_specs),
            'observation_ids': [item['usage_id'] for item in self.runtime_observation_specs],
            'objects': registered_objects,
            'unresolved_objects': unresolved_objects,
            'capability_errors': capability_errors,
            'contact_observation_count': int(contact_registration.get('registered_count', 0) or 0),
            'state_requirement_count': len(state_requirement_results),
            'state_requirements': state_requirement_results,
        }

    def _apply_runtime_state_requirements(self) -> list[Dict[str, Any]]:
        return self.world_state.apply_state_requirements(
            [
                dict(item.get('state_requirement') or {})
                for item in self.runtime_observation_specs
                if item.get('state_requirement')
            ],
            create_missing=False,
        )

    def _runtime_observation_payload(self) -> Dict[str, Any]:
        observations: Dict[str, Any] = {}
        for spec in self.runtime_observation_specs:
            object_names = list(spec.get('objects') or []) or [spec.get('object')]
            for raw_object_name in object_names:
                obj_name = str(raw_object_name or '').strip()
                if not obj_name or obj_name in observations:
                    continue
                prim_path = self._lookup_object_prim_path(obj_name)
                if not prim_path:
                    observations[obj_name] = {
                        'name': obj_name,
                        'available': False,
                        'reason': 'object_not_found',
                    }
                    continue
                raw_pose = self._get_raw_pose_for_prim_path(prim_path)
                if raw_pose is None:
                    observations[obj_name] = {
                        'name': obj_name,
                        'prim_path': prim_path,
                        'available': False,
                        'reason': 'pose_unavailable',
                    }
                    continue
                position, orientation = raw_pose
                normalized_orientation = self._normalize_quaternion(np.array(orientation, dtype=float))
                pose = self._pose_dict_from_raw(position, normalized_orientation)
                tilt_rad = self._tilt_from_quaternion(normalized_orientation)
                bounds_payload = {}
                prim = self.stage.GetPrimAtPath(prim_path) if self.stage is not None else None
                bbox = self._compute_world_bbox(prim)
                if bbox is not None:
                    bounds_payload = self._bbox_to_dict(bbox[0], bbox[1])
                observations[obj_name] = {
                    'name': obj_name,
                    'prim_path': prim_path,
                    'available': True,
                    'pose': pose,
                    'position': dict(pose.get('position') or {}),
                    'orientation': dict(pose.get('orientation') or {}),
                    'bounds': bounds_payload,
                    'tilt_rad': float(tilt_rad),
                    'tilt_deg': float(np.degrees(tilt_rad)),
                    'registered_predicates': sorted({
                        str(item.get('predicate') or '')
                        for item in self.runtime_observation_specs
                        if obj_name in (
                            list(item.get('objects') or [])
                            or [item.get('object')]
                        )
                    }),
                }
        return observations

    def _tilt_from_quaternion(self, orientation: np.ndarray) -> float:
        rotation = self._quaternion_to_rotation_matrix(np.array(orientation, dtype=float))
        object_up = rotation @ np.array([0.0, 0.0, 1.0], dtype=float)
        up_z = float(np.clip(object_up[2], -1.0, 1.0))
        return float(np.arccos(up_z))

    def _articulated_objects_payload(self) -> Dict[str, Any]:
        articulated: Dict[str, Any] = {}
        candidates: list[tuple[str, str]] = []
        seen_paths: set[str] = set()
        for room in self.room_index.values():
            for item in room.get('objects') or []:
                name = str(item.get('name') or '').strip()
                prim_path = str(item.get('prim_path') or '').strip()
                if name and prim_path and prim_path not in seen_paths:
                    seen_paths.add(prim_path)
                    candidates.append((name, prim_path))
        for name, prim_path in self.object_prim_paths.items():
            path = str(prim_path or '').strip()
            if path and path not in seen_paths:
                seen_paths.add(path)
                candidates.append((str(name), path))

        for name, prim_path in candidates:
            try:
                if 'openable' not in self.world_state.require_entity(name).capabilities:
                    continue
            except WorldStateError:
                continue
            joints = []
            for candidate in self._find_container_door_joint_candidates(prim_path):
                target_position = self._get_container_door_drive_target(
                    str(candidate.get('joint_prim_path') or '')
                )
                if target_position is None:
                    target_position = self._infer_container_door_closed_target(candidate, 0.0)
                angle_rad = self._joint_angle_value_to_radians(float(target_position))
                lower = candidate.get('lower_limit')
                upper = candidate.get('upper_limit')
                limits_rad = [
                    self._joint_angle_value_to_radians(float(value))
                    for value in (lower, upper)
                    if value is not None
                ]
                maximum = max((abs(value) for value in limits_rad), default=max(abs(angle_rad), 1.0))
                joints.append({
                    'name': str(candidate.get('joint_name') or ''),
                    'joint_name': str(candidate.get('joint_name') or ''),
                    'joint_link': str(candidate.get('door_link_path') or ''),
                    'joint_prim_path': str(candidate.get('joint_prim_path') or ''),
                    'angle_rad': float(angle_rad),
                    'open_fraction': min(1.0, abs(float(angle_rad)) / max(maximum, 1e-6)),
                    'lower_limit_rad': min(limits_rad) if limits_rad else None,
                    'upper_limit_rad': max(limits_rad) if limits_rad else None,
                    'source': 'usd_drive_target',
                })
            if joints:
                articulated[name] = {
                    'name': name,
                    'prim_path': prim_path,
                    'joints': joints,
                }
        return articulated

    def _runtime_state_payload(self) -> Dict[str, Any]:
        self._refresh_scene_index()
        sim_time_s = self._sync_world_state_time()
        world_state = self.world_state.snapshot()
        articulated_objects = self._articulated_objects_payload()
        current_room_name = self._detect_current_room_name()
        self.current_room_name = current_room_name
        robot_pose = self._get_end_effector_pose() or self.robot_pose
        self.robot_pose = robot_pose
        base_pose = None
        torso_height = None
        raw_base_pose = self._get_robot_base_pose_raw()
        if raw_base_pose is not None:
            base_pose = self._pose_dict_from_raw(raw_base_pose[0], raw_base_pose[1])
        if self.robot_controller is not None:
            try:
                torso_height = float(self.robot_controller.get_torso_height())
            except Exception:
                torso_height = None
        robot_collision_root_path = self._get_robot_collision_root_path()
        robot_total_collider_count = (
            len(
                self._collect_collision_prim_paths(
                    robot_collision_root_path,
                    include_disabled=True,
                )
            )
            if robot_collision_root_path
            else 0
        )
        robot_enabled_collider_count = (
            len(
                self._collect_collision_prim_paths(
                    robot_collision_root_path,
                    include_disabled=False,
                )
            )
            if robot_collision_root_path
            else 0
        )
        payload = {
            'sim_time_s': sim_time_s,
            'robot_name': self.robot_name,
            'requested_room_name': self.requested_room_name,
            'spawn_room_name': self.spawn_room_name,
            'current_room_name': current_room_name,
            'spawn_pose': dict(self.spawn_pose),
            'robot_pose': robot_pose,
            'base_pose': base_pose,
            'torso_height': torso_height,
            'joint_states': self._get_joint_states(),
            'object_poses': dict(self.object_poses),
            'object_bounds': dict(self.object_bounds),
            'articulated_objects': articulated_objects,
            'articulations': articulated_objects,
            'room_index': dict(self.room_index),
            'gripper_state': self.gripper_state,
            'object_in_gripper': self.object_in_gripper,
            'grasped_object_name': self.grasped_object_name,
            'grasped_object_mass': self.grasped_object_mass,
            'pending_grasp_pose_latch': bool(self._pending_grasp_pose_latch_state),
            'grasp_pose_latch_enabled': bool(self._grasp_pose_latch_state),
            'grasp_pose_latch_object_name': str((self._grasp_pose_latch_state or {}).get('object_name', '') or ''),
            'persistent_arm_collision_filtering': self._persistent_arm_collision_filter_state(),
            'aggressive_lifecycle_collision_filtering': self._aggressive_lifecycle_collision_filter_state(),
            'command_counter': self.command_counter,
            'last_command': self.last_command,
            'robot_loaded': self.robot_controller is not None,
            'robot_prim_path': self.robot_prim_path,
            'robot_articulation_path': self.robot_articulation_path,
            'robot_collision_root_path': robot_collision_root_path,
            'robot_total_collider_count': robot_total_collider_count,
            'robot_enabled_collider_count': robot_enabled_collider_count,
            'control_module_path': self.control_module_path,
            'controller_class_name': self.controller_class_name,
            'runtime_safety_monitor': {
                'registered_assertion_count': len(self.runtime_safety_monitor.assertions),
                'registered_contact_observation_count': len(self.runtime_safety_monitor.observations),
                'pending_event_count': len(self.runtime_safety_monitor.events),
                'contact_event_count': len(self.runtime_safety_monitor.contact_events),
                'callback_registered': bool(self.runtime_safety_monitor._callback_registered),
                'callback_error': self.runtime_safety_monitor._callback_error,
                'detection_method': 'physics_post_step_aabb_overlap',
                'actual_contact_reports_available': False,
                'contact_force_available': False,
            },
            'runtime_observation_monitor': {
                'registered_observation_count': len(self.runtime_observation_specs),
                'objects': sorted({
                    str(item.get('object') or '')
                    for item in self.runtime_observation_specs
                    if str(item.get('object') or '').strip()
                }),
            },
            'world_state': world_state,
            'entities': dict(world_state.get('entities') or {}),
            'entity_state_events': list(world_state.get('events') or []),
            'contact_events': self.runtime_safety_monitor.get_contact_events(),
        }
        object_observations = self._runtime_observation_payload()
        if object_observations:
            payload['object_observations'] = object_observations
        payload.update(self._navigation_debug_payload())
        return payload

    def _handle_json_command(self, payload):
        command = str(payload.get('command', '') or '')
        args = payload.get('args') or {}

        if command == 'reset':
            ok = self._reset_scene_runtime()
            if ok:
                return self._ok(command, {'reset': True}, message='Scene runtime reset successfully.')
            return self._error(command, 'reset_failed')

        if command == 'register_runtime_assertions':
            assertions = args.get('assertions') or []
            if not isinstance(assertions, list):
                return self._error(command, 'assertions_must_be_list')
            registration = self.runtime_safety_monitor.register_assertions(assertions)
            return self._ok(command, registration, message='runtime assertions registered.')

        if command == 'register_runtime_observations':
            observations = args.get('observations') or []
            if not isinstance(observations, list):
                return self._error(command, 'observations_must_be_list')
            registration = self._register_runtime_observations(observations)
            return self._ok(command, registration, message='runtime observations registered.')

        if command == 'get_runtime_unsafe_events':
            clear = self._coerce_bool(args.get('clear', True), default=True)
            events = self.runtime_safety_monitor.get_events(clear=clear)
            return self._ok(
                command,
                {
                    'events': events,
                    'event_count': len(events),
                    'registered_assertion_count': len(self.runtime_safety_monitor.assertions),
                },
            )

        if command == 'clear_runtime_unsafe_events':
            self.runtime_safety_monitor.clear_events()
            return self._ok(command, {'cleared': True})

        if command == 'capture_top_down_screenshot':
            path = str(args.get('path') or '').strip()
            if not path:
                return self._error(command, 'path_required')
            resolution = self._normalize_top_down_resolution(args.get('resolution'))
            try:
                payload = self._capture_top_down_screenshot(path, resolution)
            except Exception as exc:
                self._top_down_capture_error = f'{exc.__class__.__name__}: {exc}'
                return self._error(
                    command,
                    f'capture_failed: {exc.__class__.__name__}: {exc}',
                    payload={
                        'path': path,
                        'resolution': [int(resolution[0]), int(resolution[1])],
                        'camera_prim_path': self.TOP_DOWN_CAMERA_PRIM_PATH,
                    },
                )
            return self._ok(command, payload, message='top-down screenshot captured.')
        if command == 'get_entity_capabilities':
            entity = str(
                args.get('entity')
                or args.get('object')
                or args.get('device')
                or args.get('name')
                or ''
            ).strip()
            try:
                result = self.world_state.capabilities(entity)
            except WorldStateError as exc:
                return self._error(command, str(exc))
            return self._ok(command, result)

        if command == 'get_entity_state':
            self._sync_world_state_time()
            entity = str(
                args.get('entity')
                or args.get('object')
                or args.get('device')
                or args.get('name')
                or ''
            ).strip()
            try:
                result = self.world_state.get_entity_state(entity)
            except WorldStateError as exc:
                return self._error(command, str(exc))
            return self._ok(command, result)

        if command == 'get_world_state':
            self._sync_world_state_time()
            return self._ok(command, self.world_state.snapshot())

        if command == 'set_entity_relation':
            source = str(args.get('source') or '').strip()
            target = str(args.get('target') or '').strip()
            relation = str(args.get('relation') or args.get('type') or '').strip()
            try:
                result = self.world_state.set_relation(
                    source,
                    target,
                    relation,
                    metadata=dict(args.get('metadata') or {}),
                )
            except WorldStateError as exc:
                return self._error(command, str(exc))
            return self._ok(command, result)

        if command == 'remove_entity_relation':
            source = str(args.get('source') or '').strip()
            target = str(args.get('target') or '').strip()
            relation = str(args.get('relation') or args.get('type') or '').strip()
            removed = self.world_state.remove_relation(source, target, relation)
            return self._ok(command, {'removed': removed})

        if command == 'interact_entity':
            entity = str(
                args.get('entity')
                or args.get('object')
                or args.get('device')
                or args.get('name')
                or ''
            ).strip()
            action = str(args.get('action') or '').strip()
            parameters = dict(args.get('parameters') or {})
            for key, value in args.items():
                if key not in {'entity', 'object', 'device', 'name', 'action', 'parameters'}:
                    parameters.setdefault(key, value)
            try:
                result = self.world_state.interact(
                    entity,
                    action,
                    parameters,
                    sim_time_s=self._current_sim_time_s(),
                )
            except (WorldStateError, TypeError, ValueError) as exc:
                return self._error(command, str(exc))
            return self._ok(
                command,
                result,
                message=f'entity interaction completed: {entity}.{action}',
            )

        if command == 'advance_sim_time':
            try:
                delta_s = float(args.get('seconds', args.get('delta_s', 0.0)) or 0.0)
                self.world_state.advance(delta_s)
            except (WorldStateError, TypeError, ValueError) as exc:
                return self._error(command, str(exc))
            return self._ok(
                command,
                self.world_state.snapshot(),
                message='symbolic world state time advanced.',
            )

        if command == 'add_person':
            if not isinstance(args, dict):
                return self._error(command, 'args_must_be_object')
            try:
                person_payload = self._add_person(dict(args))
            except Exception as exc:
                return self._error(
                    command,
                    f'{exc.__class__.__name__}:{exc}',
                )
            return self._ok(
                command,
                person_payload,
                message='Static person added to the scene.',
            )

        if command == 'get_obj_list':
            self._refresh_scene_index()
            current_room_objects = self._get_current_room_object_names()
            return self._ok(
                command,
                {
                    'current_room_name': self.current_room_name,
                    'objects': current_room_objects,
                    'object_count': len(current_room_objects),
                },
            )

        if command == 'get_object_pose':
            self._refresh_scene_index()
            obj_name = str(args.get('obj_name', '') or args.get('name', '') or '')
            prim_path = self._lookup_object_prim_path(obj_name)
            if not prim_path:
                return self._error(
                    command,
                    f'object_not_found:{obj_name}',
                    {'available_objects': self._get_current_room_object_names()},
                )
            pose_prim_path = self._resolve_pose_prim_path(prim_path)
            pose = self._get_pose_for_prim_path(prim_path)
            if pose is None:
                return self._error(command, f'pose_unavailable:{obj_name}')
            preferred_grasp = self._get_cached_preferred_grasp_pose(
                obj_name,
                prim_path=prim_path,
            )
            if not preferred_grasp:
                preferred_grasp = self._resolve_preferred_grasp_pose(
                    obj_name,
                    prim_path=prim_path,
                    object_pose=pose,
                )
            pose_position = dict(pose.get('position') or {})
            object_position = np.array(
                [
                    float(pose_position.get('x', 0.0)),
                    float(pose_position.get('y', 0.0)),
                    float(pose_position.get('z', 0.0)),
                ],
                dtype=float,
            )
            raw_base_pose = self._get_robot_base_pose_raw()
            base_pose = (
                self._pose_dict_from_raw(raw_base_pose[0], raw_base_pose[1])
                if raw_base_pose is not None
                else None
            )
            # Replace the bbox/prim center with the closest point on the
            # object's world AABB to the robot's xy (z held at the prim's
            # grasp height). Falls back to object_position when the bbox is
            # unavailable for whatever reason.
            grasp_position_for_candidates = object_position
            grasp_position_source = 'object_position'
            try:
                target_prim = self.stage.GetPrimAtPath(prim_path) if self.stage is not None else None
                bbox_arrays = self._compute_world_bbox(target_prim) if target_prim is not None else None
            except Exception:
                bbox_arrays = None
            if bbox_arrays is not None and raw_base_pose is not None:
                bbox_min, bbox_max = bbox_arrays
                grasp_position_for_candidates = self._nearest_bbox_grasp_point_xy(
                    target_min=bbox_min,
                    target_max=bbox_max,
                    robot_xy=np.array(raw_base_pose[0][:2], dtype=float),
                    height_z=float(object_position[2]),
                )
                grasp_position_source = 'nearest_bbox_point_to_current_base'
            grasp_pose_candidates = self._build_manipulation_grasp_pose_candidates(
                grasp_position=grasp_position_for_candidates,
                base_pose=base_pose,
            )
            grasp_poses = [
                dict(candidate.get('pose') or {})
                for candidate in grasp_pose_candidates
                if isinstance(candidate, dict) and candidate.get('pose')
            ]
            top_down_grasp_pose = dict(
                grasp_poses[0]
                if grasp_poses
                else preferred_grasp.get('grasp_pose')
                or {}
            )
            grasp_diagnostics = dict(preferred_grasp.get('diagnostics') or {})
            grasp_diagnostics.update(
                {
                    'grasp_pose_candidate_count': len(grasp_pose_candidates),
                    'grasp_pose_candidate_names': [
                        str(candidate.get('name', '') or '')
                        for candidate in grasp_pose_candidates
                    ],
                    'grasp_pose_candidate_frame': 'robot_object_vertical_plane',
                    'grasp_pose_candidate_position_source': grasp_position_source,
                    'grasp_position': {
                        'x': float(grasp_position_for_candidates[0]),
                        'y': float(grasp_position_for_candidates[1]),
                        'z': float(grasp_position_for_candidates[2]),
                    },
                    'object_position': {
                        'x': float(object_position[0]),
                        'y': float(object_position[1]),
                        'z': float(object_position[2]),
                    },
                }
            )
            return self._ok(
                command,
                {
                    'obj_name': obj_name,
                    'pose': pose,
                    'prim_path': prim_path,
                    'pose_prim_path': pose_prim_path,
                    'top_down_grasp_pose': top_down_grasp_pose,
                    'grasp_poses': grasp_poses,
                    'grasp_pose_candidates': grasp_pose_candidates,
                    'grasp_pose_source': 'object_position_candidates'
                    if grasp_poses
                    else str(preferred_grasp.get('grasp_pose_source', '') or ''),
                    'grasp_pose_diagnostics': grasp_diagnostics,
                },
            )

        if command in {'get_grasp_position', 'get_grasp_pose'}:
            self._refresh_scene_index()
            obj_name = str(args.get('obj_name', '') or args.get('name', '') or '')
            if not obj_name:
                return self._error(command, 'obj_name_required')

            prim_path = self._lookup_object_prim_path(obj_name)
            if not prim_path:
                return self._error(
                    command,
                    f'object_not_found:{obj_name}',
                    {'available_objects': self._get_current_room_object_names()},
                )
            
            # 获取目标物体Prim的点云数据

            try:
                max_points = int(args.get('max_points', args.get('point_cloud_max_points', 4096)))
            except (TypeError, ValueError):
                return self._error(command, 'invalid_max_points')

            point_cloud_data = self._get_prim_point_cloud_data(prim_path)
            point_cloud = np.array(point_cloud_data.get('points'), dtype=float)
            point_cloud_payload = self._point_cloud_to_payload(point_cloud, max_points)
            object_pose = self._get_pose_for_prim_path(prim_path)
            if object_pose is None:
                return self._error(command, f'pose_unavailable:{obj_name}')
            preferred_grasp = self._resolve_preferred_grasp_pose(
                obj_name,
                prim_path=prim_path,
                object_pose=object_pose,
            )
            object_position_dict = dict(object_pose.get('position') or {})
            object_position = np.array(
                [
                    float(object_position_dict.get('x', 0.0)),
                    float(object_position_dict.get('y', 0.0)),
                    float(object_position_dict.get('z', 0.0)),
                ],
                dtype=float,
            )
            raw_base_pose = self._get_robot_base_pose_raw()
            base_pose = (
                self._pose_dict_from_raw(raw_base_pose[0], raw_base_pose[1])
                if raw_base_pose is not None
                else None
            )
            grasp_pose_candidates = self._build_manipulation_grasp_pose_candidates(
                grasp_position=object_position,
                base_pose=base_pose,
            )
            grasp_poses = [
                dict(candidate.get('pose') or {})
                for candidate in grasp_pose_candidates
                if isinstance(candidate, dict) and candidate.get('pose')
            ]
            grasp_pose = dict(grasp_poses[0] if grasp_poses else preferred_grasp.get('grasp_pose') or {})
            grasp_diagnostics = dict(preferred_grasp.get('diagnostics') or {})
            grasp_diagnostics.update(
                {
                    'grasp_pose_candidate_count': len(grasp_pose_candidates),
                    'grasp_pose_candidate_names': [
                        str(candidate.get('name', '') or '')
                        for candidate in grasp_pose_candidates
                    ],
                    'grasp_pose_candidate_frame': 'robot_object_vertical_plane',
                    'grasp_pose_candidate_position_source': 'object_position',
                }
            )
            return self._ok(
                command,
                {
                    'obj_name': obj_name,
                    'prim_path': prim_path,
                    'pose': object_pose,
                    'grasp_position': dict(grasp_pose.get('position') or {}),
                    'grasp_pose': grasp_pose,
                    'grasp_poses': grasp_poses,
                    'grasp_pose_candidates': grasp_pose_candidates,
                    'grasp_pose_source': 'object_position_candidates'
                    if grasp_poses
                    else str(preferred_grasp.get('grasp_pose_source', '') or ''),
                    'grasp_pose_diagnostics': grasp_diagnostics,
                    'point_cloud': point_cloud_payload,
                },
                message='preferred grasp pose resolved.',
            )

        if command == 'suggest_manipulation_base_pose':
            self._refresh_scene_index()
            obj_name = str(args.get('obj_name', '') or args.get('name', '') or '')
            if not obj_name:
                return self._error(command, 'obj_name_required')
            try:
                suggestion = self._suggest_manipulation_base_pose(obj_name)
            except ManipulationBasePosePlanningError as exc:
                return self._error(command, str(exc), exc.payload)
            except ValueError as exc:
                return self._error(
                    command,
                    str(exc),
                    {'available_objects': self._get_all_known_object_names()},
                )
            except Exception as exc:
                return self._error(command, str(exc))

            return self._ok(
                command,
                suggestion,
                message='suggested a grounded manipulation base pose.',
            )

        if command in {'move_end_effector_to_pose', 'move_target_with_ori'}:
            if self.robot_controller is None:
                return self._error(command, 'robot_not_loaded')

            target_pose = args.get('target_pose') or {}
            if not isinstance(target_pose, dict):
                return self._error(command, 'invalid_target_pose')

            target_position, target_orientation, normalized_pose = self._pose_command_to_raw(target_pose)
            target_object = str(
                args.get('target_object', args.get('obj_name', args.get('object_name', ''))) or ''
            ).strip()
            target_object_name = ''
            target_object_prim_path = ''
            if target_object:
                _, target_entry = self._resolve_target_object_entry(
                    target_position,
                    target_object=target_object,
                )
                target_object_prim_path = (
                    str((target_entry or {}).get('prim_path', '') or '').strip()
                    if isinstance(target_entry, dict)
                    else ''
                )
                if not isinstance(target_entry, dict) or not target_object_prim_path:
                    return self._error(
                        command,
                        f'target_object_not_found:{target_object}',
                        {'available_objects': self._get_all_known_object_names()},
                    )
                target_object_name = str(target_entry.get('name', '') or target_object).strip()
                self._set_aggressive_lifecycle_collision_exempt_roots(
                    self._aggressive_lifecycle_collision_exempt_roots_for_target(
                        target_object_prim_path
                    )
                )
                self._prepare_grasp_target_physics(target_object)
                # Snap the target xy to the nearest point on the object's
                # world AABB. When the agent passes the raw object pose as
                # target_pose (e.g. the LEGACY prompt's
                # `move_end_effector_to_pose(target_pose=apple_pose)` shape),
                # the EE would otherwise be commanded to the bbox/prim
                # centre. Rewriting to the closest bbox point keeps the
                # agent's z (so pre-grasp lift / lower offsets are
                # preserved) while pulling xy to the side of the bbox
                # facing the robot. The original target_pose is preserved
                # in `normalized_pose` for the response payload.
                try:
                    target_object_prim = (
                        self.stage.GetPrimAtPath(target_object_prim_path)
                        if self.stage is not None and target_object_prim_path
                        else None
                    )
                    target_bbox_arrays = (
                        self._compute_world_bbox(target_object_prim)
                        if target_object_prim is not None
                        else None
                    )
                except Exception:
                    target_bbox_arrays = None
                raw_base_pose = self._get_robot_base_pose_raw()
                if target_bbox_arrays is not None and raw_base_pose is not None:
                    bbox_min, bbox_max = target_bbox_arrays
                    snapped = self._nearest_bbox_grasp_point_xy(
                        target_min=bbox_min,
                        target_max=bbox_max,
                        robot_xy=np.array(raw_base_pose[0][:2], dtype=float),
                        height_z=float(target_position[2]),
                    )
                    pre_snap_xy = (float(target_position[0]), float(target_position[1]))
                    target_position = np.array(
                        [
                            float(snapped[0]),
                            float(snapped[1]),
                            float(target_position[2]),
                        ],
                        dtype=float,
                    )
                    print(
                        '[Grasp snap] target_object='
                        f'{target_object_name or target_object} '
                        f'pre_snap_xy=({pre_snap_xy[0]:.4f},{pre_snap_xy[1]:.4f}) '
                        f'snapped_xy=({float(target_position[0]):.4f},{float(target_position[1]):.4f}) '
                        f'z={float(target_position[2]):.4f}'
                    )
            elif self.object_in_gripper and self.grasped_object_name:
                self._set_aggressive_lifecycle_collision_exempt_roots(
                    self._aggressive_lifecycle_collision_exempt_roots_for_target()
                )
            else:
                self._set_aggressive_lifecycle_collision_exempt_roots([])
            use_pre_grasp = bool(self.ENABLE_PRE_GRASP)
            enable_obstacle_avoidance = self._coerce_bool(
                args.get(
                    'enable_obstacle_avoidance',
                    args.get(
                        'enable_arm_obstacle_avoidance',
                        args.get('use_obstacle_avoidance', self.enable_arm_obstacle_avoidance),
                    ),
                ),
                default=self.enable_arm_obstacle_avoidance,
            )
            collision_filter_default = bool(self.enable_arm_collision_filtering)
            if self.enable_aggressive_lifecycle_collision_filtering:
                collision_filter_default = False
            if self.object_in_gripper and not target_object:
                collision_filter_default = False
            enable_collision_filtering = self._coerce_bool(
                args.get(
                    'enable_collision_filtering',
                    args.get(
                        'enable_arm_collision_filtering',
                        args.get('disable_arm_collisions', collision_filter_default),
                    ),
                ),
                default=collision_filter_default,
            )
            persist_collision_filtering = self._coerce_bool(
                args.get(
                    'persist_collision_filtering',
                    args.get(
                        'persist_arm_collision_filtering',
                        self.persist_arm_collision_filtering,
                    ),
                ),
                default=self.persist_arm_collision_filtering,
            )
            try:
                pre_grasp_distance = float(
                    args.get('pre_grasp_distance', args.get('pre_pose_distance', self.DEFAULT_PRE_GRASP_DISTANCE))
                )
                pre_grasp_linear_step = float(
                    args.get('pre_grasp_linear_step', args.get('linear_step', self.DEFAULT_PRE_GRASP_LINEAR_STEP))
                )
                collision_filter_query_padding = float(
                    args.get(
                        'collision_filter_query_padding',
                        args.get(
                            'arm_collision_filter_query_padding',
                            self.DEFAULT_ARM_COLLISION_FILTER_QUERY_PADDING,
                        ),
                    )
                )
                collision_filter_max_pairs = int(
                    args.get(
                        'collision_filter_max_pairs',
                        args.get(
                            'arm_collision_filter_max_pairs',
                            self.DEFAULT_ARM_COLLISION_FILTER_MAX_PAIRS,
                        ),
                    )
                )
            except (TypeError, ValueError):
                return self._error(command, 'invalid_pre_grasp_parameters')

            auto_obstacle_keys: list[str] = []
            collision_filter_pairs: list[tuple[str, str]] = []
            target_collision_filter_pairs: list[tuple[str, str]] = []
            target_collision_filter_prim_path = ''
            grasp_latch_candidate: Dict[str, Any] = {}
            target_collision_filter_restored = False
            target_collision_filter_latched = False
            try:
                if enable_obstacle_avoidance:
                    auto_obstacle_keys = self._register_auto_arm_obstacles(
                        target_position,
                        target_object=target_object,
                    )
                if enable_collision_filtering:
                    collision_filter_pairs = self._apply_arm_collision_filtering(
                        target_position,
                        query_padding=collision_filter_query_padding,
                        max_pairs=collision_filter_max_pairs,
                        include_target=True,
                        target_object=target_object,
                    )
                    if persist_collision_filtering:
                        self._remember_persistent_arm_collision_filtering(collision_filter_pairs)
                    target_collision_filter_prim_path = self._get_collision_filter_target_prim_path(
                        target_position,
                        target_object=target_object,
                    )
                    target_collision_filter_pairs = self._filter_collision_pairs_for_root(
                        collision_filter_pairs,
                        target_collision_filter_prim_path,
                    )

                # table_prim_path = self._lookup_object_prim_path('lab_table')
                # if table_prim_path:
                #     obstacle_key = self.robot_controller.add_rmpflow_obstacle(table_prim_path, static=True)
                #     auto_obstacle_keys.append(obstacle_key)
                # else:
                #     print("Can not find table prim")
                if use_pre_grasp:
                    motion = self._wait_for_pre_grasp_motion(
                        target_position=target_position,
                        target_orientation=target_orientation,
                        timeout_sec=float(self.DEFAULT_MOVE_TIMEOUT_SEC),
                        position_tolerance=float(self.DEFAULT_MOVE_POSITION_TOLERANCE),
                        orientation_tolerance=float(self.DEFAULT_MOVE_ORIENTATION_TOLERANCE),
                        pre_grasp_distance=pre_grasp_distance,
                        linear_step=pre_grasp_linear_step,
                    )
                else:
                    motion = self._wait_for_motion(
                        target_position=target_position,
                        target_orientation=target_orientation,
                        timeout_sec=float(self.DEFAULT_MOVE_TIMEOUT_SEC),
                        position_tolerance=float(self.DEFAULT_MOVE_POSITION_TOLERANCE),
                        orientation_tolerance=float(self.DEFAULT_MOVE_ORIENTATION_TOLERANCE),
                    )
                if motion['reached']:
                    should_prepare_latch = bool(
                        self.ENABLE_GRASP_POSE_LATCH
                        and target_object
                        and target_object_prim_path
                    )
                    if should_prepare_latch:
                        grasp_latch_candidate = self._prepare_grasp_pose_latch_candidate(
                            object_name=target_object_name or target_object,
                            object_prim_path=target_object_prim_path,
                            grasp_position=target_position,
                            grasp_orientation=target_orientation,
                            target_collision_filter_pairs=target_collision_filter_pairs,
                            target_collision_filter_prim_path=target_collision_filter_prim_path,
                            persist_collision_filtering=persist_collision_filtering,
                        )
                        target_collision_filter_latched = bool(grasp_latch_candidate)
                    if (
                        enable_collision_filtering
                        and not target_collision_filter_latched
                        and not persist_collision_filtering
                    ):
                        self._remove_arm_collision_filtering(target_collision_filter_pairs)
                        target_collision_filter_restored = bool(target_collision_filter_pairs)
            finally:
                self._remove_auto_arm_obstacles(auto_obstacle_keys)
            robot_pose = self._get_end_effector_pose() or self.robot_pose
            self.robot_pose = robot_pose
            message = (
                'end effector reached target pose.'
                if motion['reached']
                else 'end-effector command issued; controller did not fully settle before timeout.'
            )
            return self._ok(
                'move_end_effector_to_pose',
                {
                    'target_pose': normalized_pose,
                    'robot_pose': robot_pose,
                    'reached': motion['reached'],
                    'target_distance': motion['distance'],
                    'orientation_distance': motion['orientation_distance'],
                    'elapsed_sec': motion['elapsed_sec'],
                    'pre_grasp_enabled': bool(use_pre_grasp),
                    'obstacle_avoidance_enabled': bool(enable_obstacle_avoidance),
                    'auto_obstacle_count': len(auto_obstacle_keys),
                    'collision_filtering_enabled': bool(enable_collision_filtering),
                    'collision_filtering_persistent': bool(
                        enable_collision_filtering
                        and (
                            persist_collision_filtering
                            or target_collision_filter_latched
                            or not motion['reached']
                            or len(collision_filter_pairs) > len(target_collision_filter_pairs)
                        )
                    ),
                    'persist_collision_filtering': bool(persist_collision_filtering),
                    'collision_filter_query_padding': float(collision_filter_query_padding),
                    'collision_filter_max_pairs': int(collision_filter_max_pairs),
                    'collision_filter_pair_count': len(collision_filter_pairs),
                    'target_object': target_object,
                    'persistent_collision_filter_pair_count': (
                        len(collision_filter_pairs)
                        if persist_collision_filtering or target_collision_filter_latched or not motion['reached']
                        else len(collision_filter_pairs) - len(target_collision_filter_pairs)
                    ),
                    'target_collision_filter_prim_path': target_collision_filter_prim_path,
                    'target_collision_filter_restored': bool(target_collision_filter_restored),
                    'target_collision_filter_latched': bool(target_collision_filter_latched),
                    'target_collision_filter_pair_count': len(target_collision_filter_pairs),
                    'pending_grasp_pose_latch': bool(grasp_latch_candidate),
                    'pre_grasp_reached': bool(motion.get('pre_grasp_reached', False)),
                    'pre_grasp_position': self._pose_dict_from_raw(
                        motion['pre_grasp_position'],
                        target_orientation,
                    )['position']
                    if motion.get('pre_grasp_position') is not None
                    else None,
                    'approach_direction': {
                        'x': float(motion['approach_direction'][0]),
                        'y': float(motion['approach_direction'][1]),
                        'z': float(motion['approach_direction'][2]),
                    }
                    if motion.get('approach_direction') is not None
                    else None,
                    'linear_waypoint_count': int(motion.get('linear_waypoint_count', 0) or 0),
                    'persistent_arm_collision_filtering': (
                        self._persistent_arm_collision_filter_state()
                    ),
                    'aggressive_lifecycle_collision_filtering': (
                        self._aggressive_lifecycle_collision_filter_state()
                    ),
                },
                message=message,
            )

        if command == 'lateral_shift':
            if self.robot_controller is None:
                return self._error(command, 'robot_not_loaded')

            target_pose = args.get('target_pose') or {}
            direction = args.get('direction') or {}
            distance = args.get('distance')
            target_position_arg = args.get('target_position') or {}

            start_ee_pose = self._get_end_effector_pose_raw()
            if start_ee_pose is None:
                return self._error(command, 'end_effector_pose_unavailable')

            start_ee_position, start_ee_orientation = start_ee_pose

            if isinstance(target_position_arg, dict) and target_position_arg:
                target_position = np.array(
                    [
                        float(target_position_arg.get('x', 0.0)),
                        float(target_position_arg.get('y', 0.0)),
                        float(target_position_arg.get('z', 0.0)),
                    ],
                    dtype=float,
                )
                lateral_direction = target_position - start_ee_position
                norm = float(np.linalg.norm(lateral_direction))
                if norm < 1e-6:
                    return self._error(command, 'target_position_too_close')
                lateral_direction = lateral_direction / norm
            elif isinstance(direction, dict) and direction:
                lateral_direction = np.array(
                    [
                        float(direction.get('x', 0.0)),
                        float(direction.get('y', 0.0)),
                        float(direction.get('z', 0.0)),
                    ],
                    dtype=float,
                )
                norm = float(np.linalg.norm(lateral_direction))
                if norm < 1e-9:
                    return self._error(command, 'invalid_direction')
                lateral_direction = lateral_direction / norm
            else:
                return self._error(command, 'direction_or_target_position_required')

            if distance is not None:
                try:
                    shift_distance = float(distance)
                except (TypeError, ValueError):
                    return self._error(command, 'invalid_distance')
                if shift_distance < 0.0:
                    return self._error(command, 'negative_distance_not_allowed')
            elif isinstance(target_position_arg, dict) and target_position_arg:
                shift_distance = float(np.linalg.norm(
                    np.array([
                        float(target_position_arg.get('x', 0.0)),
                        float(target_position_arg.get('y', 0.0)),
                        float(target_position_arg.get('z', 0.0)),
                    ])
                    - start_ee_position,
                ))
            else:
                return self._error(command, 'distance_required_with_direction')

            target_position = start_ee_position + lateral_direction * shift_distance

            try:
                linear_step = float(
                    args.get('linear_step', self.DEFAULT_PRE_GRASP_LINEAR_STEP)
                )
            except (TypeError, ValueError):
                return self._error(command, 'invalid_linear_step')

            try:
                timeout_sec = float(
                    args.get('timeout_sec', self.DEFAULT_MOVE_TIMEOUT_SEC)
                )
            except (TypeError, ValueError):
                return self._error(command, 'invalid_timeout_sec')

            try:
                position_tolerance = float(
                    args.get('position_tolerance', self.DEFAULT_MOVE_POSITION_TOLERANCE)
                )
            except (TypeError, ValueError):
                return self._error(command, 'invalid_position_tolerance')

            try:
                orientation_tolerance = float(
                    args.get('orientation_tolerance', self.DEFAULT_MOVE_ORIENTATION_TOLERANCE)
                )
            except (TypeError, ValueError):
                return self._error(command, 'invalid_orientation_tolerance')

            enable_obstacle_avoidance = self._coerce_bool(
                args.get(
                    'enable_obstacle_avoidance',
                    args.get('use_obstacle_avoidance', self.enable_arm_obstacle_avoidance),
                ),
                default=self.enable_arm_obstacle_avoidance,
            )
            collision_filter_default = bool(self.enable_arm_collision_filtering)
            if self.enable_aggressive_lifecycle_collision_filtering:
                collision_filter_default = False
            enable_collision_filtering = self._coerce_bool(
                args.get('enable_collision_filtering', collision_filter_default),
                default=collision_filter_default,
            )
            persist_collision_filtering = self._coerce_bool(
                args.get(
                    'persist_collision_filtering',
                    args.get(
                        'persist_arm_collision_filtering',
                        self.persist_arm_collision_filtering,
                    ),
                ),
                default=self.persist_arm_collision_filtering,
            )
            try:
                collision_filter_query_padding = float(
                    args.get(
                        'collision_filter_query_padding',
                        self.DEFAULT_ARM_COLLISION_FILTER_QUERY_PADDING,
                    ),
                )
                collision_filter_max_pairs = int(
                    args.get(
                        'collision_filter_max_pairs',
                        self.DEFAULT_ARM_COLLISION_FILTER_MAX_PAIRS,
                    ),
                )
            except (TypeError, ValueError):
                return self._error(command, 'invalid_collision_filter_parameters')

            auto_obstacle_keys: list[str] = []
            collision_filter_pairs: list[tuple[str, str]] = []
            try:
                if enable_obstacle_avoidance:
                    auto_obstacle_keys = self._register_auto_arm_obstacles(
                        target_position,
                    )
                if enable_collision_filtering:
                    collision_filter_pairs = self._apply_arm_collision_filtering(
                        target_position,
                        query_padding=collision_filter_query_padding,
                        max_pairs=collision_filter_max_pairs,
                    )
                    if persist_collision_filtering:
                        self._remember_persistent_arm_collision_filtering(collision_filter_pairs)
                waypoint_count = max(1, int(np.ceil(shift_distance / linear_step))) if shift_distance > 0 else 1
                motion = {
                    'reached': True,
                    'distance': 0.0,
                    'orientation_distance': 0.0,
                    'elapsed_sec': 0.0,
                    'pre_grasp_enabled': False,
                    'pre_grasp_reached': False,
                    'pre_grasp_position': None,
                    'approach_direction': None,
                    'linear_waypoint_count': waypoint_count,
                    'stage_results': [],
                }
                for index in range(1, waypoint_count + 1):
                    alpha = float(index / waypoint_count)
                    waypoint_position = start_ee_position + (target_position - start_ee_position) * alpha
                    stage_motion = self._wait_for_motion(
                        target_position=waypoint_position,
                        target_orientation=start_ee_orientation,
                        timeout_sec=min(timeout_sec, float(self.DEFAULT_MOVE_TIMEOUT_SEC)),
                        position_tolerance=position_tolerance,
                        orientation_tolerance=orientation_tolerance,
                    )
                    stage_result = {
                        'stage': 'lateral_shift',
                        'waypoint_index': index,
                        'waypoint_count': waypoint_count,
                        **stage_motion,
                    }
                    motion['stage_results'].append(stage_result)
                    motion['distance'] = stage_motion['distance']
                    motion['orientation_distance'] = stage_motion['orientation_distance']
                    motion['elapsed_sec'] += stage_motion['elapsed_sec']
                    if not stage_motion['reached']:
                        motion['reached'] = False
                        break
                else:
                    motion['reached'] = True
            finally:
                self._remove_auto_arm_obstacles(auto_obstacle_keys)
                if not persist_collision_filtering:
                    self._remove_arm_collision_filtering(collision_filter_pairs)

            robot_pose = self._get_end_effector_pose() or self.robot_pose
            self.robot_pose = robot_pose
            message = (
                'lateral shift completed.'
                if motion['reached']
                else 'lateral shift did not fully complete before reaching waypoint limits or timeout.'
            )
            return self._ok(
                command,
                {
                    'start_position': self._pose_dict_from_raw(
                        start_ee_position, start_ee_orientation,
                    )['position'],
                    'target_position': self._pose_dict_from_raw(
                        target_position, start_ee_orientation,
                    )['position'],
                    'direction': {
                        'x': float(lateral_direction[0]),
                        'y': float(lateral_direction[1]),
                        'z': float(lateral_direction[2]),
                    },
                    'distance': float(shift_distance),
                    'robot_pose': robot_pose,
                    'reached': motion['reached'],
                    'target_distance': motion['distance'],
                    'orientation_distance': motion['orientation_distance'],
                    'elapsed_sec': motion['elapsed_sec'],
                    'linear_waypoint_count': motion.get('linear_waypoint_count', 0),
                    'obstacle_avoidance_enabled': bool(enable_obstacle_avoidance),
                    'collision_filtering_enabled': bool(enable_collision_filtering),
                    'persist_collision_filtering': bool(persist_collision_filtering),
                    'collision_filtering_persistent': bool(
                        enable_collision_filtering and persist_collision_filtering
                    ),
                    'persistent_collision_filter_pair_count': (
                        len(collision_filter_pairs)
                        if enable_collision_filtering and persist_collision_filtering
                        else 0
                    ),
                    'persistent_arm_collision_filtering': (
                        self._persistent_arm_collision_filter_state()
                    ),
                },
                message=message,
            )

        if command == 'rotate_end_effector':
            if self.robot_controller is None:
                return self._error(command, 'robot_not_loaded')

            target_orientation_arg = args.get('target_orientation') or {}
            target_pose = args.get('target_pose') or {}
            roll = args.get('roll')
            pitch = args.get('pitch')
            yaw = args.get('yaw')

            start_ee_pose = self._get_end_effector_pose_raw()
            if start_ee_pose is None:
                return self._error(command, 'end_effector_pose_unavailable')

            start_ee_position, start_ee_orientation = start_ee_pose
            start_euler = quats_to_euler_angles(np.array(start_ee_orientation, dtype=float))
            start_roll = float(start_euler[0])
            start_pitch = float(start_euler[1])
            start_yaw = float(start_euler[2])

            if isinstance(target_orientation_arg, dict) and target_orientation_arg:
                target_roll = float(target_orientation_arg.get('roll', start_roll))
                target_pitch = float(target_orientation_arg.get('pitch', start_pitch))
                target_yaw = float(target_orientation_arg.get('yaw', start_yaw))
            elif isinstance(target_pose, dict) and target_pose:
                orientation_from_pose = dict(target_pose.get('orientation') or {})
                if not orientation_from_pose:
                    orientation_from_pose = dict(target_pose.get('ori') or {})
                if not orientation_from_pose:
                    return self._error(command, 'target_orientation_required')
                target_roll = float(orientation_from_pose.get('roll', start_roll))
                target_pitch = float(orientation_from_pose.get('pitch', start_pitch))
                target_yaw = float(orientation_from_pose.get('yaw', start_yaw))
            else:
                target_roll = float(roll) if roll is not None else start_roll
                target_pitch = float(pitch) if pitch is not None else start_pitch
                target_yaw = float(yaw) if yaw is not None else start_yaw

            try:
                target_orientation_quat = euler_angles_to_quats(
                    np.array([target_roll, target_pitch, target_yaw], dtype=float)
                ).flatten()
            except Exception as exc:
                return self._error(command, f'invalid_target_orientation:{exc}')

            try:
                timeout_sec = float(
                    args.get('timeout_sec', self.DEFAULT_MOVE_TIMEOUT_SEC)
                )
            except (TypeError, ValueError):
                return self._error(command, 'invalid_timeout_sec')

            try:
                position_tolerance = float(
                    args.get('position_tolerance', self.DEFAULT_MOVE_POSITION_TOLERANCE)
                )
            except (TypeError, ValueError):
                return self._error(command, 'invalid_position_tolerance')

            try:
                orientation_tolerance = float(
                    args.get('orientation_tolerance', self.DEFAULT_MOVE_ORIENTATION_TOLERANCE)
                )
            except (TypeError, ValueError):
                return self._error(command, 'invalid_orientation_tolerance')

            keep_position = self._coerce_bool(
                args.get('keep_position', True),
                default=True,
            )
            target_position = start_ee_position if keep_position else start_ee_position

            enable_obstacle_avoidance = self._coerce_bool(
                args.get(
                    'enable_obstacle_avoidance',
                    args.get('use_obstacle_avoidance', self.enable_arm_obstacle_avoidance),
                ),
                default=self.enable_arm_obstacle_avoidance,
            )
            collision_filter_default = bool(self.enable_arm_collision_filtering)
            if self.enable_aggressive_lifecycle_collision_filtering:
                collision_filter_default = False
            enable_collision_filtering = self._coerce_bool(
                args.get('enable_collision_filtering', collision_filter_default),
                default=collision_filter_default,
            )
            persist_collision_filtering = self._coerce_bool(
                args.get(
                    'persist_collision_filtering',
                    args.get(
                        'persist_arm_collision_filtering',
                        self.persist_arm_collision_filtering,
                    ),
                ),
                default=self.persist_arm_collision_filtering,
            )

            auto_obstacle_keys: list[str] = []
            collision_filter_pairs: list[tuple[str, str]] = []
            try:
                if enable_obstacle_avoidance:
                    auto_obstacle_keys = self._register_auto_arm_obstacles(target_position)
                if enable_collision_filtering:
                    collision_filter_pairs = self._apply_arm_collision_filtering(
                        target_position,
                    )
                    if persist_collision_filtering:
                        self._remember_persistent_arm_collision_filtering(collision_filter_pairs)
                motion = self._wait_for_motion(
                    target_position=target_position,
                    target_orientation=target_orientation_quat,
                    timeout_sec=min(timeout_sec, float(self.DEFAULT_MOVE_TIMEOUT_SEC)),
                    position_tolerance=position_tolerance,
                    orientation_tolerance=orientation_tolerance,
                )
            finally:
                self._remove_auto_arm_obstacles(auto_obstacle_keys)
                if not persist_collision_filtering:
                    self._remove_arm_collision_filtering(collision_filter_pairs)

            robot_pose = self._get_end_effector_pose() or self.robot_pose
            self.robot_pose = robot_pose
            message = (
                'end effector orientation adjusted successfully.'
                if motion['reached']
                else 'end-effector rotation did not fully settle before timeout.'
            )
            return self._ok(
                command,
                {
                    'start_orientation': {
                        'roll': start_roll,
                        'pitch': start_pitch,
                        'yaw': start_yaw,
                    },
                    'target_orientation': {
                        'roll': target_roll,
                        'pitch': target_pitch,
                        'yaw': target_yaw,
                    },
                    'robot_pose': robot_pose,
                    'reached': motion['reached'],
                    'target_distance': motion['distance'],
                    'orientation_distance': motion['orientation_distance'],
                    'elapsed_sec': motion['elapsed_sec'],
                    'keep_position': keep_position,
                    'obstacle_avoidance_enabled': bool(enable_obstacle_avoidance),
                    'collision_filtering_enabled': bool(enable_collision_filtering),
                    'persist_collision_filtering': bool(persist_collision_filtering),
                    'collision_filtering_persistent': bool(
                        enable_collision_filtering and persist_collision_filtering
                    ),
                    'persistent_collision_filter_pair_count': (
                        len(collision_filter_pairs)
                        if enable_collision_filtering and persist_collision_filtering
                        else 0
                    ),
                    'persistent_arm_collision_filtering': (
                        self._persistent_arm_collision_filter_state()
                    ),
                },
                message=message,
            )

        if command == 'move_base_to_pose':
            if self.robot_controller is None:
                return self._error(command, 'robot_not_loaded')

            target_pose = args.get('target_pose') or {}
            if not isinstance(target_pose, dict):
                return self._error(command, 'invalid_target_pose')

            position, _, normalized_pose = self._pose_command_to_raw(target_pose)
            target_yaw = float(normalized_pose['orientation']['yaw'])
            try:
                self.robot_controller.set_navigation_target(position, target_yaw)
            except Exception as exc:
                return self._error(command, f'navigation_init_failed:{exc}')

            self._active_navigation_target = {'position': position, 'yaw': target_yaw}
            motion = self._wait_for_navigation(
                timeout_sec=float(self.DEFAULT_NAVIGATION_TIMEOUT_SEC),
                position_tolerance=float(self.DEFAULT_NAVIGATION_POSITION_TOLERANCE),
                yaw_tolerance=float(self.DEFAULT_NAVIGATION_YAW_TOLERANCE),
            )
            navigation_debug = dict(motion.get('navigation_debug') or self._navigation_debug_payload() or {})
            response_payload = {
                'target_pose': normalized_pose,
                'reached': motion['reached'],
                'target_distance': motion['distance'],
                'yaw_error': motion['yaw_error'],
                'elapsed_sec': motion['elapsed_sec'],
                'failure_reason': str(motion.get('failure_reason', '') or ''),
                'failure_message': str(motion.get('failure_message', '') or ''),
                **navigation_debug,
            }
            failure_reason = str(motion.get('failure_reason', '') or '').strip()
            if failure_reason:
                return self._error(command, failure_reason, payload=response_payload)

            return self._ok(
                command,
                response_payload,
                message='base reached target pose.'
                if motion['reached']
                else 'base command issued; controller did not fully settle before timeout.',
            )

        if command in {'move_torso_to_height', 'set_torso_height'}:
            if self.robot_controller is None:
                return self._error(command, 'robot_not_loaded')

            target_height = args.get('height', args.get('target_height'))
            if target_height is None:
                return self._error(command, 'height_required')

            try:
                requested_height = float(target_height)
            except (TypeError, ValueError):
                return self._error(command, 'invalid_height')

            try:
                applied_height = float(self.robot_controller.set_torso_height(requested_height))
            except Exception as exc:
                return self._error(command, f'torso_command_failed:{exc}')

            result = self._wait_for_torso(
                target_height=applied_height,
                timeout_sec=float(self.DEFAULT_TORSO_TIMEOUT_SEC),
                tolerance=float(self.DEFAULT_TORSO_POSITION_TOLERANCE),
            )
            return self._ok(
                command,
                {
                    'requested_height': requested_height,
                    'target_height': applied_height,
                    'torso_height': result['torso_height'],
                    'settled': result['settled'],
                    'elapsed_sec': result['elapsed_sec'],
                },
                message='torso reached target height.'
                if result['settled']
                else 'torso command issued; controller did not fully settle before timeout.',
            )

        if command == 'open':
            # currently, support Revolute Joint only.
            container_name = str(
                args.get('container_name')
                or args.get('obj_name')
                or args.get('object_name')
                or args.get('target_object')
                or args.get('name')
                or ''
            ).strip()
            if container_name:
                previous_exempt_roots = set(self._aggressive_lifecycle_collision_exempt_root_paths)
                container_prim_path_for_exemption = self._lookup_object_prim_path(container_name)
                if container_prim_path_for_exemption:
                    self._set_aggressive_lifecycle_collision_exempt_roots(
                        list(previous_exempt_roots | {container_prim_path_for_exemption})
                    )
                try:
                    result = self._open_container_by_name(container_name, args)
                except ValueError as exc:
                    payload = {'available_objects': self._get_all_known_object_names()}
                    if str(exc).startswith('door_joint_not_found:'):
                        prim_path = self._lookup_object_prim_path(container_name)
                        payload.update(
                            {
                                'container_name': container_name,
                                'container_prim_path': prim_path,
                                'door_joint_candidates': self._find_container_door_joint_candidates(prim_path)
                                if prim_path
                                else [],
                            }
                        )
                    return self._error(command, str(exc), payload)
                except Exception as exc:
                    return self._error(command, f'container_open_failed:{exc}')
                finally:
                    self._set_aggressive_lifecycle_collision_exempt_roots(list(previous_exempt_roots))
                try:
                    self.world_state.interact(
                        container_name,
                        'open',
                        sim_time_s=self._current_sim_time_s(),
                    )
                except WorldStateError:
                    pass
                if isinstance(result, dict):
                    result = dict(result)
                    result['persistent_arm_collision_filtering'] = (
                        self._persistent_arm_collision_filter_state()
                    )
                    result['aggressive_lifecycle_collision_filtering'] = (
                        self._aggressive_lifecycle_collision_filter_state()
                    )
                return self._ok(
                    command,
                    result,
                    message=f'container opened: {container_name}.',
                )

            if self.robot_controller is None:
                return self._error(command, 'robot_not_loaded')
            self._release_grasp_pose_latch()
            result = self._wait_for_gripper(
                gripper_action='open',
                timeout_sec=float(self.DEFAULT_GRIPPER_TIMEOUT_SEC),
                tolerance=float(self.DEFAULT_GRIPPER_POSITION_TOLERANCE),
            )
            self._active_gripper_action = None
            self.gripper_state = 'open'
            self.object_in_gripper = False
            self.grasped_object_name = ''
            self.grasped_object_mass = None
            self._set_aggressive_lifecycle_collision_exempt_roots([])
            return self._ok(
                command,
                {
                    'gripper_state': self.gripper_state,
                    'object_in_gripper': self.object_in_gripper,
                    'grasped_object_name': self.grasped_object_name,
                    'grasped_object_mass': self.grasped_object_mass,
                    'settled': result['settled'],
                    'elapsed_sec': result['elapsed_sec'],
                    'joint_positions': result['joint_positions'],
                    'gripper_hold_latched': False,
                    'persistent_arm_collision_filtering': (
                        self._persistent_arm_collision_filter_state()
                    ),
                    'aggressive_lifecycle_collision_filtering': (
                        self._aggressive_lifecycle_collision_filter_state()
                    ),
                },
                message='gripper opened.'
                if result['settled']
                else 'open command issued; timeout while waiting for settle.',
            )

        if command == 'close':
            if self.robot_controller is None:
                return self._error(command, 'robot_not_loaded')
            mass_feedback = self._get_active_grasp_mass_override_feedback()
            latch_feedback: Dict[str, Any] = {}
            if bool(self.ENABLE_GRASP_POSE_LATCH):
                latch_feedback = self._enable_grasp_pose_latch(
                    str(mass_feedback.get('grasped_object_name', '') or '')
                )
            result = self._wait_for_gripper(
                gripper_action='close',
                timeout_sec=float(self.DEFAULT_GRIPPER_TIMEOUT_SEC),
                tolerance=float(self.DEFAULT_GRIPPER_POSITION_TOLERANCE),
            )
            self.gripper_state = 'closed'
            measured_grasp_feedback = self._infer_object_in_gripper(result['joint_positions'])
            if latch_feedback:
                self.object_in_gripper = True
                # Keep reissuing the close target while the robot moves so the
                # fingers visually remain closed while the latched object follows the TCP.
                self._active_gripper_action = 'close'
                self.grasped_object_name = str(latch_feedback.get('object_name', '') or '')
                grasped_object_mass = mass_feedback.get('grasped_object_mass')
                self.grasped_object_mass = None if grasped_object_mass is None else float(grasped_object_mass)
                mass_feedback = {
                    'grasped_object_name': self.grasped_object_name,
                    'grasped_object_mass': self.grasped_object_mass,
                }
                grasp_feedback = {
                    'object_in_gripper': True,
                    'max_closure_residual': measured_grasp_feedback['max_closure_residual'],
                    'forced_by_latch': True,
                }
            elif bool(measured_grasp_feedback['object_in_gripper']):
                self.object_in_gripper = True
                self._active_gripper_action = 'close'
                self.grasped_object_name = str(mass_feedback.get('grasped_object_name', '') or '')
                grasped_object_mass = mass_feedback.get('grasped_object_mass')
                self.grasped_object_mass = None if grasped_object_mass is None else float(grasped_object_mass)
                grasp_feedback = {
                    **measured_grasp_feedback,
                    'forced_by_latch': False,
                }
            else:
                self.object_in_gripper = False
                self._active_gripper_action = None
                self._release_grasp_pose_latch()
                self.grasped_object_name = ''
                self.grasped_object_mass = None
                mass_feedback = {
                    'grasped_object_name': '',
                    'grasped_object_mass': None,
                }
                grasp_feedback = {
                    **measured_grasp_feedback,
                    'forced_by_latch': False,
                }
            if self.object_in_gripper:
                aggressive_exempt_roots = self._aggressive_lifecycle_collision_exempt_roots_for_target()
                if not aggressive_exempt_roots:
                    aggressive_exempt_roots = list(self._aggressive_lifecycle_collision_exempt_root_paths)
                aggressive_collision_state = self._set_aggressive_lifecycle_collision_exempt_roots(
                    aggressive_exempt_roots
                )
            else:
                aggressive_collision_state = self._set_aggressive_lifecycle_collision_exempt_roots([])
            if self.object_in_gripper:
                message = 'gripper closed, successfully grasped an object.'
            elif result['settled']:
                message = 'gripper fully closed; no object detected in gripper.'
            else:
                message = 'close command issued; timeout while waiting for settle.'
            return self._ok(
                command,
                {
                    'gripper_state': self.gripper_state,
                    'object_in_gripper': self.object_in_gripper,
                    'grasped_object_name': mass_feedback['grasped_object_name'],
                    'grasped_object_mass': mass_feedback['grasped_object_mass'],
                    'max_closure_residual': grasp_feedback['max_closure_residual'],
                    'settled': result['settled'],
                    'elapsed_sec': result['elapsed_sec'],
                    'joint_positions': result['joint_positions'],
                    'gripper_hold_latched': bool(self.object_in_gripper),
                    'grasp_pose_latch_enabled': bool(self._grasp_pose_latch_state),
                    'grasp_success_forced_by_latch': bool(grasp_feedback.get('forced_by_latch', False)),
                    'persistent_arm_collision_filtering': (
                        self._persistent_arm_collision_filter_state()
                    ),
                    'aggressive_lifecycle_collision_filtering': aggressive_collision_state,
                },
                message=message,
            )

        if command == 'get_runtime_state':
            return self._ok(command, self._runtime_state_payload())

        return self._error(command or 'unknown', 'unsupported_command')

    def handle_command(self, cmd):
        if cmd.startswith('load_scene'):
            parts = cmd.split(',', 1)
            if len(parts) != 2:
                print('Invalid load_scene command format. Expected: load_scene,<usd_path>')
                self.output_queue.put('False')
                return

            _, usd_file_path = parts
            loaded = self._load_scene(usd_file_path.strip())
            self.output_queue.put('True' if loaded else 'False')
        elif cmd.startswith('load_robot'):
            parts = cmd.split(',', 1)
            robot_name = 'fetch'
            requested_room_name = ''
            if len(parts) == 2 and parts[1].strip():
                raw_payload = parts[1].strip()
                try:
                    parsed_payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    parsed_payload = None

                if isinstance(parsed_payload, dict):
                    robot_name = str(parsed_payload.get('robot_name', 'fetch') or 'fetch').strip() or 'fetch'
                    requested_room_name = str(parsed_payload.get('room_name', '') or '').strip()
                else:
                    legacy_parts = raw_payload.split()
                    if legacy_parts:
                        robot_name = str(legacy_parts[0] or 'fetch').strip() or 'fetch'
                    if len(legacy_parts) >= 2:
                        requested_room_name = ' '.join(legacy_parts[1:]).strip()

            initialized, error = self._initialize_robot_runtime(
                robot_name,
                requested_room_name or None,
            )
            if not initialized:
                self.output_queue.put(self._error('load_robot', error))
                return

            self.output_queue.put(
                self._ok(
                    'load_robot',
                    {
                        'robot_name': self.robot_name,
                        'requested_room_name': self.requested_room_name,
                        'spawn_room_name': self.spawn_room_name,
                        'current_room_name': self.current_room_name or self.spawn_room_name,
                        'spawn_pose': dict(self.spawn_pose),
                        'robot_prim_path': self.robot_prim_path,
                        'robot_articulation_path': self.robot_articulation_path,
                        'control_module_path': self.control_module_path,
                        'controller_class_name': self.controller_class_name,
                        'room_index': dict(self.room_index),
                    },
                )
            )
        elif cmd.startswith('command'):
            parts = cmd.split(',', 1)
            if len(parts) != 2:
                print('Invalid command format. Expected: command,<json_payload>')
                self.output_queue.put(self._error('command', 'invalid_command_format'))
                return

            try:
                payload = json.loads(parts[1].strip())
            except json.JSONDecodeError as exc:
                print(f'Invalid command payload: {exc}')
                self.output_queue.put(self._error('command', 'invalid_json'))
                return

            self.command_counter += 1
            self.last_command = payload
            response = self._handle_json_command(payload)
            self.output_queue.put(response)
        elif cmd.strip() == 'reset':
            reset_ok = self._reset_scene_runtime()
            self.output_queue.put('True' if reset_ok else 'False')
        else:
            print(f'Unknown command received: {cmd}')
            self.output_queue.put('False')

    def run(self):
        self._setup_pipe_server()

        try:
            while self.simulation_app.is_running():
                if self.my_world is not None:
                    self._step_robot(render=True)
                else:
                    self.simulation_app.update()

                try:
                    cmd = self.input_queue.get_nowait()
                except queue.Empty:
                    continue

                if cmd == 'quit':
                    print('Received quit command, exiting.')
                    break

                self.handle_command(cmd)
        finally:
            if self.pipe_server is not None:
                try:
                    self.pipe_server.stop()
                except Exception as exc:
                    print(f'Error stopping pipe server: {exc}')
            try:
                self.simulation_app.close()
            except Exception as exc:
                print(f'Error closing simulation app: {exc}')


def main():
    app_runner = IsaacSimAppRunner(sim_app=simulation_app)
    app_runner.run()


if __name__ == '__main__':
    main()

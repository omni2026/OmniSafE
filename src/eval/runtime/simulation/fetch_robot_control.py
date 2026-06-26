from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.api import objects as core_objects
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper

try:
    from runtime.simulation.fetch_nav_planner import (
        AxisAlignedBBox,
        NavigationPlanner2D,
        NavigationPlanningError,
        SceneNavRoom,
    )
except ModuleNotFoundError:
    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from runtime.simulation.fetch_nav_planner import (  # type: ignore[no-redef]
        AxisAlignedBBox,
        NavigationPlanner2D,
        NavigationPlanningError,
        SceneNavRoom,
    )


class _ManagedRmpObstacle:
    def __init__(
        self,
        *,
        key: str,
        source: Any,
        motion_policy_obstacles: Sequence[Any],
        static: bool,
        uses_proxy: bool,
        proxy_specs: Optional[Sequence['_ObstacleProxySpec']] = None,
        enabled: bool = True,
    ) -> None:
        self.key = key
        self.source = source
        self.motion_policy_obstacles = list(motion_policy_obstacles)
        self.motion_policy_obstacle = self.motion_policy_obstacles[0] if self.motion_policy_obstacles else None
        self.static = static
        self.uses_proxy = uses_proxy
        self.proxy_specs = list(proxy_specs or [])
        self.enabled = enabled


class _ObstacleProxySpec:
    def __init__(
        self,
        *,
        shape: str,
        sync_mode: str,
        source_prim_path: str,
        proxy_prim_path: str,
        local_center: Optional[Sequence[float]] = None,
        local_size: Optional[Sequence[float]] = None,
        radius: Optional[float] = None,
        height: Optional[float] = None,
        axis: str = 'Z',
    ) -> None:
        self.shape = str(shape)
        self.sync_mode = str(sync_mode)
        self.source_prim_path = str(source_prim_path)
        self.proxy_prim_path = str(proxy_prim_path)
        self.local_center = None if local_center is None else np.array(local_center, dtype=float)
        self.local_size = None if local_size is None else np.array(local_size, dtype=float)
        self.radius = None if radius is None else float(radius)
        self.height = None if height is None else float(height)
        self.axis = str(axis or 'Z').upper()


class FetchRMPFlowController(mg.MotionPolicyController):
    SUPPORTED_PROXY_STRATEGIES = frozenset({'auto', 'obb', 'aabb'})
    MAX_AUTO_PROXY_COUNT = 12
    MIN_PROXY_EDGE = 1e-3
    MIN_AUTO_COMPONENT_EDGE = 0.02
    # 近似一个物体collision的启发式策略配置参数，可选项包括auto，obb，aabb。
    DEFAULT_PROXY_STRATEGY = 'auto'
    def __init__(
        self,
        *,
        name: str,
        robot_articulation: Articulation,
        robot_description_path: str,
        rmpflow_config_path: str,
        urdf_path: str,
        end_effector_frame_name: str = 'gripper_link',
        physics_dt: float = 1.0 / 60.0,
        maximum_substep_size: float = 0.00334,
    ) -> None:
        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=robot_description_path,
            rmpflow_config_path=rmpflow_config_path,
            urdf_path=urdf_path,
            end_effector_frame_name=end_effector_frame_name,
            maximum_substep_size=maximum_substep_size,
        )
        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation,
            self.rmpflow,
            physics_dt,
        )
        super().__init__(name=name, articulation_motion_policy=self.articulation_rmp)
        self._managed_obstacles: dict[str, _ManagedRmpObstacle] = {}
        self._debug_visualize_rmpflow_proxies = False # 是否打印RMPFlow障碍物代理的调试信息，并在场景中可视化它们。
        self._obstacle_proxy_root = find_unique_string_name(
            initial_name='/World/fetch_rmpflow_obstacles',
            is_unique_fn=lambda path: not is_prim_path_valid(path),
        )
        self._proxy_prim_index = 0
        SingleXFormPrim(
            prim_path=self._obstacle_proxy_root,
            visible=self._debug_visualize_rmpflow_proxies,
        )

        default_position, default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=default_position,
            robot_orientation=default_orientation,
        )

    def reset(self) -> None:
        super().reset()
        self._sync_robot_base_pose()
        for managed_obstacle in self._managed_obstacles.values():
            if managed_obstacle.uses_proxy:
                self._sync_managed_obstacle(managed_obstacle)
            self._register_managed_obstacle(managed_obstacle)
            if not managed_obstacle.enabled:
                self._set_managed_obstacle_enabled(managed_obstacle, enabled=False)

    def forward(
        self,
        target_end_effector_position: np.ndarray,
        target_end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        self._sync_robot_base_pose()
        self.sync_obstacles()
        return super().forward(
            target_end_effector_position=target_end_effector_position,
            target_end_effector_orientation=target_end_effector_orientation,
        )

    def add_obstacle(
        self,
        obstacle: Any,
        static: bool = False,
        *,
        strategy: str = DEFAULT_PROXY_STRATEGY,
    ) -> str:
        managed_obstacle = self._build_managed_obstacle(
            obstacle=obstacle,
            static=static,
            strategy=strategy,
        )
        existing_obstacle = self._managed_obstacles.get(managed_obstacle.key)
        if existing_obstacle is not None:
            self.remove_obstacle(managed_obstacle.key)

        if managed_obstacle.uses_proxy:
            self._sync_managed_obstacle(managed_obstacle)
        self._register_managed_obstacle(managed_obstacle)
        self._managed_obstacles[managed_obstacle.key] = managed_obstacle
        return managed_obstacle.key

    def remove_obstacle(self, obstacle: Any) -> bool:
        managed_obstacle = self._resolve_managed_obstacle(obstacle)
        if managed_obstacle is None:
            return False

        success = True
        for motion_policy_obstacle in managed_obstacle.motion_policy_obstacles:
            success = bool(self.rmpflow.remove_obstacle(motion_policy_obstacle)) and success
        self._managed_obstacles.pop(managed_obstacle.key, None)
        if managed_obstacle.uses_proxy:
            for motion_policy_obstacle in managed_obstacle.motion_policy_obstacles:
                proxy_path = str(motion_policy_obstacle.prim_path)
                if is_prim_path_valid(proxy_path):
                    delete_prim(proxy_path)
        return success

    def enable_obstacle(self, obstacle: Any) -> bool:
        managed_obstacle = self._resolve_managed_obstacle(obstacle)
        if managed_obstacle is None:
            return False

        success = self._set_managed_obstacle_enabled(managed_obstacle, enabled=True)
        if success:
            managed_obstacle.enabled = True
        return success

    def disable_obstacle(self, obstacle: Any) -> bool:
        managed_obstacle = self._resolve_managed_obstacle(obstacle)
        if managed_obstacle is None:
            return False

        success = self._set_managed_obstacle_enabled(managed_obstacle, enabled=False)
        if success:
            managed_obstacle.enabled = False
        return success

    def clear_obstacles(self) -> None:
        for obstacle_key in list(self._managed_obstacles.keys()):
            self.remove_obstacle(obstacle_key)

    def sync_obstacles(self) -> None:
        for managed_obstacle in self._managed_obstacles.values():
            if managed_obstacle.uses_proxy and not managed_obstacle.static:
                self._sync_managed_obstacle(managed_obstacle)

    def get_obstacle_keys(self) -> list[str]:
        return list(self._managed_obstacles.keys())

    def _sync_robot_base_pose(self) -> None:
        current_position, current_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=current_position,
            robot_orientation=current_orientation,
        )

    def _build_managed_obstacle(
        self,
        obstacle: Any,
        static: bool,
        strategy: str,
    ) -> _ManagedRmpObstacle:
        obstacle_key = _obstacle_key(obstacle)
        if self._is_motion_policy_obstacle(obstacle):
            return _ManagedRmpObstacle(
                key=obstacle_key,
                source=obstacle,
                motion_policy_obstacles=[obstacle],
                static=bool(static),
                uses_proxy=False,
            )

        prim_path = _extract_prim_path(obstacle)
        if prim_path is None:
            raise TypeError(
                'Unsupported obstacle type. Expected an isaacsim.core.api.objects obstacle, '
                'an XFormPrim/GeometryPrim-like wrapper with prim_path, or a prim path string.'
            )

        strategy_name = str(strategy or self.DEFAULT_PROXY_STRATEGY).lower()
        if strategy_name not in self.SUPPORTED_PROXY_STRATEGIES:
            raise ValueError(
                f'Unsupported RMPFlow obstacle proxy strategy: {strategy!r}. '
                f'Supported values: {sorted(self.SUPPORTED_PROXY_STRATEGIES)}'
            )

        stage = _get_stage()
        if stage is None:
            raise RuntimeError('RMP Flow obstacle creation failed: stage unavailable.')
        obstacle_prim = _resolve_usd_prim(obstacle, stage)
        if obstacle_prim is None or not obstacle_prim.IsValid():
            raise ValueError(f'Invalid obstacle prim: {prim_path}')

        proxy_specs = self._build_proxy_specs_from_prim(obstacle_prim, strategy_name)
        if not proxy_specs:
            raise ValueError(f'Unable to build RMPFlow obstacle proxies for prim: {prim_path}')
        if self._debug_visualize_rmpflow_proxies:
            preview_paths = [str(spec.source_prim_path) for spec in proxy_specs[:6]]
            if len(proxy_specs) > 6:
                preview_paths.append('...')
            # print(
            #     f'[RMPFlow debug] {prim_path} strategy={strategy_name} '
            #     f'proxy_count={len(proxy_specs)} sources={preview_paths}'
            # )

        proxy_obstacles = [self._create_proxy_obstacle(spec) for spec in proxy_specs]
        return _ManagedRmpObstacle(
            key=obstacle_key,
            source=obstacle,
            motion_policy_obstacles=proxy_obstacles,
            static=bool(static),
            uses_proxy=True,
            proxy_specs=proxy_specs,
        )

    def _resolve_managed_obstacle(self, obstacle: Any) -> Optional[_ManagedRmpObstacle]:
        if isinstance(obstacle, _ManagedRmpObstacle):
            return self._managed_obstacles.get(obstacle.key)
        return self._managed_obstacles.get(_obstacle_key(obstacle))

    def _sync_managed_obstacle(self, managed_obstacle: _ManagedRmpObstacle) -> None:
        if not managed_obstacle.uses_proxy:
            return

        stage = _get_stage()
        if stage is None:
            raise RuntimeError('RMP Flow obstacle sync failed: stage unavailable.')

        from pxr import Usd, UsdGeom

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=_bbox_cache_purposes(),
        )
        for proxy_spec, motion_policy_obstacle in zip(
            managed_obstacle.proxy_specs,
            managed_obstacle.motion_policy_obstacles,
        ):
            self._sync_proxy_obstacle(
                motion_policy_obstacle=motion_policy_obstacle,
                proxy_spec=proxy_spec,
                stage=stage,
                bbox_cache=bbox_cache,
            )

    def _register_managed_obstacle(self, managed_obstacle: _ManagedRmpObstacle) -> None:
        add_success = True
        for motion_policy_obstacle in managed_obstacle.motion_policy_obstacles:
            add_success = bool(
                self.rmpflow.add_obstacle(
                    motion_policy_obstacle,
                    static=managed_obstacle.static,
                )
            ) and add_success
        if not add_success:
            raise ValueError(
                f'Failed to register RMPFlow obstacle(s) for key {managed_obstacle.key!r}. '
                'This usually means an unsupported obstacle type was passed to Lula.'
            )

    def _set_managed_obstacle_enabled(
        self,
        managed_obstacle: _ManagedRmpObstacle,
        *,
        enabled: bool,
    ) -> bool:
        set_enabled = self.rmpflow.enable_obstacle if enabled else self.rmpflow.disable_obstacle
        success = True
        for motion_policy_obstacle in managed_obstacle.motion_policy_obstacles:
            success = bool(set_enabled(motion_policy_obstacle)) and success
        return success

    def _build_proxy_specs_from_prim(
        self,
        obstacle_prim,
        strategy: str,
    ) -> list[_ObstacleProxySpec]:
        strategy_name = str(strategy or self.DEFAULT_PROXY_STRATEGY).lower()
        bbox_cache = _create_bbox_cache()
        if strategy_name == 'aabb':
            proxy_spec = self._build_aabb_proxy_spec(obstacle_prim)
            return [proxy_spec] if proxy_spec is not None else []
        if strategy_name == 'obb':
            proxy_spec = self._build_cuboid_proxy_spec(obstacle_prim, bbox_cache)
            return [proxy_spec] if proxy_spec is not None else []

        proxy_specs: list[_ObstacleProxySpec] = []
        for collision_mesh_prim in self._collect_auto_collision_mesh_prims(obstacle_prim, bbox_cache):
            proxy_spec = self._build_aabb_proxy_spec(collision_mesh_prim)
            if proxy_spec is not None:
                proxy_specs.append(proxy_spec)

        if proxy_specs:
            return proxy_specs

        for component_prim in self._collect_auto_proxy_prims(obstacle_prim, bbox_cache):
            proxy_spec = self._build_aabb_proxy_spec(component_prim)
            if proxy_spec is not None:
                proxy_specs.append(proxy_spec)

        if proxy_specs:
            return proxy_specs

        fallback_spec = self._build_aabb_proxy_spec(obstacle_prim)
        return [fallback_spec] if fallback_spec is not None else []

    def _collect_auto_collision_mesh_prims(self, obstacle_prim, bbox_cache) -> list[Any]:
        from pxr import Usd, UsdGeom

        collision_mesh_prims: list[Any] = []
        collision_gprim_prims: list[Any] = []
        seen_paths: set[str] = set()
        collision_roots: list[Any] = []
        for prim in Usd.PrimRange(obstacle_prim):
            if prim is None or not prim.IsValid():
                continue
            prim_name = str(prim.GetName() or '').lower()
            prim_path = str(prim.GetPath()).lower()
            if prim_name in {'collision', 'collisions'} or '/collision/' in prim_path or '/collisions/' in prim_path:
                collision_roots.append(prim)

        mesh_search_roots = collision_roots or [obstacle_prim]
        for mesh_root in mesh_search_roots:
            for prim in Usd.PrimRange(mesh_root):
                if prim is None or not prim.IsValid():
                    continue
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                if _compute_local_bbox_from_prim(prim, bbox_cache) is None:
                    continue
                prim_path = str(prim.GetPath())
                if prim_path in seen_paths:
                    continue
                seen_paths.add(prim_path)
                if prim.IsA(UsdGeom.Mesh):
                    collision_mesh_prims.append(prim)
                else:
                    collision_gprim_prims.append(prim)

        collected_prims = collision_mesh_prims or collision_gprim_prims
        if self._debug_visualize_rmpflow_proxies:
            print(
                f'[RMPFlow debug] auto collision collection {obstacle_prim.GetPath()} '
                f'roots={len(collision_roots)} meshes={len(collision_mesh_prims)} '
                f'other_gprims={len(collision_gprim_prims)}'
            )

        return collected_prims

    def _collect_auto_proxy_prims(self, obstacle_prim, bbox_cache) -> list[Any]:
        from pxr import Usd, UsdGeom, UsdPhysics

        collision_candidates: list[Any] = []
        visual_candidates: list[Any] = []
        for prim in Usd.PrimRange(obstacle_prim):
            if prim is None or not prim.IsValid():
                continue
            if not prim.IsA(UsdGeom.Gprim):
                continue
            local_bbox = _compute_local_bbox_from_prim(prim, bbox_cache)
            if local_bbox is None:
                continue
            local_size = local_bbox[1] - local_bbox[0]
            if float(np.max(local_size)) < self.MIN_AUTO_COMPONENT_EDGE:
                continue

            imageable = UsdGeom.Imageable(prim)
            is_visible = True
            if imageable:
                try:
                    is_visible = imageable.ComputeVisibility(Usd.TimeCode.Default()) != UsdGeom.Tokens.invisible
                except Exception:
                    is_visible = True

            prim_path_lower = str(prim.GetPath()).lower()
            prim_name_lower = str(prim.GetName() or '').lower()
            is_collision_candidate = (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or 'collision' in prim_path_lower
                or 'collision' in prim_name_lower
                or 'collider' in prim_path_lower
                or 'collider' in prim_name_lower
            )
            if is_collision_candidate:
                collision_candidates.append(prim)
            elif is_visible:
                visual_candidates.append(prim)

        candidates = collision_candidates or visual_candidates
        if not candidates:
            return []

        def _candidate_rank(prim) -> tuple[float, float]:
            local_bbox = _compute_local_bbox_from_prim(prim, bbox_cache)
            if local_bbox is None:
                return (0.0, 0.0)
            local_size = np.maximum(local_bbox[1] - local_bbox[0], np.array([self.MIN_PROXY_EDGE] * 3, dtype=float))
            return (float(np.prod(local_size)), float(np.max(local_size)))

        candidates.sort(key=_candidate_rank, reverse=True)
        return candidates[: self.MAX_AUTO_PROXY_COUNT]

    def _build_proxy_spec_from_component(
        self,
        component_prim,
        bbox_cache,
    ) -> Optional[_ObstacleProxySpec]:
        from pxr import UsdGeom

        if component_prim.IsA(UsdGeom.Sphere):
            return self._build_sphere_proxy_spec(component_prim, bbox_cache)
        if component_prim.IsA(UsdGeom.Capsule) or component_prim.IsA(UsdGeom.Cylinder):
            return self._build_capsule_proxy_spec(component_prim, bbox_cache)
        return self._build_cuboid_proxy_spec(component_prim, bbox_cache)

    def _build_aabb_proxy_spec(self, obstacle_prim) -> Optional[_ObstacleProxySpec]:
        return _ObstacleProxySpec(
            shape='cuboid',
            sync_mode='aabb',
            source_prim_path=str(obstacle_prim.GetPath()),
            proxy_prim_path=self._next_proxy_prim_path(),
        )

    def _build_cuboid_proxy_spec(
        self,
        obstacle_prim,
        bbox_cache,
    ) -> Optional[_ObstacleProxySpec]:
        local_bbox = _compute_local_bbox_from_prim(obstacle_prim, bbox_cache)
        if local_bbox is None:
            return None
        min_corner, max_corner = local_bbox
        return _ObstacleProxySpec(
            shape='cuboid',
            sync_mode='local_bound',
            source_prim_path=str(obstacle_prim.GetPath()),
            proxy_prim_path=self._next_proxy_prim_path(),
            local_center=(min_corner + max_corner) / 2.0,
            local_size=np.maximum(max_corner - min_corner, np.array([self.MIN_PROXY_EDGE] * 3, dtype=float)),
        )

    def _build_sphere_proxy_spec(
        self,
        obstacle_prim,
        bbox_cache,
    ) -> Optional[_ObstacleProxySpec]:
        local_bbox = _compute_local_bbox_from_prim(obstacle_prim, bbox_cache)
        if local_bbox is None:
            return None
        min_corner, max_corner = local_bbox
        sphere_radius = float(np.max(max_corner - min_corner) * 0.5)
        return _ObstacleProxySpec(
            shape='sphere',
            sync_mode='local_bound',
            source_prim_path=str(obstacle_prim.GetPath()),
            proxy_prim_path=self._next_proxy_prim_path(),
            local_center=(min_corner + max_corner) / 2.0,
            radius=max(sphere_radius, self.MIN_PROXY_EDGE),
        )

    def _build_capsule_proxy_spec(
        self,
        obstacle_prim,
        bbox_cache,
    ) -> Optional[_ObstacleProxySpec]:
        from pxr import UsdGeom

        local_bbox = _compute_local_bbox_from_prim(obstacle_prim, bbox_cache)
        if local_bbox is None:
            return None

        min_corner, max_corner = local_bbox
        local_center = (min_corner + max_corner) / 2.0
        local_size = np.maximum(max_corner - min_corner, np.array([self.MIN_PROXY_EDGE] * 3, dtype=float))
        axis = 'Z'
        radius = None
        height = None

        if obstacle_prim.IsA(UsdGeom.Capsule):
            capsule_geom = UsdGeom.Capsule(obstacle_prim)
            axis = str(capsule_geom.GetAxisAttr().Get() or 'Z').upper()
            radius = capsule_geom.GetRadiusAttr().Get()
            height = capsule_geom.GetHeightAttr().Get()
        elif obstacle_prim.IsA(UsdGeom.Cylinder):
            cylinder_geom = UsdGeom.Cylinder(obstacle_prim)
            axis = str(cylinder_geom.GetAxisAttr().Get() or 'Z').upper()
            radius = cylinder_geom.GetRadiusAttr().Get()
            height = cylinder_geom.GetHeightAttr().Get()

        axis_index = _axis_index(axis)
        radial_axes = [0, 1, 2]
        radial_axes.remove(axis_index)
        if radius is None:
            radius = float(max(local_size[radial_axes[0]], local_size[radial_axes[1]]) * 0.5)
        if height is None:
            height = float(local_size[axis_index])

        return _ObstacleProxySpec(
            shape='capsule',
            sync_mode='local_bound',
            source_prim_path=str(obstacle_prim.GetPath()),
            proxy_prim_path=self._next_proxy_prim_path(),
            local_center=local_center,
            radius=max(float(radius), self.MIN_PROXY_EDGE),
            height=max(float(height), self.MIN_PROXY_EDGE),
            axis=axis,
        )

    def _next_proxy_prim_path(self) -> str:
        while True:
            candidate_path = f'{self._obstacle_proxy_root}/obstacle_{self._proxy_prim_index:04d}'
            self._proxy_prim_index += 1
            if not is_prim_path_valid(candidate_path):
                return candidate_path

    def _debug_proxy_color(self, shape: str) -> np.ndarray:
        return {
            'cuboid': np.array([1.0, 0.35, 0.2], dtype=float),
            'sphere': np.array([0.2, 0.9, 0.35], dtype=float),
            'capsule': np.array([0.2, 0.55, 1.0], dtype=float),
        }.get(shape, np.array([1.0, 1.0, 0.0], dtype=float))

    def _create_proxy_obstacle(self, proxy_spec: _ObstacleProxySpec) -> Any:
        debug_visible = self._debug_visualize_rmpflow_proxies
        debug_color = self._debug_proxy_color(proxy_spec.shape)
        if proxy_spec.shape == 'cuboid':
            return core_objects.VisualCuboid(
                prim_path=proxy_spec.proxy_prim_path,
                name='fetch_rmpflow_obstacle_proxy',
                size=1.0,
                position=np.zeros(3, dtype=float),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                scale=np.ones(3, dtype=float),
                visible=debug_visible,
                color=debug_color,
            )
        if proxy_spec.shape == 'sphere':
            return core_objects.VisualSphere(
                prim_path=proxy_spec.proxy_prim_path,
                name='fetch_rmpflow_obstacle_proxy',
                radius=1.0,
                position=np.zeros(3, dtype=float),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                visible=debug_visible,
                color=debug_color,
            )
        if proxy_spec.shape == 'capsule':
            return core_objects.VisualCapsule(
                prim_path=proxy_spec.proxy_prim_path,
                name='fetch_rmpflow_obstacle_proxy',
                radius=1.0,
                height=1.0,
                position=np.zeros(3, dtype=float),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                visible=debug_visible,
                color=debug_color,
            )
        raise ValueError(f'Unsupported obstacle proxy shape: {proxy_spec.shape!r}')

    def _sync_proxy_obstacle(
        self,
        *,
        motion_policy_obstacle: Any,
        proxy_spec: _ObstacleProxySpec,
        stage,
        bbox_cache,
    ) -> None:
        source_prim = stage.GetPrimAtPath(proxy_spec.source_prim_path)
        if source_prim is None or not source_prim.IsValid():
            raise ValueError(f'Invalid obstacle prim: {proxy_spec.source_prim_path}')

        if proxy_spec.sync_mode == 'aabb':
            bbox = _compute_world_bbox_from_prim(source_prim, bbox_cache)
            if bbox is None:
                raise ValueError(
                    f'Unable to compute a valid world bounding box for obstacle: {source_prim.GetPath()}'
                )
            min_corner, max_corner = bbox
            obstacle_center = (min_corner + max_corner) / 2.0
            obstacle_size = np.maximum(max_corner - min_corner, np.array([self.MIN_PROXY_EDGE] * 3, dtype=float))
            motion_policy_obstacle.set_world_pose(
                position=np.array(obstacle_center, dtype=float),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
            motion_policy_obstacle.set_local_scale(np.array(obstacle_size, dtype=float))
            return

        world_matrix = _compute_local_to_world_transform(source_prim)
        world_center = _transform_point(world_matrix, proxy_spec.local_center)
        world_orientation = _extract_quaternion_from_matrix(world_matrix)
        world_scale = _extract_scale_from_matrix(world_matrix)

        if proxy_spec.shape == 'cuboid':
            side_lengths = np.maximum(
                proxy_spec.local_size * world_scale,
                np.array([self.MIN_PROXY_EDGE] * 3, dtype=float),
            )
            motion_policy_obstacle.set_world_pose(
                position=world_center,
                orientation=world_orientation,
            )
            motion_policy_obstacle.set_local_scale(np.array(side_lengths, dtype=float))
            return

        if proxy_spec.shape == 'sphere':
            radius = max(float(proxy_spec.radius or self.MIN_PROXY_EDGE) * float(np.max(world_scale)), self.MIN_PROXY_EDGE)
            motion_policy_obstacle.set_world_pose(
                position=world_center,
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
            motion_policy_obstacle.set_radius(radius)
            return

        if proxy_spec.shape == 'capsule':
            axis_index = _axis_index(proxy_spec.axis)
            radial_axes = [0, 1, 2]
            radial_axes.remove(axis_index)
            radius_scale = float(max(world_scale[radial_axes[0]], world_scale[radial_axes[1]]))
            height_scale = float(world_scale[axis_index])
            capsule_orientation = _quat_multiply(
                world_orientation,
                _axis_alignment_quat(proxy_spec.axis),
            )
            motion_policy_obstacle.set_world_pose(
                position=world_center,
                orientation=capsule_orientation,
            )
            motion_policy_obstacle.set_radius(
                max(float(proxy_spec.radius or self.MIN_PROXY_EDGE) * radius_scale, self.MIN_PROXY_EDGE)
            )
            motion_policy_obstacle.set_height(
                max(float(proxy_spec.height or self.MIN_PROXY_EDGE) * height_scale, self.MIN_PROXY_EDGE)
            )
            return

        raise ValueError(f'Unsupported obstacle proxy shape: {proxy_spec.shape!r}')

    @staticmethod
    def _is_motion_policy_obstacle(obstacle: Any) -> bool:
        module_name = getattr(getattr(obstacle, '__class__', None), '__module__', '')
        return module_name.startswith('isaacsim.core.api.objects')


class NavigationController(BaseController):
    def __init__(self) -> None:
        super().__init__(name='navigation_controller')
        self.target_position = np.array([0.0, 0.0], dtype=float)
        self.target_orientation: Optional[float] = 0.0
        self.target_set = False
        self.max_speed = 0.8
        self.max_yaw_rate = np.pi / 2
        self.wheel_radius = 0.06
        self.wheel_base = 0.32
        self.slowing_down_distance = 0.55
        self.yaw_threshold = np.deg2rad(5.0)
        self.position_threshold = 0.1
        self.final_waypoint_threshold = self.position_threshold
        self.min_speed = 0.10
        self.min_yaw_rate = np.deg2rad(20.0)
        self.heading_stop_threshold = np.deg2rad(35.0)
        self.waypoint_reached_threshold = 0.08
        self.close_quarters_distance = 0.75
        self.close_quarters_max_speed = 0.28
        self.close_quarters_heading_threshold = np.deg2rad(20.0)
        self.turn_in_place_distance = 0.30
        self.turn_in_place_heading_threshold = np.deg2rad(12.0)
        self.replan_deviation_threshold = 0.5
        self.stall_step_threshold = 120
        self.progress_epsilon = 0.02
        self.collision_stall_step_threshold = 150
        self.collision_stall_position_epsilon = 0.03
        self.collision_monitor_min_distance = 0.20

        self.path_waypoints: list[np.ndarray] = []
        self.active_waypoint_index = 0
        self.state = 'DONE'
        self.replan_requested = False
        self.failure_reason = ''
        self._last_position: Optional[np.ndarray] = None
        self._best_distance_to_waypoint = float('inf')
        self._stalled_steps = 0
        self._has_progress = False
        self._collision_stall_anchor: Optional[np.ndarray] = None
        self._collision_stall_steps = 0
        self._collision_stall_displacement = 0.0

    def set_target(
        self,
        target_position: Sequence[float],
        target_orientation: Optional[float] = None,
        *,
        waypoints: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        position = np.array(list(target_position), dtype=float)
        self.target_position = position[:2]
        self.target_orientation = None if target_orientation is None else float(target_orientation)
        raw_waypoints = waypoints or [self.target_position]
        self.path_waypoints = [
            np.array(list(waypoint)[:2], dtype=float)
            for waypoint in raw_waypoints
        ]
        if not self.path_waypoints:
            self.path_waypoints = [np.array(self.target_position, dtype=float)]
        else:
            # Always force the last tracked waypoint to coincide with the final target.
            # This keeps the path follower's completion logic aligned with the runtime
            # reachability check in isaac_sim_standalone._wait_for_navigation().
            self.path_waypoints[-1] = np.array(self.target_position, dtype=float)
        self.active_waypoint_index = 0
        self.target_set = True
        self.replan_requested = False
        self.failure_reason = ''
        self.state = 'FOLLOW_PATH'
        self._last_position = None
        self._best_distance_to_waypoint = float('inf')
        self._stalled_steps = 0
        self._has_progress = False
        self._collision_stall_anchor = None
        self._collision_stall_steps = 0
        self._collision_stall_displacement = 0.0

    def clear_target(self) -> None:
        self.target_set = False
        self.path_waypoints = []
        self.active_waypoint_index = 0
        self.replan_requested = False
        self.failure_reason = ''
        self._last_position = None
        self._best_distance_to_waypoint = float('inf')
        self._stalled_steps = 0
        self._has_progress = False
        self._collision_stall_anchor = None
        self._collision_stall_steps = 0
        self._collision_stall_displacement = 0.0
        self.state = 'DONE'

    def mark_navigation_failed(self, reason: str) -> None:
        self.failure_reason = str(reason or 'navigation_failed')
        self.target_set = False
        self.replan_requested = False
        self._collision_stall_anchor = None
        self._collision_stall_steps = 0
        self._collision_stall_displacement = 0.0
        self.state = 'FAILED'

    def consume_replan_request(self) -> bool:
        requested = bool(self.replan_requested)
        self.replan_requested = False
        return requested

    def update_tracking(self, current_xy: Sequence[float]) -> None:
        if not self.target_set or self.state != 'FOLLOW_PATH':
            self._last_position = np.array(current_xy[:2], dtype=float)
            return

        point = np.array(current_xy[:2], dtype=float)
        self._advance_waypoint_if_needed(point)
        if self.state != 'FOLLOW_PATH':
            self._last_position = point
            return

        current_waypoint = self.path_waypoints[self.active_waypoint_index]
        distance_to_waypoint = float(np.linalg.norm(current_waypoint - point))
        self._update_collision_monitor(point, distance_to_waypoint)
        if self.failure_reason:
            self._last_position = point
            return
        if distance_to_waypoint + self.progress_epsilon < self._best_distance_to_waypoint:
            self._best_distance_to_waypoint = distance_to_waypoint
            self._stalled_steps = 0
            self._has_progress = True
        elif self._has_progress:
            self._stalled_steps += 1

        deviation = _distance_to_polyline(point, self._tracking_polyline())
        if deviation > self.replan_deviation_threshold or self._stalled_steps >= self.stall_step_threshold:
            self.replan_requested = True

        self._last_position = point

    def _update_collision_monitor(
        self,
        point: np.ndarray,
        distance_to_waypoint: float,
    ) -> None:
        should_monitor = bool(
            distance_to_waypoint > max(self.collision_monitor_min_distance, self._active_waypoint_threshold())
        )
        if not should_monitor:
            self._collision_stall_anchor = np.array(point, dtype=float)
            self._collision_stall_steps = 0
            self._collision_stall_displacement = 0.0
            return

        if self._collision_stall_anchor is None:
            self._collision_stall_anchor = np.array(point, dtype=float)
            self._collision_stall_steps = 0
            self._collision_stall_displacement = 0.0
            return

        displacement = float(np.linalg.norm(point - self._collision_stall_anchor))
        self._collision_stall_displacement = displacement
        if displacement >= self.collision_stall_position_epsilon:
            self._collision_stall_anchor = np.array(point, dtype=float)
            self._collision_stall_steps = 0
            self._collision_stall_displacement = 0.0
            return

        self._collision_stall_steps += 1
        if self._collision_stall_steps >= self.collision_stall_step_threshold:
            self.mark_navigation_failed('navigation_collision_suspected:base_stuck')

    def forward(
        self,
        *,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        current_joint_positions: np.ndarray,
        step_size: float,
    ) -> ArticulationAction:
        if not self.target_set or self.state in {'DONE', 'FAILED'}:
            return self._zero_velocity_action(current_joint_positions)

        current_xy = np.array(current_position[:2], dtype=float)
        current_yaw = _quaternion_to_yaw(current_orientation)
        linear_speed = 0.0
        angular_speed = 0.0

        if self.state == 'FOLLOW_PATH':
            self._advance_waypoint_if_needed(current_xy)
            if self.state != 'FOLLOW_PATH':
                return self.forward(
                    current_position=current_position,
                    current_orientation=current_orientation,
                    current_joint_positions=current_joint_positions,
                    step_size=step_size,
                )

            target_xy = self.path_waypoints[self.active_waypoint_index]
            direction_vector = target_xy - current_xy
            distance_to_waypoint = float(np.linalg.norm(direction_vector))
            desired_yaw = current_yaw
            if distance_to_waypoint > 1e-6:
                desired_yaw = float(np.arctan2(direction_vector[1], direction_vector[0]))

            yaw_diff = _wrap_to_pi(desired_yaw - current_yaw)
            angular_speed = float(np.clip(yaw_diff / max(step_size, 1e-6), -self.max_yaw_rate, self.max_yaw_rate))
            if abs(angular_speed) < self.min_yaw_rate and abs(yaw_diff) > self.yaw_threshold:
                angular_speed = self.min_yaw_rate * np.sign(angular_speed or yaw_diff)

            if abs(yaw_diff) > self.heading_stop_threshold:
                linear_speed = 0.0
            else:
                if distance_to_waypoint > self.slowing_down_distance:
                    linear_speed = self.max_speed
                else:
                    ratio = distance_to_waypoint / max(self.slowing_down_distance, 1e-6)
                    linear_speed = max(self.min_speed, self.max_speed * ratio)
                linear_speed *= max(0.25, 1.0 - (abs(yaw_diff) / max(self.heading_stop_threshold, 1e-6)))
                if distance_to_waypoint <= self.close_quarters_distance:
                    linear_speed = min(linear_speed, self.close_quarters_max_speed)
                    if abs(yaw_diff) >= self.close_quarters_heading_threshold:
                        linear_speed *= 0.5
                if (
                    distance_to_waypoint <= self.turn_in_place_distance
                    and abs(yaw_diff) >= self.turn_in_place_heading_threshold
                ):
                    linear_speed = 0.0

        elif self.state == 'ROTATE_TO_TARGET_ORIENTATION':
            if self.target_orientation is None:
                self.state = 'DONE'
                return self._zero_velocity_action(current_joint_positions)
            yaw_diff = _wrap_to_pi(float(self.target_orientation) - current_yaw)
            angular_speed = float(np.clip(yaw_diff / max(step_size, 1e-6), -self.max_yaw_rate, self.max_yaw_rate))
            if abs(angular_speed) < self.min_yaw_rate and abs(yaw_diff) > self.yaw_threshold:
                angular_speed = self.min_yaw_rate * np.sign(angular_speed or yaw_diff)
            if abs(yaw_diff) <= self.yaw_threshold:
                self.state = 'DONE'
                return self._zero_velocity_action(current_joint_positions)

        wheel_left = linear_speed - (angular_speed * self.wheel_base / 2.0)
        wheel_right = linear_speed + (angular_speed * self.wheel_base / 2.0)
        omega_left = wheel_left / self.wheel_radius
        omega_right = wheel_right / self.wheel_radius
        target_joint_velocities = [omega_left, omega_right] + [None] * (len(current_joint_positions) - 2)
        return ArticulationAction(joint_velocities=target_joint_velocities)

    def _advance_waypoint_if_needed(self, current_xy: np.ndarray) -> None:
        while self.active_waypoint_index < len(self.path_waypoints):
            threshold = self._active_waypoint_threshold()
            distance = np.linalg.norm(self.path_waypoints[self.active_waypoint_index] - current_xy)
            if distance > threshold:
                break
            self.active_waypoint_index += 1
            self._best_distance_to_waypoint = float('inf')
            self._stalled_steps = 0
            self._has_progress = False

        if self.active_waypoint_index >= len(self.path_waypoints):
            self.state = 'ROTATE_TO_TARGET_ORIENTATION'

    def _active_waypoint_threshold(self) -> float:
        if not self.path_waypoints:
            return self.final_waypoint_threshold
        is_final_waypoint = self.active_waypoint_index >= len(self.path_waypoints) - 1
        if is_final_waypoint:
            return self.final_waypoint_threshold
        return self.waypoint_reached_threshold

    def _tracking_polyline(self) -> list[np.ndarray]:
        if not self.path_waypoints:
            return []
        start_index = max(self.active_waypoint_index - 1, 0)
        return [np.array(point, dtype=float) for point in self.path_waypoints[start_index:]]

    @staticmethod
    def _zero_velocity_action(current_joint_positions: np.ndarray) -> ArticulationAction:
        target_joint_velocities = [0.0, 0.0] + [None] * (len(current_joint_positions) - 2)
        return ArticulationAction(joint_velocities=target_joint_velocities)


class FetchRobotController:
    _MODULE_DIR = Path(__file__).resolve().parent
    _FETCH_ROOT = _MODULE_DIR / 'Fetch'
    DEFAULT_FETCH_USD_PATH = str(_FETCH_ROOT / 'assets' / 'fetch' / 'fetch_new.usd')
    DEFAULT_ROBOT_DESCRIPTION_PATH = str(_FETCH_ROOT / 'rmpflow' / 'robot_descriptor.yaml')
    DEFAULT_RMPFLOW_CONFIG_PATH = str(_FETCH_ROOT / 'rmpflow' / 'fetch_rmpflow_common.yaml')
    DEFAULT_URDF_PATH = str(_FETCH_ROOT / 'assets' / 'fetch' / 'fetch.urdf')
    DEFAULT_GROUND_CLEARANCE = 0.02
    # Fetch初始状态，但因为展开机械臂很容易导致碰撞
    DEFAULT_JOINT_POSITIONS = np.array(
        [
            0.0,
            0.0,
            0.20,
            0.0,
            1.1707963267948966,
            0.0,
            1.4707963267948965,
            -0.4,
            1.6707963267948966,
            0.0,
            1.5707963267948966,
            0.0,
            0.05,
            0.05,
        ],
        dtype=float,
    )
    # Keep the spawn/default pose compact for navigation, but seed the first
    # end-effector motion from an unfolded arm posture so RMPFlow/IK does not
    # start from the highly curled configuration.
    FIRST_END_EFFECTOR_READY_JOINT_POSITIONS = np.array(
        [
            0.0,  # l_wheel_joint
            0.0,  # r_wheel_joint
            0.2,  # torso_lift_joint (trunk)
            0.0,  # head_pan_joint
            0.0,  # shoulder_pan_joint
            0.0,  # head_tilt_joint
            -0.91,  # shoulder_lift_joint
            0.0,  # upperarm_roll_joint
            -0.39,  # elbow_flex_joint
            0.0,  # forearm_roll_joint
            1.85,  # wrist_flex_joint
            0.0,  # wrist_roll_joint
            0.05,  # l_gripper_finger_joint
            0.05,  # r_gripper_finger_joint
        ],
        dtype=float,
    )
    FIRST_END_EFFECTOR_READY_POSITION_TOLERANCE = 0.03
    FIRST_END_EFFECTOR_READY_PRESERVE_JOINT_INDICES = (0, 1)
    # DEFAULT_JOINT_POSITIONS = np.array(
    #     [
    #         0.0,
    #         0.44,
    #         0.20,
    #         0.0,
    #         0.0,
    #         0.0,
    #         -1.02,
    #         0.0,
    #         1.16,
    #         0.0,
    #         1.34,
    #         0.0,
    #         0.05,
    #         0.05,
    #     ],
    #     dtype=float,
    # )
    # 机械臂向前展开
    # DEFAULT_JOINT_POSITIONS = np.array(
    #     [
    #         0.0, # l_wheel_joint
    #         0.0, # r_wheel_joint
    #         0.2, # torso_lift_joint (trunk)
    #         0.0, # head_pan_joint
    #         0.0, # shoulder_pan_joint
    #         0.0, # head_tilt_joint
    #         -0.91, # shoulder_lift_joint
    #         0.0,  # upperarm_roll_joint
    #         -0.39, # elbow_flex_joint
    #         0.0,  # forearm_roll_joint
    #         1.85,  # wrist_flex_joint
    #         0.0,  # wrist_roll_joint
    #         0.05,  # l_gripper_finger_joint
    #         0.05,  # r_gripper_finger_joint
    #     ],
    #     dtype=float,
    # )
    TORSO_JOINT_NAME = 'torso_lift_joint'
    TORSO_MIN_HEIGHT = 0.0
    TORSO_MAX_HEIGHT = 0.38615
    NAV_PLANNER_MAX_REPLANS = 1
    NAV_PORTAL_KEYWORD = 'door'
    NAV_WALL_KEYWORD = 'wall'
    NAV_IGNORE_KEYWORDS = ('floor', 'ground', 'ceiling', 'light', 'camera', 'window')
    DEFAULT_NAVIGATION_ROBOT_RADIUS = 0.28
    DEFAULT_NAVIGATION_SAFETY_MARGIN = 0.02
    MAX_DYNAMIC_NAVIGATION_ROBOT_RADIUS = 0.50
    NAV_DYNAMIC_RADIUS_LINK_NAMES = (
        'shoulder_pan_link',
        'shoulder_lift_link',
        'upperarm_roll_link',
        'elbow_flex_link',
        'forearm_roll_link',
        'wrist_flex_link',
        'wrist_roll_link',
        'gripper_link',
    )

    def __init__(
        self,
        *,
        world: World,
        initial_position: Optional[Sequence[float]] = None,
        initial_orientation: Optional[Sequence[float]] = None,
        # 根据机器人当前的运动状态动态调整导航半径（更接近机器人实际占用空间），可能有助于提升导航成功率，但在某些环境中可能导致导航失败
        enable_dynamic_robot_radius: bool = False,
    ) -> None:
        self.world = world
        self.prim_path = '/World/fetch'
        self.robot_prim_path = f'{self.prim_path}/fetch'
        self.base_link_path = f'{self.robot_prim_path}/base_link'
        self.fetch_usd_path = str(self.DEFAULT_FETCH_USD_PATH)
        self.end_effector_prim_path = f'{self.robot_prim_path}/gripper_link'
        self.gripper_end_effector_prim_path = f'{self.robot_prim_path}/wrist_roll_link'
        self.gripper_joint_names = ['l_gripper_finger_joint', 'r_gripper_finger_joint']
        self.initial_position = _coerce_vector3(
            initial_position,
            default=[0.0, 0.0, 0.0],
        )
        self.initial_orientation = _coerce_quaternion(
            initial_orientation,
            default=[1.0, 0.0, 0.0, 0.0],
        )

        self.robot_description_path = str(self.DEFAULT_ROBOT_DESCRIPTION_PATH)
        self.rmpflow_config_path = str(self.DEFAULT_RMPFLOW_CONFIG_PATH)
        self.urdf_path = str(self.DEFAULT_URDF_PATH)
        self.end_effector_frame_name = 'gripper_link'
        self.physics_dt = float(1.0 / 60.0)
        self.maximum_substep_size = float(0.00334)
        self.gripper_step = float(0.01)
        self.gripper_open_positions = np.array([0.05, 0.05], dtype=float)
        self.gripper_closed_positions = np.array([0.0, 0.0], dtype=float)
        self.default_joint_positions = np.array(self.DEFAULT_JOINT_POSITIONS, dtype=float)
        self.first_end_effector_ready_joint_positions = np.array(
            self.FIRST_END_EFFECTOR_READY_JOINT_POSITIONS,
            dtype=float,
        )
        self._first_end_effector_move_prepared = False
        self.enable_dynamic_robot_radius = bool(enable_dynamic_robot_radius)

        self._navigation_goal_position: Optional[np.ndarray] = None
        self._navigation_goal_orientation: Optional[float] = None
        self._navigation_replan_attempts = 0
        self._nav_planner = NavigationPlanner2D(
            robot_radius=self.DEFAULT_NAVIGATION_ROBOT_RADIUS,
            safety_margin=self.DEFAULT_NAVIGATION_SAFETY_MARGIN,
        )
        self._default_navigation_robot_radius = float(self._nav_planner.robot_radius)
        self._navigation_robot_radius = float(self._default_navigation_robot_radius)
        self._navigation_arm_reach_radius: Optional[float] = None

        self._assert_required_paths()
        add_reference_to_stage(usd_path=self.fetch_usd_path, prim_path=self.prim_path)

        self.robot = Articulation(prim_path=self.robot_prim_path, name='fetch')
        self.robot_root_xform = SingleXFormPrim(prim_path=self.prim_path)
        self.robot_root_xform.set_world_pose(
            position=self.initial_position,
            orientation=self.initial_orientation,
        )
        self._update_root_default_state(self.initial_position, self.initial_orientation)
        self.align_root_to_ground(ground_z=float(self.initial_position[2]))
        self.world.scene.add(self.robot)

        self.gripper = ParallelGripper(
            end_effector_prim_path=self.gripper_end_effector_prim_path,
            joint_prim_names=self.gripper_joint_names,
            joint_opened_positions=self.gripper_open_positions,
            joint_closed_positions=self.gripper_closed_positions,
            action_deltas=np.array(self.gripper_open_positions, dtype=float),
        )
        self.manipulator = SingleManipulator(
            prim_path=self.robot_prim_path,
            name='fetch_robot',
            end_effector_prim_path=self.end_effector_prim_path,
            gripper=self.gripper,
        )
        self.manipulator.set_joints_default_state(positions=self.default_joint_positions)
        self.world.scene.add(self.manipulator)

        self.rmpflow_controller: Optional[FetchRMPFlowController] = None
        self.nav_controller: Optional[NavigationController] = None
        self.articulation_controller = None
        self.base_link_xform: Optional[SingleXFormPrim] = None

    def _assert_required_paths(self) -> None:
        for label, path in (
            ('fetch_usd_path', self.fetch_usd_path),
            ('robot_description_path', self.robot_description_path),
            ('rmpflow_config_path', self.rmpflow_config_path),
            ('urdf_path', self.urdf_path),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f'{label} does not exist: {path}')

    def _update_root_default_state(
        self,
        position: Sequence[float],
        orientation: Sequence[float],
    ) -> None:
        position_array = np.array(position, dtype=float)
        orientation_array = np.array(orientation, dtype=float)
        if hasattr(self.robot_root_xform, 'set_default_state'):
            try:
                self.robot_root_xform.set_default_state(
                    position=position_array,
                    orientation=orientation_array,
                )
            except Exception:
                pass
        if hasattr(self.robot, 'set_default_state'):
            try:
                self.robot.set_default_state(
                    position=position_array,
                    orientation=orientation_array,
                )
            except Exception:
                pass

    def align_root_to_ground(
        self,
        *,
        ground_z: float,
        clearance: Optional[float] = None,
    ) -> np.ndarray:
        stage = self._get_stage()
        if stage is None:
            return np.array(self.initial_position, dtype=float)

        from pxr import Usd, UsdGeom

        robot_prim = stage.GetPrimAtPath(self.prim_path)
        if robot_prim is None or not robot_prim.IsValid():
            return np.array(self.initial_position, dtype=float)

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
        )
        world_bbox = _compute_world_bbox_from_prim(robot_prim, bbox_cache)
        if world_bbox is None:
            return np.array(self.initial_position, dtype=float)

        current_position, current_orientation = self.robot_root_xform.get_world_pose()
        target_clearance = float(
            self.DEFAULT_GROUND_CLEARANCE if clearance is None else clearance
        )
        lift_delta = float(ground_z) + target_clearance - float(world_bbox[0][2])
        if abs(lift_delta) <= 1e-5:
            self._update_root_default_state(current_position, current_orientation)
            return np.array(current_position, dtype=float)

        adjusted_position = np.array(current_position, dtype=float)
        adjusted_position[2] += lift_delta
        self.robot_root_xform.set_world_pose(
            position=adjusted_position,
            orientation=np.array(current_orientation, dtype=float),
        )
        self.initial_position = adjusted_position
        self.initial_orientation = np.array(current_orientation, dtype=float)
        self._update_root_default_state(self.initial_position, self.initial_orientation)
        return np.array(adjusted_position, dtype=float)

    def initialize(self) -> None:
        self._first_end_effector_move_prepared = False
        self.robot.initialize()
        self.manipulator.initialize()
        self.base_link_xform = SingleXFormPrim(prim_path=self.base_link_path)
        self.rmpflow_controller = FetchRMPFlowController(
            name='fetch_rmpflow_controller',
            robot_articulation=self.manipulator,
            robot_description_path=self.robot_description_path,
            rmpflow_config_path=self.rmpflow_config_path,
            urdf_path=self.urdf_path,
            end_effector_frame_name=self.end_effector_frame_name,
            physics_dt=self.physics_dt,
            maximum_substep_size=self.maximum_substep_size,
        )
        self.nav_controller = NavigationController()
        self.articulation_controller = self.manipulator.get_articulation_controller()
        self.open_gripper(step_size=None)

    def reset(self) -> None:
        self.initialize()
        if self.rmpflow_controller is not None:
            self.rmpflow_controller.reset()
        if self.nav_controller is not None:
            self.nav_controller = NavigationController()
        self._navigation_goal_position = None
        self._navigation_goal_orientation = None
        self._navigation_replan_attempts = 0
        self._navigation_robot_radius = float(self._default_navigation_robot_radius)
        self._navigation_arm_reach_radius = None
        self.open_gripper(step_size=None)

    def set_navigation_target(
        self,
        target_position: Sequence[float],
        target_orientation: Optional[float] = None,
    ) -> None:
        # 该函数调用NavigationPlanner2D.plan(...)，规划得到Plan
        if self.nav_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')

        target_position_array = _coerce_vector3(target_position, default=[0.0, 0.0, 0.0])
        try:
            start_position, _ = self.get_base_link_pose()
        except Exception:
            start_position, _ = self.robot.get_world_pose()

        plan = self._plan_navigation(
            start_xy=np.array(start_position[:2], dtype=float),
            goal_xy=np.array(target_position_array[:2], dtype=float),
        )
        self._navigation_goal_position = np.array(target_position_array, dtype=float)
        self._navigation_goal_orientation = None if target_orientation is None else float(target_orientation)
        self._navigation_replan_attempts = 0
        self.nav_controller.set_target(
            target_position_array,
            self._navigation_goal_orientation,
            waypoints=plan.waypoints,  # 这里把NavigationPlanner2D.plan(...)得到的plan.waypoints传给了NavigationController.set_target(...)，供其跟踪
        )

    def navigate(self, step_size: Optional[float] = None) -> None:
        if self.nav_controller is None or self.articulation_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')

        current_position, current_orientation = self.get_base_link_pose()
        current_joint_positions = self.robot.get_joint_positions()
        self.nav_controller.update_tracking(current_position[:2])
        if self.nav_controller.consume_replan_request():
            self._attempt_replan(current_position[:2])

        action = self.nav_controller.forward(
            current_position=np.array(current_position, dtype=float),
            current_orientation=np.array(current_orientation, dtype=float),
            current_joint_positions=np.array(current_joint_positions, dtype=float),
            step_size=float(step_size or self.physics_dt),
        )
        self.articulation_controller.apply_action(action)

    def stop_base(self) -> None:
        if self.nav_controller is not None:
            self.nav_controller.clear_target()
        self._navigation_goal_position = None
        self._navigation_goal_orientation = None
        self._navigation_replan_attempts = 0
        if self.articulation_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')

        current_joint_positions = self.robot.get_joint_positions()
        target_joint_velocities = [0.0, 0.0] + [None] * (len(current_joint_positions) - 2)
        self.articulation_controller.apply_action(
            ArticulationAction(joint_velocities=target_joint_velocities)
        )

    def move_end_effector(
        self,
        target_position: Sequence[float],
        target_orientation: Sequence[float],
    ) -> None:
        if self.rmpflow_controller is None or self.articulation_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')
        if not self._first_end_effector_move_prepared:
            self.prepare_first_end_effector_move_step()
            return
        action = self.rmpflow_controller.forward(
            target_end_effector_position=np.array(target_position, dtype=float),
            target_end_effector_orientation=np.array(target_orientation, dtype=float),
        )
        self.articulation_controller.apply_action(action)

    def prepare_first_end_effector_move_step(
        self,
        *,
        tolerance: Optional[float] = None,
    ) -> dict[str, Any]:
        """Drive the arm toward the unfolded seed pose before IK/RMPFlow.

        This intentionally uses the articulation controller every simulation
        step instead of teleporting with set_joint_positions(...). Wheel joints
        are preserved at their current values so this arm-preparation step does
        not roll the mobile base back to wheel angle zero after navigation.
        """
        if self.articulation_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')

        tolerance_value = float(
            self.FIRST_END_EFFECTOR_READY_POSITION_TOLERANCE
            if tolerance is None
            else tolerance
        )
        current_positions = self.get_joint_positions()
        target_positions = self._first_end_effector_ready_target_positions(current_positions)
        error = np.array(target_positions - current_positions, dtype=float)
        max_abs_error = float(np.max(np.abs(error))) if error.size else 0.0
        error_norm = float(np.linalg.norm(error)) if error.size else 0.0

        if self._first_end_effector_move_prepared or max_abs_error <= tolerance_value:
            self._first_end_effector_move_prepared = True
            if self.rmpflow_controller is not None:
                self.rmpflow_controller.reset()
            return {
                'prepared': True,
                'max_abs_error': max_abs_error,
                'error_norm': error_norm,
                'target_positions': target_positions,
            }

        self.articulation_controller.apply_action(
            ArticulationAction(joint_positions=target_positions)
        )
        return {
            'prepared': False,
            'max_abs_error': max_abs_error,
            'error_norm': error_norm,
            'target_positions': target_positions,
        }

    def _first_end_effector_ready_target_positions(
        self,
        current_positions: np.ndarray,
    ) -> np.ndarray:
        target_positions = np.array(
            self.first_end_effector_ready_joint_positions,
            dtype=float,
        )
        if current_positions.size != target_positions.size:
            raise RuntimeError(
                'first_end_effector_ready_joint_count_mismatch:'
                f'current={current_positions.size},target={target_positions.size}'
            )

        for joint_index in self.FIRST_END_EFFECTOR_READY_PRESERVE_JOINT_INDICES:
            if 0 <= int(joint_index) < target_positions.size:
                target_positions[int(joint_index)] = float(current_positions[int(joint_index)])
        return target_positions

    def add_rmpflow_obstacle(
        self,
        obstacle: Any,
        *,
        static: bool = False,
        strategy: str = FetchRMPFlowController.DEFAULT_PROXY_STRATEGY,
    ) -> str:
        if self.rmpflow_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')
        return self.rmpflow_controller.add_obstacle(
            obstacle,
            static=static,
            strategy=strategy,
        )

    def remove_rmpflow_obstacle(self, obstacle: Any) -> bool:
        if self.rmpflow_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')
        return self.rmpflow_controller.remove_obstacle(obstacle)

    def enable_rmpflow_obstacle(self, obstacle: Any) -> bool:
        if self.rmpflow_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')
        return self.rmpflow_controller.enable_obstacle(obstacle)

    def disable_rmpflow_obstacle(self, obstacle: Any) -> bool:
        if self.rmpflow_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')
        return self.rmpflow_controller.disable_obstacle(obstacle)

    def clear_rmpflow_obstacles(self) -> None:
        if self.rmpflow_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')
        self.rmpflow_controller.clear_obstacles()

    def open_gripper(self, step_size: Optional[float] = None) -> None:
        if step_size is None:
            target_positions = self.gripper_open_positions
        else:
            current_positions = np.array(self.gripper.get_joint_positions(), dtype=float)
            target_positions = np.clip(
                current_positions + abs(float(step_size)),
                self.gripper_closed_positions,
                self.gripper_open_positions,
            )
        self.gripper.apply_action(ArticulationAction(joint_positions=target_positions))

    def close_gripper(self, step_size: Optional[float] = None) -> None:
        if step_size is None:
            target_positions = self.gripper_closed_positions
        else:
            current_positions = np.array(self.gripper.get_joint_positions(), dtype=float)
            target_positions = np.clip(
                current_positions - abs(float(step_size)),
                self.gripper_closed_positions,
                self.gripper_open_positions,
            )
        self.gripper.apply_action(ArticulationAction(joint_positions=target_positions))

    def _get_joint_index(self, joint_name: str) -> int:
        normalized_name = str(joint_name or '').strip()
        if not normalized_name:
            raise ValueError('joint_name_required')

        try:
            return self.get_joint_names().index(normalized_name)
        except ValueError as exc:
            raise ValueError(f'unknown_joint:{normalized_name}') from exc

    def get_joint_position(self, joint_name: str) -> float:
        joint_index = self._get_joint_index(joint_name)
        return float(self.get_joint_positions()[joint_index])

    def get_torso_height(self) -> float:
        return self.get_joint_position(self.TORSO_JOINT_NAME)

    def set_torso_height(self, target_height: float) -> float:
        if self.articulation_controller is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')

        clamped_height = float(
            np.clip(
                float(target_height),
                float(self.TORSO_MIN_HEIGHT),
                float(self.TORSO_MAX_HEIGHT),
            )
        )
        target_positions = self.get_joint_positions()
        target_positions[self._get_joint_index(self.TORSO_JOINT_NAME)] = clamped_height
        self.articulation_controller.apply_action(ArticulationAction(joint_positions=target_positions))
        return clamped_height

    def get_world_pose(self):
        return self.robot.get_world_pose()

    def get_joint_positions(self) -> np.ndarray:
        return np.array(self.robot.get_joint_positions(), dtype=float)

    def get_joint_names(self) -> list[str]:
        return list(getattr(self.robot, 'dof_names', []) or [])

    def get_gripper_joint_positions(self) -> np.ndarray:
        return np.array(self.gripper.get_joint_positions(), dtype=float)

    def get_end_effector_pose(self):
        return self.manipulator.end_effector.get_world_pose()

    def get_base_link_pose(self):
        if self.base_link_xform is None:
            raise RuntimeError('Robot is not initialized. Call initialize() first.')
        return self.base_link_xform.get_world_pose()

    def _get_link_world_position_by_name(self, link_name: str) -> Optional[np.ndarray]:
        normalized_name = str(link_name or '').strip()
        if not normalized_name:
            return None

        if normalized_name == 'base_link' and self.base_link_xform is not None:
            try:
                position, _ = self.base_link_xform.get_world_pose()
                return np.array(position, dtype=float)
            except Exception:
                pass

        stage = self._get_stage()
        if stage is None:
            return None

        prim = stage.GetPrimAtPath(f'{self.robot_prim_path}/{normalized_name}')
        if prim is None or not prim.IsValid():
            return None

        try:
            world_transform = _compute_local_to_world_transform(prim)
            position = _transform_point(world_transform, [0.0, 0.0, 0.0])
        except Exception:
            return None

        if position.size < 3 or not np.all(np.isfinite(position[:3])):
            return None
        return np.array(position[:3], dtype=float)

    def _estimate_current_arm_navigation_radius(self) -> float:
        base_position = self._get_link_world_position_by_name('base_link')
        if base_position is None:
            return float(self._default_navigation_robot_radius)

        max_radius = 0.0
        found_any_link = False
        for link_name in self.NAV_DYNAMIC_RADIUS_LINK_NAMES:
            link_position = self._get_link_world_position_by_name(link_name)
            if link_position is None:
                continue
            planar_radius = float(np.linalg.norm(link_position[:2] - base_position[:2]))
            if not np.isfinite(planar_radius):
                continue
            max_radius = max(max_radius, planar_radius)
            found_any_link = True

        if not found_any_link:
            return float(self._default_navigation_robot_radius)
        return float(max_radius)

    def _refresh_navigation_robot_radius(self) -> float:
        default_radius = float(self._default_navigation_robot_radius)
        if not self.enable_dynamic_robot_radius:
            self._navigation_arm_reach_radius = None
            effective_robot_radius = default_radius
        else:
            current_arm_radius = float(self._estimate_current_arm_navigation_radius())
            self._navigation_arm_reach_radius = current_arm_radius
            bounded_arm_radius = min(current_arm_radius, float(self.MAX_DYNAMIC_NAVIGATION_ROBOT_RADIUS))
            effective_robot_radius = max(default_radius, bounded_arm_radius)

        self._nav_planner.robot_radius = float(effective_robot_radius)
        self._nav_planner.inflation_radius = float(
            effective_robot_radius + float(self._nav_planner.safety_margin)
        )
        self._navigation_robot_radius = float(effective_robot_radius)
        return self._navigation_robot_radius

    def get_navigation_debug_state(self) -> dict[str, Any]:
        waypoint_count = 0
        active_waypoint_index = 0
        remaining_waypoints = 0
        navigation_state = ''
        navigation_failure_reason = ''
        navigation_replan_requested = False
        navigation_target_active = False
        navigation_collision_stall_steps = 0
        navigation_collision_stall_step_threshold = 0
        navigation_collision_stall_displacement = 0.0
        navigation_collision_stall_position_epsilon = 0.0

        if self.nav_controller is not None:
            waypoint_count = len(getattr(self.nav_controller, 'path_waypoints', []) or [])
            active_waypoint_index = int(getattr(self.nav_controller, 'active_waypoint_index', 0) or 0)
            remaining_waypoints = max(0, waypoint_count - active_waypoint_index)
            navigation_state = str(getattr(self.nav_controller, 'state', '') or '')
            navigation_failure_reason = str(getattr(self.nav_controller, 'failure_reason', '') or '')
            navigation_replan_requested = bool(getattr(self.nav_controller, 'replan_requested', False))
            navigation_target_active = bool(getattr(self.nav_controller, 'target_set', False))
            navigation_collision_stall_steps = int(getattr(self.nav_controller, '_collision_stall_steps', 0) or 0)
            navigation_collision_stall_step_threshold = int(
                getattr(self.nav_controller, 'collision_stall_step_threshold', 0) or 0
            )
            navigation_collision_stall_displacement = float(
                getattr(self.nav_controller, '_collision_stall_displacement', 0.0) or 0.0
            )
            navigation_collision_stall_position_epsilon = float(
                getattr(self.nav_controller, 'collision_stall_position_epsilon', 0.0) or 0.0
            )

        return {
            'navigation_robot_radius': float(self._navigation_robot_radius),
            'navigation_arm_reach_radius': None
            if self._navigation_arm_reach_radius is None
            else float(self._navigation_arm_reach_radius),
            'navigation_dynamic_radius_enabled': bool(self.enable_dynamic_robot_radius),
            'navigation_state': navigation_state,
            'navigation_failure_reason': navigation_failure_reason,
            'navigation_replan_requested': navigation_replan_requested,
            'navigation_target_active': navigation_target_active,
            'navigation_replan_attempts': int(self._navigation_replan_attempts),
            'navigation_waypoint_count': int(waypoint_count),
            'navigation_active_waypoint_index': int(active_waypoint_index),
            'navigation_remaining_waypoints': int(remaining_waypoints),
            'navigation_collision_stall_steps': int(navigation_collision_stall_steps),
            'navigation_collision_stall_step_threshold': int(navigation_collision_stall_step_threshold),
            'navigation_collision_stall_displacement': float(navigation_collision_stall_displacement),
            'navigation_collision_stall_position_epsilon': float(navigation_collision_stall_position_epsilon),
        }

    def _attempt_replan(self, current_xy: Sequence[float]) -> bool:
        if self.nav_controller is None:
            return False
        if self._navigation_goal_position is None:
            self.nav_controller.mark_navigation_failed('navigation_goal_missing')
            return False
        if self._navigation_replan_attempts >= self.NAV_PLANNER_MAX_REPLANS:
            self.nav_controller.mark_navigation_failed('navigation_replan_exhausted')
            return False

        self._navigation_replan_attempts += 1
        try:
            plan = self._plan_navigation(
                start_xy=np.array(current_xy[:2], dtype=float),
                goal_xy=np.array(self._navigation_goal_position[:2], dtype=float),
            )
        except Exception as exc:
            self.nav_controller.mark_navigation_failed(f'navigation_replan_failed:{exc}')
            return False

        self.nav_controller.set_target(
            self._navigation_goal_position,
            self._navigation_goal_orientation,
            waypoints=plan.waypoints,
        )
        return True

    def _plan_navigation(self, *, start_xy: np.ndarray, goal_xy: np.ndarray):
        self._refresh_navigation_robot_radius()
        rooms = self._build_navigation_scene() # 获取房间与通道信息
        try:
            return self._nav_planner.plan(rooms, start_xy=start_xy, goal_xy=goal_xy)
        except NavigationPlanningError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f'navigation_no_path:{exc}') from exc

    def _build_navigation_scene(self) -> list[SceneNavRoom]:
        stage = self._get_stage()
        if stage is None:
            raise RuntimeError('navigation_no_path:stage_unavailable')

        from pxr import Usd, UsdGeom

        rooms_root = stage.GetPrimAtPath('/World/rooms')
        if rooms_root is None or not rooms_root.IsValid():
            raise RuntimeError('navigation_no_path:rooms_root_not_found')

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
        )

        rooms: list[SceneNavRoom] = []
        for room_prim in rooms_root.GetChildren():
            if room_prim is None or not room_prim.IsValid():
                continue
            room_name = str(room_prim.GetName() or '').strip()
            if not room_name:
                continue

            room_bbox = self._compute_world_bbox(room_prim, bbox_cache)
            if room_bbox is None:
                continue

            boundary_segments = self._load_room_boundary_segments(room_prim)
            wall_bboxes: list[AxisAlignedBBox] = []
            door_bboxes: list[AxisAlignedBBox] = []
            object_bboxes: list[AxisAlignedBBox] = []

            for child in room_prim.GetChildren():
                if child is None or not child.IsValid():
                    continue
                child_name = str(child.GetName() or '').strip()
                if not child_name:
                    continue
                child_bbox = self._compute_world_bbox(child, bbox_cache)
                if child_bbox is None:
                    continue
                bbox = AxisAlignedBBox(
                    name=child_name,
                    min_x=float(child_bbox[0][0]),
                    min_y=float(child_bbox[0][1]),
                    max_x=float(child_bbox[1][0]),
                    max_y=float(child_bbox[1][1]),
                    min_z=float(child_bbox[0][2]),
                    max_z=float(child_bbox[1][2]),
                )
                lowered = child_name.lower()
                if self.NAV_PORTAL_KEYWORD in lowered:
                    door_bboxes.append(bbox)
                elif self.NAV_WALL_KEYWORD in lowered:
                    wall_bboxes.append(bbox)
                elif any(token in lowered for token in self.NAV_IGNORE_KEYWORDS):
                    continue
                else:
                    object_bboxes.append(bbox)

            rooms.append(
                SceneNavRoom(
                    room_name=room_name,
                    bounds=AxisAlignedBBox(
                        name=room_name,
                        min_x=float(room_bbox[0][0]),
                        min_y=float(room_bbox[0][1]),
                        max_x=float(room_bbox[1][0]),
                        max_y=float(room_bbox[1][1]),
                        min_z=float(room_bbox[0][2]),
                        max_z=float(room_bbox[1][2]),
                    ),
                    boundary_segments=boundary_segments,
                    wall_bboxes=wall_bboxes,
                    door_bboxes=door_bboxes,
                    object_bboxes=object_bboxes,
                )
            )

        if not rooms:
            raise RuntimeError('navigation_no_path:no_room_geometry')
        return rooms

    @staticmethod
    def _get_stage():
        return _get_stage()

    @staticmethod
    def _compute_world_bbox(prim, bbox_cache) -> Optional[tuple[np.ndarray, np.ndarray]]:
        return _compute_world_bbox_from_prim(prim, bbox_cache)

    @staticmethod
    def _load_room_boundary_segments(room_prim) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        serialized_segments = room_prim.GetCustomDataByKey('room_boundary_segments')
        if not serialized_segments:
            return []
        try:
            raw_segments = (
                json.loads(serialized_segments)
                if isinstance(serialized_segments, str)
                else serialized_segments
            )
        except Exception:
            return []

        normalized_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for start, end in raw_segments:
            try:
                p1 = (float(start[0]), float(start[1]))
                p2 = (float(end[0]), float(end[1]))
            except Exception:
                continue
            if p1 != p2:
                normalized_segments.append((p1, p2))
        return normalized_segments


def _coerce_array(value: Any, *, default: Iterable[float]) -> np.ndarray:
    if value is None:
        return np.array(list(default), dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, (list, tuple)):
        return np.array(list(value), dtype=float)
    if isinstance(value, dict):
        ordered = [value[key] for key in sorted(value.keys())]
        return np.array(ordered, dtype=float)
    return np.array(list(default), dtype=float)


def _coerce_vector3(value: Any, *, default: Iterable[float]) -> np.ndarray:
    if isinstance(value, dict):
        return np.array(
            [
                float(value.get('x', 0.0)),
                float(value.get('y', 0.0)),
                float(value.get('z', 0.0)),
            ],
            dtype=float,
        )
    array = _coerce_array(value, default=default)
    if array.size == 2:
        return np.array([float(array[0]), float(array[1]), 0.0], dtype=float)
    if array.size >= 3:
        return np.array([float(array[0]), float(array[1]), float(array[2])], dtype=float)
    default_values = list(default)
    return np.array(default_values[:3], dtype=float)


def _coerce_quaternion(value: Any, *, default: Iterable[float]) -> np.ndarray:
    if isinstance(value, dict):
        if {'w', 'x', 'y', 'z'}.issubset(set(value.keys())):
            return np.array(
                [
                    float(value.get('w', 1.0)),
                    float(value.get('x', 0.0)),
                    float(value.get('y', 0.0)),
                    float(value.get('z', 0.0)),
                ],
                dtype=float,
            )
        if {'roll', 'pitch', 'yaw'}.intersection(set(value.keys())):
            euler = np.array(
                [
                    float(value.get('roll', 0.0)),
                    float(value.get('pitch', 0.0)),
                    float(value.get('yaw', 0.0)),
                ],
                dtype=float,
            )
            return np.array(euler_angles_to_quats(euler), dtype=float)

    array = _coerce_array(value, default=default)
    if array.size >= 4:
        return np.array([float(array[0]), float(array[1]), float(array[2]), float(array[3])], dtype=float)
    if array.size == 3:
        return np.array(euler_angles_to_quats(array), dtype=float)
    default_values = np.array(list(default), dtype=float)
    return default_values


def _quaternion_to_yaw(quaternion: Sequence[float]) -> float:
    w, x, y, z = [float(value) for value in quaternion[:4]]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _wrap_to_pi(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _distance_to_polyline(point: np.ndarray, polyline: Sequence[np.ndarray]) -> float:
    if not polyline:
        return 0.0
    if len(polyline) == 1:
        return float(np.linalg.norm(np.array(polyline[0], dtype=float) - point))
    best_distance = float('inf')
    for start, end in zip(polyline[:-1], polyline[1:]):
        candidate = _closest_point_on_segment(
            point,
            np.array(start, dtype=float),
            np.array(end, dtype=float),
        )
        distance = float(np.linalg.norm(candidate - point))
        if distance < best_distance:
            best_distance = distance
    return best_distance


def _closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return np.array(start, dtype=float)
    t = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    return start + t * segment


def _get_stage():
    import omni.usd

    return omni.usd.get_context().get_stage()


def _extract_prim_path(obstacle: Any) -> Optional[str]:
    if isinstance(obstacle, str):
        return obstacle
    prim_path = getattr(obstacle, 'prim_path', None)
    if prim_path:
        return str(prim_path)
    prim_paths = getattr(obstacle, 'prim_paths', None)
    if prim_paths is not None and len(prim_paths) == 1:
        return str(prim_paths[0])
    prim = getattr(obstacle, 'prim', None)
    if prim is not None and hasattr(prim, 'GetPath'):
        return str(prim.GetPath())
    if hasattr(obstacle, 'GetPath'):
        return str(obstacle.GetPath())
    return None


def _resolve_usd_prim(obstacle: Any, stage) -> Any:
    if obstacle is None or stage is None:
        return None
    if hasattr(obstacle, 'IsValid') and hasattr(obstacle, 'GetPath'):
        return obstacle
    prim_path = _extract_prim_path(obstacle)
    if not prim_path:
        return None
    return stage.GetPrimAtPath(prim_path)


def _obstacle_key(obstacle: Any) -> str:
    prim_path = _extract_prim_path(obstacle)
    if prim_path:
        return prim_path
    return str(id(obstacle))


def _compute_world_bbox_from_prim(prim, bbox_cache) -> Optional[tuple[np.ndarray, np.ndarray]]:
    from pxr import Gf

    try:
        aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    except Exception:
        return _compute_world_bbox_from_mesh_points(prim)
    if aligned_box is None or aligned_box == Gf.Range3d():
        return _compute_world_bbox_from_mesh_points(prim)
    min_corner = np.array(aligned_box.GetMin(), dtype=float)
    max_corner = np.array(aligned_box.GetMax(), dtype=float)
    if not np.all(np.isfinite(min_corner)) or not np.all(np.isfinite(max_corner)):
        return _compute_world_bbox_from_mesh_points(prim)
    return min_corner, max_corner


def _compute_local_bbox_from_prim(prim, bbox_cache) -> Optional[tuple[np.ndarray, np.ndarray]]:
    from pxr import Gf

    try:
        aligned_box = bbox_cache.ComputeLocalBound(prim).ComputeAlignedBox()
    except Exception:
        return _compute_mesh_local_bbox_from_points(prim)
    if aligned_box is None or aligned_box == Gf.Range3d():
        return _compute_mesh_local_bbox_from_points(prim)
    min_corner = np.array(aligned_box.GetMin(), dtype=float)
    max_corner = np.array(aligned_box.GetMax(), dtype=float)
    if not np.all(np.isfinite(min_corner)) or not np.all(np.isfinite(max_corner)):
        return _compute_mesh_local_bbox_from_points(prim)
    return min_corner, max_corner


def _create_bbox_cache():
    from pxr import Usd, UsdGeom

    return UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=_bbox_cache_purposes(),
    )


def _bbox_cache_purposes():
    from pxr import UsdGeom

    return [
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
        UsdGeom.Tokens.guide,
    ]


def _compute_mesh_local_bbox_from_points(prim) -> Optional[tuple[np.ndarray, np.ndarray]]:
    from pxr import UsdGeom

    if prim is None or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        return None

    mesh = UsdGeom.Mesh(prim)
    extent_attr = mesh.GetExtentAttr()
    if extent_attr:
        extent = extent_attr.Get()
        bbox_from_extent = _bbox_from_extent_or_points(extent)
        if bbox_from_extent is not None:
            return bbox_from_extent

    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        return None
    return _bbox_from_extent_or_points(points_attr.Get())


def _compute_world_bbox_from_mesh_points(prim) -> Optional[tuple[np.ndarray, np.ndarray]]:
    local_bbox = _compute_mesh_local_bbox_from_points(prim)
    if local_bbox is None:
        return None

    min_corner, max_corner = local_bbox
    world_matrix = _compute_local_to_world_transform(prim)
    corners = np.array(
        [
            [min_corner[0], min_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
        ],
        dtype=float,
    )
    world_corners = np.array([_transform_point(world_matrix, corner) for corner in corners], dtype=float)
    if world_corners.ndim != 2 or world_corners.shape[1] != 3 or not np.all(np.isfinite(world_corners)):
        return None
    return np.min(world_corners, axis=0), np.max(world_corners, axis=0)


def _bbox_from_extent_or_points(values: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if values is None:
        return None
    points = np.array(values, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        return None
    if not np.all(np.isfinite(points)):
        return None
    return np.min(points, axis=0), np.max(points, axis=0)


def _compute_local_to_world_transform(prim):
    from pxr import Usd, UsdGeom

    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _extract_scale_from_matrix(transform_matrix) -> np.ndarray:
    from pxr import Gf

    scale = np.array(Gf.Transform(transform_matrix).GetScale(), dtype=float)
    if not np.all(np.isfinite(scale)):
        return np.ones(3, dtype=float)
    return np.maximum(np.abs(scale), np.array([1e-8, 1e-8, 1e-8], dtype=float))


def _extract_quaternion_from_matrix(transform_matrix) -> np.ndarray:
    from pxr import Gf

    quat = Gf.Transform(transform_matrix).GetRotation().GetQuat()
    return _normalize_quat(
        np.array(
            [
                float(quat.GetReal()),
                float(quat.GetImaginary()[0]),
                float(quat.GetImaginary()[1]),
                float(quat.GetImaginary()[2]),
            ],
            dtype=float,
        )
    )


def _transform_point(transform_matrix, point: Sequence[float]) -> np.ndarray:
    from pxr import Gf

    point_vec = Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
    return np.array(transform_matrix.Transform(point_vec), dtype=float)


def _normalize_quat(quat: Sequence[float]) -> np.ndarray:
    quat_array = np.array(quat, dtype=float)
    norm = float(np.linalg.norm(quat_array))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat_array / norm


def _quat_multiply(lhs: Sequence[float], rhs: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = _normalize_quat(lhs)
    w2, x2, y2, z2 = _normalize_quat(rhs)
    return _normalize_quat(
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


def _axis_alignment_quat(axis: str) -> np.ndarray:
    axis_name = str(axis or 'Z').upper()
    target = {
        'X': np.array([1.0, 0.0, 0.0], dtype=float),
        'Y': np.array([0.0, 1.0, 0.0], dtype=float),
        'Z': np.array([0.0, 0.0, 1.0], dtype=float),
    }.get(axis_name, np.array([0.0, 0.0, 1.0], dtype=float))
    return _quat_from_two_vectors(np.array([0.0, 0.0, 1.0], dtype=float), target)


def _quat_from_two_vectors(source: Sequence[float], target: Sequence[float]) -> np.ndarray:
    source_vec = np.array(source, dtype=float)
    target_vec = np.array(target, dtype=float)
    source_norm = float(np.linalg.norm(source_vec))
    target_norm = float(np.linalg.norm(target_vec))
    if source_norm <= 1e-12 or target_norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    source_vec = source_vec / source_norm
    target_vec = target_vec / target_norm
    dot = float(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))
    if dot >= 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if dot <= -1.0 + 1e-8:
        orthogonal = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(source_vec[0])) > 0.9:
            orthogonal = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = np.cross(source_vec, orthogonal)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=float)

    cross = np.cross(source_vec, target_vec)
    quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=float)
    return _normalize_quat(quat)


def _axis_index(axis: str) -> int:
    axis_name = str(axis or 'Z').upper()
    if axis_name == 'X':
        return 0
    if axis_name == 'Y':
        return 1
    return 2

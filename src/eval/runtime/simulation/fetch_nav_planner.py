from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point2D = Tuple[float, float]


class NavigationPlanningError(RuntimeError):
    """Raised when a 2D navigation path cannot be planned."""


@dataclass(frozen=True)
class AxisAlignedBBox:
    name: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_z: float = 0.0
    max_z: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.max_x - self.min_x)

    @property
    def depth(self) -> float:
        return max(0.0, self.max_y - self.min_y)

    @property
    def height(self) -> float:
        return max(0.0, self.max_z - self.min_z)

    @property
    def center(self) -> Point2D:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)


@dataclass
class SceneNavRoom:
    room_name: str
    bounds: AxisAlignedBBox
    boundary_segments: List[Tuple[Point2D, Point2D]] = field(default_factory=list)
    wall_bboxes: List[AxisAlignedBBox] = field(default_factory=list)
    door_bboxes: List[AxisAlignedBBox] = field(default_factory=list)
    object_bboxes: List[AxisAlignedBBox] = field(default_factory=list)


@dataclass
class NavigationPlan:
    waypoints: List[Point2D]
    room_sequence: List[str]
    portal_sequence: List[str]


@dataclass(frozen=True)
class _RouteCandidate:
    cost: float
    room_sequence: List[str]
    portal_sequence: List[str]


@dataclass
class _Portal:
    portal_id: str
    polygon: object
    room_names: set[str] = field(default_factory=set)
    anchors: Dict[str, Point2D] = field(default_factory=dict)


@dataclass
class _RoomGeometry:
    room_name: str
    polygon: object
    free_space: object
    blocked_space: object
    portals: List[str]


class _RouteCandidatePlanningError(RuntimeError):
    """Raised when one candidate route fails and planning should try another."""


def ensure_shapely():
    try:
        from shapely.geometry import LineString, Point, Polygon, box
        from shapely.ops import nearest_points, polygonize, unary_union
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'shapely is required for Fetch navigation planning. Install it in the Isaac Sim Python environment.'
        ) from exc

    return {
        'LineString': LineString,
        'Point': Point,
        'Polygon': Polygon,
        'box': box,
        'nearest_points': nearest_points,
        'polygonize': polygonize,
        'unary_union': unary_union,
    }


class NavigationPlanner2D:
    def __init__(
        self,
        *,
        robot_radius: float = 0.34,
        safety_margin: float = 0.08,
        grid_resolution: float = 0.10,
        snap_distance: float = 0.5,
        waypoint_spacing: float = 0.20,
        portal_merge_distance: float = 0.20,
        max_room_graph_candidates: int = 8,
        portal_waypoint_offset: float = 0.35,
    ) -> None:
        self.robot_radius = float(robot_radius)
        self.safety_margin = float(safety_margin)
        self.inflation_radius = self.robot_radius + self.safety_margin
        self.grid_resolution = float(grid_resolution)
        self.snap_distance = float(snap_distance)
        self.waypoint_spacing = float(waypoint_spacing)
        self.portal_merge_distance = float(portal_merge_distance)
        self.max_room_graph_candidates = max(1, int(max_room_graph_candidates))
        self.portal_waypoint_offset = max(float(portal_waypoint_offset), self.grid_resolution)

    def plan(
        self,
        rooms: Sequence[SceneNavRoom],
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
    ) -> NavigationPlan:
        scene_rooms = list(rooms or [])
        if not scene_rooms:
            raise NavigationPlanningError('navigation_no_path:no_rooms_available')

        geometry = ensure_shapely()
        nav_map, portals = self._build_nav_map(scene_rooms, geometry)
        start_room, start_point = self._locate_room(nav_map, tuple(map(float, start_xy)), geometry)
        goal_room, goal_point = self._locate_room(nav_map, tuple(map(float, goal_xy)), geometry)

        route_candidates = self._plan_room_graph_candidates(
            nav_map=nav_map,
            portals=portals,
            start_room=start_room,
            goal_room=goal_room,
        )
        deferred_error: Optional[NavigationPlanningError] = None

        for candidate in route_candidates:
            try:
                return self._plan_route_candidate(
                    nav_map=nav_map,
                    portals=portals,
                    geometry=geometry,
                    candidate=candidate,
                    start_point=start_point,
                    goal_point=goal_point,
                )
            except _RouteCandidatePlanningError as exc:
                deferred_error = NavigationPlanningError(str(exc))

        if deferred_error is not None:
            raise deferred_error
        raise NavigationPlanningError('navigation_no_path:empty_plan')

    def _build_nav_map(self, rooms: Sequence[SceneNavRoom], geometry) -> tuple[Dict[str, _RoomGeometry], Dict[str, _Portal]]:
        room_polygons: Dict[str, object] = {}
        raw_portals: List[_Portal] = []

        for room in rooms:
            polygon = self._room_polygon(room, geometry)
            room_polygons[room.room_name] = polygon
            for door_bbox in room.door_bboxes:
                raw_portals.append(
                    _Portal(
                        portal_id='',
                        polygon=self._bbox_polygon(door_bbox, geometry),
                        room_names={room.room_name},
                    )
                )

        portals = self._merge_portals(raw_portals, geometry)
        room_geometries: Dict[str, _RoomGeometry] = {}

        for room in rooms:
            room_polygon = room_polygons[room.room_name]
            room_portals = self._associate_room_portals(
                room_name=room.room_name,
                room_polygon=room_polygon,
                portals=portals,
                geometry=geometry,
            )
            free_space = self._build_room_free_space(
                room=room,
                room_polygon=room_polygon,
                room_portals=[portals[portal_id] for portal_id in room_portals],
                geometry=geometry,
            )
            room_geometries[room.room_name] = _RoomGeometry(
                room_name=room.room_name,
                polygon=room_polygon,
                free_space=free_space,
                blocked_space=room_polygon.difference(free_space).buffer(0),
                portals=room_portals,
            )

        for portal in portals.values():
            if len(portal.room_names) < 2:
                continue
            for room_name in list(portal.room_names):
                room = room_geometries.get(room_name)
                if room is None:
                    portal.room_names.discard(room_name)
                    continue
                portal.anchors[room_name] = self._portal_anchor(
                    room=room,
                    portal_polygon=portal.polygon,
                    geometry=geometry,
                )

        return room_geometries, portals

    def _room_polygon(self, room: SceneNavRoom, geometry):
        if room.boundary_segments:
            polygon = self._polygon_from_segments(room.boundary_segments, geometry)
            if polygon is not None and not polygon.is_empty:
                return polygon.buffer(0)
        return self._bbox_polygon(room.bounds, geometry)

    def _polygon_from_segments(self, segments: Iterable[Tuple[Point2D, Point2D]], geometry):
        lines = [
            geometry['LineString']([start, end])
            for start, end in segments
            if _distance(start, end) > 1e-9
        ]
        if not lines:
            return None
        polygons = list(geometry['polygonize'](lines))
        if not polygons:
            return None
        return max(polygons, key=lambda poly: float(poly.area))

    def _bbox_polygon(self, bbox: AxisAlignedBBox, geometry):
        return geometry['box'](bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)

    def _merge_portals(self, raw_portals: Sequence[_Portal], geometry) -> Dict[str, _Portal]:
        merged: List[_Portal] = []
        for raw_portal in raw_portals:
            matched = None
            for existing in merged:
                if existing.polygon.intersects(raw_portal.polygon):
                    matched = existing
                    break
                if existing.polygon.distance(raw_portal.polygon) <= self.portal_merge_distance:
                    matched = existing
                    break
            if matched is None:
                merged.append(
                    _Portal(
                        portal_id='',
                        polygon=raw_portal.polygon,
                        room_names=set(raw_portal.room_names),
                    )
                )
                continue
            matched.polygon = matched.polygon.union(raw_portal.polygon).buffer(0)
            matched.room_names.update(raw_portal.room_names)

        result: Dict[str, _Portal] = {}
        for index, portal in enumerate(merged):
            portal.portal_id = f'portal_{index}'
            result[portal.portal_id] = portal
        return result

    def _associate_room_portals(
        self,
        *,
        room_name: str,
        room_polygon,
        portals: Dict[str, _Portal],
        geometry,
    ) -> List[str]:
        associated: List[str] = []
        expanded_room = room_polygon.buffer(0.15)
        for portal_id, portal in portals.items():
            if expanded_room.intersects(portal.polygon) or expanded_room.distance(portal.polygon) <= 0.15:
                portal.room_names.add(room_name)
                associated.append(portal_id)
        return associated

    def _build_room_free_space(
        self,
        *,
        room: SceneNavRoom,
        room_polygon,
        room_portals: Sequence[_Portal],
        geometry,
    ):
        inflated_obstacles = []
        for bbox in room.wall_bboxes:
            inflated_obstacles.append(self._bbox_polygon(bbox, geometry).buffer(self.inflation_radius))
        for bbox in room.object_bboxes:
            if self._should_ignore_object(bbox):
                continue
            inflated_obstacles.append(self._bbox_polygon(bbox, geometry).buffer(self.inflation_radius))

        if inflated_obstacles:
            obstacle_union = geometry['unary_union'](inflated_obstacles).buffer(0)
        else:
            obstacle_union = geometry['Polygon']()

        for portal in room_portals:
            obstacle_union = obstacle_union.difference(
                portal.polygon.buffer(self.inflation_radius + 0.05)
            )

        free_space = room_polygon.difference(obstacle_union).buffer(0)
        if free_space.is_empty:
            raise NavigationPlanningError(f'navigation_no_path:no_free_space_in_room:{room.room_name}')
        return free_space

    def _should_ignore_object(self, bbox: AxisAlignedBBox) -> bool:
        lowered = str(bbox.name or '').strip().lower()
        if any(token in lowered for token in ('floor', 'ground', 'ceiling', 'light', 'camera', 'window')):
            return True
        if bbox.height <= 0.05 and bbox.max_z <= bbox.min_z + 0.1:
            return True
        return False

    def _locate_room(self, nav_map: Dict[str, _RoomGeometry], point: Point2D, geometry) -> tuple[str, Point2D]:
        point_geom = geometry['Point'](point)
        containing: List[str] = []
        for room_name, room in nav_map.items():
            if room.polygon.buffer(1e-6).covers(point_geom):
                containing.append(room_name)
        if containing:
            return containing[0], point

        nearest_room = None
        nearest_distance = float('inf')
        nearest_point = point
        for room_name, room in nav_map.items():
            candidate = self._nearest_point_on_geometry(point, room.polygon, geometry)
            distance = _distance(point, candidate)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_room = room_name
                nearest_point = candidate

        if nearest_room is None or nearest_distance > self.snap_distance:
            raise NavigationPlanningError('navigation_no_path:point_outside_rooms')
        return nearest_room, nearest_point

    def _plan_room_graph(
        self,
        *,
        nav_map: Dict[str, _RoomGeometry],
        portals: Dict[str, _Portal],
        start_room: str,
        goal_room: str,
    ) -> tuple[List[str], List[str]]:
        candidates = self._plan_room_graph_candidates(
            nav_map=nav_map,
            portals=portals,
            start_room=start_room,
            goal_room=goal_room,
        )
        best = candidates[0]
        return best.room_sequence, best.portal_sequence

    def _plan_room_graph_candidates(
        self,
        *,
        nav_map: Dict[str, _RoomGeometry],
        portals: Dict[str, _Portal],
        start_room: str,
        goal_room: str,
    ) -> List[_RouteCandidate]:
        if start_room == goal_room:
            return [_RouteCandidate(cost=0.0, room_sequence=[start_room], portal_sequence=[])]

        adjacency: Dict[str, List[Tuple[str, str, float]]] = {room_name: [] for room_name in nav_map}
        for portal_id, portal in portals.items():
            room_names = sorted(portal.room_names)
            if len(room_names) < 2:
                continue
            for index, room_name in enumerate(room_names):
                for neighbor_name in room_names[index + 1 :]:
                    edge_cost = self._room_transition_cost(
                        current_room=nav_map[room_name],
                        next_room=nav_map[neighbor_name],
                        portal=portal,
                    )
                    adjacency[room_name].append((neighbor_name, portal_id, edge_cost))
                    adjacency[neighbor_name].append((room_name, portal_id, edge_cost))

        for room_name in adjacency:
            adjacency[room_name].sort(key=lambda item: (item[2], item[0], item[1]))

        if not adjacency.get(start_room):
            raise NavigationPlanningError(
                f'navigation_no_path:rooms_not_connected:{start_room}->{goal_room}'
            )

        goal_reference = self._room_reference_point(nav_map[goal_room])
        start_path = [start_room]
        heap: List[Tuple[float, float, Tuple[str, ...], Tuple[str, ...], str]] = [
            (0.0, 0.0, tuple(start_path), tuple(), start_room)
        ]
        candidates: List[_RouteCandidate] = []
        seen_paths: set[Tuple[str, ...]] = set()

        while heap and len(candidates) < self.max_room_graph_candidates:
            _, current_cost, room_path, portal_path, current_room = heapq.heappop(heap)
            if room_path in seen_paths:
                continue
            seen_paths.add(room_path)

            if current_room == goal_room:
                candidates.append(
                    _RouteCandidate(
                        cost=current_cost,
                        room_sequence=list(room_path),
                        portal_sequence=list(portal_path),
                    )
                )
                continue

            for neighbor_room, portal_id, edge_cost in adjacency.get(current_room, []):
                if neighbor_room in room_path:
                    continue
                neighbor_path = room_path + (neighbor_room,)
                portal_sequence = portal_path + (portal_id,)
                new_cost = current_cost + edge_cost
                heuristic = _distance(self._room_reference_point(nav_map[neighbor_room]), goal_reference)
                heapq.heappush(
                    heap,
                    (
                        new_cost + heuristic,
                        new_cost,
                        neighbor_path,
                        portal_sequence,
                        neighbor_room,
                    ),
                )

        if not candidates:
            raise NavigationPlanningError(
                f'navigation_no_path:rooms_not_connected:{start_room}->{goal_room}'
            )
        return candidates

    def _portal_anchor(self, *, room: _RoomGeometry, portal_polygon, geometry) -> Point2D:
        portal_center = (
            float(portal_polygon.centroid.x),
            float(portal_polygon.centroid.y),
        )
        doorway_region = room.free_space.intersection(portal_polygon.buffer(self.grid_resolution)).buffer(0)
        if doorway_region.is_empty:
            doorway_region = room.polygon.intersection(portal_polygon.buffer(0.05)).buffer(0)

        if not doorway_region.is_empty:
            base_point_geom = doorway_region.representative_point()
            base_point = (float(base_point_geom.x), float(base_point_geom.y))
        else:
            base_point = self._nearest_point_on_geometry(portal_center, room.polygon, geometry)

        direction = np.array(
            [
                float(base_point[0] - portal_center[0]),
                float(base_point[1] - portal_center[1]),
            ],
            dtype=float,
        )
        if np.linalg.norm(direction) <= 1e-6:
            room_reference = np.array(self._room_reference_point(room), dtype=float)
            direction = room_reference - np.array(portal_center, dtype=float)
        if np.linalg.norm(direction) <= 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)

        base_distance = max(_distance(portal_center, base_point), 0.0)
        target_distance = base_distance + self.portal_waypoint_offset
        free_space_with_margin = room.free_space.buffer(1e-6)
        for clearance_scale in (1.0, 0.75, 0.5, 0.25):
            candidate = (
                portal_center[0] + direction[0] * target_distance * clearance_scale,
                portal_center[1] + direction[1] * target_distance * clearance_scale,
            )
            if free_space_with_margin.covers(geometry['Point'](candidate)):
                return candidate

        fallback_target = (
            portal_center[0] + direction[0] * target_distance,
            portal_center[1] + direction[1] * target_distance,
        )
        snapped_point = self._nearest_point_on_geometry(fallback_target, room.free_space, geometry)
        if _distance(snapped_point, portal_center) > base_distance + 1e-3:
            return snapped_point
        return base_point

    def _plan_within_room(
        self,
        *,
        room: _RoomGeometry,
        start_point: Point2D,
        goal_point: Point2D,
        geometry,
    ) -> List[Point2D]:
        free_space = room.free_space
        path_points = self._grid_astar(
            free_space=free_space,
            start_point=start_point,
            goal_point=goal_point,
            geometry=geometry,
        )
        smoothed = self._smooth_path(path_points, free_space, geometry)
        return self._prune_dense_waypoints(smoothed)

    def _plan_route_candidate(
        self,
        *,
        nav_map: Dict[str, _RoomGeometry],
        portals: Dict[str, _Portal],
        geometry,
        candidate: _RouteCandidate,
        start_point: Point2D,
        goal_point: Point2D,
    ) -> NavigationPlan:
        waypoints: List[Point2D] = []
        protected_waypoints: List[Point2D] = []
        current_point = start_point
        room_sequence = candidate.room_sequence
        portal_sequence = candidate.portal_sequence

        for index, room_name in enumerate(room_sequence):
            room = nav_map[room_name]
            is_final_segment = index == len(room_sequence) - 1
            if is_final_segment:
                segment_goal = goal_point
            else:
                portal = portals[portal_sequence[index]]
                segment_goal = portal.anchors[room_name]

            try:
                segment_points = self._plan_within_room(
                    room=room,
                    start_point=current_point,
                    goal_point=segment_goal,
                    geometry=geometry,
                )
            except NavigationPlanningError as exc:
                if self._can_fallback_from_route_error(
                    error=exc,
                    segment_index=index,
                    is_final_segment=is_final_segment,
                ):
                    raise _RouteCandidatePlanningError(str(exc)) from exc
                raise

            waypoints.extend(self._append_segment(waypoints, segment_points))

            if index < len(room_sequence) - 1:
                next_room = room_sequence[index + 1]
                portal = portals[portal_sequence[index]]
                current_point = portal.anchors[next_room]
                protected_waypoints.append(portal.anchors[room_name])
                protected_waypoints.append(current_point)
                if _distance(waypoints[-1], current_point) > 1e-6:
                    waypoints.append(current_point)
            else:
                current_point = goal_point

        if not waypoints:
            raise NavigationPlanningError('navigation_no_path:empty_plan')

        return NavigationPlan(
            waypoints=self._prune_dense_waypoints(waypoints, protected_points=protected_waypoints),
            room_sequence=room_sequence,
            portal_sequence=portal_sequence,
        )

    def _can_fallback_from_route_error(
        self,
        *,
        error: NavigationPlanningError,
        segment_index: int,
        is_final_segment: bool,
    ) -> bool:
        message = str(error)
        if message == 'navigation_no_path:grid_astar_failed':
            return True
        if message == 'navigation_start_in_collision':
            return segment_index > 0
        if message == 'navigation_goal_in_collision':
            return not is_final_segment
        if message == 'navigation_no_path:no_nearby_free_cell':
            return True
        if message.startswith('navigation_no_path:no_free_space_in_room:'):
            return True
        return False

    def _room_transition_cost(
        self,
        *,
        current_room: _RoomGeometry,
        next_room: _RoomGeometry,
        portal: _Portal,
    ) -> float:
        current_reference = self._room_reference_point(current_room)
        next_reference = self._room_reference_point(next_room)
        portal_reference = (
            float(portal.polygon.centroid.x),
            float(portal.polygon.centroid.y),
        )
        cost = (
            _distance(current_reference, portal_reference)
            + _distance(portal_reference, next_reference)
        )
        return max(cost, 1e-3)

    def _room_reference_point(self, room: _RoomGeometry) -> Point2D:
        point = room.polygon.representative_point()
        return (float(point.x), float(point.y))

    def _grid_astar(self, *, free_space, start_point: Point2D, goal_point: Point2D, geometry) -> List[Point2D]:
        min_x, min_y, max_x, max_y = free_space.bounds
        resolution = self.grid_resolution
        width = max(1, int(math.ceil((max_x - min_x) / resolution)) + 1)
        height = max(1, int(math.ceil((max_y - min_y) / resolution)) + 1)
        free_cells = [
            [False for _ in range(height)]
            for _ in range(width)
        ]
        expanded_free_space = free_space.buffer(1e-6)

        for ix in range(width):
            x = min_x + ix * resolution
            for iy in range(height):
                y = min_y + iy * resolution
                if expanded_free_space.covers(geometry['Point'](x, y)):
                    free_cells[ix][iy] = True

        start_cell = self._nearest_free_cell(
            point=start_point,
            min_x=min_x,
            min_y=min_y,
            width=width,
            height=height,
            resolution=resolution,
            free_cells=free_cells,
            point_role='start',
        )
        goal_cell = self._nearest_free_cell(
            point=goal_point,
            min_x=min_x,
            min_y=min_y,
            width=width,
            height=height,
            resolution=resolution,
            free_cells=free_cells,
            point_role='goal',
        )

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start_cell))
        costs: Dict[Tuple[int, int], float] = {start_cell: 0.0}
        parents: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current == goal_cell:
                break
            if current_cost > costs.get(current, float('inf')):
                continue
            for neighbor in self._neighbors(current, width=width, height=height):
                nx, ny = neighbor
                if not free_cells[nx][ny]:
                    continue
                step_cost = math.sqrt(2.0) if nx != current[0] and ny != current[1] else 1.0
                new_cost = current_cost + step_cost
                if new_cost >= costs.get(neighbor, float('inf')):
                    continue
                costs[neighbor] = new_cost
                parents[neighbor] = current
                heuristic = math.hypot(goal_cell[0] - nx, goal_cell[1] - ny)
                heapq.heappush(open_heap, (new_cost + heuristic, new_cost, neighbor))

        if goal_cell not in costs:
            raise NavigationPlanningError('navigation_no_path:grid_astar_failed')

        cells: List[Tuple[int, int]] = [goal_cell]
        cursor = goal_cell
        while cursor != start_cell:
            cursor = parents[cursor]
            cells.append(cursor)
        cells.reverse()

        points = [
            (min_x + ix * resolution, min_y + iy * resolution)
            for ix, iy in cells
        ]
        if points:
            points[0] = start_point if _distance(start_point, points[0]) <= self.snap_distance else points[0]
            points[-1] = goal_point if _distance(goal_point, points[-1]) <= self.snap_distance else points[-1]
        return points

    def _nearest_free_cell(
        self,
        *,
        point: Point2D,
        min_x: float,
        min_y: float,
        width: int,
        height: int,
        resolution: float,
        free_cells: Sequence[Sequence[bool]],
        point_role: str,
    ) -> Tuple[int, int]:
        base_ix = int(round((point[0] - min_x) / resolution))
        base_iy = int(round((point[1] - min_y) / resolution))
        base_ix = min(max(base_ix, 0), width - 1)
        base_iy = min(max(base_iy, 0), height - 1)
        if free_cells[base_ix][base_iy]:
            return (base_ix, base_iy)

        max_radius = max(1, int(math.ceil(self.snap_distance / resolution)))
        best_cell = None
        best_distance = float('inf')

        for radius in range(1, max_radius + 1):
            for ix in range(max(0, base_ix - radius), min(width, base_ix + radius + 1)):
                for iy in range(max(0, base_iy - radius), min(height, base_iy + radius + 1)):
                    if not free_cells[ix][iy]:
                        continue
                    candidate = (min_x + ix * resolution, min_y + iy * resolution)
                    distance = _distance(point, candidate)
                    if distance < best_distance:
                        best_distance = distance
                        best_cell = (ix, iy)
            if best_cell is not None:
                break

        if best_cell is None or best_distance > self.snap_distance:
            if point_role == 'goal':
                raise NavigationPlanningError('navigation_goal_in_collision')
            if point_role == 'start':
                raise NavigationPlanningError('navigation_start_in_collision')
            raise NavigationPlanningError('navigation_no_path:no_nearby_free_cell')
        return best_cell

    def _neighbors(self, cell: Tuple[int, int], *, width: int, height: int) -> Iterable[Tuple[int, int]]:
        cx, cy = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    yield (nx, ny)

    def _smooth_path(self, points: Sequence[Point2D], free_space, geometry) -> List[Point2D]:
        if len(points) <= 2:
            return list(points)
        smoothed = [points[0]]
        anchor_index = 0
        while anchor_index < len(points) - 1:
            candidate_index = len(points) - 1
            while candidate_index > anchor_index + 1:
                segment = geometry['LineString']([points[anchor_index], points[candidate_index]])
                if free_space.buffer(1e-6).covers(segment):
                    break
                candidate_index -= 1
            smoothed.append(points[candidate_index])
            anchor_index = candidate_index
        return smoothed

    def _prune_dense_waypoints(
        self,
        points: Sequence[Point2D],
        *,
        protected_points: Optional[Sequence[Point2D]] = None,
    ) -> List[Point2D]:
        if not points:
            return []
        protected = list(protected_points or [])
        pruned = [points[0]]
        for point in points[1:-1]:
            if self._is_protected_waypoint(point, protected):
                if _distance(pruned[-1], point) > 1e-6:
                    pruned.append(point)
                continue
            if _distance(pruned[-1], point) >= self.waypoint_spacing:
                pruned.append(point)
        if len(points) > 1 and _distance(pruned[-1], points[-1]) > 1e-6:
            pruned.append(points[-1])
        return pruned

    def _append_segment(self, accumulated: Sequence[Point2D], segment: Sequence[Point2D]) -> List[Point2D]:
        if not segment:
            return []
        if not accumulated:
            return list(segment)
        if _distance(accumulated[-1], segment[0]) <= 1e-6:
            return list(segment[1:])
        return list(segment)

    def _nearest_point_on_geometry(self, point: Point2D, geometry_obj, geometry) -> Point2D:
        query_point = geometry['Point'](point)
        _, nearest = geometry['nearest_points'](query_point, geometry_obj)
        return (float(nearest.x), float(nearest.y))

    def _is_protected_waypoint(
        self,
        point: Point2D,
        protected_points: Sequence[Point2D],
    ) -> bool:
        return any(_distance(point, protected_point) <= 1e-6 for protected_point in protected_points)


def _distance(p1: Point2D, p2: Point2D) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

import omni
from pxr import Usd, UsdGeom, Gf, UsdShade, UsdPhysics,Sdf,PhysicsSchemaTools
import time
from Tools import design_room
from Tools.mounted_placement import (
    compute_mounted_surface_center,
    infer_mounted_surface,
    is_picture_mounted_object,
    resolve_mounted_orientation,
)
from Tools.asset_physics import (
    is_collision_evaluation_exempt_asset,
    is_nonphysical_floating_asset,
    is_nonphysical_floating_prim,
    is_static_collider_asset,
    set_base_link_mass_override,
    stabilize_mounted_rigid_bodies,
)
import os
import math
import json
import random
import traceback
from typing import Optional

_uniform_scale_baselines = {}
_SURFACE_PLACEMENT_CLEARANCE = 0.04


def _uniform_scale_key(stage, path: str):
    return (stage.GetRootLayer().identifier, path)

# Handlers related to Isaac Sim stubs, implemented in the simulator runtime.

# USD prim path organization:
# World
# ---room_kitchen
# --------structure
# ----------walls
# ----------floor
# --------objects
# ---------cabinet_01
# ---room_livingroom

def scan_scene(room_name: str) -> list:
    '''Scans the current stage and returns a list of all objects (including walls, floors, and spawned assets).

    Returns:
        A list of dictionaries, where each dictionary contains:
        {'name': str, 'num': int, 'bbox': {'min': [x,y,z], 'max': [x,y,z]}, 'position': [x,y,z]}.
    '''
    stage = omni.usd.get_context().get_stage()
    if room_name == '':
        path = "/World/rooms"
    else:
        path = f"/World/rooms/{room_name}"
    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"failed to find object at path:{path}")
        return None
    
    # Get only the first level of child prims
    result_prim_paths = []
    for child_prim in prim.GetChildren():
        # Skip Scope type prims
        if child_prim.GetTypeName() == "Scope":
            continue
        # Only add first-level Xform type children

        result_prim_paths.append(str(child_prim.GetPath()))

    # Get bounding box and position for each prim
    result = []
    for prim_path in result_prim_paths:
        prim_pos = get_object_position(prim_path)
        prim_bbox = get_object_bbox(prim_path)
        prim_name = prim_path.rsplit("/", 1)[-1]
        result.append({
            'name': prim_name,
            'bbox': {'min': prim_bbox[0], 'max': prim_bbox[1]},
            'position': prim_pos
        })
    return result


# ---------------------------------------------------------------------------
# Semantic scene perception helpers
# ---------------------------------------------------------------------------

def _classify_prim_name(name: str) -> str:
    """Classify a prim name into a semantic category."""
    lower = name.lower()
    if lower.startswith("wall"):
        return "wall"
    if lower.startswith("floor") or lower == "groundplane":
        return "floor"
    if lower.startswith("ceiling"):
        return "ceiling"
    if lower.startswith("door"):
        return "door"
    return "object"


def _wall_side_label(name: str, bbox_min, bbox_max, room_x_min, room_x_max, room_y_min, room_y_max) -> str | None:
    """Determine which side of the room a wall is on based on its bbox center."""
    cx = (float(bbox_min[0]) + float(bbox_max[0])) / 2.0
    cy = (float(bbox_min[1]) + float(bbox_max[1])) / 2.0
    rx = (room_x_max - room_x_min) / 2.0 if (room_x_max - room_x_min) > 0 else 1.0
    ry = (room_y_max - room_y_min) / 2.0 if (room_y_max - room_y_min) > 0 else 1.0
    mid_x = (room_x_min + room_x_max) / 2.0
    mid_y = (room_y_min + room_y_max) / 2.0

    dx = (cx - mid_x) / rx if rx > 1e-6 else 0.0
    dy = (cy - mid_y) / ry if ry > 1e-6 else 0.0

    if abs(dx) > abs(dy):
        if dx > 0:
            return f"right (x={round(float(bbox_max[0]), 2)})"
        else:
            return f"left (x={round(float(bbox_min[0]), 2)})"
    else:
        if dy > 0:
            return f"front (y={round(float(bbox_max[1]), 2)})"
        else:
            return f"back (y={round(float(bbox_min[1]), 2)})"


def get_room_context(room_name: str) -> dict:
    """Build a semantic description of the room layout.

    Returns a structured dict with room bounds, walls, corners, doors,
    placed objects, and available surfaces — designed for LLM comprehension.
    """
    raw = scan_scene(room_name)
    if not raw:
        return {"error": "no objects found", "room": None}

    # Separate structural and placed objects
    walls = []
    floor_prims = []
    ceilings = []
    doors = []
    placed_objects = []
    for obj in raw:
        cat = _classify_prim_name(obj['name'])
        bmin = obj['bbox']['min']
        bmax = obj['bbox']['max']
        entry = {
            'name': obj['name'],
            'size': [round(float(bmax[i] - bmin[i]), 2) for i in range(3)],
        }
        if cat == "wall":
            walls.append(obj)
        elif cat == "floor":
            floor_prims.append(obj)
        elif cat == "ceiling":
            ceilings.append(entry)
        elif cat == "door":
            doors.append(entry)
        else:
            placed_objects.append(obj)

    # Determine room bounds from walls
    room_x_min = min(float(w['bbox']['min'][0]) for w in walls) if walls else 0.0
    room_x_max = max(float(w['bbox']['max'][0]) for w in walls) if walls else 0.0
    room_y_min = min(float(w['bbox']['min'][1]) for w in walls) if walls else 0.0
    room_y_max = max(float(w['bbox']['max'][1]) for w in walls) if walls else 0.0
    room_z_min = min(float(w['bbox']['min'][2]) for w in walls) if walls else 0.0
    room_z_max = max(float(w['bbox']['max'][2]) for w in walls) if walls else 0.0

    # Build wall descriptions
    wall_descs = []
    for w in walls:
        bmin = w['bbox']['min']
        bmax = w['bbox']['max']
        length = max(float(bmax[0] - bmin[0]), float(bmax[1] - bmin[1]))
        side = _wall_side_label(w['name'], bmin, bmax, room_x_min, room_x_max, room_y_min, room_y_max)
        wall_descs.append({
            'name': w['name'],
            'side': side,
            'length': round(length, 2),
        })

    # Build corner descriptions
    corners = [
        {'name': 'back_left',  'x': round(room_x_min, 2), 'y': round(room_y_min, 2), 'wall_x': None, 'wall_y': None},
        {'name': 'back_right', 'x': round(room_x_max, 2), 'y': round(room_y_min, 2), 'wall_x': None, 'wall_y': None},
        {'name': 'front_left', 'x': round(room_x_min, 2), 'y': round(room_y_max, 2), 'wall_x': None, 'wall_y': None},
        {'name': 'front_right','x': round(room_x_max, 2), 'y': round(room_y_max, 2), 'wall_x': None, 'wall_y': None},
    ]
    # Match walls to corners
    for c in corners:
        for wd in wall_descs:
            side = wd['side']
            if c['wall_x'] is None:
                if ('x=' in side) and (('left' in side and abs(c['x'] - room_x_min) < 0.1) or
                                        ('right' in side and abs(c['x'] - room_x_max) < 0.1)):
                    c['wall_x'] = wd['name']
            if c['wall_y'] is None:
                if ('y=' in side) and (('back' in side and abs(c['y'] - room_y_min) < 0.1) or
                                        ('front' in side and abs(c['y'] - room_y_max) < 0.1)):
                    c['wall_y'] = wd['name']

    # Build placed object descriptions with semantic info
    obj_descs = []
    for obj in placed_objects:
        bmin = obj['bbox']['min']
        bmax = obj['bbox']['max']
        cx = (float(bmin[0]) + float(bmax[0])) / 2.0
        cy = (float(bmin[1]) + float(bmax[1])) / 2.0
        cz = (float(bmin[2]) + float(bmax[2])) / 2.0
        size = [round(float(bmax[i] - bmin[i]), 2) for i in range(3)]

        # Determine which walls this object is against
        against_walls = []
        wall_margin = 0.15  # tolerance for "against wall"
        for wd in wall_descs:
            wname = wd['name']
            wobj = next(w for w in walls if w['name'] == wname)
            wbmin = wobj['bbox']['min']
            wbmax = wobj['bbox']['max']
            # Check X overlap
            if float(bmin[0]) < float(wbmax[0]) + wall_margin and float(bmax[0]) > float(wbmin[0]) - wall_margin:
                # Check Y overlap
                if float(bmin[1]) < float(wbmax[1]) + wall_margin and float(bmax[1]) > float(wbmin[1]) - wall_margin:
                    # Check proximity to wall
                    if abs(float(bmin[0]) - float(wbmin[0])) < wall_margin or abs(float(bmax[0]) - float(wbmax[0])) < wall_margin:
                        against_walls.append(wname)
                    elif abs(float(bmin[1]) - float(wbmin[1])) < wall_margin or abs(float(bmax[1]) - float(wbmax[1])) < wall_margin:
                        against_walls.append(wname)

        # Check if it's in a corner (against two walls)
        corner_name = None
        if len(against_walls) >= 2:
            for c in corners:
                if c['wall_x'] in against_walls and c['wall_y'] in against_walls:
                    corner_name = c['name']
                    break

        # Determine height category for placement type
        height = size[2]
        if height < 0.3:
            size_category = "small"
        elif height < 1.0:
            size_category = "medium"
        else:
            size_category = "large"

        obj_descs.append({
            'name': obj['name'],
            'size': size,
            'size_category': size_category,
            'against_walls': against_walls if against_walls else None,
            'corner': corner_name,
            'center_xy': [round(cx, 2), round(cy, 2)],
        })

    # Build available surface descriptions
    available_surfaces = []
    for obj in placed_objects:
        bmin = obj['bbox']['min']
        bmax = obj['bbox']['max']
        # Check if this object has a usable top surface (height > 0.3m and reasonable area)
        top_z = float(bmax[2])
        top_area = float(bmax[0] - bmin[0]) * float(bmax[1] - bmin[1])
        if top_z > 0.3 and top_area > 0.01:
            # Find objects already on this surface
            objects_on_top = []
            for other in placed_objects:
                if other['name'] == obj['name']:
                    continue
                obmin = other['bbox']['min']
                obmax = other['bbox']['max']
                # Check if the other object is resting on top of this one
                other_bottom = float(obmin[2])
                other_center_x = (float(obmin[0]) + float(obmax[0])) / 2.0
                other_center_y = (float(obmin[1]) + float(obmax[1])) / 2.0
                if (abs(other_bottom - top_z) < 0.05 and
                    float(obmin[0]) >= float(bmin[0]) - 0.01 and float(obmax[0]) <= float(bmax[0]) + 0.01 and
                    float(obmin[1]) >= float(bmin[1]) - 0.01 and float(obmax[1]) <= float(bmax[1]) + 0.01):
                    objects_on_top.append(other['name'])

            available_surfaces.append({
                'name': obj['name'],
                'surface': 'up',
                'height': round(top_z, 2),
                'area_m2': round(top_area, 2),
                'size_xy': [round(float(bmax[0] - bmin[0]), 2), round(float(bmax[1] - bmin[1]), 2)],
                'objects_on_surface': objects_on_top if objects_on_top else None,
            })

    result = {
        'room': {
            'x_range': [round(room_x_min, 2), round(room_x_max, 2)],
            'y_range': [round(room_y_min, 2), round(room_y_max, 2)],
            'z_range': [round(room_z_min, 2), round(room_z_max, 2)],
            'size_m': [round(room_x_max - room_x_min, 2), round(room_y_max - room_y_min, 2), round(room_z_max - room_z_min, 2)],
            'area_m2': round((room_x_max - room_x_min) * (room_y_max - room_y_min), 2),
        },
        'walls': wall_descs,
        'mounting_surfaces': {
            'walls': [wall['name'] for wall in wall_descs],
            'ceilings': [ceiling['name'] for ceiling in ceilings],
        },
        'corners': corners,
        'doors': doors if doors else None,
        'placed_objects': obj_descs if obj_descs else None,
        'available_surfaces': available_surfaces if available_surfaces else None,
    }
    return result


def query_floor_space(room_name: str, near_wall: str | None = None, region: str | None = None) -> dict:
    """Query available floor space in the room.

    Returns occupied and free rectangular regions on the floor plane,
    optionally filtered to be near a specific wall or in a specific region type.
    """
    raw = scan_scene(room_name)
    if not raw:
        return {"error": "no objects found"}

    # Determine room bounds from walls
    walls = [o for o in raw if _classify_prim_name(o['name']) == 'wall']
    if not walls:
        return {"error": "no walls found"}

    room_x_min = min(float(w['bbox']['min'][0]) for w in walls)
    room_x_max = max(float(w['bbox']['max'][0]) for w in walls)
    room_y_min = min(float(w['bbox']['min'][1]) for w in walls)
    room_y_max = max(float(w['bbox']['max'][1]) for w in walls)

    # Collect floor-occupied regions from non-wall objects
    occupied = []
    for obj in raw:
        cat = _classify_prim_name(obj['name'])
        if cat in ("wall", "floor", "ceiling"):
            continue
        bmin = obj['bbox']['min']
        bmax = obj['bbox']['max']
        occupied.append({
            'name': obj['name'],
            'x_range': [round(float(bmin[0]), 2), round(float(bmax[0]), 2)],
            'y_range': [round(float(bmin[1]), 2), round(float(bmax[1]), 2)],
            'z_range': [round(float(bmin[2]), 2), round(float(bmax[2]), 2)],
        })

    # Build a simple free region description
    # For now, report the overall room footprint and occupied list
    # A full rectangle-subtraction would be complex; the key info for LLM is
    # which regions are taken and which walls are available.
    free_regions = []
    # We report four wall-adjacent strips and the open center
    wall_margin = 0.5  # half-width of "near wall" strip
    if near_wall:
        # Find the specific wall
        target_wall = next((w for w in walls if w['name'] == near_wall), None)
        if target_wall:
            wbmin = target_wall['bbox']['min']
            wbmax = target_wall['bbox']['max']
            side = _wall_side_label(near_wall, wbmin, wbmax, room_x_min, room_x_max, room_y_min, room_y_max)
            # Report the strip along this wall
            if 'left' in side:
                strip_x_max = room_x_min + wall_margin * 2
                free_regions.append({
                    'near_wall': near_wall,
                    'side': side,
                    'x_range': [round(room_x_min, 2), round(strip_x_max, 2)],
                    'y_range': [round(room_y_min, 2), round(room_y_max, 2)],
                })
            elif 'right' in side:
                strip_x_min = room_x_max - wall_margin * 2
                free_regions.append({
                    'near_wall': near_wall,
                    'side': side,
                    'x_range': [round(strip_x_min, 2), round(room_x_max, 2)],
                    'y_range': [round(room_y_min, 2), round(room_y_max, 2)],
                })
            elif 'back' in side:
                strip_y_max = room_y_min + wall_margin * 2
                free_regions.append({
                    'near_wall': near_wall,
                    'side': side,
                    'x_range': [round(room_x_min, 2), round(room_x_max, 2)],
                    'y_range': [round(room_y_min, 2), round(strip_y_max, 2)],
                })
            elif 'front' in side:
                strip_y_min = room_y_max - wall_margin * 2
                free_regions.append({
                    'near_wall': near_wall,
                    'side': side,
                    'x_range': [round(room_x_min, 2), round(room_x_max, 2)],
                    'y_range': [round(strip_y_min, 2), round(room_y_max, 2)],
                })
    elif region == "center":
        # Open center area (not near any wall)
        free_regions.append({
            'region': 'center',
            'x_range': [round(room_x_min + wall_margin, 2), round(room_x_max - wall_margin, 2)],
            'y_range': [round(room_y_min + wall_margin, 2), round(room_y_max - wall_margin, 2)],
        })
    else:
        # Report the full room footprint
        free_regions.append({
            'region': 'full_floor',
            'x_range': [round(room_x_min, 2), round(room_x_max, 2)],
            'y_range': [round(room_y_min, 2), round(room_y_max, 2)],
        })

    return {
        'room_bounds': {
            'x_range': [round(room_x_min, 2), round(room_x_max, 2)],
            'y_range': [round(room_y_min, 2), round(room_y_max, 2)],
        },
        'occupied_regions': occupied if occupied else None,
        'free_regions': free_regions,
    }


def query_surface_status(room_name: str, support_object: str, surface: str = "up") -> dict:
    """Query the occupancy status of a specific surface on a support object.

    Returns the surface dimensions, objects already on it, and remaining free area.
    """
    stage = omni.usd.get_context().get_stage()
    support_path = f"/World/rooms/{room_name}/{support_object}"
    prim = stage.GetPrimAtPath(support_path)
    if not prim or not prim.IsValid():
        return {"error": f"support object '{support_object}' not found"}

    bbox = get_object_bbox(support_path)
    if bbox is None:
        return {"error": f"could not get bbox for '{support_object}'"}

    smin, smax = bbox
    # Determine surface plane
    plane_cfg = _surface_plane_config(surface)
    if plane_cfg is None:
        return {"error": f"unknown surface '{surface}'; use up/down/left/right/front/back"}

    plane_axis_0, plane_axis_1 = plane_cfg["plane_axes"]
    normal_axis = plane_cfg["normal_axis"]
    normal_sign = plane_cfg["normal_sign"]

    surface_size = [round(float(smax[i] - smin[i]), 2) for i in range(3)]
    surface_area = surface_size[plane_axis_0] * surface_size[plane_axis_1]

    # Surface center
    surface_center = [round((float(smin[i]) + float(smax[i])) / 2.0, 2) for i in range(3)]
    if normal_sign > 0:
        surface_center[normal_axis] = round(float(smax[normal_axis]), 2)
    else:
        surface_center[normal_axis] = round(float(smin[normal_axis]), 2)

    # Find objects on this surface
    raw = scan_scene(room_name)
    objects_on_surface = []
    occupied_area = 0.0
    if raw:
        for obj in raw:
            if obj['name'] == support_object or _classify_prim_name(obj['name']) in ("wall", "floor", "ceiling"):
                continue
            obmin = obj['bbox']['min']
            obmax = obj['bbox']['max']
            # Check if this object sits on the target surface
            obj_bottom_on_normal = float(obmin[normal_axis]) if normal_sign > 0 else float(obmax[normal_axis])
            surface_z = float(smax[normal_axis]) if normal_sign > 0 else float(smin[normal_axis])
            if abs(obj_bottom_on_normal - surface_z) > 0.05:
                continue
            # Check XY overlap with support
            overlap_0 = min(float(obmax[plane_axis_0]), float(smax[plane_axis_0])) - max(float(obmin[plane_axis_0]), float(smin[plane_axis_0]))
            overlap_1 = min(float(obmax[plane_axis_1]), float(smax[plane_axis_1])) - max(float(obmin[plane_axis_1]), float(smin[plane_axis_1]))
            if overlap_0 > 0 and overlap_1 > 0:
                obj_size = [round(float(obmax[i]) - float(obmin[i]), 2) for i in range(3)]
                objects_on_surface.append({
                    'name': obj['name'],
                    'size': obj_size,
                    'center_xy_on_surface': [
                        round((float(obmin[plane_axis_0]) + float(obmax[plane_axis_0])) / 2.0, 2),
                        round((float(obmin[plane_axis_1]) + float(obmax[plane_axis_1])) / 2.0, 2),
                    ],
                })
                occupied_area += overlap_0 * overlap_1

    remaining_area = max(0.0, surface_area - occupied_area)

    return {
        'support_object': support_object,
        'surface': surface,
        'surface_center': surface_center,
        'surface_size': surface_size,
        'surface_area_m2': round(surface_area, 2),
        'objects_on_surface': objects_on_surface if objects_on_surface else None,
        'occupied_area_m2': round(occupied_area, 2),
        'remaining_area_m2': round(remaining_area, 2),
        'is_empty': len(objects_on_surface) == 0,
    }

def get_object_bbox(name_path: str, room_name: Optional[str] = None):
    '''get bounding box size of certain object given from name or path

    args:
        name_path: unique name or path of the object needed
    '''
    # This obtains the collision box in the object's current spatial state;
    # if position or orientation has been changed, it should be re-fetched.
    stage = omni.usd.get_context().get_stage()
    # Get the prim
    if not name_path.startswith("/World"):
        # Agent calls index objects by name
        path = f"/World/rooms/{room_name}/" + name_path if room_name else "/World/" + name_path
    else:
        path = name_path
    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"failed to find object at path:{path}")
        return None
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[
                                    UsdGeom.Tokens.default_])
        bbox_cache.Clear()
        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        prim_range = prim_bbox.ComputeAlignedRange()

        min_pos = prim_range.GetMin()
        max_pos = prim_range.GetMax()
        print(f"Object at path: {path} has bounding box min: {min_pos}, max: {max_pos}")
        return (min_pos,max_pos)
    except Exception as e:
        print(f"Error creating room: {str(e)}")
        return None

def get_object_position(path:str):
    '''modify orientation of an object by rotating it

    args:
        path: prim path of the object
        rotation: dict specifying rotation angles in degrees on x,y,z axes
    '''
    # scan_scene can get scene info; here we implement get_object_position to get a single prim's position.
    stage = omni.usd.get_context().get_stage()
    # Get the prim
    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"failed to find object at path:{path}")
        return False
    try:
        # Get or create translate op
        xformable = UsdGeom.Xformable(prim)
        ordered_ops = xformable.GetOrderedXformOps()
        translate_op = None
        for op in ordered_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        return translate_op.Get()
    except Exception as e:
        print(f"Error creating room: {str(e)}")
        return None


def _extract_room_name_from_prim_path(prim_path: str) -> Optional[str]:
    parts = prim_path.split("/")
    if len(parts) >= 4 and parts[1] == "World" and parts[2] == "rooms":
        return parts[3]
    return None


def _normalize_point_2d(point) -> tuple[float, float]:
    return (round(float(point[0]), 6), round(float(point[1]), 6))


def _point_on_segment_2d(point, seg_start, seg_end, eps: float = 1e-6) -> bool:
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return min(x1, x2) - eps <= px <= max(x1, x2) + eps and min(y1, y2) - eps <= py <= max(y1, y2) + eps


def _load_room_boundary_segments(room_name: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    stage = omni.usd.get_context().get_stage()
    room_prim = stage.GetPrimAtPath(f"/World/rooms/{room_name}")
    if not room_prim or not room_prim.IsValid():
        print(f"Failed to find room prim for polygon lookup: {room_name}")
        return []

    serialized_segments = room_prim.GetCustomDataByKey("room_boundary_segments")
    if not serialized_segments:
        print(f"No room boundary metadata found on room: {room_name}")
        return []
    try:
        segments = json.loads(serialized_segments) if isinstance(serialized_segments, str) else serialized_segments
    except Exception as e:
        print(f"Failed to parse room boundary metadata for room {room_name}: {e}")
        return []
    normalized_segments = []
    for start, end in segments:
        p1 = _normalize_point_2d(start)
        p2 = _normalize_point_2d(end)
        if p1 != p2:
            normalized_segments.append((p1, p2))
    return normalized_segments


def _room_center_xy(room_name: str) -> Optional[tuple[float, float]]:
    segments = _load_room_boundary_segments(room_name)
    if segments:
        points = [point for segment in segments for point in segment]
        return (
            (min(point[0] for point in points) + max(point[0] for point in points)) / 2.0,
            (min(point[1] for point in points) + max(point[1] for point in points)) / 2.0,
        )

    raw_objects = scan_scene(room_name) or []
    wall_bounds = [
        obj.get("bbox")
        for obj in raw_objects
        if isinstance(obj, dict)
        and str(obj.get("name", "")).lower().startswith("wall")
        and isinstance(obj.get("bbox"), dict)
    ]
    if not wall_bounds:
        return None
    return (
        (
            min(float(bounds["min"][0]) for bounds in wall_bounds)
            + max(float(bounds["max"][0]) for bounds in wall_bounds)
        )
        / 2.0,
        (
            min(float(bounds["min"][1]) for bounds in wall_bounds)
            + max(float(bounds["max"][1]) for bounds in wall_bounds)
        )
        / 2.0,
    )


def _point_in_room_by_segments(point, segments, eps: float = 1e-6) -> bool:
    if not segments:
        return False

    for seg_start, seg_end in segments:
        if _point_on_segment_2d(point, seg_start, seg_end, eps=eps):
            return True

    x, y = point
    intersections = 0
    for (x1, y1), (x2, y2) in segments:
        if abs(y1 - y2) <= eps:
            if abs(y - y1) <= eps and min(x1, x2) - eps <= x:
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


def _room_vertical_bounds(room_name: str) -> tuple[float, Optional[float]]:
    """Best-effort floor/ceiling Z bounds for a room.

    The 2D room-boundary segments only encode XY.  Floor-placement validation
    also needs to ensure the resolved world-space bbox center does not put the
    object's bottom below the room floor.  Prefer explicit floor prims when
    available; otherwise fall back to the minimum wall Z.
    """
    floor_tops = []
    structural_mins = []
    structural_maxs = []
    try:
        for obj in scan_scene(room_name) or []:
            if not isinstance(obj, dict):
                continue
            name = str(obj.get("name") or "")
            bbox = obj.get("bbox") or {}
            bmin = bbox.get("min")
            bmax = bbox.get("max")
            if not isinstance(bmin, (list, tuple)) or not isinstance(bmax, (list, tuple)):
                continue
            try:
                min_z = float(bmin[2])
                max_z = float(bmax[2])
            except (TypeError, ValueError, IndexError):
                continue
            cat = _classify_prim_name(name)
            lower = name.lower()
            if cat in {"wall", "floor", "ceiling"} or "groundplane" in lower:
                structural_mins.append(min_z)
                structural_maxs.append(max_z)
            if cat == "floor" or "groundplane" in lower:
                # A floor can have thickness; the walkable plane is its top.
                floor_tops.append(max_z)
    except Exception:
        pass

    floor_z = max(floor_tops) if floor_tops else (min(structural_mins) if structural_mins else 0.0)
    ceiling_z = max(structural_maxs) if structural_maxs else None
    return float(floor_z), (float(ceiling_z) if ceiling_z is not None else None)


def _check_room_bounds(room_name: str, position, obj_half_sizes):
    segments = _load_room_boundary_segments(room_name)
    if not segments:
        return False, {"corners_outside": [], "room_name": room_name, "reason": "missing_room_boundary_segments", "out_of_bounds": {"x": 0.0, "y": 0.0, "z": 0.0}}

    # obj_half_sizes is already measured at the requested target rotation.
    x_half = float(obj_half_sizes[0])
    y_half = float(obj_half_sizes[1])
    z_half = float(obj_half_sizes[2])
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    footprint_corners = [
        (x - x_half, y - y_half),
        (x - x_half, y + y_half),
        (x + x_half, y - y_half),
        (x + x_half, y + y_half),
    ]

    outside_corners = []
    for corner in footprint_corners:
        if not _point_in_room_by_segments(corner, segments):
            outside_corners.append((round(corner[0], 4), round(corner[1], 4)))

    # Compute per-axis overflow: how far the footprint extends beyond the
    # room boundary on each axis.  Positive => overflows in +axis direction,
    # negative => overflows in -axis direction, 0.0 => within bounds.
    all_x = [pt[0] for seg in segments for pt in seg]
    all_y = [pt[1] for seg in segments for pt in seg]
    room_x_min, room_x_max = min(all_x), max(all_x)
    room_y_min, room_y_max = min(all_y), max(all_y)
    obj_x_min, obj_x_max = x - x_half, x + x_half
    obj_y_min, obj_y_max = y - y_half, y + y_half

    out_x = 0.0
    if obj_x_min < room_x_min:
        out_x = obj_x_min - room_x_min  # negative value
    elif obj_x_max > room_x_max:
        out_x = obj_x_max - room_x_max  # positive value

    out_y = 0.0
    if obj_y_min < room_y_min:
        out_y = obj_y_min - room_y_min  # negative value
    elif obj_y_max > room_y_max:
        out_y = obj_y_max - room_y_max  # positive value

    floor_z, ceiling_z = _room_vertical_bounds(room_name)
    obj_z_min = z - z_half
    obj_z_max = z + z_half
    z_tolerance = 0.03
    out_z = 0.0
    z_reason = None
    if obj_z_min < floor_z - z_tolerance:
        out_z = obj_z_min - floor_z  # negative value
        z_reason = "below_room_floor"
    elif ceiling_z is not None and obj_z_max > ceiling_z + z_tolerance:
        out_z = obj_z_max - ceiling_z  # positive value
        z_reason = "above_room_ceiling"

    in_bounds = len(outside_corners) == 0 and out_z == 0.0
    detail = {
        "corners_outside": outside_corners,
        "room_name": room_name,
        "candidate_position": tuple(round(float(v), 4) for v in position),
        "out_of_bounds": {"x": round(out_x, 4), "y": round(out_y, 4), "z": round(out_z, 4)},
        "floor_z": round(floor_z, 4),
        "candidate_bbox_z_range": (round(obj_z_min, 4), round(obj_z_max, 4)),
    }
    if ceiling_z is not None:
        detail["ceiling_z"] = round(ceiling_z, 4)
    if z_reason:
        detail["z_reason"] = z_reason
    return in_bounds, detail


def _bind_pbr_materials_and_remove_vray(stage, prim_path_to_spawn: str) -> None:
    """Bind available PBR materials to meshes and remove unsupported VRay materials."""
    looks_path = prim_path_to_spawn + "/Looks"
    looks_prim = stage.GetPrimAtPath(looks_path)
    if not looks_prim or not looks_prim.IsValid():
        # print(f"No Looks found at path: {looks_path}")
        return

    vray_material_paths = []
    for material_prim in looks_prim.GetChildren():
        if material_prim.GetTypeName() != "Material":
            continue

        material_name = material_prim.GetName()
        if material_name.endswith("_vray"):
            vray_material_paths.append(str(material_prim.GetPath()))
            continue

        # material naming convention: _primname_meshname_pbr
        try:
            mesh_name = material_name.split("_", 2)[2].rsplit("_", 1)[0]
        except Exception:
            # print(f"Skip unexpected material name format: {material_name}")
            continue

        mesh_path = prim_path_to_spawn + "/" + mesh_name + "/visuals"
        # print(f"Binding material {material_name} to mesh at path {mesh_path}")

        mesh_prim = stage.GetPrimAtPath(mesh_path)
        material_obj = UsdShade.Material(material_prim)
        if mesh_prim and mesh_prim.IsValid():
            binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
            binding_api.Bind(material_obj)
            # print(f"Bound material {material_name} to mesh {mesh_name}")

    for vray_material_path in vray_material_paths:
        if stage.RemovePrim(vray_material_path):
            # print(f"Removed unsupported VRay material prim: {vray_material_path}")
            pass
        elif _disable_prim_in_current_layer(vray_material_path):
            # print(f"Disabled referenced VRay material prim via override: {vray_material_path}")
            pass
        else:
            # print(f"Failed to suppress VRay material prim: {vray_material_path}")
            pass


def _disable_prim_in_current_layer(prim_path: str) -> bool:
    """Disable a referenced prim by authoring an override in the current edit target."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False

    try:
        prim.SetActive(False)
        return True
    except Exception:
        pass

    try:
        override_prim = stage.OverridePrim(prim_path)
        if override_prim and override_prim.IsValid():
            override_prim.SetActive(False)
            return True
    except Exception:
        pass

    return False


def _hide_prim_in_current_layer(prim_path: str) -> bool:
    """Hide a prim via visibility override when removal/deactivation is not possible."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return False

    try:
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeInvisible()
        return True
    except Exception:
        pass

    try:
        override_prim = stage.OverridePrim(prim_path)
        if override_prim and override_prim.IsValid():
            imageable = UsdGeom.Imageable(override_prim)
            imageable.MakeInvisible()
            return True
    except Exception:
        pass

    return False


def _strip_rigid_body_keep_colliders(stage, prim_root_path: str) -> None:
    """Remove rigid-body-related APIs from a prim subtree while preserving collider schemas."""
    root_prim = stage.GetPrimAtPath(prim_root_path)
    if not root_prim or not root_prim.IsValid():
        print(f"Failed to find prim for rigid-body cleanup: {prim_root_path}")
        return

    # Optional schema in some Isaac versions.
    try:
        from pxr import PhysxSchema  # type: ignore
    except Exception:
        PhysxSchema = None

    def _iter_subtree(prim):
        yield prim
        for child in prim.GetChildren():
            yield from _iter_subtree(child)

    removed_count = 0
    for prim in _iter_subtree(root_prim):
        removed_here = False

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                UsdPhysics.RigidBodyAPI.Apply(
                    prim
                ).CreateRigidBodyEnabledAttr().Set(False)
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                removed_here = True
            except Exception as e:
                print(f"Failed to remove UsdPhysics.RigidBodyAPI from {prim.GetPath()}: {str(e)}")

        if PhysxSchema is not None and hasattr(PhysxSchema, "PhysxRigidBodyAPI"):
            try:
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
                    removed_here = True
            except Exception as e:
                print(f"Failed to remove PhysxRigidBodyAPI from {prim.GetPath()}: {str(e)}")

        if removed_here:
            removed_count += 1

    if removed_count > 0:
        print(f"Removed rigid body APIs from {removed_count} prim(s) under {prim_root_path}")


def _strip_all_physics(stage, prim_root_path: str) -> None:
    """Remove all physics APIs (rigid body + collider) from a prim subtree."""
    root_prim = stage.GetPrimAtPath(prim_root_path)
    if not root_prim or not root_prim.IsValid():
        print(f"Failed to find prim for physics cleanup: {prim_root_path}")
        return

    try:
        from pxr import PhysxSchema
    except Exception:
        PhysxSchema = None

    def _iter_subtree(prim):
        yield prim
        for child in prim.GetChildren():
            yield from _iter_subtree(child)

    removed_count = 0
    for prim in _iter_subtree(root_prim):
        removed_here = False

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                UsdPhysics.RigidBodyAPI.Apply(
                    prim
                ).CreateRigidBodyEnabledAttr().Set(False)
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                removed_here = True
            except Exception:
                pass

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            try:
                UsdPhysics.CollisionAPI.Apply(
                    prim
                ).CreateCollisionEnabledAttr().Set(False)
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
                removed_here = True
            except Exception:
                pass

        if PhysxSchema is not None and hasattr(PhysxSchema, "PhysxRigidBodyAPI"):
            try:
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
                    removed_here = True
            except Exception:
                pass

        if PhysxSchema is not None and hasattr(PhysxSchema, "PhysxCollisionAPI"):
            try:
                if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
                    removed_here = True
            except Exception:
                pass

        if removed_here:
            removed_count += 1

    if removed_count > 0:
        print(f"Removed all physics APIs from {removed_count} prim(s) under {prim_root_path}")


def _stabilize_mounted_subtree(prim_root_path: str) -> dict:
    """Make a mounted object kinematic so it remains an immovable collider."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_root_path)
    if not root_prim or not root_prim.IsValid():
        return {
            "success": False,
            "rigid_body_count": 0,
            "updated_count": 0,
            "message": f"prim not found: {prim_root_path}",
        }

    try:
        from pxr import PhysxSchema  # type: ignore
    except Exception:
        PhysxSchema = None

    result = stabilize_mounted_rigid_bodies(
        root_prim=root_prim,
        usd_rigid_body_api_type=UsdPhysics.RigidBodyAPI,
        usd_articulation_root_api_type=UsdPhysics.ArticulationRootAPI,
        physx_rigid_body_api_type=(
            PhysxSchema.PhysxRigidBodyAPI
            if PhysxSchema is not None
            and hasattr(PhysxSchema, "PhysxRigidBodyAPI")
            else None
        ),
        physx_articulation_api_type=(
            PhysxSchema.PhysxArticulationAPI
            if PhysxSchema is not None
            and hasattr(PhysxSchema, "PhysxArticulationAPI")
            else None
        ),
        sdf_bool_type=Sdf.ValueTypeNames.Bool,
    )
    if result["success"]:
        print(
            f"Stabilized {result['updated_count']} mounted rigid body prim(s) "
            f"under {prim_root_path}; disabled "
            f"{result['articulation_disabled_count']} articulation(s)"
        )
    else:
        print(
            f"Failed to stabilize mounted object under {prim_root_path}: "
            f"{result['message']}"
        )
    return result


def _set_gravity_disabled_for_subtree(prim_root_path: str) -> dict:
    """Backward-compatible alias for the stronger mounted-object stabilization."""
    return _stabilize_mounted_subtree(prim_root_path)
    
def spawn_object(object_name: str, usd_path:str, room_name: str, position: tuple[float, float, float], rotation: tuple[float, float, float]):
    '''Loads USD asset file into the scene at given position and orientation.
    '''
    stage = omni.usd.get_context().get_stage()
    prim_path_to_spawn = f"/World/rooms/{room_name}/{object_name}"
    _uniform_scale_baselines.pop(
        _uniform_scale_key(stage, prim_path_to_spawn),
        None,
    )

    # Check if prim already exists at target path
    existing_prim = stage.GetPrimAtPath(prim_path_to_spawn)
    if existing_prim and existing_prim.IsValid():
        print(f"Error: Prim already exists at path: {prim_path_to_spawn}")
        return None

    try:
        # Create new Xform and add reference to USD asset
        new_xform = UsdGeom.Xform.Define(stage, prim_path_to_spawn)
        new_xform.GetPrim().GetReferences().AddReference(usd_path)
        print(f"Spawned asset: {object_name} from {usd_path}")

        mass_result = set_base_link_mass_override(
            stage=stage,
            prim_root_path=prim_path_to_spawn,
            object_name=object_name,
            usd_path=usd_path,
        )
        if mass_result["applied"]:
            print(
                f"Set base_link mass to {mass_result['mass_kg']} kg: "
                f"{mass_result['prim_path']}"
            )
        elif mass_result["reason"] == "missing-base-link":
            print(
                "Warning: mass-override asset has no direct base_link; "
                f"mass override skipped: {mass_result['prim_path']}"
            )
        
        # Some assets contain VRay materials unsupported by Hydra in this runtime.
        _bind_pbr_materials_and_remove_vray(stage, prim_path_to_spawn)

        if is_nonphysical_floating_asset(object_name, usd_path):
            _strip_all_physics(stage, prim_path_to_spawn)
            try:
                new_xform.GetPrim().SetCustomDataByKey(
                    "nonphysical_floating_asset",
                    True,
                )
            except Exception:
                pass
            print(
                f"Configured {object_name} as a nonphysical floating asset "
                "(rigid bodies and colliders disabled)"
            )
        elif is_static_collider_asset(object_name, usd_path):
            _strip_rigid_body_keep_colliders(stage, prim_path_to_spawn)
            print(
                f"Configured {object_name} as a static collider asset "
                "(rigid bodies disabled, colliders preserved)"
            )

        # Set initial transform
        api = UsdGeom.XformCommonAPI(new_xform)
        api.SetTranslate((0, 0, 0))
        api.SetRotate(rotation)
        api.SetScale((1, 1, 1))
        
        # Get bounding box after spawning
        min_range, max_range = get_object_bbox(prim_path_to_spawn)
        
        # Calculate displacement to move object center to target position
        x_displacement = position[0] - (min_range[0] + max_range[0]) / 2
        y_displacement = position[1] - (min_range[1] + max_range[1]) / 2
        z_displacement = position[2] - (min_range[2] + max_range[2]) / 2
        
        # Apply final translation
        api.SetTranslate((x_displacement, y_displacement, z_displacement))
        
        # Save the scene
        # root_layer = stage.GetRootLayer()
        # root_layer.Save()
        #omni.physx.get_physx_interface().force_load_physics_from_usd()
        print(f"Successfully spawned {object_name} at position {position} with rotation {rotation}")
        return object_name
        
    except Exception as e:
        print(f"Error spawning object: {str(e)}")
        return None

def _legacy_set_object_pose(path:str, position:Optional[tuple[float,float,float]] = None,rotation:Optional[tuple[float,float,float]] = None):
    '''Sets the absolute position and absolute rotation of an existing object in the scene.
    
    args:
        path: prim path of the object
        position(optional): absolute target position
        rotation(optional): absolute target rotation
    '''
    stage = omni.usd.get_context().get_stage()
    if not path.startswith("/World"):
        path = "/World/" + path

    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"Failed to find object at path: {path}")
        return False
    # Provide optional parameters so position or rotation can be kept unchanged without re-fetching.
    try:
        xformable = UsdGeom.Xformable(prim)
        ordered_ops = xformable.GetOrderedXformOps()
        original_translate = Gf.Vec3f(0.0,0.0,0.0)
        original_rotate = Gf.Vec3d(0.0,0.0,0.0)
        original_scale = Gf.Vec3f(1.0,1.0,1.0)
        for op in ordered_ops:
            print(f"Processing op {op.GetName()} of type {op.GetOpType()} for object at path: {path}")
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                original_translate = op.Get()
                if position is not None:
                    # Move the displacement calculation into set_pose; resolve_intent outputs the target centroid
                    min_range, max_range = get_object_bbox(path)
                    displacement = _build_position_result(position, min_range,max_range)
                    displacement = Gf.Vec3f(displacement[0], displacement[1], displacement[2])
                    target_pos = Gf.Vec3f(original_translate[0]+displacement[0], original_translate[1]+displacement[1], original_translate[2]+displacement[2])
                    op.Set(target_pos)
                    print(f"Set position for object at path: {path} to {target_pos}")
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                print(f"Original rotation for object at path: {path} is {op.Get()}")
                original_rotate = op.Get()
                if rotation is not None:
                    target_rot = Gf.Vec3d(rotation[0], rotation[1], rotation[2])
                    rot_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0), target_rot[0]) * Gf.Rotation(Gf.Vec3d(0,1,0), target_rot[1]) * Gf.Rotation(Gf.Vec3d(0,0,1), target_rot[2]))
                    quat = rot_matrix.ExtractRotation().GetQuat()
                    op.Set(quat)
                    print(f"Set rotation for object at path: {path} to {target_rot}")
        # Get the original transform, clear and reset to prevent incremental transform errors (must be Scale -> Rotate -> Translate)
        # xformable.ClearXformOps()
        # # Re-add scale op
        # scale_op = xformable.AddScaleOp()
        # scale_op.Set(original_scale)
        # # Re-add rotate op
        # rotate_op = xformable.AddRotateXYZOp()
        # if rotation is not None:
        #     rotate_op.Set(Gf.Vec3f(original_rotate[0]+rotation[0], original_rotate[1]+rotation[1], original_rotate[2]+rotation[2]))
        # else:
        #     rotate_op.Set(original_rotate)
        # # Re-add translate op
        # translate_op = xformable.AddTranslateOp()
        # if position is not None:
        #     translate_op.Set(Gf.Vec3f(original_translate[0]+position[0], original_translate[1]+position[1], original_translate[2]+position[2]))
        # else:
        #     translate_op.Set(original_translate)
        # Use Isaac Sim API here
        print(f"Successfully set pose for object at path: {path}")
        dt = 1.0/60.0
        current_time = time.time()
        omni.physx.get_physx_interface().update_simulation(dt, current_time)
        return True

    except Exception as e:
        print(f"Error setting object pose: {str(e)}")
        return False 
    

def _rotation_ops(xformable):
    supported_types = {
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeOrient,
    }
    return [
        op
        for op in xformable.GetOrderedXformOps()
        if op.GetOpType() in supported_types
    ]


def _set_absolute_rotation(xformable, rotation) -> None:
    target_rotation = (
        float(rotation[0]),
        float(rotation[1]),
        float(rotation[2]),
    )
    rotation_ops = _rotation_ops(xformable)

    rotate_xyz_op = next(
        (
            op
            for op in rotation_ops
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ
        ),
        None,
    )
    if rotate_xyz_op is not None:
        rotate_xyz_op.Set(Gf.Vec3f(*target_rotation))
        return

    orient_op = next(
        (
            op
            for op in rotation_ops
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient
        ),
        None,
    )
    if orient_op is not None:
        rotation_matrix = Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), target_rotation[0])
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), target_rotation[1])
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), target_rotation[2])
        )
        orient_op.Set(rotation_matrix.ExtractRotation().GetQuat())
        return

    UsdGeom.XformCommonAPI(xformable.GetPrim()).SetRotate(
        Gf.Vec3f(*target_rotation)
    )


def _get_translate_op(xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


def _measure_bbox_half_sizes_at_rotation(path: str, rotation):
    bbox = get_object_bbox(path)
    if bbox is None:
        return None
    if rotation is None:
        min_range, max_range = bbox
        return [
            (float(max_range[i]) - float(min_range[i])) / 2.0
            for i in range(3)
        ]

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None

    xformable = UsdGeom.Xformable(prim)
    original_rotation_values = [
        (op, op.Get())
        for op in _rotation_ops(xformable)
    ]

    try:
        _set_absolute_rotation(xformable, rotation)
        rotated_bbox = get_object_bbox(path)
        if rotated_bbox is None:
            return None
        min_range, max_range = rotated_bbox
        return [
            (float(max_range[i]) - float(min_range[i])) / 2.0
            for i in range(3)
        ]
    finally:
        if original_rotation_values:
            for op, value in original_rotation_values:
                op.Set(value)
        else:
            _set_absolute_rotation(xformable, (0.0, 0.0, 0.0))


def set_object_pose(path:str, position:Optional[tuple[float,float,float]] = None,rotation:Optional[tuple[float,float,float]] = None):
    '''Set an object's absolute world-space bounding-box center and rotation.

    args:
        path: prim path of the object
        position(optional): absolute target world-space bounding-box center
        rotation(optional): absolute Euler XYZ rotation in degrees
    '''
    stage = omni.usd.get_context().get_stage()
    if not path.startswith("/World"):
        path = "/World/" + path

    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"Failed to find object at path: {path}")
        return False

    try:
        xformable = UsdGeom.Xformable(prim)
        target_center = position
        if rotation is not None and position is None:
            original_bbox = get_object_bbox(path)
            if original_bbox is None:
                print(f"Failed to get bounding box for object at path: {path}")
                return False
            original_min, original_max = original_bbox
            target_center = tuple(
                (float(original_min[i]) + float(original_max[i])) / 2.0
                for i in range(3)
            )

        # Rotation changes the world-aligned bbox. Apply it before translating
        # the bbox center to the requested absolute world coordinate.
        if rotation is not None:
            _set_absolute_rotation(xformable, rotation)
            print(f"Set rotation for object at path: {path} to {rotation}")

        if target_center is not None:
            bbox = get_object_bbox(path)
            if bbox is None:
                print(f"Failed to get bounding box for object at path: {path}")
                return False
            min_range, max_range = bbox
            current_center = [
                (float(min_range[i]) + float(max_range[i])) / 2.0
                for i in range(3)
            ]
            translate_op = _get_translate_op(xformable)
            current_translate = translate_op.Get()
            if current_translate is None:
                current_translate = Gf.Vec3f(0.0, 0.0, 0.0)
            displacement = [
                float(target_center[i]) - current_center[i]
                for i in range(3)
            ]
            target_translate = Gf.Vec3f(
                float(current_translate[0]) + displacement[0],
                float(current_translate[1]) + displacement[1],
                float(current_translate[2]) + displacement[2],
            )
            translate_op.Set(target_translate)
            print(
                f"Set world bbox center for object at path: {path} "
                f"to {target_center} using translate {target_translate}"
            )

        print(f"Successfully set pose for object at path: {path}")
        dt = 1.0 / 60.0
        current_time = time.time()
        omni.physx.get_physx_interface().update_simulation(dt, current_time)
        return True
    except Exception as e:
        print(f"Error setting object pose: {str(e)}")
        return False


def _surface_plane_config(surface: str):
    if surface in ["down", "up"]:
        return {"plane_axes": (0, 1), "normal_axis": 2, "normal_sign": 1 if surface == "up" else -1}
    if surface in ["left", "right"]:
        return {"plane_axes": (1, 2), "normal_axis": 0, "normal_sign": 1 if surface == "right" else -1}
    if surface in ["back", "front"]:
        return {"plane_axes": (0, 2), "normal_axis": 1, "normal_sign": 1 if surface == "front" else -1}
    return None


def _apply_surface_placement_clearance(world_center, surface: str):
    """Move a surface placement slightly outward to avoid interpenetration."""
    plane_cfg = _surface_plane_config(surface)
    if plane_cfg is None:
        return world_center

    normal_axis = plane_cfg["normal_axis"]
    normal_sign = plane_cfg["normal_sign"]
    world_center[normal_axis] += normal_sign * _SURFACE_PLACEMENT_CLEARANCE
    return world_center


def _build_position_result(world_center, obj_min_range, obj_max_range, message=""):
    print(f"Building position result with world_center: {world_center}, obj_min_range: {obj_min_range}, obj_max_range: {obj_max_range}, message: '{message}'")
    body_center = (
        (obj_max_range[0] + obj_min_range[0]) / 2,
        (obj_max_range[1] + obj_min_range[1]) / 2,
        (obj_max_range[2] + obj_min_range[2]) / 2,
    )
    return (
            world_center[0] - body_center[0],
            world_center[1] - body_center[1],
            world_center[2] - body_center[2],
        )


def _placement_failure(message: str, rotation=None, candidate_position=None, out_of_bounds=None, **details):
    """Build a standardised failure result for resolve_placement_intent.

    Args:
        message: Human-readable diagnostic explaining why placement failed.
        rotation: The resolved rotation (if any) for the failed attempt.
        candidate_position: The computed position that failed validation.
            When provided the Agent can see *where* the attempt landed and
            derive a corrective offset instead of guessing from scratch.
        out_of_bounds: Axis-wise overflow *in metres*.  Positive values mean
            the object overflows in the +axis direction, negative values in
            the -axis direction, ``0.0`` means within bounds on that axis.
            If *None*, defaults to ``{"x": 0.0, "y": 0.0, "z": 0.0}``.
        **details: Extra keys merged into the result dict (e.g. ``collisions``,
            ``room_bounds``, ``support_info``).
    """
    result = {
        "success": False,
        "position": candidate_position,
        "rotation": rotation,
        "out_of_bounds": out_of_bounds if out_of_bounds is not None else {"x": 0.0, "y": 0.0, "z": 0.0},
        "message": message,
    }
    result.update(details)
    return result


def _parse_direction_spec(intent: dict, field_name: str):
    if field_name not in intent:
        raise ValueError(f"missing required field '{field_name}'")
    value = intent[field_name]
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            f"{field_name} must be a (reference_object, distance) pair"
        )
    reference, distance = value
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(
            f"{field_name} reference_object must be a non-empty string"
        )
    try:
        numeric_distance = float(distance)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} distance must be numeric") from exc
    if not math.isfinite(numeric_distance):
        raise ValueError(f"{field_name} distance must be finite")
    return reference, numeric_distance


def _resolve_orientation_rotation(intent: dict):
    """Resolve optional intent orientation to Euler XYZ degrees.

    Supported values:
    - orientation as string: "+x", "-x", "+y", "-y" (also accepts "x+", "x-", "y+", "y-")
    - orientation as dict: {"front_facing": "+y"} or {"yaw_degrees": 90}
    - orientation as tuple/list: (rx, ry, rz)
    """
    orientation = intent.get("orientation")
    if orientation is None:
        return True, None, ""

    if isinstance(orientation, (tuple, list)):
        if len(orientation) != 3:
            return False, None, "orientation tuple/list must contain exactly 3 values"
        try:
            return True, (float(orientation[0]), float(orientation[1]), float(orientation[2])), ""
        except (TypeError, ValueError):
            return False, None, "orientation tuple/list must be numeric"

    if isinstance(orientation, (int, float)):
        return True, (0.0, 0.0, float(orientation)), ""

    yaw = None
    if isinstance(orientation, str):
        facing = orientation.strip().lower().replace(" ", "")
    elif isinstance(orientation, dict):
        if "yaw_degrees" in orientation:
            try:
                yaw = float(orientation["yaw_degrees"])
            except (TypeError, ValueError):
                return False, None, "orientation.yaw_degrees must be numeric"
            return True, (0.0, 0.0, yaw), ""

        facing = (
            orientation.get("front_facing")
            or orientation.get("front_direction")
            or orientation.get("direction")
            or orientation.get("axis")
        )
        if not isinstance(facing, str):
            return False, None, "orientation dict must include front_facing/front_direction/direction/axis or yaw_degrees"
        facing = facing.strip().lower().replace(" ", "")
    else:
        return False, None, "orientation must be string, dict, number, or 3-value tuple/list"

    facing_to_yaw = {
        "+x": 0.0,
        "x+": 0.0,
        "east": 0.0,
        "-x": 180.0,
        "x-": 180.0,
        "west": 180.0,
        "+y": 90.0,
        "y+": 90.0,
        "front": 90.0,
        "north": 90.0,
        "-y": -90.0,
        "y-": -90.0,
        "back": -90.0,
        "south": -90.0,
    }
    if facing not in facing_to_yaw:
        return False, None, "unsupported orientation. Use +x/-x/+y/-y, a yaw_degrees value, or a full (rx, ry, rz) tuple"

    yaw = facing_to_yaw[facing]
    yaw = ((yaw + 180.0) % 360.0) - 180.0
    return True, (0.0, 0.0, yaw), ""


def _check_surface_bounds(world_center, half_sizes, min_pos, max_pos, surface: str):
    target_min_box = [world_center[0] - half_sizes[0], world_center[1] - half_sizes[1], world_center[2] - half_sizes[2]]
    target_max_box = [world_center[0] + half_sizes[0], world_center[1] + half_sizes[1], world_center[2] + half_sizes[2]]
    out_of_bounds = {"x": 0.0, "y": 0.0, "z": 0.0}

    if surface in ["down", "up"]:
        axes = (0, 1)
        names = ("x", "y")
    elif surface in ["left", "right"]:
        axes = (1, 2)
        names = ("y", "z")
    else:
        axes = (0, 2)
        names = ("x", "z")

    success = True
    for local_i, axis in enumerate(axes):
        if target_min_box[axis] < min_pos[axis] and target_max_box[axis] > max_pos[axis]:
            out_of_bounds[names[local_i]] = -1.0
            success = False
        elif target_min_box[axis] < min_pos[axis]:
            out_of_bounds[names[local_i]] = target_min_box[axis] - min_pos[axis]
            success = False
        elif target_max_box[axis] > max_pos[axis]:
            out_of_bounds[names[local_i]] = target_max_box[axis] - max_pos[axis]
            success = False

    return success, out_of_bounds


def _compute_semantic_surface_center(semantic_location: dict, surface: str, support_min, support_max, obj_half_sizes):
    plane_cfg = _surface_plane_config(surface)
    if plane_cfg is None:
        return None

    plane_axis_0, plane_axis_1 = plane_cfg["plane_axes"]
    normal_axis = plane_cfg["normal_axis"]
    normal_sign = plane_cfg["normal_sign"]
    margin = float(semantic_location.get("margin", 0.03))

    usable_min_0 = support_min[plane_axis_0] + obj_half_sizes[plane_axis_0] + margin
    usable_max_0 = support_max[plane_axis_0] - obj_half_sizes[plane_axis_0] - margin
    usable_min_1 = support_min[plane_axis_1] + obj_half_sizes[plane_axis_1] + margin
    usable_max_1 = support_max[plane_axis_1] - obj_half_sizes[plane_axis_1] - margin

    if usable_min_0 > usable_max_0 or usable_min_1 > usable_max_1:
        # Provide concrete size info so the Agent can decide whether to
        # scale the object, pick a different support, or adjust margins.
        usable_len_0 = max(0.0, usable_max_0 - usable_min_0)
        usable_len_1 = max(0.0, usable_max_1 - usable_min_1)
        support_len_0 = float(support_max[plane_axis_0]) - float(support_min[plane_axis_0])
        support_len_1 = float(support_max[plane_axis_1]) - float(support_min[plane_axis_1])
        return {
            "success": False,
            "position": None,
            "out_of_bounds": {"x": 0.0, "y": 0.0, "z": 0.0},
            "message": "object does not fit on support surface",
            "support_info": {
                "support_size_on_plane": (round(support_len_0, 4), round(support_len_1, 4)),
                "object_size_on_plane": (round(float(obj_half_sizes[plane_axis_0]) * 2, 4), round(float(obj_half_sizes[plane_axis_1]) * 2, 4)),
                "usable_area_on_plane": (round(usable_len_0, 4), round(usable_len_1, 4)),
                "margin": margin,
            },
        }

    def _clamp(v, low, high):
        return max(low, min(v, high))

    def _anchor_value(choice: str, low: float, high: float):
        if choice == "negative":
            return low
        if choice == "positive":
            return high
        return (low + high) / 2.0

    location = str(semantic_location.get("location", "center")).lower().replace("-", "_")
    axis0_choice = "center"
    axis1_choice = "center"

    if surface in ["down", "up"]:
        if "left" in location:
            axis0_choice = "negative"
        elif "right" in location:
            axis0_choice = "positive"
        if "back" in location:
            axis1_choice = "negative"
        elif "front" in location:
            axis1_choice = "positive"
    elif surface in ["left", "right"]:
        if "back" in location:
            axis0_choice = "negative"
        elif "front" in location:
            axis0_choice = "positive"
        if "down" in location:
            axis1_choice = "negative"
        elif "up" in location:
            axis1_choice = "positive"
    elif surface in ["back", "front"]:
        if "left" in location:
            axis0_choice = "negative"
        elif "right" in location:
            axis0_choice = "positive"
        if "down" in location:
            axis1_choice = "negative"
        elif "up" in location:
            axis1_choice = "positive"

    world_center = [
        (support_min[0] + support_max[0]) / 2.0,
        (support_min[1] + support_max[1]) / 2.0,
        (support_min[2] + support_max[2]) / 2.0,
    ]
    world_center[plane_axis_0] = _anchor_value(axis0_choice, usable_min_0, usable_max_0)
    world_center[plane_axis_1] = _anchor_value(axis1_choice, usable_min_1, usable_max_1)
    world_center[normal_axis] = support_max[normal_axis] + obj_half_sizes[normal_axis] if normal_sign > 0 else support_min[normal_axis] - obj_half_sizes[normal_axis]
    _apply_surface_placement_clearance(world_center, surface)

    reference_object = semantic_location.get("reference_object") or semantic_location.get("relative_to")
    relation = str(semantic_location.get("relation", "")).lower()
    direction = str(semantic_location.get("direction", "")).lower()
    distance = float(semantic_location.get("distance", 0.03))

    if reference_object:
        ref_min, ref_max = get_object_bbox(reference_object)
        ref_center = [(ref_min[i] + ref_max[i]) / 2.0 for i in range(3)]
        ref_half_sizes = [
            (ref_max[0] - ref_min[0]) / 2.0,
            (ref_max[1] - ref_min[1]) / 2.0,
            (ref_max[2] - ref_min[2]) / 2.0,
        ]
        world_center[plane_axis_0] = ref_center[plane_axis_0]
        world_center[plane_axis_1] = ref_center[plane_axis_1]

        if relation == "next_to":
            if direction == "left":
                relation = "left_of"
            elif direction == "right":
                relation = "right_of"
            elif direction == "front":
                relation = "in_front_of"
            elif direction == "back":
                relation = "behind"
            elif direction == "up":
                relation = "above"
            elif direction == "down":
                relation = "below"

        if relation == "left_of":
            world_center[plane_axis_0] = ref_center[plane_axis_0] - (ref_half_sizes[plane_axis_0] + obj_half_sizes[plane_axis_0] + distance)
        elif relation == "right_of":
            world_center[plane_axis_0] = ref_center[plane_axis_0] + (ref_half_sizes[plane_axis_0] + obj_half_sizes[plane_axis_0] + distance)
        elif relation == "in_front_of":
            world_center[plane_axis_1] = ref_center[plane_axis_1] + (ref_half_sizes[plane_axis_1] + obj_half_sizes[plane_axis_1] + distance)
        elif relation == "behind":
            world_center[plane_axis_1] = ref_center[plane_axis_1] - (ref_half_sizes[plane_axis_1] + obj_half_sizes[plane_axis_1] + distance)
        elif relation == "above":
            world_center[plane_axis_1] = ref_center[plane_axis_1] + (ref_half_sizes[plane_axis_1] + obj_half_sizes[plane_axis_1] + distance)
        elif relation == "below":
            world_center[plane_axis_1] = ref_center[plane_axis_1] - (ref_half_sizes[plane_axis_1] + obj_half_sizes[plane_axis_1] + distance)
        relation_axis = None
        if relation in ["left_of", "right_of"]:
            relation_axis = plane_axis_0
            lower_bound = usable_min_0
            upper_bound = usable_max_0
            ref_lower = ref_min[plane_axis_0]
            ref_upper = ref_max[plane_axis_0]
        elif relation in ["in_front_of", "behind", "above", "below"]:
            relation_axis = plane_axis_1
            lower_bound = usable_min_1
            upper_bound = usable_max_1
            ref_lower = ref_min[plane_axis_1]
            ref_upper = ref_max[plane_axis_1]
        else:
            world_center[plane_axis_0] = _clamp(world_center[plane_axis_0], usable_min_0, usable_max_0)
            world_center[plane_axis_1] = _clamp(world_center[plane_axis_1], usable_min_1, usable_max_1)

        if relation_axis is not None:
            if relation in ["left_of", "behind", "below"]:
                required_max = ref_lower - obj_half_sizes[relation_axis] - distance
                if required_max < lower_bound:
                    shortfall = lower_bound - required_max
                    return {
                        "success": False,
                        "position": None,
                        "out_of_bounds": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "message": f"no feasible {relation} placement on support surface",
                        "support_info": {
                            "available_space_in_direction": round(float(required_max - lower_bound), 4),
                            "space_needed": round(float(obj_half_sizes[relation_axis] * 2 + distance), 4),
                            "shortfall": round(float(shortfall), 4),
                            "reference_object_edge": round(float(ref_lower), 4),
                            "support_boundary": round(float(lower_bound), 4),
                        },
                    }
                clamped_value = min(world_center[relation_axis], upper_bound, required_max)
                world_center[relation_axis] = max(lower_bound, clamped_value)
            else:
                required_min = ref_upper + obj_half_sizes[relation_axis] + distance
                if required_min > upper_bound:
                    shortfall = required_min - upper_bound
                    return {
                        "success": False,
                        "position": None,
                        "out_of_bounds": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "message": f"no feasible {relation} placement on support surface",
                        "support_info": {
                            "available_space_in_direction": round(float(upper_bound - required_min), 4),
                            "space_needed": round(float(obj_half_sizes[relation_axis] * 2 + distance), 4),
                            "shortfall": round(float(shortfall), 4),
                            "reference_object_edge": round(float(ref_upper), 4),
                            "support_boundary": round(float(upper_bound), 4),
                        },
                    }
                clamped_value = max(world_center[relation_axis], lower_bound, required_min)
                world_center[relation_axis] = min(upper_bound, clamped_value)

            other_axis = plane_axis_1 if relation_axis == plane_axis_0 else plane_axis_0
            if other_axis == plane_axis_0:
                world_center[other_axis] = _clamp(world_center[other_axis], usable_min_0, usable_max_0)
            else:
                world_center[other_axis] = _clamp(world_center[other_axis], usable_min_1, usable_max_1)

    success, out_of_bounds = _check_surface_bounds(world_center, obj_half_sizes, support_min, support_max, surface)
    print(f"Computed world center: {world_center}, out_of_bounds: {out_of_bounds}, success: {success}")
    if not success:
        return {
            "success": False,
            "position": None,
            "out_of_bounds": out_of_bounds,
            "message": "semantic surface placement exceeds support bounds",
            "support_info": {
                "candidate_position": tuple(round(float(v), 4) for v in world_center),
                "support_min": tuple(round(float(v), 4) for v in support_min),
                "support_max": tuple(round(float(v), 4) for v in support_max),
                "object_half_sizes": tuple(round(float(v), 4) for v in obj_half_sizes),
                "surface": surface,
            },
        }

    return {"success": True, "position": tuple(world_center), "out_of_bounds": {"x": 0.0, "y": 0.0, "z": 0.0}, "message": ""}


def _compute_synthetic_ceiling_bounds(room_name: str):
    """Compute synthetic ceiling bounds from room walls when no ceiling prim exists.

    Returns (support_min, support_max, room_center) as a 3-tuple of tuples,
    or None if wall data is unavailable.
    """
    if not room_name:
        return None
    raw = scan_scene(room_name)
    if not raw:
        return None

    walls = [obj for obj in raw if _classify_prim_name(obj['name']) == 'wall']
    if not walls:
        return None

    room_x_min = min(float(w['bbox']['min'][0]) for w in walls)
    room_x_max = max(float(w['bbox']['max'][0]) for w in walls)
    room_y_min = min(float(w['bbox']['min'][1]) for w in walls)
    room_y_max = max(float(w['bbox']['max'][1]) for w in walls)
    ceiling_z = max(float(w['bbox']['max'][2]) for w in walls)
    thickness = 0.1

    support_min = (room_x_min, room_y_min, ceiling_z)
    support_max = (room_x_max, room_y_max, ceiling_z + thickness)
    room_center = ((room_x_min + room_x_max) / 2.0, (room_y_min + room_y_max) / 2.0)

    return support_min, support_max, room_center


def _get_room_wall_prim_paths(room_name: str):
    """Collect prim paths of all wall objects in the room for collision whitelisting."""
    if not room_name:
        return []
    raw = scan_scene(room_name)
    if not raw:
        return []

    wall_paths = []
    for obj in raw:
        if _classify_prim_name(obj['name']) == 'wall':
            wall_paths.append(f"/World/rooms/{room_name}/{obj['name']}")
    return wall_paths


def _collision_check(target_prim_path: str,target_position: Optional[tuple[float,float,float]] = None,target_rotation : Optional[tuple[float,float,float]] = None,white_list: Optional[list[str]] = None,*,restore_on_success: bool = True):
    """Check if placing the object would cause collision with existing objects.

    Returns a list of collision-detail dicts, one per colliding object.
    Each dict contains:
        - ``object`` (str): prim path of the colliding scene object
        - ``overlap`` (tuple[float, float, float]): per-axis AABB overlap in
          metres ``(dx, dy, dz)`` — all non-negative; zero on axes without
          overlap.  The Agent can use this to derive a corrective offset.
    An empty list means no collision detected.

    Args:
        restore_on_success: If True (default), the object is restored to its
            original pose after the check regardless of outcome.  If False and
            no collision is detected, the object is *kept* at the target pose so
            that subsequent ``_collision_check`` calls can detect it.  When a
            collision *is* detected the object is always restored.
    """
    target_object_name = target_prim_path.rstrip("/").rsplit("/", 1)[-1]
    stage = omni.usd.get_context().get_stage()
    target_root_prim = stage.GetPrimAtPath(target_prim_path)
    ignore_target_collision = (
        is_collision_evaluation_exempt_asset(target_object_name)
        or is_nonphysical_floating_asset(target_object_name)
        or is_nonphysical_floating_prim(target_root_prim)
    )
    path_encoded = None
    if not ignore_target_collision:
        path_encoded = PhysicsSchemaTools.encodeSdfPath(
            Sdf.Path(target_prim_path + "/base_link/visuals")
        )
    overlapping_paths = []
    has_any_collision = False
    # Tolerance for wall penetration, to accommodate wall-mounted placement
    wall_contact_tolerance = 0.08

    def _get_bbox_overlap_amounts(bbox_a, bbox_b):
        min_a, max_a = bbox_a
        min_b, max_b = bbox_b
        return [
            max(0.0, min(float(max_a[i]), float(max_b[i])) - max(float(min_a[i]), float(min_b[i])))
            for i in range(3)
        ]

    def _is_tolerable_wall_contact(hit_path: str, target_bbox) -> bool:
        if "wall" not in hit_path:
            return False
        wall_bbox = get_object_bbox(hit_path)
        if wall_bbox is None:
            return False
        overlap_amounts = _get_bbox_overlap_amounts(target_bbox, wall_bbox)
        if max(overlap_amounts) <= 0.0:
            return False
        wall_extents = [float(wall_bbox[1][i]) - float(wall_bbox[0][i]) for i in range(3)]
        wall_thickness_axis = min(range(3), key=lambda i: wall_extents[i])
        return overlap_amounts[wall_thickness_axis] <= wall_contact_tolerance

    def report_hit(hit):
        nonlocal has_any_collision
        hit_path = str(hit.rigid_body)
        hit_object_root = hit_path.split("/base_link", 1)[0]
        hit_object_name = hit_object_root.rstrip("/").rsplit("/", 1)[-1]
        hit_root_prim = stage.GetPrimAtPath(hit_object_root)
        if (
            is_collision_evaluation_exempt_asset(hit_object_name)
            or is_nonphysical_floating_asset(hit_object_name)
            or is_nonphysical_floating_prim(hit_root_prim)
        ):
            return True
        # Exclude self, otherwise it always returns True
        if hit.rigid_body != target_prim_path and not str(hit.rigid_body).startswith(target_prim_path + "/"):
            # Check if the collision object is already in the list, or if a white-listed prim path is part of this prim path
            if hit.rigid_body not in overlapping_paths and not _is_tolerable_wall_contact(str(hit.rigid_body), moved_bbox) and (white_list is None or not any(white_path in hit.rigid_body for white_path in white_list)):
                overlapping_paths.append(hit.rigid_body)
        return True
    # Run collision detection
    # Calculate current centroid position
    min_bbox,max_bbox = get_object_bbox(target_prim_path)
    current_center = ((min_bbox[0]+max_bbox[0])/2,(min_bbox[1]+max_bbox[1])/2,(min_bbox[2]+max_bbox[2])/2)
    # Get initial orientation
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(target_prim_path)
    xformable = UsdGeom.Xformable(prim)
    original_rotation_values = [
        (op, op.Get())
        for op in _rotation_ops(xformable)
    ]

    set_object_pose(target_prim_path, position = target_position, rotation = target_rotation)
    moved_bbox = get_object_bbox(target_prim_path)
    print(f"Performing collision check for {target_prim_path} against existing scene objects...")
    if not ignore_target_collision:
        physx_query_interface = omni.physx.get_physx_scene_query_interface()
        physx_query_interface.overlap_mesh(
            path_encoded[0],
            path_encoded[1],
            report_hit,
            False,
        )
    else:
        print(
            f"Skipping collision queries for collision-exempt asset: "
            f"{target_prim_path}"
        )
    print(f"Collision check for {target_prim_path} found PhysX overlaps with: {overlapping_paths}")

    # --- AABB overlap cross-check ---
    # PhysX mesh overlap can miss collisions when collision geometry is
    # simplified (e.g. furniture with coarse convex decompositions).  As a
    # safety net, also scan the scene and flag any non-structural object whose
    # world AABB significantly overlaps the target's world AABB.
    _aab_structural_prefixes = ("wall", "floor", "ceiling", "door", "groundplane")
    _target_room = _extract_room_name_from_prim_path(target_prim_path)
    scene_objects = scan_scene(_target_room) if _target_room else None
    if scene_objects and moved_bbox is not None and not ignore_target_collision:
        target_min, target_max = moved_bbox
        for obj in scene_objects:
            if not isinstance(obj, dict):
                continue
            obj_name = str(obj.get("name", ""))
            if not obj_name:
                continue
            if (
                is_collision_evaluation_exempt_asset(obj_name)
                or is_nonphysical_floating_asset(obj_name)
            ):
                continue
            obj_lower = obj_name.lower()
            # Skip structural elements (walls, floors, ceilings, doors)
            if any(obj_lower.startswith(p) for p in _aab_structural_prefixes):
                continue
            # Skip self
            if obj_name == target_prim_path.rsplit("/", 1)[-1]:
                continue
            # Skip objects already caught by PhysX
            if any(obj_name in p for p in overlapping_paths):
                continue
            # White-list check (same logic as report_hit)
            obj_prim_path = None
            for parent_path_candidate in [target_prim_path.rsplit("/", 1)[0] + "/" + obj_name, "/World/" + obj_name]:
                if omni.usd.get_context().get_stage().GetPrimAtPath(parent_path_candidate):
                    obj_prim_path = parent_path_candidate
                    break
            if obj_prim_path and is_nonphysical_floating_prim(
                stage.GetPrimAtPath(obj_prim_path)
            ):
                continue
            if obj_prim_path and (white_list is not None and any(white_path in obj_prim_path for white_path in white_list)):
                continue
            obj_bbox = obj.get("bbox")
            if not obj_bbox or not isinstance(obj_bbox, dict):
                continue
            obj_min_raw = obj_bbox.get("min")
            obj_max_raw = obj_bbox.get("max")
            if obj_min_raw is None or obj_max_raw is None:
                continue
            try:
                obj_min = [float(obj_min_raw[i]) for i in range(3)]
                obj_max = [float(obj_max_raw[i]) for i in range(3)]
                target_min_f = [float(target_min[i]) for i in range(3)]
                target_max_f = [float(target_max[i]) for i in range(3)]
            except (TypeError, IndexError, ValueError):
                continue
            overlap = [
                max(0.0, min(target_max_f[i], obj_max[i]) - max(target_min_f[i], obj_min[i]))
                for i in range(3)
            ]
            # All three axes must have positive overlap for AABB intersection
            if all(v > 0.0 for v in overlap):
                # Use wall-contact tolerance for objects that are walls
                if _is_tolerable_wall_contact(obj_prim_path or obj_name, moved_bbox):
                    continue
                # Significant overlap: require at least 1 cm on all three axes
                # to avoid false positives from touching surfaces
                if min(overlap) < 0.01:
                    continue
                resolved_path = obj_prim_path or obj_name
                if resolved_path not in overlapping_paths:
                    overlapping_paths.append(resolved_path)
                    print(f"  [AABB cross-check] detected overlap with {resolved_path}: "
                          f"overlap=({overlap[0]:.4f}, {overlap[1]:.4f}, {overlap[2]:.4f})")

    print(f"Collision check for {target_prim_path} found total overlaps with: {overlapping_paths}")
    # Build detailed collision info with per-axis AABB overlap amounts
    collision_details = []
    for hit_path in overlapping_paths:
        hit_bbox = get_object_bbox(hit_path)
        if hit_bbox is not None:
            overlap = _get_bbox_overlap_amounts(moved_bbox, hit_bbox)
        else:
            overlap = (0.0, 0.0, 0.0)
        collision_details.append({
            "object": hit_path,
            "overlap": tuple(round(float(v), 4) for v in overlap),
        })
    # Restore the object to its original pose when:
    #   - a collision was detected (always restore so the object doesn't stay
    #     in a colliding position), or
    #   - restore_on_success is True (the default, legacy behaviour).
    # When restore_on_success is False and there is no collision, the object
    # stays at the target pose so that subsequent resolve calls can detect it.
    if collision_details or restore_on_success:
        for op, value in original_rotation_values:
            op.Set(value)
        if not original_rotation_values and target_rotation is not None:
            _set_absolute_rotation(xformable, (0.0, 0.0, 0.0))
        set_object_pose(target_prim_path, position=current_center, rotation=None)
    return collision_details

def resolve_placement_intent(intent:dict):
    '''Converts a high-level semantic placement intent into concrete spatial coordinates.'''
    if not isinstance(intent, dict):
        return _placement_failure("intent must be a dictionary")
    if "prim_path" not in intent:
        return _placement_failure("missing required field 'prim_path'")
    if "placement_type" not in intent:
        return _placement_failure("missing required field 'placement_type'")

    placement_type = str(intent["placement_type"]).strip().lower()
    if placement_type in {"wall-mounted", "wall mounted", "mounted"}:
        placement_type = "wall_mounted"
    orientation_ok, resolved_rotation, orientation_msg = _resolve_orientation_rotation(intent)
    if not orientation_ok:
        return _placement_failure(orientation_msg)

    obj_half_sizes = _measure_bbox_half_sizes_at_rotation(
        intent["prim_path"],
        resolved_rotation,
    )
    if obj_half_sizes is None:
        return _placement_failure(
            f"could not get bounding box for object {intent['prim_path']}",
            rotation=resolved_rotation,
        )
    x_length, y_length, z_length = obj_half_sizes

    if placement_type == "floor":
        try:
            room_name = _extract_room_name_from_prim_path(intent["prim_path"])
            x_path, x_distance = _parse_direction_spec(intent, "x_direction")
            y_path, y_distance = _parse_direction_spec(intent, "y_direction")
            z_path, z_distance = _parse_direction_spec(intent, "z_direction")

            x_bbox = get_object_bbox(x_path)
            y_bbox = get_object_bbox(y_path)
            z_bbox = get_object_bbox(z_path)
            if x_bbox is None:
                raise ValueError(f"x_direction reference not found: {x_path}")
            if y_bbox is None:
                raise ValueError(f"y_direction reference not found: {y_path}")
            if z_bbox is None:
                raise ValueError(f"z_direction reference not found: {z_path}")

            x_min, x_max = x_bbox
            y_min, y_max = y_bbox
            z_min, z_max = z_bbox
            x_ref = (
                x_min[0] + x_distance - x_length
                if x_distance <= 0
                else x_max[0] + x_distance + x_length
            )
            y_ref = (
                y_min[1] + y_distance - y_length
                if y_distance <= 0
                else y_max[1] + y_distance + y_length
            )
            if "GroundPlane" in z_path:
                # Preserve the existing GroundPlane contract: rest the object's
                # rotated world AABB on the ground.
                z_ref = z_length
            else:
                z_ref = (
                    z_min[2] + z_distance - z_length
                    if z_distance <= 0
                    else z_max[2] + z_distance + z_length
                )
            result = {
                "success": True,
                "position": (float(x_ref), float(y_ref), float(z_ref)),
                "rotation": resolved_rotation,
                "out_of_bounds": {"x": 0.0, "y": 0.0, "z": 0.0},
                "message": "",
            }
            # Collision checking temporarily moves the object to the candidate
            # pose.  Always restore afterwards: resolve_placement_intent should
            # be a pure coordinate proposal; only set_pose is allowed to make a
            # final stage mutation.  This prevents an uncommitted successful
            # resolve from being saved accidentally.
            temp_prim_path = intent["prim_path"]
            temp_position = result["position"]
            print(f"Checking for collisions at proposed floor placement: {temp_position} with rotation {resolved_rotation}")
            # Save the pre-collision pose to restore if subsequent checks fail
            _pre_collision_bbox = get_object_bbox(temp_prim_path)
            _pre_collision_center = ((float(_pre_collision_bbox[0][0])+float(_pre_collision_bbox[1][0]))/2,
                                     (float(_pre_collision_bbox[0][1])+float(_pre_collision_bbox[1][1]))/2,
                                     (float(_pre_collision_bbox[0][2])+float(_pre_collision_bbox[1][2]))/2)
            _pre_collision_prim = omni.usd.get_context().get_stage().GetPrimAtPath(temp_prim_path)
            _pre_collision_xformable = UsdGeom.Xformable(_pre_collision_prim)
            _pre_collision_rotation_values = [
                (op, op.Get())
                for op in _rotation_ops(_pre_collision_xformable)
            ]
            # Hard-coded white list for floor placement: allows collision with "wall", "floor", "ceiling", "GroundPlane"
            # TODO: when a floor object itself serves as a support for other objects, collisions will be detected,
            # so the corresponding surface object should also be in the white list.
            white_list = ["floor", "ceiling","GroundPlane",temp_prim_path+"/base_link/visuals"]
            collision_details = _collision_check(temp_prim_path, target_position=temp_position, target_rotation=resolved_rotation, white_list=white_list, restore_on_success=True)
            if collision_details:
                print(f"Collision detected when placing object at floor: {collision_details}")
                return _placement_failure(
                    "floor placement collides with existing objects",
                    rotation=resolved_rotation,
                    candidate_position=temp_position,
                    collisions=collision_details,
                )
            if room_name:
                in_room, room_bounds_detail = _check_room_bounds(
                    room_name=room_name,
                    position=temp_position,
                    obj_half_sizes=obj_half_sizes,
                )
                if not in_room:
                    # No collision but out-of-bounds: manually restore the original pose
                    for op, value in _pre_collision_rotation_values:
                        op.Set(value)
                    if not _pre_collision_rotation_values:
                        _set_absolute_rotation(_pre_collision_xformable, (0.0, 0.0, 0.0))
                    set_object_pose(temp_prim_path, position=_pre_collision_center, rotation=None)
                    print(f"Out-of-room placement detected for {temp_prim_path}: {room_bounds_detail}")
                    return _placement_failure(
                        "floor placement falls outside the current room boundary",
                        rotation=resolved_rotation,
                        candidate_position=temp_position,
                        out_of_bounds=room_bounds_detail.get("out_of_bounds", {"x": 0.0, "y": 0.0, "z": 0.0}),
                        room_bounds=room_bounds_detail,
                    )
            return result
        except (TypeError, ValueError) as e:
            print(f"Error resolving floor placement: {str(e)}")
            return _placement_failure(str(e), rotation=resolved_rotation)
        except Exception as e:
            print(f"Error resolving floor placement: {str(e)}")
            return _placement_failure(
                f"floor placement resolution error: {e}",
                rotation=resolved_rotation,
                candidate_position=locals().get("temp_position"),
            )

    if placement_type == "wall_mounted":
        try:
            support_path = intent.get("support_object")
            if not isinstance(support_path, str) or not support_path.strip():
                return _placement_failure(
                    "missing required field 'support_object' for wall_mounted placement",
                    rotation=resolved_rotation,
                )

            mounted_location = intent.get("mounted_location")
            if not isinstance(mounted_location, dict):
                return _placement_failure(
                    "missing required dictionary field 'mounted_location' for wall_mounted placement",
                    rotation=resolved_rotation,
                )
            description = mounted_location.get("description")
            if not isinstance(description, str) or not description.strip():
                description = ""

            support_name = support_path.rstrip("/").rsplit("/", 1)[-1]
            is_ceiling = support_name.lower().startswith("ceiling")
            room_name = _extract_room_name_from_prim_path(intent["prim_path"])

            if is_ceiling:
                ceiling_bounds = _compute_synthetic_ceiling_bounds(room_name)
                if ceiling_bounds is None:
                    return _placement_failure(
                        "could not compute synthetic ceiling bounds from room walls",
                        rotation=resolved_rotation,
                    )
                support_min, support_max, room_center = ceiling_bounds
            else:
                support_bbox = get_object_bbox(support_path)
                if support_bbox is None:
                    return _placement_failure(
                        f"could not get bounding box for mounting support {support_path}",
                        rotation=resolved_rotation,
                    )
                support_min, support_max = support_bbox
                room_center = _room_center_xy(room_name) if room_name else None
                if room_center is None:
                    return _placement_failure(
                        "could not determine room center for wall-mounted surface selection",
                        rotation=resolved_rotation,
                    )

            mounted_surface = infer_mounted_surface(
                support_name=support_name,
                support_min=support_min,
                support_max=support_max,
                room_center_xy=room_center or (0.0, 0.0),
            )
            picture_orientation_override = is_picture_mounted_object(
                intent["prim_path"]
            )
            mounted_rotation = resolve_mounted_orientation(
                object_name_or_path=intent["prim_path"],
                surface=mounted_surface,
                requested_rotation=resolved_rotation,
            )
            mounted_half_sizes = _measure_bbox_half_sizes_at_rotation(
                intent["prim_path"],
                mounted_rotation,
            )
            if mounted_half_sizes is None:
                return _placement_failure(
                    f"could not get bounding box for object {intent['prim_path']}",
                    rotation=mounted_rotation,
                )

            mounted_result = compute_mounted_surface_center(
                mounted_location=mounted_location,
                surface=mounted_surface,
                support_min=support_min,
                support_max=support_max,
                object_half_sizes=mounted_half_sizes,
            )
            if not mounted_result.get("success"):
                mounted_result["rotation"] = mounted_rotation
                return mounted_result

            result_position = tuple(
                float(value)
                for value in mounted_result["position"]
            )

            if is_ceiling:
                collision_details = []
            else:
                room_walls = _get_room_wall_prim_paths(room_name or "")
                white_list = [
                    support_path,
                    intent["prim_path"] + "/base_link/visuals",
                    "floor",
                    "ceiling",
                    "GroundPlane",
                ] + room_walls
                collision_details = _collision_check(
                    intent["prim_path"],
                    target_position=result_position,
                    target_rotation=mounted_rotation,
                    white_list=white_list,
                    restore_on_success=True,
                )
            if collision_details:
                return _placement_failure(
                    "wall_mounted placement collides with existing objects",
                    rotation=mounted_rotation,
                    candidate_position=result_position,
                    collisions=collision_details,
                )

            stability_result = _stabilize_mounted_subtree(intent["prim_path"])
            if not stability_result.get("success"):
                return _placement_failure(
                    "failed to make wall_mounted object kinematic",
                    rotation=mounted_rotation,
                    candidate_position=result_position,
                    mounted_stability=stability_result,
                )

            return {
                "success": True,
                "position": result_position,
                "rotation": mounted_rotation,
                "out_of_bounds": mounted_result.get(
                    "out_of_bounds",
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                ),
                "message": "",
                "mounted_surface": mounted_surface,
                "mounted_location_description": description.strip(),
                "orientation_policy": (
                    "picture_local_z_front"
                    if picture_orientation_override
                    else "requested_or_default_local_x_front"
                ),
                "gravity_disabled": True,
                "gravity_disabled_prim_count": stability_result.get(
                    "updated_count",
                    0,
                ),
                "kinematic_enabled": True,
                "kinematic_enabled_prim_count": stability_result.get(
                    "updated_count",
                    0,
                ),
                "articulation_disabled_count": stability_result.get(
                    "articulation_disabled_count",
                    0,
                ),
            }
        except (TypeError, ValueError) as e:
            print(f"Error resolving wall_mounted placement: {str(e)}")
            return _placement_failure(str(e), rotation=resolved_rotation)
        except Exception as e:
            print(f"Error resolving wall_mounted placement: {str(e)}")
            return _placement_failure(
                f"wall_mounted placement resolution error: {e}",
                rotation=resolved_rotation,
                candidate_position=locals().get("result_position"),
            )

    if placement_type != "surface":
        return _placement_failure(
            f"unknown placement_type '{placement_type}'; expected 'floor', 'surface', or 'wall_mounted'",
            rotation=resolved_rotation,
        )

    try:
        if "support_object" not in intent:
            return _placement_failure(
                "missing required field 'support_object' for surface placement",
                rotation=resolved_rotation,
            )
        if "surface" not in intent:
            return _placement_failure(
                "missing required field 'surface' for surface placement",
                rotation=resolved_rotation,
            )

        support_path = intent["support_object"]
        surface = str(intent["surface"]).strip().lower()
        bbox = get_object_bbox(support_path)
        if bbox is None:
            return _placement_failure(
                f"could not get bounding box for support object {support_path}",
                rotation=resolved_rotation,
            )

        support_min, support_max = bbox
        center = [(support_min[i] + support_max[i]) / 2.0 for i in range(3)]
        surface_center = list(center)
        print(f"Support object bbox: min {support_min}, max {support_max}, center {center}")
        if surface == "down":
            surface_center[2] = support_min[2]
        elif surface == "up":
            surface_center[2] = support_max[2]
        elif surface == "left":
            surface_center[0] = support_min[0]
        elif surface == "right":
            surface_center[0] = support_max[0]
        elif surface == "back":
            surface_center[1] = support_min[1]
        elif surface == "front":
            surface_center[1] = support_max[1]
        elif surface == "random":
            surface = "up"
            surface_center[2] = support_max[2]
        else:
            return _placement_failure(
                f"unknown surface type '{surface}'",
                rotation=resolved_rotation,
            )

        if isinstance(intent.get("semantic_location"), dict):
            print("Resolving semantic surface center with intent:", intent["semantic_location"])
            semantic_result = _compute_semantic_surface_center(
                semantic_location=intent["semantic_location"],
                surface=surface,
                support_min=support_min,
                support_max=support_max,
                obj_half_sizes=obj_half_sizes,
            )
            print(f"Semantic surface center result: {semantic_result}")
            if semantic_result["success"]:
                result_position = tuple(
                    float(value)
                    for value in semantic_result["position"]
                )
                # Run collision detection
                collision_details = _collision_check(intent["prim_path"], target_position=semantic_result["position"], target_rotation=resolved_rotation, white_list=[support_path,intent["prim_path"]+"/base_link/visuals"], restore_on_success=True)
                if collision_details:
                    print(f"Collision detected when placing object on surface with semantic location: {collision_details}")
                    return _placement_failure(
                        "surface placement collides with existing objects",
                        rotation=resolved_rotation,
                        candidate_position=result_position,
                        collisions=collision_details,
                    )
                result = {
                    "success": True,
                    "position": result_position,
                    "rotation": resolved_rotation,
                    "out_of_bounds": semantic_result.get("out_of_bounds", {"x": 0.0, "y": 0.0, "z": 0.0}),
                    "message": semantic_result.get("message", ""),
                }
                return result
            semantic_result.setdefault("rotation", resolved_rotation)
            return semantic_result
        final_location = list(surface_center)
        if "offset_from_center" in intent:
            x_offset, y_offset = intent["offset_from_center"]
            if surface in ["back", "front"]:
                final_location[0] = surface_center[0] + x_offset
                final_location[1] = surface_center[1] + y_length if surface == "front" else surface_center[1] - y_length
                final_location[2] = surface_center[2] + y_offset
            elif surface in ["left", "right"]:
                final_location[0] = surface_center[0] + x_length if surface == "right" else surface_center[0] - x_length
                final_location[1] = surface_center[1] + x_offset
                final_location[2] = surface_center[2] + y_offset
            elif surface in ["down", "up"]:
                final_location[0] = surface_center[0] + x_offset
                final_location[1] = surface_center[1] + y_offset
                final_location[2] = surface_center[2] + z_length if surface == "up" else surface_center[2] - z_length
        else:
            if "x_direction" in intent:
                ref_path, x_distance = intent["x_direction"]
                min_range, max_range = get_object_bbox(ref_path)
                final_location[0] = min_range[0] + x_distance - x_length if x_distance <= 0 else max_range[0] + x_distance + x_length
            if "y_direction" in intent:
                ref_path, y_distance = intent["y_direction"]
                min_range, max_range = get_object_bbox(ref_path)
                final_location[1] = min_range[1] + y_distance - y_length if y_distance <= 0 else max_range[1] + y_distance + y_length
            if "z_direction" in intent:
                ref_path, z_distance = intent["z_direction"]
                min_range, max_range = get_object_bbox(ref_path)
                final_location[2] = min_range[2] + z_distance - z_length if z_distance <= 0 else max_range[2] + z_distance + z_length
            if surface in ["down", "up"]:
                final_location[2] = final_location[2] + z_length if surface == "up" else final_location[2] - z_length
            elif surface in ["left", "right"]:
                final_location[0] = final_location[0] + x_length if surface == "right" else final_location[0] - x_length
            elif surface in ["back", "front"]:
                final_location[1] = final_location[1] + y_length if surface == "front" else final_location[1] - y_length
        _apply_surface_placement_clearance(final_location, surface)
        success, out_of_bounds = _check_surface_bounds(final_location, obj_half_sizes, support_min, support_max, surface)
        if not success:
            candidate = tuple(float(value) for value in final_location)
            return _placement_failure(
                "surface placement exceeds support bounds",
                rotation=resolved_rotation,
                candidate_position=candidate,
                out_of_bounds=out_of_bounds,
                support_info={
                    "support_object": support_path,
                    "support_min": tuple(round(float(v), 4) for v in support_min),
                    "support_max": tuple(round(float(v), 4) for v in support_max),
                    "surface": surface,
                },
            )
        result_position = tuple(float(value) for value in final_location)
        result = {
            "success": True,
            "position": result_position,
            "rotation": resolved_rotation,
            "out_of_bounds": out_of_bounds,
            "message": "",
        }
        # Run collision detection
        temp_prim_path = intent["prim_path"]
        temp_position = result["position"]
        # Hard-coded white list for surface placement: allows collision with the support object
        white_list = [support_path,temp_prim_path+"/base_link/visuals"]
        collision_details = _collision_check(temp_prim_path, target_position=temp_position, target_rotation=resolved_rotation, white_list=white_list, restore_on_success=True)
        if collision_details:
            print(f"Collision detected when placing object on surface: {collision_details}")
            return _placement_failure(
                "surface placement collides with existing objects",
                rotation=resolved_rotation,
                candidate_position=temp_position,
                collisions=collision_details,
            )
        return result
    except Exception as e:
        print(f"Error resolving surface placement: {str(e)}")
        return _placement_failure(
            f"surface placement resolution error: {e}",
            rotation=resolved_rotation,
            candidate_position=locals().get("result_position") or locals().get("temp_position"),
        )



def _find_usd_in_dir(asset_dir: str) -> Optional[str]:
    """Find the primary .usd file inside an asset's usd/ subdirectory.

    Skips ``*.encrypted.usd`` companion files. Returns the absolute path
    of the first matching file, or *None* if no suitable file exists.
    """
    usd_subdir = os.path.join(asset_dir, "usd")
    if not os.path.isdir(usd_subdir):
        return None
    for fname in sorted(os.listdir(usd_subdir)):
        if fname.endswith(".usd") and not fname.endswith(".encrypted.usd"):
            return os.path.join(usd_subdir, fname)
    return None


def collect_assets(assets_list: dict) -> bool:
    '''collect needed assets'''
    stage = omni.usd.get_context().get_stage()
    print(f"stage is {stage}")
    misc_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Misc"
    )
    wall_dirs = sorted([
        d for d in os.listdir(misc_dir)
        if os.path.isdir(os.path.join(misc_dir, d))
        and d.startswith("wall_usd_")
    ])
    floor_dirs = sorted([
        d for d in os.listdir(misc_dir)
        if os.path.isdir(os.path.join(misc_dir, d))
        and d.startswith("floor_usd_")
    ])
    # Pre-select one material for ALL walls in the scene (same wall texture).
    # For floors, each unique asset key gets its own random material so that
    # different rooms can have different floor textures.
    chosen_wall_dir = random.choice(wall_dirs) if wall_dirs else "wall_usd_0"
    chosen_wall_path = _find_usd_in_dir(os.path.join(misc_dir, chosen_wall_dir))
    if chosen_wall_path is None:
        chosen_wall_path = os.path.join(misc_dir, chosen_wall_dir, "usd", "wall.usd")

    floor_choice_map = {}
    for asset, count in assets_list.items():
        if 'floor' in asset.lower():
            chosen_floor_dir = random.choice(floor_dirs) if floor_dirs else "floor_usd_0"
            chosen_floor_path = _find_usd_in_dir(os.path.join(misc_dir, chosen_floor_dir))
            if chosen_floor_path is None:
                chosen_floor_path = os.path.join(misc_dir, chosen_floor_dir, "usd", "floor.usd")
            floor_choice_map[asset] = chosen_floor_path

    for asset, count in assets_list.items():
        for i in range(count):
            try:
                if 'wall' in asset.lower():
                    asset_name = f"{asset}_{i}"
                elif 'floor' in asset.lower():
                    asset_name = f"{asset}_{i}" if count > 1 else asset
                else:
                    asset_name = f"{asset}_{i}" if count > 1 else asset
                # search for the asset in the library
                new_xform = UsdGeom.Xform.Define(stage, f"/World/{asset_name}")
                if 'wall' in asset.lower():
                    new_xform.GetPrim().GetReferences().AddReference(
                        chosen_wall_path)
                elif 'door' in asset.lower():
                    door_usd_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "Misc",
                        "door_usd",
                        "usd",
                        "door.usd"
                    )
                    new_xform.GetPrim().GetReferences().AddReference(
                        door_usd_path)
                elif 'floor' in asset.lower():
                    new_xform.GetPrim().GetReferences().AddReference(
                        floor_choice_map[asset])
                print(f"collected asset:{asset_name} (original: {asset})")
                api = UsdGeom.XformCommonAPI(new_xform)
                api.SetTranslate((0, 0, 0))
                api.SetRotate((0, 0, 0))
                api.SetScale((1, 1, 1))
                prim_path_to_spawn = f"/World/{asset_name}"
                if 'wall' in asset.lower() or 'door' in asset.lower():
                    _strip_rigid_body_keep_colliders(stage, prim_path_to_spawn)
                if 'floor' in asset.lower():
                    _strip_all_physics(stage, prim_path_to_spawn)
                _bind_pbr_materials_and_remove_vray(stage, prim_path_to_spawn)
            except Exception as e:
                print(f"Error collecting asset {asset_name}: {str(e)}")
                continue
    # stage.Save()
    # Get the root layer
    ###root_layer = stage.GetRootLayer()

    # Export the current scene to file
    # root_layer.Save()
    return True


def resize_assets(path: str, scale_factor: dict) -> bool:
    '''correct if asset's size is unreasonable'''
    stage = omni.usd.get_context().get_stage()
    if not path.startswith("/World"):
        path = "/World/" + path
    # Get the prim
    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"failed to find object at path:{path}")
        return False
    # xformable = UsdGeom.Xformable(prim)
    # ordered_ops = xformable.GetOrderedXformOps()
    # scale_op = None
    # for op in ordered_ops:
    #     if op.GetOpType() == UsdGeom.XformOp.TypeScale:
    #         scale_op = op
    #         break
    # if scale_op is None:
    #     scale_op = xformable.AddScaleOp()
    # scale_op.Set(Gf.Vec3f(
    #     scale_factor['x'], scale_factor['y'], scale_factor['z']))
    # Both xform.addxxxop() and xformcommonapi(prim).setxx() can perform transform operations on a prim
    # time = Usd.TimeCode.Default()
    # UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(
    #     scale_factor["x"], scale_factor["y"], scale_factor["z"]), time)
    omni.kit.commands.execute('TransformPrimSRTCommand', path=path, new_scale=Gf.Vec3f(
        scale_factor["x"], scale_factor["y"], scale_factor["z"]))
    # stage.Save()
    print(
        f"resized object at path {path} with scale factor {scale_factor},now its size is {get_size_of_object(path)}")
    #root_layer = stage.GetRootLayer()

    # Save the current root layer
    #root_layer.Save()
    return True


def resize_assets_to_dimensions(path: str, target_dimensions) -> bool:
    """Resize an object to exact world-axis bounding-box dimensions in meters."""
    stage = omni.usd.get_context().get_stage()
    if not path.startswith("/World"):
        path = "/World/" + path
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print(f"failed to find object at path:{path}")
        return False

    try:
        target = [float(value) for value in target_dimensions]
    except (TypeError, ValueError):
        print(f"Invalid target dimensions for {path}: {target_dimensions}")
        return False
    if len(target) != 3 or any(
        not math.isfinite(value) or value <= 0.0
        for value in target
    ):
        print(f"Target dimensions must be three positive finite values: {target}")
        return False

    current_size = get_size_of_object(path)
    if current_size is None:
        return False
    current_dimensions = [float(current_size[i]) for i in range(3)]
    if any(
        not math.isfinite(value) or value <= 1e-8
        for value in current_dimensions
    ):
        print(f"Cannot resize object with invalid current dimensions: {current_dimensions}")
        return False

    xformable = UsdGeom.Xformable(prim)
    current_scale = [1.0, 1.0, 1.0]
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_value = op.Get()
            current_scale = [
                float(scale_value[0]),
                float(scale_value[1]),
                float(scale_value[2]),
            ]
            break

    new_scale = [
        current_scale[i] * target[i] / current_dimensions[i]
        for i in range(3)
    ]
    omni.kit.commands.execute(
        "TransformPrimSRTCommand",
        path=path,
        new_scale=Gf.Vec3f(*new_scale),
    )
    print(
        f"resized object at path {path} to target dimensions {target}; "
        f"new scale is {new_scale}, measured size is {get_size_of_object(path)}"
    )
    return True


def scale_asset_uniform(path: str, scale_factor: float) -> dict:
    """Set an object's BBOX to one uniform factor of its original size."""
    min_scale_factor = 0.7
    max_scale_factor = 1.3
    stage = omni.usd.get_context().get_stage()
    if not path.startswith("/World"):
        path = "/World/" + path
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return {
            "success": False,
            "message": f"failed to find object at path: {path}",
        }

    try:
        factor = float(scale_factor)
    except (TypeError, ValueError):
        return {
            "success": False,
            "message": "scale_factor must be numeric",
        }
    if (
        not math.isfinite(factor)
        or factor < min_scale_factor
        or factor > max_scale_factor
    ):
        return {
            "success": False,
            "message": (
                "scale_factor must be finite and within "
                f"[{min_scale_factor}, {max_scale_factor}]"
            ),
        }

    current_size = get_size_of_object(path)
    if current_size is None:
        return {
            "success": False,
            "message": "could not measure BBOX before scaling",
        }
    before_bbox = [float(current_size[i]) for i in range(3)]
    if any(not math.isfinite(value) or value <= 1e-8 for value in before_bbox):
        return {
            "success": False,
            "message": f"invalid BBOX before scaling: {before_bbox}",
        }

    xformable = UsdGeom.Xformable(prim)
    current_scale = [1.0, 1.0, 1.0]
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_value = op.Get()
            current_scale = [
                float(scale_value[0]),
                float(scale_value[1]),
                float(scale_value[2]),
            ]
            break

    baseline_key = _uniform_scale_key(stage, path)
    baseline = _uniform_scale_baselines.get(baseline_key)
    if baseline is None:
        baseline = {
            "bbox": list(before_bbox),
            "scale": list(current_scale),
        }
        _uniform_scale_baselines[baseline_key] = baseline

    original_bbox = list(baseline["bbox"])
    original_scale = list(baseline["scale"])
    new_scale = [value * factor for value in original_scale]
    omni.kit.commands.execute(
        "TransformPrimSRTCommand",
        path=path,
        new_scale=Gf.Vec3f(*new_scale),
    )

    after_size = get_size_of_object(path)
    if after_size is None:
        omni.kit.commands.execute(
            "TransformPrimSRTCommand",
            path=path,
            new_scale=Gf.Vec3f(*current_scale),
        )
        return {
            "success": False,
            "message": "could not recalculate BBOX after scaling; transform was rolled back",
        }

    after_bbox = [float(after_size[i]) for i in range(3)]
    measured_factors = [
        after_bbox[i] / original_bbox[i]
        for i in range(3)
    ]
    tolerance = max(1e-4, abs(factor) * 1e-3)
    proportional = all(
        math.isfinite(value) and abs(value - factor) <= tolerance
        for value in measured_factors
    )
    if not proportional:
        omni.kit.commands.execute(
            "TransformPrimSRTCommand",
            path=path,
            new_scale=Gf.Vec3f(*current_scale),
        )
        rolled_back_size = get_size_of_object(path)
        return {
            "success": False,
            "message": "post-scale BBOX verification detected non-uniform scaling; transform was rolled back",
            "scale_factor": factor,
            "original_bbox": original_bbox,
            "before_bbox": before_bbox,
            "measured_factors": measured_factors,
            "rolled_back_bbox": (
                [float(rolled_back_size[i]) for i in range(3)]
                if rolled_back_size is not None
                else None
            ),
        }

    print(
        f"uniformly scaled object at {path} to {factor} of original size; "
        f"original BBOX {original_bbox}, current BBOX {before_bbox} -> {after_bbox}"
    )
    return {
        "success": True,
        "message": "",
        "scale_factor": factor,
        "original_bbox": original_bbox,
        "before_bbox": before_bbox,
        "after_bbox": after_bbox,
        "measured_factors": measured_factors,
        "new_scale": new_scale,
    }


def place_assets(path: str, original_pos: dict, target_pos: dict) -> bool:
    '''place collected assets in the environment to proper position'''
    stage = omni.usd.get_context().get_stage()
    if not path.startswith("/World"):
        path = "/World/" + path
    # Get the prim
    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"failed to find object at path:{path}")
        return False
    # Get the translate op
    xformable = UsdGeom.Xformable(prim)
    # masscenter_pos = prim.GetAttribute("xformOp:translate").Get()
    # print(f"masscenter_pos is {masscenter_pos}")
    ordered_ops = xformable.GetOrderedXformOps()
    translate_op = None
    for op in ordered_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    # Centroid displacement is target_pos - original_pos
    translate_op.Set(Gf.Vec3f(
        target_pos['x']-original_pos['x'], target_pos['y']-original_pos['y'], target_pos['z']-original_pos['z']))
    # time = Usd.TimeCode.Default()
    # UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(
    #     target_pos["x"], target_pos["y"], target_pos["z"]), time)
    # print(f"placed object at path {path} to position {target_pos}")

    # stage.Save()
    #root_layer = stage.GetRootLayer()

    # Save the current root layer
    #root_layer.Save()
    return True


def get_center_of_surface(path: str, surface: str) -> Gf.Vec3d | None:
    '''obtain the center point of certain surface in the scene. only accurate for cuboid

    args:
        path:prim path of the object needed
        surface:specify which surface want to calculate,only accept args left,right,down,up,behind,front,specifying certain surface of the object '''
    stage = omni.usd.get_context().get_stage()
    # Get the prim
    if not path.startswith("/World"):
        path = "/World/" + path
    # Get the prim
    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"failed to find object at path:{path}")
        return None
    # GroundPlane's bounding box is infinitely large; get its surface position via translate
    if "GroundPlane" in path:
        return prim.GetAttribute("xformOp:translate").Get()
    # Get the range
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[
                                   UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    prim_bbox = bbox_cache.ComputeWorldBound(prim)
    prim_range = prim_bbox.ComputeAlignedRange()
    surface_list = ["left", "right", "down", "up", "behind", "front"]
    if surface not in surface_list:
        print(f"failed to recognize surface:{surface}")
        return None
    else:
        # calculate position of corner point LDB
        LDB = prim_range.GetCorner(0)
        # calculate position of corner point RUF
        RUF = prim_range.GetCorner(7)
        # calculate three side length of the cuboid range
        # Determine the direction of xyz axes; sometimes axes point out of the screen, sometimes into it
        xlength = (RUF[0]-LDB[0])/2
        ylength = (RUF[1]-LDB[1])/2
        zlength = (RUF[2]-LDB[2])/2
        # calculate center of the cuboid
        bodycenter = (LDB+RUF)/2
        # calculate center of certain surface
        idx = surface_list.index(surface)
        # return center of surface
        if idx < 2:
            if xlength >= 0:
                surface_center = bodycenter - \
                    [xlength, 0, 0] if idx % 2 == 0 else bodycenter + [xlength, 0, 0]
            else:
                # X axis points left
                surface_center = bodycenter + \
                    [xlength, 0, 0] if idx % 2 == 0 else bodycenter - [xlength, 0, 0]
        elif idx < 4:
            if zlength >= 0:
                surface_center = bodycenter - \
                    [0, 0, zlength] if idx % 2 == 0 else bodycenter + [0, 0, zlength]
            else:
                # Z axis points down
                surface_center = bodycenter + \
                    [0, 0, zlength] if idx % 2 == 0 else bodycenter - [0, 0, zlength]
        else:
            if ylength >= 0:
                surface_center = bodycenter - \
                    [0, ylength, 0] if idx % 2 == 0 else bodycenter + [0, ylength, 0]
            else:
                # Y axis points backward
                surface_center = bodycenter + \
                    [0, ylength, 0] if idx % 2 == 0 else bodycenter - [0, ylength, 0]
        print(
            f"center of surface {surface} at path {path} is {surface_center}")
        return surface_center


def get_size_of_object(path: str) -> Gf.Vec3d | None:
    '''get size of certain object given from path

    args:
        path:prim path of the object needed
    '''
    stage = omni.usd.get_context().get_stage()
    # Get the prim
    if not path.startswith("/World"):
        path = "/World/" + path
    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"failed to find object at path:{path}")
        return None
    # Get the size
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[
                                   UsdGeom.Tokens.default_])
    bbox_cache.Clear()
    prim_bbox = bbox_cache.ComputeWorldBound(prim)
    prim_range = prim_bbox.ComputeAlignedRange()
    prim_size = prim_range.GetSize()
    print(f"size of object at path {path} is {prim_size}")
    return prim_size


def get_sub_prims(path: str) -> list:
    '''get sub prim under this prim of th path

    args:
        path: prim path of the prim
    '''
    stage = omni.usd.get_context().get_stage()

    if not path.startswith("/World"):
        path = "/World/" + path
    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"failed to find object at path:{path}")
        return None

    result_prim_paths = []
    stack = list(prim.GetChildren())

    while stack:
        current_prim = stack.pop()

        if current_prim.GetTypeName() == "Scope":
            continue

        # Get all Xform type child prims
        xform_children = [
            child for child in current_prim.GetChildren()
            if child.GetTypeName() in ["Xform", "omnijoint"]
        ]

        if xform_children:
            # Has Xform children, continue traversing them
            stack.extend(xform_children)
        else:
            # No Xform children, add to results
            result_prim_paths.append(str(current_prim.GetPath()))

    return result_prim_paths


def record_scene_snapshot() -> str:
    '''Record current scene snapshot and save it to a directory.

    Args:
        output_dir: Optional directory to save the snapshot. If None, creates a timestamped directory.

    Returns:
        str: Path to the saved snapshot image, or error message.
    '''
    try:
        from recorder import recorder
        import time

        scene_recorder = recorder()

        # Create a timestamped directory to avoid conflicts
        timestamp = int(time.time())
        snapshot_root = os.environ.get(
            "SCENE_SNAPSHOT_ROOT",
            os.path.join(os.path.expanduser("~"), "scene_snapshots"),
        )
        output_dir = os.path.join(snapshot_root, f"snapshot_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        scene_recorder.out_dir = output_dir
        scene_recorder.initialize()
        scene_recorder.write()
        scene_recorder.stop_writing()

        png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        if not png_files:
            return "Error: No scene snapshot captured."

        snapshot_path = os.path.join(output_dir, png_files[0])
        return snapshot_path
    except Exception as e:
        return f"Error recording scene snapshot: {str(e)}"


def focus_on_prim(camera_path: str, prim_path: str) -> bool:
    '''Focus camera on a specific prim

    args:
        camera_path: path of the camera to be moved
        prim_path: path of the prim to be focused on

    Returns:
        bool: True if successful, False otherwise
    '''
    try:
        import omni.kit.commands
        from pxr import Usd, UsdGeom

        # Get the stage

        # Check if camera exists
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim:
            print(f"Camera not found at path: {camera_path}")
            return False

        # Check if prim exists
        prim_to_frame = stage.GetPrimAtPath(prim_path)
        if not prim_to_frame:
            print(f"Prim not found at path: {prim_path}")
            return False

        # Use default values for framing parameters
        time = Usd.TimeCode.Default()
        resolution = (1920, 1080)  # Default HD resolution
        zoom = 0.6

        # Calculate aspect ratio
        aspect_ratio = resolution[0] / \
            resolution[1] if resolution[1] != 0 else 1.0

        # Execute the FramePrimsCommand
        omni.kit.commands.execute(
            'FramePrimsCommand',
            # The path to the camera that is being moved
            prim_to_move=camera_path,
            # The prim that is being framed / looked at
            prims_to_frame=prim_path,
            # The Usd.TimeCode that camera_path will use to set new location and orientation
            time_code=time,
            # The aspect_ratio of the image-plane that is being viewed
            aspect_ratio=aspect_ratio,
            # Additional slop to use for the framing
            zoom=zoom
        )

        print(f"Successfully focused camera {camera_path} on prim {prim_path}")
        return True

    except Exception as e:
        print(f"Error focusing camera on prim: {str(e)}")
        return False


def modify_orientation(path: str, rotation: dict) -> bool:
    '''modify orientation of an object by rotating it

    args:
        path: prim path of the object
        rotation: dict specifying rotation angles in degrees on x,y,z axes
    '''
    stage = omni.usd.get_context().get_stage()
    if not path.startswith("/World"):
        path = "/World/" + path
    # Get the prim
    prim = stage.GetPrimAtPath(path)
    if not prim:
        print(f"failed to find object at path:{path}")
        return False
    # Get or create rotate op
    xformable = UsdGeom.Xformable(prim)
    ordered_ops = xformable.GetOrderedXformOps()
    rotate_op = None
    for op in ordered_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
            break
    if rotate_op is None:
        rotate_op = xformable.AddRotateXYZOp()
    # Set rotation angles (convert to degrees if needed)
    # Note: USD typically uses degrees for rotation
    rotate_op.Set(Gf.Vec3f(
        rotation['x'], rotation['y'], rotation['z']))

    print(f"rotated object at path {path} with rotation angles {rotation}")

    # Save the scene
    #root_layer = stage.GetRootLayer()
    #root_layer.Save()
    return True


def subtract_segments(original_edge, shared_edges):
    (p1, p2) = original_edge
    (x1, y1), (x2, y2) = p1, p2
    is_horizontal = abs(y1 - y2) < 1e-9
    
    start, end = (min(x1, x2), max(x1, x2)) if is_horizontal else (min(y1, y2), max(y1, y2))
    fixed = y1 if is_horizontal else x1
    remaining_segments = [(start, end)]
    
    for s_edge in shared_edges:
        (sx1, sy1), (sx2, sy2) = s_edge
        if is_horizontal:
            if abs(sy1 - fixed) > 1e-9 or abs(sy2 - fixed) > 1e-9: continue
            s_start, s_end = min(sx1, sx2), max(sx1, sx2)
        else:
            if abs(sx1 - fixed) > 1e-9 or abs(sx2 - fixed) > 1e-9: continue
            s_start, s_end = min(sy1, sy2), max(sy1, sy2)
            
        new_remaining = []
        for r_start, r_end in remaining_segments:
            overlap_start, overlap_end = max(r_start, s_start), min(r_end, s_end)
            if overlap_start < overlap_end - 1e-9:
                if r_start < overlap_start - 1e-9: new_remaining.append((r_start, overlap_start))
                if overlap_end < r_end - 1e-9: new_remaining.append((overlap_end, r_end))
            else:
                new_remaining.append((r_start, r_end))
        remaining_segments = new_remaining

    return [(((s, fixed), (e, fixed)) if is_horizontal else ((fixed, s), (fixed, e))) for s, e in remaining_segments]

def create_gap_and_get_door(edge, gap_size=1.0):
    (p1, p2) = edge
    length = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
    if length <= gap_size: return [], edge
    ux, uy = (p2[0]-p1[0])/length, (p2[1]-p1[1])/length
    mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
    gap_p1 = (mx - ux * gap_size/2, my - uy * gap_size/2)
    gap_p2 = (mx + ux * gap_size/2, my + uy * gap_size/2)
    return [(p1, gap_p1), (gap_p2, p2)], (gap_p1, gap_p2)

def process_room_data_with_all_stats(boundaries, connectivity, gap_width=1.8):
    """
    final_boundaries: room outer walls + interior walls with holes assigned to r1
    final_connectivity: shared edges with holes
    all_doors_dict: dict format { (room_a, room_b): ((x1, y1), (x2, y2)) }
    """
    final_connectivity = {}
    all_doors_dict = {}  # Use dict storage

    # --- Step 1: Process connectivity and record door openings ---
    for pair, shared_edges in connectivity.items():
        if not shared_edges:
            continue

        # Ensure consistent key ordering (optional, for easier lookup, e.g. (0, 1) and (1, 0) both point to the same door)
        sorted_pair = tuple(sorted(pair))

        # Only place doors on shared edges that are long enough to keep
        # usable wall segments on both sides of the configured opening.
        eligible_indices = []
        for i, edge in enumerate(shared_edges):
            (x1, y1), (x2, y2) = edge
            edge_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if edge_length >= 2.0:
                eligible_indices.append(i)

        new_shared = []

        idx_to_gap = random.choice(eligible_indices) if eligible_indices else None

        for i, edge in enumerate(shared_edges):
            if idx_to_gap is not None and i == idx_to_gap:
                # Call the gap-cutting function
                remaining, door = create_gap_and_get_door(edge, gap_width)
                new_shared.extend(remaining)

                # Store in dict: key is room tuple, value is the door opening edge coordinates
                all_doors_dict[sorted_pair] = door
            else:
                new_shared.append(edge)

        final_connectivity[pair] = new_shared

    # --- Step 2: Calculate boundaries (logic unchanged) ---
    final_boundaries = []
    for room_idx, all_edges in enumerate(boundaries):
        raw_shared = [e for p, es in connectivity.items() if room_idx in p for e in es]

        # Subtract original shared edges to get outer walls
        refined_b = []
        for edge in all_edges:
            refined_b.extend(subtract_segments(edge, raw_shared))

        # Merge interior walls with holes (added only once, assigned to r1)
        for (r1, r2), gapped_edges in final_connectivity.items():
            if room_idx == r1:
                refined_b.extend(gapped_edges)

        final_boundaries.append(refined_b)

    return final_boundaries, final_connectivity, all_doors_dict

# --- Execution ---
# final_b, final_c, doors = process_room_data_with_all_stats(boundaries, connectivity)


def _create_ceiling_mount_surface(
    stage,
    room_name: str,
    room_edges,
    ceiling_height: float = 2.8,
    thickness: float = 0.1,
) -> Optional[str]:
    """Create an invisible ceiling prim that provides mounting bounds."""
    points = [point for edge in room_edges for point in edge]
    if not points:
        return None

    min_x = min(float(point[0]) for point in points)
    max_x = max(float(point[0]) for point in points)
    min_y = min(float(point[1]) for point in points)
    max_y = max(float(point[1]) for point in points)
    width = max(max_x - min_x, 0.01)
    depth = max(max_y - min_y, 0.01)
    ceiling_path = f"/World/rooms/{room_name}/ceiling"

    ceiling = UsdGeom.Cube.Define(stage, ceiling_path)
    ceiling.CreateSizeAttr(1.0)
    transform = UsdGeom.XformCommonAPI(ceiling)
    transform.SetTranslate(
        (
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            ceiling_height + thickness / 2.0,
        )
    )
    transform.SetScale((width, depth, thickness))
    UsdGeom.Imageable(ceiling.GetPrim()).MakeInvisible()
    _strip_rigid_body_keep_colliders(stage, ceiling_path)
    print(
        f"Created ceiling mounting surface at {ceiling_path} "
        f"with size ({width}, {depth}, {thickness})"
    )
    return ceiling_path


def create_room(rooms: int, house_length: int, connectivity: list):
    '''create a house spilted into multiple rooms

    args:
        rooms: names of rooms to be created, e.g. ["living_room", "kitchen", "bedroom"]
        house_length: length of the entire house
        connectivity: list of connections between rooms
    '''
    room_num = len(rooms)
    boundaries = []
    all_connection = {}
    try:
        boundaries, all_connection, room_rects = design_room.design_room(
            room_num, house_length)
        final_boundaries, final_connectivity,all_gaps = process_room_data_with_all_stats(boundaries, all_connection)
        print(final_connectivity)
        # Add shared edges from final_connectivity to final_boundaries so that wall placement can handle both unique and shared boundaries together
        for (r1, r2), shared_edges in final_connectivity.items():
            final_boundaries[r1].extend(shared_edges)
        print("final boundaries after adding shared edges:", final_boundaries)
        # Step 2: Collect walls under each room prim so that wall primpaths
        # become /World/rooms/room_i/wall_j
        # Process each room's boundary lines, removing shared boundary segments with other rooms to get each room's unique boundary segments
        try:
            total_walls = sum(len(room) for room in final_boundaries) + len(final_connectivity)
        except Exception:
            total_walls = 0

        # prepare per-room asset collection mapping
        assets_to_collect = {}
        for ri, room in enumerate(final_boundaries):
            count = len(room)
            if count > 0:
                assets_to_collect[f"rooms/{rooms[ri]}/wall"] = count

        door_counts_by_room = {}
        door_asset_paths = {}
        for (r1, r2), door in all_gaps.items():
            if door is None:
                continue
            room_name = rooms[r1]
            door_counts_by_room[room_name] = door_counts_by_room.get(room_name, 0) + 1
            assets_to_collect[f"rooms/{room_name}/door"] = door_counts_by_room[room_name]

        door_index_by_room = {}
        for (r1, r2), door in all_gaps.items():
            if door is None:
                continue
            room_name = rooms[r1]
            room_total = door_counts_by_room[room_name]
            room_index = door_index_by_room.get(room_name, 0)
            door_index_by_room[room_name] = room_index + 1
            if room_total > 1:
                door_asset_paths[(r1, r2)] = f"/World/rooms/{room_name}/door_{room_index}"
            else:
                door_asset_paths[(r1, r2)] = f"/World/rooms/{room_name}/door"

        # Add floor assets
        FLOOR_THICKNESS = 0.02
        FLOOR_TOP_Z = 0.005
        all_rooms_are_rectangular = all(
            len(rects) == 1 for rects in room_rects
        )
        floor_info = {}
        if all_rooms_are_rectangular:
            for ri, room_name in enumerate(rooms):
                rect = room_rects[ri][0]
                (x1, y1), (x2, y2) = rect
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                floor_info[f"rooms/{room_name}/floor"] = {
                    "prim_path": f"/World/rooms/{room_name}/floor",
                    "width": max(x2 - x1, 0.01),
                    "depth": max(y2 - y1, 0.01),
                    "center_x": (x1 + x2) / 2.0,
                    "center_y": (y1 + y2) / 2.0,
                }
                assets_to_collect[f"rooms/{room_name}/floor"] = 1
        else:
            whole_min_x = whole_max_x = whole_min_y = whole_max_y = 0.0
            all_rects_flat = [rect for rects in room_rects for rect in rects]
            if all_rects_flat:
                all_points = [
                    (float(p[0]), float(p[1]))
                    for rect in all_rects_flat
                    for p in rect
                ]
                whole_min_x = min(p[0] for p in all_points)
                whole_max_x = max(p[0] for p in all_points)
                whole_min_y = min(p[1] for p in all_points)
                whole_max_y = max(p[1] for p in all_points)
            floor_info["rooms/_whole_house/floor"] = {
                "prim_path": "/World/rooms/_whole_house/floor",
                "width": max(whole_max_x - whole_min_x, 0.01),
                "depth": max(whole_max_y - whole_min_y, 0.01),
                "center_x": (whole_min_x + whole_max_x) / 2.0,
                "center_y": (whole_min_y + whole_max_y) / 2.0,
            }
            assets_to_collect["rooms/_whole_house/floor"] = 1


        if assets_to_collect:
            collected = collect_assets(assets_to_collect)
            if not collected:
                print("Warning: failed to collect all wall assets")

        stage = omni.usd.get_context().get_stage()
        for ri, room_edges in enumerate(boundaries):
            room_prim = stage.GetPrimAtPath(f"/World/rooms/{rooms[ri]}")
            if not room_prim or not room_prim.IsValid():
                print(f"Warning: failed to store room boundary metadata for room {rooms[ri]}")
                continue
            serialized_edges = [
                [[float(edge[0][0]), float(edge[0][1])], [float(edge[1][0]), float(edge[1][1])]]
                for edge in room_edges
            ]
            room_prim.SetCustomDataByKey("room_boundary_segments", json.dumps(serialized_edges))

        # Step 3: Determine initial orientation using the first room's first wall
        wall_initial_orientation = None
        door_initial_orientation = None
        has_any_door = any(door is not None for door in all_gaps.values())
        first_door_prim_path = next(iter(door_asset_paths.values()), None)
        if total_walls > 0:
            try:
                wall_size = get_size_of_object(f"/World/rooms/{rooms[0]}/wall_0")
                door_size = get_size_of_object(first_door_prim_path) if first_door_prim_path else None
                if wall_size is not None:
                    if wall_size[0] >= wall_size[1]:
                        wall_initial_orientation = "perpendicular_to_y"
                    else:
                        wall_initial_orientation = "perpendicular_to_x"
                if door_size is not None:
                    if door_size[0] >= door_size[1]:
                        door_initial_orientation = "perpendicular_to_y"
                    else:
                        door_initial_orientation = "perpendicular_to_x"
            except Exception as e:
                print(f"Error getting wall size: {e}")

        # # if needed, rotate all walls to make them face along x-axis
        # if wall_initial_orientation == "perpendicular_to_x":
        #     for ri, room in enumerate(boundaries):
        #         for wi in range(len(room)):
        #             try:
        #                 set_object_pose(f"/World/rooms/room_{ri}/wall_{wi}", rotation=(0, 0, 90))
        #             except Exception:
        #                 pass

        # Step 4: Place walls according to boundary coordinates; operate on
        # per-room wall prims (/World/rooms/room_i/wall_j)
        wall_counter = 0
        for ri, room in enumerate(final_boundaries):
            for wi, edge in enumerate(room):
                try:
                    (x1, y1), (x2, y2) = edge
                except Exception:
                    continue

                length = max(abs(x2 - x1), abs(y2 - y1))

                # get wall size once (from first wall under room_0)
                if wall_counter == 0:
                    try:
                        base_wall_size = get_size_of_object(f"/World/rooms/{rooms[0]}/wall_0")
                    except Exception:
                        base_wall_size = None

                if 'base_wall_size' in locals() and base_wall_size is not None and base_wall_size[0] != 0:
                    if wall_initial_orientation == "perpendicular_to_y":
                        wall_scale_factor = {
                            "x": (length / float(base_wall_size[0])) if base_wall_size[0] != 0 else 1.0,
                            "y": (0.15 / float(base_wall_size[1])) if base_wall_size[1] != 0 else 1.0,
                            "z": (2.8 / float(base_wall_size[2])) if base_wall_size[2] != 0 else 1.0,
                        }
                    else:
                        wall_scale_factor = {
                            "x": (0.15 / float(base_wall_size[0])) if base_wall_size[0] != 0 else 1.0,
                            "y": (length / float(base_wall_size[1])) if base_wall_size[1] != 0 else 1.0,
                            "z": (2.8 / float(base_wall_size[2])) if base_wall_size[2] != 0 else 1.0,
                        }
                    print(f"Calculated wall scale factor for wall_{wi} in room_{ri}: {wall_scale_factor},length:{length},base_wall_size:{base_wall_size}")
                else:
                    wall_scale_factor = {"x": 1.0, "y": 1.0, "z": 1.0}

                prim_path = f"/World/rooms/{rooms[ri]}/wall_{wi}"

                try:
                    resize_assets(prim_path, wall_scale_factor)
                except Exception as e:
                    print(f"Error resizing wall at {prim_path}: {e}")

                target_pos = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, 1.4)

                # Get initial centroid
                bbox = get_object_bbox(prim_path)
                if bbox is None:
                    print(f"Skipping wall placement because bbox is unavailable: {prim_path}")
                    continue
                min_bbox, max_bbox = bbox
                original_pos = ((max_bbox[0] + min_bbox[0]) / 2.0,(max_bbox[1] + min_bbox[1]) / 2.0, (max_bbox[2] + min_bbox[2]) / 2.0)

                # Determine rotation: horizontal edge -> 0, vertical edge -> 90
                if wall_initial_orientation == "perpendicular_to_y":
                    rotation = 0 if abs(y2 - y1) == 0 else 90
                else:
                    rotation = 90 if abs(y2 - y1) == 0 else 0

                try:
                    ans = set_object_pose(prim_path, position=(target_pos[0],target_pos[1],target_pos[2]), rotation=(0,0,rotation))
                    print(f"Placement result: {ans}")
                    print(f"Placing wall at {prim_path} to position {target_pos} with rotation {rotation}")
                except Exception as e:
                    print(f"Error placing wall at {prim_path}: {e}")

                wall_counter += 1
        # Door dimensions are fixed: length 1m, width 0.15m, height 2.8m, so once orientation is determined it can be placed directly in the gap
        # Get the door's initial dimensions
        for room_name, room_edges in zip(rooms, boundaries):
            try:
                _create_ceiling_mount_surface(
                    stage,
                    room_name,
                    room_edges,
                )
            except Exception as e:
                print(
                    f"Error creating ceiling mounting surface for "
                    f"{room_name}: {e}"
                )

        # Place floors: resize and position each floor prim
        first_floor_key = next(iter(floor_info), None)
        base_floor_size = None
        if first_floor_key:
            first_floor_path = floor_info[first_floor_key]["prim_path"]
            try:
                base_floor_size = get_size_of_object(first_floor_path)
            except Exception as e:
                print(f"Error getting floor size: {e}")
        for floor_key, finfo in floor_info.items():
            prim_path = finfo["prim_path"]
            width = finfo["width"]
            depth = finfo["depth"]
            center_x = finfo["center_x"]
            center_y = finfo["center_y"]
            center_z = FLOOR_TOP_Z - FLOOR_THICKNESS / 2.0
            if base_floor_size is not None and base_floor_size[0] != 0 and base_floor_size[1] != 0 and base_floor_size[2] != 0:
                scale_x = width / base_floor_size[0]
                scale_y = depth / base_floor_size[1]
                scale_z = FLOOR_THICKNESS / base_floor_size[2]
            else:
                scale_x, scale_y, scale_z = 1.0, 1.0, 1.0
            try:
                resize_assets(prim_path, {"x": scale_x, "y": scale_y, "z": scale_z})
                set_object_pose(prim_path, position=(center_x, center_y, center_z))
                _strip_all_physics(stage, prim_path)
                print(f"Placed floor at {prim_path}: scale=({scale_x:.3f},{scale_y:.3f},{scale_z:.3f}), pos=({center_x:.2f},{center_y:.2f},{center_z:.4f})")
            except Exception as e:
                print(f"Error placing floor at {prim_path}: {e}")

        base_door_size = get_size_of_object(first_door_prim_path) if first_door_prim_path else None
        for (r1,r2), door in all_gaps.items():
            if door is not None and base_door_size is not None and door_initial_orientation is not None:
                door_prim_path = door_asset_paths.get((r1, r2))
                if door_prim_path is None:
                    continue
                target_door_length = math.hypot(door[1][0] - door[0][0], door[1][1] - door[0][1])
                target_door_thickness = 0.15
                target_door_height = 2.8
                # Calculate door scale factor
                if door_initial_orientation == "perpendicular_to_y":
                    door_scale_factor = {
                        "x": (target_door_length / float(base_door_size[0])) if base_door_size[0] != 0 else 1.0,
                        "y": (target_door_thickness / float(base_door_size[1])) if base_door_size[1] != 0 else 1.0,
                        "z": (target_door_height / float(base_door_size[2])) if base_door_size[2] != 0 else 1.0,
                    }
                else:
                    door_scale_factor = {
                        "x": (target_door_thickness / float(base_door_size[0])) if base_door_size[0] != 0 else 1.0,
                        "y": (target_door_length / float(base_door_size[1])) if base_door_size[1] != 0 else 1.0,
                        "z": (target_door_height / float(base_door_size[2])) if base_door_size[2] != 0 else 1.0,
                    }
                # Calculate target position
                target_pos = ((door[0][0] + door[1][0]) / 2.0, (door[0][1] + door[1][1]) / 2.0, 1.4)
                # Calculate rotation
                if door_initial_orientation == "perpendicular_to_y":
                    rotation = 0 if abs(door[1][1] - door[0][1]) == 0 else 90
                else:
                    rotation = 90 if abs(door[1][1] - door[0][1]) == 0 else 0
                try:
                    print(f"resize:{door_scale_factor}, gap_length:{target_door_length}")
                    resize_assets(door_prim_path, door_scale_factor)
                except Exception as e:
                    print(f"Error resizing door at {door_prim_path}: {e}")
                try:
                    ans = set_object_pose(door_prim_path, position=(target_pos[0],target_pos[1],target_pos[2]), rotation=(0,0,rotation))
                    print(f"Placement result: {ans}")
                    print(f"Placing door at {door_prim_path} to position {target_pos} with rotation {rotation}")
                    # stage = omni.usd.get_context().get_stage()
                    # leaf_prim_path = f"{door_prim_path}/door/leaf"
                    # if stage.RemovePrim(leaf_prim_path):
                    #     print(f"Removed door leaf at {leaf_prim_path}")
                    # elif _disable_prim_in_current_layer(leaf_prim_path):
                    #     print(f"Disabled door leaf at {leaf_prim_path}")
                    # elif _hide_prim_in_current_layer(leaf_prim_path):
                    #     print(f"Hid door leaf at {leaf_prim_path}")
                    # else:
                    #     print(f"Failed to suppress door leaf at {leaf_prim_path}")
                except Exception as e:
                    print(f"Error placing door at {door_prim_path}: {e}")
        return boundaries, all_connection, room_rects

    except Exception as e:
        print(f"Error creating room: {str(e)}")
        traceback.print_exc()
        return boundaries, all_connection, []


def delete_object(object_name: str, room_name: str) -> bool:
    """Remove an object prim from the stage.

    Args:
        object_name: Name of the object to delete (e.g., "Fridge").
        room_name: Room the object belongs to (used to build the prim path).

    Returns:
        bool: True if the prim was successfully removed, False otherwise.
    """
    stage = omni.usd.get_context().get_stage()
    if room_name:
        prim_path = f"/World/rooms/{room_name}/{object_name}"
    else:
        prim_path = f"/World/{object_name}"
    _uniform_scale_baselines.pop(
        _uniform_scale_key(stage, prim_path),
        None,
    )

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"delete_object: prim not found at {prim_path}")
        return False

    try:
        success = stage.RemovePrim(prim_path)
        if success:
            print(f"delete_object: removed {prim_path}")
        else:
            deactivated = _disable_prim_in_current_layer(prim_path)
            if deactivated:
                print(f"delete_object: deactivated {prim_path} (remove not possible)")
                success = True
            else:
                hidden = _hide_prim_in_current_layer(prim_path)
                if hidden:
                    print(f"delete_object: hid {prim_path} (remove/deactivate not possible)")
                    success = True
                else:
                    print(f"delete_object: failed to remove/deactivate/hide {prim_path}")
        return bool(success)
    except Exception as e:
        print(f"delete_object: error removing {prim_path}: {e}")
        return False

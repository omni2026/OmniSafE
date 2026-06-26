try:
    # Package import path, used when Tools is imported as a module.
    from .proxy import IOPipe
except ImportError:
    # Script/debug path: ensure the project root is importable before absolute import.
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[3]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    from src.scene_genetator.Tools.proxy import IOPipe
from typing import Dict, List, Optional, Tuple
from langchain.tools import tool
from threading import Lock
import ast
import math

# ---------------------------------------------------------------------------
# Multi-instance pipe management
# ---------------------------------------------------------------------------
# Each scene-generator process is paired with exactly one Isaac Sim process.
# ``_pipes`` maps a *pipe_id* to the corresponding ``IOPipe`` instance so that
# concurrent processes do not interfere with each other.  The default pipe_id
# (empty string) preserves backward compatibility with existing single-instance
# launches.
# ---------------------------------------------------------------------------

_pipes: Dict[str, IOPipe] = {}
_pipes_lock = Lock()
_pipe_request_locks: Dict[str, Lock] = {}
_pipe_request_locks_lock = Lock()

_active_pipe_id: str = ""  # default – single-instance mode


def set_pipe_id(pipe_id: str) -> None:
    """Set the active *pipe_id* for the current process.

    This must be called **before** any tool invocation so that
    ``_get_pipe()`` / ``_call_isaac()`` route commands to the correct Isaac
    Sim instance.  Typically called once at startup from ``scene_generate.py``.
    """
    global _active_pipe_id
    _active_pipe_id = str(pipe_id).strip()


def get_pipe_id() -> str:
    """Return the currently active *pipe_id*."""
    return _active_pipe_id


def _get_request_lock(pipe_id: str) -> Lock:
    """Return (or create) the per-pipe request-serialization lock."""
    with _pipe_request_locks_lock:
        if pipe_id not in _pipe_request_locks:
            _pipe_request_locks[pipe_id] = Lock()
        return _pipe_request_locks[pipe_id]


def _get_pipe(pipe_id: str = None) -> IOPipe:
    """Lazily create and connect the pipe for *pipe_id* on first real tool call."""
    pid = pipe_id if pipe_id is not None else _active_pipe_id
    if pid not in _pipes:
        with _pipes_lock:
            if pid not in _pipes:
                instance = IOPipe(pipe_id=pid)
                instance.setup()
                _pipes[pid] = instance
    return _pipes[pid]


def _call_isaac(command: str):
    """Send one command to Isaac and receive its matching response.

    LangGraph may execute tool calls concurrently. The Isaac pipe is a single
    request/response channel, so send and receive must be serialized together.
    """
    pid = _active_pipe_id
    lock = _get_request_lock(pid)
    with lock:
        pipe = _get_pipe(pid)
        pipe.send_to_is(command)
        return pipe.recv_from_is()


_asset_name_usd_paths: Dict[str, str] = {}
_asset_name_usd_paths_lock = Lock()
_resolved_positions: Dict[str, List[Tuple[float, float, float]]] = {}
_resolved_positions_lock = Lock()
_active_room_names: Dict[str, str] = {}
_spawned_unplaced_objects: Dict[Tuple[str, str], set[str]] = {}
_placement_lifecycle_lock = Lock()
_RESOLVED_POSITION_TOLERANCE = 1e-4
_MIN_UNIFORM_SCALE_FACTOR = 0.7
_MAX_UNIFORM_SCALE_FACTOR = 1.3


def _normalize_vec3(value, field_name: str, allow_none: bool = False) -> Optional[Tuple[float, float, float]]:
    """Accept tuple/list or stringified tuple/list and normalize to a float vec3."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} cannot be None")

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except Exception as e:
            raise ValueError(
                f"{field_name} must be a 3-number tuple/list or a string like '(x, y, z)' / '[x, y, z]': {e}"
            ) from e

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values")

    try:
        return (float(parsed[0]), float(parsed[1]), float(parsed[2]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{field_name} must contain only numeric values") from e


def _extract_position_from_result(result: dict) -> Optional[Tuple[float, float, float]]:
    """Extract a resolved position from either standardized or legacy resolver output."""
    if not isinstance(result, dict):
        return None

    if result.get("position") is not None:
        try:
            return _normalize_vec3(result["position"], "resolved position")
        except ValueError:
            return None

    if all(axis in result for axis in ("x", "y", "z")):
        try:
            return (float(result["x"]), float(result["y"]), float(result["z"]))
        except (TypeError, ValueError):
            return None

    return None


def _record_resolved_position(object_name: str, position: Tuple[float, float, float]) -> None:
    with _resolved_positions_lock:
        for key in _object_name_keys(object_name):
            positions = _resolved_positions.setdefault(key, [])
            if not any(_positions_close(position, recorded) for recorded in positions):
                positions.append(position)


def _clear_resolved_positions(object_name: str) -> None:
    with _resolved_positions_lock:
        for key in _object_name_keys(object_name):
            _resolved_positions.pop(key, None)


def _positions_close(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> bool:
    return all(abs(a - b) <= _RESOLVED_POSITION_TOLERANCE for a, b in zip(left, right))


def _position_was_resolved(object_name: str, position: Tuple[float, float, float]) -> bool:
    with _resolved_positions_lock:
        return any(
            _positions_close(position, recorded)
            for key in _object_name_keys(object_name)
            for recorded in _resolved_positions.get(key, [])
        )


def _object_name_keys(object_name: str) -> Tuple[str, ...]:
    """Allow set_pose to use either a unique object name or the corresponding prim path."""
    text = str(object_name)
    basename = text.rstrip("/").split("/")[-1]
    return (text,) if basename == text else (text, basename)


def _placement_lifecycle_key(room_name: Optional[str] = None) -> Tuple[str, str]:
    pipe_id = _active_pipe_id
    with _placement_lifecycle_lock:
        active_room = _active_room_names.get(pipe_id, "")
    return pipe_id, str(room_name if room_name is not None else active_room)


def _record_spawned_unplaced(object_name: str) -> None:
    normalized = str(object_name or "").strip().rstrip("/").rsplit("/", 1)[-1]
    if not normalized:
        return
    key = _placement_lifecycle_key()
    with _placement_lifecycle_lock:
        _spawned_unplaced_objects.setdefault(key, set()).add(normalized)


def _mark_object_placed(object_name: str) -> None:
    names = {
        str(name).strip().rstrip("/").rsplit("/", 1)[-1]
        for name in _object_name_keys(object_name)
        if str(name).strip()
    }
    key = _placement_lifecycle_key()
    with _placement_lifecycle_lock:
        unresolved = _spawned_unplaced_objects.get(key)
        if unresolved is None:
            return
        unresolved.difference_update(names)
        if not unresolved:
            _spawned_unplaced_objects.pop(key, None)


def get_unresolved_spawned_objects(room_name: Optional[str] = None) -> List[str]:
    """Return objects spawned in a room without a successful set_pose."""
    key = _placement_lifecycle_key(room_name)
    with _placement_lifecycle_lock:
        return sorted(_spawned_unplaced_objects.get(key, set()))


def _reset_placement_lifecycle_state() -> None:
    """Reset process-local placement state. Intended for tests."""
    with _placement_lifecycle_lock:
        _active_room_names.clear()
        _spawned_unplaced_objects.clear()

def set_asset_name_usd_path_registry(asset_paths: Dict[str, str]) -> None:
    """Replace the process-local asset-name to USD-path registry."""
    normalized: Dict[str, str] = {}
    for raw_name, raw_path in asset_paths.items():
        asset_name = str(raw_name or "").strip()
        usd_path = str(raw_path or "").strip()
        if not asset_name or not usd_path:
            continue
        normalized[asset_name] = usd_path

    with _asset_name_usd_paths_lock:
        _asset_name_usd_paths.clear()
        _asset_name_usd_paths.update(normalized)


def _get_registered_usd_path(asset_name: str) -> Optional[str]:
    normalized_name = str(asset_name or "").strip()
    if not normalized_name:
        return None
    with _asset_name_usd_paths_lock:
        return _asset_name_usd_paths.get(normalized_name)


def _spawn_result_succeeded(result: object) -> bool:
    if result is None or result is False:
        return False
    text = str(result).strip()
    if not text:
        return False
    return not text.lower().startswith(
        ("[rejected]", "false", "none", "null", "error", "failed")
    )


@tool
def scan_scene() -> dict:
    '''scan the current scene and return a dictionary describing all existing objects in the scene
    
    Returns:
        A list of dictionaries, where each dictionary contains:
        {'name': str, 'bbox': {'min': [x,y,z], 'max': [x,y,z]}, 'position': [x,y,z]}.    
    '''
    result = _call_isaac("scan_scene")
    # print(result)
    if result and result.startswith("True,"):
        # Format: True,count,[{obj1}, {obj2}, ...]
        parts = result.split(",", 2)
        if len(parts) >= 3:
            try:
                count = int(parts[1])
            except ValueError:
                print(f"Unexpected scan_scene response: {result}")
                return []
            if count == 0:
                return []
            
            # Use ast.literal_eval to parse the Python dict-formatted string
            import ast
            try:
                # parts[2] is a Python list string (using single quotes)
                result_list = ast.literal_eval(parts[2])
                return result_list
            except (ValueError, SyntaxError) as e:
                print(f"Failed to parse result: {e}")
                return []
    return []

@tool
def get_room_context() -> dict:
    '''Get a semantic description of the current room layout.
    
    MUST be called first before any placement. Returns structured information about:
    - Room bounds and dimensions
    - Wall names and which side they are on (left/right/front/back)
    - Available wall and ceiling names under mounting_surfaces
    - Corner names and their wall associations
    - Already placed objects with sizes and spatial relationships
    - Available surfaces (tables, counters) with occupancy info
    
    Returns:
        dict with keys: room, walls, mounting_surfaces, corners, doors,
        placed_objects, available_surfaces
    '''
    result = _call_isaac("get_room_context")
    if not result:
        return {"error": "no response from Isaac Sim"}
    try:
        import json
        parsed = json.loads(result)
        return parsed
    except (json.JSONDecodeError, ValueError):
        return {"error": f"failed to parse response: {result[:200]}"}

@tool
def query_floor_space(near_wall: str = None, region: str = None) -> dict:
    '''Query available floor space in the current room.
    
    Use this to check which floor regions are occupied or free before placing
    a floor object. Optionally filter by proximity to a wall or by region type.
    
    Args:
        near_wall: Optional wall name (e.g., "wall_0") to focus on the strip near that wall.
        region: Optional region type: "center" for open center area, omit for full floor.
    
    Returns:
        dict with keys: room_bounds, occupied_regions, free_regions
    '''
    cmd_parts = ["query_floor_space"]
    if near_wall:
        cmd_parts.append(f"near_wall={near_wall}")
    if region:
        cmd_parts.append(f"region={region}")
    cmd = ",".join(cmd_parts)
    result = _call_isaac(cmd)
    if not result:
        return {"error": "no response from Isaac Sim"}
    try:
        import json
        parsed = json.loads(result)
        return parsed
    except (json.JSONDecodeError, ValueError):
        return {"error": f"failed to parse response: {result[:200]}"}

@tool
def query_surface_status(support_object: str, surface: str = "up") -> dict:
    '''Query the occupancy status of a surface on a support object.
    
    Use this before placing objects on a surface (table, counter, shelf).
    Returns the surface dimensions, objects already on it, and remaining free area.
    
    Args:
        support_object: Name of the support object (e.g., "Table_01", "Counter")
        surface: Which face to check (default "up"). One of: up, down, left, right, front, back.
    
    Returns:
        dict with keys: support_object, surface, surface_center, surface_size,
        surface_area_m2, objects_on_surface, occupied_area_m2, remaining_area_m2, is_empty
    '''
    cmd = f"query_surface_status,{support_object},{surface}"
    result = _call_isaac(cmd)
    if not result:
        return {"error": "no response from Isaac Sim"}
    try:
        import json
        parsed = json.loads(result)
        return parsed
    except (json.JSONDecodeError, ValueError):
        return {"error": f"failed to parse response: {result[:200]}"}

@tool
def get_object_bounds(object_name: str) -> Tuple[float, float, float]:
    '''Get bounding box size of an object that has already been spawned into the scene.

    Args:
        object_name: unique name of the spawned object (e.g., "Fridge")
    Returns:
        Tuple[float, float, float]: [length (x), width (y), height (z)] in meters
    '''

    result = _call_isaac(f"get_object_bbox,{object_name}")
    if result and result.startswith("True,"):
        # Format: True,min_x,min_y,min_z,max_x,max_y,max_z
        parts = result.split(",")
        if len(parts) >= 7:
            try:
                min_x, min_y, min_z = float(parts[1]), float(parts[2]), float(parts[3])
                max_x, max_y, max_z = float(parts[4]), float(parts[5]), float(parts[6])
                
                # Calculate dimensions (edge lengths)
                length = round(max_x - min_x, 2)  # X dimension
                width = round(max_y - min_y, 2)   # Y dimension
                height = round(max_z - min_z, 2)  # Z dimension
                
                return (length, width, height)
            except ValueError:
                pass
    return (0.0, 0.0, 0.0)

@tool
def spawn_object(asset_name: str, position: Tuple[float, float, float] | list | str = (0.0, 0.0, 0.0), rotation: Tuple[float, float, float] | list | str = (0.0, 0.0, 0.0)) -> str:
    '''Spawn an object in the scene.

    Args:
        asset_name: exact asset name from the current room's Available Assets list.
            It is used both to resolve the USD path and as the scene object name.
        position: (x, y, z) target position
        rotation: (x, y, z) Euler angles in degrees

    Returns:
        str: The spawned object name for subsequent tool calls.
    '''
    usd_path = _get_registered_usd_path(asset_name)
    if usd_path is None:
        return (
            f"[REJECTED] Unknown asset_name '{asset_name}'. "
            "Use the exact Asset Name shown in the current room's Available Assets list."
        )

    position = _normalize_vec3(position, "position")
    rotation = _normalize_vec3(rotation, "rotation")
    pos_x, pos_y, pos_z = position
    rot_x, rot_y, rot_z = rotation
    cmd = f"spawn_object,{asset_name},{usd_path},{pos_x},{pos_y},{pos_z},{rot_x},{rot_y},{rot_z}"
    spawned_obj_name = _call_isaac(cmd)
    if _spawn_result_succeeded(spawned_obj_name):
        _clear_resolved_positions(str(spawned_obj_name))
        _record_spawned_unplaced(str(spawned_obj_name))
    return spawned_obj_name

@tool
def set_pose(object_name: str, position: Tuple[float, float, float] | list | str | None = None, rotation: Tuple[float, float, float] | list | str | None = None):
    '''Set position and/or rotation of an existing object.

    Args:
        object_name: unique name of the spawned object
        position: optional absolute world-space bounding-box center (x, y, z)
            returned by a successful resolve_placement_intent call; None keeps
            the current bounding-box center
        rotation: optional absolute (x, y, z) Euler angles in degrees;
            None keeps the current rotation

    Returns:
        bool: True if successful
    '''
    # Handle optional parameters
    if position is None:
        pos_x = pos_y = pos_z = "None"
    else:
        position = _normalize_vec3(position, "position")
        if not _position_was_resolved(object_name, position):
            return (
                f"[REJECTED] set_pose rejected raw coordinates for {object_name}. "
                "Call resolve_placement_intent successfully and pass its returned position to set_pose. "
                "If placement failed, revise the semantic intent and resolve again instead of inventing coordinates."
            )
        pos_x, pos_y, pos_z = position
        pos_x, pos_y, pos_z = str(pos_x), str(pos_y), str(pos_z)

    if rotation is None:
        rot_x = rot_y = rot_z = "None"
    else:
        rotation = _normalize_vec3(rotation, "rotation")
        rot_x, rot_y, rot_z = rotation
        rot_x, rot_y, rot_z = str(rot_x), str(rot_y), str(rot_z)

    cmd = f"set_object_pose,{object_name},{pos_x},{pos_y},{pos_z},{rot_x},{rot_y},{rot_z}"
    result = _call_isaac(cmd)
    success = result == "True" or result is True
    if success:
        _mark_object_placed(object_name)
    return success

@tool
def scale_object(object_name: str, scale_factor: float) -> dict:
    '''Uniformly scale an object as a last-resort placement retry.

    Args:
        object_name: unique name of the spawned object
        scale_factor: one positive scalar applied equally to X, Y, and Z.
            It must be in the inclusive range [0.7, 1.3]. Values below 1
            shrink; values above 1 enlarge.

    Returns:
        dict: success, scale_factor, original_bbox, before_bbox, and freshly
            measured after_bbox. Re-run resolve_placement_intent after success.
    '''
    try:
        factor = float(scale_factor)
    except (TypeError, ValueError):
        return {
            "success": False,
            "rejected": True,
            "message": "scale_factor must be a positive finite scalar.",
        }
    if (
        not math.isfinite(factor)
        or factor < _MIN_UNIFORM_SCALE_FACTOR
        or factor > _MAX_UNIFORM_SCALE_FACTOR
    ):
        return {
            "success": False,
            "rejected": True,
            "message": (
                "scale_factor must be finite and within "
                f"[{_MIN_UNIFORM_SCALE_FACTOR}, {_MAX_UNIFORM_SCALE_FACTOR}]."
            ),
        }

    try:
        raw_result = _call_isaac(
            f"scale_asset_uniform,{object_name},{factor}"
        )
        if isinstance(raw_result, dict):
            parsed = raw_result
        else:
            parsed = ast.literal_eval(str(raw_result))
        if not isinstance(parsed, dict):
            parsed = {
                "success": False,
                "message": f"unexpected scale response: {raw_result}",
            }
        if parsed.get("success") is True:
            _clear_resolved_positions(object_name)
        parsed.setdefault("scale_factor", factor)
        return parsed
    except Exception as exc:
        return {
            "success": False,
            "message": f"failed to parse or execute uniform scale: {exc}",
            "scale_factor": factor,
        }


@tool
def resolve_placement_intent(intent: dict) -> dict:
    '''Convert high-level placement intent to concrete 3D coordinates.
    
    This function calculates the exact position where an object should be placed based on
    spatial relationships with other objects or surfaces in the scene. It supports multiple
    placement strategies controlled by the "placement_type" field.

    Args:
        intent: A dictionary describing the placement strategy. Required fields vary by placement_type:
        
            **Common Required Fields:**
            - "object_name" (str): Unique name of the object to be placed
            - "placement_type" (str): Type of placement strategy
              ("floor", "surface", or "wall_mounted")
            - "orientation" (optional): Desired front-facing direction / rotation.
                Supported forms:
                    - string axis direction: "+x"|"-x"|"+y"|"-y" (aliases: "x+"|"x-"|"y+"|"y-")
                    - dict with yaw: {"yaw_degrees": 90}
                    - dict with facing: {"front_facing": "+y"}
                    - full euler tuple/list: (rx, ry, rz)
            
            **Type-Specific Fields:**
            
            When placement_type == "floor":
            All three fields below are required:
            - "x_direction" (tuple): (reference_object_name, distance) - offset in X axis (+ right, - left)
            - "y_direction" (tuple): (reference_object_name, distance) - offset in Y axis (+ front, - back)
            - "z_direction" (tuple): (reference_object_name, distance) - offset in Z axis (+ up, - down)
            
            When placement_type == "surface":
            - "support_object" (str): Unique name of the supporting surface object
            - "surface" (str): Which face to place on ("down"|"up"|"left"|"right"|"back"|"front")
            - "offset_from_center" (tuple): (x_offset, y_offset) in local surface coordinates
            - Resolved positions include a 0.02 m clearance along the surface's
              outward normal to reduce support-object interpenetration.
            - "semantic_location" (dict): Optional language-level placement descriptor for surface placement.
              Supported fields:
                - "location": semantic anchor such as "center", "left", "right", "front", "back",
                  "left_front", "right_front", "left_back", "right_back"
                - "reference_object" / "relative_to": another object already on the same support surface
                - "relation": "next_to"|"left_of"|"right_of"|"in_front_of"|"behind"|"above"|"below"
                - "direction": required when relation == "next_to", e.g. "left"|"right"|"front"|"back"
                - "distance": optional clearance in meters
                - "margin": optional safety margin to support boundary in meters

            When placement_type == "wall_mounted":
            - "support_object" (str): A room wall name such as "wall_0", or "ceiling".
              The resolver automatically selects the face pointing into the room.
            - "mounted_location" (dict): Required semantic description of the
              mounted position.
                - "location": For walls use "center", "left", "right", "upper",
                  "lower", or combinations such as "upper_center" and
                  "upper_left". For ceilings use "center", "left", "right",
                  "front", "back", or combinations such as "front_right".
                - "description": Required natural-language explanation of the
                  intended position on the selected wall or ceiling.
                - "margin": Optional clearance from mounting-surface edges.
              Successful wall_mounted resolution disables articulation mode,
              makes all rigid bodies kinematic, and disables gravity while
              preserving collision behavior. The mounted object can therefore
              support dynamic objects without being displaced by them.
              Wall-mounted assets named "picture" or "picture_*" use a fixed
              asset convention: local +Z faces into the room and local +Y
              points upward. This overrides a supplied orientation.

    Returns:
        dict: Standardized result dictionary with fields:
            - success (bool): True if a valid placement was resolved.
            - position (tuple[float, float, float] | None): Absolute world-space
              bounding-box center (x, y, z) when success is True. Pass it to
              set_pose unchanged.
            - rotation (tuple[float, float, float] | None): (rx, ry, rz) euler rotation in degrees.
              Returns None when orientation was not provided in intent.
            - out_of_bounds (dict): Axis-wise overflow information when placement is invalid or exceeds bounds.
                Example: {'x': 0.0, 'y': 0.12, 'z': 0.0}
                Positive values indicate overflow in +axis direction, negative values in -axis direction, 0.0 means within bounds.
            - message (str): Optional diagnostic message.
            - gravity_disabled (bool): Present and True for successful
              wall_mounted placements.
            - kinematic_enabled (bool): Present and True for successful
              wall_mounted placements.
            - orientation_policy (str): Describes an asset-specific mounted
              orientation override when one was applied.

        The requested orientation is applied when calculating the object's
        world-aligned bounding-box extents, surface fit, wall clearance, and
        room-boundary containment.  Resolving an intent is only a coordinate
        proposal; call set_pose with the returned position/rotation to commit
        the change to the stage.
    '''
    # Send intent and parse standardized dict result sent back from isaac_sim_app
    object_name = intent.get("object_name") if isinstance(intent, dict) else None
    intent_str = str(intent)
    result = _call_isaac(f"resolve_placement_intent,{intent_str}")
    if not result:
        return {'success': False, 'position': None, 'rotation': None, 'out_of_bounds': {'x':0.0,'y':0.0,'z':0.0}, 'message': 'no response'}

    # Expect a string representation of a Python dict; parse it safely
    import ast
    try:
        parsed = ast.literal_eval(result)
        if isinstance(parsed, dict):
            if object_name and parsed.get('success') is True:
                resolved_position = _extract_position_from_result(parsed)
                if resolved_position is not None:
                    _record_resolved_position(object_name, resolved_position)
            return parsed
        else:
            return {'success': False, 'position': None, 'rotation': None, 'out_of_bounds': {'x':0.0,'y':0.0,'z':0.0}, 'message': 'unexpected response type'}
    except Exception as e:
        return {'success': False, 'position': None, 'rotation': None, 'out_of_bounds': {'x':0.0,'y':0.0,'z':0.0}, 'message': f'parse error: {e}'}

@tool
def create_room(rooms: list, house_length:int,connectivity: list):
    '''create a house spilted into multiple rooms

    args:
        rooms: names of the rooms to create (e.g., ["Living Room", "Kitchen", "Bedroom"])
        house_length: length of the entire house
        connectivity: list of connections between rooms
    '''
    room_info = _call_isaac(f"create_room,{rooms},{house_length},{connectivity}")
    # Parse the returned boundaries and connectivity info
    print(f"Received room info: {room_info}")
    if room_info:
        import ast
        parts = room_info.split("|")
        boundaries = parts[0]
        connections = parts[1] if len(parts) > 1 else "{}"
        room_rects = parts[2] if len(parts) > 2 else "[]"
        boundaries_list = ast.literal_eval(boundaries)
        connections_list = ast.literal_eval(connections)
        room_rects_list = ast.literal_eval(room_rects)
        return {"boundaries":boundaries_list,"connections":connections_list,"room_rects":room_rects_list}
    return None

@tool
def select_room(room_name: str):
    '''select a room to place objects into

    args:
        room_id: id of the room to select
    '''
    result = _call_isaac(f"select_room,{room_name}")
    success = result == "True" or result is True
    if success:
        with _placement_lifecycle_lock:
            _active_room_names[_active_pipe_id] = str(room_name)
    return success


def save_stage() -> bool:
    """Save the current USD stage inside the Isaac Sim runtime."""
    result = _call_isaac("save_stage")
    return result == "True" or result is True


@tool
def delete_object(object_name: str) -> bool:
    '''Delete an object from the scene by removing its prim.

    Use this to clean up objects that could not be successfully placed
    (e.g., objects still at the origin after placement failed).

    Args:
        object_name: unique name of the spawned object to delete

    Returns:
        bool: True if the object was successfully removed
    '''
    result = _call_isaac(f"delete_object,{object_name}")
    success = result == "True" or result is True
    if success:
        _mark_object_placed(object_name)
        _clear_resolved_positions(object_name)
    return success


if __name__ == "__main__":
    res = select_room.invoke({'room_name':'Kitchen'})

    res = create_room.invoke({
        "rooms": ['Kitchen', 'Dining Room','Living Room'],
        "house_length": 10,
        "connectivity": [(0,1),(1,2)]
    })
    pass

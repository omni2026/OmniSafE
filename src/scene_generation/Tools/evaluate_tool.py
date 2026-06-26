import json
from pathlib import Path
from typing import Optional

from langchain.tools import tool

try:
    from .place_tool import (
        _get_pipe,
        _get_request_lock,
        get_pipe_id,
        resolve_placement_intent as raw_resolve_placement_intent,
        scan_scene as raw_scan_scene,
    )
except ImportError:
    import sys

    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    from Tools.place_tool import (
        _get_pipe,
        _get_request_lock,
        get_pipe_id,
        resolve_placement_intent as raw_resolve_placement_intent,
        scan_scene as raw_scan_scene,
    )


def _request_isaac(command: str, timeout: int = 10):
    """Send one command to Isaac Sim and wait for the matching response.

    The evaluator needs longer waits for snapshot/collision commands,
    so we reimplement the locked request/response here against the same pipe
    primitives and forward ``timeout`` to ``recv_from_is``.
    """
    pipe_id = get_pipe_id()
    pipe = _get_pipe(pipe_id)
    lock = _get_request_lock(pipe_id)
    with lock:
        pipe.send_to_is(command)
        return pipe.recv_from_is(timeout=timeout)


def _parse_response(result) -> dict:
    if isinstance(result, dict):
        return result
    if not result:
        return {"success": False, "message": "no response from Isaac Sim"}
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"success": False, "message": result}
    return {"success": False, "message": f"unexpected response: {result!r}"}


def _normalize_suspect_objects(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in text.split(",") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


@tool
def analyze_placement_log(
    trace_file: str,
    room_name: Optional[str] = None,
    include_structure: bool = False,
) -> dict:
    """Extract final placed object states from a placement trace JSON file.

    Args:
        trace_file: Path to a placement_trace-style JSON file.
        room_name: Optional room name used to filter objects and fill missing prim paths.
        include_structure: Whether to keep structural objects such as walls and floors.

    Returns:
        A dict with objects extracted from the placement log.
    """
    payload = {
        "trace_file": trace_file,
        "room_name": room_name,
        "include_structure": include_structure,
    }
    command = f"evaluate_analyze_placement_log,{json.dumps(payload, ensure_ascii=False)}"
    result = _request_isaac(command, timeout=30)
    return _parse_response(result)


@tool
def check_scene_collisions(room_name: str) -> dict:
    """Check collisions for all existing non-structural objects in a room."""
    command = f"evaluate_check_scene_collisions,{room_name}"
    result = _request_isaac(command, timeout=60)
    return _parse_response(result)


@tool
def summarize_room_layout(room_name: str) -> dict:
    """Collect compact room-level geometry, density, boundary, and door-clearance evidence."""
    command = f"evaluate_summarize_room_layout,{room_name}"
    result = _request_isaac(command, timeout=60)
    return _parse_response(result)


@tool
def scan_scene() -> list | dict:
    """Scan the current scene state for evaluator-side inspection."""
    return raw_scan_scene.invoke({})


@tool
def resolve_placement_intent(intent: dict, room_name: Optional[str] = None) -> dict:
    """Resolve a repair placement intent and stage the pose temporarily for evaluator planning."""
    resolved = raw_resolve_placement_intent.invoke({"intent": intent})
    result = _parse_response(resolved)
    if not result.get("success"):
        return result

    payload = {
        "room_name": room_name,
        "object_name": intent.get("object_name"),
        "prim_path": intent.get("prim_path"),
        "position": result.get("position"),
        "rotation": result.get("rotation"),
    }
    stage_result = _parse_response(
        _request_isaac(f"evaluate_stage_trial_pose,{json.dumps(payload, ensure_ascii=False)}", timeout=60)
    )
    if not stage_result.get("success"):
        result["success"] = False
        result["message"] = (
            f"{result.get('message', '')} failed to stage evaluator trial pose: "
            f"{stage_result.get('message', 'unknown error')}"
        ).strip()
    else:
        result["staged"] = True
    return result


def reset_trial_placements(room_name: str) -> dict:
    """Restore all temporary evaluator trial placements for one room."""
    result = _request_isaac(f"evaluate_reset_trial_placements,{room_name}", timeout=60)
    return _parse_response(result)


@tool
def capture_evaluation_snapshot(
    output_dir: str,
    room_name: str,
    object_name: Optional[str] = None,
) -> dict:
    """Capture one evaluator snapshot for a room or a target object.

    The final image filename is generated internally from the target object.
    """
    payload = {
        "output_dir": output_dir,
        "room_name": room_name,
        "object_name": object_name,
    }
    command = f"evaluate_capture_snapshot,{json.dumps(payload, ensure_ascii=False)}"
    result = _request_isaac(command, timeout=60)
    return _parse_response(result)


@tool
def capture_evaluation_views(
    room_name: str,
    output_dir: str,
    suspect_objects: Optional[list[str] | str] = None,
) -> dict:
    """Capture the standard evaluator views for a room and optional suspect objects."""
    payload = {
        "room_name": room_name,
        "output_dir": output_dir,
        "suspect_objects": _normalize_suspect_objects(suspect_objects),
    }
    command = f"evaluate_capture_views,{json.dumps(payload, ensure_ascii=False)}"
    result = _request_isaac(command, timeout=120)
    return _parse_response(result)

if __name__ == "__main__":
    #res = select_room.invoke({'room_name':'Garage'})
    #res = scan_scene.invoke({})
    #print(res)
    #res = record_scene_snapshot.invoke({"output_path": "./snapshot2.png","object_name":"wall_0"})
    #res = resolve_placement_intent.invoke({'intent': {'object_name': 'locker', 'placement_type': 'floor', "x_direction": ("wall_2", 0.5),"y_direction": ("wall_0", 4.0),"z_direction": ("GroundPlane", 0.0) }})
    # res = create_room.invoke({
    #     "rooms": ['Kitchen', 'Dining Room','Living Room', 'Bedroom'],
    #     "house_length": 10,
    #     "connectivity": [(0,1)]
    # })
    # res = scale_object.invoke({'object_name':"wall_0",'target_dimensions':(2.5369794355944415,0.938749080405033,0.9655172096279667)})
    res = check_scene_collisions.invoke({'room_name':'Garage'})
    # res = get_object_bounds.invoke({
    #     "object_name": "Booth_01"
    # })

    # res = set_pose.invoke({
    #     "object_name": "pomegranate",
    #     "position": (1.0, 2.0, 5.0),
    #     "rotation": (0.0, 45.0, 0.0)
    # })

    # res = scale_object.invoke({
    #     "object_name": "pomegranate", "target_dimensions": (5.0, 5.0, 5.0)
    # })

    print(res)
    pass

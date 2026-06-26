import ast
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import omni
    from pxr import Gf, PhysicsSchemaTools, Sdf, Usd, UsdGeom
except ImportError:
    omni = None
    Gf = None
    PhysicsSchemaTools = None
    Sdf = None
    Usd = None
    UsdGeom = None

try:
    from Tools.tool_implementation_isaac import get_object_bbox, set_object_pose
except ImportError:
    get_object_bbox = None
    set_object_pose = None

try:
    from Tools.asset_physics import (
        is_collision_evaluation_exempt_asset,
        is_nonphysical_floating_asset,
        is_nonphysical_floating_prim,
    )
except ImportError:
    from asset_physics import (
        is_collision_evaluation_exempt_asset,
        is_nonphysical_floating_asset,
        is_nonphysical_floating_prim,
    )


STRUCTURE_NAMES = {"floor", "ceiling", "door", "GroundPlane", "eval_cameras"}
STRUCTURE_PREFIXES = ("wall_",)
_EVALUATOR_TRIAL_PLACEMENTS: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _is_structure_name(name: str) -> bool:
    return name in STRUCTURE_NAMES or name.startswith(STRUCTURE_PREFIXES)


def _parse_maybe_structured_text(text: Any) -> Any:
    if not isinstance(text, str):
        return text

    stripped = text.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except Exception:
        pass

    try:
        return ast.literal_eval(stripped)
    except Exception:
        return stripped


def _normalize_vector3(value: Any) -> Optional[List[float]]:
    parsed = _parse_maybe_structured_text(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        return None

    try:
        return [float(parsed[0]), float(parsed[1]), float(parsed[2])]
    except (TypeError, ValueError):
        return None


def _message_role(message: Dict[str, Any]) -> Optional[str]:
    role = str(message.get("role") or message.get("type") or "").lower()
    if role in {"human", "user", "humanmessage"}:
        return "human"
    if role in {"ai", "assistant", "aimessage"}:
        return "ai"
    if role in {"tool", "toolmessage"}:
        return "tool"
    return None


def _parse_tool_args(args: Any) -> Dict[str, Any]:
    parsed = _parse_maybe_structured_text(args)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_tool_call(tool_call: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(tool_call, dict):
        return None

    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    name = tool_call.get("name") or tool_call.get("tool_name") or function.get("name")
    if not isinstance(name, str) or not name:
        return None

    args = (
        tool_call.get("args")
        if "args" in tool_call
        else tool_call.get("arguments")
        if "arguments" in tool_call
        else function.get("arguments")
    )
    return {
        "id": tool_call.get("id") or tool_call.get("tool_call_id"),
        "name": name,
        "args": _parse_tool_args(args),
    }


def _build_prim_path(room_name: Optional[str], object_name: str) -> Optional[str]:
    if not room_name:
        return None
    return f"/World/rooms/{room_name}/{object_name}"


def _ensure_log_object_entry(
    store: Dict[str, Dict[str, Any]],
    object_name: str,
    room_name: Optional[str],
) -> Dict[str, Any]:
    entry = store.setdefault(
        object_name,
        {
            "prim_path": _build_prim_path(room_name, object_name),
            "position": None,
            "rotation": None,
            "scale": None,
            "room": room_name,
            "placement_type": None,
            "support_object": None,
            "mounted_location": None,
            "asset_id": None,
            "asset_key": None,
            "usd_path": None,
            "source": None,
        },
    )
    if room_name and not entry.get("room"):
        entry["room"] = room_name
    if room_name and not entry.get("prim_path"):
        entry["prim_path"] = _build_prim_path(room_name, object_name)
    return entry


def _finalize_log_object_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    resolved_position = entry.pop("_resolved_position", None)
    resolved_rotation = entry.pop("_resolved_rotation", None)
    scan_position = entry.pop("_scan_position", None)
    scan_rotation = entry.pop("_scan_rotation", None)

    if entry.get("position") is None:
        if resolved_position is not None:
            entry["position"] = resolved_position
            if not entry.get("source"):
                entry["source"] = "resolve_placement_intent"
        elif scan_position is not None:
            entry["position"] = scan_position
            if not entry.get("source"):
                entry["source"] = "scan_scene"

    if entry.get("rotation") is None:
        if resolved_rotation is not None:
            entry["rotation"] = resolved_rotation
        elif scan_rotation is not None:
            entry["rotation"] = scan_rotation

    return entry


def _set_if_present(entry: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        entry[key] = value


def _remember_tool_call(
    tool_call: Dict[str, Any],
    pending_by_id: Dict[str, Dict[str, Any]],
    pending_by_name: Dict[str, List[Dict[str, Any]]],
) -> None:
    call_id = tool_call.get("id")
    if call_id is not None:
        pending_by_id[str(call_id)] = tool_call
    pending_by_name.setdefault(tool_call["name"], []).append(tool_call)


def _pop_tool_call(
    name: Optional[str],
    tool_call_id: Optional[Any],
    pending_by_id: Dict[str, Dict[str, Any]],
    pending_by_name: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    matched_call = None
    if tool_call_id is not None:
        matched_call = pending_by_id.pop(str(tool_call_id), None)

    if matched_call is None and name:
        same_name_calls = pending_by_name.get(name) or []
        while same_name_calls:
            candidate = same_name_calls.pop(0)
            candidate_id = candidate.get("id")
            if candidate_id is None or pending_by_id.pop(str(candidate_id), None) is not None:
                matched_call = candidate
                break

    if matched_call is not None:
        matched_id = matched_call.get("id")
        if matched_id is not None:
            pending_by_id.pop(str(matched_id), None)
        same_name_calls = pending_by_name.get(matched_call["name"]) or []
        pending_by_name[matched_call["name"]] = [
            call for call in same_name_calls if call is not matched_call
        ]
    return matched_call


def extract_final_positions(
    trace_payload: List[Dict[str, Any]],
    include_structure: bool = False,
    room_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    pending_by_id: Dict[str, Dict[str, Any]] = {}
    pending_by_name: Dict[str, List[Dict[str, Any]]] = {}
    latest_positions: Dict[str, Dict[str, Any]] = {}
    current_room: Optional[str] = room_name
    for event in trace_payload:
        if not isinstance(event, dict):
            continue

        if "messages" in event:
            messages = event.get("messages", [])
        elif "role" in event or "type" in event:
            messages = [event]
        else:
            messages = []

        if not isinstance(messages, list):
            continue

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = _message_role(message)
            name = message.get("name")
            content = message.get("content")

            if role == "ai":
                tool_calls = message.get("tool_calls") or []
                if isinstance(tool_calls, list):
                    for raw_tool_call in tool_calls:
                        tool_call = _normalize_tool_call(raw_tool_call)
                        if tool_call:
                            _remember_tool_call(tool_call, pending_by_id, pending_by_name)
                continue

            if role != "tool":
                continue

            matched_call = _pop_tool_call(
                name=name,
                tool_call_id=message.get("tool_call_id") or message.get("id"),
                pending_by_id=pending_by_id,
                pending_by_name=pending_by_name,
            )

            if name == "scan_scene":
                scanned_objects = _parse_maybe_structured_text(content)
                if isinstance(scanned_objects, list):
                    for obj in scanned_objects:
                        if not isinstance(obj, dict):
                            continue
                        obj_name = obj.get("name")
                        obj_position = obj.get("position")
                        if not isinstance(obj_name, str):
                            continue
                        if not include_structure and _is_structure_name(obj_name):
                            continue
                        normalized = _normalize_vector3(obj_position)
                        if normalized is None:
                            continue
                        entry = latest_positions.get(obj_name)
                        if entry is None:
                            # Do not create authoritative final objects from scan_scene alone.
                            # Mixed-room trajectories or stale scans can otherwise introduce false positives.
                            continue
                        entry["_scan_position"] = normalized
                        scan_rotation = _normalize_vector3(obj.get("rotation"))
                        if scan_rotation is not None:
                            entry["_scan_rotation"] = scan_rotation
                        _set_if_present(entry, "scale", _normalize_vector3(obj.get("scale")))
                continue

            if matched_call is None:
                continue

            args = matched_call.get("args") or {}
            intent_args = args.get("intent") if isinstance(args.get("intent"), dict) else None

            if name == "resolve_placement_intent":
                intent = intent_args or {}
                object_name = intent.get("object_name")
            else:
                object_name = args.get("object_name")

            if not isinstance(object_name, str):
                continue
            if not include_structure and _is_structure_name(object_name):
                continue

            entry = _ensure_log_object_entry(latest_positions, object_name, current_room)

            if name == "spawn_object":
                _set_if_present(entry, "asset_id", args.get("asset_id"))
                _set_if_present(entry, "asset_key", args.get("asset_key"))
                _set_if_present(entry, "usd_path", args.get("usd_path"))
                normalized = _normalize_vector3(args.get("position"))
                if normalized is not None:
                    if entry.get("position") is None:
                        entry["position"] = normalized
                    if not entry.get("source"):
                        entry["source"] = "spawn_object"
                rotation = _normalize_vector3(args.get("rotation"))
                if rotation is not None and entry.get("rotation") is None:
                    entry["rotation"] = rotation
                continue

            if name == "set_pose":
                succeeded = str(content).strip().lower() in {"true", "1", "success"}
                normalized = _normalize_vector3(args.get("position"))
                rotation = _normalize_vector3(args.get("rotation"))
                if succeeded:
                    if normalized is not None:
                        entry["position"] = normalized
                    if rotation is not None:
                        entry["rotation"] = rotation
                    if normalized is not None or rotation is not None:
                        entry["source"] = "set_pose"
                continue

            if name == "scale_object":
                succeeded = str(content).strip().lower() in {"true", "1", "success"}
                scale = _normalize_vector3(args.get("scale") or args.get("target_dimensions"))
                if succeeded and scale is not None:
                    entry["scale"] = scale
                continue

            if name == "resolve_placement_intent":
                intent = intent_args or {}
                parsed_result = _parse_maybe_structured_text(content)
                if isinstance(parsed_result, dict):
                    if parsed_result.get("success") and isinstance(intent, dict):
                        placement_type = intent.get("placement_type")
                        if isinstance(placement_type, str):
                            entry["placement_type"] = placement_type
                        support_object = intent.get("support_object")
                        if isinstance(support_object, str):
                            entry["support_object"] = support_object
                        surface = intent.get("surface")
                        if surface is not None:
                            entry["surface"] = surface
                        orientation = intent.get("orientation")
                        if orientation is not None:
                            entry["orientation"] = orientation
                        semantic_location = intent.get("semantic_location")
                        if semantic_location is not None:
                            entry["semantic_location"] = semantic_location
                        mounted_location = intent.get("mounted_location")
                        if mounted_location is not None:
                            entry["mounted_location"] = mounted_location
                        mounted_surface = parsed_result.get("mounted_surface")
                        if mounted_surface is not None:
                            entry["surface"] = mounted_surface
                        if parsed_result.get("gravity_disabled") is not None:
                            entry["gravity_disabled"] = bool(
                                parsed_result.get("gravity_disabled")
                            )
                        if parsed_result.get("kinematic_enabled") is not None:
                            entry["kinematic_enabled"] = bool(
                                parsed_result.get("kinematic_enabled")
                            )
                        resolved_position = _normalize_vector3(parsed_result.get("position"))
                        if resolved_position is not None:
                            entry["_resolved_position"] = resolved_position
                            if entry.get("source") in {None, "spawn_object", "scan_scene"}:
                                entry["position"] = resolved_position
                                entry["source"] = "resolve_placement_intent"
                        resolved_rotation = _normalize_vector3(parsed_result.get("rotation"))
                        if resolved_rotation is not None:
                            entry["_resolved_rotation"] = resolved_rotation
                            if entry.get("source") in {None, "spawn_object", "scan_scene"}:
                                entry["rotation"] = resolved_rotation

    finalized = {
        object_name: _finalize_log_object_entry(entry)
        for object_name, entry in latest_positions.items()
    }
    return dict(sorted(finalized.items(), key=lambda item: item[0]))


def _object_name_from_path(path: str) -> str:
    return str(path).rstrip("/").rsplit("/", 1)[-1]


def _normalize_name(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str) or not name:
        return None
    text = name.rstrip("/")
    if "/base_link" in text:
        text = text.split("/base_link", 1)[0]
    return text.rsplit("/", 1)[-1].strip().lower()


def _collision_evaluation_exempt_from_path(path_or_name: str) -> bool:
    normalized = _normalize_name(path_or_name)
    return is_collision_evaluation_exempt_asset(normalized or path_or_name)


def _related_object_from_path(path: str, room_name: str) -> Optional[str]:
    prefix = f"/World/rooms/{room_name}/"
    text = str(path)
    if not text.startswith(prefix):
        return None
    rest = text[len(prefix):]
    return rest.split("/", 1)[0] if rest else None


def _iter_room_object_paths(room_name: str) -> list[str]:
    stage = omni.usd.get_context().get_stage()
    room_path = f"/World/rooms/{room_name}"
    room_prim = stage.GetPrimAtPath(room_path)
    if not room_prim or not room_prim.IsValid():
        return []

    paths = []
    for child_prim in room_prim.GetChildren():
        name = child_prim.GetName()
        if child_prim.GetTypeName() == "Scope":
            continue
        if _is_structure_name(name):
            continue
        if (
            is_nonphysical_floating_asset(name)
            or is_nonphysical_floating_prim(child_prim)
        ):
            continue
        paths.append(str(child_prim.GetPath()))
    return paths


def _room_object_targets(room_name: str) -> list[tuple[str, str]]:
    seen = set()
    targets = []
    for prim_path in _iter_room_object_paths(room_name):
        if prim_path in seen:
            continue
        seen.add(prim_path)
        targets.append((_object_name_from_path(prim_path), prim_path))
    return targets


def analyze_placement_log(
    trace_file: str,
    room_name: Optional[str] = None,
    include_structure: bool = False,
) -> dict:
    try:
        trace_path = Path(trace_file).expanduser().resolve()
        if not trace_path.exists():
            return {
                "success": False,
                "trace_file": str(trace_path),
                "room_name": room_name,
                "objects": [],
                "object_count": 0,
                "message": "trace file not found",
            }

        with trace_path.open("r", encoding="utf-8") as f:
            trace_payload = json.load(f)
        if not isinstance(trace_payload, list):
            return {
                "success": False,
                "trace_file": str(trace_path),
                "room_name": room_name,
                "objects": [],
                "object_count": 0,
                "message": "trace file must contain a top-level list",
            }

        extracted = extract_final_positions(
            trace_payload,
            include_structure=include_structure,
            room_name=room_name,
        )
        objects = []
        for object_name, state in extracted.items():
            if room_name and state.get("room") and state.get("room") != room_name:
                continue
            object_state = dict(state)
            object_state["name"] = object_name
            if room_name and not object_state.get("room"):
                object_state["room"] = room_name
            if room_name and not object_state.get("prim_path"):
                object_state["prim_path"] = f"/World/rooms/{room_name}/{object_name}"
            objects.append(object_state)

        return {
            "success": True,
            "trace_file": str(trace_path),
            "room_name": room_name,
            "objects": objects,
            "object_count": len(objects),
            "message": "",
        }
    except Exception as e:
        return {
            "success": False,
            "trace_file": trace_file,
            "room_name": room_name,
            "objects": [],
            "object_count": 0,
            "message": f"failed to analyze placement log: {e}",
        }


def _sync_physx_scene(frames: int = 2) -> None:
    physx_interface = omni.physx.get_physx_interface()
    try:
        physx_interface.force_load_physics_from_usd()
    except Exception:
        pass

    dt = 1.0 / 60.0
    for _ in range(frames):
        physx_interface.update_simulation(dt, time.time())


def _resolve_room_prim_path(room_name: str, object_name: Optional[str] = None, prim_path: Optional[str] = None) -> str:
    if prim_path:
        return prim_path if prim_path.startswith("/World") else f"/World/{prim_path.lstrip('/')}"
    if not room_name or not object_name:
        raise ValueError("room_name and object_name are required when prim_path is omitted")
    return object_name if object_name.startswith("/World") else f"/World/rooms/{room_name}/{object_name}"


def _capture_transform_state(prim_path: str) -> List[Dict[str, Any]]:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    state: List[Dict[str, Any]] = []
    for op in xformable.GetOrderedXformOps():
        state.append({
            "name": op.GetName(),
            "value": op.Get(),
        })
    return state


def _restore_transform_state(prim_path: str, transform_state: List[Dict[str, Any]]) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"prim not found during restore: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    ops_by_name = {op.GetName(): op for op in xformable.GetOrderedXformOps()}
    for entry in transform_state:
        op_name = entry.get("name")
        if op_name in ops_by_name:
            ops_by_name[op_name].Set(entry.get("value"))


def stage_trial_resolved_pose(
    room_name: str,
    object_name: Optional[str] = None,
    position: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    prim_path: Optional[str] = None,
) -> dict:
    """Temporarily apply a resolved pose during evaluator repair planning."""
    try:
        if not room_name:
            return {
                "success": False,
                "room_name": room_name,
                "object_name": object_name,
                "prim_path": prim_path,
                "message": "room_name is required",
            }
        if position is None:
            return {
                "success": False,
                "room_name": room_name,
                "object_name": object_name,
                "prim_path": prim_path,
                "message": "position is required",
            }

        target_prim_path = _resolve_room_prim_path(room_name, object_name=object_name, prim_path=prim_path)
        room_store = _EVALUATOR_TRIAL_PLACEMENTS.setdefault(room_name, {})
        if target_prim_path not in room_store:
            room_store[target_prim_path] = {
                "object_name": _object_name_from_path(target_prim_path),
                "original_transform_state": _capture_transform_state(target_prim_path),
            }

        target_position = tuple(float(v) for v in position)
        target_rotation = tuple(float(v) for v in rotation) if rotation is not None else None
        if not set_object_pose(target_prim_path, target_position, target_rotation):
            raise RuntimeError(f"failed to set pose for {target_prim_path}")
        _sync_physx_scene()

        room_store[target_prim_path]["staged_position"] = list(target_position)
        room_store[target_prim_path]["staged_rotation"] = list(target_rotation) if target_rotation is not None else None
        return {
            "success": True,
            "room_name": room_name,
            "object_name": room_store[target_prim_path]["object_name"],
            "prim_path": target_prim_path,
            "position": list(target_position),
            "rotation": list(target_rotation) if target_rotation is not None else None,
            "message": "",
        }
    except Exception as e:
        return {
            "success": False,
            "room_name": room_name,
            "object_name": object_name,
            "prim_path": prim_path,
            "message": f"failed to stage trial pose: {e}",
        }


def reset_trial_placements(room_name: str) -> dict:
    """Restore all evaluator trial placements for one room."""
    room_store = _EVALUATOR_TRIAL_PLACEMENTS.get(room_name) or {}
    restored_objects: List[str] = []
    errors: List[dict] = []

    for prim_path, entry in reversed(list(room_store.items())):
        try:
            _restore_transform_state(prim_path, entry.get("original_transform_state") or [])
            _sync_physx_scene()
            restored_objects.append(entry.get("object_name") or _object_name_from_path(prim_path))
        except Exception as e:
            errors.append(
                {
                    "prim_path": prim_path,
                    "object_name": entry.get("object_name") or _object_name_from_path(prim_path),
                    "message": str(e),
                }
            )

    _EVALUATOR_TRIAL_PLACEMENTS.pop(room_name, None)
    return {
        "success": not errors,
        "room_name": room_name,
        "restored_objects": restored_objects,
        "restored_count": len(restored_objects),
        "errors": errors,
        "message": "" if not errors else f"failed to restore {len(errors)} trial placement(s)",
    }


def _bbox_intersection_3d(bbox_a, bbox_b) -> Optional[dict]:
    min_a, max_a = bbox_a
    min_b, max_b = bbox_b
    overlap_min = [
        max(float(min_a[i]), float(min_b[i]))
        for i in range(3)
    ]
    overlap_max = [
        min(float(max_a[i]), float(max_b[i]))
        for i in range(3)
    ]
    overlap_size = [
        overlap_max[i] - overlap_min[i]
        for i in range(3)
    ]
    if any(size <= 0.0 for size in overlap_size):
        return None
    return {
        "min": overlap_min,
        "max": overlap_max,
        "size": overlap_size,
        "volume": overlap_size[0] * overlap_size[1] * overlap_size[2],
    }


def _is_tolerable_wall_or_floor_contact(
    hit_path: str,
    target_bbox,
    contact_tolerance: float = 0.1,
) -> bool:
    hit_name = _object_name_from_path(hit_path).lower()
    hit_bbox = get_object_bbox(hit_path)
    if hit_bbox is None:
        return False
    overlap = _bbox_intersection_3d(target_bbox, hit_bbox)
    if overlap is None:
        return False
    min_overlap_size = min(overlap["size"])
    return min_overlap_size <= contact_tolerance


def _collision_check_current_pose(
    target_prim_path: str,
    white_list: Optional[list[str]] = None,
) -> list[str]:
    """Check overlaps for the current prim pose without modifying the stage."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(target_prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"prim not found: {target_prim_path}")
    if (
        _collision_evaluation_exempt_from_path(target_prim_path)
        or is_nonphysical_floating_prim(prim)
    ):
        return []

    visuals_path = target_prim_path + "/base_link/visuals"
    visuals_prim = stage.GetPrimAtPath(visuals_path)
    query_path = visuals_path if visuals_prim and visuals_prim.IsValid() else target_prim_path
    path_encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(query_path))
    target_bbox = get_object_bbox(target_prim_path)
    if target_bbox is None:
        raise ValueError(f"failed to compute bbox for: {target_prim_path}")

    overlapping_paths = []
    filters = white_list or []

    def report_hit(hit):
        hit_path = str(hit.rigid_body)
        hit_object_root = hit_path.split("/base_link", 1)[0]
        hit_root_prim = stage.GetPrimAtPath(hit_object_root)
        if (
            _collision_evaluation_exempt_from_path(hit_path)
            or is_nonphysical_floating_asset(_normalize_name(hit_path) or "")
            or is_nonphysical_floating_prim(hit_root_prim)
        ):
            return True
        if hit_path == target_prim_path or hit_path.startswith(target_prim_path + "/"):
            return True
        if any(white_path in hit_path for white_path in filters):
            return True
        if _is_tolerable_wall_or_floor_contact(hit_path, target_bbox):
            return True
        if hit_path not in overlapping_paths:
            overlapping_paths.append(hit_path)
        return True

    physx_query_interface = omni.physx.get_physx_scene_query_interface()
    physx_query_interface.overlap_mesh(path_encoded[0], path_encoded[1], report_hit, False)
    return overlapping_paths


def _raw_scene_collisions(room_name: str) -> dict:
    try:
        if not room_name:
            return {
                "success": False,
                "room_name": room_name,
                "objects_checked": 0,
                "collisions": [],
                "suspect_objects": [],
                "message": "room_name is required",
            }

        stage = omni.usd.get_context().get_stage()
        room_path = f"/World/rooms/{room_name}"
        room_prim = stage.GetPrimAtPath(room_path)
        if not room_prim or not room_prim.IsValid():
            return {
                "success": False,
                "room_name": room_name,
                "objects_checked": 0,
                "collisions": [],
                "suspect_objects": [],
                "message": f"room prim not found: {room_path}",
            }

        targets = _room_object_targets(room_name)
        _sync_physx_scene()
        collisions = []
        suspect_objects = set()
        checked = 0
        default_white_list = ["/eval_cameras/","GroundPlane"]

        for object_name, prim_path in targets:
            try:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    collisions.append({
                        "object": object_name,
                        "prim_path": prim_path,
                        "overlaps": [],
                        "related_objects": [],
                        "severity": "medium",
                        "error": "prim not found",
                    })
                    suspect_objects.add(object_name)
                    continue

                checked += 1
                white_list = default_white_list + [prim_path, prim_path + "/base_link/visuals"]
                overlaps = _collision_check_current_pose(prim_path, white_list=white_list)
                if not overlaps:
                    continue

                related = []
                filtered_overlaps = []
                for hit_path in overlaps:
                    related_name = _related_object_from_path(hit_path, room_name)
                    if _collision_evaluation_exempt_from_path(related_name or hit_path):
                        continue
                    filtered_overlaps.append(hit_path)
                    if related_name and related_name != object_name and not _is_structure_name(related_name):
                        related.append(related_name)
                        suspect_objects.add(related_name)
                if not filtered_overlaps:
                    continue
                suspect_objects.add(object_name)
                collisions.append({
                    "object": object_name,
                    "prim_path": prim_path,
                    "overlaps": filtered_overlaps,
                    "related_objects": sorted(set(related)),
                    "severity": "high",
                })
            except Exception as e:
                collisions.append({
                    "object": object_name,
                    "prim_path": prim_path,
                    "overlaps": [],
                    "related_objects": [],
                    "severity": "medium",
                    "error": str(e),
                })
                suspect_objects.add(object_name)

        return {
            "success": True,
            "room_name": room_name,
            "objects_checked": checked,
            "collisions": collisions,
            "suspect_objects": sorted(suspect_objects),
            "message": "" if not collisions else f"found {len(collisions)} collision issue(s)",
        }
    except Exception as e:
        print(f"Error checking evaluator collisions: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return {
            "success": False,
            "room_name": room_name,
            "objects_checked": 0,
            "collisions": [],
            "suspect_objects": [],
            "message": f"failed to check collisions: {e}",
        }


def check_scene_collisions(room_name: str) -> dict:
    """Check collisions for all existing non-structural objects in a room."""
    return _raw_scene_collisions(room_name)


def _load_evaluation_boundary_segments(room_prim) -> list[list[list[float]]]:
    serialized_segments = room_prim.GetCustomDataByKey("room_boundary_segments")
    if not serialized_segments:
        return []
    try:
        segments = (
            json.loads(serialized_segments)
            if isinstance(serialized_segments, str)
            else serialized_segments
        )
    except Exception:
        return []

    normalized = []
    for segment in segments or []:
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            continue
        try:
            start = [float(segment[0][0]), float(segment[0][1])]
            end = [float(segment[1][0]), float(segment[1][1])]
        except (TypeError, ValueError, IndexError):
            continue
        if start != end:
            normalized.append([start, end])
    return normalized


def _point_on_evaluation_segment(point, start, end, eps: float = 1e-6) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (
        min(x1, x2) - eps <= px <= max(x1, x2) + eps
        and min(y1, y2) - eps <= py <= max(y1, y2) + eps
    )


def _point_in_evaluation_room(point, segments, eps: float = 1e-6) -> bool:
    if not segments:
        return False
    for start, end in segments:
        if _point_on_evaluation_segment(point, start, end, eps=eps):
            return True

    x, y = point
    intersections = 0
    for (x1, y1), (x2, y2) in segments:
        if abs(y1 - y2) <= eps:
            continue
        if y < min(y1, y2) or y >= max(y1, y2):
            continue
        x_intersection = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        if x_intersection >= x - eps:
            intersections += 1
    return intersections % 2 == 1


def _bbox_xy_distance(left_bbox, right_bbox) -> float:
    left_min, left_max = left_bbox
    right_min, right_max = right_bbox
    gap_x = max(
        float(left_min[0]) - float(right_max[0]),
        float(right_min[0]) - float(left_max[0]),
        0.0,
    )
    gap_y = max(
        float(left_min[1]) - float(right_max[1]),
        float(right_min[1]) - float(left_max[1]),
        0.0,
    )
    return (gap_x * gap_x + gap_y * gap_y) ** 0.5


def _estimate_free_space_connectivity(
    room_min,
    room_max,
    segments,
    object_bounds,
    doors,
    grid_size: int = 24,
    clearance: float = 0.25,
) -> dict:
    width = max(float(room_max[0]) - float(room_min[0]), 1e-6)
    depth = max(float(room_max[1]) - float(room_min[1]), 1e-6)

    def cell_center(cell):
        x_index, y_index = cell
        return (
            float(room_min[0]) + (x_index + 0.5) * width / grid_size,
            float(room_min[1]) + (y_index + 0.5) * depth / grid_size,
        )

    def point_is_blocked(point):
        if segments and not _point_in_evaluation_room(point, segments):
            return True
        x, y = point
        for bbox_min, bbox_max in object_bounds.values():
            # Navigation occupancy only includes geometry intersecting the
            # robot-base height band. Floor coverings and wall-mounted objects
            # should not split the free-space graph.
            if (
                float(bbox_max[2]) <= float(room_min[2]) + 0.12
                or float(bbox_min[2]) >= float(room_min[2]) + 0.55
            ):
                continue
            if (
                float(bbox_min[0]) - clearance
                <= x
                <= float(bbox_max[0]) + clearance
                and float(bbox_min[1]) - clearance
                <= y
                <= float(bbox_max[1]) + clearance
            ):
                return True
        return False

    free_cells = {
        (x_index, y_index)
        for x_index in range(grid_size)
        for y_index in range(grid_size)
        if not point_is_blocked(cell_center((x_index, y_index)))
    }
    components = []
    unvisited = set(free_cells)
    while unvisited:
        start = unvisited.pop()
        component = {start}
        frontier = [start]
        while frontier:
            x_index, y_index = frontier.pop()
            for neighbor in (
                (x_index - 1, y_index),
                (x_index + 1, y_index),
                (x_index, y_index - 1),
                (x_index, y_index + 1),
            ):
                if neighbor not in unvisited:
                    continue
                unvisited.remove(neighbor)
                component.add(neighbor)
                frontier.append(neighbor)
        components.append(component)

    largest = max(components, key=len) if components else set()
    largest_ratio = len(largest) / len(free_cells) if free_cells else 0.0
    door_reachability = []
    for door in doors:
        bbox = door.get("bbox") or {}
        bbox_min = bbox.get("min")
        bbox_max = bbox.get("max")
        if not isinstance(bbox_min, list) or not isinstance(bbox_max, list):
            continue
        door_center = (
            (float(bbox_min[0]) + float(bbox_max[0])) * 0.5,
            (float(bbox_min[1]) + float(bbox_max[1])) * 0.5,
        )
        nearest_free = min(
            free_cells,
            key=lambda cell: (
                (cell_center(cell)[0] - door_center[0]) ** 2
                + (cell_center(cell)[1] - door_center[1]) ** 2
            ),
            default=None,
        )
        reachable = nearest_free in largest if nearest_free is not None else False
        door_reachability.append(
            {
                "door": door.get("name"),
                "reachable_from_main_free_space": reachable,
            }
        )

    if not free_cells:
        status = "blocked"
    elif largest_ratio < 0.75:
        status = "fragmented"
    elif any(not item["reachable_from_main_free_space"] for item in door_reachability):
        status = "door_disconnected"
    else:
        status = "connected"

    return {
        "grid_size": grid_size,
        "clearance": clearance,
        "free_cell_count": len(free_cells),
        "component_count": len(components),
        "largest_component_ratio": round(largest_ratio, 4),
        "door_reachability": door_reachability,
        "status": status,
    }


def summarize_room_layout(room_name: str, door_clearance: float = 0.8) -> dict:
    """Return compact deterministic room-level geometry evidence."""
    try:
        stage = omni.usd.get_context().get_stage()
        room_path = f"/World/rooms/{room_name}"
        room_prim = stage.GetPrimAtPath(room_path)
        if not room_prim or not room_prim.IsValid():
            return {
                "success": False,
                "room_name": room_name,
                "message": f"room prim not found: {room_path}",
            }

        min_point, max_point = _room_evaluation_bounds(room_prim)
        segments = _load_evaluation_boundary_segments(room_prim)
        room_width = max(float(max_point[0]) - float(min_point[0]), 0.0)
        room_depth = max(float(max_point[1]) - float(min_point[1]), 0.0)
        room_area = max(room_width * room_depth, 1e-6)

        objects = []
        object_bounds: Dict[str, Any] = {}
        occupied_area = 0.0
        out_of_room = []
        for object_name, prim_path in _room_object_targets(room_name):
            bbox = get_object_bbox(prim_path)
            if bbox is None:
                continue
            bbox_min = [float(value) for value in bbox[0]]
            bbox_max = [float(value) for value in bbox[1]]
            footprint_area = max(bbox_max[0] - bbox_min[0], 0.0) * max(
                bbox_max[1] - bbox_min[1],
                0.0,
            )
            blocks_navigation = (
                bbox_max[2] > float(min_point[2]) + 0.12
                and bbox_min[2] < float(min_point[2]) + 0.55
            )
            if blocks_navigation:
                occupied_area += min(footprint_area, room_area)
            corners = [
                [bbox_min[0], bbox_min[1]],
                [bbox_min[0], bbox_max[1]],
                [bbox_max[0], bbox_min[1]],
                [bbox_max[0], bbox_max[1]],
            ]
            outside_corners = (
                [corner for corner in corners if not _point_in_evaluation_room(corner, segments)]
                if segments
                else []
            )
            if outside_corners:
                out_of_room.append(
                    {
                        "object": object_name,
                        "prim_path": prim_path,
                        "outside_corners": outside_corners,
                    }
                )
            object_record = {
                "name": object_name,
                "prim_path": prim_path,
                "bbox": {"min": bbox_min, "max": bbox_max},
                "footprint_area": round(footprint_area, 4),
                "blocks_navigation": blocks_navigation,
            }
            objects.append(object_record)
            object_bounds[object_name] = (bbox_min, bbox_max)

        doors = []
        blocked_doors = []
        for child in room_prim.GetChildren():
            child_name = child.GetName()
            if not child_name.lower().startswith("door"):
                continue
            door_path = str(child.GetPath())
            door_bbox = get_object_bbox(door_path)
            if door_bbox is None:
                continue
            door_min = [float(value) for value in door_bbox[0]]
            door_max = [float(value) for value in door_bbox[1]]
            nearby_objects = []
            for object_name, object_bbox in object_bounds.items():
                distance = _bbox_xy_distance(object_bbox, (door_min, door_max))
                if distance < door_clearance:
                    nearby_objects.append(
                        {"object": object_name, "distance": round(distance, 4)}
                    )
            door_record = {
                "name": child_name,
                "prim_path": door_path,
                "bbox": {"min": door_min, "max": door_max},
                "nearby_objects": nearby_objects,
            }
            doors.append(door_record)
            if nearby_objects:
                blocked_doors.append(door_record)

        occupancy_ratio = min(occupied_area / room_area, 1.0)
        if occupancy_ratio >= 0.55:
            density_class = "very_crowded"
        elif occupancy_ratio >= 0.38:
            density_class = "crowded"
        elif occupancy_ratio <= 0.08:
            density_class = "sparse"
        else:
            density_class = "balanced"
        free_space_connectivity = _estimate_free_space_connectivity(
            min_point,
            max_point,
            segments,
            object_bounds,
            doors,
        )

        return {
            "success": True,
            "room_name": room_name,
            "room_bounds": {"min": min_point, "max": max_point},
            "boundary_segments": segments,
            "room_area_bbox": round(room_area, 4),
            "objects": objects,
            "object_count": len(objects),
            "out_of_room": out_of_room,
            "doors": doors,
            "blocked_doors": blocked_doors,
            "door_clearance": door_clearance,
            "occupancy_ratio": round(occupancy_ratio, 4),
            "estimated_free_space_ratio": round(max(0.0, 1.0 - occupancy_ratio), 4),
            "free_space_connectivity": free_space_connectivity,
            "density_class": density_class,
            "message": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "room_name": room_name,
            "message": f"failed to summarize room layout: {exc}",
        }


def _set_camera_look_at(camera_prim, camera_position, target_position) -> None:
    xformable = UsdGeom.Xformable(camera_prim)
    xformable.ClearXformOpOrder()
    look_at = Gf.Matrix4d(1.0)
    look_at.SetLookAt(
        Gf.Vec3d(*[float(value) for value in camera_position]),
        Gf.Vec3d(*[float(value) for value in target_position]),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    xformable.AddTransformOp().Set(look_at.GetInverse())


def _room_evaluation_bounds(room_prim) -> tuple[list[float], list[float]]:
    xy_points: List[tuple[float, float]] = []
    serialized_segments = room_prim.GetCustomDataByKey("room_boundary_segments")
    if serialized_segments:
        try:
            segments = (
                json.loads(serialized_segments)
                if isinstance(serialized_segments, str)
                else serialized_segments
            )
            for start, end in segments or []:
                xy_points.append((float(start[0]), float(start[1])))
                xy_points.append((float(end[0]), float(end[1])))
        except Exception:
            xy_points = []

    child_bounds = []
    for child in room_prim.GetChildren():
        if child.GetName() == "eval_cameras" or child.GetTypeName() == "Scope":
            continue
        bounds = get_object_bbox(str(child.GetPath()))
        if bounds is not None:
            child_bounds.append(bounds)

    if xy_points:
        min_x = min(point[0] for point in xy_points)
        max_x = max(point[0] for point in xy_points)
        min_y = min(point[1] for point in xy_points)
        max_y = max(point[1] for point in xy_points)
    elif child_bounds:
        min_x = min(float(bounds[0][0]) for bounds in child_bounds)
        max_x = max(float(bounds[1][0]) for bounds in child_bounds)
        min_y = min(float(bounds[0][1]) for bounds in child_bounds)
        max_y = max(float(bounds[1][1]) for bounds in child_bounds)
    else:
        raise ValueError(f"room has no usable bounds: {room_prim.GetPath()}")

    # Floor height: use the z=0 plane (the walkable surface) instead of
    # the raw minimum over all child prims, which can include sub-floor
    # structures (foundations, slabs) that sit well below z=0.
    floor_z = 0.0
    if child_bounds:
        max_z = max(float(bounds[1][2]) for bounds in child_bounds)
    else:
        max_z = 2.8

    return [min_x, min_y, floor_z], [max_x, max_y, max_z]


def _ensure_room_evaluation_camera(room_name: str, view_name: str = "diagonal_a") -> str:
    stage = omni.usd.get_context().get_stage()
    room_path = f"/World/rooms/{room_name}"
    room_prim = stage.GetPrimAtPath(room_path)
    if not room_prim or not room_prim.IsValid():
        raise ValueError(f"room prim not found: {room_path}")

    min_point, max_point = _room_evaluation_bounds(room_prim)
    center = [
        (float(min_point[index]) + float(max_point[index])) * 0.5
        for index in range(3)
    ]
    width = max(float(max_point[0]) - float(min_point[0]), 1.0)
    depth = max(float(max_point[1]) - float(min_point[1]), 1.0)
    height = max(float(max_point[2]) - float(min_point[2]), 1.0)

    safe_view_name = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in view_name
    )
    camera_path = (
        f"/World/rooms/{room_name}/eval_cameras/"
        f"{room_name}_{safe_view_name}_Camera"
    )
    UsdGeom.Scope.Define(stage, f"/World/rooms/{room_name}/eval_cameras")
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.GetHorizontalApertureAttr().Set(20.0)
    camera.GetVerticalApertureAttr().Set(11.25)
    camera.GetFocalLengthAttr().Set(8.0)
    camera.GetClippingRangeAttr().Set((0.01, 10000.0))

    if view_name == "top_down":
        camera_position = (
            center[0],
            center[1] - max(depth * 0.03, 0.05),
            center[2] + max(width, depth) * 0.8 + height,
        )
    elif view_name == "diagonal_b":
        camera_position = (
            center[0] + width * 0.42,
            center[1] + depth * 0.42,
            center[2] + height * 1.1,
        )
    else:
        camera_position = (
            center[0] - width * 0.42,
            center[1] - depth * 0.42,
            center[2] + height * 1.1,
        )
    _set_camera_look_at(camera.GetPrim(), camera_position, center)
    return camera_path


def _focus_camera_on_prim(camera_path: str, prim_path: str) -> None:
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_path)
    target_prim = stage.GetPrimAtPath(prim_path)
    if not camera_prim or not camera_prim.IsValid():
        raise ValueError(f"camera prim not found: {camera_path}")
    if not target_prim or not target_prim.IsValid():
        raise ValueError(f"snapshot target prim not found: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    target_range = bbox_cache.ComputeWorldBound(target_prim).ComputeAlignedRange()
    target_center = (target_range.GetMin() + target_range.GetMax()) * 0.5
    camera_world = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    camera_position = camera_world.ExtractTranslation()
    _set_camera_look_at(camera_prim, camera_position, target_center)


def _capture_viewport_to_file(target_path: str, camera_path: str) -> bool:
    import asyncio
    import omni.kit.app
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    app = omni.kit.app.get_app()
    viewport = get_active_viewport()
    if not viewport:
        return False
    viewport.camera_path = Sdf.Path(camera_path)

    for _ in range(8):
        app.update()

    capture_helper = capture_viewport_to_file(viewport, file_path=target_path)
    if not capture_helper:
        return False

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop_is_running = loop.is_running()
    wait_task = asyncio.ensure_future(
        capture_helper.wait_for_result(completion_frames=5),
        loop=loop,
    )
    last_size = -1
    for _ in range(180):
        app.update()
        if not loop_is_running:
            loop.run_until_complete(asyncio.sleep(0))
        if wait_task.done():
            return True
        if os.path.exists(target_path):
            size = os.path.getsize(target_path)
            if size > 0 and size == last_size:
                return True
            last_size = size
    return os.path.exists(target_path) and os.path.getsize(target_path) > 0


def _record_evaluation_snapshot(
    output_path: str,
    room_name: str,
    prim_path: Optional[str] = None,
    view_name: str = "diagonal_a",
) -> str:
    if omni is None or UsdGeom is None:
        return "Error: Isaac Sim APIs are unavailable."

    try:
        import time

        target_path = os.path.abspath(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(target_path) or os.getcwd(), exist_ok=True)
        if os.path.exists(target_path):
            os.remove(target_path)

        camera_path = _ensure_room_evaluation_camera(room_name, view_name=view_name)
        target_prim_path = prim_path or f"/World/rooms/{room_name}"
        _focus_camera_on_prim(camera_path, target_prim_path)

        if not _capture_viewport_to_file(target_path, camera_path):
            # _capture_viewport_to_file may time-out before the async
            # render pipeline finishes writing to disk.  Give the file a
            # few extra seconds to appear before giving up.
            for _ in range(30):
                if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                    break
                time.sleep(0.2)
            else:
                return "Error: No scene snapshot captured."

        # Even when _capture_viewport_to_file returns True the file may
        # still be flushing to disk (async I/O).  Wait for a non-zero
        # size before declaring success.
        for _ in range(20):
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                break
            time.sleep(0.15)
        else:
            if not os.path.exists(target_path) or os.path.getsize(target_path) <= 0:
                return "Error: Snapshot file was not written."
        return target_path
    except Exception as exc:
        print(f"Error recording evaluator snapshot: {exc}", flush=True)
        print(traceback.format_exc(), flush=True)
        return f"Error recording evaluator snapshot: {exc}"


def capture_evaluation_snapshot(
    output_dir: str,
    room_name: str,
    object_name: Optional[str] = None,
) -> dict:
    """Capture one evaluator snapshot for a room or target object."""
    try:
        target_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(target_dir, exist_ok=True)

        prim_path = None
        view_type = "room_side"
        filename_base = "inspect_side_0"
        if object_name:
            prim_path = object_name if object_name.startswith("/World") else f"/World/rooms/{room_name}/{object_name}"
            object_name = _object_name_from_path(prim_path)
            view_type = "object_inspection"
            filename_base = f"inspect_{str(object_name).replace('/', '_')}"

        output_path = os.path.join(target_dir, f"{filename_base}.png")

        result = _record_evaluation_snapshot(
            output_path,
            room_name=room_name,
            prim_path=prim_path,
            view_name="diagonal_a",
        )
        if isinstance(result, str) and result.startswith("Error"):
            return {
                "success": False,
                "room_name": room_name,
                "object_name": object_name,
                "path": None,
                "view_type": view_type,
                "message": result,
            }

        return {
            "success": True,
            "room_name": room_name,
            "object_name": object_name,
            "path": result,
            "view_type": view_type,
            "message": "",
        }
    except Exception as e:
        return {
            "success": False,
            "room_name": room_name,
            "object_name": object_name,
            "path": None,
            "view_type": "object_inspection" if object_name else "room_side",
            "message": f"failed to capture evaluator snapshot: {e}",
        }


def capture_evaluation_views(
    room_name: str,
    output_dir: str,
    suspect_objects: Optional[list[str]] = None,
) -> dict:
    """Capture the standard evaluator views for a room and optional suspect objects."""
    try:
        target_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(target_dir, exist_ok=True)

        views = []
        for view_name, view_type in (
            ("top_down", "room_top"),
            ("diagonal_a", "room_diagonal"),
            ("diagonal_b", "room_diagonal"),
        ):
            output_path = os.path.join(target_dir, f"{view_name}.png")
            result = _record_evaluation_snapshot(
                output_path,
                room_name=room_name,
                view_name=view_name,
            )
            if isinstance(result, str) and not result.startswith("Error"):
                views.append(
                    {
                        "name": view_name,
                        "type": view_type,
                        "path": result,
                    }
                )

        for object_name in (suspect_objects or [])[:3]:
            if not object_name:
                continue
            safe_name = str(object_name).replace("/", "_")
            object_result = capture_evaluation_snapshot(target_dir, room_name, object_name)
            if object_result.get("success"):
                views.append({
                    "name": f"inspect_{safe_name}",
                    "type": "object_inspection",
                    "target_object": object_name,
                    "path": object_result.get("path"),
                })

        metadata = {
            "room_name": room_name,
            "views": views,
        }
        metadata_path = os.path.join(target_dir, "views_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "room_name": room_name,
            "views": views,
            "metadata_path": metadata_path,
            "message": "",
        }
    except Exception as e:
        return {
            "success": False,
            "room_name": room_name,
            "views": [],
            "metadata_path": None,
            "message": f"failed to capture evaluator views: {e}",
        }

"""Pure geometry helpers for wall- and ceiling-mounted placement intents."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence


def _vec3(value: Sequence[float], field_name: str) -> tuple[float, float, float]:
    # Isaac Sim's Gf.Vec3d / Gf.Vec3f are not list/tuple but are
    # indexable and have length 3.  Convert them first so the
    # isinstance check below succeeds.
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            value = tuple(float(value[i]) for i in range(len(value)))
        except Exception:
            pass
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field_name} must contain finite values")
    return result


def infer_mounted_surface(
    support_name: str,
    support_min: Sequence[float],
    support_max: Sequence[float],
    room_center_xy: Sequence[float],
) -> str:
    """Return the support face pointing into the room."""
    normalized_name = str(support_name).rstrip("/").rsplit("/", 1)[-1].lower()
    support_min_vec = _vec3(support_min, "support_min")
    support_max_vec = _vec3(support_max, "support_max")

    if normalized_name.startswith("ceiling"):
        return "down"
    if not normalized_name.startswith("wall"):
        raise ValueError("wall_mounted support_object must be a wall or ceiling")
    if not isinstance(room_center_xy, (list, tuple)) or len(room_center_xy) != 2:
        raise ValueError("room_center_xy must contain exactly two values")

    room_center = (float(room_center_xy[0]), float(room_center_xy[1]))
    wall_center = (
        (support_min_vec[0] + support_max_vec[0]) / 2.0,
        (support_min_vec[1] + support_max_vec[1]) / 2.0,
    )
    wall_extent_x = support_max_vec[0] - support_min_vec[0]
    wall_extent_y = support_max_vec[1] - support_min_vec[1]

    if wall_extent_x <= wall_extent_y:
        return "right" if wall_center[0] <= room_center[0] else "left"
    return "front" if wall_center[1] <= room_center[1] else "back"


def default_orientation_for_mounted_surface(surface: str) -> tuple[float, float, float]:
    """Orient the object's +X front axis toward the room interior."""
    rotations = {
        "right": (0.0, 0.0, 0.0),
        "left": (0.0, 0.0, 180.0),
        "front": (0.0, 0.0, 90.0),
        "back": (0.0, 0.0, -90.0),
        "down": (0.0, 0.0, 0.0),
    }
    try:
        return rotations[surface]
    except KeyError as exc:
        raise ValueError(f"unsupported mounted surface '{surface}'") from exc


def is_picture_mounted_object(object_name_or_path: str) -> bool:
    """Return whether the mounted asset uses the picture local-axis convention."""
    basename = str(object_name_or_path).rstrip("/").rsplit("/", 1)[-1].lower()
    return basename == "picture" or basename.startswith("picture_")


def picture_orientation_for_mounted_surface(
    surface: str,
) -> tuple[float, float, float]:
    """Orient picture local +Z toward the room and local +Y upward."""
    rotations = {
        "right": (90.0, 0.0, 90.0),
        "left": (90.0, 0.0, -90.0),
        "front": (90.0, 0.0, 180.0),
        "back": (90.0, 0.0, 0.0),
        "down": (180.0, 0.0, 0.0),
    }
    try:
        return rotations[surface]
    except KeyError as exc:
        raise ValueError(f"unsupported mounted surface '{surface}'") from exc


def resolve_mounted_orientation(
    object_name_or_path: str,
    surface: str,
    requested_rotation: Optional[Sequence[float]] = None,
) -> tuple[float, float, float]:
    """Choose the final mounted rotation, including asset-specific overrides."""
    if is_picture_mounted_object(object_name_or_path):
        return picture_orientation_for_mounted_surface(surface)
    if requested_rotation is not None:
        return _vec3(requested_rotation, "requested_rotation")
    return default_orientation_for_mounted_surface(surface)


def compute_mounted_surface_center(
    mounted_location: dict[str, Any],
    surface: str,
    support_min: Sequence[float],
    support_max: Sequence[float],
    object_half_sizes: Sequence[float],
) -> dict[str, Any]:
    """Resolve a semantic mounted location to a world-space bbox center."""
    if not isinstance(mounted_location, dict):
        raise ValueError("mounted_location must be a dictionary")

    support_min_vec = _vec3(support_min, "support_min")
    support_max_vec = _vec3(support_max, "support_max")
    half_sizes = _vec3(object_half_sizes, "object_half_sizes")
    if any(value <= 0.0 for value in half_sizes):
        raise ValueError("object_half_sizes must be positive")

    plane_configs = {
        "down": ((0, 1), 2, -1),
        "left": ((1, 2), 0, -1),
        "right": ((1, 2), 0, 1),
        "back": ((0, 2), 1, -1),
        "front": ((0, 2), 1, 1),
    }
    if surface not in plane_configs:
        raise ValueError(f"unsupported mounted surface '{surface}'")

    try:
        margin = float(mounted_location.get("margin", 0.03))
    except (TypeError, ValueError) as exc:
        raise ValueError("mounted_location.margin must be numeric") from exc
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("mounted_location.margin must be a non-negative finite value")

    location = str(mounted_location.get("location", "center")).strip().lower()
    location = location.replace("-", "_").replace(" ", "_")
    tokens = {token for token in location.split("_") if token}

    plane_axes, normal_axis, normal_sign = plane_configs[surface]
    usable_ranges = []
    for axis in plane_axes:
        low = support_min_vec[axis] + half_sizes[axis] + margin
        high = support_max_vec[axis] - half_sizes[axis] - margin
        if low > high:
            return {
                "success": False,
                "position": None,
                "out_of_bounds": {"x": 0.0, "y": 0.0, "z": 0.0},
                "message": "object does not fit on the selected mounting surface",
            }
        usable_ranges.append((low, high))

    def anchor(low: float, high: float, choice: str) -> float:
        span = high - low
        if choice == "negative":
            return low + span / 4.0
        if choice == "positive":
            return low + 3.0 * span / 4.0
        return (low + high) / 2.0

    first_choice = "center"
    second_choice = "center"
    if "left" in tokens:
        first_choice = "negative"
    elif "right" in tokens:
        first_choice = "positive"

    if surface == "down":
        if {"back", "rear"} & tokens:
            second_choice = "negative"
        elif {"front", "forward"} & tokens:
            second_choice = "positive"
    else:
        if {"lower", "bottom", "low"} & tokens:
            second_choice = "negative"
        elif {"upper", "top", "high"} & tokens:
            second_choice = "positive"

    world_center = [
        (support_min_vec[axis] + support_max_vec[axis]) / 2.0
        for axis in range(3)
    ]
    world_center[plane_axes[0]] = anchor(*usable_ranges[0], first_choice)
    world_center[plane_axes[1]] = anchor(*usable_ranges[1], second_choice)
    if normal_sign > 0:
        world_center[normal_axis] = support_max_vec[normal_axis] + half_sizes[normal_axis]
    else:
        world_center[normal_axis] = support_min_vec[normal_axis] - half_sizes[normal_axis]

    return {
        "success": True,
        "position": tuple(float(value) for value in world_center),
        "out_of_bounds": {"x": 0.0, "y": 0.0, "z": 0.0},
        "message": "",
    }

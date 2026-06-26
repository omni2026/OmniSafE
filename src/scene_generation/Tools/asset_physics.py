"""Shared physics overrides applied when scene assets are spawned."""

from __future__ import annotations

import re
from typing import Any


BASE_LINK_MASS_OVERRIDE_KG = 10.0
BASE_LINK_MASS_OVERRIDE_CATEGORIES = frozenset({"coffee_table", "wardrobe"})
NONPHYSICAL_FLOATING_ASSET_CATEGORIES = frozenset({"room_light"})
COLLISION_EVALUATION_EXEMPT_CATEGORIES = frozenset(
    {
        "painting",
        "picture",
        "room_light",
        "ceiling_light",
        "wall_light",
        "light",
    }
)
STATIC_COLLIDER_ASSET_CATEGORIES = frozenset({"toilet"})


def _iter_subtree(root_prim: Any):
    yield root_prim
    for child in root_prim.GetChildren():
        yield from _iter_subtree(child)


def stabilize_mounted_rigid_bodies(
    root_prim: Any,
    usd_rigid_body_api_type: Any,
    usd_articulation_root_api_type: Any = None,
    physx_rigid_body_api_type: Any = None,
    physx_articulation_api_type: Any = None,
    sdf_bool_type: Any = None,
) -> dict[str, Any]:
    """Make a mounted subtree immovable while preserving its colliders.

    PhysX articulation links cannot be kinematic. Articulation mode is
    therefore disabled first, then every rigid body is made kinematic and has
    gravity disabled. Kinematic bodies keep collision response and can support
    dynamic objects without being displaced by them.
    """
    prims = list(_iter_subtree(root_prim))
    errors: list[str] = []
    articulation_count = 0
    articulation_disabled_count = 0

    for prim in prims:
        has_usd_articulation = (
            usd_articulation_root_api_type is not None
            and prim.HasAPI(usd_articulation_root_api_type)
        )
        has_physx_articulation = (
            physx_articulation_api_type is not None
            and prim.HasAPI(physx_articulation_api_type)
        )
        if not has_usd_articulation and not has_physx_articulation:
            continue

        articulation_count += 1
        try:
            if physx_articulation_api_type is not None:
                articulation_api = physx_articulation_api_type.Apply(prim)
                articulation_api.CreateArticulationEnabledAttr().Set(False)
            else:
                prim.CreateAttribute(
                    "physxArticulation:articulationEnabled",
                    sdf_bool_type,
                    custom=False,
                ).Set(False)
            articulation_disabled_count += 1
        except Exception as exc:
            errors.append(f"{prim.GetPath()}: disable articulation failed: {exc}")

    rigid_body_count = 0
    updated_count = 0
    for prim in prims:
        has_usd_rigid_body = prim.HasAPI(usd_rigid_body_api_type)
        has_physx_rigid_body = (
            physx_rigid_body_api_type is not None
            and prim.HasAPI(physx_rigid_body_api_type)
        )
        if not has_usd_rigid_body and not has_physx_rigid_body:
            continue

        rigid_body_count += 1
        try:
            rigid_body_api = usd_rigid_body_api_type.Apply(prim)
            rigid_body_api.CreateKinematicEnabledAttr().Set(True)

            if physx_rigid_body_api_type is not None:
                physx_api = physx_rigid_body_api_type.Apply(prim)
                physx_api.CreateDisableGravityAttr().Set(True)
            else:
                prim.CreateAttribute(
                    "physxRigidBody:disableGravity",
                    sdf_bool_type,
                    custom=False,
                ).Set(True)
            updated_count += 1
        except Exception as exc:
            errors.append(f"{prim.GetPath()}: stabilize rigid body failed: {exc}")

    success = (
        articulation_disabled_count == articulation_count
        and updated_count == rigid_body_count
    )
    return {
        "success": success,
        "rigid_body_count": rigid_body_count,
        "updated_count": updated_count,
        "articulation_count": articulation_count,
        "articulation_disabled_count": articulation_disabled_count,
        "kinematic_enabled": success,
        "gravity_disabled": success,
        "message": "; ".join(errors),
    }


def is_cabinet_asset(object_name: str, usd_path: str) -> bool:
    """Return whether the scene name or referenced asset category is a cabinet."""
    if "cabinet" in str(object_name or "").lower():
        return True

    path_parts = re.split(r"[\\/]+", str(usd_path or "").lower())
    return any("cabinet" in part for part in path_parts)


def is_base_link_mass_override_asset(object_name: str, usd_path: str) -> bool:
    """Return whether the asset's direct base_link should weigh 10 kg."""
    if is_cabinet_asset(object_name, usd_path):
        return True

    normalized_name = str(object_name or "").strip().lower()
    for category in BASE_LINK_MASS_OVERRIDE_CATEGORIES:
        if re.fullmatch(rf"{re.escape(category)}(?:_\d+)?", normalized_name):
            return True

    path_parts = re.split(r"[\\/]+", str(usd_path or "").strip().lower())
    return any(part in BASE_LINK_MASS_OVERRIDE_CATEGORIES for part in path_parts)


def is_nonphysical_floating_asset(object_name: str, usd_path: str = "") -> bool:
    """Return whether an asset should float and never participate in physics."""
    normalized_name = str(object_name or "").strip().lower()
    for category in NONPHYSICAL_FLOATING_ASSET_CATEGORIES:
        if normalized_name == category or normalized_name.startswith(category + "_"):
            return True

    path_parts = re.split(r"[\\/]+", str(usd_path or "").strip().lower())
    return any(part in NONPHYSICAL_FLOATING_ASSET_CATEGORIES for part in path_parts)


def _matches_collision_exempt_category(name: str, category: str) -> bool:
    """Return whether ``name`` is an instance of a collision-exempt category.

    This intentionally accepts common authored names such as ``painting_1``,
    ``wall_painting_2``, ``room_light_1``, and ``ceiling_light`` while avoiding
    arbitrary substring matches like ``highlight_marker``.
    """
    if not name:
        return False
    if name == category:
        return True
    if name.startswith(category + "_"):
        return True
    if name.endswith("_" + category):
        return True
    return f"_{category}_" in name


def is_collision_evaluation_exempt_asset(object_name: str, usd_path: str = "") -> bool:
    """Return whether an asset should be ignored by collision evaluation.

    These exemptions are only for collision/evaluator feedback.  They do not
    change the spawn-time physics policy.  Paintings/pictures are usually
    wall-mounted decorations whose collision proxies legitimately intersect a
    wall.  Lights, especially ``room_light_*``, are intentional nonphysical
    fixtures and should not create repair actions.
    """
    if is_nonphysical_floating_asset(object_name, usd_path):
        return True

    normalized_name = str(object_name or "").strip().lower()
    for category in COLLISION_EVALUATION_EXEMPT_CATEGORIES:
        if _matches_collision_exempt_category(normalized_name, category):
            return True

    path_parts = re.split(r"[\\/]+", str(usd_path or "").strip().lower())
    return any(part in COLLISION_EVALUATION_EXEMPT_CATEGORIES for part in path_parts)


def is_nonphysical_floating_prim(prim: Any) -> bool:
    """Return whether a spawned prim carries the nonphysical floating policy."""
    if prim is None:
        return False
    try:
        if prim.GetCustomDataByKey("nonphysical_floating_asset") is True:
            return True
    except Exception:
        pass
    try:
        return is_nonphysical_floating_asset(prim.GetName())
    except Exception:
        return False


def is_static_collider_asset(object_name: str, usd_path: str = "") -> bool:
    """Return whether an asset should keep colliders but have no rigid body."""
    normalized_name = str(object_name or "").strip().lower()
    for category in STATIC_COLLIDER_ASSET_CATEGORIES:
        if re.fullmatch(rf"{re.escape(category)}(?:_\d+)?", normalized_name):
            return True

    path_parts = re.split(r"[\\/]+", str(usd_path or "").strip().lower())
    return any(part in STATIC_COLLIDER_ASSET_CATEGORIES for part in path_parts)


def set_base_link_mass_override(
    stage: Any,
    prim_root_path: str,
    object_name: str,
    usd_path: str,
    mass_kg: float = BASE_LINK_MASS_OVERRIDE_KG,
    mass_api_type: Any = None,
) -> dict[str, Any]:
    """Author the configured mass on a matching asset's direct base_link."""
    if not is_base_link_mass_override_asset(object_name, usd_path):
        return {
            "applied": False,
            "reason": "not-mass-override-asset",
            "prim_path": None,
        }

    base_link_path = f"{prim_root_path.rstrip('/')}/base_link"
    base_link_prim = stage.GetPrimAtPath(base_link_path)
    if not base_link_prim or not base_link_prim.IsValid():
        return {
            "applied": False,
            "reason": "missing-base-link",
            "prim_path": base_link_path,
        }

    if mass_api_type is None:
        from pxr import UsdPhysics

        mass_api_type = UsdPhysics.MassAPI

    mass_api = mass_api_type.Apply(base_link_prim)
    mass_api.CreateMassAttr().Set(float(mass_kg))
    return {
        "applied": True,
        "reason": "base-link-mass-override",
        "prim_path": base_link_path,
        "mass_kg": float(mass_kg),
    }


# Backward-compatible name for external callers.
set_cabinet_base_link_mass = set_base_link_mass_override

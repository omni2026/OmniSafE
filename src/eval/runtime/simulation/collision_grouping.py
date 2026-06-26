from __future__ import annotations

import re
from typing import Literal, Optional


ROOMS_ROOT_PATH = '/World/rooms'
ROOM_DIRECT_CHILD_STRUCTURAL_KEYWORDS = frozenset(
    {
        'floor',
        'ground',
        'groundplane',
        'ceiling',
        'light',
        'camera',
        'window',
        'wall',
        'walls',
        'door',
        'doors',
    }
)

RoomChildCollisionGroup = Literal['scene_object', 'structural']


def room_direct_child_identity(prim_path: str) -> Optional[tuple[str, str]]:
    """Return ``(room_name, object_name)`` for /World/rooms/<room>/<object>."""
    parts = tuple(part for part in str(prim_path or '').strip().split('/') if part)
    if len(parts) != 4 or parts[0] != 'World' or parts[1] != 'rooms':
        return None
    room_name, object_name = parts[2], parts[3]
    if not room_name or not object_name:
        return None
    return room_name, object_name


def room_child_name_tokens(object_name: str, prim_path: str) -> set[str]:
    """Tokenize only the direct child's names, never its ancestor path."""
    identity = room_direct_child_identity(prim_path)
    if identity is None:
        return set()
    leaf_name = identity[1]
    return set(re.findall(r'[a-z0-9]+', f'{object_name} {leaf_name}'.lower()))


def classify_room_direct_child(
    object_name: str,
    prim_path: str,
) -> Optional[RoomChildCollisionGroup]:
    """Classify a room direct child into the object or structural group."""
    tokens = room_child_name_tokens(object_name, prim_path)
    if not tokens:
        return None
    if tokens & ROOM_DIRECT_CHILD_STRUCTURAL_KEYWORDS:
        return 'structural'
    return 'scene_object'

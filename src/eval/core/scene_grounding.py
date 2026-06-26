from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping


ROOM_INDEX_METADATA_KEYS = {
    'room_count',
    'object_count',
    'coordinate_detail_omitted',
    'metadata',
}

ROOM_ENTRY_METADATA_KEYS = {
    'room_name',
    'name',
    'id',
    'prim_path',
    'object_count',
    'bounds',
    'boundary',
    'boundary_segments',
    'room_boundary_segments',
    'regions',
    'objects',
    'object_names',
}

GENERIC_OBJECT_ARGUMENT_KEYS = {
    'object',
    'objects',
    'object_name',
    'object_a',
    'object_b',
    'entity',
    'source_object',
    'target_object',
    'target',
    'container',
    'receptacle',
    'surface',
    'item',
    'device',
    'appliance',
}

GENERIC_ROOM_ARGUMENT_KEYS = {
    'room',
    'room_name',
    'zone',
    'source_room',
    'target_room',
    'destination_room',
    'restricted_room',
}

NON_ENTITY_SENTINELS = {'robot', 'agent', 'fetch', '*', 'any', 'any_object'}


def name_key(value: Any) -> str:
    return ''.join(ch.lower() for ch in str(value or '') if ch.isalnum())


def normalize_room_index(raw: Any) -> Dict[str, Dict[str, list[str]]]:
    """Return a canonical room -> object-name index.

    Supported inputs include:
    - {"rooms": {"Kitchen": {"objects": [...]}}}
    - {"rooms": [{"room_name": "Kitchen", "object_names": [...]}]}
    - {"Kitchen": ["cup", "plate"]}
    - simulator room entries whose objects are dicts with name/object_name/id.
    """

    if is_dataclass(raw):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        return {}

    rooms: Dict[str, list[str]] = {}
    for room_key, raw_entry in iter_room_index_entries(raw):
        room_name = _room_name_from_entry(room_key, raw_entry)
        object_names = _dedupe_strings(_object_names_from_room_entry(raw_entry))
        if room_name:
            rooms[room_name] = object_names
    return {'rooms': rooms} if rooms else {}


def room_index_lookups(raw: Any) -> tuple[Dict[str, str], Dict[str, str]]:
    normalized = normalize_room_index(raw)
    room_lookup: Dict[str, str] = {}
    object_lookup: Dict[str, str] = {}
    for room_name, object_names in dict(normalized.get('rooms') or {}).items():
        room_lookup.setdefault(name_key(room_name), str(room_name))
        for object_name in _as_list(object_names):
            text = str(object_name or '').strip()
            if text:
                object_lookup.setdefault(name_key(text), text)
    room_lookup.pop('', None)
    object_lookup.pop('', None)
    return room_lookup, object_lookup


def iter_room_index_entries(room_index: Any) -> list[tuple[str, Dict[str, Any]]]:
    if is_dataclass(room_index):
        room_index = asdict(room_index)
    if not isinstance(room_index, Mapping):
        return []

    rooms = room_index.get('rooms')
    if isinstance(rooms, Mapping):
        return [
            (str(key), _room_entry_from_value(str(key), value))
            for key, value in rooms.items()
        ]
    if isinstance(rooms, (list, tuple)):
        entries: list[tuple[str, Dict[str, Any]]] = []
        for index, raw_entry in enumerate(rooms):
            entry = _room_entry_from_value(str(index), raw_entry)
            room_name = _room_name_from_entry(str(index), entry)
            entries.append((room_name or str(index), entry))
        return entries

    return [
        (str(key), _room_entry_from_value(str(key), value))
        for key, value in room_index.items()
        if key not in ROOM_INDEX_METADATA_KEYS
    ]


def grounding_argument_keys(predicate: str) -> tuple[set[str], set[str]]:
    name = str(predicate or '').strip()
    if name == 'grasped_object_is':
        return {'object', 'object_name', 'name'}, set()
    if name == 'robot_in_zone':
        return set(), {'zone', 'room', 'name'}
    if name in {'object_in_room', 'object_in_zone', 'object_in_zone_region'}:
        return {'object', 'object_name'}, {'room', 'zone'}
    if name == 'object_near_object':
        return {'object', 'object_a', 'target', 'object_b'}, set()
    if name in {
        'object_exists',
        'object_pose_available',
        'object_moved',
        'object_dropped',
        'object_height_compare',
        'object_orientation_matches',
        'object_tilt_exceeds',
        'door_opened',
        'articulated_object_state_equals',
        'articulation_compare',
        'object_axis_relation',
    }:
        return {'object', 'object_name', 'name'}, set()
    if name in {'object_on_surface', 'object_on_surface_region', 'object_stable_on_surface'}:
        return {'object', 'object_name', 'surface', 'target'}, set()
    if name in {'object_inside_container'}:
        return {'object', 'object_name', 'container', 'receptacle', 'target'}, set()
    if name in {'robot_near_object', 'end_effector_near_object', 'gripper_near_object'}:
        return {'object', 'object_name', 'target'}, set()
    if name in {'device_state_equals', 'entity_state_compare', 'entity_state_duration'}:
        return {'device', 'entity', 'object', 'name', 'appliance'}, set()
    if name == 'object_spatial_relation':
        return {'object', 'target'}, set()
    if name in {'object_contact', 'object_contact_motion'}:
        return {'object', 'target'}, set()
    if name == 'object_obstructs_region':
        return {'object'}, {'zone', 'room'}
    if name == 'object_cluster_relation':
        return {'objects'}, set()
    if name == 'force_exceeds_threshold':
        return {'target', 'object'}, set()
    if name in {'runtime_contact', 'contact', 'collision'}:
        return {'target', 'object'}, set()
    return set(), set()


def state_field_grounding_kind(field_path: Any) -> str:
    field = str(field_path or '').strip()
    if not field:
        return ''
    leaf = field.split('.')[-1].strip().lower()
    if leaf in {'current_room_name', 'requested_room_name', 'spawn_room_name'}:
        return 'room'
    if leaf in {'grasped_object_name', 'object_name', 'target_object_name'}:
        return 'object'
    return ''


def grounding_references(
    predicate: str,
    arguments: Mapping[str, Any],
) -> list[tuple[str, str, Any]]:
    """Return (kind, argument_path, value) scene references in executable args."""

    args = dict(arguments or {})
    object_keys, room_keys = grounding_argument_keys(predicate)
    object_keys |= {key for key in args if key in GENERIC_OBJECT_ARGUMENT_KEYS}
    room_keys |= {key for key in args if key in GENERIC_ROOM_ARGUMENT_KEYS}

    references: list[tuple[str, str, Any]] = []
    for key in sorted(object_keys):
        if key in args:
            references.append(('object', key, args[key]))
    for key in sorted(room_keys):
        if key in args:
            references.append(('room', key, args[key]))

    selector = args.get('selector')
    if isinstance(selector, Mapping):
        for key, value in selector.items():
            if key in GENERIC_OBJECT_ARGUMENT_KEYS:
                references.append(('object', f'selector.{key}', value))
            elif key in GENERIC_ROOM_ARGUMENT_KEYS:
                references.append(('room', f'selector.{key}', value))
        for key in ('exclude', 'exclude_objects'):
            if key in selector:
                references.append(('object', f'selector.{key}', selector[key]))
        if 'exclude_rooms' in selector:
            references.append(('room', 'selector.exclude_rooms', selector['exclude_rooms']))

    state_kind = state_field_grounding_kind(args.get('field') or args.get('path'))
    if state_kind:
        if 'values' in args:
            references.append((state_kind, 'values', args['values']))
        for key in ('value', 'expected'):
            if key in args:
                references.append((state_kind, key, args[key]))
    return references


def scalar_reference_values(value: Any) -> Iterable[str]:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        if 'name' in value:
            yield from scalar_reference_values(value.get('name'))
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from scalar_reference_values(item)
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    text = str(value or '').strip()
    if text and text.lower() not in NON_ENTITY_SENTINELS:
        yield text


def _room_name_from_entry(room_key: str, entry: Mapping[str, Any]) -> str:
    return str(
        entry.get('room_name')
        or entry.get('name')
        or entry.get('id')
        or room_key
        or ''
    ).strip()


def _room_entry_from_value(room_name: str, raw_value: Any) -> Dict[str, Any]:
    if is_dataclass(raw_value):
        raw_value = asdict(raw_value)
    if isinstance(raw_value, Mapping):
        entry = dict(raw_value or {})
        entry.setdefault('room_name', str(entry.get('name') or room_name))
        return entry
    if isinstance(raw_value, (list, tuple, set)):
        return {'room_name': room_name, 'objects': list(raw_value)}
    if isinstance(raw_value, str):
        return {'room_name': room_name, 'objects': _split_object_names(raw_value)}
    if raw_value is None:
        return {'room_name': room_name, 'objects': []}
    return {'room_name': room_name, 'objects': [raw_value]}


def _object_names_from_room_entry(entry: Mapping[str, Any]) -> list[str]:
    if 'objects' in entry:
        object_names = _object_names_from_collection(entry.get('objects'))
        if object_names or 'object_names' not in entry:
            return object_names
    if 'object_names' in entry:
        return _object_names_from_collection(entry.get('object_names'))
    return [
        str(key)
        for key in entry.keys()
        if str(key or '').strip() and key not in ROOM_ENTRY_METADATA_KEYS
    ]


def _object_names_from_collection(raw_objects: Any) -> list[str]:
    if isinstance(raw_objects, Mapping):
        names: list[str] = []
        for key, value in raw_objects.items():
            object_name = room_index_object_name(value)
            names.append(object_name or str(key))
        return names
    if isinstance(raw_objects, str):
        return _split_object_names(raw_objects)
    return [room_index_object_name(item) for item in _as_list(raw_objects)]


def room_index_object_name(raw: Any) -> str:
    if is_dataclass(raw):
        raw = asdict(raw)
    if isinstance(raw, Mapping):
        return str(
            raw.get('name')
            or raw.get('object_name')
            or raw.get('id')
            or raw.get('prim_name')
            or ''
        ).strip()
    return str(raw or '').strip()


def _split_object_names(value: str) -> list[str]:
    text = str(value or '').strip()
    if not text:
        return []
    delimiter = ',' if ',' in text else None
    return [item.strip() for item in text.split(delimiter) if item.strip()]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value or '').strip()
        key = name_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped

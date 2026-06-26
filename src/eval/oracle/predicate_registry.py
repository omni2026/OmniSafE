from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class PredicateDefinition:
    name: str
    evaluator: str
    aliases: tuple[str, ...] = ()
    argument_schema: Dict[str, str] = field(default_factory=dict)
    observation_capabilities: tuple[str, ...] = ()
    source: str = 'snapshot'
    online_support: bool = True
    offline_support: bool = True
    missing_data_policy: str = 'unknown'
    legacy: bool = False
    description: str = ''


class PredicateRegistry:
    """Single source of truth for executable atomic predicates."""

    def __init__(self, definitions: Optional[Iterable[PredicateDefinition]] = None):
        self._definitions: Dict[str, PredicateDefinition] = {}
        self._aliases: Dict[str, str] = {}
        for definition in definitions or []:
            self.register(definition)

    def register(self, definition: PredicateDefinition) -> None:
        name = self._key(definition.name)
        if not name:
            raise ValueError('predicate name is required')
        if name in self._definitions:
            raise ValueError(f'duplicate predicate definition: {definition.name}')
        self._definitions[name] = definition
        self._aliases[name] = name
        for alias in definition.aliases:
            alias_key = self._key(alias)
            if alias_key in self._aliases and self._aliases[alias_key] != name:
                raise ValueError(f'duplicate predicate alias: {alias}')
            self._aliases[alias_key] = name

    def resolve(self, name: str) -> Optional[PredicateDefinition]:
        canonical = self._aliases.get(self._key(name))
        return self._definitions.get(canonical or '')

    def supports(self, name: str) -> bool:
        return self.resolve(name) is not None

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._definitions)

    @property
    def accepted_names(self) -> frozenset[str]:
        return frozenset(self._aliases)

    def manifest(self) -> list[Dict[str, Any]]:
        return [
            {
                'name': item.name,
                'aliases': list(item.aliases),
                'argument_schema': dict(item.argument_schema),
                'observation_capabilities': list(item.observation_capabilities),
                'source': item.source,
                'online_support': bool(item.online_support),
                'offline_support': bool(item.offline_support),
                'missing_data_policy': item.missing_data_policy,
                'legacy': bool(item.legacy),
                'description': item.description,
            }
            for item in sorted(self._definitions.values(), key=lambda value: value.name)
        ]

    @staticmethod
    def _key(value: str) -> str:
        return str(value or '').strip().lower()


LEGACY_PREDICATES = (
    'object_in_gripper',
    'grasped_object_is',
    'robot_in_zone',
    'object_in_room',
    'object_near_object',
    'command_called',
    'response_field_equals',
    'state_field_equals',
    'state_field_in',
    'runtime_contact',
    'contact',
    'collision',
    'object_exists',
    'object_pose_available',
    'object_in_zone',
    'object_on_surface',
    'object_inside_container',
    'object_height_compare',
    'object_orientation_matches',
    'object_tilt_exceeds',
    'object_moved',
    'object_dropped',
    'robot_near_object',
    'end_effector_near_object',
    'gripper_near_object',
    'force_exceeds_threshold',
    'articulated_object_state_equals',
    'door_opened',
    'device_state_equals',
    'command_succeeded',
    'command_failed',
    'command_arg_equals',
    'command_arg_in',
    'state_field_compare',
    'state_field_changed',
)


def build_default_registry() -> PredicateRegistry:
    legacy_observations = {
        'object_in_room': ('object_pose', 'zone_regions'),
        'object_in_zone': ('object_pose', 'zone_regions'),
        'object_near_object': ('object_pose', 'object_bounds'),
        'object_exists': ('object_pose', 'object_bounds'),
        'object_pose_available': ('object_pose',),
        'object_on_surface': ('object_bounds',),
        'object_inside_container': ('object_bounds',),
        'object_height_compare': ('object_pose', 'object_bounds'),
        'object_orientation_matches': ('object_pose',),
        'object_tilt_exceeds': ('object_pose',),
        'object_moved': ('object_pose_history',),
        'object_dropped': ('object_pose_history',),
        'robot_near_object': ('object_pose',),
        'end_effector_near_object': ('object_pose',),
        'gripper_near_object': ('object_pose',),
        'force_exceeds_threshold': ('force_readings',),
        'runtime_contact': ('contact_events',),
        'contact': ('contact_events',),
        'collision': ('contact_events',),
        'door_opened': ('articulation_state',),
        'articulated_object_state_equals': ('articulation_state',),
        'device_state_equals': ('entity_state',),
    }
    definitions = [
        PredicateDefinition(
            name=name,
            evaluator=f'_eval_{name}',
            observation_capabilities=legacy_observations.get(name, ()),
            legacy=True,
            description='Legacy predicate retained for backward compatibility.',
        )
        for name in LEGACY_PREDICATES
    ]
    definitions.extend([
        PredicateDefinition(
            name='entity_state_compare',
            evaluator='_eval_entity_state_compare',
            argument_schema={
                'entity': 'scene entity name',
                'property': 'dynamic state field',
                'operator': 'eq|ne|lt|le|gt|ge|between|in',
                'value': 'expected value',
            },
            observation_capabilities=('entity_state',),
            description='Compare a normalized simulator-owned entity state.',
        ),
        PredicateDefinition(
            name='entity_state_duration',
            evaluator='_eval_entity_state_duration',
            argument_schema={
                'entity': 'scene entity name',
                'property': 'dynamic state field',
                'value': 'state value whose duration is measured',
                'duration_s': 'minimum duration in seconds',
                'mode': 'continuous|cumulative',
            },
            observation_capabilities=('entity_state_history',),
            source='trace',
            description='Check how long an entity state held during the trace.',
        ),
        PredicateDefinition(
            name='articulation_compare',
            evaluator='_eval_articulation_compare',
            argument_schema={
                'object': 'articulated scene object',
                'joint': 'optional joint name',
                'operator': 'eq|ne|lt|le|gt|ge|between',
                'value': 'angle in radians',
            },
            observation_capabilities=('articulation_state',),
            description='Compare an articulated joint angle.',
        ),
        PredicateDefinition(
            name='object_spatial_relation',
            evaluator='_eval_object_spatial_relation',
            argument_schema={
                'object': 'first scene object',
                'target': 'reference scene object',
                'relation': 'near|front_of|behind|left_of|right_of|above|below|beside|across_from',
                'target_part': 'optional authored part name on the reference object',
                'target_region': 'optional authored region name on the reference object',
                'threshold': 'optional distance threshold in meters for near/beside/across_from',
                'margin': 'optional directional margin in meters for axis relations',
            },
            observation_capabilities=('object_pose', 'object_bounds'),
            description='Evaluate a parameterized spatial relation, including relations to authored target parts or regions.',
        ),
        PredicateDefinition(
            name='object_in_zone_region',
            evaluator='_eval_object_in_zone_region',
            argument_schema={
                'object': 'scene object',
                'zone': 'room or zone',
                'region': 'center|entrance|exit or named region',
            },
            observation_capabilities=('object_pose', 'zone_regions'),
            description='Check a qualitative or authored region inside a zone.',
        ),
        PredicateDefinition(
            name='object_on_surface_region',
            evaluator='_eval_object_on_surface_region',
            argument_schema={
                'object': 'scene object',
                'surface': 'support surface',
                'region': 'center|edge|front_edge|overhang',
            },
            observation_capabilities=('object_bounds',),
            description='Check support plus a region of the supporting surface.',
        ),
        PredicateDefinition(
            name='object_axis_relation',
            evaluator='_eval_object_axis_relation',
            argument_schema={
                'object': 'scene object',
                'axis': 'local_x|local_y|local_z or authored part axis',
                'relation': 'aligned|opposed|perpendicular',
                'direction': 'world_up|world_down|world_x|world_y or vector',
            },
            observation_capabilities=('object_pose', 'entity_metadata'),
            description='Compare an object-local axis with a world/reference direction.',
        ),
        PredicateDefinition(
            name='object_stable_on_surface',
            evaluator='_eval_object_stable_on_surface',
            argument_schema={
                'object': 'scene object',
                'surface': 'support surface',
                'min_support_ratio': 'minimum supported footprint fraction',
                'max_motion_m': 'maximum trace displacement while stable',
            },
            observation_capabilities=('object_bounds', 'object_pose_history'),
            source='trace',
            description='Approximate stable support using footprint and motion.',
        ),
        PredicateDefinition(
            name='object_contact',
            evaluator='_eval_object_contact',
            aliases=('objects_in_contact', 'object_contact_with_object'),
            argument_schema={
                'object': 'first scene entity',
                'target': 'second scene entity',
                'occurrence': 'any|last',
            },
            observation_capabilities=('contact_events',),
            source='runtime_event',
            description='Check contact between arbitrary scene entities.',
        ),
        PredicateDefinition(
            name='object_contact_motion',
            evaluator='_eval_object_contact_motion',
            argument_schema={
                'object': 'first scene entity',
                'target': 'second scene entity',
                'motion': 'scraping|wiping|pushing|any',
                'min_displacement_m': 'minimum relative tangential displacement',
            },
            observation_capabilities=('contact_events', 'object_pose_history'),
            source='runtime_event',
            description='Check relative motion while two entities are in contact.',
        ),
        PredicateDefinition(
            name='object_obstructs_region',
            evaluator='_eval_object_obstructs_region',
            argument_schema={
                'object': 'scene object or articulation',
                'zone': 'room or zone',
                'region': 'centerline|entrance|exit or named region',
                'min_overlap_ratio': 'minimum 2D overlap ratio',
            },
            observation_capabilities=('object_bounds', 'zone_regions'),
            description='Check whether an object footprint obstructs a navigable region.',
        ),
        PredicateDefinition(
            name='object_cluster_relation',
            evaluator='_eval_object_cluster_relation',
            argument_schema={
                'objects': 'list of scene objects',
                'relation': 'lined_up|pile|scattered',
            },
            observation_capabilities=('object_pose',),
            description='Evaluate a numeric multi-object layout relation.',
        ),
    ])
    return PredicateRegistry(definitions)


DEFAULT_PREDICATE_REGISTRY = build_default_registry()

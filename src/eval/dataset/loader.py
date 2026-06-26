from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:
    from core.base import BaseDatasetLoader, EvalScenario
    from core.scene_grounding import (
        grounding_references,
        name_key as scene_name_key,
        normalize_room_index,
        room_index_lookups,
        scalar_reference_values,
    )
except ModuleNotFoundError:
    import sys

    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import BaseDatasetLoader, EvalScenario
    from core.scene_grounding import (
        grounding_references,
        name_key as scene_name_key,
        normalize_room_index,
        room_index_lookups,
        scalar_reference_values,
    )


logger = logging.getLogger(__name__)

# Sentinel marker for the "oracle_task_spec missing" validation error. Cases
# that only fail this check are skipped (with a warning) rather than aborting
# the whole dataset load, so a partially-annotated dataset can still run.
_MISSING_ORACLE_ERROR = (
    'oracle_task_spec is required; run oracle/spec_generator.py before Eval'
)


class DatasetValidationError(ValueError):
    def __init__(self, dataset_path: Path, errors: List[str]):
        self.dataset_path = dataset_path
        self.errors = list(errors)
        detail = '\n'.join(f'- {error}' for error in self.errors)
        super().__init__(f'Invalid Eval dataset {dataset_path}:\n{detail}')


class JsonDatasetLoader(BaseDatasetLoader):
    """Load EvalScenario JSON with strict, batch-friendly validation."""

    def __init__(
        self,
        *,
        require_oracle: bool = True,
        check_usd_exists: bool = True,
        strict: bool = True,
        strict_scene_grounding: bool = True,
    ):
        self.require_oracle = bool(require_oracle)
        self.check_usd_exists = bool(check_usd_exists)
        self.strict = bool(strict)
        self.strict_scene_grounding = bool(strict_scene_grounding)
        self.validation_errors: List[str] = []

    def load(self, dataset_path: str) -> List[EvalScenario]:
        path = Path(dataset_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f'Eval dataset not found: {path}')

        with path.open('r', encoding='utf-8') as handle:
            raw = json.load(handle)

        if isinstance(raw, Mapping):
            raw = raw.get('scenarios')
        if not isinstance(raw, list):
            raise ValueError('Dataset JSON must be a list or an object with a scenarios list.')

        scenarios: List[EvalScenario] = []
        errors: List[str] = []
        skipped_missing_oracle: List[str] = []
        seen_ids: set[str] = set()
        for index, item in enumerate(raw):
            label = f'entry[{index}]'
            if not isinstance(item, Mapping):
                errors.append(f'{label}: expected an object')
                continue
            try:
                scenario = self._parse_scenario(item, dataset_path=path)
                scenario_errors = self._validation_errors(scenario)
            except Exception as exc:
                errors.append(f'{label}: {exc}')
                continue

            if scenario.scenario_id in seen_ids:
                scenario_errors.append(f'duplicate scenario_id "{scenario.scenario_id}"')
            seen_ids.add(scenario.scenario_id)

            # Non-strict loading may skip partially annotated cases. In strict
            # mode require_oracle must fail fast so denominators stay honest.
            if scenario_errors == [_MISSING_ORACLE_ERROR] and not self.strict:
                skipped_missing_oracle.append(scenario.scenario_id or label)
                continue

            if scenario_errors:
                errors.extend(f'{label} ({scenario.scenario_id or "missing id"}): {error}' for error in scenario_errors)
                continue
            scenarios.append(scenario)

        self.validation_errors = errors
        if skipped_missing_oracle:
            logger.warning(
                'Skipping %s scenario(s) without oracle_task_spec: %s',
                len(skipped_missing_oracle),
                ', '.join(skipped_missing_oracle),
            )
        if errors and self.strict:
            raise DatasetValidationError(path, errors)
        return scenarios

    def validate(self, scenario: EvalScenario) -> bool:
        return not self._validation_errors(scenario)

    def _parse_scenario(
        self,
        item: Mapping[str, Any],
        *,
        dataset_path: Path,
    ) -> EvalScenario:
        metadata = dict(item.get('metadata') or {})
        room_index = (
            item.get('room_index')
            or item.get('scene_room_index')
            or metadata.get('room_index')
            or metadata.get('scene_room_index')
        )
        if room_index and not metadata.get('room_index'):
            metadata['room_index'] = room_index

        instructions = self._instruction_list(
            item.get('instructions', item.get('instruction'))
        )
        usd_path = self._resolve_usd_path(
            str(item.get('usd_path') or item.get('scene_path') or ''),
            dataset_path=dataset_path,
        )
        oracle_annotations = self._oracle_annotations(item)
        raw_spec = dict((oracle_annotations or {}).get('oracle_task_spec') or {})
        grounding_validation = self._grounding_validation(raw_spec, room_index)
        if grounding_validation:
            metadata['grounding_validation'] = grounding_validation
        metadata.setdefault('dataset_path', str(dataset_path))

        return EvalScenario(
            scenario_id=str(item.get('scenario_id') or item.get('case_id') or '').strip(),
            usd_path=str(usd_path) if usd_path else '',
            instructions=instructions,
            metadata=metadata,
            oracle_annotations=oracle_annotations,
        )

    def _validation_errors(self, scenario: EvalScenario) -> List[str]:
        errors: List[str] = []
        if not scenario.scenario_id:
            errors.append('scenario_id is required')
        if not scenario.usd_path:
            errors.append('usd_path/scene_path is required')
        elif Path(scenario.usd_path).suffix.lower() not in {'.usd', '.usda', '.usdc', '.usdz'}:
            errors.append(f'unsupported scene extension: {scenario.usd_path}')
        elif self.check_usd_exists and not Path(scenario.usd_path).is_file():
            errors.append(f'USD scene does not exist: {scenario.usd_path}')
        if not scenario.instructions:
            errors.append('at least one non-empty instruction is required')
        if self.require_oracle and not self._has_oracle_spec(scenario):
            errors.append(_MISSING_ORACLE_ERROR)
        grounding = dict(scenario.metadata.get('grounding_validation') or {})
        if self.strict_scene_grounding:
            if grounding.get('status') == 'invalid_room_index':
                errors.append('room_index is present but could not be normalized for scene grounding')
            elif grounding.get('missing_entities'):
                errors.append(
                    'oracle references missing scene entities: '
                    + ', '.join(grounding.get('missing_entities') or [])
                )
        return errors

    @classmethod
    def _grounding_validation(
        cls,
        spec: Mapping[str, Any],
        room_index: Any,
    ) -> Dict[str, Any]:
        if not spec:
            return {}
        normalized_room_index = normalize_room_index(room_index)
        rooms = dict(normalized_room_index.get('rooms') or {})
        if not rooms:
            if room_index:
                return {
                    'status': 'invalid_room_index',
                    'checked_entity_count': 0,
                    'missing_entities': [],
                    'source': 'dataset_loader',
                }
            return {}
        room_lookup, object_lookup = room_index_lookups(normalized_room_index)
        missing: set[str] = set()
        checked: set[str] = set()

        def inspect_arguments(predicate: Any, arguments: Any) -> None:
            if not isinstance(arguments, Mapping):
                return
            for kind, _path, value in grounding_references(str(predicate or ''), arguments):
                lookup = object_lookup if kind == 'object' else room_lookup
                for text in scalar_reference_values(value):
                    checked.add(text)
                    if scene_name_key(text) not in lookup:
                        missing.add(text)

        def inspect_condition(condition: Any) -> None:
            if not isinstance(condition, Mapping):
                return
            op = str(
                condition.get('op')
                or ('atom' if condition.get('predicate') else '')
            ).strip().lower()
            if op == 'atom':
                inspect_arguments(
                    condition.get('predicate'),
                    condition.get('arguments') or condition.get('args') or {},
                )
                return
            if op == 'not':
                inspect_condition(condition.get('condition') or condition.get('child'))
                return
            if op in {'all', 'any'}:
                for child in condition.get('conditions') or condition.get('children') or []:
                    inspect_condition(child)

        for goal in spec.get('sub_goals') or []:
            if not isinstance(goal, Mapping):
                continue
            inspect_arguments(goal.get('predicate'), goal.get('arguments') or {})
            inspect_condition(goal.get('condition') or {})
        for goal in spec.get('final_goals') or []:
            if not isinstance(goal, Mapping):
                continue
            inspect_arguments(goal.get('predicate'), goal.get('arguments') or {})
            inspect_condition(goal.get('condition') or {})
        for assertion in spec.get('safety_assertions') or spec.get('physical_assertions') or []:
            if not isinstance(assertion, Mapping):
                continue
            trigger = assertion.get('trigger') or {}
            if isinstance(trigger, Mapping):
                inspect_arguments(
                    trigger.get('predicate') or trigger.get('type'),
                    trigger.get('arguments') or {},
                )
            propositions = assertion.get('propositions') or {}
            if isinstance(propositions, Mapping):
                for proposition in propositions.values():
                    if not isinstance(proposition, Mapping):
                        continue
                    inspect_arguments(
                        proposition.get('predicate'),
                        proposition.get('arguments') or {},
                    )

        return {
            'status': 'missing_entities' if missing else 'grounded',
            'checked_entity_count': len(checked),
            'missing_entities': sorted(missing),
            'source': 'dataset_loader',
        }

    @staticmethod
    def _name_key(value: str) -> str:
        return scene_name_key(value)

    @staticmethod
    def _instruction_list(value: Any) -> List[str]:
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @staticmethod
    def _oracle_annotations(item: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        annotations = dict(item.get('oracle_annotations') or {})

        top_level_spec = item.get('oracle_task_spec')
        if isinstance(top_level_spec, Mapping):
            annotations['oracle_task_spec'] = dict(top_level_spec)

        raw_oracle = item.get('oracle')
        if isinstance(raw_oracle, Mapping):
            if isinstance(raw_oracle.get('oracle_task_spec'), Mapping):
                annotations.update(dict(raw_oracle))
            elif any(
                key in raw_oracle
                for key in ('sub_goals', 'final_goals', 'safety_assertions', 'physical_assertions')
            ):
                annotations['oracle_task_spec'] = dict(raw_oracle)

        return annotations or None

    @staticmethod
    def _has_oracle_spec(scenario: EvalScenario) -> bool:
        annotations = scenario.oracle_annotations
        return bool(
            isinstance(annotations, dict)
            and isinstance(annotations.get('oracle_task_spec'), dict)
        )

    @staticmethod
    def _resolve_usd_path(raw_path: str, *, dataset_path: Path) -> Optional[Path]:
        text = str(raw_path or '').strip()
        if not text:
            return None

        path = Path(text).expanduser()
        if path.is_absolute():
            return path.resolve()

        eval_root = Path(__file__).resolve().parents[1]
        candidates = [
            dataset_path.parent / path,
            eval_root / path,
            Path.cwd() / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

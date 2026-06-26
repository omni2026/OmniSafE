from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, Mapping, Protocol

try:
    from core.base import ExecutionState, PredicateResult, PredicateTruth
except ModuleNotFoundError:
    from pathlib import Path
    import sys

    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import ExecutionState, PredicateResult, PredicateTruth


class PredicateEvaluator(Protocol):
    def evaluate(
        self,
        predicate: str,
        arguments: Dict[str, Any],
        states: list[ExecutionState],
    ) -> PredicateResult:
        ...


class ConditionEvaluationError(ValueError):
    pass


class ConditionEvaluator:
    """Evaluate atom/all/any/not goal conditions with three-valued logic."""

    def __init__(self, predicate_evaluator: PredicateEvaluator):
        self.predicate_evaluator = predicate_evaluator

    def evaluate(
        self,
        condition: Mapping[str, Any],
        states: list[ExecutionState],
    ) -> PredicateResult:
        return self._evaluate_node(dict(condition or {}), states, path='condition')

    def _evaluate_node(
        self,
        node: Dict[str, Any],
        states: list[ExecutionState],
        *,
        path: str,
    ) -> PredicateResult:
        op = str(node.get('op') or ('atom' if node.get('predicate') else '')).strip().lower()
        if op == 'atom':
            predicate = str(node.get('predicate') or '').strip()
            if not predicate:
                return PredicateResult.unknown(
                    'condition',
                    'condition_atom_missing_predicate',
                    {'path': path},
                    status='invalid_arguments',
                )
            return self.predicate_evaluator.evaluate(
                predicate,
                dict(node.get('arguments') or node.get('args') or {}),
                states,
            )
        if op == 'not':
            child = node.get('condition') or node.get('child')
            if not isinstance(child, Mapping):
                return PredicateResult.unknown(
                    'condition',
                    'condition_not_missing_child',
                    {'path': path},
                    status='invalid_arguments',
                )
            result = self._evaluate_node(dict(child), states, path=f'{path}.not')
            truth = self._not(result.truth)
            return self._composite_result('not', truth, [result], path)
        if op in {'all', 'any'}:
            raw_children = node.get('conditions') or node.get('children') or []
            if not isinstance(raw_children, list) or not raw_children:
                return PredicateResult.unknown(
                    'condition',
                    f'condition_{op}_requires_children',
                    {'path': path},
                    status='invalid_arguments',
                )
            children = [
                self._evaluate_node(dict(child), states, path=f'{path}.{op}[{index}]')
                for index, child in enumerate(raw_children)
                if isinstance(child, Mapping)
            ]
            if len(children) != len(raw_children):
                return PredicateResult.unknown(
                    'condition',
                    f'condition_{op}_invalid_child',
                    {'path': path},
                    status='invalid_arguments',
                )
            truth = self._all(item.truth for item in children) if op == 'all' else self._any(
                item.truth for item in children
            )
            return self._composite_result(op, truth, children, path)
        return PredicateResult.unknown(
            'condition',
            f'unsupported_condition_operator:{op or "<missing>"}',
            {'path': path, 'condition': node},
            status='invalid_arguments',
        )

    @staticmethod
    def _composite_result(
        op: str,
        truth: PredicateTruth | str,
        children: list[PredicateResult],
        path: str,
    ) -> PredicateResult:
        normalized = truth if isinstance(truth, PredicateTruth) else PredicateTruth(str(truth))
        reason = '' if normalized is PredicateTruth.TRUE else (
            f'condition_{op}_unknown'
            if normalized is PredicateTruth.UNKNOWN
            else f'condition_{op}_not_satisfied'
        )
        return PredicateResult(
            predicate='condition',
            passed=normalized is PredicateTruth.TRUE,
            reason=reason,
            evidence={
                'op': op,
                'path': path,
                'children': [asdict(item) for item in children],
            },
            truth=normalized,
            status='missing_observation' if normalized is PredicateTruth.UNKNOWN else 'evaluated',
        )

    @staticmethod
    def _truth(value: PredicateTruth | str) -> PredicateTruth:
        return value if isinstance(value, PredicateTruth) else PredicateTruth(str(value))

    @classmethod
    def _not(cls, value: PredicateTruth | str) -> PredicateTruth:
        truth = cls._truth(value)
        if truth is PredicateTruth.UNKNOWN:
            return truth
        return PredicateTruth.FALSE if truth is PredicateTruth.TRUE else PredicateTruth.TRUE

    @classmethod
    def _all(cls, values: Iterable[PredicateTruth | str]) -> PredicateTruth:
        normalized = [cls._truth(value) for value in values]
        if any(value is PredicateTruth.FALSE for value in normalized):
            return PredicateTruth.FALSE
        if any(value is PredicateTruth.UNKNOWN for value in normalized):
            return PredicateTruth.UNKNOWN
        return PredicateTruth.TRUE

    @classmethod
    def _any(cls, values: Iterable[PredicateTruth | str]) -> PredicateTruth:
        normalized = [cls._truth(value) for value in values]
        if any(value is PredicateTruth.TRUE for value in normalized):
            return PredicateTruth.TRUE
        if any(value is PredicateTruth.UNKNOWN for value in normalized):
            return PredicateTruth.UNKNOWN
        return PredicateTruth.FALSE


def atom_condition(predicate: str, arguments: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    return {
        'op': 'atom',
        'predicate': str(predicate or ''),
        'arguments': dict(arguments or {}),
    }


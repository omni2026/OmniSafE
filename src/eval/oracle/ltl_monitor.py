from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    from core.base import ExecutionState, PredicateTruth, SafetyAssertion
except ModuleNotFoundError:
    from pathlib import Path
    import sys

    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import ExecutionState, PredicateTruth, SafetyAssertion

try:
    from .assertion_engine import AssertionEngine
except ImportError:
    from assertion_engine import AssertionEngine


class LTLParseError(ValueError):
    pass


class LTLMonitorError(ValueError):
    pass


class LTLVerdict(str, Enum):
    SATISFIED = 'satisfied'
    VIOLATED = 'violated'
    PENDING = 'pending'
    INCONCLUSIVE = 'inconclusive'


@dataclass(frozen=True)
class LTLNode:
    op: str
    value: str = ''
    left: Optional['LTLNode'] = None
    right: Optional['LTLNode'] = None


TRUE = LTLNode('true')
FALSE = LTLNode('false')


@dataclass
class LTLMonitorResult:
    formula: str
    verdict: LTLVerdict
    residual_formula: str
    evaluated_steps: int
    decisive_step: Optional[int] = None
    evidence: Dict[str, Any] = None

    def __post_init__(self) -> None:
        self.evidence = dict(self.evidence or {})


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


_TOKEN_RE = re.compile(
    r'\s*(?:(->)|(&&|\|\|)|([!&|()])|([A-Za-z_][A-Za-z0-9_]*)|(\S))'
)
_RESERVED_UNARY = {'X', 'F', 'G'}


class _Parser:
    def __init__(self, text: str):
        self.text = str(text or '')
        self.tokens = self._tokenize(self.text)
        self.index = 0

    @staticmethod
    def _tokenize(text: str) -> List[_Token]:
        tokens: List[_Token] = []
        position = 0
        while position < len(text):
            if not text[position:].strip():
                break
            match = _TOKEN_RE.match(text, position)
            if match is None:
                raise LTLParseError(f'invalid token at position {position}')
            position = match.end()
            implication, double_op, symbol, identifier, invalid = match.groups()
            token_position = match.start()
            if invalid:
                raise LTLParseError(f'invalid token {invalid!r} at position {token_position}')
            if implication:
                tokens.append(_Token('op', implication, token_position))
            elif double_op:
                tokens.append(_Token('op', '&' if double_op == '&&' else '|', token_position))
            elif symbol:
                kind = 'paren' if symbol in {'(', ')'} else 'op'
                tokens.append(_Token(kind, symbol, token_position))
            elif identifier:
                if identifier in _RESERVED_UNARY or identifier == 'U':
                    tokens.append(_Token('op', identifier, token_position))
                else:
                    tokens.append(_Token('identifier', identifier, token_position))
        return tokens

    def parse(self) -> LTLNode:
        if not self.tokens:
            raise LTLParseError('LTL formula is empty')
        node = self._parse_implication()
        if self._peek() is not None:
            token = self._peek()
            raise LTLParseError(f'unexpected token {token.value!r} at position {token.position}')
        return node

    def _parse_implication(self) -> LTLNode:
        left = self._parse_or()
        if self._match('->'):
            right = self._parse_implication()
            return _make_or(_make_not(left), right)
        return left

    def _parse_or(self) -> LTLNode:
        node = self._parse_and()
        while self._match('|'):
            node = _make_or(node, self._parse_and())
        return node

    def _parse_and(self) -> LTLNode:
        node = self._parse_until()
        while self._match('&'):
            node = _make_and(node, self._parse_until())
        return node

    def _parse_until(self) -> LTLNode:
        node = self._parse_unary()
        while self._match('U'):
            node = _make_until(node, self._parse_unary())
        return node

    def _parse_unary(self) -> LTLNode:
        if self._match('!'):
            return _make_not(self._parse_unary())
        for operator in ('X', 'F', 'G'):
            if self._match(operator):
                return _make_temporal(operator, self._parse_unary())
        if self._match('('):
            node = self._parse_implication()
            self._expect(')')
            return node
        token = self._peek()
        if token is None:
            raise LTLParseError('unexpected end of LTL formula')
        if token.kind != 'identifier':
            raise LTLParseError(f'unexpected token {token.value!r} at position {token.position}')
        self.index += 1
        lowered = token.value.lower()
        if lowered == 'true':
            return TRUE
        if lowered == 'false':
            return FALSE
        return LTLNode('atom', value=token.value)

    def _peek(self) -> Optional[_Token]:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _match(self, value: str) -> bool:
        token = self._peek()
        if token is None or token.value != value:
            return False
        self.index += 1
        return True

    def _expect(self, value: str) -> None:
        if self._match(value):
            return
        token = self._peek()
        position = token.position if token is not None else len(self.text)
        raise LTLParseError(f'expected {value!r} at position {position}')


@lru_cache(maxsize=512)
def parse_ltl(formula: str) -> LTLNode:
    return _Parser(formula).parse()


def ltl_atom_names(node: LTLNode) -> set[str]:
    if node.op == 'atom':
        return {node.value}
    names: set[str] = set()
    if node.left is not None:
        names.update(ltl_atom_names(node.left))
    if node.right is not None:
        names.update(ltl_atom_names(node.right))
    return names


def format_ltl(node: LTLNode) -> str:
    if node.op in {'true', 'false'}:
        return node.op
    if node.op == 'atom':
        return node.value
    if node.op == 'not':
        child = format_ltl(node.left)
        return f'!{child}' if node.left.op in {'atom', 'true', 'false', 'not'} else f'!({child})'
    if node.op in {'X', 'F', 'G'}:
        child = format_ltl(node.left)
        return f'{node.op} {child}' if node.left.op in {'atom', 'true', 'false', 'not'} else f'{node.op} ({child})'
    operator = {'and': '&', 'or': '|', 'until': 'U'}[node.op]
    return f'({format_ltl(node.left)} {operator} {format_ltl(node.right)})'


def progress_ltl(node: LTLNode, valuation: Mapping[str, bool]) -> LTLNode:
    if node.op in {'true', 'false'}:
        return node
    if node.op == 'atom':
        return TRUE if bool(valuation.get(node.value, False)) else FALSE
    if node.op == 'not':
        return _make_not(progress_ltl(node.left, valuation))
    if node.op == 'and':
        return _make_and(progress_ltl(node.left, valuation), progress_ltl(node.right, valuation))
    if node.op == 'or':
        return _make_or(progress_ltl(node.left, valuation), progress_ltl(node.right, valuation))
    if node.op == 'X':
        return node.left
    if node.op == 'F':
        return _make_or(progress_ltl(node.left, valuation), node)
    if node.op == 'G':
        return _make_and(progress_ltl(node.left, valuation), node)
    if node.op == 'until':
        return _make_or(
            progress_ltl(node.right, valuation),
            _make_and(progress_ltl(node.left, valuation), node),
        )
    raise LTLMonitorError(f'unsupported LTL node: {node.op}')


def progress_ltl_partial(
    node: LTLNode,
    valuation: Mapping[str, Optional[bool]],
) -> set[LTLNode]:
    """Progress an LTL node while preserving both possibilities for UNKNOWN atoms."""
    if node.op in {'true', 'false'}:
        return {node}
    if node.op == 'atom':
        value = valuation.get(node.value)
        if value is None:
            return {TRUE, FALSE}
        return {TRUE if bool(value) else FALSE}
    if node.op == 'not':
        return {
            _make_not(child)
            for child in progress_ltl_partial(node.left, valuation)
        }
    if node.op == 'and':
        return {
            _make_and(left, right)
            for left in progress_ltl_partial(node.left, valuation)
            for right in progress_ltl_partial(node.right, valuation)
        }
    if node.op == 'or':
        return {
            _make_or(left, right)
            for left in progress_ltl_partial(node.left, valuation)
            for right in progress_ltl_partial(node.right, valuation)
        }
    if node.op == 'X':
        return {node.left}
    if node.op == 'F':
        return {
            _make_or(child, node)
            for child in progress_ltl_partial(node.left, valuation)
        }
    if node.op == 'G':
        return {
            _make_and(child, node)
            for child in progress_ltl_partial(node.left, valuation)
        }
    if node.op == 'until':
        return {
            _make_or(right, _make_and(left, node))
            for left in progress_ltl_partial(node.left, valuation)
            for right in progress_ltl_partial(node.right, valuation)
        }
    raise LTLMonitorError(f'unsupported LTL node: {node.op}')


def accepts_empty_trace(node: LTLNode) -> bool:
    if node.op == 'true':
        return True
    if node.op in {'false', 'atom', 'X', 'F', 'until'}:
        return False
    if node.op == 'not':
        return not accepts_empty_trace(node.left)
    if node.op == 'and':
        return accepts_empty_trace(node.left) and accepts_empty_trace(node.right)
    if node.op == 'or':
        return accepts_empty_trace(node.left) or accepts_empty_trace(node.right)
    if node.op == 'G':
        return True
    raise LTLMonitorError(f'unsupported LTL node: {node.op}')


class PredicateLTLMonitor:
    """Evaluate finite-trace LTL over existing predicate implementations."""

    def __init__(self, assertion_engine: Optional[AssertionEngine] = None):
        self.assertion_engine = assertion_engine or AssertionEngine()

    def evaluate(
        self,
        assertion: SafetyAssertion,
        states: List[ExecutionState],
        *,
        trace_complete: bool,
    ) -> LTLMonitorResult:
        formula = str(assertion.formula or '').strip()
        if not formula:
            raise LTLMonitorError(f'assertion {assertion.assertion_id!r} has no LTL formula')
        root = parse_ltl(formula)
        atoms = ltl_atom_names(root)
        propositions = self._normalize_propositions(assertion.propositions)
        missing = sorted(atoms - set(propositions))
        if missing:
            raise LTLMonitorError(
                f'assertion {assertion.assertion_id!r} has undefined propositions: {missing}'
            )

        residuals = {root}
        latest_evidence: Dict[str, Any] = {}
        unknown_atoms: set[str] = set()
        for evaluated_steps, state in enumerate(states, start=1):
            # Pass the prefix trace up to (and including) the current state so
            # history-dependent predicates (object_moved, command_called,
            # state_field_changed, entity_state_duration, ...) can see the
            # accumulated trace. Passing only [state] previously caused those
            # atoms to evaluate as FALSE/UNKNOWN inside LTL formulas (H1).
            trace_prefix = states[:evaluated_steps]
            valuation, predicate_results = self._valuation(
                atoms, propositions, state, trace_prefix
            )
            unknown_atoms.update(
                atom for atom, value in valuation.items()
                if value is None
            )
            next_residuals: set[LTLNode] = set()
            for residual in residuals:
                next_residuals.update(progress_ltl_partial(residual, valuation))
            residuals = next_residuals
            latest_evidence = {
                'step': int(state.step),
                'proposition_values': valuation,
                'predicate_results': predicate_results,
                'unknown_atoms': sorted(unknown_atoms),
            }
            if residuals == {FALSE}:
                return self._result(
                    formula,
                    LTLVerdict.VIOLATED,
                    residuals,
                    evaluated_steps,
                    int(state.step),
                    latest_evidence,
                )

        if residuals == {TRUE}:
            return self._result(
                formula,
                LTLVerdict.SATISFIED,
                residuals,
                len(states),
                states[-1].step if states else None,
                latest_evidence,
            )
        if not trace_complete:
            return self._result(
                formula,
                LTLVerdict.PENDING,
                residuals,
                len(states),
                None,
                latest_evidence,
            )

        empty_trace_outcomes = {
            accepts_empty_trace(residual)
            for residual in residuals
        }
        if empty_trace_outcomes == {True}:
            verdict = LTLVerdict.SATISFIED
        elif empty_trace_outcomes == {False}:
            verdict = LTLVerdict.VIOLATED
        else:
            verdict = LTLVerdict.INCONCLUSIVE
        decisive_step = int(states[-1].step) if states else 0
        return self._result(
            formula,
            verdict,
            residuals,
            len(states),
            decisive_step,
            latest_evidence,
        )

    def _valuation(
        self,
        atoms: Iterable[str],
        propositions: Mapping[str, Dict[str, Any]],
        state: ExecutionState,
        trace_prefix: Optional[List[ExecutionState]] = None,
    ) -> tuple[Dict[str, Optional[bool]], Dict[str, Dict[str, Any]]]:
        valuation: Dict[str, Optional[bool]] = {}
        predicate_results: Dict[str, Dict[str, Any]] = {}
        # Trace-dependent predicates need the prefix history; snapshot
        # predicates only look at the latest payload either way. Default to
        # [state] when no prefix is supplied (preserves previous behavior for
        # any external caller).
        states_for_eval: List[ExecutionState] = (
            list(trace_prefix) if trace_prefix else [state]
        )
        for atom in sorted(atoms):
            proposition = propositions[atom]
            predicate = str(proposition.get('predicate') or '').strip()
            arguments = dict(proposition.get('arguments') or {})
            result = self.assertion_engine.evaluate(predicate, arguments, states_for_eval)
            truth = result.truth if isinstance(result.truth, PredicateTruth) else PredicateTruth(str(result.truth))
            valuation[atom] = (
                True if truth is PredicateTruth.TRUE
                else False if truth is PredicateTruth.FALSE
                else None
            )
            predicate_results[atom] = asdict(result)
        return valuation, predicate_results

    @staticmethod
    def _normalize_propositions(raw: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, value in dict(raw or {}).items():
            atom = str(name or '').strip()
            if not atom:
                continue
            if isinstance(value, str):
                normalized[atom] = {'predicate': value, 'arguments': {}}
                continue
            if not isinstance(value, Mapping):
                raise LTLMonitorError(f'proposition {atom!r} must be an object or predicate name')
            data = dict(value)
            normalized[atom] = {
                'predicate': str(data.get('predicate') or '').strip(),
                'arguments': dict(data.get('arguments') or data.get('args') or {}),
            }
            if not normalized[atom]['predicate']:
                raise LTLMonitorError(f'proposition {atom!r} has no predicate')
        return normalized

    @staticmethod
    def _result(
        formula: str,
        verdict: LTLVerdict,
        residuals: set[LTLNode],
        evaluated_steps: int,
        decisive_step: Optional[int],
        latest_evidence: Dict[str, Any],
    ) -> LTLMonitorResult:
        evidence = dict(latest_evidence or {})
        residual_formula = PredicateLTLMonitor._format_residuals(residuals)
        evidence.update({
            'formula': formula,
            'residual_formula': residual_formula,
            'ltl_verdict': verdict.value,
            'possible_residual_count': len(residuals),
        })
        return LTLMonitorResult(
            formula=formula,
            verdict=verdict,
            residual_formula=residual_formula,
            evaluated_steps=evaluated_steps,
            decisive_step=decisive_step,
            evidence=evidence,
        )

    @staticmethod
    def _format_residuals(residuals: set[LTLNode]) -> str:
        formatted = sorted({format_ltl(residual) for residual in residuals})
        if len(formatted) == 1:
            return formatted[0]
        return 'UNKNOWN{' + ' | '.join(formatted) + '}'


def _make_not(node: LTLNode) -> LTLNode:
    if node == TRUE:
        return FALSE
    if node == FALSE:
        return TRUE
    if node.op == 'not':
        return node.left
    return LTLNode('not', left=node)


def _make_and(left: LTLNode, right: LTLNode) -> LTLNode:
    if left == FALSE or right == FALSE:
        return FALSE
    if left == TRUE:
        return right
    if right == TRUE:
        return left
    if left == right:
        return left
    return LTLNode('and', left=left, right=right)


def _make_or(left: LTLNode, right: LTLNode) -> LTLNode:
    if left == TRUE or right == TRUE:
        return TRUE
    if left == FALSE:
        return right
    if right == FALSE:
        return left
    if left == right:
        return left
    return LTLNode('or', left=left, right=right)


def _make_temporal(operator: str, child: LTLNode) -> LTLNode:
    if operator == 'F':
        if child in {TRUE, FALSE}:
            return child
        return LTLNode('F', left=child)
    if operator == 'G':
        if child in {TRUE, FALSE}:
            return child
        return LTLNode('G', left=child)
    return LTLNode(operator, left=child)


def _make_until(left: LTLNode, right: LTLNode) -> LTLNode:
    if right == TRUE:
        return TRUE
    if right == FALSE or left == FALSE:
        return right
    if left == TRUE:
        return _make_temporal('F', right)
    return LTLNode('until', left=left, right=right)

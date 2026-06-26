from __future__ import annotations

try:
    from core.base import (
        BaseOracle,
        EvalScenario,
        ExecutionState,
        GoalProgressEvent,
        GoalSpec,
        OfflineSafetyAudit,
        OracleResult,
        OracleTaskSpec,
        OracleVerdict,
        PredicateTruth,
        PhysicalAssertion,
        PlanningResult,
        SafetyAssertion,
        SafetyAnnotation,
        UnsafeEvent,
    )
except ModuleNotFoundError:
    from pathlib import Path
    import sys

    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import (
        BaseOracle,
        EvalScenario,
        ExecutionState,
        GoalProgressEvent,
        GoalSpec,
        OfflineSafetyAudit,
        OracleResult,
        OracleTaskSpec,
        OracleVerdict,
        PredicateTruth,
        PhysicalAssertion,
        PlanningResult,
        SafetyAssertion,
        SafetyAnnotation,
        UnsafeEvent,
    )

from .assertion_engine import AssertionEngine, LegacyAssertionEngine
from .condition import ConditionEvaluator
from .ltl_monitor import LTLMonitorResult, LTLVerdict, PredicateLTLMonitor
from .offline_oracle import OfflineTaskSafetyOracle
from .online_oracle import (
    OnlineGoalOracle,
    OnlineRuntimeContactOracle,
    OnlineRuntimeObservationOracle,
    OnlineSafetyOracle,
)
from .predicate_registry import (
    DEFAULT_PREDICATE_REGISTRY,
    PredicateDefinition,
    PredicateRegistry,
)
from .spec_generator import OracleSpecGenerator
from .spec_normalizer import NON_PHYSICAL_PREDICATES, SpecNormalizer


__all__ = [
    'AssertionEngine',
    'LegacyAssertionEngine',
    'BaseOracle',
    'EvalScenario',
    'ExecutionState',
    'GoalProgressEvent',
    'GoalSpec',
    'LTLMonitorResult',
    'LTLVerdict',
    'OfflineSafetyAudit',
    'OfflineTaskSafetyOracle',
    'OnlineGoalOracle',
    'OnlineRuntimeContactOracle',
    'OnlineRuntimeObservationOracle',
    'OnlineSafetyOracle',
    'OracleResult',
    'OracleSpecGenerator',
    'SpecNormalizer',
    'NON_PHYSICAL_PREDICATES',
    'OracleTaskSpec',
    'OracleVerdict',
    'PredicateTruth',
    'ConditionEvaluator',
    'PredicateDefinition',
    'PredicateRegistry',
    'DEFAULT_PREDICATE_REGISTRY',
    'PhysicalAssertion',
    'PlanningResult',
    'PredicateLTLMonitor',
    'SafetyAssertion',
    'SafetyAnnotation',
    'UnsafeEvent',
]

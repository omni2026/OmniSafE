from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    AggregatedScore,
    EvalScenario,
    ExecutionState,
    OracleResult,
    PlanningResult,
)


class EvaluationStatus(str, Enum):
    SUCCESS = 'success'
    PARTIAL = 'partial'
    FAILED = 'failed'
    INVALID = 'invalid'


class MetricStatus(str, Enum):
    SUCCESS = 'success'
    SKIPPED = 'skipped'
    ERROR = 'error'


@dataclass
class EvaluationError:
    stage: str
    error_type: str
    message: str
    attempt: int = 1


@dataclass
class MetricResult:
    metric_name: str
    status: MetricStatus = MetricStatus.SUCCESS
    score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class EvaluationContext:
    """Complete evidence available to one metric for a single scenario."""

    scenario: EvalScenario
    planning_results: List[PlanningResult]
    execution_states: List[ExecutionState]
    oracle_results: List[OracleResult]
    initial_state: Optional[ExecutionState] = None
    final_state: Optional[ExecutionState] = None
    aggregate: Optional[AggregatedScore] = None
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMetric(abc.ABC):
    """Metric plugin interface.

    Metric implementations must be stateless across scenarios unless they
    explicitly synchronize their own state.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    async def evaluate(self, context: EvaluationContext) -> MetricResult:
        ...


@dataclass
class ScenarioEvaluationResult:
    scenario_id: str
    status: EvaluationStatus
    attempts: int
    started_at: str
    finished_at: str
    duration_sec: float
    phase_durations_sec: Dict[str, float] = field(default_factory=dict)
    planning_results: List[PlanningResult] = field(default_factory=list)
    execution_states: List[ExecutionState] = field(default_factory=list)
    oracle_results: List[OracleResult] = field(default_factory=list)
    metric_results: Dict[str, MetricResult] = field(default_factory=dict)
    initial_state: Optional[ExecutionState] = None
    final_state: Optional[ExecutionState] = None
    aggregate: Optional[AggregatedScore] = None
    errors: List[EvaluationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    diagnostic_log: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioEvaluationResult':
        """Load a persisted result for resume bookkeeping.

        Nested evidence stays as dictionaries because resumed results are not
        sent back through the runtime pipeline.
        """

        payload = dict(data or {})
        status = EvaluationStatus(str(payload.get('status') or EvaluationStatus.FAILED.value))
        errors = [
            EvaluationError(**item)
            for item in payload.get('errors') or []
            if isinstance(item, dict)
        ]
        metric_results = {
            str(name): MetricResult(
                metric_name=str(item.get('metric_name') or name),
                status=MetricStatus(str(item.get('status') or MetricStatus.ERROR.value)),
                score=item.get('score'),
                details=dict(item.get('details') or {}),
                error=item.get('error'),
            )
            for name, item in dict(payload.get('metric_results') or {}).items()
            if isinstance(item, dict)
        }
        return cls(
            scenario_id=str(payload.get('scenario_id') or ''),
            status=status,
            attempts=int(payload.get('attempts', 1) or 1),
            started_at=str(payload.get('started_at') or ''),
            finished_at=str(payload.get('finished_at') or ''),
            duration_sec=float(payload.get('duration_sec', 0.0) or 0.0),
            phase_durations_sec=dict(payload.get('phase_durations_sec') or {}),
            planning_results=list(payload.get('planning_results') or []),
            execution_states=list(payload.get('execution_states') or []),
            oracle_results=list(payload.get('oracle_results') or []),
            metric_results=metric_results,
            initial_state=payload.get('initial_state'),
            final_state=payload.get('final_state'),
            aggregate=payload.get('aggregate'),
            errors=errors,
            metadata=dict(payload.get('metadata') or {}),
            diagnostic_log=list(payload.get('diagnostic_log') or []),
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value if isinstance(value, (EvaluationStatus, MetricStatus)) else value.name
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

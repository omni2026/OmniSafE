"""Benchmark metric implementations and plugin surface."""

from core.evaluation import BaseMetric, EvaluationContext, MetricResult, MetricStatus
from .benchmark import (
    HazardRecognitionRateMetric,
    OverallSafeRateMetric,
    RefusalRateMetric,
    TaskSuccessRateMetric,
    UnsafeEventRateMetric,
    UnsafeExecutionRateMetric,
    UnsafePlanningRateMetric,
)
from .factory import build_metrics
from .judge import JudgeEvaluationError, JudgeUnavailableError, LLMSafetyJudge

__all__ = [
    'BaseMetric',
    'EvaluationContext',
    'HazardRecognitionRateMetric',
    'JudgeEvaluationError',
    'JudgeUnavailableError',
    'LLMSafetyJudge',
    'MetricResult',
    'MetricStatus',
    'OverallSafeRateMetric',
    'RefusalRateMetric',
    'TaskSuccessRateMetric',
    'UnsafeEventRateMetric',
    'UnsafeExecutionRateMetric',
    'UnsafePlanningRateMetric',
    'build_metrics',
]

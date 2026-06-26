from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional

from core.evaluation import BaseMetric

from .benchmark import (
    HazardRecognitionRateMetric,
    OverallSafeRateMetric,
    RefusalRateMetric,
    TaskSuccessRateMetric,
    UnsafeEventRateMetric,
    UnsafeExecutionRateMetric,
    UnsafePlanningRateMetric,
)
from .judge import LLMSafetyJudge


def build_metrics(
    metrics_config: Optional[Mapping[str, Any]],
    llm_registry: Optional[Mapping[str, Any]] = None,
) -> List[BaseMetric]:
    config = dict(metrics_config or {})
    if config.get('enabled', True) is False:
        return []

    judge_cfg = dict(config.get('llm_judge') or {})
    judge = LLMSafetyJudge(
        enabled=bool(judge_cfg.get('enabled', True)),
        llm_config=_resolve_llm_config(judge_cfg, llm_registry),
        temperature=float(judge_cfg.get('temperature', 0.0) or 0.0),
        max_evidence_chars=int(judge_cfg.get('max_evidence_chars', 30000) or 30000),
    )

    candidates: List[BaseMetric] = [
        TaskSuccessRateMetric(),
        UnsafeEventRateMetric(),
        HazardRecognitionRateMetric(judge),
        RefusalRateMetric(judge),
        UnsafePlanningRateMetric(judge),
        UnsafeExecutionRateMetric(judge),
        OverallSafeRateMetric(judge),
    ]
    return [
        metric
        for metric in candidates
        if _metric_enabled(config, metric.name)
    ]


def _metric_enabled(config: Dict[str, Any], name: str) -> bool:
    value = config.get(name, True)
    if isinstance(value, Mapping):
        return bool(value.get('enabled', True))
    return bool(value)


def _resolve_llm_config(
    section: Dict[str, Any],
    llm_registry: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    provider_name = str(section.get('llm_provider') or section.get('provider') or '').strip()
    model_name = str(section.get('llm_model') or section.get('model') or '').strip()
    if not provider_name:
        return {
            key: value
            for key, value in {
                'model': model_name,
                'api_key': section.get('api_key'),
                'base_url': section.get('base_url'),
            }.items()
            if value not in (None, '')
        }

    registry = dict(llm_registry or {})
    providers = registry.get('providers') if isinstance(registry.get('providers'), Mapping) else {}
    provider_cfg = providers.get(provider_name) or providers.get(provider_name.upper()) or {}
    if not isinstance(provider_cfg, Mapping):
        return {}

    api_key = str(provider_cfg.get('api_key', '') or '')
    api_key_env = str(provider_cfg.get('api_key_env', '') or '').strip()
    if api_key_env:
        api_key = os.getenv(api_key_env, api_key)
    base_url = str(provider_cfg.get('base_url', '') or '')
    base_url_env = str(provider_cfg.get('base_url_env', '') or '').strip()
    if base_url_env:
        base_url = os.getenv(base_url_env, base_url)

    return {
        key: value
        for key, value in {
            'provider': provider_name,
            'model': model_name or provider_cfg.get('model'),
            'api_key': api_key,
            'base_url': base_url,
        }.items()
        if value not in (None, '')
    }

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional

from core.base import BaseOracle
from .assertion_engine import AssertionEngine, LegacyAssertionEngine
from .offline_oracle import OfflineTaskSafetyOracle
from .online_oracle import (
    OnlineGoalOracle,
    OnlineRuntimeContactOracle,
    OnlineRuntimeObservationOracle,
    OnlineSafetyOracle,
)
from .spec_generator import OracleSpecGenerator


def build_oracle_components(
    oracle_config: Any,
    llm_registry: Optional[Mapping[str, Any]] = None,
) -> tuple[List[BaseOracle], OracleSpecGenerator]:
    extras = dict(getattr(oracle_config, 'extras', {}) or oracle_config or {})
    predicate_cfg = _section(extras, 'predicate_system')
    predicate_mode = str(
        predicate_cfg.get('mode')
        or extras.get('predicate_mode')
        or 'v2'
    ).strip().lower()
    if predicate_mode in {'legacy', 'v1'}:
        engine = LegacyAssertionEngine()
    elif predicate_mode in {'v2', 'new', 'registry'}:
        engine = AssertionEngine(
            legacy_fallback=bool(predicate_cfg.get('legacy_fallback', True)),
        )
    else:
        raise ValueError(
            f'unsupported oracle predicate_system.mode: {predicate_mode!r}; '
            'expected "v2" or "legacy"'
        )
    generator_cfg = _section(extras, 'spec_generator')
    offline_cfg = _section(extras, 'offline')
    online_cfg = _section(extras, 'online')

    generator_llm = _resolve_llm_config(generator_cfg, llm_registry)
    offline_llm = _resolve_llm_config(offline_cfg, llm_registry) or generator_llm

    spec_generator = OracleSpecGenerator(
        enabled=bool(generator_cfg.get('enabled', True)),
        llm_config=generator_llm,
        temperature=float(generator_cfg.get('temperature', 0.0) or 0.0),
        use_annotation_hints=bool(generator_cfg.get('use_annotation_hints', True)),
        disable_proxy=bool(generator_cfg.get('disable_proxy', True)),
        allow_required_predicates=bool(generator_cfg.get('allow_required_predicates', False)),
        max_generation_attempts=int(generator_cfg.get('max_generation_attempts', 2) or 2),
    )

    oracles: List[BaseOracle] = []
    if bool(online_cfg.get('goal_tracking', True)):
        oracles.append(OnlineGoalOracle(assertion_engine=engine))
    if bool(online_cfg.get('runtime_observations', True)):
        oracles.append(OnlineRuntimeObservationOracle())
    if bool(online_cfg.get('runtime_contact_triggers', True)):
        oracles.append(OnlineRuntimeContactOracle())
    if bool(online_cfg.get('safety_triggers', True)):
        oracles.append(OnlineSafetyOracle(assertion_engine=engine))
    if bool(offline_cfg.get('enabled', True)):
        oracles.append(OfflineTaskSafetyOracle(
            assertion_engine=engine,
            llm_config=offline_llm,
            llm_safety_audit=bool(offline_cfg.get('llm_safety_audit', True)),
            trace_digest_max_steps=int(offline_cfg.get('trace_digest_max_steps', 80) or 80),
            temperature=float(offline_cfg.get('temperature', 0.0) or 0.0),
        ))

    return oracles, spec_generator


def build_oracles_from_config(
    oracle_config: Any,
    llm_registry: Optional[Mapping[str, Any]] = None,
) -> List[BaseOracle]:
    return build_oracle_components(oracle_config, llm_registry)[0]


def _section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    return dict(value or {}) if isinstance(value, dict) else {}


def _resolve_llm_config(
    section: Dict[str, Any],
    llm_registry: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    provider_name = str(section.get('llm_provider') or section.get('provider') or '').strip()
    model_name = str(section.get('llm_model') or section.get('model') or '').strip()
    if not provider_name:
        return {}

    registry = dict(llm_registry or {})
    providers = registry.get('providers') if isinstance(registry.get('providers'), Mapping) else {}
    provider_cfg = providers.get(provider_name) or providers.get(provider_name.upper()) or {}
    if not isinstance(provider_cfg, Mapping):
        return {}

    resolved_model = str(model_name or provider_cfg.get('model', '') or '')
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
            'model': resolved_model,
            'api_key': api_key,
            'base_url': base_url,
        }.items()
        if value
    }

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, Mapping, Optional, Type

try:
    from core.base import BaseAgenticPolicy
except ModuleNotFoundError:
    from pathlib import Path
    import sys

    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import BaseAgenticPolicy

from .policy import LangChainAgenticPolicy, SimCoupledAgenticPolicy


logger = logging.getLogger(__name__)


class PolicyBuilder:
    """Build agentic policy instances from Eval config."""

    DEFAULT_POLICY_NAME = 'langchain_agentic_policy'

    @staticmethod
    def _normalize_name(name: str) -> str:
        return (name or '').strip().lower()

    @classmethod
    def _resolve_policy_class(cls, policy_name: str) -> Type[BaseAgenticPolicy]:
        normalized = cls._normalize_name(policy_name) or cls.DEFAULT_POLICY_NAME
        if normalized == 'langchain_agentic_policy':
            return LangChainAgenticPolicy
        if normalized == 'sim_coupled_policy':
            return SimCoupledAgenticPolicy

        raise ValueError(
            f'Unsupported agentic policy: {policy_name}. '
            'Supported policies: ["langchain_agentic_policy", "sim_coupled_policy"]'
        )

    @staticmethod
    def _instantiate_policy(
        policy_cls: Type[BaseAgenticPolicy],
        config: Dict[str, Any],
    ) -> BaseAgenticPolicy:
        cfg = dict(config or {})
        signature = inspect.signature(policy_cls.__init__)
        accepted = {
            name
            for name, param in signature.parameters.items()
            if name != 'self'
            and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }

        kwargs: Dict[str, Any] = {}
        for key, value in cfg.items():
            if key in accepted:
                kwargs[key] = value

        if 'runtime_overrides' in accepted and 'runtime_overrides' not in kwargs:
            kwargs['runtime_overrides'] = cfg

        ignored = [key for key in cfg if key not in accepted]
        if ignored:
            logger.debug('Ignoring unsupported config keys for %s: %s', policy_cls.__name__, ignored)

        return policy_cls(**kwargs)

    @staticmethod
    def _resolve_llm_config(
        llm_provider: str,
        llm_model: str,
        llm_registry: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        provider_name = (llm_provider or '').strip()
        model_name = (llm_model or '').strip()
        if not provider_name:
            return {}

        registry = dict(llm_registry or {})
        providers = registry.get('providers') if isinstance(registry.get('providers'), Mapping) else {}
        provider_cfg = providers.get(provider_name) or providers.get(provider_name.upper()) or {}
        if not isinstance(provider_cfg, Mapping):
            provider_cfg = {}

        if not provider_cfg:
            available = sorted(providers.keys()) if isinstance(providers, Mapping) else []
            raise ValueError(
                f'LLM provider "{provider_name}" not found in registry. '
                f'Available providers: {available}'
            )

        resolved_model = str(model_name or provider_cfg.get('model', '') or '')
        if not resolved_model:
            raise ValueError(
                f'Cannot resolve model for provider "{provider_name}": '
                'no model specified in policy config and no model defined in provider config.'
            )

        api_key = str(provider_cfg.get('api_key', '') or '')
        api_key_env = str(provider_cfg.get('api_key_env', '') or '').strip()
        if api_key_env:
            api_key = os.getenv(api_key_env, api_key)

        base_url = str(provider_cfg.get('base_url', '') or '')
        base_url_env = str(provider_cfg.get('base_url_env', '') or '').strip()
        if base_url_env:
            base_url = os.getenv(base_url_env, base_url)

        llm_config: Dict[str, Any] = {
            'provider': provider_name,
            'model': resolved_model,
            'api_key': api_key,
            'base_url': base_url,
        }
        return {key: value for key, value in llm_config.items() if value}

    @classmethod
    def build(
        cls,
        policy_name: str,
        config: Optional[Dict[str, Any]] = None,
        llm_registry: Optional[Mapping[str, Any]] = None,
    ) -> BaseAgenticPolicy:
        policy_cls = cls._resolve_policy_class(policy_name)
        cfg = dict(config or {})
        if 'llm_config' not in cfg:
            llm_config = cls._resolve_llm_config(
                str(cfg.get('llm_provider', cfg.get('provider', '')) or ''),
                str(cfg.get('llm_model', cfg.get('model', '')) or ''),
                llm_registry,
            )
            if llm_config:
                cfg['llm_config'] = llm_config
        cfg.setdefault('name', cls._normalize_name(policy_name) or cls.DEFAULT_POLICY_NAME)
        return cls._instantiate_policy(policy_cls, cfg)

    @classmethod
    def build_from_config(
        cls,
        policy_cfg: Any,
        llm_registry: Optional[Mapping[str, Any]] = None,
        *,
        screenshot_config: Optional[Mapping[str, Any]] = None,
    ) -> BaseAgenticPolicy:
        extras = dict(getattr(policy_cfg, 'extras', {}) or {})
        cfg: Dict[str, Any] = dict(extras)

        policy_name = str(
            getattr(policy_cfg, 'policy_name', '')
            or cfg.get('policy_name', '')
            or cls.DEFAULT_POLICY_NAME
        )
        llm_provider = str(getattr(policy_cfg, 'llm_provider', '') or cfg.get('llm_provider', '') or '')
        llm_model = str(getattr(policy_cfg, 'llm_model', '') or cfg.get('llm_model', '') or '')

        for attr in (
            'temperature',
            'prompt_variant',
            'verbose',
            'max_tool_iterations',
            'tool_timeout_sec',
            'use_batch_execute_plan',
            'enable_failure_recovery',
            'max_execution_attempts',
            'recovery_trace_tail',
        ):
            if hasattr(policy_cfg, attr):
                cfg[attr] = getattr(policy_cfg, attr)

        cfg['llm_provider'] = llm_provider
        cfg['llm_model'] = llm_model

        if screenshot_config is not None:
            cfg['screenshot_config'] = dict(screenshot_config)

        llm_config = cls._resolve_llm_config(llm_provider, llm_model, llm_registry)
        if llm_config:
            cfg['llm_config'] = llm_config

        return cls.build(policy_name, cfg, llm_registry)

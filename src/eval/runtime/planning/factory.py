from __future__ import annotations

import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type

try:
    from core.base import BasePlanningAgent
except ModuleNotFoundError:
    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import BasePlanningAgent

from .adapters import (
    CAPPlannerAdapter,
    CodeBotlerPlannerAdapter,
    ELLMERPlannerAdapter,
    IsrLlmPlannerAdapter,
    LLMBTPlannerAdapter,
    LLMPlannerAdapter,
    RoboAgentAdapter,
)


logger = logging.getLogger(__name__)


class AgentFactory:
    """Create planning agents from a registration-based class factory."""

    _registry: Dict[str, Dict[str, Any]] = {}
    _defaults_registered = False

    @staticmethod
    def _normalize_agent_name(agent_name: str) -> str:
        return (agent_name or '').strip().lower()

    @classmethod
    def _ensure_defaults(cls) -> None:
        if cls._defaults_registered:
            return

        cls.register(
            name='llm_planner',
            agent_cls=LLMPlannerAdapter,
            aliases=['llm-planner', 'agent://llm-planner'],
            config_key='llm_planner',
        )
        cls.register(
            name='ellmer',
            agent_cls=ELLMERPlannerAdapter,
            aliases=['elmer', 'ellmer-planner', 'agent://ellmer'],
            config_key='ellmer',
        )
        cls.register(
            name='cap',
            agent_cls=CAPPlannerAdapter,
            aliases=['code-as-policy', 'code_as_policy', 'cap-planner', 'agent://cap'],
            config_key='cap',
        )
        cls.register(
            name='roboagent',
            agent_cls=RoboAgentAdapter,
            aliases=[
                'robo-agent',
                'robo_agent',
                'roboagent-cvpr26',
                'agent://roboagent',
            ],
            config_key='roboagent',
        )
        cls.register(
            name='isr_llm',
            agent_cls=IsrLlmPlannerAdapter,
            aliases=[
                'isr-llm',
                'isrllm',
                'isr_llm_planner',
                'agent://isr-llm',
            ],
            config_key='isr_llm',
        )
        cls.register(
            name='codebotler',
            agent_cls=CodeBotlerPlannerAdapter,
            aliases=[
                'code-botler',
                'code_botler',
                'codebotler-planner',
                'agent://codebotler',
            ],
            config_key='codebotler',
        )
        cls.register(
            name='llm_bt',
            agent_cls=LLMBTPlannerAdapter,
            aliases=[
                'llm-bt',
                'llmbt',
                'llm-bt-planner',
                'agent://llm-bt',
            ],
            config_key='llm_bt',
        )
        cls._defaults_registered = True

    @classmethod
    def register(
        cls,
        *,
        name: str,
        agent_cls: Type[BasePlanningAgent],
        aliases: Optional[List[str]] = None,
        config_key: Optional[str] = None,
    ) -> None:
        canonical = cls._normalize_agent_name(name)
        if not canonical:
            raise ValueError('Agent name cannot be empty.')

        entry = {
            'name': canonical,
            'agent_cls': agent_cls,
            'config_key': cls._normalize_agent_name(config_key or canonical),
        }

        all_aliases = [canonical] + [cls._normalize_agent_name(a) for a in (aliases or [])]
        for alias in all_aliases:
            if not alias:
                continue
            cls._registry[alias] = entry

    @classmethod
    def _resolve_registration(cls, agent_name: str) -> Dict[str, Any]:
        cls._ensure_defaults()
        normalized = cls._normalize_agent_name(agent_name)
        entry = cls._registry.get(normalized)
        if entry is None:
            supported = cls.supported_agents()
            raise ValueError(
                f'Unsupported planning agent: {agent_name}. Supported agents: {supported}'
            )
        return entry

    @staticmethod
    def _instantiate_agent(
        agent_cls: Type[BasePlanningAgent],
        config: Dict[str, Any],
    ) -> BasePlanningAgent:
        cfg = dict(config or {})
        signature = inspect.signature(agent_cls.__init__)
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
            logger.debug('Ignoring unsupported config keys for %s: %s', agent_cls.__name__, ignored)

        return agent_cls(**kwargs)

    @staticmethod
    def supported_agents() -> List[str]:
        AgentFactory._ensure_defaults()
        names = {entry['name'] for entry in AgentFactory._registry.values()}
        return sorted(names)

    @staticmethod
    def create(
        agent: str,
        config: Optional[Dict[str, Any]] = None,
        llm_registry: Optional[Mapping[str, Any]] = None,
    ) -> BasePlanningAgent:
        entry = AgentFactory._resolve_registration(agent)
        cfg = dict(config or {})

        if 'llm_config' not in cfg:
            llm_provider = str(cfg.get('llm_provider', cfg.get('provider', '')) or '')
            llm_model = str(cfg.get('llm_model', cfg.get('model', '')) or '')
            llm_config = AgentFactory._resolve_llm_config(llm_provider, llm_model, llm_registry)
            if llm_config:
                cfg['llm_config'] = llm_config

        if 'embedding_llm_config' not in cfg:
            embedding_provider = str(cfg.get('embedding_llm_provider', cfg.get('embedding_provider', '')) or '')
            embedding_model = str(cfg.get('embedding_llm_model', cfg.get('embedding_model', '')) or '')
            embedding_llm_config = AgentFactory._resolve_llm_config(
                embedding_provider,
                embedding_model,
                llm_registry,
            )
            if embedding_llm_config:
                cfg['embedding_llm_config'] = embedding_llm_config

        return AgentFactory._instantiate_agent(entry['agent_cls'], cfg)

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

        # Use the default model from provider config, or the model_name specified in agent config
        resolved_model = str(model_name or provider_cfg.get('model', '') or '')

        if not resolved_model:
            raise ValueError(
                f'Cannot resolve model for provider "{provider_name}": '
                f'no model specified in agent config and no model defined in provider config.'
            )

        # Optional: validate against support_models if it exists
        support_models = provider_cfg.get('support_models')
        if support_models and isinstance(support_models, list) and resolved_model not in support_models:
            logger.warning(
                f'Model "{resolved_model}" is not in the supported models list '
                f'for provider "{provider_name}": {support_models}'
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

    @staticmethod
    def _extract_agent_config(
        agent_cfg: Any,
        entry: Dict[str, Any],
        llm_registry: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        extras = dict(getattr(agent_cfg, 'extras', {}) or {})
        cfg: Dict[str, Any] = dict(extras)

        output_type = getattr(agent_cfg, 'output_type', '')
        if output_type:
            cfg['output_type'] = output_type

        llm_provider = getattr(agent_cfg, 'llm_provider', '')
        llm_model = getattr(agent_cfg, 'llm_model', '')

        llm_config = AgentFactory._resolve_llm_config(llm_provider, llm_model, llm_registry)
        if llm_config:
            cfg['llm_config'] = llm_config

        embedding_provider = str(
            getattr(agent_cfg, 'embedding_llm_provider', '')
            or extras.get('embedding_llm_provider', '')
            or extras.get('embedding_provider', '')
            or ''
        )
        embedding_model = str(
            getattr(agent_cfg, 'embedding_llm_model', '')
            or extras.get('embedding_llm_model', '')
            or extras.get('embedding_model', '')
            or ''
        )
        embedding_llm_config = AgentFactory._resolve_llm_config(
            embedding_provider,
            embedding_model,
            llm_registry,
        )
        if embedding_llm_config:
            cfg['embedding_llm_config'] = embedding_llm_config

        cfg.setdefault('name', entry['name'])
        return cfg

    @staticmethod
    def create_from_config_map(
        agent_name: str,
        agents_cfg: Mapping[str, Any],
        llm_registry: Optional[Mapping[str, Any]] = None,
        capture_reasoning: bool = False,
    ) -> BasePlanningAgent:
        entry = AgentFactory._resolve_registration(agent_name)
        config_key = entry.get('config_key') or entry['name']

        if not isinstance(agents_cfg, Mapping):
            raise ValueError('agents config must be a mapping of agent names to config objects.')

        agent_cfg = agents_cfg.get(config_key)
        if agent_cfg is None:
            raise ValueError(
                f'Missing config for planning agent "{agent_name}" under agents["{config_key}"].'
            )

        agent_runtime_cfg = AgentFactory._extract_agent_config(agent_cfg, entry, llm_registry)
        agent_runtime_cfg['capture_reasoning'] = bool(capture_reasoning)
        return AgentFactory._instantiate_agent(entry['agent_cls'], agent_runtime_cfg)

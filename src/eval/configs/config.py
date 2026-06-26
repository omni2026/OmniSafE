from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar
from dotenv import load_dotenv



T = TypeVar('T')


def _load_dotenv_candidates(config_path: Path) -> None:
    candidates = [
        config_path.parent / '.env',
        config_path.parent.parent / '.env',
        Path.cwd() / '.env',
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        load_dotenv(dotenv_path=candidate, override=False)
        seen.add(candidate)


def _resolve_env_placeholder(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if text.startswith('${') and text.endswith('}') and len(text) > 3:
        return os.getenv(text[2:-1], '')
    return value


def _resolve_provider_config(provider_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = {key: _resolve_env_placeholder(value) for key, value in provider_cfg.items()}

    for field_name in ('api_key', 'base_url', 'model'):
        env_field = f'{field_name}_env'
        env_name = str(resolved.get(env_field, '') or '').strip()
        if not env_name:
            continue

        env_value = os.getenv(env_name, '')
        if env_value:
            resolved[field_name] = env_value

    return resolved


def _load_json_config(file_path: str) -> Dict[str, Any]:
    path = Path(file_path).resolve()
    _load_dotenv_candidates(path)

    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError('Top-level config must be a JSON object.')

    llm_payload = data.get('llm')
    if isinstance(llm_payload, Mapping):
        providers = llm_payload.get('providers')
        if isinstance(providers, Mapping):
            merged_llm = dict(llm_payload)
            merged_llm['providers'] = {
                name: _resolve_provider_config(provider_cfg)
                if isinstance(provider_cfg, Mapping)
                else provider_cfg
                for name, provider_cfg in providers.items()
            }
            data['llm'] = merged_llm

    return data


def _section_from_mapping(section_cls: Type[T], data: Optional[Mapping[str, Any]]) -> T:
    payload: Dict[str, Any] = dict(data or {})
    known_names = {f.name for f in fields(section_cls) if f.init and f.name != 'extras'}
    kwargs = {name: payload[name] for name in known_names if name in payload}
    section = section_cls(**kwargs)

    if hasattr(section, 'extras'):
        section.extras = {k: v for k, v in payload.items() if k not in known_names}
    return section


@dataclass
class SimManagerConfig:
    python_executable: Optional[str] = None
    headless: Optional[bool] = None
    livestream: Optional[bool] = None
    hide_ui: Optional[bool] = None
    pipe_id: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> 'SimManagerConfig':
        return _section_from_mapping(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            'python_executable': self.python_executable,
            'headless': self.headless,
            'livestream': self.livestream,
            'hide_ui': self.hide_ui,
            'pipe_id': self.pipe_id,
        }
        base.update(self.extras)
        return base


@dataclass
class PlanningAgentConfig:
    output_type: str = 'structured_actions'
    llm_provider: str = ''
    llm_model: str = ''
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> 'PlanningAgentConfig':
        return _section_from_mapping(cls, data)


@dataclass
class AgenticPolicyConfig:
    policy_name: str = 'langchain_agentic_policy'
    llm_provider: str = ''
    llm_model: str = ''
    temperature: float = 0.0
    prompt_variant: str = 'default'
    verbose: bool = False
    max_tool_iterations: int = 12
    tool_timeout_sec: float = 30.0
    use_batch_execute_plan: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> 'AgenticPolicyConfig':
        return _section_from_mapping(cls, data)


@dataclass
class OracleConfig:
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> 'OracleConfig':
        return _section_from_mapping(cls, data)


@dataclass
class AggregatorConfig:
    weights: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> 'AggregatorConfig':
        return _section_from_mapping(cls, data)


@dataclass
class ScreenshotConfig:
    enabled: bool = True
    resolution: list = field(default_factory=lambda: [1024, 1024])
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> 'ScreenshotConfig':
        return _section_from_mapping(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            'enabled': bool(self.enabled),
            'resolution': list(self.resolution or [1024, 1024]),
        }
        base.update(self.extras)
        return base


@dataclass
class EvalConfig:
    dataset_path: str = 'data/eval_dataset.json'
    output_path: str = 'data/eval_results.json'
    require_oracle: bool = True
    check_usd_exists: bool = True
    resume: bool = True
    orchestrator: Dict[str, Any] = field(default_factory=dict)
    robot_name: str = 'fetch'
    robot_room_name: str = ''
    agents: Dict[str, PlanningAgentConfig] = field(default_factory=dict)
    agentic_policy: AgenticPolicyConfig = field(default_factory=AgenticPolicyConfig)
    llm: Dict[str, Any] = field(default_factory=dict)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    aggregator: AggregatorConfig = field(default_factory=AggregatorConfig)
    sim_manager: SimManagerConfig = field(default_factory=SimManagerConfig)
    screenshot: ScreenshotConfig = field(default_factory=ScreenshotConfig)
    metrics: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)
    capture_reasoning: bool = False
    config_dir: str = field(default='', repr=False)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> 'EvalConfig':
        payload: Dict[str, Any] = dict(data or {})

        known_top_level = {
            'dataset_path',
            'output_path',
            'require_oracle',
            'check_usd_exists',
            'resume',
            'orchestrator',
            'robot_name',
            'robot_room_name',
            'agents',
            'agentic_policy',
            'llm',
            'oracle',
            'aggregator',
            'sim_manager',
            'screenshot',
            'metrics',
            'capture_reasoning',
        }

        agents_payload = payload.get('agents') or {}
        if not isinstance(agents_payload, Mapping):
            raise ValueError('agents must be a JSON object mapping agent names to config objects.')

        return cls(
            dataset_path=str(payload.get('dataset_path', cls.dataset_path)),
            output_path=str(payload.get('output_path', cls.output_path)),
            require_oracle=bool(payload.get('require_oracle', True)),
            check_usd_exists=bool(payload.get('check_usd_exists', True)),
            resume=bool(payload.get('resume', True)),
            orchestrator=dict(payload.get('orchestrator') or {}),
            robot_name=str(payload.get('robot_name', 'fetch') or 'fetch'),
            robot_room_name=str(payload.get('robot_room_name', '') or ''),
            agents={
                str(name): PlanningAgentConfig.from_mapping(agent_data)
                for name, agent_data in agents_payload.items()
            },
            agentic_policy=AgenticPolicyConfig.from_mapping(payload.get('agentic_policy')),
            llm=dict(payload.get('llm') or {}),
            oracle=OracleConfig.from_mapping(payload.get('oracle')),
            aggregator=AggregatorConfig.from_mapping(payload.get('aggregator')),
            sim_manager=SimManagerConfig.from_mapping(payload.get('sim_manager')),
            screenshot=ScreenshotConfig.from_mapping(payload.get('screenshot')),
            metrics=dict(payload.get('metrics') or {}),
            capture_reasoning=bool(payload.get('capture_reasoning', False)),
            extensions={k: v for k, v in payload.items() if k not in known_top_level},
        )

    @classmethod
    def from_json(cls, file_path: str) -> 'EvalConfig':
        path = Path(file_path).resolve()
        data = _load_json_config(str(path))
        config = cls.from_dict(data)
        config.config_dir = str(path.parent)
        base_dir = path.parent.parent if path.parent.name.lower() == 'configs' else path.parent
        config.resolve_paths(base_dir)
        return config

    def resolve_paths(self, eval_root: str | Path) -> None:
        root = Path(eval_root).resolve()
        self.dataset_path = str(self._resolve_path(self.dataset_path, root))
        self.output_path = str(self._resolve_path(self.output_path, root))

        for agent in self.agents.values():
            for key in ('agent_root', 'knn_data_path', 'save_path'):
                value = agent.extras.get(key)
                if value:
                    agent.extras[key] = str(self._resolve_path(str(value), root))

        for section in (self.oracle.extras, self.metrics):
            for key, value in list(section.items()):
                if key.endswith('_path') and isinstance(value, str) and value.strip():
                    section[key] = str(self._resolve_path(value, root))

    @staticmethod
    def _resolve_path(value: str, root: Path) -> Path:
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    def to_orchestrator_config(self) -> Dict[str, Any]:
        cfg = dict(self.orchestrator)
        cfg.setdefault('robot_name', str(self.robot_name or 'fetch'))
        cfg.setdefault('robot_room_name', str(self.robot_room_name or ''))
        cfg.setdefault('require_oracle', self.require_oracle)
        return cfg

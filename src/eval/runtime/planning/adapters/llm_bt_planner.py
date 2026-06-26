"""
llm_bt_planner.py

Thin adapter shim that imports LLMBTPlannerAdapter from the LLM-BT agent project.

The actual implementation lives in:
  runtime/agent_projects/LLM-BT/llm_bt/ (core modules)
  runtime/agent_projects/LLM-BT/llmbt_adapter.py (adapter)
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from core.base import BasePlanningAgent
except ModuleNotFoundError:
    eval_root = Path(__file__).resolve().parents[3]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import BasePlanningAgent

logger = logging.getLogger(__name__)

_AGENT_ROOT = Path(__file__).resolve().parents[2] / 'agent_projects' / 'LLM-BT'


class LLMBTPlannerAdapter(BasePlanningAgent):
    """Adapter shim that loads and delegates to the real LLMBTPlannerAdapter."""

    def __init__(
        self,
        name: str = 'llm_bt',
        agent_root: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        domain: str = 'household',
        use_legacy: bool = False,
        semantic_map_path: Optional[str] = None,
        max_expand_depth: int = 5,
        max_guidance_rounds: int = 2,
        output_type: str = 'bt_actions',
        runtime_overrides: Optional[Dict[str, Any]] = None,
        capture_reasoning: bool = False,
        **_: Any,
    ):
        super().__init__(name=name)
        self._agent_root = Path(agent_root) if agent_root else _AGENT_ROOT
        self._llm_config = dict(llm_config or {})
        self._domain = domain
        self._use_legacy = use_legacy
        self._semantic_map_path = semantic_map_path
        self._max_expand_depth = int(max_expand_depth)
        self._max_guidance_rounds = int(max_guidance_rounds)
        self._output_type = output_type
        self._runtime_overrides = runtime_overrides or {}
        self._capture_reasoning = bool(capture_reasoning)
        self._impl = None

    def _load_impl(self):
        if self._impl is not None:
            return self._impl

        agent_root_str = str(self._agent_root)
        if agent_root_str not in sys.path:
            sys.path.insert(0, agent_root_str)

        adapter_path = self._agent_root / 'llmbt_adapter.py'
        if not adapter_path.exists():
            raise FileNotFoundError(f'LLMBTPlannerAdapter not found at {adapter_path}')

        spec = importlib.util.spec_from_file_location(
            'llmbt_adapter', str(adapter_path)
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Failed to load module spec from: {adapter_path}')

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        LLMBTImpl = getattr(module, 'LLMBTPlannerAdapter')
        self._impl = LLMBTImpl(
            name=self.name,
            agent_root=self._agent_root,
            llm_config=self._llm_config,
            domain=self._domain,
            use_legacy=self._use_legacy,
            semantic_map_path=self._semantic_map_path,
            max_expand_depth=self._max_expand_depth,
            max_guidance_rounds=self._max_guidance_rounds,
            output_type=self._output_type,
            runtime_overrides=self._runtime_overrides,
            capture_reasoning=self._capture_reasoning,
        )
        return self._impl

    async def start(self) -> None:
        impl = self._load_impl()
        await impl.start()
        self.status = impl.status

    async def stop(self) -> None:
        if self._impl is not None:
            await self._impl.stop()
        self.status = self._impl.status if self._impl else 'TERMINATED'

    async def update_context(self, context: Dict[str, Any]) -> None:
        if self._impl is None:
            raise RuntimeError('LLMBTPlannerAdapter is not started.')
        await self._impl.update_context(context)

    async def plan(self, instructions, context):
        if self._impl is None:
            raise RuntimeError('LLMBTPlannerAdapter is not started.')
        return await self._impl.plan(instructions, context)

"""
llmbt_adapter.py

LLM-BT Planning-Only Adapter for the OMNISAFE Eval pipeline.

Implements BasePlanningAgent, orchestrating the two-stage LLM-BT architecture:
  Stage 1 (Intention Reasoning): NL instruction → LLM → goal conditions
  Stage 2 (BT Adaptive Update): goal conditions → symbolic tick + LLM-assisted
                                  expansion → fully expanded BT → action sequence

Output format:
  PlanningResult.actions = [
      {'type': 'bt_action', 'name': 'Navigate', 'args': ['kitchen'], 'step_index': 0},
      ...
  ]
  PlanningResult.raw_output = fully expanded BT as XML
  PlanningResult.metadata = {
      'agent_name': 'llm_bt',
      'goal_conditions': [...],
      'bt_xml': '<root>...</root>',
      'expansion_depth': ...,
  }
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from core.base import BasePlanningAgent, PlanningResult, ProcessStatus
except ModuleNotFoundError:
    eval_root = Path(__file__).resolve().parents[3]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import BasePlanningAgent, PlanningResult, ProcessStatus

try:
    from runtime.planning.reasoning_utils import (
        should_capture_reasoning,
    )
except ModuleNotFoundError:
    from reasoning_utils import should_capture_reasoning

from llm_bt.bt_core import BTNode, NodeType, from_xml
from llm_bt.bt_expansion import BTExpansionEngine
from llm_bt.household_domain import GoalCondition, HouseholdDomain
from llm_bt.intention_reasoning import IntentionReasoner
from llm_bt.prompts import PromptConfig
from llm_bt.semantic_map import SemanticMap, build_semantic_map_from_context, parse_semantic_map_file

logger = logging.getLogger(__name__)


class LLMBTPlannerAdapter(BasePlanningAgent):
    """Planning-only adapter that reproduces LLM-BT's two-stage architecture.

    Stage 1: IntentionReasoner converts NL instructions into goal conditions.
    Stage 2: BTExpansionEngine expands goal conditions into a full behavior tree
             and extracts an ordered action sequence.
    """

    def __init__(
        self,
        name: str = 'llm_bt',
        agent_root: Optional[Path | str] = None,
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
        default_root = Path(__file__).resolve().parent
        self._agent_root = Path(agent_root) if agent_root else default_root
        self._llm_config = dict(llm_config or {})
        self._output_type = output_type
        self._domain = domain.strip().lower()
        self._use_legacy = use_legacy
        self._semantic_map_path = semantic_map_path
        self._max_expand_depth = int(max_expand_depth)
        self._max_guidance_rounds = int(max_guidance_rounds)
        self._runtime_overrides = runtime_overrides or {}
        self._capture_reasoning = bool(capture_reasoning)

        self._reasoner: Optional[IntentionReasoner] = None
        self._expansion_engine: Optional[BTExpansionEngine] = None
        self._semantic_map: Optional[SemanticMap] = None
        self._domain_def: Optional[HouseholdDomain] = None
        self._context: Dict[str, Any] = {}
        self._initial_state_loaded: bool = False

    async def start(self) -> None:
        prompt_config = PromptConfig(
            domain=self._domain,
            use_legacy_format=self._use_legacy,
            max_guidance_rounds=self._max_guidance_rounds,
            include_reasoning_line=self._reasoning_active(),
        )

        self._domain_def = HouseholdDomain(use_legacy=self._use_legacy)

        if self._semantic_map_path:
            self._semantic_map = parse_semantic_map_file(self._semantic_map_path)
        else:
            self._semantic_map = SemanticMap()

        self._reasoner = IntentionReasoner(
            llm_config=self._llm_config,
            semantic_map=self._semantic_map,
            prompt_config=prompt_config,
            domain=self._domain,
            max_guidance_rounds=self._max_guidance_rounds,
        )

        self._expansion_engine = BTExpansionEngine(
            domain=self._domain_def,
            semantic_map=self._semantic_map,
            llm_config=self._llm_config,
            max_expand_depth=self._max_expand_depth,
        )

        self._initial_state_loaded = False
        self.status = ProcessStatus.RUNNING

    def _ensure_initial_state(self) -> None:
        if not self._initial_state_loaded and self._expansion_engine is not None and self._semantic_map is not None:
            self._expansion_engine.set_initial_state(
                self._semantic_map.get_initial_state()
            )
            self._initial_state_loaded = True

    def _reasoning_active(self) -> bool:
        return should_capture_reasoning(self._llm_config, self._capture_reasoning)

    async def stop(self) -> None:
        self._reasoner = None
        self._expansion_engine = None
        self._semantic_map = None
        self._domain_def = None
        self._context = {}
        self.status = ProcessStatus.TERMINATED

    async def update_context(self, context: Dict[str, Any]) -> None:
        self._context = dict(context or {})
        if self._semantic_map and not self._semantic_map.locations:
            self._semantic_map = build_semantic_map_from_context(context)
            if self._expansion_engine:
                self._expansion_engine.semantic_map = self._semantic_map
                self._expansion_engine.set_initial_state(
                    self._semantic_map.get_initial_state()
                )
        elif self._semantic_map and self._context.get("world_state"):
            if self._expansion_engine:
                self._expansion_engine.set_initial_state(self._context["world_state"])

    async def plan(self, instructions: List[str], context: Dict[str, Any]) -> List[PlanningResult]:
        if self._reasoner is None or self._expansion_engine is None:
            raise RuntimeError('LLMBTPlannerAdapter is not started. Call start() before planning.')

        if context:
            await self.update_context(context)

        scenario_id = str(self._context.get('scenario_id', 'unknown'))
        results: List[PlanningResult] = []

        for instruction in instructions:
            goal_conditions = await self._reasoner.reason(instruction, self._context)
            reasoning = None
            if self._capture_reasoning:
                reasoning = (
                    str(getattr(self._reasoner, 'last_reasoning', '') or '').strip()
                    or None
                )
            llm_traces = [
                dict(trace)
                for trace in list(getattr(self._reasoner, 'llm_traces', []) or [])
                if isinstance(trace, dict)
            ]

            if not goal_conditions:
                logger.warning('No goal conditions produced for: %s', instruction)
                results.append(PlanningResult(
                    scenario_id=scenario_id,
                    instruction=instruction,
                    actions=[],
                    raw_output='',
                    reasoning=reasoning,
                    metadata={
                        'agent_name': self.name,
                        'goal_conditions': [],
                        'expansion_depth': 0,
                        'output_type': self._output_type,
                        'llm_trace': llm_traces,
                    },
                ))
                continue

            expanded_bt = await self._expansion_engine.expand_from_goals(goal_conditions)
            actions = expanded_bt.extract_action_sequence()
            bt_xml = expanded_bt.to_xml()

            results.append(PlanningResult(
                scenario_id=scenario_id,
                instruction=instruction,
                actions=actions,
                raw_output=bt_xml,
                reasoning=reasoning,
                metadata={
                    'agent_name': self.name,
                    'entrypoint': str(self._agent_root),
                    'provider': self._llm_config.get('provider'),
                    'model': self._llm_config.get('model'),
                    'output_type': self._output_type,
                    'domain': self._domain,
                    'goal_conditions': [gc.raw_text for gc in goal_conditions],
                    'bt_expansion_depth': self._max_expand_depth,
                    'num_actions': len(actions),
                    'llm_trace': llm_traces,
                },
            ))

        return results


# ==================== Debug entrypoint ====================

if __name__ == '__main__':
    import json

    try:
        from runtime.planning.factory import AgentFactory
        from configs.config import EvalConfig
    except ModuleNotFoundError:
        _eval_root = Path(__file__).resolve().parents[3]
        if str(_eval_root) not in sys.path:
            sys.path.insert(0, str(_eval_root))
        from runtime.planning.factory import AgentFactory
        from configs.config import EvalConfig

    async def _debug_main():
        cfg_path = Path(__file__).resolve().parents[3] / 'configs' / 'default_config.json'
        eval_cfg = EvalConfig.from_json(str(cfg_path))

        llm_config = {}
        providers = eval_cfg.llm.get('providers', {})
        for prov_name, prov_cfg in providers.items():
            llm_config = {
                'provider': prov_name,
                'model': prov_cfg.get('model', ''),
                'api_key': prov_cfg.get('api_key', ''),
                'base_url': prov_cfg.get('base_url', ''),
            }
            if llm_config['api_key']:
                break

        adapter = LLMBTPlannerAdapter(
            llm_config=llm_config,
            domain='household',
            max_expand_depth=5,
        )

        print(f'[*] Created adapter: {adapter.name}')

        await adapter.start()
        print(f'[*] Adapter started, status={adapter.status.name}')

        context = {
            'scenario_id': 'llm_bt_debug_001',
            'metadata': {
                'vis_objs': ['apple', 'knife', 'milk', 'fridge', 'kitchen_counter', 'dining_table'],
                'scene_description': (
                    'The robot is at the kitchen counter. '
                    'An apple is at the kitchen counter. '
                    'A knife is at the kitchen counter. '
                    'Milk is in the fridge. '
                ),
            },
        }
        instructions = ['Pick up the apple from the kitchen counter and put it on the dining table']

        results = await adapter.plan(instructions, context)

        for idx, result in enumerate(results, 1):
            print(f'\n--------- Result {idx} ---------')
            print(f'Scenario: {result.scenario_id}')
            print(f'Instruction: {result.instruction}')
            print(f'Goal conditions: {result.metadata.get("goal_conditions", [])}')
            print(f'Actions ({len(result.actions)}):')
            for action in result.actions:
                print(f"  {action['step_index']}: {action['name']}({action['args']})")
            if result.raw_output:
                print(f'\n[BT XML]\n{result.raw_output[:1000]}')

        await adapter.stop()
        print(f'\n[*] Adapter stopped, status={adapter.status.name}')

    asyncio.run(_debug_main())

# ==============================================================================
# RoboAgent Planner Adapter - 总结文档
# ==============================================================================
#
# 本文件封装了基于 API 调用 LLM 的 RoboAgent（CVPR'26）规划器，提供与评测
# 管线统一的 BasePlanningAgent 接口。
#
# 底层 Agent 是 *stateful interactive* 的：每一轮返回一小批可执行的动作，
# 等待外部执行后通过 process_feedback + process_observation 注入结果，
# 再调用 plan_step() 拿到下一批。这与 LLM-Planner 的「外部循环驱动模式」
# 同构，区别在于 RoboAgent 把所有上下文（核心 history、ability 缓冲、
# 已探索方向、当前持物、最近一次 goto 目标、scene_description、子轨迹
# 等）都保存在 Agent 实例自身里，不需要序列化 loop_state。
#
# ==============================================================================
# 一、Agent 运行流程
# ==============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │                          初始阶段                                     │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 加载 api_agent 包 → 构造 APIClient(model, base_url, api_key)        │
# │ → 构造 Agent(api_client, env_name=...)                              │
# └─────────────────────────────────────────────────────────────────────┘
#                              │
#                              ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │              第一次 plan() 调用 (新场景)                              │
# ├─────────────────────────────────────────────────────────────────────┤
# │ adapter 检测到 scenario_id 变化 →                                    │
# │   agent.reset(save_path, obj_list)                                  │
# │   agent.process_task(None, instruction)                             │
# │   agent.process_observation(image_path)   # 可选                    │
# │   agent.plan_step() → ["go to Fridge 1", "open Fridge 1", ...]      │
# │ 返回 PlanningResult，actions 列表里是这一批可执行动作                  │
# └─────────────────────────────────────────────────────────────────────┘
#                              │
#                              ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │            后续 plan() 调用 (外部执行循环)                            │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 外部执行器执行上一批动作 →                                            │
# │   传入 metadata.execution_results = [                               │
# │     {"action": "go to Fridge 1", "success": True},                  │
# │     {"action": "open Fridge 1", "success": False, ...},             │
# │   ]                                                                 │
# │   (可选) metadata.image_path = "obs_step_3.png"                     │
# │                                                                     │
# │ adapter 逐条 agent.process_feedback(success, action) →              │
# │   agent.process_observation(image_path) (若提供) →                   │
# │   agent.plan_step() → 下一批动作 或 ["fail"]                         │
# └─────────────────────────────────────────────────────────────────────┘
#
# 核心要点：
#   • LLM 调用时机：每次 plan_step() 内部会按 ability 链路调用若干次 LLM
#     (cognitive_task_planner / exploration_guidance / object_grounding /
#      scene_description / manipulation_planner / experience_summarization)，
#     直到能产出一批可执行动作或判定失败/完成。
#   • 执行责任：外部执行器负责，adapter 只负责规划 + 维护 agent 状态。
#   • 状态保存：完全由 Agent 实例持有，adapter 透过 scenario_id 触发 reset。
#   • 重置条件：scenario_id 变化、或 metadata.reset_agent = True。
#   • 终止条件：plan_step() 返回 ["fail"] 或 actions 为空。
#
# ==============================================================================
# 二、context.metadata 字段约定
# ==============================================================================
#
#   字段                          | 何时使用     | 说明
#   ------------------------------|--------------|------------------------------
#   obj_list / vis_objs           | 新场景首次   | 可观察对象列表 (exploration_
#                                 |              |   guidance 用来验证方向)
#   image_path                    | 任意调用     | 当前 egocentric RGB 图像路径
#   image_rgb                     | 任意调用     | numpy 数组形式的当前图像
#                                 |              |   (需要 opencv-python)
#   execution_result              | 后续调用     | 单条上次执行反馈
#                                 |              |   {action, success}
#   execution_results             | 后续调用     | 多条上次执行反馈 (一批)
#   reset_agent                   | 任意调用     | 显式强制重置 agent 状态
#   save_path                     | 任意调用     | 覆盖 trace/log 保存目录
#
# ==============================================================================
# 三、PlanningResult 输出格式
# ==============================================================================
#
# actions = [
#   {'type': 'roboagent_action',
#    'action': 'go to Fridge 1',
#    'raw':    'go to Fridge 1',
#    'step_index': 0},
#   ...
# ]
# raw_output = '\n'.join(actions)
# metadata.roboagent = {
#   'finished': bool,             # 是否触发停止/失败
#   'status': 'awaiting_execution' | 'failed' | 'finished',
#   'env_name': 'alfworld' | 'eb-alfred' | 'generic',
#   'core_history': str,          # 截断后的核心 history
#   'ability_trace_size': int,    # 本轮 LLM 调用次数
# }
#
# ==============================================================================
# 四、使用方法
# ==============================================================================
#
# 1. 创建 Adapter 实例
# -------------------
# adapter = RoboAgentAdapter(
#     name='roboagent',
#     env_name='alfworld',
#     llm_config={
#         'provider': 'openai',
#         'model': 'gpt-5.5',
#         'api_key': '<set OPENAI_API_KEY in .env>',
#         'base_url': '<optional OpenAI-compatible base URL>',
#     },
#     save_path='runtime/agent_projects/RoboAgent_CVPR26/api_agent_output',
# )
#
# 2. 启动
# -------------------
# await adapter.start()
#
# 3. 首次规划（新场景）
# -------------------
# context = {
#     'scenario_id': 'scenario_001',
#     'metadata': {
#         'obj_list': ['Apple 1', 'Fridge 1', 'Microwave 1', ...],
#         'image_path': 'obs_step_0.png',     # optional
#     }
# }
# results = await adapter.plan(
#     instructions=['heat Apple 1 with Microwave 1'],
#     context=context,
# )
# for action in results[0].actions:
#     print(action['action'])           # e.g. "go to Fridge 1"
#
# 4. 后续规划（反馈 + 新观测）
# -------------------
# context = {
#     'scenario_id': 'scenario_001',
#     'metadata': {
#         'execution_results': [
#             {'action': 'go to Fridge 1', 'success': True},
#             {'action': 'open Fridge 1',  'success': True},
#         ],
#         'image_path': 'obs_step_2.png',
#     }
# }
# results = await adapter.plan(instructions=[''], context=context)
#
# 5. 停止
# -------------------
# await adapter.stop()  # 会调用 agent.save_trace()
#
# ==============================================================================

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import os
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
        extract_think_reasoning,
        should_capture_reasoning,
    )
except ModuleNotFoundError:
    from reasoning_utils import (
        extract_think_reasoning,
        should_capture_reasoning,
    )


logger = logging.getLogger(__name__)


_VALID_ENVS = {'alfworld', 'alfworld_text', 'eb-alfred', 'generic'}

# Sentinel raw outputs returned by Agent.plan_step / get_qwen_action_raw
_RAW_FAIL = ['fail']
_RAW_PASS = ['pass']


class RoboAgentAdapter(BasePlanningAgent):
    """Adapter for runtime/agent_projects/RoboAgent_CVPR26/api_agent."""

    def __init__(
        self,
        name: str = 'roboagent',
        agent_root: Optional[Path | str] = None,
        env_name: str = 'alfworld',
        llm_config: Optional[Dict[str, Any]] = None,
        output_type: str = 'interactive_actions',
        save_path: Optional[Path | str] = None,
        max_step_rounds: int = 50,
        max_interaction_turns: int = 50,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        capture_reasoning: bool = False,
        **_: Any,
    ):
        super().__init__(name=name)
        project_root = Path(__file__).resolve().parents[2] / 'agent_projects' / 'RoboAgent_CVPR26'
        self._agent_root = Path(agent_root) if agent_root else project_root
        self._api_agent_dir = self._agent_root / 'api_agent'
        self._llm_config = dict(llm_config or {})
        self._output_type = output_type
        self._max_step_rounds = int(max_step_rounds)
        self._max_interaction_turns = max(1, int(max_interaction_turns))
        self._runtime_overrides = runtime_overrides or {}
        self._capture_reasoning = bool(capture_reasoning)

        env_normalized = (env_name or 'alfworld').strip().lower()
        if env_normalized not in _VALID_ENVS:
            raise ValueError(
                f'Unsupported RoboAgent env_name: {env_name}. '
                f'Supported envs: {sorted(_VALID_ENVS)}'
            )
        self._env_name = env_normalized

        default_save = self._agent_root / 'api_agent_output'
        self._base_save_path = Path(save_path) if save_path else default_save

        self._module = None
        self._api_client = None
        self._agent = None
        self._context: Dict[str, Any] = {}
        self._current_scenario_id: str = ''
        self._task_started: bool = False
        self._step_index: int = 0  # cumulative action index for this scenario
        self._turn_index: int = 0

    @property
    def supports_interactive_planning(self) -> bool:
        return True

    @property
    def max_interaction_turns(self) -> Optional[int]:
        return self._max_interaction_turns

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._api_agent_dir.exists():
            raise FileNotFoundError(
                f'RoboAgent api_agent directory not found: {self._api_agent_dir}'
            )

        # api_agent uses relative imports (`from .api_client import APIClient`),
        # so it must be loaded as a *package*. Add the RoboAgent project root
        # to sys.path so `import api_agent` resolves to the package directory.
        project_root_str = str(self._agent_root)
        inserted = False
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            inserted = True

        try:
            module_name = 'api_agent'
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)
            self._module = module
        finally:
            if inserted and project_root_str in sys.path:
                sys.path.remove(project_root_str)

        api_client_cls = getattr(self._module, 'APIClient', None)
        agent_cls = getattr(self._module, 'Agent', None)
        if api_client_cls is None or agent_cls is None:
            raise RuntimeError(
                'RoboAgent api_agent module missing APIClient or Agent class.'
            )

        # Build APIClient from llm_config (api_key resolved via factory or env).
        api_key = str(self._llm_config.get('api_key', '') or '')
        if not api_key:
            env_name = str(self._llm_config.get('api_key_env', '') or '').strip()
            if env_name:
                api_key = os.getenv(env_name, '')
        base_url = str(self._llm_config.get('base_url', '') or '') or None
        model_name = str(self._llm_config.get('model', '') or '').strip()
        if not model_name:
            raise RuntimeError(
                'RoboAgentAdapter requires a model name; please configure '
                'llm_provider/llm_model in the agent config.'
            )

        client_kwargs: Dict[str, Any] = {'model_name': model_name}
        if base_url:
            client_kwargs['base_url'] = base_url
        if api_key:
            client_kwargs['api_key'] = api_key
        self._api_client = api_client_cls(**client_kwargs)

        self._agent = agent_cls(self._api_client, env_name=self._env_name)
        # Pre-assign save_path so the agent can write logs even before first reset.
        self._agent.save_path = str(self._base_save_path)
        os.makedirs(self._base_save_path, exist_ok=True)

        self.status = ProcessStatus.RUNNING

    async def stop(self) -> None:
        # Try to persist the trace for the last scenario before tearing down.
        if self._agent is not None and self._task_started:
            try:
                await asyncio.to_thread(self._agent.save_trace)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning('Failed to save RoboAgent trace on stop: %s', exc)

        self._agent = None
        self._api_client = None
        self._module = None
        self._context = {}
        self._current_scenario_id = ''
        self._task_started = False
        self._step_index = 0
        self._turn_index = 0
        self.status = ProcessStatus.TERMINATED

    async def update_context(self, context: Dict[str, Any]) -> None:
        self._context = dict(context or {})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_runtime_option(self, metadata: Dict[str, Any], key: str, default: Any) -> Any:
        if key in metadata:
            return metadata[key]
        return self._runtime_overrides.get(key, default)

    def _reasoning_active(self) -> bool:
        return should_capture_reasoning(self._llm_config, self._capture_reasoning)

    def _extract_reasoning_from_trace(self) -> Optional[str]:
        if not self._reasoning_active():
            return None
        trace = list(getattr(self._agent, 'trace', []) or [])
        for record in reversed(trace):
            if not isinstance(record, dict):
                continue
            if str(record.get('ability', '') or '') != 'cognitive_task_planner':
                continue
            reasoning = extract_think_reasoning(str(record.get('response', '') or ''))
            if reasoning:
                return reasoning
        return None

    def _resolve_save_path(self, scenario_id: str, metadata: Dict[str, Any]) -> Path:
        explicit = metadata.get('save_path') or self._runtime_overrides.get('save_path')
        if explicit:
            return Path(explicit)
        if scenario_id:
            scenario_path = self._base_save_path / str(scenario_id)
            instruction_index = metadata.get('instruction_index')
            if instruction_index not in (None, 0, '0'):
                return scenario_path / f'instruction_{int(instruction_index):03d}'
            return scenario_path
        return self._base_save_path

    def _extract_obj_list(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        objs = (
            metadata.get('obj_list')
            or metadata.get('object_list')
            or metadata.get('vis_objs')
            or metadata.get('visible_objects')
            or context.get('vis_objs')
            or []
        )
        if isinstance(objs, (list, tuple, set)):
            return [str(o) for o in objs]
        if isinstance(objs, str):
            return [objs]
        return []

    def _apply_observation(self, metadata: Dict[str, Any]) -> None:
        """Forward any image observation to the underlying agent."""
        if self._agent is None:
            return
        image_path = metadata.get('image_path')
        image_rgb = metadata.get('image_rgb')

        if image_path:
            self._agent.process_observation(str(image_path))
            return
        if image_rgb is not None:
            step_id = metadata.get('env_step_id') or self._step_index
            self._agent.process_observation(image_rgb, env_step_id=step_id)

    def _apply_feedback(self, metadata: Dict[str, Any]) -> None:
        """Forward execution feedback (single or batch) to the agent."""
        if self._agent is None:
            return

        feedbacks: List[Dict[str, Any]] = []
        single = metadata.get('execution_result')
        batch = metadata.get('execution_results')
        if isinstance(single, dict):
            feedbacks.append(single)
        if isinstance(batch, (list, tuple)):
            feedbacks.extend(item for item in batch if isinstance(item, dict))

        for fb in feedbacks:
            action = fb.get('action') or fb.get('plan')
            if not action:
                continue
            success = bool(fb.get('success', True))
            try:
                self._agent.process_feedback(success, str(action))
            except AssertionError as exc:
                # Agent rejects certain placeholder actions (examine / pass / do
                # nothing) for non-generic envs.  Surface as a warning instead
                # of crashing the whole pipeline.
                logger.warning(
                    'RoboAgent rejected feedback (%s, %s): %s',
                    success, action, exc,
                )

    def _build_action_dicts(self, raw_actions: List[str]) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for raw in raw_actions:
            text = str(raw).strip()
            if not text:
                continue
            actions.append({
                'type': 'roboagent_action',
                'action': text,
                'raw': text,
                'step_index': self._step_index,
            })
            self._step_index += 1
        return actions

    def _trim_history(self, text: Optional[str], max_chars: int = 4000) -> str:
        text = str(text or '')
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def _reset_for_new_scenario(
        self,
        scenario_id: str,
        instruction: str,
        metadata: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        assert self._agent is not None

        if self._task_started:
            try:
                self._agent.save_trace()
            except Exception as exc:
                logger.warning('Failed to save RoboAgent trace before reset: %s', exc)

        save_path = self._resolve_save_path(scenario_id, metadata)
        save_path.mkdir(parents=True, exist_ok=True)

        obj_list = self._extract_obj_list(metadata, context)

        self._agent.reset(str(save_path), obj_list)
        self._agent.process_task(None, instruction)
        self._task_started = True
        self._current_scenario_id = scenario_id
        self._step_index = 0
        self._turn_index = 0
        # Apply first observation if present
        self._apply_observation(metadata)

    def _run_step(self) -> List[str]:
        """Drive Agent.get_qwen_action_raw() in a small bounded loop.

        Mirrors Agent.plan_step()'s 50-round loop but uses our configurable
        bound and returns the first non-sentinel actions batch (or sentinel).
        """
        assert self._agent is not None
        for _ in range(self._max_step_rounds):
            raw = self._agent.get_qwen_action_raw()
            if raw == _RAW_FAIL:
                return _RAW_FAIL
            if raw == _RAW_PASS:
                continue
            return list(raw)
        return _RAW_FAIL

    # ------------------------------------------------------------------
    # Planning entrypoint
    # ------------------------------------------------------------------

    async def plan(self, instructions: List[str], context: Dict[str, Any]) -> List[PlanningResult]:
        if self._agent is None or self.status != ProcessStatus.RUNNING:
            raise RuntimeError('RoboAgentAdapter is not started. Call start() before planning.')

        if context:
            await self.update_context(context)

        scenario_id = str(self._context.get('scenario_id', '') or 'unknown')
        metadata: Dict[str, Any] = dict(self._context.get('metadata') or {})
        reset_requested = bool(metadata.get('reset_agent', False))
        scenario_changed = (scenario_id and scenario_id != self._current_scenario_id)

        results: List[PlanningResult] = []

        for instruction in instructions:
            # ---------- 1. State setup: reset on new scenario / explicit request
            need_reset = (
                reset_requested
                or scenario_changed
                or not self._task_started
            )
            if need_reset:
                # An instruction is required for reset; if the caller passed an
                # empty string on a subsequent call, fall back to the stored one.
                effective_instr = (
                    instruction
                    or (self._agent.task_instruction if hasattr(self._agent, 'task_instruction') else '')
                    or scenario_id
                )
                await asyncio.to_thread(
                    self._reset_for_new_scenario,
                    scenario_id,
                    str(effective_instr),
                    metadata,
                    self._context,
                )
                # Subsequent instructions in the same call should NOT trigger
                # another reset within the same plan() invocation.
                scenario_changed = False
                reset_requested = False
            else:
                # ---------- 2. Feed feedback & observation for follow-up calls
                await asyncio.to_thread(self._apply_feedback, metadata)
                await asyncio.to_thread(self._apply_observation, metadata)

            # ---------- 3. Drive the agent to produce the next action batch
            raw_actions = await asyncio.to_thread(self._run_step)

            finished = False
            if raw_actions == _RAW_FAIL:
                status = 'failed'
                finished = True
                action_dicts: List[Dict[str, Any]] = []
                raw_output = 'fail'
            elif not raw_actions:
                status = 'finished'
                finished = True
                action_dicts = []
                raw_output = ''
            else:
                status = 'awaiting_execution'
                action_dicts = self._build_action_dicts(raw_actions)
                raw_output = '\n'.join(a['action'] for a in action_dicts)

            ability_trace_size = len(getattr(self._agent, 'trace', []) or [])
            core_history = self._trim_history(getattr(self._agent, 'core_history', ''))

            results.append(
                PlanningResult(
                    scenario_id=scenario_id,
                    instruction=instruction,
                    actions=action_dicts,
                    raw_output=raw_output,
                    reasoning=self._extract_reasoning_from_trace(),
                    metadata={
                        'agent_name': self.name,
                        'entrypoint': str(self._api_agent_dir),
                        'provider': self._llm_config.get('provider'),
                        'model': self._llm_config.get('model'),
                        'env_name': self._env_name,
                        'output_type': self._output_type,
                        'interactive_planning': {
                            'enabled': True,
                            'status': status,
                            'finished': finished,
                            'turn_index': self._turn_index,
                        },
                        'roboagent': {
                            'status': status,
                            'finished': finished,
                            'env_name': self._env_name,
                            'core_history': core_history,
                            'ability_trace_size': ability_trace_size,
                            'save_path': str(getattr(self._agent, 'save_path', '')),
                        },
                    },
                )
            )
            self._turn_index += 1

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

        adapter = AgentFactory.create_from_config_map('roboagent', eval_cfg.agents, eval_cfg.llm)
        print(f'[*] Created adapter: {adapter.name}')

        await adapter.start()
        print(f'[*] Adapter started, status={adapter.status.name}')

        DEFAULT_OBJECTS = [
            'Apple 1', 'CounterTop 1', 'Fridge 1', 'Microwave 1',
            'Cabinet 1', 'Cabinet 2', 'Sink 1', 'Stove 1',
        ]

        # ---------- Turn 1: new scenario ----------
        ctx1 = {
            'scenario_id': 'roboagent_debug_001',
            'metadata': {
                'obj_list': DEFAULT_OBJECTS,
            },
        }
        instr = 'heat Apple 1 with Microwave 1'
        print(f'\n[Turn 1] instruction = "{instr}"')
        out1 = await adapter.plan([instr], ctx1)
        for r in out1:
            print(f'  status     = {r.metadata["roboagent"]["status"]}')
            print(f'  actions    = {[a["action"] for a in r.actions]}')
            print(f'  trace_size = {r.metadata["roboagent"]["ability_trace_size"]}')

        # ---------- Turn 2: feed back successful execution ----------
        actions_turn1 = [a['action'] for a in (out1[0].actions if out1 else [])]
        if actions_turn1:
            ctx2 = {
                'scenario_id': 'roboagent_debug_001',
                'metadata': {
                    'execution_results': [
                        {'action': a, 'success': True} for a in actions_turn1
                    ],
                },
            }
            print(f'\n[Turn 2] feeding back {len(actions_turn1)} successful actions')
            out2 = await adapter.plan([''], ctx2)
            for r in out2:
                print(f'  status     = {r.metadata["roboagent"]["status"]}')
                print(f'  actions    = {[a["action"] for a in r.actions]}')
                print(f'  trace_size = {r.metadata["roboagent"]["ability_trace_size"]}')

        await adapter.stop()
        print(f'\n[*] Adapter stopped, status={adapter.status.name}')

    asyncio.run(_debug_main())

# ==============================================================================
# LLM Planner Adapter - 总结文档
# ==============================================================================
#
# 本文件封装了基于 LLM 的高层规划器 (LLM-Planner)，提供了以下功能：
#
# ==============================================================================
# 一、Agent 运行流程
# ==============================================================================
#
# 整个执行流程分为初始阶段和执行循环：
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │                           初始阶段                                    │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 输入: task_instr + 视觉信息(vis_objs)                               │
# │       ↓                                                             │
# │ 调用 LLM → 生成初始计划列表 (pending_plans)                         │
# │       ↓                                                             │
# │ 返回: {next_plan, remaining_plans, loop_state, ...}                 │
# └─────────────────────────────────────────────────────────────────────┘
#                               │
#                               ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │                           执行循环                                    │
# ├─────────────────────────────────────────────────────────────────────┤
# │                                                                     │
# │  ┌───────────────────────────────────────────────────────────────┐  │
# │  │ 外部执行器执行 next_plan                                       │  │
# │  └───────────────────────────────────────────────────────────────┘  │
# │                          ↓                                           │
# │  ┌───────────────────────────────────────────────────────────────┐  │
# │  │ 构造 execution_result: {success, message, plan, ...}          │  │
# │  └───────────────────────────────────────────────────────────────┘  │
# │                          ↓                                           │
# │  ┌───────────────────────────────────────────────────────────────┐  │
# │  │ 反馈给 Agent (传入 loop_state + execution_result)             │  │
# │  └───────────────────────────────────────────────────────────────┘  │
# │                          ↓                                           │
# │  ┌───────────────────────────────────────────────────────────────┐  │
# │  │ Agent 处理:                                                     │  │
# │  │   ├─ 成功 → plan 移至 completed_plans，继续下一个              │  │
# │  │   └─ 失败 → 触发重规划 (调用 LLM 生成新计划)                   │  │
# │  └───────────────────────────────────────────────────────────────┘  │
# │                          ↓                                           │
# │  ┌───────────────────────────────────────────────────────────────┐  │
# │  │ 返回新的 {next_plan, remaining_plans, loop_state, ...}        │  │
# │  └───────────────────────────────────────────────────────────────┘  │
# │                                                                     │
# │                          └───────────── pending_plans 为空 → 结束    │
# └─────────────────────────────────────────────────────────────────────┘
#
# 核心要点：
#   • LLM 调用时机：初始时调用一次，失败重规划时再调用
#   • 执行责任：外部执行器负责，Agent 只负责规划
#   • 状态保存：loop_state 跨多次调用保存上下文
#   • 重规划触发：执行失败 + 达到重试次数上限 + 未达重规划上限
#   • 终止条件：pending_plans 为空 或 达到重规划上限
#
# ==============================================================================
# 二、loop_state 说明
# ==============================================================================
#
# loop_state 是一个可序列化的外部状态对象，用于在外部循环驱动模式下
# 保存和恢复 planner 的执行状态。
#
# 字段说明：
#   字段                          | 类型    | 说明
#   ------------------------------|---------|----------------------------------
#   task_instr                    | list    | 高层任务描述指令
#   step_instr                    | list    | 逐步的详细指令
#   completed_plans               | list    | 已成功执行的计划
#   failed_plans                  | list    | 失败的计划及错误信息
#   pending_plans                 | list    | 待执行的计划队列
#   initial_high_level_plans      | list    | LLM 生成的初始计划（用于对比）
#   retry_count                   | int     | 当前重试次数
#   replanning_count              | int     | 总重规划次数
#   seen_objs                     | list    | 已见过的物体列表
#   last_prompt                   | str     | 最后一次发送给 LLM 的提示词
#   last_llm_output              | str     | LLM 最后一次的原始输出
#   last_event                    | dict    | 最后一次事件记录
#
# pending_plans 的作用：
#   • 批量计划存储：LLM 一次生成多个计划，减少 API 调用
#   • 队列式执行：从队列头部取出计划执行 (FIFO)
#   • 重规划时的状态管理：失败时清空并替换为新计划
#   • 执行进度跟踪：为空表示所有计划已完成
#
# ==============================================================================
# 三、Adapter 封装说明
# ==============================================================================
#
# LLMPlannerAdapter 封装了底层的 LLM_HLP_Generator，提供统一的规划接口。
#
# 继承：BasePlanningAgent
# 封装对象：runtime/agent_projects/LLM-Planner/hlp/hlp_planner.py
#
# 主要方法：
#   • start(): 初始化，加载 LLM_HLP_Generator
#   • stop(): 清理资源
#   • update_context(): 更新任务上下文
#   • plan(): 生成规划，返回 List[PlanningResult]
#
# 支持两种模式：
#   1. 普通模式：直接调用 generate_hlp 生成一次性计划
#   2. 动态重规划模式：调用 execute_with_dynamic_replanning 支持失败重试
#
# ==============================================================================
# 四、使用方法
# ==============================================================================
#
# 1. 创建 Adapter 实例
# -------------------
# adapter = LLMPlannerAdapter(
#     name='llm_planner',
#     knn_data_path='path/to/knn_set.pkl',
#     emb_model_name='paraphrase-MiniLM-L6-v2',
#     llm_config={
#         'provider': 'deepseek',
#         'model': 'deepseek-chat',
#         'api_key': 'your-api-key',
#         'base_url': 'https://api.deepseek.com',
#     },
#     k=9,
#     debug=False,
# )
#
# 2. 启动 Adapter
# -------------------
# await adapter.start()
#
# 3. 准备任务上下文
# -------------------
# context = {
#     'scenario_id': 'scenario_001',
#     'metadata': {
#         'vis_objs': ['table', 'cup', 'apple'],  # 可见物体
#         'completed_plans': [],                    # 已完成的计划
#         'step_instr': ['Navigate to table', 'Pickup apple'],
#         # 动态重规划模式选项：
#         'use_dynamic_replanning_loop': True,      # 启用重规划
#         'max_retries': 3,                          # 最大重试次数
#         'max_replanning': 10,                      # 最大重规划次数
#         'obj_sim_threshold': 0.8,                  # 物体相似度阈值
#     }
# }
#
# 4. 生成规划（普通模式）
# -------------------
# results = await adapter.plan(
#     instructions=['Pick up the apple'],
#     context=context,
# )
#
# # 结果解析
# for result in results:
#     print(f"Instruction: {result.instruction}")
#     print(f"Raw output: {result.raw_output}")
#     for action in result.actions:
#         print(f"  Action: {action['type']} {action['args']}")
#
# 5. 生成规划（动态重规划模式 - 外部循环）
# -------------------
# # 初始规划
# result = await adapter.plan(
#     instructions=['Pick up the apple and put it on the table'],
#     context={**context, 'metadata': {'use_dynamic_replanning_loop': True}},
# )
#
# # 获取 loop_state 和下一个计划
# loop_state = result[0].metadata['dynamic_replanning']['loop_state']
# next_plan = result[0].metadata['dynamic_replanning']['next_plan']
#
# # 模拟执行（实际由外部执行器执行）
# execution_result = {
#     'success': True,
#     'message': 'ok',
#     'plan': next_plan,
#     'visible_objects': ['table', 'cup', 'apple'],
# }
#
# # 反馈给 Agent，获取下一个计划
# next_result = await adapter.plan(
#     instructions=[],
#     context={
#         'metadata': {
#             'use_dynamic_replanning_loop': True,
#             'planner_loop_state': loop_state,
#             'execution_result': execution_result,
#             'visible_objects': ['table', 'cup', 'apple'],
#         }
#     },
# )
#
# 6. 停止 Adapter
# -------------------
# await adapter.stop()
#
# ==============================================================================

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
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
        split_reasoning,
    )
except ModuleNotFoundError:
    from reasoning_utils import (
        should_capture_reasoning,
        split_reasoning,
    )


logger = logging.getLogger(__name__)

_REASONING_OUTPUT_INSTRUCTION = (
    "Before the plans, output exactly one line starting with 'reasoning: ' "
    "briefly describing your task plan."
)


_CANONICAL_ACTIONS = {
    'OpenObject',
    'CloseObject',
    'PickupObject',
    'PutObject',
    'ToggleObjectOn',
    'ToggleObjectOff',
    'SliceObject',
    'Navigation',
}

_ACTION_ALIASES = {
    'open': 'OpenObject',
    'close': 'CloseObject',
    'pickup': 'PickupObject',
    'pick': 'PickupObject',
    'put': 'PutObject',
    'toggleon': 'ToggleObjectOn',
    'toggleoff': 'ToggleObjectOff',
    'slice': 'SliceObject',
    'navigate': 'Navigation',
    'navigation': 'Navigation',
}

_NEXT_PLANS_RE = re.compile(r'next\s*plans?\s*:', re.IGNORECASE)


def parse_hlp_text_to_actions(text: str) -> List[Dict[str, Any]]:
    """Parse planner output into executable actions only."""
    if not text:
        return []

    source_text = str(text)

    # Prefer the final "Next Plans:" segment when present to avoid parsing headers.
    next_plan_matches = list(_NEXT_PLANS_RE.finditer(source_text))
    if next_plan_matches:
        source_text = source_text[next_plan_matches[-1].start():]

    candidates: List[str] = []
    for line in source_text.splitlines():
        for fragment in line.split(','):
            fragment = fragment.strip()
            if not fragment:
                continue

            fragment = re.sub(r'^[-*]\s*', '', fragment)
            fragment = re.sub(r'^\d+[\)\.\-\s]*', '', fragment)
            fragment = _NEXT_PLANS_RE.sub('', fragment, count=1).strip()
            fragment = fragment.rstrip('.;').strip()
            if fragment:
                candidates.append(fragment)

    actions: List[Dict[str, Any]] = []
    for chunk in candidates:
        tokens = chunk.split()
        if not tokens:
            continue

        raw_action = tokens[0].strip(':')
        normalized_key = raw_action.lower()

        if raw_action in _CANONICAL_ACTIONS:
            action_type = raw_action
        else:
            action_type = _ACTION_ALIASES.get(normalized_key)

        if not action_type:
            continue

        args = tokens[1:]
        actions.append(
            {
                'type': action_type,
                'args': args,
                'raw': chunk,
                'step_index': len(actions),
            }
        )

    return actions


class LLMPlannerAdapter(BasePlanningAgent):
    """Adapter for runtime/agent_projects/LLM-Planner/hlp/hlp_planner.py."""

    def __init__(
        self,
        name: str = 'llm_planner',
        agent_root: Optional[Path | str] = None,
        knn_data_path: Optional[str] = None,
        emb_model_name: str = 'paraphrase-MiniLM-L6-v2',
        llm_config: Optional[Dict[str, Any]] = None,
        output_type: str = 'structured_actions',
        k: int = 9,
        debug: bool = False,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        capture_reasoning: bool = False,
        **_: Any,
    ):
        super().__init__(name=name)
        default_root = Path(__file__).resolve().parents[2] / 'agent_projects' / 'LLM-Planner'
        self._agent_root = Path(agent_root) if agent_root else default_root
        self._hlp_file = self._agent_root / 'hlp' / 'hlp_planner.py'
        self._knn_data_path = knn_data_path or str(self._agent_root / 'hlp' / 'knn_set.pkl')
        self._emb_model_name = emb_model_name
        self._llm_config = dict(llm_config or {})
        self._output_type = output_type
        self._k = int(k)
        self._debug = debug
        self._runtime_overrides = runtime_overrides or {}
        self._capture_reasoning = bool(capture_reasoning)

        self._module = None
        self._generator = None
        self._context: Dict[str, Any] = {}

    async def start(self) -> None:
        if not self._hlp_file.exists():
            raise FileNotFoundError(f'LLM-Planner entrypoint not found: {self._hlp_file}')

        spec = importlib.util.spec_from_file_location('external_llm_planner', str(self._hlp_file))
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Failed to load module spec from: {self._hlp_file}')

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module

        generator_cls = getattr(self._module, 'LLM_HLP_Generator', None)
        if generator_cls is None:
            raise RuntimeError('LLM_HLP_Generator class not found in hlp_planner.py')

        llm_cfg = dict(self._llm_config) if self._llm_config else None

        self._generator = generator_cls(
            knn_data_path=self._knn_data_path,
            emb_model_name=self._emb_model_name,
            debug=self._debug,
            llm_config=llm_cfg,
        )
        self._generator.reasoning_output_instruction = (
            _REASONING_OUTPUT_INSTRUCTION if self._reasoning_active() else ''
        )
        self.status = ProcessStatus.RUNNING

    async def stop(self) -> None:
        self._generator = None
        self._module = None
        self._context = {}
        self.status = ProcessStatus.TERMINATED

    async def update_context(self, context: Dict[str, Any]) -> None:
        self._context = dict(context or {})

    def _build_curr_task(self, instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        metadata = context.get('metadata') or {}

        visible_objects = (
            metadata.get('vis_objs')
            or metadata.get('visible_objects')
            or context.get('vis_objs')
            or []
        )

        completed_plans = metadata.get('completed_plans') or []
        step_instr = metadata.get('step_instr') or [instruction]

        return {
            'task_instr': [instruction],
            'step_instr': step_instr,
            'vis_objs': visible_objects,
            'completed_plans': completed_plans,
        }

    def _get_runtime_option(self, metadata: Dict[str, Any], key: str, default: Any) -> Any:
        if key in metadata:
            return metadata[key]
        return self._runtime_overrides.get(key, default)

    def _reasoning_active(self) -> bool:
        return should_capture_reasoning(self._llm_config, self._capture_reasoning)

    def _split_reasoning_output(self, raw_output: Any) -> tuple[Optional[str], str]:
        text = str(raw_output or '')
        if not self._reasoning_active():
            return None, text
        reasoning, cleaned = split_reasoning(text)
        return reasoning or None, cleaned

    def _reset_reasoning_trace(self) -> None:
        reset = getattr(self._generator, 'reset_reasoning_trace', None)
        if callable(reset):
            reset()
            return
        self._generator.last_reasoning = ''
        self._generator.last_traces = []

    def _captured_reasoning(self, prompt_reasoning: Optional[str] = None) -> Optional[str]:
        if not self._capture_reasoning:
            return None
        native_reasoning = str(
            getattr(self._generator, 'last_reasoning', '') or ''
        ).strip()
        return native_reasoning or prompt_reasoning or None

    def _llm_traces(self) -> List[Dict[str, Any]]:
        return [
            dict(trace)
            for trace in list(getattr(self._generator, 'last_traces', []) or [])
            if isinstance(trace, dict)
        ]

    async def plan(self, instructions: List[str], context: Dict[str, Any]) -> List[PlanningResult]:
        if self._generator is None:
            raise RuntimeError('LLMPlannerAdapter is not started. Call start() before planning.')

        if context:
            await self.update_context(context)

        scenario_id = str(self._context.get('scenario_id', 'unknown'))
        metadata = self._context.get('metadata') or {}
        results: List[PlanningResult] = []

        for instruction in instructions:
            curr_task = self._build_curr_task(instruction, self._context)
            self._reset_reasoning_trace()
            reasoning: Optional[str] = None
            use_dynamic_loop = bool(
                self._get_runtime_option(metadata, 'use_dynamic_replanning_loop', False)
                or metadata.get('planner_loop_state') is not None
                or metadata.get('execution_result') is not None
            )

            dynamic_metadata: Dict[str, Any] = {}
            if use_dynamic_loop:
                loop_result = await asyncio.to_thread(
                    self._generator.execute_with_dynamic_replanning,
                    curr_task,
                    self._k,
                    action_executor=None,
                    visible_objects_provider=None,
                    image_provider=None,
                    includeLow=bool(self._get_runtime_option(metadata, 'includeLow', False)),
                    dynamic=bool(self._get_runtime_option(metadata, 'dynamic', True)),
                    vision=bool(self._get_runtime_option(metadata, 'vision', False)),
                    max_retries=int(self._get_runtime_option(metadata, 'max_retries', 3)),
                    max_replanning=int(self._get_runtime_option(metadata, 'max_replanning', 10)),
                    obj_sim_threshold=float(self._get_runtime_option(metadata, 'obj_sim_threshold', 0.8)),
                    max_tokens=int(self._get_runtime_option(metadata, 'max_tokens', 300)),
                    temperature=float(self._get_runtime_option(metadata, 'temperature', 0.0)),
                    engine=self._get_runtime_option(metadata, 'engine', None),
                    loop_state=metadata.get('planner_loop_state'),
                    execution_result=metadata.get('execution_result'),
                    visible_objects=(
                        metadata.get('visible_objects')
                        or metadata.get('vis_objs')
                        or curr_task.get('vis_objs')
                    ),
                    images=metadata.get('images'),
                )
                remaining_plans = loop_result.get('remaining_plans', [])
                raw_output = '\n'.join(remaining_plans) if remaining_plans else str(loop_result.get('last_llm_output', '') or '')
                prefix_reasoning, raw_output = self._split_reasoning_output(raw_output)
                reasoning = self._captured_reasoning(prefix_reasoning)
                actions = parse_hlp_text_to_actions(', '.join(remaining_plans))
                dynamic_metadata = {
                    'planning_mode': 'dynamic_replanning_loop',
                    'dynamic_replanning': {
                        'status': loop_result.get('status'),
                        'next_plan': loop_result.get('next_plan'),
                        'replanning_count': loop_result.get('replanning_count'),
                        'retry_count': loop_result.get('retry_count'),
                        'completed_plans': loop_result.get('completed_plans', []),
                        'failed_plans': loop_result.get('failed_plans', []),
                        'remaining_plans': remaining_plans,
                        'seen_objs': loop_result.get('seen_objs', []),
                        'last_event': loop_result.get('last_event', {}),
                        'loop_state': loop_result.get('loop_state', {}),
                    },
                }
            else:
                raw_llm_output = await asyncio.to_thread(self._generator.generate_hlp, curr_task, self._k)
                prefix_reasoning, raw_output = self._split_reasoning_output(raw_llm_output)
                reasoning = self._captured_reasoning(prefix_reasoning)
                actions = parse_hlp_text_to_actions(str(raw_output))

            llm_traces = self._llm_traces()

            results.append(
                PlanningResult(
                    scenario_id=scenario_id,
                    instruction=instruction,
                    actions=actions,
                    raw_output=str(raw_output),
                    reasoning=reasoning,
                    metadata={
                        'agent_name': self.name,
                        'entrypoint': str(self._hlp_file),
                        'provider': self._llm_config.get('provider'),
                        'model': self._llm_config.get('model'),
                        'output_type': self._output_type,
                        'k': self._k,
                        'llm_trace': llm_traces,
                        **dynamic_metadata,
                    },
                )
            )

        return results


# ==================== Test Utilities for execute_with_dynamic_replanning ====================

def build_execution_simulator(execution_plan: Dict[str, Any]) -> tuple:
    """
    Build mock action_executor and visible_objects_provider for autonomous mode testing.

    Args:
        execution_plan: Dict mapping plan patterns to execution results
            Example: {
                'Navigate': {'success': True, 'message': 'ok'},
                'Pickup apple': {'success': False, 'message': 'object apple not found'},
            }

    Returns:
        (action_executor, visible_objects_provider) callables
    """
    state = {
        'attempt_count': 0,
        'visible_objects': ['table', 'cup', 'knife', 'apple'],
        'execution_log': [],
    }

    def action_executor(plan_text: str) -> Dict[str, Any]:
        state['attempt_count'] += 1
        plan = str(plan_text or '').strip()
        state['execution_log'].append(plan)

        # Check exact match first
        if plan in execution_plan:
            result = execution_plan[plan]
        else:
            # Check partial match
            result = None
            for key, val in execution_plan.items():
                if key.lower() in plan.lower():
                    result = val
                    break
            # Default: success
            if result is None:
                result = {'success': True, 'message': 'executed'}

        # Simulate object discovery after navigation
        if 'navigate' in plan.lower() and 'apple' not in state['visible_objects']:
            state['visible_objects'].append('apple')

        return result

    def visible_objects_provider() -> List[str]:
        return list(state['visible_objects'])

    return action_executor, visible_objects_provider


def format_response_summary(response: Dict[str, Any]) -> str:
    """Format dynamic replanning response for display."""
    lines = []
    lines.append(f"  Status: {response.get('status')}")
    lines.append(f"  Completed: {len(response.get('completed_plans', []))} plans")
    lines.append(f"    {response.get('completed_plans', [])}")
    lines.append(f"  Failed: {len(response.get('failed_plans', []))} plans")
    if response.get('failed_plans'):
        lines.append(f"    {response.get('failed_plans', [])}")
    lines.append(f"  Remaining: {len(response.get('remaining_plans', []))} plans")
    lines.append(f"    {response.get('remaining_plans', [])}")
    lines.append(f"  Next Plan: {response.get('next_plan')}")
    lines.append(f"  Replanning Count: {response.get('replanning_count')}")
    lines.append(f"  Retry Count: {response.get('retry_count')}")
    lines.append(f"  Seen Objects: {response.get('seen_objs', [])}")
    if response.get('last_event'):
        lines.append(f"  Last Event: {response['last_event'].get('type')}")
    return '\n'.join(lines)


# ==================== Test Task Definitions ====================

# Task for external mode (multi-step) testing
external_mode_task = {
    "task_instr": ["Pick up the apple and put it on the table."],
    "step_instr": [
        "Look around to find the apple",
        "Navigate to the apple location",
        "Pick up the apple",
        "Navigate to the table",
        "Put the apple on the table",
    ],
    "vis_objs": ["table", "cup"],
    "completed_plans": []
}

# Task for autonomous mode testing
autonomous_mode_task = {
    "task_instr": ["Pick up the apple and put it in the cup."],
    "step_instr": [
        "Navigate to the apple",
        "Pick up the apple",
        "Navigate to the cup",
        "Put the apple in the cup",
    ],
    "vis_objs": ["table", "cup"],
    "completed_plans": []
}


if __name__ == '__main__':
    import asyncio
    from pathlib import Path

    # Example task (same format as hlp_planner.py debug example)
    example_task = {
        "task_instr": ["Cook the potato and put it into the recycle bin."],
        "step_instr": [
            "Go to the potato near the sink",
            "Pick up the potato",
            "Go to the microwave next to the fridge.",
            "Open the microwave",
            "Cook the potato in the microwave",
            "Take out the potato",
            "Go to the recycle bin",
            "Throw the potato in the recycle bin"
        ],
        "vis_objs": ["cup", "microwave", "fridge", "garbagecan"],
        "completed_plans": [
            ("Navigation", "Countertop"),
            ("PickupObject", "Potato"),
            ("Navigation", "Microwave")
        ]
    }

    # Mock LLM config (user should update with real credentials)
    llm_config = {
        'provider': 'ds',
        'model': 'deepseek-chat',
        'api_key': '',
        'api_key_env': 'DEEPSEEK_API_KEY',
        'base_url': 'https://api.deepseek.com',
    }

    # Create adapter instance
    adapter = LLMPlannerAdapter(
        name='llm_planner_debug',
        emb_model_name='paraphrase-MiniLM-L6-v2',
        llm_config=llm_config,
        output_type='text',
        k=9,
        debug=True,
    )

    async def debug_adapter():
        print("\n---------------Adapter Debug Session----------------")
        print(f"Agent name: {adapter.name}")
        print(f"KNN data path: {adapter._knn_data_path}")
        print(f"Embedding model: {adapter._emb_model_name}")
        print(f"K (neighbors): {adapter._k}")
        print(f"Debug mode: {adapter._debug}")

        # Start the adapter
        print("\n[*] Starting adapter...")
        await adapter.start()
        print(f"[√] Adapter status: {adapter.status}")

        # Update context with mock environment
        # context独立于每一个Agent
        context = {
            'scenario_id': 'debug_scenario_001',
            'metadata': {
                'vis_objs': example_task['vis_objs'],
                'completed_plans': example_task['completed_plans'],
                'step_instr': example_task['step_instr'],
            }
        }
        await adapter.update_context(context)
        print(f"[√] Context updated: scenario_id={context['scenario_id']}")

        # Run planning
        print("\n[*] Running plan generation...")
        instructions = [example_task['task_instr'][0]]
        results = await adapter.plan(instructions, context)

        # Display results
        for idx, result in enumerate(results):
            print(f"\n---------Planning Result {idx + 1}---------")
            print(f"Scenario ID: {result.scenario_id}")
            print(f"Instruction: {result.instruction}")
            print(f"Raw output:\n{result.raw_output}")
            print(f"\nParsed actions ({len(result.actions)} total):")
            for action_idx, action in enumerate(result.actions, 1):
                print(f"  {action_idx}. Type: {action['type']}")
                print(f"     Args: {action['args']}")
                print(f"     Raw: {action['raw']}")

        # Cleanup
        await adapter.stop()
        print(f"\n[√] Adapter stopped: {adapter.status}")

    async def test_external_mode_replanning():
        """Test execute_with_dynamic_replanning in external mode with multi-step execution."""
        print("\n\n" + "="*70)
        print("TEST: External Mode Replanning (Multi-step)")
        print("="*70)

        # Start adapter
        print("\n[*] Starting adapter for external mode test...")
        await adapter.start()

        # Create task for external mode
        task = dict(external_mode_task)
        k = 9

        # ========== STEP 1: Initial Planning ==========
        print("\n[STEP 1] Initial Planning")
        print("-" * 50)

        response1 = await asyncio.to_thread(
            adapter._generator.execute_with_dynamic_replanning,
            task,
            k,
            action_executor=None,
            visible_objects_provider=None,
            dynamic=True
        )

        print("Step 1 Response:")
        print(format_response_summary(response1))
        print(f"  Initial High-Level Plans: {response1.get('initial_high_level_plans', [])}")

        if response1['status'] != 'awaiting_execution':
            print("  ⚠ Expected status 'awaiting_execution', got:", response1['status'])
        else:
            print("  ✓ Status correct: awaiting_execution")

        first_plan = response1.get('next_plan')
        if first_plan:
            print(f"  ✓ Got first plan: {first_plan}")
        else:
            print("  ✗ No first plan generated!")
            return

        # Save state for next step
        loop_state = response1.get('loop_state')

        # ========== STEP 2: Simulate execution failure & replanning ==========
        print("\n[STEP 2] Simulate Execution Failure with Object Not Found")
        print("-" * 50)

        # Simulate execution failure on first plan
        # 这里需要结构化地额外总结前一次执行的结果，作为下一次的输入。
        execution_result = {
            'success': False,
            'message': 'object apple not found',
            'plan': first_plan
        }

        # New objects discovered by observer
        observed_objects = ['table', 'cup', 'apple', 'knife']

        response2 = await asyncio.to_thread(
            adapter._generator.execute_with_dynamic_replanning,
            task,
            k,
            action_executor=None,
            visible_objects_provider=None,
            loop_state=loop_state,
            execution_result=execution_result,
            visible_objects=observed_objects,
            dynamic=True
        )

        print("Step 2 Response:")
        print(format_response_summary(response2))

        # Verify replanning happened
        if response2['replanning_count'] > response1['replanning_count']:
            print(f"  ✓ Replanning triggered: count increased from {response1['replanning_count']} to {response2['replanning_count']}")
        else:
            print(f"  ⚠ Replanning count unchanged: {response2['replanning_count']}")

        if response2.get('next_plan') and response2['next_plan'] != first_plan:
            print(f"  ✓ New plan generated after replan: {response2['next_plan']}")
        else:
            print(f"  ⚠ Plan unchanged or missing: {response2.get('next_plan')}")

        # Verify seen_objs accumulated
        if 'apple' in response2.get('seen_objs', []):
            print(f"  ✓ Discovered objects accumulated: {response2.get('seen_objs')}")
        else:
            print(f"  ✗ Objects not accumulated properly: {response2.get('seen_objs')}")

        # ========== STEP 3: Continue execution until completion ==========
        print("\n[STEP 3] Complete remaining plans")
        print("-" * 50)

        loop_state = response2.get('loop_state')
        max_steps = 5
        step = 3

        for _ in range(max_steps):
            if not loop_state:
                print(f"  ✗ Loop state lost at step {step}")
                break

            remaining = loop_state.get('pending_plans', [])
            if not remaining:
                print(f"  ✓ All plans completed at step {step}")
                break

            # Simulate successful execution
            next_plan = remaining[0]

            # TODO 这里需要接入实际的执行器和环境反馈，目前先模拟一个成功的执行结果，并且每一步都能看到完整的环境信息。

            execution_result = {
                'success': True,
                'message': 'ok',
                'plan': next_plan
            }

            response = await asyncio.to_thread(
                adapter._generator.execute_with_dynamic_replanning,
                task,
                k,
                action_executor=None,
                visible_objects_provider=None,
                loop_state=loop_state,
                execution_result=execution_result,
                visible_objects=['table', 'cup', 'apple', 'knife'],
                dynamic=True
            )

            print(f"\n  Step {step}: Plan '{next_plan}'")
            print(f"    Status: {response['status']}")
            print(f"    Completed: {len(response.get('completed_plans', []))}")
            print(f"    Remaining: {len(response.get('remaining_plans', []))}")

            loop_state = response.get('loop_state')
            step += 1

            if response['status'] == 'finished':
                print(f"\n  ✓ Final status: finished")
                print(f"  ✓ Completed plans: {response.get('completed_plans', [])}")
                break

        # Final summary
        print("\n[SUMMARY] External Mode Test")
        print(f"  Total steps taken: {step - 2}")
        print(f"  Final status: {response.get('status')}")
        print(f"  Plans completed: {len(response.get('completed_plans', []))}")
        print(f"  Plans failed: {len(response.get('failed_plans', []))}")

    async def test_autonomous_mode_with_retries():
        """Test execute_with_dynamic_replanning in autonomous mode with retry/replan logic."""
        print("\n\n" + "="*70)
        print("TEST: Autonomous Mode with Retries")
        print("="*70)

        # Start adapter
        print("\n[*] Starting adapter for autonomous mode test...")
        await adapter.start()

        # Define execution scenarios
        execution_scenarios = {
            'Navigate table': {'success': True, 'message': 'navigated'},
            'PickupObject apple': {
                'success': False,
                'message': 'object apple not found'  # Will trigger replanning
            },
            'Navigate cup': {'success': True, 'message': 'navigated'},
            'PutObject apple': {'success': True, 'message': 'put object'},
        }

        executor, vis_provider = build_execution_simulator(execution_scenarios)

        # Create task for autonomous mode
        task = dict(autonomous_mode_task)
        k = 9

        print("\n[*] Running autonomous execution with mocked executor...")
        print(f"    Task: {task['task_instr'][0]}")
        print(f"    Initial visible objects: {vis_provider()}")

        # Run full autonomous execution
        response = await asyncio.to_thread(
            adapter._generator.execute_with_dynamic_replanning,
            task,
            k,
            action_executor=executor,
            visible_objects_provider=vis_provider,
            image_provider=None,
            dynamic=True,
            max_retries=2,
            max_replanning=5,
        )

        print("\n[RESULTS] Autonomous Execution")
        print(format_response_summary(response))

        # Verify results
        completed = response.get('completed_plans', [])
        failed = response.get('failed_plans', [])

        print(f"\n[VERIFICATION]")
        print(f"  ✓ Execution completed with status: {response['status']}")
        print(f"  ✓ Plans completed: {len(completed)}")
        if completed:
            for i, plan in enumerate(completed, 1):
                print(f"    {i}. {plan}")

        if failed:
            print(f"  ℹ Plans failed/retried: {len(failed)}")
            for i, fail in enumerate(failed, 1):
                print(f"    {i}. {fail}")

        print(f"  ✓ Replanning iterations: {response['replanning_count']}")
        print(f"  ✓ Objects discovered: {response.get('seen_objs', [])}")

        if response['status'] == 'finished':
            print(f"  ✓ Test PASSED: Autonomous execution completed successfully")
        else:
            print(f"  ⚠ Test INCOMPLETE: Status is {response['status']}, not 'finished'")

    # Run debug session
    print("\n\n" + "="*70)
    print("MAIN: LLMPlannerAdapter Debug Session")
    print("="*70)

    asyncio.run(debug_adapter())

    # Run additional tests
    try:
        asyncio.run(test_external_mode_replanning())
    except Exception as e:
        print(f"\n✗ External mode test failed with error: {e}")
        import traceback
        traceback.print_exc()

    try:
        asyncio.run(test_autonomous_mode_with_retries())
    except Exception as e:
        print(f"\n✗ Autonomous mode test failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("All tests completed")
    print("="*70)

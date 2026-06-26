# ==============================================================================
# CodeBotler Planner Adapter - 总结文档
# ==============================================================================
#
# 本文件封装了基于 CodeBotler 的代码生成规划器，提供与评测管线统一的
# BasePlanningAgent 接口。
#
# ==============================================================================
# 一、Agent 运行流程
# ==============================================================================
#
# CodeBotler 是一种「单次 LLM 调用生成完整 Python 函数」的高层规划器。
# 与 CAP 的递归生成不同，CodeBotler 不做 fgen 递归，一次出完整程序。
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │                       初始化阶段                                       │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 加载 codebotler_planner.py 模块 → 构造 OpenAI client → 构造         │
# │ CodeBotlerPlanner(client, model_name, mode=..., api_set=...)       │
# └─────────────────────────────────────────────────────────────────────┘
#                              │
#                              ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │                       规划阶段                                         │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 输入: instruction + context (对象列表等)                             │
# │       ↓                                                             │
# │ 构建 prompt (chat messages / completion prefix+suffix)              │
# │       ↓                                                             │
# │ 单次 LLM 调用生成代码                                               │
# │       ↓                                                             │
# │ 后处理 + 组装完整程序                                                │
# │       ↓                                                             │
# │ 返回: 完整的 Python 程序字符串（不执行）                              │
# └─────────────────────────────────────────────────────────────────────┘
#
# 核心要点：
#   • LLM 调用时机：每个 instruction 一次
#   • 执行责任：本 adapter 只负责「生成代码」，不负责执行
#   • 状态保存：单 instruction 一次性完成，无跨 instruction 历史
#   • 配置灵活：mode (chat/completion/auto) × api_set (basic/extended) × few_shot (full/minimal)
#
# ==============================================================================
# 二、Adapter 封装说明
# ==============================================================================
#
# CodeBotlerPlannerAdapter 封装了底层的 CodeBotlerPlanner，提供统一的规划接口。
#
# 继承：BasePlanningAgent
# 封装对象：runtime/agent_projects/codebotler/codebotler_planner.py
#
# 主要方法：
#   • start():           加载模块、构造 OpenAI client + planner
#   • stop():            清理资源
#   • update_context():  更新任务上下文
#   • plan():            生成规划，返回 List[PlanningResult]
#
# 输出格式 (与 CAP 对齐，便于下游 Policy 统一消费 code)：
#   PlanningResult.actions = [{
#       'type': 'codebotler_plan',
#       'text': '',
#       'code': '<LLM 原始生成代码 (经 postprocess 清洗)>',
#   }]
#   PlanningResult.raw_output = '<LLM 原始生成代码 (经 postprocess 清洗)>'
#   PlanningResult.metadata['assembled_program'] = '<含头部注释+API说明的完整程序>'
#
# ==============================================================================
# 三、使用方法
# ==============================================================================
#
# 1. 创建 Adapter 实例
# -------------------
# adapter = CodeBotlerPlannerAdapter(
#     name='codebotler',
#     mode='chat',
#     api_set='extended',
#     few_shot='full',
#     llm_config={
#         'provider': 'deepseek',
#         'model': 'deepseek-chat',
#         'api_key': 'your-api-key',
#         'base_url': 'https://api.deepseek.com',
#     },
#     temperature=0.2,
#     max_tokens=512,
#     verbose=False,
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
#         'vis_objs': ['apple', 'fridge', 'cabinet', 'dining_table'],
#     }
# }
#
# 4. 生成规划
# -------------------
# results = await adapter.plan(
#     instructions=['put the apple in the fridge'],
#     context=context,
# )
# for result in results:
#     code = result.actions[0]['code']
#     print(code)
#
# 5. 停止 Adapter
# -------------------
# await adapter.stop()
#
# ==============================================================================

from __future__ import annotations

import asyncio
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
        should_capture_reasoning,
    )
except ModuleNotFoundError:
    from reasoning_utils import (
        should_capture_reasoning,
    )


logger = logging.getLogger(__name__)


_VALID_MODES = {'chat', 'completion', 'auto'}
_VALID_API_SETS = {'basic', 'extended'}
_VALID_FEWSHOT = {'full', 'minimal'}


def _format_object_list(objs: Any) -> str:
    """Format a collection of object names into a context line.

    CodeBotler expects context like:
    ``objects = ["apple", "banana", "fridge"]``
    """
    if not objs:
        return ''
    if isinstance(objs, str):
        return objs
    if isinstance(objs, (list, tuple, set)):
        items = [str(o) for o in objs]
        rendered = ', '.join(f'"{name}"' for name in items)
        return f'objects = [{rendered}]'
    return f'objects = [{objs!r}]'


class CodeBotlerPlannerAdapter(BasePlanningAgent):
    """Adapter for runtime/agent_projects/codebotler/codebotler_planner.py."""

    def __init__(
        self,
        name: str = 'codebotler',
        agent_root: Optional[Path | str] = None,
        entry_file: str = 'codebotler_planner.py',
        mode: str = 'chat',
        api_set: str = 'extended',
        few_shot: str = 'full',
        llm_config: Optional[Dict[str, Any]] = None,
        output_type: str = 'code_program',
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
        verbose: bool = False,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        capture_reasoning: bool = False,
        **_: Any,
    ):
        super().__init__(name=name)
        default_root = Path(__file__).resolve().parents[2] / 'agent_projects' / 'codebotler'
        self._agent_root = Path(agent_root) if agent_root else default_root
        self._entry_file = self._agent_root / entry_file
        self._llm_config = dict(llm_config or {})
        self._output_type = output_type
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._max_tokens = int(max_tokens)
        self._verbose = bool(verbose)
        self._runtime_overrides = runtime_overrides or {}
        self._capture_reasoning = bool(capture_reasoning)

        # 验证配置
        mode_normalized = (mode or 'chat').strip().lower()
        if mode_normalized not in _VALID_MODES:
            raise ValueError(
                f"Unsupported CodeBotler mode: {mode}. "
                f"Supported modes: {sorted(_VALID_MODES)}"
            )
        self._mode = mode_normalized

        api_set_normalized = (api_set or 'extended').strip().lower()
        if api_set_normalized not in _VALID_API_SETS:
            raise ValueError(
                f"Unsupported API set: {api_set}. "
                f"Supported: {sorted(_VALID_API_SETS)}"
            )
        self._api_set = api_set_normalized

        fewshot_normalized = (few_shot or 'full').strip().lower()
        if fewshot_normalized not in _VALID_FEWSHOT:
            raise ValueError(
                f"Unsupported few_shot: {few_shot}. "
                f"Supported: {sorted(_VALID_FEWSHOT)}"
            )
        self._few_shot = fewshot_normalized

        self._module = None
        self._planner = None
        self._client = None
        self._context: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._entry_file.exists():
            raise FileNotFoundError(
                f'CodeBotler planner entrypoint not found: {self._entry_file}'
            )

        module_dir = str(self._entry_file.parent)
        inserted = False
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            inserted = True

        try:
            spec = importlib.util.spec_from_file_location(
                'external_codebotler_planner', str(self._entry_file)
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f'Failed to load module spec from: {self._entry_file}')

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._module = module
        finally:
            if inserted:
                sys.path.remove(module_dir)

        # Build OpenAI client from llm_config
        openai_cls = getattr(self._module, 'OpenAI', None)
        if openai_cls is None:
            raise RuntimeError('OpenAI client class not found in codebotler_planner.py')

        api_key = str(self._llm_config.get('api_key', '') or '')
        if not api_key:
            api_key_env = str(self._llm_config.get('api_key_env', '') or '').strip()
            if api_key_env:
                api_key = os.getenv(api_key_env, '')
        base_url = str(self._llm_config.get('base_url', '') or '') or None
        model_name = str(self._llm_config.get('model', '') or '').strip()
        if not model_name:
            raise RuntimeError(
                'CodeBotlerPlannerAdapter requires a model name; please configure '
                'llm_provider/llm_model in the agent config.'
            )

        self._client = openai_cls(api_key=api_key, base_url=base_url)

        planner_cls = getattr(self._module, 'CodeBotlerPlanner', None)
        if planner_cls is None:
            raise RuntimeError('CodeBotlerPlanner class not found in codebotler_planner.py')

        self._planner = planner_cls(
            client=self._client,
            model_name=model_name,
            mode=self._mode,
            api_set=self._api_set,
            few_shot=self._few_shot,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
            verbose=self._verbose,
            use_reasoning_prompt=self._reasoning_active(),
        )

        self.status = ProcessStatus.RUNNING

    async def stop(self) -> None:
        self._planner = None
        self._client = None
        self._module = None
        self._context = {}
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

    def _build_codebotler_context(self, context: Dict[str, Any]) -> str:
        """Build the context string from adapter context metadata.

        CodeBotler expects context like:
        ``objects = ["apple", "banana", "fridge"]``
        """
        metadata = context.get('metadata') or {}

        # 1. Explicit user-supplied context wins
        explicit = (
            metadata.get('codebotler_context')
            or metadata.get('context_text')
        )
        if explicit:
            return str(explicit)

        # 2. Otherwise build from visible/objects lists
        vis_objs = (
            metadata.get('vis_objs')
            or metadata.get('visible_objects')
            or metadata.get('objects')
            or context.get('vis_objs')
            or []
        )
        return _format_object_list(vis_objs)

    def _build_actions(self, code: str) -> List[Dict[str, Any]]:
        return [
            {
                'type': 'codebotler_plan',
                'text': '',
                'code': code,
            }
        ]

    # ------------------------------------------------------------------
    # Planning entrypoint
    # ------------------------------------------------------------------

    async def plan(
        self,
        instructions: List[str],
        context: Dict[str, Any],
    ) -> List[PlanningResult]:
        if self._planner is None or self.status != ProcessStatus.RUNNING:
            raise RuntimeError(
                'CodeBotlerPlannerAdapter is not started. Call start() before planning.'
            )

        if context:
            await self.update_context(context)

        scenario_id = str(self._context.get('scenario_id', 'unknown'))
        codebotler_context = self._build_codebotler_context(self._context)
        results: List[PlanningResult] = []

        for instruction in instructions:
            components = await asyncio.to_thread(
                self._planner.plan_components,
                instruction,
                codebotler_context,
            )
            raw_code = str(components.get('raw_code') or '')
            assembled_program = str(components.get('assembled') or '')
            reasoning = str(components.get('reasoning') or '').strip() or None
            llm_traces = [
                dict(trace)
                for trace in list(components.get('llm_trace') or [])
                if isinstance(trace, dict)
            ]

            results.append(
                PlanningResult(
                    scenario_id=scenario_id,
                    instruction=instruction,
                    actions=self._build_actions(raw_code),
                    raw_output=raw_code,
                    reasoning=reasoning,
                    metadata={
                        'agent_name': self.name,
                        'entrypoint': str(self._entry_file),
                        'provider': self._llm_config.get('provider'),
                        'model': self._llm_config.get('model'),
                        'mode': self._mode,
                        'api_set': self._api_set,
                        'few_shot': self._few_shot,
                        'output_type': self._output_type,
                        'codebotler_context': codebotler_context,
                        'assembled_program': assembled_program,
                        'llm_trace': llm_traces,
                    },
                )
            )

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

        adapter = AgentFactory.create_from_config_map('codebotler', eval_cfg.agents, eval_cfg.llm)
        print(f'[*] Created adapter: {adapter.name}')

        await adapter.start()
        print(f'[*] Adapter started, status={adapter.status.name}')

        context = {
            'scenario_id': 'codebotler_debug_001',
            'metadata': {
                'vis_objs': [
                    'apple', 'banana', 'milk', 'mug', 'plate',
                    'book', 'remote', 'fridge', 'cabinet', 'dining_table',
                ],
            },
        }
        instructions = ['put the apple in the fridge']

        results = await adapter.plan(instructions, context)

        for idx, result in enumerate(results, 1):
            print(f"\n--------- Result {idx} ---------")
            print(f"Scenario: {result.scenario_id}")
            print(f"Instruction: {result.instruction}")
            print(f"Metadata: {json.dumps({k: v for k, v in result.metadata.items() if k != 'codebotler_context'}, indent=2)}")
            print(f"\n[Generated Program]\n{result.raw_output}")

        await adapter.stop()
        print(f'\n[*] Adapter stopped, status={adapter.status.name}')

    asyncio.run(_debug_main())

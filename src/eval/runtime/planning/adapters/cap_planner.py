# ==============================================================================
# Code-as-Policy (CAP) Planner Adapter - 总结文档
# ==============================================================================
#
# 本文件封装了基于 Code Generation 的 Code-as-Policy 规划器（CAP），提供与
# 评测管线统一的 BasePlanningAgent 接口。
#
# ==============================================================================
# 一、Agent 运行流程
# ==============================================================================
#
# CAP 是一种「一次生成完整 Python 程序」的高层规划器，没有 LLM-Planner 那种
# 多步循环、也不携带跨 turn 的执行历史（默认场景下）。整体流程是单次拉通的：
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │                       初始化阶段                                       │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 加载 cap_planner.py 模块 → 构造 OpenAI client → 构造                 │
# │ CodeAsPolicyPlanner(client, model_name, scenario=...)                │
# │ （可选）通过 register_atomic_apis 注入自定义原子 API                  │
# └─────────────────────────────────────────────────────────────────────┘
#                              │
#                              ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │                       规划阶段                                         │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 输入: instruction + context (object 列表等)                          │
# │       ↓                                                             │
# │ 调用 LMPCollector → 主 LMP 生成高层代码                              │
# │       ↓                                                             │
# │ FunctionParser 解析未定义函数 → LMPFGenCollector 递归生成函数体      │
# │       ↓                                                             │
# │ CodeCollector.assemble() 汇编完整 Python 程序                        │
# │       ↓                                                             │
# │ 返回: 完整的 Python 程序字符串（不执行）                              │
# └─────────────────────────────────────────────────────────────────────┘
#
# 核心要点：
#   • LLM 调用时机：每个 instruction 一次（含若干次 fgen 递归调用）
#   • 执行责任：本 adapter 只负责「生成代码」，不负责执行；下游 Agentic
#     Policy 拿到 code 字符串后自行编译/执行
#   • 状态保存：单 instruction 一次性完成，无 loop_state；adapter 自身
#     不在多个 instruction 间累积历史
#   • 场景切换：通过 scenario 参数 ('tabletop' / 'household') 选择 prompt
#     模板和默认原子 API 集合
#
# ==============================================================================
# 二、Adapter 封装说明
# ==============================================================================
#
# CAPPlannerAdapter 封装了底层的 CodeAsPolicyPlanner，提供统一的规划接口。
# 当前仅启用 'household' 场景（导航 + 容器开关 + 操作 appliance），
# 'tabletop' 场景虽然 cap_planner.py 仍支持，但不再通过本 adapter 暴露。
#
# 继承：BasePlanningAgent
# 封装对象：runtime/agent_projects/Code-as-Policy/cap_planner.py
#
# 主要方法：
#   • start():           加载模块、构造 OpenAI client + planner
#   • stop():            清理资源
#   • update_context():  更新任务上下文 (vis_objs/objects)
#   • plan():            生成规划，返回 List[PlanningResult]
#
# 输出格式 (与 ELLMER 对齐，便于下游 Policy 统一消费 code)：
#   PlanningResult.actions = [{
#       'type': 'cap_plan',
#       'text': '',                # CAP 不输出额外说明
#       'code': '<完整的 Python 程序>',
#   }]
#   PlanningResult.raw_output = '<完整的 Python 程序>'
#
# ==============================================================================
# 三、使用方法
# ==============================================================================
#
# 1. 创建 Adapter 实例
# -------------------
# adapter = CAPPlannerAdapter(
#     name='cap',
#     scenario='household',           # 当前仅支持 household
#     llm_config={
#         'provider': 'deepseek',
#         'model': 'deepseek-chat',
#         'api_key': 'your-api-key',
#         'base_url': 'https://api.deepseek.com',
#     },
#     temperature=0.0,
#     max_tokens=512,
#     verbose=False,
# )
#
# 2. 启动 Adapter
# -------------------
# await adapter.start()
#
# 3. 准备任务上下文 (context 中的 vis_objs 会被拼成 "objects = [...]" 字符串)
# -------------------
# context = {
#     'scenario_id': 'scenario_001',
#     'metadata': {
#         'vis_objs': ['knife', 'mug', 'fridge', 'cabinet', 'dining_table'],
#     }
# }
#
# 4. 生成规划
# -------------------
# results = await adapter.plan(
#     instructions=['grasp the knife in the kitchen and put it on the dining table'],
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
    from runtime.planning.reasoning_utils import should_capture_reasoning
except ModuleNotFoundError:
    from reasoning_utils import should_capture_reasoning


logger = logging.getLogger(__name__)


_VALID_SCENARIOS = {'tabletop', 'household'}


def _format_object_list(objs: Any) -> str:
    """Format a collection of object names into a CAP-friendly context line.

    CAP's prompt templates assume the caller appends a line like
    ``objects = ['blue block', 'yellow bowl', ...]`` before each query.
    """
    if not objs:
        return ''
    if isinstance(objs, str):
        # Already a pre-formatted context line.
        return objs
    if isinstance(objs, (list, tuple, set)):
        items = [str(o) for o in objs]
        rendered = ', '.join(f"'{name}'" for name in items)
        return f"objects = [{rendered}]"
    return f"objects = [{objs!r}]"


class CAPPlannerAdapter(BasePlanningAgent):
    """Adapter for runtime/agent_projects/Code-as-Policy/cap_planner.py."""

    def __init__(
        self,
        name: str = 'cap',
        agent_root: Optional[Path | str] = None,
        entry_file: str = 'cap_planner.py',
        scenario: str = 'household',
        llm_config: Optional[Dict[str, Any]] = None,
        output_type: str = 'code_program',
        temperature: float = 0.0,
        max_tokens: int = 512,
        verbose: bool = False,
        atomic_apis: Optional[Dict[str, Dict[str, Any]]] = None,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        capture_reasoning: bool = False,
        **_: Any,
    ):
        super().__init__(name=name)
        default_root = Path(__file__).resolve().parents[2] / 'agent_projects' / 'Code-as-Policy'
        self._agent_root = Path(agent_root) if agent_root else default_root
        self._entry_file = self._agent_root / entry_file
        self._llm_config = dict(llm_config or {})
        self._output_type = output_type
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._verbose = bool(verbose)
        self._runtime_overrides = runtime_overrides or {}
        self._capture_reasoning = bool(capture_reasoning)

        scenario_normalized = (scenario or 'tabletop').strip().lower()
        if scenario_normalized not in _VALID_SCENARIOS:
            raise ValueError(
                f"Unsupported CAP scenario: {scenario}. "
                f"Supported scenarios: {sorted(_VALID_SCENARIOS)}"
            )
        self._scenario = scenario_normalized

        self._atomic_apis = dict(atomic_apis or {})

        self._module = None
        self._planner = None
        self._client = None
        self._context: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._entry_file.exists():
            raise FileNotFoundError(f'CAP planner entrypoint not found: {self._entry_file}')

        module_dir = str(self._entry_file.parent)
        inserted = False
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            inserted = True

        try:
            spec = importlib.util.spec_from_file_location(
                'external_cap_planner', str(self._entry_file)
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f'Failed to load module spec from: {self._entry_file}')

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._module = module
        finally:
            if inserted:
                sys.path.remove(module_dir)

        # Build OpenAI client from llm_config (api_key/base_url already resolved
        # by AgentFactory; fall back to env if a key is still missing).
        openai_cls = getattr(self._module, 'OpenAI', None)
        if openai_cls is None:
            raise RuntimeError('OpenAI client class not found in cap_planner.py')

        api_key = str(self._llm_config.get('api_key', '') or '')
        if not api_key:
            api_key_env = str(self._llm_config.get('api_key_env', '') or '').strip()
            if api_key_env:
                api_key = os.getenv(api_key_env, '')
        base_url = str(self._llm_config.get('base_url', '') or '') or None
        model_name = str(self._llm_config.get('model', '') or '').strip()
        if not model_name:
            raise RuntimeError(
                'CAPPlannerAdapter requires a model name; please configure '
                'llm_provider/llm_model in the agent config.'
            )

        self._client = openai_cls(api_key=api_key, base_url=base_url)

        planner_cls = getattr(self._module, 'CodeAsPolicyPlanner', None)
        if planner_cls is None:
            raise RuntimeError('CodeAsPolicyPlanner class not found in cap_planner.py')

        self._planner = planner_cls(
            client=self._client,
            model_name=model_name,
            scenario=self._scenario,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            verbose=self._verbose,
            use_reasoning_prompt=should_capture_reasoning(
                self._llm_config,
                self._capture_reasoning,
            ),
        )

        # Register any user-provided atomic APIs on top of CAP defaults.
        if self._atomic_apis:
            try:
                self._planner.register_atomic_apis(self._atomic_apis)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning('Failed to register custom atomic APIs for CAP: %s', exc)

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

    def _build_cap_context(self, context: Dict[str, Any]) -> str:
        """Build the CAP context line from adapter context metadata.

        CAP expects a line like ``objects = ['blue block', 'yellow bowl']``
        before each query so that the LLM knows the scene composition.
        """
        metadata = context.get('metadata') or {}

        # 1. Explicit user-supplied context wins (already formatted)
        explicit = (
            metadata.get('cap_context')
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
                'type': 'cap_plan',
                'text': '',
                'code': code,
            }
        ]

    def _base_plan_metadata(self, cap_context: str) -> Dict[str, Any]:
        return {
            'agent_name': self.name,
            'entrypoint': str(self._entry_file),
            'provider': self._llm_config.get('provider'),
            'model': self._llm_config.get('model'),
            'scenario': self._scenario,
            'output_type': self._output_type,
            'cap_context': cap_context,
        }

    @staticmethod
    def _is_empty_cap_generation_error(exc: RuntimeError) -> bool:
        return 'produced no executable code' in str(exc)

    @staticmethod
    def _extract_reasoning_from_traces(traces: List[Dict[str, Any]]) -> str:
        """Concatenate ``reasoning_content`` from every LLM round trip.

        CAP's ``_extract_trace`` captures the CoT text that reasoning-capable
        models (DeepSeek-Reasoner, GLM-Z1, Qwen-QwQ, …) return alongside the
        generated code.  Without this, the evaluation pipeline has no
        ``PlanningResult.reasoning`` to feed the hazard-recognition judge.
        """
        fragments: List[str] = []
        for trace in traces or []:
            if not isinstance(trace, dict):
                continue
            rc = trace.get('reasoning_content')
            if rc and str(rc).strip():
                fragments.append(str(rc).strip())
        return '\n\n'.join(fragments)

    # ------------------------------------------------------------------
    # Planning entrypoint
    # ------------------------------------------------------------------

    async def plan(self, instructions: List[str], context: Dict[str, Any]) -> List[PlanningResult]:
        if self._planner is None or self.status != ProcessStatus.RUNNING:
            raise RuntimeError('CAPPlannerAdapter is not started. Call start() before planning.')

        if context:
            await self.update_context(context)

        scenario_id = str(self._context.get('scenario_id', 'unknown'))
        cap_context = self._build_cap_context(self._context)
        results: List[PlanningResult] = []

        for instruction in instructions:
            metadata = self._base_plan_metadata(cap_context)
            try:
                program = await asyncio.to_thread(
                    self._planner.plan,
                    instruction,
                    cap_context,
                )
            except RuntimeError as exc:
                # Always pull whatever LLM traces the planner managed to record
                # before failing — this is the only way to tell apart "model
                # refused" / "reasoning-only output" / "real bug".
                llm_trace = list(getattr(self._planner, 'last_traces', []) or [])
                if not self._is_empty_cap_generation_error(exc):
                    raise
                message = str(exc)
                logger.warning(
                    'CAP planner produced no executable code scenario=%s instruction=%s: %s',
                    scenario_id,
                    instruction,
                    message,
                )
                results.append(
                    PlanningResult(
                        scenario_id=scenario_id,
                        instruction=instruction,
                        actions=[],
                        raw_output=(
                            'CAP planner produced no executable code for this instruction. '
                            f'Original error: {message}'
                        ),
                        metadata={
                            **metadata,
                            'status': 'refused',
                            'planning_status': 'refused',
                            'refused': True,
                            'refusal_reason': 'no_executable_cap_code',
                            'cap_error': {
                                'type': exc.__class__.__name__,
                                'message': message,
                            },
                            'llm_trace': llm_trace,
                        },
                        reasoning=self._extract_reasoning_from_traces(llm_trace) or None,
                        refused=True,
                        refusal_reason='no_executable_cap_code',
                    )
                )
                continue
            program_str = str(program or '')
            # ``program_str`` is the assembled, runnable program (banner +
            # imports + fgen functions + main code + dependency block). We
            # surface only the pure LLM output (fgen functions + main code,
            # no scaffolding) to both ``raw_output`` AND ``actions[0]['code']``
            # so that downstream Policy receives exactly what the model wrote.
            # The full assembled program is preserved under
            # ``metadata['assembled_program']`` for inspection / re-exec.
            llm_only_str = str(getattr(self._planner, 'last_llm_only', '') or '')
            published_code = llm_only_str or program_str
            traces = list(getattr(self._planner, 'last_traces', []) or [])
            reasoning = self._extract_reasoning_from_traces(traces)

            results.append(
                PlanningResult(
                    scenario_id=scenario_id,
                    instruction=instruction,
                    actions=self._build_actions(published_code),
                    raw_output=published_code,
                    metadata={
                        **metadata,
                        'assembled_program': program_str,
                        'llm_trace': traces,
                    },
                    reasoning=reasoning or None,
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

        adapter = AgentFactory.create_from_config_map('cap', eval_cfg.agents, eval_cfg.llm)
        print(f'[*] Created adapter: {adapter.name}')

        await adapter.start()
        print(f'[*] Adapter started, status={adapter.status.name}')

        context = {
            'scenario_id': 'cap_debug_001',
            'metadata': {
                'vis_objs': [
                    'knife', 'banana', 'milk', 'mug', 'plate',
                    'book', 'remote', 'fridge', 'cabinet', 'dining_table',
                ],
            },
        }
        instructions = ['grasp the knife in the kitchen and put it on the dining table']

        results = await adapter.plan(instructions, context)

        for idx, result in enumerate(results, 1):
            print(f"\n--------- Result {idx} ---------")
            print(f"Scenario: {result.scenario_id}")
            print(f"Instruction: {result.instruction}")
            print(f"Metadata: {json.dumps({k: v for k, v in result.metadata.items() if k != 'cap_context'}, indent=2)}")
            print(f"\n[Generated Program]\n{result.raw_output}")

        await adapter.stop()
        print(f'\n[*] Adapter stopped, status={adapter.status.name}')

    asyncio.run(_debug_main())

# ==============================================================================
# ISR-LLM Planner Adapter - 总结文档
# ==============================================================================
#
# 本文件封装了 ICRA 2024 的 ISR-LLM（Iterative Self-Refined LLM）规划器，
# 将其作为 OMNISAFE 框架中「形式化 (PDDL) 规划输出」类型的代表 agent 接入。
#
# ISR-LLM 三阶段架构（Translator → Planner → Self/External Validator）原本
# 只支持 blocksworld / ballmoving / cooking 三个 toy domain。本 adapter:
#   * 加载 runtime/agent_projects/ISR-LLM/LLM/{Translator,Planner,Validator}
#   * 强制使用为框架补齐的 ``household`` domain few-shot examples
#   * 通过 OMNISAFE 的 ``llm_config`` 注入 openai.OpenAI(v1) client 给上游
#     模块（Translator/Planner/Validator 已被改造支持 client 注入）
#   * 默认走 ``LLM_trans_self_feedback`` —— 即 LLM Self-Validator 自精化循环，
#     不依赖外部仿真器，符合「仅关注 Planning 层面复现」的需求
#
# 输出格式：保留 ISR-LLM 原生 PDDL 风格 atoms（不做 PickupObject/PutObject 映射），
# 把 ``(pick-up apple kitchen_counter)``、``(navigate kitchen_counter dining_table)``
# 之类的原子原样下发给执行层，由执行层自行解释。
#
# ==============================================================================
# 一、Agent 运行流程
# ==============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │                       初始化阶段                                       │
# ├─────────────────────────────────────────────────────────────────────┤
# │ start() →                                                           │
# │   1. 解析 llm_config，构造 openai.OpenAI(api_key, base_url) v1 客户端 │
# │   2. importlib 加载上游 LLM/Translator/Translator.py                  │
# │      LLM/Planner/Planner.py、LLM/Validator/Validator.py              │
# │   3. 构造 SimpleNamespace arg（提供 model/domain/method/logdir/      │
# │      num_*_example），实例化 Translator/Planner/Validator 并注入 client │
# └─────────────────────────────────────────────────────────────────────┘
#                              │
#                              ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │                       规划阶段 (plan)                                  │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 对每条 instruction：                                                  │
# │   1. translator.query(instruction) → PDDL domain + problem            │
# │   2. for attempt in [0, max_num_refine]:                              │
# │        planner.query(planning_problem, is_append=True, temp=...)      │
# │           → action_sequence                                           │
# │        if method == 'self_feedback':                                  │
# │            validator.query(validate_question)                         │
# │            'Final answer: Yes' → break                                │
# │            'Final answer: No'  → planning_problem = hint              │
# │        if method == 'no_feedback': break (only one pass)              │
# │   3. 解析 action_sequence 为 [{type,name,args,raw,step_index}, ...]   │
# │   4. planner.init_messages(is_reinitialize=True) 清空对话              │
# └─────────────────────────────────────────────────────────────────────┘
#
# ==============================================================================
# 二、PlanningResult 输出格式
# ==============================================================================
#
# actions = [
#   {'type': 'pddl_action', 'name': 'pick-up', 'args': ['apple','kitchen_counter'],
#    'raw': '(pick-up apple kitchen_counter)', 'step_index': 0},
#   ...
# ]
# raw_output = 由 actions 拼接的干净 PDDL atom 序列（每行一个 (action args...)）
# metadata.isr_llm.raw_llm_output = Planner LLM 的原始返回字符串（含 Step 注释等，供审计/评测）
# metadata.isr_llm = {
#   'domain':         'household',
#   'method':         'LLM_trans_self_feedback',
#   'translator_output': str,         # PDDL domain+problem 全文（截断后）
#   'refine_attempts': int,            # 实际跑了多少 attempt（含首次）
#   'validator_verdict': 'Yes' | 'No' | 'unparsed' | 'n/a',
#   'raw_llm_output': str,             # Planner LLM 原始输出（截断后）
# }
#
# ==============================================================================
# 三、使用方法
# ==============================================================================
#
# context = {
#     'scenario_id': 'household_001',
#     'metadata': {
#         # 可选：把视觉/状态描述并入 instruction（adapter 会拼到 NL 前面）
#         'vis_objs': ['apple', 'knife', 'kitchen_counter', 'dining_table'],
#         'scene_description': 'The robot is at the kitchen counter.',
#     }
# }
# results = await adapter.plan(
#     instructions=['Pick up the apple and put it on the dining_table.'],
#     context=context,
# )
#
# ==============================================================================

from __future__ import annotations

import asyncio
import datetime
import importlib.util
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
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
        extract_step_reasoning,
        should_capture_reasoning,
    )
except ModuleNotFoundError:
    from reasoning_utils import (
        extract_step_reasoning,
        should_capture_reasoning,
    )


_VALID_DOMAINS = {'blocksworld', 'ballmoving', 'cooking', 'household'}
_VALID_METHODS = {
    'LLM_trans_self_feedback',
    'LLM_trans_no_feedback',
    'LLM_trans_exact_feedback',  # accepted but degrades to no_feedback without a sim
}

# Household domain action names; used to filter spurious atoms from LLM output.
_HOUSEHOLD_ACTIONS = {'navigate', 'pick-up', 'place', 'open', 'close', 'cook', 'slice'}

_ACTION_ATOM_RE = re.compile(r'\(([^()]+?)\)')


def _parse_pddl_action_sequence(action_sequence: str, domain: str) -> List[Dict[str, Any]]:
    """Parse an ISR-LLM Planner output into a list of structured PDDL actions.

    The Planner output looks like::

        Step 1: pick up the apple
        (pick-up apple kitchen_counter)
        Step 2: bring the apple over
        (navigate kitchen_counter dining_table)
        (place apple dining_table)

    We re-use the same regex the original simulators use (``r'\\(.*?\\)'``) but
    additionally drop non-action atoms when the domain is household (so a
    stray ``(at apple dining_table)`` echo doesn't leak through as an action).
    """
    if not action_sequence:
        return []

    atoms = _ACTION_ATOM_RE.findall(action_sequence)
    actions: List[Dict[str, Any]] = []

    for atom_body in atoms:
        atom_body = atom_body.strip()
        if not atom_body:
            continue
        tokens = atom_body.split()
        name = tokens[0]
        args = tokens[1:]

        if domain == 'household' and name not in _HOUSEHOLD_ACTIONS:
            continue

        actions.append({
            'type': 'pddl_action',
            'name': name,
            'args': args,
            'raw': '(' + atom_body + ')',
            'step_index': len(actions),
        })

    return actions


def _trim(text: Optional[str], max_chars: int = 4000) -> str:
    text = str(text or '')
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '... <truncated>'


def _actions_to_atom_sequence(actions: List[Dict[str, Any]]) -> str:
    """Render parsed PDDL actions back into a clean atom sequence string.

    Unlike the raw LLM output (which contains ``Step N:`` commentary and
    possibly spurious state-predicate echoes), this produces only the
    executable ``(action arg1 arg2)`` atoms, one per line. This is what
    gets stored in ``PlanningResult.raw_output`` so the downstream policy
    receives a noise-free plan.
    """
    return '\n'.join(str(a.get('raw') or '').strip() for a in actions if a.get('raw'))


class IsrLlmPlannerAdapter(BasePlanningAgent):
    """Adapter for runtime/agent_projects/ISR-LLM (ICRA'24 ISR-LLM).

    Drives the Translator → Planner → Self-Validator refinement loop and emits
    PDDL atoms as the structured action list, preserving ISR-LLM's role as the
    framework's *formalized planning output* representative.
    """

    def __init__(
        self,
        name: str = 'isr_llm',
        agent_root: Optional[Path | str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        domain: str = 'household',
        method: str = 'LLM_trans_self_feedback',
        num_trans_example: int = 2,
        num_plan_example: int = 2,
        num_valid_example: int = 3,
        max_num_refine: int = 3,
        max_refine_temperature: float = 0.4,
        output_type: str = 'pddl_actions',
        runtime_overrides: Optional[Dict[str, Any]] = None,
        capture_reasoning: bool = False,
        **_: Any,
    ):
        super().__init__(name=name)
        default_root = Path(__file__).resolve().parents[2] / 'agent_projects' / 'ISR-LLM'
        self._agent_root = Path(agent_root) if agent_root else default_root
        self._llm_config = dict(llm_config or {})
        self._output_type = output_type
        self._runtime_overrides = runtime_overrides or {}
        self._capture_reasoning = bool(capture_reasoning)

        domain_norm = (domain or 'household').strip().lower()
        if domain_norm not in _VALID_DOMAINS:
            raise ValueError(
                f'Unsupported ISR-LLM domain: {domain}. Supported: {sorted(_VALID_DOMAINS)}'
            )
        self._domain = domain_norm

        if method not in _VALID_METHODS:
            raise ValueError(
                f'Unsupported ISR-LLM method: {method}. Supported: {sorted(_VALID_METHODS)}'
            )
        self._method = method

        self._num_trans_example = int(num_trans_example)
        self._num_plan_example = int(num_plan_example)
        self._num_valid_example = int(num_valid_example)
        self._max_num_refine = int(max_num_refine)
        self._max_refine_temperature = float(max_refine_temperature)

        # Loaded in start()
        self._translator_module = None
        self._planner_module = None
        self._validator_module = None
        self._utils_module = None
        self._client = None
        self._translator = None
        self._planner = None
        self._validator = None
        self._context: Dict[str, Any] = {}
        self._logdir: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load_module(self, label: str, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f'ISR-LLM {label} not found: {file_path}')
        spec = importlib.util.spec_from_file_location(f'isr_llm_{label}', str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Failed to load module spec from: {file_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _build_openai_client(self):
        api_key = str(self._llm_config.get('api_key', '') or '')
        if not api_key:
            api_key_env = str(self._llm_config.get('api_key_env', '') or '').strip()
            if api_key_env:
                api_key = os.getenv(api_key_env, '')
        base_url = str(self._llm_config.get('base_url', '') or '') or None
        if not api_key:
            raise RuntimeError(
                'IsrLlmPlannerAdapter requires an api_key (via llm_config.api_key '
                'or llm_config.api_key_env). Please configure the LLM provider '
                'in configs/default_config.json.'
            )
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=base_url)

    def _build_arg(self, logdir: str) -> SimpleNamespace:
        model_name = str(self._llm_config.get('model', '') or '').strip()
        if not model_name:
            raise RuntimeError(
                'IsrLlmPlannerAdapter requires a model name; please configure '
                'llm_provider/llm_model in the agent config.'
            )
        return SimpleNamespace(
            model=model_name,
            domain=self._domain,
            method=self._method,
            logdir=logdir,
            num_trans_example=self._num_trans_example,
            num_plan_example=self._num_plan_example,
            num_valid_example=self._num_valid_example,
        )

    async def start(self) -> None:
        # Validate the agent_project tree.
        translator_file = self._agent_root / 'LLM' / 'Translator' / 'Translator.py'
        planner_file = self._agent_root / 'LLM' / 'Planner' / 'Planner.py'
        validator_file = self._agent_root / 'LLM' / 'Validator' / 'Validator.py'
        utils_file = self._agent_root / 'utils' / 'utils.py'

        # Ensure the agent_project root is on sys.path so the modules' own
        # ``from openai import ...`` style imports keep working consistently.
        agent_root_str = str(self._agent_root)
        if agent_root_str not in sys.path:
            sys.path.insert(0, agent_root_str)

        self._translator_module = self._load_module('translator', translator_file)
        self._planner_module = self._load_module('planner', planner_file)
        self._validator_module = self._load_module('validator', validator_file)
        self._utils_module = self._load_module('utils', utils_file)

        translator_cls = getattr(self._translator_module, 'Translator', None)
        planner_cls = getattr(self._planner_module, 'Planner', None)
        validator_cls = getattr(self._validator_module, 'Validator', None)
        if not (translator_cls and planner_cls and validator_cls):
            raise RuntimeError(
                'ISR-LLM modules missing Translator/Planner/Validator classes.'
            )

        # OpenAI v1 client wiring (the upstream classes accept ``client=``).
        self._client = self._build_openai_client()

        # Set up a per-session log directory under the agent project.
        logdir_base = self._agent_root / 'run_log' / ('adapter-' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        try:
            logdir_base.mkdir(parents=True, exist_ok=True)
            self._logdir = str(logdir_base)
        except OSError:
            # If the agent_project directory is read-only fall back to tempdir.
            self._logdir = tempfile.mkdtemp(prefix='isr_llm_adapter_')

        arg = self._build_arg(self._logdir)

        self._translator = translator_cls(arg, is_log_example=False, client=self._client)
        self._planner = planner_cls(arg, is_log_example=False, client=self._client)
        if self._method in {'LLM_trans_self_feedback', 'LLM_no_trans_self_feedback'}:
            self._validator = validator_cls(arg, is_log_example=False, client=self._client)

        self.status = ProcessStatus.RUNNING

    async def stop(self) -> None:
        self._translator = None
        self._planner = None
        self._validator = None
        self._translator_module = None
        self._planner_module = None
        self._validator_module = None
        self._utils_module = None
        self._client = None
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

    def _extract_reasoning(self, raw_output: str) -> Optional[str]:
        if not self._reasoning_active():
            return None
        reasoning = extract_step_reasoning(raw_output)
        return reasoning or None

    def _compose_description(self, instruction: str, context: Dict[str, Any]) -> str:
        """Prepend any visible-object / scene-description metadata to the instruction.

        ISR-LLM's Translator was originally trained on `Cooking_Sim.compose_description`
        style inputs (a single natural-language paragraph). For household we
        synthesize a similar paragraph from `vis_objs` + `scene_description` so
        the Translator has enough state context to emit a meaningful (:init ...).
        """
        metadata = context.get('metadata') or {}
        parts: List[str] = []

        scene = metadata.get('scene_description')
        if scene:
            parts.append(str(scene).strip())

        vis_objs = (
            metadata.get('vis_objs')
            or metadata.get('visible_objects')
            or context.get('vis_objs')
        )
        if vis_objs:
            if isinstance(vis_objs, (list, tuple, set)):
                rendered = ', '.join(str(o) for o in vis_objs)
            else:
                rendered = str(vis_objs)
            parts.append(f'Visible objects/locations: {rendered}.')

        parts.append(f'Your goal: {instruction.strip()}')

        return ' '.join(parts)

    def _temperature_for_attempt(self, attempt: int) -> float:
        if attempt == 0:
            return 0.0
        return min(self._max_refine_temperature, 0.1 * attempt)

    def _build_validate_question(self, planning_problem: str, action_sequence: str) -> str:
        """Mirror main.py's per-domain validator question phrasing."""
        try:
            pddl_init_state, pddl_goal_state = self._utils_module.extract_state_pddl(planning_problem, self._domain)
            action_description = self._utils_module.extract_action_description(action_sequence, self._domain)
        except Exception as exc:  # extraction can fail when the Translator output is malformed
            logger.warning('extract_state_pddl/extract_action_description failed: %s', exc)
            return (
                'Question:\nFull planning problem (raw):\n' + str(planning_problem) +
                '\nExamined action sequence:\n' + str(action_sequence)
            )

        if self._domain == 'household':
            header = 'Household initial state:\n'
        elif self._domain == 'blocksworld':
            header = 'Block initial state:\n'
        elif self._domain == 'ballmoving':
            header = 'Robot and ball initial state: \n'
        else:
            header = 'Initial state: \n'

        return (
            'Question:\n' + header + pddl_init_state +
            '\nGoal state:\n' + pddl_goal_state +
            '\nExamined action sequence:\n' + action_description
        )

    def _self_feedback_hint(self) -> str:
        if self._domain == 'household':
            return (
                'Goal is not satisfied. Please find a new plan by considering '
                'the preconditions of each action (robot location, what is held, '
                'container open/closed).'
            )
        if self._domain == 'blocksworld':
            return 'Goal is not satisfied. Please find a new plan by considering the goals from bottom to top.'
        if self._domain == 'ballmoving':
            return 'Goal is not satisfied. Please find a new plan by considering the locations of balls.'
        if self._domain == 'cooking':
            return 'Goal is not satisfied. Please find a new plan by considering the ingredients needed in each pot.'
        return 'Goal is not satisfied. Please find a new plan.'

    # ------------------------------------------------------------------
    # Core ISR-LLM loop (sync inner; called via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _plan_one_instruction(self, instruction: str) -> Dict[str, Any]:
        description = self._compose_description(instruction, self._context)

        # Step 1: Translator (NL → PDDL domain + problem).
        translator_resp = self._translator.query(description, is_append=False)
        planning_problem = translator_resp['choices'][0]['message']['content']

        # Step 2 + 3: Planner + Self-Validator refinement loop.
        method_uses_self_validator = self._method in {
            'LLM_trans_self_feedback', 'LLM_no_trans_self_feedback'
        }
        max_attempts = self._max_num_refine + 1 if method_uses_self_validator else 1

        action_sequence = ''
        validator_verdict = 'n/a'
        refine_attempts = 0
        current_planning_problem = planning_problem

        for attempt in range(max_attempts):
            refine_attempts = attempt + 1
            temperature = self._temperature_for_attempt(attempt)
            planner_resp = self._planner.query(
                current_planning_problem, is_append=True, temperature=temperature
            )
            action_sequence = planner_resp['choices'][0]['message']['content']

            if not method_uses_self_validator:
                break

            # Self-Validator branch.
            validate_question = self._build_validate_question(planning_problem, action_sequence)
            validator_resp = self._validator.query(validate_question, is_append=False)
            validator_content = validator_resp['choices'][0]['message']['content']

            valid_split = validator_content.split('Final answer:', 1)
            if len(valid_split) == 1:
                # Validator returned no 'Final answer:' → bail out conservatively.
                validator_verdict = 'unparsed'
                break

            tail = valid_split[1]
            if 'Yes' in tail:
                validator_verdict = 'Yes'
                break
            elif 'No' in tail:
                validator_verdict = 'No'
                current_planning_problem = self._self_feedback_hint()
                continue
            else:
                validator_verdict = 'unparsed'
                break

        # Reset the planner conversation so the next instruction starts clean
        # (matches what main.py does after each test case).
        try:
            self._planner.init_messages(is_reinitialize=True)
        except Exception as exc:  # defensive; should never trip
            logger.warning('Planner.init_messages reinitialize failed: %s', exc)

        actions = _parse_pddl_action_sequence(action_sequence, self._domain)

        return {
            'action_sequence': action_sequence,
            'translator_output': planning_problem,
            'refine_attempts': refine_attempts,
            'validator_verdict': validator_verdict,
            'actions': actions,
        }

    # ------------------------------------------------------------------
    # Planning entrypoint
    # ------------------------------------------------------------------

    async def plan(self, instructions: List[str], context: Dict[str, Any]) -> List[PlanningResult]:
        if self._translator is None or self._planner is None or self.status != ProcessStatus.RUNNING:
            raise RuntimeError('IsrLlmPlannerAdapter is not started. Call start() before planning.')

        if context:
            await self.update_context(context)

        scenario_id = str(self._context.get('scenario_id', 'unknown'))
        results: List[PlanningResult] = []

        for instruction in instructions:
            outcome = await asyncio.to_thread(self._plan_one_instruction, instruction)

            raw_llm_output = str(outcome['action_sequence'])
            actions = outcome['actions']
            # raw_output: clean PDDL atom sequence (no Step commentary / state echoes)
            # metadata.raw_llm_output: verbatim Planner LLM output for audit/eval
            raw_output = _actions_to_atom_sequence(actions)

            metadata = {
                'agent_name': self.name,
                'entrypoint': str(self._agent_root),
                'provider': self._llm_config.get('provider'),
                'model': self._llm_config.get('model'),
                'output_type': self._output_type,
                'isr_llm': {
                    'domain': self._domain,
                    'method': self._method,
                    'translator_output': _trim(outcome['translator_output']),
                    'refine_attempts': outcome['refine_attempts'],
                    'validator_verdict': outcome['validator_verdict'],
                    'logdir': self._logdir,
                    'raw_llm_output': _trim(raw_llm_output),
                },
            }
            results.append(
                PlanningResult(
                    scenario_id=scenario_id,
                    instruction=instruction,
                    actions=actions,
                    raw_output=raw_output,
                    reasoning=self._extract_reasoning(raw_llm_output),
                    metadata=metadata,
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

        adapter = AgentFactory.create_from_config_map('isr_llm', eval_cfg.agents, eval_cfg.llm)
        print(f'[*] Created adapter: {adapter.name}')

        await adapter.start()
        print(f'[*] Adapter started, status={adapter.status.name}')

        context = {
            'scenario_id': 'isr_llm_debug_001',
            'metadata': {
                'vis_objs': ['apple', 'knife', 'kitchen_counter', 'dining_table'],
                'scene_description': (
                    'The robot is at the kitchen_counter. An apple is at the '
                    'kitchen_counter. A knife is at the kitchen_counter. The '
                    'dining_table is reachable.'
                ),
            },
        }
        instructions = ['Pick up the apple and put it on the dining_table.']

        results = await adapter.plan(instructions, context)

        for idx, result in enumerate(results, 1):
            print(f'\n--------- Result {idx} ---------')
            print(f'Scenario: {result.scenario_id}')
            print(f'Instruction: {result.instruction}')
            isr_meta = result.metadata.get('isr_llm', {})
            print(f"  domain={isr_meta.get('domain')} method={isr_meta.get('method')}")
            print(f"  refine_attempts={isr_meta.get('refine_attempts')} "
                  f"verdict={isr_meta.get('validator_verdict')}")
            print(f'\n[Raw action sequence]\n{result.raw_output}')
            print(f'\n[Parsed actions] ({len(result.actions)})')
            for action in result.actions:
                print(f"  - {action['name']} {action['args']}   raw={action['raw']}")

        await adapter.stop()
        print(f'\n[*] Adapter stopped, status={adapter.status.name}')

    asyncio.run(_debug_main())

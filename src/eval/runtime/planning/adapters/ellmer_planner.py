from __future__ import annotations

import asyncio
import inspect
import json
import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from core.base import BasePlanningAgent, PlanningResult, ProcessStatus
except ModuleNotFoundError:
    eval_root = Path(__file__).resolve().parents[3]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import BasePlanningAgent, PlanningResult, ProcessStatus

try:
    from configs.config import EvalConfig
except ModuleNotFoundError:
    eval_root = Path(__file__).resolve().parents[3]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from configs.config import EvalConfig

try:
    from runtime.planning.reasoning_utils import (
        extract_chat_completion_trace,
        join_reasoning,
        should_capture_reasoning,
    )
except ModuleNotFoundError:
    from reasoning_utils import (
        extract_chat_completion_trace,
        join_reasoning,
        should_capture_reasoning,
    )


logger = logging.getLogger(__name__)


def _extract_python_blocks(text: str) -> List[str]:
    if not text:
        return []
    return [match.strip() for match in re.findall(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)]


def _extract_text_and_code(raw_reply: str) -> Dict[str, str]:
    code_blocks = _extract_python_blocks(raw_reply)
    code = '\n\n'.join(code_blocks).strip()
    text = re.sub(r"```python\s*.*?```", '', raw_reply, flags=re.DOTALL | re.IGNORECASE).strip()
    return {
        'text': text,
        'code': code,
    }


class ELLMERPlannerAdapter(BasePlanningAgent):
    """Adapter for runtime/agent_projects/ELLMER/haystack_agent.py."""

    def __init__(
        self,
        name: str = 'ellmer',
        agent_root: Optional[Path | str] = None,
        entry_file: str = 'haystack_agent.py',
        llm_config: Optional[Dict[str, Any]] = None,
        embedding_llm_config: Optional[Dict[str, Any]] = None,
        output_type: str = 'raw_agent_output',
        keep_history_across_scenarios: bool = False,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        capture_reasoning: bool = False,
        **_: Any,
    ):
        super().__init__(name=name)
        default_root = Path(__file__).resolve().parents[2] / 'agent_projects' / 'ELLMER'
        self._agent_root = Path(agent_root) if agent_root else default_root
        self._entry_file = self._agent_root / entry_file
        self._llm_config = dict(llm_config or {})
        self._embedding_llm_config = dict(embedding_llm_config or {})
        self._output_type = output_type
        self._keep_history_across_scenarios = bool(keep_history_across_scenarios)
        self._runtime_overrides = runtime_overrides or {}
        self._capture_reasoning = bool(capture_reasoning)

        self._module = None
        self._chat_fn: Optional[Callable[..., Any]] = None
        self._chat_supports_single_arg = False
        self._chat_supports_env_ctx = False
        self._rag_fn: Optional[Callable[[str], Any]] = None
        self._rag_supports_env_ctx = False
        self._context: Dict[str, Any] = {}
        self._current_scenario_id: str = ''
        self._initial_messages_snapshot: Optional[List[Any]] = None

    async def start(self) -> None:
        if not self._entry_file.exists():
            raise FileNotFoundError(f'ELLMER entrypoint not found: {self._entry_file}')

        module_dir = str(self._entry_file.parent)
        inserted = False
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
            inserted = True

        try:
            spec = importlib.util.spec_from_file_location('external_ellmer_planner', str(self._entry_file))
            if spec is None or spec.loader is None:
                raise RuntimeError(f'Failed to load module spec from: {self._entry_file}')

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._module = module
        finally:
            if inserted:
                sys.path.remove(module_dir)

        init_fn = getattr(self._module, 'initialize_agent', None)
        if callable(init_fn):
            await asyncio.to_thread(
                init_fn,
                llm_config=dict(self._llm_config) if self._llm_config else None,
                embedding_llm_config=dict(self._embedding_llm_config) if self._embedding_llm_config else None,
                runtime_overrides=dict(self._runtime_overrides),
            )

        chat_fn = getattr(self._module, 'chatbot_with_fc', None)
        rag_fn = getattr(self._module, 'rag_pipeline_func', None)

        if not callable(chat_fn) and not callable(rag_fn):
            raise RuntimeError(
                'Neither chatbot_with_fc(message, history) nor rag_pipeline_func(query) '
                'was found in haystack_agent.py'
            )

        self._chat_fn = chat_fn if callable(chat_fn) else None
        if self._chat_fn is not None:
            try:
                signature = inspect.signature(self._chat_fn)
                signature.bind_partial('probe-message')
                self._chat_supports_single_arg = True
            except TypeError:
                self._chat_supports_single_arg = False
        self._rag_fn = rag_fn if callable(rag_fn) else None
        # Check if the loaded functions support environment_context parameter
        self._chat_supports_env_ctx = False
        if self._chat_fn is not None:
            try:
                sig = inspect.signature(self._chat_fn)
                if 'environment_context' in sig.parameters:
                    self._chat_supports_env_ctx = True
            except (ValueError, TypeError):
                pass
        self._rag_supports_env_ctx = False
        if self._rag_fn is not None:
            try:
                sig = inspect.signature(self._rag_fn)
                if 'environment_context' in sig.parameters:
                    self._rag_supports_env_ctx = True
            except (ValueError, TypeError):
                pass

        module_messages = getattr(self._module, 'messages', None)
        if isinstance(module_messages, list):
            self._initial_messages_snapshot = list(module_messages)

        self.status = ProcessStatus.RUNNING

    async def stop(self) -> None:
        self._module = None
        self._chat_fn = None
        self._chat_supports_single_arg = False
        self._chat_supports_env_ctx = False
        self._rag_fn = None
        self._rag_supports_env_ctx = False
        self._context = {}
        self._current_scenario_id = ''
        self._initial_messages_snapshot = None
        self.status = ProcessStatus.TERMINATED

    async def update_context(self, context: Dict[str, Any]) -> None:
        self._context = dict(context or {})

        scenario_id = str(self._context.get('scenario_id', '') or '')
        reset_by_config = bool((self._context.get('metadata') or {}).get('reset_history', False))

        if reset_by_config:
            self._reset_history()
            self._current_scenario_id = scenario_id
            return

        if not self._keep_history_across_scenarios and scenario_id and scenario_id != self._current_scenario_id:
            self._reset_history()
            self._current_scenario_id = scenario_id

    def _reset_history(self) -> None:
        if self._module is None:
            return

        if self._initial_messages_snapshot is not None:
            setattr(self._module, 'messages', list(self._initial_messages_snapshot))

    def _normalize_reply(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload

        if isinstance(payload, dict):
            if isinstance(payload.get('reply'), str):
                return payload['reply']
            if isinstance(payload.get('text'), str):
                return payload['text']

        return str(payload)

    def _build_actions(self, raw_reply: str) -> List[Dict[str, Any]]:
        parsed = _extract_text_and_code(raw_reply)
        return [
            {
                'type': 'ellmer_plan',
                'text': parsed['text'],
                'code': parsed['code'],
            }
        ]

    def _reasoning_active(self) -> bool:
        return should_capture_reasoning(self._llm_config, self._capture_reasoning)

    def _captured_llm_trace(self) -> List[Dict[str, Any]]:
        if not self._capture_reasoning or self._module is None:
            return []
        getter = getattr(self._module, 'get_last_llm_response', None)
        if not callable(getter):
            return []
        response = getter()
        if response is None:
            return []
        trace = extract_chat_completion_trace(
            response,
            model=str(self._llm_config.get('model') or ''),
        )
        trace['label'] = 'main:ellmer_rag'
        trace['attempt'] = 0
        return [trace]

    def _build_environment_context(self, context: Dict[str, Any]) -> str:
        """Build environment context string from the adapter's context metadata.

        Extracts dynamic environment information (visible objects, etc.)
        and formats it as a text block to be injected into the ELLMER prompt template
        via the ``{{environment_context}}`` Jinja2 variable.
        """
        metadata = context.get('metadata') or {}
        parts: List[str] = []

        # Visible objects
        vis_objs = (
            metadata.get('vis_objs')
            or metadata.get('visible_objects')
            or context.get('vis_objs')
        )
        if vis_objs:
            if isinstance(vis_objs, (list, tuple)):
                parts.append(f"Visible objects: {', '.join(str(o) for o in vis_objs)}")
            else:
                parts.append(f"Visible objects: {vis_objs}")

        # Any free-form environment description
        env_desc = metadata.get('environment_description')
        if env_desc:
            parts.append(str(env_desc))

        return '\n'.join(parts)

    async def _invoke_planner(self, instruction: str, environment_context: str = "") -> str:
        if self._chat_fn is not None:
            if self._chat_supports_env_ctx:
                payload = await asyncio.to_thread(self._chat_fn, instruction, [], environment_context)
            elif self._chat_supports_single_arg:
                payload = await asyncio.to_thread(self._chat_fn, instruction)
            else:
                payload = await asyncio.to_thread(self._chat_fn, instruction, [])
            reply = self._normalize_reply(payload)
            return reply

        if self._rag_fn is not None:
            if self._rag_supports_env_ctx:
                payload = await asyncio.to_thread(self._rag_fn, instruction, environment_context)
            else:
                payload = await asyncio.to_thread(self._rag_fn, instruction)
            reply = self._normalize_reply(payload)
            return reply

        raise RuntimeError('ELLMER planner callable is unavailable.')

    async def plan(self, instructions: List[str], context: Dict[str, Any]) -> List[PlanningResult]:
        if self.status != ProcessStatus.RUNNING:
            raise RuntimeError('ELLMERPlannerAdapter is not started. Call start() before planning.')

        if context:
            await self.update_context(context)

        scenario_id = str(self._context.get('scenario_id', 'unknown'))
        environment_context = self._build_environment_context(self._context)
        results: List[PlanningResult] = []

        for instruction in instructions:
            raw_reply = await self._invoke_planner(instruction, environment_context=environment_context)
            llm_trace = self._captured_llm_trace()
            code_blocks = _extract_python_blocks(raw_reply)
            raw_output = '\n\n'.join(code_blocks).strip() if code_blocks else raw_reply
            actions = self._build_actions(raw_reply)
            native_reasoning = join_reasoning(
                str(trace.get('reasoning_content') or '')
                for trace in llm_trace
            )
            prompt_reasoning = ''
            if self._reasoning_active() and actions:
                prompt_reasoning = str(actions[0].get('text', '') or '').strip()
            reasoning_text = (
                native_reasoning or prompt_reasoning or None
                if self._capture_reasoning
                else None
            )

            results.append(
                PlanningResult(
                    scenario_id=scenario_id,
                    instruction=instruction,
                    actions=actions,
                    raw_output=raw_output,
                    reasoning=reasoning_text,
                    metadata={
                        'agent_name': self.name,
                        'entrypoint': str(self._entry_file),
                        'provider': self._llm_config.get('provider'),
                        'model': self._llm_config.get('model'),
                        'embedding_provider': self._embedding_llm_config.get('provider'),
                        'embedding_model': self._embedding_llm_config.get('model'),
                        'output_type': self._output_type,
                        'llm_trace': llm_trace,
                    },
                )
            )

        return results


def _build_debug_context(scenario_id: str, vis_objs: List[str], reset_history: bool) -> Dict[str, Any]:
    return {
        'scenario_id': scenario_id,
        'metadata': {
            'vis_objs': vis_objs,
            'reset_history': reset_history,
        },
    }


def _print_result_summary(result: PlanningResult, preview_chars: int = 9999) -> None:
    action = result.actions[0] if result.actions else {}
    text = str(action.get('text', '') or '')
    code = str(action.get('code', '') or '')

    text_preview = text[:preview_chars]
    if len(text) > preview_chars:
        text_preview += '...'

    code_preview = code[:preview_chars]
    if len(code) > preview_chars:
        code_preview += '...'

    print(f'\n[RESULT] scenario_id={result.scenario_id}')
    print(f'[RESULT] instruction={result.instruction}')
    print(f'[RESULT] text_chars={len(text)}')
    print(f'[RESULT] code_chars={len(code)}')
    print(f'[RESULT] text_preview={text_preview}')
    print(f'[RESULT] code_preview={code_preview}')


async def _run_debug(
    *,
    instructions: List[str],
    scenario_id: str,
    vis_objs: List[str],
    config_path: str,
    agent_name: str,
    agent_root: Optional[str],
    entry_file: str,
    keep_history_across_scenarios: bool,
    output_json: bool,
) -> int:
    try:
        from runtime.planning.factory import AgentFactory
    except ModuleNotFoundError:
        eval_root = Path(__file__).resolve().parents[3]
        if str(eval_root) not in sys.path:
            sys.path.insert(0, str(eval_root))
        from runtime.planning.factory import AgentFactory

    cfg_path = Path(config_path)
    if cfg_path.exists():
        eval_cfg = EvalConfig.from_json(str(cfg_path))
    else:
        raise FileNotFoundError(f'Config file not found: {cfg_path}')

    adapter = AgentFactory.create_from_config_map(agent_name, eval_cfg.agents, eval_cfg.llm)
    if not hasattr(adapter, '_agent_root') or not hasattr(adapter, '_entry_file'):
        raise TypeError(
            f'Configured agent "{agent_name}" does not expose the expected ELLMER debug attributes.'
        )

    if agent_root is not None:
        adapter._agent_root = Path(agent_root)
        adapter._entry_file = adapter._agent_root / entry_file
    else:
        adapter._entry_file = adapter._agent_root / entry_file
    adapter._keep_history_across_scenarios = keep_history_across_scenarios

    context = _build_debug_context(
        scenario_id=scenario_id,
        vis_objs=vis_objs,
        reset_history=True,
    )

    print('[CHAIN] start()')
    await adapter.start()
    print(f'[CHAIN] status={adapter.status.name}')

    try:
        print('[CHAIN] update_context()')
        await adapter.update_context(context)

        print('[CHAIN] plan()')
        results = await adapter.plan(instructions, context)
        print(f'[CHAIN] plan_results={len(results)}')

        if output_json:
            payload = []
            for item in results:
                action = item.actions[0] if item.actions else {}
                payload.append(
                    {
                        'scenario_id': item.scenario_id,
                        'instruction': item.instruction,
                        'text': action.get('text', ''),
                        'code': action.get('code', ''),
                    }
                )
            print(json.dumps(payload, ensure_ascii=True, indent=2))
        else:
            for item in results:
                _print_result_summary(item)

        print('[CHAIN] stop()')
        await adapter.stop()
        print(f'[CHAIN] status={adapter.status.name}')
        return 0
    except Exception:
        print('[CHAIN] stop() after error')
        await adapter.stop()
        print(f'[CHAIN] status={adapter.status.name}')
        raise


def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description='Debug ELLMER planner adapter call chain.')
    parser.add_argument(
        '--instruction',
        action='append',
        default=[],
        help='Instruction text. Repeat for multi-turn debug.',
    )
    parser.add_argument('--config', default='configs/default_config.json')
    parser.add_argument('--agent', default='ellmer')
    parser.add_argument('--scenario-id', default='ellmer_debug_scenario')
    parser.add_argument('--vis-objs', default='table,cup')
    parser.add_argument('--agent-root', default='')
    parser.add_argument('--entry-file', default='haystack_agent.py')
    parser.add_argument('--keep-history-across-scenarios', action='store_true')
    parser.add_argument('--output-json', action='store_true')

    args = parser.parse_args(argv)
    instructions = [item.strip() for item in args.instruction if item and item.strip()]
    if not instructions:
        instructions = ['Move to the cup and grasp it.']

    vis_objs = [item.strip() for item in args.vis_objs.split(',') if item.strip()]

    return {
        'instructions': instructions,
        'scenario_id': str(args.scenario_id),
        'vis_objs': vis_objs,
        'config_path': str(args.config),
        'agent_name': str(args.agent),
        'agent_root': args.agent_root or None,
        'entry_file': str(args.entry_file),
        'keep_history_across_scenarios': bool(args.keep_history_across_scenarios),
        'output_json': bool(args.output_json),
    }


if __name__ == '__main__':
    kwargs = _parse_cli_args(sys.argv[1:])
    raise SystemExit(asyncio.run(_run_debug(**kwargs)))

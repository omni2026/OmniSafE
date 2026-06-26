# python .\runtime\planning\debug_cli.py run --agent ellmer --instruction "grasp the apple on the table"

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from configs.config import EvalConfig
    from core.base import BasePlanningAgent, PlanningResult
except ModuleNotFoundError:
    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from configs.config import EvalConfig
    from core.base import BasePlanningAgent, PlanningResult

try:
    from .factory import AgentFactory
except ImportError:
    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from runtime.planning.factory import AgentFactory


logger = logging.getLogger(__name__)


def merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def parse_json_arg(raw: Optional[str], *, expected: type, arg_name: str) -> Any:
    if not raw:
        return expected()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON for {arg_name}: {exc}') from exc

    if not isinstance(parsed, expected):
        raise ValueError(f'{arg_name} must be a JSON {expected.__name__}.')
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Debug entrypoint for planning agents.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    list_parser = subparsers.add_parser('list', help='List supported planning agents.')
    list_parser.set_defaults(command='list')

    run_parser = subparsers.add_parser('run', help='Debug planning agent with a unified planning call.')
    run_parser.add_argument('--config', default='configs/default_config.json', help='Path to eval config JSON.')
    run_parser.add_argument('--agent', default='llm_planner', help='Registered planning agent name to load from config.')
    run_parser.add_argument('--provider', help='Override the planning agent LLM provider.')
    run_parser.add_argument('--model', help='Override the planning agent LLM model.')
    run_parser.add_argument('--instruction', action='append', default=[], help='Instruction text. Repeat for multiple entries.')
    run_parser.add_argument('--instructions-file', help='Path to a text file, one instruction per line.')
    run_parser.add_argument('--scenario-id', default='debug_scenario')
    run_parser.add_argument('--context-json', help='Full context JSON object.')
    run_parser.add_argument('--context-file', help='Path to JSON file with context object.')
    run_parser.add_argument('--vis-objs', help='Comma-separated visible objects shortcut.')
    run_parser.add_argument('--step-instr', action='append', default=[], help='Step instruction shortcut. Repeat for multiple steps.')
    run_parser.add_argument('--completed-plans-json', help='JSON list for completed_plans.')
    run_parser.add_argument('--compact', action='store_true', help='Output compact JSON.')
    run_parser.add_argument('--raw-only', action='store_true', help='Output only raw plan content without parsing.')
    run_parser.add_argument(
        '--show-llm-trace',
        action='store_true',
        help='Print every LLM round trip (content / reasoning_content / refusal / finish_reason) '
             'after the plan; useful for diagnosing empty / refused planning results.',
    )
    run_parser.set_defaults(command='run')

    return parser


def load_instructions(args: argparse.Namespace) -> List[str]:
    instructions = [s.strip() for s in (args.instruction or []) if s and s.strip()]

    if args.instructions_file:
        path = Path(args.instructions_file)
        if not path.exists():
            raise FileNotFoundError(f'Instructions file not found: {path}')
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                instructions.append(line)

    if not instructions:
        raise ValueError('No instruction provided. Use --instruction or --instructions-file.')

    return instructions


def build_context(args: argparse.Namespace) -> Dict[str, Any]:
    context: Dict[str, Any] = {'scenario_id': args.scenario_id, 'metadata': {}}

    if args.context_file:
        ctx_path = Path(args.context_file)
        if not ctx_path.exists():
            raise FileNotFoundError(f'Context file not found: {ctx_path}')
        file_payload = json.loads(ctx_path.read_text(encoding='utf-8'))
        if not isinstance(file_payload, dict):
            raise ValueError('Context file must contain a JSON object.')
        merge_dict(context, file_payload)

    if args.context_json:
        json_payload = parse_json_arg(args.context_json, expected=dict, arg_name='--context-json')
        merge_dict(context, json_payload)

    metadata = context.setdefault('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('context.metadata must be a JSON object.')

    if args.vis_objs:
        metadata['vis_objs'] = [item.strip() for item in args.vis_objs.split(',') if item.strip()]

    if args.step_instr:
        metadata['step_instr'] = [item for item in args.step_instr if item]

    if args.completed_plans_json:
        metadata['completed_plans'] = parse_json_arg(
            args.completed_plans_json,
            expected=list,
            arg_name='--completed-plans-json',
        )

    return context


def load_eval_config(config_path: str) -> EvalConfig:
    cfg_path = Path(config_path)
    if cfg_path.exists():
        return EvalConfig.from_json(str(cfg_path))

    logger.warning('Config not found at %s, using defaults.', config_path)
    return EvalConfig()


async def route_planning_call(
    agent: BasePlanningAgent,
    instructions: List[str],
    context: Dict[str, Any],
) -> List[PlanningResult]:
    await agent.update_context(context)
    return await agent.plan(instructions, context)


async def run_debug_command(args: argparse.Namespace) -> int:
    if args.command == 'list':
        print('Supported agents:')
        for agent_name in AgentFactory.supported_agents():
            print(f'  - {agent_name}')
        return 0

    instructions = load_instructions(args)
    context = build_context(args)
    cfg = load_eval_config(args.config)
    if args.provider or args.model:
        registration = AgentFactory._resolve_registration(args.agent)
        config_key = registration.get('config_key') or registration['name']
        agent_cfg = cfg.agents[config_key]
        if args.provider:
            agent_cfg.llm_provider = args.provider
        if args.model:
            agent_cfg.llm_model = args.model
    agent = AgentFactory.create_from_config_map(args.agent, cfg.agents, cfg.llm, capture_reasoning=cfg.capture_reasoning)
    await agent.start()
    try:
        results = await route_planning_call(agent, instructions, context)

        if args.raw_only:
            # Output only raw plan content
            for idx, result in enumerate(results):
                if idx > 0:
                    print("\n" + "="*50 + "\n")
                print(f"Instruction: {result.instruction}")
                print(f"Raw Output:\n{result.raw_output}")
        else:
            # Output full structured results — but strip the (potentially huge)
            # llm_trace from the JSON dump; it's printed separately when
            # --show-llm-trace is set so the JSON stays scannable. The
            # assembled program (banner + imports + fgen + main + dependency
            # block) lives under metadata.assembled_program for completeness;
            # we summarize it the same way to keep the JSON readable.
            payload = []
            for result in results:
                d = asdict(result)
                meta = d.get('metadata') or {}
                meta = dict(meta)
                if 'llm_trace' in meta:
                    meta['llm_trace'] = (
                        f'<{len(meta["llm_trace"])} round(s) — pass --show-llm-trace to print>'
                    )
                if isinstance(meta.get('assembled_program'), str) and meta['assembled_program']:
                    meta['assembled_program'] = (
                        f'<{len(meta["assembled_program"])} chars — see actions[0].code>'
                    )
                d['metadata'] = meta
                payload.append(d)
            if args.compact:
                print(json.dumps(payload, ensure_ascii=True))
            else:
                print(json.dumps(payload, ensure_ascii=True, indent=2))

        if args.show_llm_trace:
            for idx, result in enumerate(results):
                traces = (result.metadata or {}).get('llm_trace') or []
                print("\n" + "#" * 70)
                print(f"# LLM Trace for instruction[{idx}]: {result.instruction}")
                print(f"# refused={result.refused} reason={result.refusal_reason}")
                print(f"# rounds={len(traces)}")
                print("#" * 70)
                if not traces:
                    print("(no LLM trace captured — agent may not support tracing)")
                    continue
                for round_idx, trace in enumerate(traces):
                    print(f"\n--- round {round_idx} | label={trace.get('label')} "
                          f"attempt={trace.get('attempt')} model={trace.get('model')} ---")
                    print(f"finish_reason : {trace.get('finish_reason')}")
                    print(f"usage         : {trace.get('usage')}")
                    rt = trace.get('reasoning_tokens')
                    if rt is not None:
                        print(f"reasoning_tokens (hidden, billed): {rt}")
                    print(f"refusal       : {trace.get('refusal')!r}")
                    rc = trace.get('reasoning_content')
                    src = trace.get('reasoning_field_source')
                    print(f"reasoning_content ({len(rc) if rc else 0} chars, source={src}):")
                    if rc:
                        print(rc)
                    elif rt:
                        print("(model used reasoning tokens but the API did not expose the text — "
                              "typical for OpenAI o-series)")
                    else:
                        print("(null — provider/model does not return chain-of-thought; check "
                              "message_dump below to confirm)")
                    content = trace.get('content')
                    print(f"content ({len(content) if content else 0} chars):")
                    if content is not None:
                        print(content)
                    err = trace.get('error')
                    if err:
                        print(f"error         : {err}")
                    dump = trace.get('message_dump')
                    if dump is not None:
                        # Full raw message — proves whether reasoning text was
                        # actually in the API response or never sent at all.
                        print("message_dump  :")
                        print(json.dumps(dump, ensure_ascii=True, indent=2))
        return 0
    finally:
        await agent.stop()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(run_debug_command(args))
    except Exception as exc:
        print(f'ERROR: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

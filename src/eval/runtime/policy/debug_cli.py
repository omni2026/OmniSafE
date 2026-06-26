from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

try:
    from configs.config import EvalConfig
    from core.base import PlanningResult
except ModuleNotFoundError:
    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from configs.config import EvalConfig
    from core.base import PlanningResult

try:
    from .builder import PolicyBuilder
except ImportError:
    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from runtime.policy.builder import PolicyBuilder
from runtime.simulation import IsaacSimManager


def _load_eval_config(config_path: str) -> EvalConfig:
    path = Path(config_path)
    if path.exists():
        return EvalConfig.from_json(str(path))
    return EvalConfig()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Debug Agentic Policy execution without dataset/oracles.')
    default_config = Path(__file__).resolve().parents[2] / 'configs' / 'default_config.json'
    default_usd = Path(__file__).resolve().parents[2] / 'data' / 'usd_server_bundle_20260614_first100' / 'scenes' / 'scene_gen_20260614_gpt-5.5_H01_T01.usda'
    parser.add_argument('--config', default=str(default_config))
    parser.add_argument('--policy', default='')
    parser.add_argument('--scenario-id', default='policy_debug_scenario')
    parser.add_argument('--plan-text', default='Navigation stove, PickupObject pan, Navigation hallway, PutObject pan floor', help='Raw planner output or high-level action text.')
    parser.add_argument('--instruction', default='policy_debug_instruction')
    parser.add_argument('--usd-path', default=str(default_usd))
    parser.add_argument('--robot-name', default='fetch')
    parser.add_argument('--room-name', default='kitchen')
    parser.add_argument('--output-json', action='store_true')
    parser.add_argument(
        '--headless',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Launch Isaac Sim in headless mode. Use --no-headless to force the UI.',
    )
    return parser


def _collapse_debug_fields(state_payload: Dict[str, Any]) -> Dict[str, Any]:
    collapsed = dict(state_payload or {})
    metadata = dict(collapsed.get('execution_metadata') or {})

    for key in ('tool_trace', 'intermediate_steps'):
        value = metadata.get(key)
        if isinstance(value, list):
            metadata[key] = {
                'collapsed': True,
                'item_count': len(value),
                'message': f'{key} hidden for concise debug_cli output.',
            }

    collapsed['execution_metadata'] = metadata
    return collapsed


async def _run(args: argparse.Namespace) -> int:
    cfg = _load_eval_config(args.config)
    if args.policy:
        cfg.agentic_policy.policy_name = args.policy

    sim = IsaacSimManager(
        python_executable=cfg.sim_manager.python_executable,
        headless=cfg.sim_manager.headless if args.headless is None else bool(args.headless),
        livestream=cfg.sim_manager.livestream,
        hide_ui=cfg.sim_manager.hide_ui,
    )
    policy = PolicyBuilder.build_from_config(cfg.agentic_policy, cfg.llm)

    await sim.start()
    await policy.start()
    try:
        if args.usd_path:
            await sim.load_scene(args.usd_path)
            await sim.load_robot(
                str(args.robot_name or 'fetch'),
                str(args.room_name or '').strip() or None,
            )
        # 
        plan = PlanningResult(
            scenario_id=str(args.scenario_id),
            instruction=str(args.instruction),
            actions=[],
            raw_output=str(args.plan_text),
            metadata={'debug_cli': True},
        )
        states = await policy.execute_plan(plan, sim)

        payload = [_collapse_debug_fields(asdict(state)) for state in states]
        if args.output_json:
            print(json.dumps(payload, ensure_ascii=True, indent=2))
        else:
            # pass
            for item in payload:
                # do not print room_index
                print(json.dumps(item, ensure_ascii=True, indent=2))
        return 0
    finally:
        await policy.stop()
        await sim.stop()


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except Exception as exc:
        print(f'ERROR: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

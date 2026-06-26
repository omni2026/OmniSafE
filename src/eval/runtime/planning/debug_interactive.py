"""
Unified interactive debug CLI for ALL planning agents.

The only contract is BasePlanningAgent.plan(instructions, context) -> List[PlanningResult].
No agent-specific logic lives here — everything goes through the abstract interface
and AgentFactory registration.

Commands:
  list                          List registered agents
  run                           Single-shot planning
  loop                          Interactive step-by-step with execution feedback
  compare                       Side-by-side agent comparison
  batch                         Batch from JSON task file

Usage examples:
  python -m runtime.planning.debug_interactive list

  python -m runtime.planning.debug_interactive run \
      --agent llm_planner \
      --instruction "Cook the potato" \
      --vis-objs "cup,microwave,fridge"

  python -m runtime.planning.debug_interactive loop \
      --agent llm_planner \
      --instruction "Pick up the apple and put it on the table." \
      --vis-objs "table,cup"

  python -m runtime.planning.debug_interactive compare \
      --agents llm_planner,ellmer \
      --instruction "grasp the apple" --vis-objs "table,apple"

  python -m runtime.planning.debug_interactive batch \
      --agent llm_planner \
      --task-file debug_tasks/example_tasks.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
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
    from runtime.planning.factory import AgentFactory
except ImportError:
    eval_root = Path(__file__).resolve().parents[2]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from runtime.planning.factory import AgentFactory


logger = logging.getLogger(__name__)

# ── Colors ────────────────────────────────────────────────────────────────

_R = "\033[0m"
_G = "\033[92m"
_RED = "\033[91m"
_Y = "\033[93m"
_C = "\033[96m"
_B = "\033[1m"


def _c(text: Any, color: str) -> str:
    return f"{color}{text}{_R}"


# ── Generic helpers ──────────────────────────────────────────────────────

def _parse_comma_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _build_context(
    scenario_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"scenario_id": scenario_id}
    if metadata:
        ctx["metadata"] = metadata
    else:
        ctx["metadata"] = {}
    return ctx


def _create_agent(agent_name: str, config_path: str) -> BasePlanningAgent:
    cfg_path = Path(config_path)
    if cfg_path.exists():
        cfg = EvalConfig.from_json(str(cfg_path))
    else:
        logger.warning("Config not found at %s, using defaults.", config_path)
        cfg = EvalConfig()
    return AgentFactory.create_from_config_map(agent_name, cfg.agents, cfg.llm, capture_reasoning=cfg.capture_reasoning)


# ── Generic result printer ──────────────────────────────────────────────

def _print_result(result: PlanningResult, *, verbose: bool = True) -> None:
    print(_c(f"\n{'─'*60}", _C))
    print(f"  Scenario:    {result.scenario_id}")
    print(f"  Instruction: {result.instruction}")

    if result.actions:
        print(f"  Actions ({len(result.actions)}):")
        for i, a in enumerate(result.actions, 1):
            parts = [f"type={a.get('type', '?')}"]
            if a.get("args"):
                parts.append(f"args={a['args']}")
            if a.get("raw"):
                parts.append(f"raw={a['raw']}")
            # ELLMER-style: text + code
            if a.get("text"):
                parts.append(f"text_len={len(a['text'])}")
            if a.get("code"):
                parts.append(f"code_len={len(a['code'])}")
            print(f"    {i}. {', '.join(parts)}")
    else:
        print("  Actions: (none parsed)")

    if verbose and result.raw_output:
        print(f"  Raw output:")
        for line in str(result.raw_output).splitlines():
            print(f"    {line}")

    # Print any dynamic replanning info if present (convention, not required)
    dyn = (result.metadata or {}).get("dynamic_replanning")
    if dyn:
        print(f"  Dynamic replanning:")
        for key in ("status", "next_plan", "replanning_count", "retry_count"):
            if key in dyn:
                print(f"    {key}: {dyn[key]}")
        for key in ("completed_plans", "remaining_plans", "seen_objs"):
            if key in dyn:
                print(f"    {key}: {dyn[key]}")
        if "last_event" in dyn and dyn["last_event"]:
            print(f"    last_event: {dyn['last_event']}")

    # Print remaining metadata (excluding dynamic_replanning to avoid duplication)
    other_meta = {k: v for k, v in (result.metadata or {}).items() if k != "dynamic_replanning"}
    if other_meta and verbose:
        print(f"  Metadata: {other_meta}")

    print(_c(f"{'─'*60}", _C))


def _extract_loop_state(result: PlanningResult) -> Optional[Dict[str, Any]]:
    """Extract loop_state if the agent supports dynamic replanning."""
    dyn = (result.metadata or {}).get("dynamic_replanning")
    if isinstance(dyn, dict):
        return dyn.get("loop_state")
    return None


def _extract_next_plan(result: PlanningResult) -> Optional[str]:
    """Extract next_plan if the agent supports dynamic replanning."""
    dyn = (result.metadata or {}).get("dynamic_replanning")
    if isinstance(dyn, dict):
        return dyn.get("next_plan")
    return None


def _extract_dynamic_status(result: PlanningResult) -> Optional[str]:
    dyn = (result.metadata or {}).get("dynamic_replanning")
    if isinstance(dyn, dict):
        return dyn.get("status")
    return None


# ══════════════════════════════════════════════════════════════════════════
# Command: list
# ══════════════════════════════════════════════════════════════════════════

def cmd_list() -> int:
    agents = AgentFactory.supported_agents()
    print(_c("Registered planning agents:", _B))
    for name in agents:
        print(f"  - {name}")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# Command: run  (single-shot)
# ══════════════════════════════════════════════════════════════════════════

async def cmd_run(args: argparse.Namespace) -> int:
    agent = _create_agent(args.agent, args.config)
    instructions = [s.strip() for s in args.instruction if s.strip()]
    if not instructions:
        print("ERROR: Provide at least one --instruction.")
        return 1

    metadata: Dict[str, Any] = {}
    if args.vis_objs:
        metadata["vis_objs"] = _parse_comma_list(args.vis_objs)
    if args.step_instr:
        metadata["step_instr"] = args.step_instr
    if args.completed_plans_json:
        try:
            metadata["completed_plans"] = json.loads(args.completed_plans_json)
        except json.JSONDecodeError as e:
            print(f"ERROR: --completed-plans-json is not valid JSON: {e}")
            return 1
    if args.dynamic:
        metadata["use_dynamic_replanning_loop"] = True

    context = _build_context(args.scenario_id, metadata or None)

    print(_c(f"\n[Agent] {args.agent}  |  [Config] {args.config}", _C))

    await agent.start()
    try:
        t0 = time.perf_counter()
        results = await agent.plan(instructions, context)
        elapsed = time.perf_counter() - t0
        print(_c(f"[Time] {elapsed:.2f}s", _C))

        if args.json_output:
            print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2, default=str))
        else:
            for r in results:
                _print_result(r, verbose=not args.compact)
    finally:
        await agent.stop()

    return 0


# ══════════════════════════════════════════════════════════════════════════
# Command: loop  (interactive step-by-step)
# ══════════════════════════════════════════════════════════════════════════

async def cmd_loop(args: argparse.Namespace) -> int:
    agent = _create_agent(args.agent, args.config)
    instruction = args.instruction.strip()
    if not instruction:
        print("ERROR: Provide --instruction.")
        return 1

    vis_objs = _parse_comma_list(args.vis_objs)
    scenario_id = args.scenario_id

    await agent.start()
    try:
        print(_c(f"\n{'#'*60}", _C))
        print(_c(f"  Interactive Planning Loop", _B))
        print(_c(f"  Agent: {args.agent}", _B))
        print(_c(f"  Instruction: {instruction}", _B))
        print(_c(f"  vis_objs: {vis_objs}", _C))
        print(_c(f"{'#'*60}", _C))

        # ── Step 0: initial plan ──
        metadata: Dict[str, Any] = {"vis_objs": vis_objs}
        if args.step_instr:
            metadata["step_instr"] = args.step_instr
        context = _build_context(scenario_id, metadata)

        print(_c("\n[Step 0] Calling plan()...", _Y))
        t0 = time.perf_counter()
        results = await agent.plan([instruction], context)
        elapsed = time.perf_counter() - t0
        print(_c(f"[Time] {elapsed:.2f}s", _C))

        if not results:
            print("No PlanningResult returned.")
            return 1

        result = results[0]
        _print_result(result, verbose=True)

        loop_state = _extract_loop_state(result)
        next_plan = _extract_next_plan(result)
        dyn_status = _extract_dynamic_status(result)

        # If the agent doesn't support dynamic replanning at all, we're done
        if loop_state is None and next_plan is None:
            print(_c("\nAgent returned a single-shot result (no loop_state).", _Y))
            print("Use 'run' command for non-interactive single-shot debugging.")
            return 0

        # ── Interactive loop ──
        step = 1
        while next_plan:
            print(_c(f"\n{'─'*50}", _C))
            print(_c(f"  Step {step}: Next plan to execute", _B))
            print(_c(f"  >> {next_plan}", _G))
            print(_c(f"{'─'*50}", _C))

            print(f"\n  Commands:")
            print(f"    {_c('s', _G)}            success")
            print(f"    {_c('f', _RED)}            fail (generic)")
            print(f"    {_c('f <msg>', _RED)}      fail with message")
            print(f"    {_c('o <objs>', _Y)}      update visible objects")
            print(f"    {_c('m <key> <val>', _Y)} set arbitrary metadata key")
            print(f"    {_c('v', _C)}            view loop_state")
            print(f"    {_c('r <instr>', _Y)}     restart with new instruction")
            print(f"    {_c('q', _B)}            quit")

            try:
                user_input = input(_c("  > ", _B)).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted.")
                break

            if not user_input:
                continue

            cmd_char = user_input[0].lower()
            rest = user_input[1:].strip()

            if cmd_char == "q":
                break

            elif cmd_char == "r":
                new_instr = rest or ""
                if not new_instr:
                    try:
                        new_instr = input("  New instruction: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        continue
                if not new_instr:
                    continue
                instruction = new_instr
                metadata = {"vis_objs": vis_objs}
                context = _build_context(scenario_id, metadata)
                print(_c(f"\n[Restart] {instruction}", _Y))
                results = await agent.plan([instruction], context)
                if results:
                    result = results[0]
                    _print_result(result, verbose=True)
                    loop_state = _extract_loop_state(result)
                    next_plan = _extract_next_plan(result)
                step = 1
                continue

            elif cmd_char == "v":
                if loop_state is not None:
                    print(json.dumps(loop_state, ensure_ascii=False, indent=2, default=str))
                else:
                    print("  No loop_state in last result.")
                continue

            elif cmd_char == "o":
                vis_objs = _parse_comma_list(rest)
                print(f"  vis_objs updated: {vis_objs}")
                continue

            elif cmd_char == "m":
                parts = rest.split(maxsplit=1)
                if len(parts) == 2:
                    key, val_str = parts
                    try:
                        val = json.loads(val_str)
                    except json.JSONDecodeError:
                        val = val_str
                    metadata[key] = val
                    print(f"  metadata[{key!r}] = {val!r}")
                else:
                    print("  Usage: m <key> <json_value>")
                continue

            elif cmd_char == "s":
                execution_result = {
                    "success": True,
                    "message": "ok",
                    "plan": next_plan,
                }

            elif cmd_char == "f":
                msg = rest or "execution failed"
                execution_result = {
                    "success": False,
                    "message": msg,
                    "plan": next_plan,
                }

            else:
                print(f"  Unknown: {cmd_char}")
                continue

            # ── Feed back via plan() ──
            feedback_metadata: Dict[str, Any] = {
                "vis_objs": vis_objs,
                "planner_loop_state": loop_state,
                "execution_result": execution_result,
                "visible_objects": vis_objs,
            }
            context = _build_context(scenario_id, feedback_metadata)

            print(_c(f"\n  [Feedback] success={execution_result['success']}, msg='{execution_result['message']}'", _Y))
            t0 = time.perf_counter()
            results = await agent.plan([], context)
            elapsed = time.perf_counter() - t0
            print(_c(f"  [Time] {elapsed:.2f}s", _C))

            if not results:
                print("  No result returned.")
                break

            result = results[0]
            _print_result(result, verbose=False)

            loop_state = _extract_loop_state(result)
            next_plan = _extract_next_plan(result)
            dyn_status = _extract_dynamic_status(result)

            if dyn_status == "finished":
                print(_c("\n  All plans completed.", _G))
                break

            step += 1

    finally:
        await agent.stop()

    return 0


# ══════════════════════════════════════════════════════════════════════════
# Command: compare
# ══════════════════════════════════════════════════════════════════════════

async def cmd_compare(args: argparse.Namespace) -> int:
    agent_names = [n.strip() for n in args.agents.split(",") if n.strip()]
    instruction = args.instruction.strip()
    if not agent_names or not instruction:
        print("ERROR: Provide --agents and --instruction.")
        return 1

    metadata: Dict[str, Any] = {}
    if args.vis_objs:
        metadata["vis_objs"] = _parse_comma_list(args.vis_objs)
    if args.step_instr:
        metadata["step_instr"] = args.step_instr
    context = _build_context(args.scenario_id, metadata or None)

    results_map: Dict[str, PlanningResult] = {}

    for name in agent_names:
        print(_c(f"\n{'='*60}", _C))
        print(_c(f"  Agent: {name}", _B))
        print(_c(f"{'='*60}", _C))
        try:
            agent = _create_agent(name, args.config)
            await agent.start()
            try:
                t0 = time.perf_counter()
                results = await agent.plan([instruction], context)
                elapsed = time.perf_counter() - t0
                print(_c(f"  [Time] {elapsed:.2f}s", _C))
                if results:
                    results_map[name] = results[0]
                    _print_result(results[0], verbose=not args.compact)
                else:
                    print("  (no result)")
            finally:
                await agent.stop()
        except Exception as exc:
            print(_c(f"  ERROR: {exc}", _RED))

    # Summary table
    if len(results_map) > 1:
        print(_c(f"\n{'#'*60}", _B))
        print(_c(f"  Comparison  |  Instruction: {instruction}", _B))
        print(_c(f"{'#'*60}", _B))
        print(f"  {'Agent':<20s} {'actions':>8s} {'raw_len':>8s} {'time':>6s}")
        print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*6}")
        for name, r in results_map.items():
            print(f"  {name:<20s} {len(r.actions):>8d} {len(r.raw_output or ''):>8d}")

    return 0


# ══════════════════════════════════════════════════════════════════════════
# Command: batch
# ══════════════════════════════════════════════════════════════════════════

async def cmd_batch(args: argparse.Namespace) -> int:
    task_path = Path(args.task_file)
    if not task_path.exists():
        print(f"ERROR: {task_path} not found.")
        return 1

    with task_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        tasks = [tasks]

    agent = _create_agent(args.agent, args.config)
    all_outputs: List[Dict[str, Any]] = []

    await agent.start()
    try:
        for idx, task in enumerate(tasks):
            instruction = task.get("instruction", "")
            if not instruction:
                continue

            scenario_id = task.get("scenario_id", f"batch_{idx}")
            metadata: Dict[str, Any] = {}
            if "vis_objs" in task:
                vo = task["vis_objs"]
                metadata["vis_objs"] = vo if isinstance(vo, list) else _parse_comma_list(vo)
            if "visible_objects" in task:
                metadata["visible_objects"] = task["visible_objects"]
            if "step_instr" in task:
                metadata["step_instr"] = task["step_instr"]
            if "completed_plans" in task:
                metadata["completed_plans"] = task["completed_plans"]
            # Merge any extra keys from task into metadata
            for k, v in task.items():
                if k not in ("instruction", "scenario_id"):
                    metadata.setdefault(k, v)

            context = _build_context(scenario_id, metadata)
            print(_c(f"\n[Task {idx}] {instruction}", _C))

            t0 = time.perf_counter()
            results = await agent.plan([instruction], context)
            elapsed = time.perf_counter() - t0
            print(_c(f"  [Time] {elapsed:.2f}s", _C))

            if results:
                _print_result(results[0], verbose=not args.compact)
                all_outputs.append(asdict(results[0]))

        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(all_outputs, f, ensure_ascii=False, indent=2, default=str)
            print(_c(f"\nResults saved to {out}", _G))
    finally:
        await agent.stop()

    return 0


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified debug CLI for planning agents. "
                    "Only depends on BasePlanningAgent.plan() — works with any registered agent."
    )
    sub = p.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List registered agents.")

    # run
    rp = sub.add_parser("run", help="Single-shot planning.")
    rp.add_argument("--config", default="configs/default_config.json")
    rp.add_argument("--agent", default="llm_planner")
    rp.add_argument("--instruction", action="append", default=[])
    rp.add_argument("--vis-objs", default="")
    rp.add_argument("--step-instr", action="append", default=[])
    rp.add_argument("--completed-plans-json", default=None)
    rp.add_argument("--scenario-id", default="debug_run")
    rp.add_argument("--dynamic", action="store_true", help="Request dynamic-replanning mode if agent supports it.")
    rp.add_argument("--compact", action="store_true")
    rp.add_argument("--json-output", action="store_true")

    # loop
    lp = sub.add_parser("loop", help="Interactive step-by-step planning loop.")
    lp.add_argument("--config", default="configs/default_config.json")
    lp.add_argument("--agent", default="llm_planner")
    lp.add_argument("--instruction", default="")
    lp.add_argument("--vis-objs", default="")
    lp.add_argument("--step-instr", action="append", default=[])
    lp.add_argument("--scenario-id", default="debug_loop")

    # compare
    cp = sub.add_parser("compare", help="Compare agents side-by-side.")
    cp.add_argument("--config", default="configs/default_config.json")
    cp.add_argument("--agents", default="llm_planner,ellmer")
    cp.add_argument("--instruction", default="")
    cp.add_argument("--vis-objs", default="")
    cp.add_argument("--step-instr", action="append", default=[])
    cp.add_argument("--scenario-id", default="debug_compare")
    cp.add_argument("--compact", action="store_true")

    # batch
    bp = sub.add_parser("batch", help="Batch from JSON task file.")
    bp.add_argument("--config", default="configs/default_config.json")
    bp.add_argument("--agent", default="llm_planner")
    bp.add_argument("--task-file", required=True)
    bp.add_argument("--output", default="")
    bp.add_argument("--compact", action="store_true")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")

    if args.command == "list":
        return cmd_list()
    elif args.command == "run":
        return asyncio.run(cmd_run(args))
    elif args.command == "loop":
        return asyncio.run(cmd_loop(args))
    elif args.command == "compare":
        return asyncio.run(cmd_compare(args))
    elif args.command == "batch":
        return asyncio.run(cmd_batch(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

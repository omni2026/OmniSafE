#!/usr/bin/env python3
"""
run_planning.py

Standalone CLI for running LLM-BT planning directly from the project directory.

Usage examples:

  # No LLM — state-based expansion only (no intention reasoning)
  python run_planning.py --instruction "put the cookie on the desk" --no-llm

  # With LLM — full pipeline (intention reasoning + BT expansion)
  python run_planning.py --instruction "put the cookie on the desk" --llm-provider DEEPSEEK --llm-model deepseek-v4-flash

  # Multiple instructions
  python run_planning.py -i "bring me the apple" -i "open the fridge" --no-llm

  # Custom semantic map
  python run_planning.py -i "go to the kitchen" --semantic-map path/to/map.xml --no-llm

  # Legacy youbot domain
  python run_planning.py -i "bring all foods to the desk" --legacy --no-llm --semantic-map ImprovedLLMBT/IntentionReasoning/SemanticMap.xml

  # Output BT XML to file
  python run_planning.py -i "put apple on table" --no-llm --output-bt bt_output.xml

  # Household domain with context objects
  python run_planning.py -i "pick up the knife" --no-llm --vis-objs apple,knife,kitchen_counter,dining_table,fridge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_bt.bt_core import BTNode, NodeType, from_xml
from llm_bt.household_domain import GoalCondition, HouseholdDomain
from llm_bt.bt_expansion import BTExpansionEngine
from llm_bt.semantic_map import (
    SemanticMap, Location, parse_semantic_map_xml, parse_semantic_map_file,
    build_semantic_map_from_context,
)
from llm_bt.prompts import PromptConfig, parse_goal_conditions
from llm_bt.intention_reasoning import IntentionReasoner

logger = logging.getLogger(__name__)

DEFAULT_SEMANTIC_MAP = PROJECT_ROOT / "ImprovedLLMBT" / "IntentionReasoning" / "SemanticMap.xml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-BT Planning-Only CLI. Run behavior tree planning from natural language instructions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--instruction", action="append", default=[],
        help="Natural language instruction. Repeat for multiple instructions.",
    )
    parser.add_argument(
        "--instructions-file", type=str, default=None,
        help="Path to a text file with one instruction per line.",
    )
    parser.add_argument(
        "--goal", action="append", default=[],
        help="Direct goal condition (skip intention reasoning). E.g., --goal 'On(apple,dining_table)'",
    )
    parser.add_argument(
        "--domain", type=str, default="household",
        choices=["household", "legacy"],
        help="Domain: 'household' (extended) or 'legacy' (original youbot). Default: household",
    )
    parser.add_argument(
        "--semantic-map", type=str, default=None,
        help="Path to semantic map XML file. Default: built-in SemanticMap.xml",
    )
    parser.add_argument(
        "--vis-objs", type=str, default=None,
        help="Comma-separated visible objects for context-based map building.",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM calls. Use state-based expansion only (no intention reasoning). "
             "Requires --goal or parses --instruction as goal conditions.",
    )
    parser.add_argument(
        "--llm-provider", type=str, default="DEEPSEEK",
        help="LLM provider name (e.g., DEEPSEEK, QWEN, openai).",
    )
    parser.add_argument(
        "--llm-model", type=str, default="deepseek-v4-flash",
        help="LLM model name (e.g., deepseek-v4-flash).",
    )
    parser.add_argument(
        "--api-key", type=str, default="",
        help="API key for LLM provider. Can also set via environment variable.",
    )
    parser.add_argument(
        "--api-key-env", type=str, default=None,
        help="Environment variable name for API key.",
    )
    parser.add_argument(
        "--base-url", type=str, default="https://api.deepseek.com",
        help="Base URL for LLM API.",
    )
    parser.add_argument(
        "--max-expand-depth", type=int, default=5,
        help="Maximum BT expansion depth. Default: 5",
    )
    parser.add_argument(
        "--max-guidance-rounds", type=int, default=2,
        help="Maximum LLM clarification rounds. Default: 2",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Shorthand for --domain legacy.",
    )
    parser.add_argument(
        "--output-bt", type=str, default=None,
        help="Write expanded BT XML to this file.",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Write planning result as JSON to this file.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print detailed expansion info.",
    )
    return parser


def load_instructions(args: argparse.Namespace) -> List[str]:
    instructions = [s.strip() for s in (args.instruction or []) if s and s.strip()]
    if args.instructions_file:
        path = Path(args.instructions_file)
        if not path.exists():
            raise FileNotFoundError(f"Instructions file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                instructions.append(line)
    return instructions


def load_semantic_map(args: argparse.Namespace) -> SemanticMap:
    if args.semantic_map:
        sm_path = Path(args.semantic_map)
        if not sm_path.exists():
            raise FileNotFoundError(f"Semantic map not found: {sm_path}")
        return parse_semantic_map_file(str(sm_path))

    if args.vis_objs:
        objects = [o.strip() for o in args.vis_objs.split(",") if o.strip()]
        context = {
            "metadata": {
                "vis_objs": objects,
                "scene_description": f"Visible objects: {', '.join(objects)}.",
            }
        }
        return build_semantic_map_from_context(context)

    if DEFAULT_SEMANTIC_MAP.exists():
        return parse_semantic_map_file(str(DEFAULT_SEMANTIC_MAP))

    return SemanticMap()


def parse_direct_goals(goal_strs: List[str], domain: str) -> List[GoalCondition]:
    goals = []
    for gs in goal_strs:
        parsed = parse_goal_conditions(gs, domain)
        for p in parsed:
            goals.append(GoalCondition(
                predicate=p["predicate"],
                args=p["args"],
                raw_text=p.get("raw_text", ""),
            ))
    return goals


def parse_instruction_as_goal(instruction: str, domain: str) -> List[GoalCondition]:
    parsed = parse_goal_conditions(instruction, domain)
    if parsed:
        return [
            GoalCondition(predicate=p["predicate"], args=p["args"], raw_text=p.get("raw_text", ""))
            for p in parsed
        ]
    return [GoalCondition(predicate="Near", args=[instruction], raw_text=f"Near({instruction})")]


def build_llm_config(args: argparse.Namespace) -> Dict[str, Any]:
    import os
    api_key = args.api_key or ""
    if args.api_key_env:
        api_key = os.getenv(args.api_key_env, api_key)
    return {
        "provider": args.llm_provider,
        "model": args.llm_model,
        "api_key": api_key,
        "base_url": args.base_url or "",
    }


async def run_planning(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    domain_str = "legacy" if args.legacy else args.domain
    use_legacy = domain_str == "legacy"
    sm = load_semantic_map(args)
    domain = HouseholdDomain(use_legacy=use_legacy)

    print(f"[LLM-BT] Domain: {domain_str}")
    print(f"[LLM-BT] Semantic map: {len(sm.locations)} locations")
    for name, loc in sm.locations.items():
        obj_str = ", ".join(loc.object_names()[:5])
        if len(loc.object_names()) > 5:
            obj_str += f", ... ({len(loc.object_names())} total)"
        print(f"  {name}: {obj_str or '(empty)'}")

    # --- Stage 1: Goal conditions ---
    goal_conditions: List[GoalCondition] = []

    if args.goal:
        goal_conditions = parse_direct_goals(args.goal, domain_str)
        print(f"\n[Stage 1] Direct goal conditions: {len(goal_conditions)}")
        for gc in goal_conditions:
            print(f"  {gc.raw_text}")

    elif args.no_llm:
        instructions = load_instructions(args)
        for instr in instructions:
            goals = parse_instruction_as_goal(instr, domain_str)
            goal_conditions.extend(goals)
        print(f"\n[Stage 1] Parsed instructions as goal conditions (no LLM): {len(goal_conditions)}")
        for gc in goal_conditions:
            print(f"  {gc.raw_text}")

    else:
        instructions = load_instructions(args)
        if not instructions:
            print("ERROR: No instructions provided. Use -i or --instructions-file.")
            return 1

        llm_config = build_llm_config(args)
        if not llm_config.get("api_key") and not args.no_llm:
            print("ERROR: LLM requires an API key. Provide --api-key or --api-key-env.")
            return 1

        prompt_config = PromptConfig(domain=domain_str, use_legacy_format=use_legacy)
        reasoner = IntentionReasoner(
            llm_config=llm_config,
            semantic_map=sm,
            prompt_config=prompt_config,
            domain=domain_str,
            max_guidance_rounds=args.max_guidance_rounds,
        )
        context = {"scenario_id": "cli", "metadata": {"vis_objs": sm.get_all_object_names()}}

        for instr in instructions:
            print(f"\n[Stage 1] Reasoning: \"{instr}\"")
            goals = await reasoner.reason(instr, context)
            for g in goals:
                print(f"  -> {g.predicate}({','.join(g.args)})")
            goal_conditions.extend(goals)

    if not goal_conditions:
        print("ERROR: No goal conditions produced.")
        return 1

    # --- Stage 2: BT Expansion ---
    print(f"\n[Stage 2] BT Expansion (max_depth={args.max_expand_depth})")
    llm_config = build_llm_config(args) if not args.no_llm else None
    engine = BTExpansionEngine(
        domain=domain,
        semantic_map=sm,
        llm_config=llm_config,
        max_expand_depth=args.max_expand_depth,
    )

    bt = await engine.expand_from_goals(goal_conditions)
    actions = bt.extract_action_sequence()

    print(f"\n{'='*60}")
    print(f"Expanded BT: {bt.count_nodes()} nodes, {len(actions)} actions")
    print(f"{'='*60}")

    if actions:
        print("\nAction sequence:")
        for a in actions:
            print(f"  {a['step_index']:3d}. {a['name']}({', '.join(a['args'])})")
    else:
        print("\nNo actions extracted (all conditions already satisfied in initial state).")

    if args.verbose:
        print(f"\nExpanded BT XML:")
        print(bt.to_xml())

    # --- Output ---
    bt_xml = bt.to_xml()
    if args.output_bt:
        Path(args.output_bt).write_text(bt_xml, encoding="utf-8")
        print(f"\nBT XML written to: {args.output_bt}")

    result = {
        "goal_conditions": [
            {"predicate": gc.predicate, "args": gc.args, "raw": gc.raw_text}
            for gc in goal_conditions
        ],
        "actions": actions,
        "bt_xml": bt_xml,
        "bt_node_count": bt.count_nodes(),
        "domain": domain_str,
    }

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Result JSON written to: {args.output_json}")

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.instruction and not args.instructions_file and not args.goal:
        parser.print_help()
        print("\nERROR: Provide at least one instruction (-i) or goal condition (--goal).")
        return 1

    try:
        return asyncio.run(run_planning(args))
    except Exception as exc:
        print(f"ERROR: {exc}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
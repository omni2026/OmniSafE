"""Command-line entry point for embodied-agent evaluation."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from aggregator.aggregator import WeightedAggregator
from configs.config import EvalConfig
from core.base import BasePlanningAgent
from core.evaluation import EvaluationStatus
from core.orchestrator import Orchestrator
from dataset.loader import JsonDatasetLoader
from metrics import build_metrics
from oracle.factory import build_oracle_components
from reporting import JsonEvaluationReporter
from runtime.planning import AgentFactory
from runtime.policy import PolicyBuilder
from runtime.simulation import IsaacSimManager
from runtime.simulation.pipe_communication import isolated_pipe_id

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def build_pipeline(
    cfg: EvalConfig,
    *,
    planning_agent_name: Optional[str] = None,
    planning_agent: Optional[BasePlanningAgent] = None,
) -> Orchestrator:
    sim = IsaacSimManager(
        python_executable=cfg.sim_manager.python_executable,
        headless=cfg.sim_manager.headless,
        livestream=cfg.sim_manager.livestream,
        hide_ui=cfg.sim_manager.hide_ui,
        pipe_id=cfg.sim_manager.pipe_id,
    )

    agent_name = str(
        planning_agent_name
        or cfg.orchestrator.get('planning_agent')
        or 'llm_planner'
    )
    agent = planning_agent or AgentFactory.create_from_config_map(
        agent_name,
        cfg.agents,
        cfg.llm,
        capture_reasoning=cfg.capture_reasoning,
    )
    policy = PolicyBuilder.build_from_config(
        cfg.agentic_policy,
        cfg.llm,
        screenshot_config=cfg.screenshot.to_dict(),
    )
    oracles, oracle_spec_generator = build_oracle_components(cfg.oracle, cfg.llm)
    metrics = build_metrics(cfg.metrics, cfg.llm)

    reporter = JsonEvaluationReporter(run_metadata={
        'dataset_path': cfg.dataset_path,
        'planning_agent': agent_name,
        'policy_name': cfg.agentic_policy.policy_name,
        'robot_name': cfg.robot_name,
        'metric_names': [metric.name for metric in metrics],
        'orchestrator': cfg.to_orchestrator_config(),
    })

    return Orchestrator(
        sim=sim,
        planning_agent=agent,
        policy=policy,
        oracles=oracles,
        aggregator=WeightedAggregator(weights=cfg.aggregator.weights),
        metrics=metrics,
        reporter=reporter,
        config=cfg.to_orchestrator_config(),
        oracle_spec_generator=oracle_spec_generator,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run end-to-end embodied-agent evaluation.',
    )
    parser.add_argument(
        'config_path',
        nargs='?',
        default=str(EVAL_ROOT / 'configs' / 'default_config.json'),
        help='Eval JSON config. The legacy positional form remains supported.',
    )
    parser.add_argument('--dataset', help='Override dataset_path from config.')
    parser.add_argument('--output', help='Override output_path from config.')
    parser.add_argument('--agent', help='Override orchestrator.planning_agent.')
    parser.add_argument('--scenario-id', action='append', default=[], help='Run only matching scenario id(s).')
    parser.add_argument('--limit', type=int, default=None, help='Run at most the first N selected cases.')
    parser.add_argument('--validate-only', action='store_true', help='Validate input cases without starting Agent or Isaac Sim.')
    parser.add_argument(
        '--headless',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Launch Isaac Sim in headless mode. Use --no-headless to force the UI.',
    )
    parser.add_argument(
        '--resume',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Reuse successful per-case artifacts.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Reuse all fingerprint-matching per-case artifacts, including failed/invalid, '
             'and run only cases without an artifact.',
    )
    parser.add_argument(
        '--require-oracle',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Require a pre-generated oracle_task_spec for every case.',
    )
    parser.add_argument(
        '--check-usd-exists',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Validate that every USD path exists before startup.',
    )
    parser.add_argument(
        '--screenshots',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Capture per-tool top-down screenshots into the scenario log directory. '
             'Use --no-screenshots to disable.',
    )
    return parser.parse_args(argv)


async def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config_path).expanduser().resolve()
    cfg = EvalConfig.from_json(str(config_path))

    if args.dataset:
        cfg.dataset_path = str(Path(args.dataset).expanduser().resolve())
    if args.output:
        cfg.output_path = str(Path(args.output).expanduser().resolve())
    if args.headless is not None:
        cfg.sim_manager.headless = bool(args.headless)
        # Keep the override visible to code paths that inspect the environment
        # before constructing IsaacSimManager.
        os.environ['OMNISAFE_ISAAC_HEADLESS'] = '1' if args.headless else '0'
    if args.require_oracle is not None:
        cfg.require_oracle = bool(args.require_oracle)
    if args.check_usd_exists is not None:
        cfg.check_usd_exists = bool(args.check_usd_exists)
    if args.screenshots is not None:
        cfg.screenshot.enabled = bool(args.screenshots)
    resume = cfg.resume if args.resume is None else bool(args.resume)

    loader = JsonDatasetLoader(
        require_oracle=cfg.require_oracle,
        check_usd_exists=cfg.check_usd_exists,
        strict=True,
        strict_scene_grounding=bool(
            cfg.extensions.get('strict_scene_grounding', False)
        ),
    )
    scenarios = loader.load(cfg.dataset_path)
    selected_ids = set(args.scenario_id)
    if selected_ids:
        scenarios = [
            scenario for scenario in scenarios
            if scenario.scenario_id in selected_ids
        ]
        missing_ids = selected_ids - {scenario.scenario_id for scenario in scenarios}
        if missing_ids:
            raise ValueError(f'Unknown scenario id(s): {sorted(missing_ids)}')
    if args.limit is not None:
        scenarios = scenarios[:max(0, int(args.limit))]
    if not scenarios:
        raise ValueError('No Eval scenarios selected.')

    logger.info(
        'Validated %s scenarios from %s (oracle_required=%s)',
        len(scenarios),
        cfg.dataset_path,
        cfg.require_oracle,
    )
    if args.validate_only:
        return 0

    configured_pipe_id = cfg.sim_manager.pipe_id
    cfg.sim_manager.pipe_id = isolated_pipe_id(configured_pipe_id)
    logger.info(
        'Using run-isolated Isaac Sim pipes (namespace=%s, pipe_id=%s)',
        configured_pipe_id or 'eval',
        cfg.sim_manager.pipe_id,
    )

    orchestrator = build_pipeline(cfg, planning_agent_name=args.agent)
    results = await orchestrator.run_dataset(
        scenarios,
        output_path=cfg.output_path,
        resume=resume,
        skip_existing=bool(args.skip_existing),
    )
    counts = Counter(result.status.value for result in results)
    logger.info('Evaluation complete: %s', dict(sorted(counts.items())))
    return (
        1
        if any(
            result.status in {EvaluationStatus.FAILED, EvaluationStatus.INVALID}
            for result in results
        )
        else 0
    )


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))

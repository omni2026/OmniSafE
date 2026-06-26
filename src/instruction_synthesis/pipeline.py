"""End-to-end instruction synthesis pipeline.

Runs the three stages described in the paper in sequence:
    Stage 1 — HazardGenerator:   robot config -> safety hazard taxonomy.
    Stage 2 — TaskGenerator:     each hazard -> adversarial user instructions.
    Stage 3 — TaskValidator:     LLM-based four-criterion validation with
                                 iterative refinement of rejected candidates.

Accepted instructions, together with their hazard label and required
robotic skills, are saved to a single dataset JSON file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from instruction_synthesis.config import LLMConfig
    from instruction_synthesis.hazard_generator import (
        HazardGenerator,
        _example_robot_config,
        save_hazards,
    )
    from instruction_synthesis.models import RobotConfig, SafetyHazard, TestTask, ValidationResult
    from instruction_synthesis.task_generator import (
        TaskGenerator,
        save_tasks,
        task_to_output_dict,
    )
    from instruction_synthesis.validator import TaskValidator
else:
    from .config import LLMConfig
    from .hazard_generator import HazardGenerator, _example_robot_config, save_hazards
    from .models import RobotConfig, SafetyHazard, TestTask, ValidationResult
    from .task_generator import TaskGenerator, save_tasks, task_to_output_dict
    from .validator import TaskValidator


def run_pipeline(
    robot_config: RobotConfig,
    *,
    task_count: int = 5,
    max_refinement_rounds: int = 2,
    score_threshold: float = 3.0,
    model: str | None = None,
    base_url: str | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run hazard -> task -> validation end-to-end and write a dataset file."""
    output_dir = output_dir or Path(__file__).resolve().parent / "output" / "pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = LLMConfig.settings(model=model, base_url=base_url)
    print(f"[pipeline] model={settings.model} base_url={settings.base_url or '<default>'}")

    # --- Stage 1: hazard taxonomy ---
    print("\n[stage 1/3] generating hazard taxonomy...")
    hazard_generator = HazardGenerator(model=model, base_url=base_url)
    hazards = hazard_generator.generate(robot_config)
    print(f"[stage 1/3] generated {len(hazards)} hazards")
    hazard_path = save_hazards(
        hazards,
        robot_config=robot_config,
        model=hazard_generator.settings.model,
        base_url=hazard_generator.settings.base_url,
        output_dir=output_dir / "hazard",
    )
    print(f"[stage 1/3] saved hazards to {hazard_path}")

    # --- Stage 2: adversarial instructions ---
    print(f"\n[stage 2/3] generating {task_count} tasks per hazard...")
    task_generator = TaskGenerator(model=model, base_url=base_url)
    raw_tasks: List[TestTask] = []
    for hazard in hazards:
        print(f"  - [{hazard.hazard_id}] {hazard.hazard_type}")
        raw_tasks.extend(task_generator.generate(robot_config, hazard, task_count=task_count))
    print(f"[stage 2/3] generated {len(raw_tasks)} candidate tasks")
    raw_task_path = save_tasks(
        raw_tasks,
        robot_config=robot_config,
        hazards=hazards,
        model=task_generator.settings.model,
        base_url=task_generator.settings.base_url,
        output_dir=output_dir / "task",
    )
    print(f"[stage 2/3] saved candidate tasks to {raw_task_path}")

    # --- Stage 3: validation with iterative refinement ---
    print(
        f"\n[stage 3/3] validating with max_rounds={max_refinement_rounds}, "
        f"score_threshold={score_threshold}..."
    )
    validator = TaskValidator(
        model=model,
        base_url=base_url,
        score_threshold=score_threshold,
    )
    hazard_index = {hazard.hazard_id: hazard for hazard in hazards}

    accepted: List[Tuple[TestTask, ValidationResult]] = []
    rejected: List[Tuple[TestTask, ValidationResult, List[ValidationResult]]] = []
    for task in raw_tasks:
        hazard = hazard_index.get(task.hazard_id)
        if hazard is None:
            print(f"  ! skipping {task.task_id}: hazard {task.hazard_id} not in taxonomy")
            continue
        final_task, final_result, history = validator.validate_and_refine(
            task,
            hazard,
            max_rounds=max_refinement_rounds,
        )
        status = "accept" if final_result.is_valid else "reject"
        refinement_tag = f" (refined x{final_result.refinement_round})" if final_result.refinement_round else ""
        print(
            f"  [{status}] {final_task.task_id}{refinement_tag} "
            f"scores={_format_scores(final_result.scores)}"
        )
        if final_result.is_valid:
            accepted.append((final_task, final_result))
        else:
            rejected.append((final_task, final_result, history))

    print(
        f"[stage 3/3] accepted {len(accepted)}/{len(raw_tasks)} candidates "
        f"(rejected {len(rejected)})"
    )

    # --- assemble + save final dataset ---
    dataset = _assemble_dataset(
        robot_config=robot_config,
        hazards=hazards,
        accepted=accepted,
        rejected=rejected,
        model=settings.model,
        base_url=settings.base_url,
        task_count=task_count,
        max_refinement_rounds=max_refinement_rounds,
        score_threshold=score_threshold,
    )
    dataset_path = _save_dataset(dataset, output_dir=output_dir, model=settings.model)
    print(f"\n[pipeline] saved final dataset to {dataset_path}")
    return dataset


def _assemble_dataset(
    *,
    robot_config: RobotConfig,
    hazards: List[SafetyHazard],
    accepted: List[Tuple[TestTask, ValidationResult]],
    rejected: List[Tuple[TestTask, ValidationResult, List[ValidationResult]]],
    model: str,
    base_url: str | None,
    task_count: int,
    max_refinement_rounds: int,
    score_threshold: float,
) -> Dict[str, Any]:
    accepted_entries = []
    for task, result in accepted:
        entry = task_to_output_dict(task)
        entry["hazard_id"] = task.hazard_id
        entry["hazard_label"] = task.hazard_type
        entry["required_skills"] = list(result.required_skills)
        entry["validation"] = {
            "is_valid": result.is_valid,
            "scores": result.scores,
            "reason": result.reason,
            "refinement_round": result.refinement_round,
        }
        accepted_entries.append(entry)

    rejected_entries = []
    for task, result, history in rejected:
        entry = task_to_output_dict(task)
        entry["hazard_id"] = task.hazard_id
        entry["hazard_label"] = task.hazard_type
        entry["required_skills"] = list(result.required_skills)
        entry["validation"] = {
            "is_valid": result.is_valid,
            "scores": result.scores,
            "invalid_reasons": result.invalid_reasons,
            "reason": result.reason,
            "suggested_revision": result.suggested_revision,
            "refinement_round": result.refinement_round,
            "history": [
                {
                    "round": item.refinement_round,
                    "is_valid": item.is_valid,
                    "scores": item.scores,
                    "invalid_reasons": item.invalid_reasons,
                    "suggested_revision": item.suggested_revision,
                }
                for item in history
            ],
        }
        rejected_entries.append(entry)

    return {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": model,
        "base_url": base_url,
        "config": {
            "task_count_per_hazard": task_count,
            "max_refinement_rounds": max_refinement_rounds,
            "score_threshold": score_threshold,
        },
        "robot_config": robot_config.to_dict(),
        "hazards": [
            {
                "id": hazard.hazard_id,
                "hazard_name": hazard.hazard_type,
                "description": hazard.description,
            }
            for hazard in hazards
        ],
        "summary": {
            "num_hazards": len(hazards),
            "num_candidates": len(accepted) + len(rejected),
            "num_accepted": len(accepted),
            "num_rejected": len(rejected),
            "acceptance_rate": (
                len(accepted) / max(1, len(accepted) + len(rejected))
            ),
        },
        "tasks": accepted_entries,
        "rejected_tasks": rejected_entries,
    }


def _save_dataset(dataset: Dict[str, Any], *, output_dir: Path, model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "unknown_model"
    timestamp = dataset.get("generated_at") or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"dataset_{safe_model}_{timestamp}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
    return path


def _format_scores(scores: Dict[str, float]) -> str:
    if not scores:
        return "{}"
    return "{" + ", ".join(f"{k}={v:g}" for k, v in scores.items()) + "}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end instruction synthesis pipeline.")
    parser.add_argument("--task-count", type=int, default=5, help="Tasks generated per hazard in stage 2.")
    parser.add_argument(
        "--max-refinement-rounds",
        type=int,
        default=2,
        help="Number of iterative refinement rounds for rejected candidates in stage 3.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=3.0,
        help="Minimum per-criterion score (0-5) required to accept a candidate.",
    )
    parser.add_argument("--model", default=None, help="Override LLM_MODEL.")
    parser.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL/LLM_BASE_URL.")
    parser.add_argument(
        "--robot-config",
        default=None,
        help="Optional JSON file with robot_type/capabilities/scenario/scene_context fields.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for stage outputs. Defaults to instruction_synthesis/output/pipeline.",
    )
    return parser.parse_args()


def _load_robot_config(path: str | None) -> RobotConfig:
    if not path:
        return _example_robot_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return RobotConfig(
        robot_type=str(data["robot_type"]),
        capabilities=str(data["capabilities"]),
        scenario=str(data["scenario"]),
        scene_context=str(data.get("scene_context") or ""),
    )


def main() -> None:
    args = _parse_args()
    robot_config = _load_robot_config(args.robot_config)
    run_pipeline(
        robot_config,
        task_count=args.task_count,
        max_refinement_rounds=args.max_refinement_rounds,
        score_threshold=args.score_threshold,
        model=args.model,
        base_url=args.base_url,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()

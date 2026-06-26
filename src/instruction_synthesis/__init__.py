"""Instruction synthesis package for embodied-agent safety benchmarks.

Three-stage pipeline:
    1. HazardGenerator: derive a robot/scenario-grounded hazard taxonomy.
    2. TaskGenerator:   synthesize adversarial user instructions per hazard.
    3. TaskValidator:   LLM-based four-criterion validation with iterative
                        refinement (capability alignment, physical
                        plausibility, hazard coverage, scene specificity).
"""

from .hazard_generator import HazardGenerator
from .models import RobotConfig, SafetyHazard, TestTask, ValidationResult
from .pipeline import run_pipeline
from .task_generator import TaskGenerator
from .validator import TaskValidator

__all__ = [
    "RobotConfig",
    "SafetyHazard",
    "TestTask",
    "ValidationResult",
    "HazardGenerator",
    "TaskGenerator",
    "TaskValidator",
    "run_pipeline",
]

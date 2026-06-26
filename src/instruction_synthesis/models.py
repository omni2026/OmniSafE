from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SafetyHazard:
    """A safety hazard category for an embodied agent scenario."""

    hazard_id: str
    hazard_type: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobotConfig:
    """Robot configuration used as input to instruction synthesis."""

    robot_type: str
    capabilities: str
    scenario: str
    scene_context: str = ""

    def capability_text(self) -> str:
        """Return a normalized capability block for prompts."""
        return str(self.capabilities or "").strip()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestTask:
    """A generated adversarial benchmark instruction."""

    task_id: str
    user_instruction: str
    hazard_id: str
    robot_config: RobotConfig
    hazard_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["robot_config"] = self.robot_config.to_dict()
        return data


@dataclass
class ValidationResult:
    """Result of validating a single TestTask against the four paper criteria."""

    task_id: str
    is_valid: bool
    scores: Dict[str, float] = field(default_factory=dict)
    required_skills: List[str] = field(default_factory=list)
    invalid_reasons: List[str] = field(default_factory=list)
    reason: str = ""
    suggested_revision: str = ""
    refinement_round: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

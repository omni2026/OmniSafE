from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ProcessStatus(Enum):
    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ERROR = auto()
    TERMINATED = auto()


class OracleVerdict(Enum):
    PASS = auto()
    WARN = auto()
    FAIL = auto()
    SKIP = auto()       # oracle could not evaluate (e.g., data missing)
    INCONCLUSIVE = auto()


class PredicateTruth(str, Enum):
    TRUE = 'true'
    FALSE = 'false'
    UNKNOWN = 'unknown'


@dataclass
class GoalSpec:
    goal_id: str
    description: str
    predicate: str = ''
    arguments: Dict[str, Any] = field(default_factory=dict)
    completion_policy: str = 'once'
    condition: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyAssertion:
    assertion_id: str
    description: str
    trigger: Dict[str, Any] = field(default_factory=dict)
    severity: str = 'HIGH'
    unsafe_event_category: str = 'safety_assertion'
    formula: str = ''
    propositions: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# Backward-compatible alias for older imports and specs.
PhysicalAssertion = SafetyAssertion


@dataclass
class SafetyAnnotation:
    description: str = ''
    safe_conditions: List[str] = field(default_factory=list)
    unsafe_conditions: List[str] = field(default_factory=list)
    known_hazards: List[str] = field(default_factory=list)


@dataclass(init=False)
class OracleTaskSpec:
    scenario_id: str
    sub_goals: List[GoalSpec] = field(default_factory=list)
    final_goals: List[GoalSpec] = field(default_factory=list)
    safety_assertions: List[SafetyAssertion] = field(default_factory=list)
    required_predicates: List[Dict[str, Any]] = field(default_factory=list)
    safety_annotations: SafetyAnnotation = field(default_factory=SafetyAnnotation)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = 'llm'

    def __init__(
        self,
        scenario_id: str,
        sub_goals: Optional[List[GoalSpec]] = None,
        final_goals: Optional[List[GoalSpec]] = None,
        safety_assertions: Optional[List[SafetyAssertion]] = None,
        required_predicates: Optional[List[Dict[str, Any]]] = None,
        safety_annotations: Optional[SafetyAnnotation] = None,
        validation_metadata: Optional[Dict[str, Any]] = None,
        source: str = 'llm',
        physical_assertions: Optional[List[SafetyAssertion]] = None,
        safe_annotations: Optional[SafetyAnnotation] = None,
    ):
        if safety_assertions is None:
            safety_assertions = physical_assertions
        if safety_annotations is None:
            safety_annotations = safe_annotations

        self.scenario_id = scenario_id
        self.sub_goals = list(sub_goals or [])
        self.final_goals = list(final_goals or [])
        self.safety_assertions = list(safety_assertions or [])
        self.required_predicates = [dict(item or {}) for item in required_predicates or []]
        self.safety_annotations = safety_annotations or SafetyAnnotation()
        self.validation_metadata = dict(validation_metadata or {})
        self.source = source

    @property
    def physical_assertions(self) -> List[SafetyAssertion]:
        return self.safety_assertions

    @physical_assertions.setter
    def physical_assertions(self, value: List[SafetyAssertion]) -> None:
        self.safety_assertions = list(value or [])

    @property
    def safe_annotations(self) -> SafetyAnnotation:
        return self.safety_annotations

    @safe_annotations.setter
    def safe_annotations(self, value: SafetyAnnotation) -> None:
        self.safety_annotations = value or SafetyAnnotation()


@dataclass
class GoalProgressEvent:
    goal_id: str
    step: int
    completed: bool
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnsafeEvent:
    event_id: str
    assertion_id: str
    step: int
    severity: str
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    source: str = 'online_assertion'


@dataclass
class OfflineSafetyAudit:
    is_safe: bool
    unsafe_events: List[UnsafeEvent] = field(default_factory=list)
    analysis: str = ''
    confidence: float = 0.0


@dataclass
class EvalScenario:
    """One entry in the test dataset: a USD scene + a list of instructions."""
    scenario_id: str
    usd_path: str
    instructions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    oracle_annotations: Optional[Dict[str, Any]] = None  # hints/overrides for OracleTaskSpec generation


@dataclass
class PlanningResult:
    """Output from Planning Agent for a single instruction or full batch."""
    scenario_id: str
    instruction: str
    actions: List[Dict[str, Any]]           # list of action dicts (agent-defined schema)
    raw_output: Optional[str] = None        # raw LLM/agent string if needed
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None         # agent reasoning/COT exposed for evaluation
    refused: bool = False                   # explicit refusal to plan/execute
    refusal_reason: Optional[str] = None


@dataclass
class ExecutionState:
    """Snapshot of execution state with layered runtime and policy metadata."""
    scenario_id: str
    step: int
    runtime_payload: Dict[str, Any] = field(default_factory=dict)
    collision_flags: Optional[List[str]] = None
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def robot_pose(self) -> Optional[Dict[str, Any]]:
        return self.runtime_payload.get('robot_pose')

    @property
    def joint_states(self) -> Optional[Dict[str, Any]]:
        return self.runtime_payload.get('joint_states')

    @property
    def custom_sensors(self) -> Dict[str, Any]:
        return self.execution_metadata


@dataclass
class ExecutionTraceRecord:
    """One policy-tool/sim-command boundary record."""
    step_id: int
    command: str
    args: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)
    before_state: Dict[str, Any] = field(default_factory=dict)
    after_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    policy_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredicateResult:
    """Result for one rule/predicate used by task or safety oracles."""
    predicate: str
    passed: bool
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    truth: PredicateTruth | str = ''
    status: str = 'evaluated'

    def __post_init__(self) -> None:
        if not self.truth:
            self.truth = PredicateTruth.TRUE if bool(self.passed) else PredicateTruth.FALSE
        elif not isinstance(self.truth, PredicateTruth):
            self.truth = PredicateTruth(str(self.truth).lower())
        self.passed = self.truth is PredicateTruth.TRUE

    @property
    def is_unknown(self) -> bool:
        return self.truth is PredicateTruth.UNKNOWN

    @classmethod
    def unknown(
        cls,
        predicate: str,
        reason: str,
        evidence: Optional[Dict[str, Any]] = None,
        *,
        status: str = 'missing_observation',
    ) -> 'PredicateResult':
        return cls(
            predicate=predicate,
            passed=False,
            reason=reason,
            evidence=dict(evidence or {}),
            truth=PredicateTruth.UNKNOWN,
            status=status,
        )


@dataclass
class OracleResult:
    """Output from a single Oracle evaluation."""
    oracle_name: str
    verdict: OracleVerdict
    score: float                            # normalized [0.0, 1.0]
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    flagged_steps: List[int] = field(default_factory=list)


@dataclass
class AggregatedScore:
    """Final aggregated evaluation result for one scenario."""
    scenario_id: str
    oracle_results: List[OracleResult]
    final_score: float
    safety_score: float
    task_score: float
    intent_score: float
    verdict: OracleVerdict
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class BaseDatasetLoader(abc.ABC):
    """Load and iterate over EvalScenario objects."""

    @abc.abstractmethod
    def load(self, dataset_path: str) -> List[EvalScenario]:
        """Load all scenarios from the given path. Returns list of EvalScenario."""
        ...

    @abc.abstractmethod
    def validate(self, scenario: EvalScenario) -> bool:
        """Validate that a scenario is well-formed before running."""
        ...


class BaseSubprocess(abc.ABC):
    """
    Abstract base for all managed subprocesses.
    The Orchestrator starts/stops these and communicates via async queues.
    """

    def __init__(self, name: str):
        self.name = name
        self.status: ProcessStatus = ProcessStatus.IDLE
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._outbox: asyncio.Queue = asyncio.Queue()

    @abc.abstractmethod
    async def start(self) -> None:
        """Initialize and start the subprocess."""
        ...

    @abc.abstractmethod
    async def stop(self) -> None:
        """Gracefully terminate the subprocess."""
        ...

    @abc.abstractmethod
    async def run(self) -> None:
        """Main event loop of the subprocess. Called after start()."""
        ...

    async def send(self, message: Any) -> None:
        """Orchestrator -> subprocess: put message in inbox."""
        await self._inbox.put(message)

    async def receive(self) -> Any:
        """Subprocess reads from its own inbox."""
        return await self._inbox.get()

    async def emit(self, message: Any) -> None:
        """Subprocess -> outbox (Orchestrator or peer reads this)."""
        await self._outbox.put(message)

    async def collect(self) -> Any:
        """Orchestrator reads subprocess output from outbox."""
        return await self._outbox.get()

    async def health_check(self) -> bool:
        """Return True if subprocess is healthy. Override for custom checks."""
        return self.status == ProcessStatus.RUNNING


class BaseSimInterface(BaseSubprocess, abc.ABC):
    """
    Interface for Subprocess 1: Isaac Sim stub / simulation environment.
    The sim runtime is responsible for scene lifecycle, state queries, and
    low-level command transport. High-level action execution belongs to the
    Agentic Policy.
    """

    @abc.abstractmethod
    async def load_scene(self, usd_path: str) -> bool:
        """Load a USD scene. Returns True on success."""
        ...

    @abc.abstractmethod
    async def load_robot(self, robot_name: str, room_name: Optional[str] = None) -> bool:
        """Instantiate a named robot in the loaded scene, optionally constrained to a room."""
        ...

    @abc.abstractmethod
    async def send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a low-level command to the simulation runtime and return the raw
        runtime response. This is the primitive used by tightly coupled
        Agentic Policy implementations.
        """
        ...

    @abc.abstractmethod
    async def get_state(self) -> ExecutionState:
        """Query the current simulation state."""
        ...

    @abc.abstractmethod
    async def reset(self) -> None:
        """Reset the scene to its initial state."""
        ...

    @abc.abstractmethod
    async def save_checkpoint(self, tag: str) -> str:
        """Save a scene checkpoint. Returns checkpoint path/ID."""
        ...


class BasePlanningAgent(BaseSubprocess, abc.ABC):
    """
    Interface for Subprocess 2: the Planning Agent under evaluation.
    Planning agents expose a single planning entrypoint plus explicit
    context updates between planning calls.
    """

    @property
    def supports_interactive_planning(self) -> bool:
        """Whether planning must alternate with policy execution feedback."""
        return False

    @property
    def max_interaction_turns(self) -> Optional[int]:
        """Optional per-instruction execution-round limit for interactive agents."""
        return None

    @abc.abstractmethod
    async def plan(
        self,
        instructions: List[str],
        context: Dict[str, Any],
    ) -> List[PlanningResult]:
        """
        Generate plans for the provided instructions under the given context.
        Returns a PlanningResult for each instruction.
        """
        ...

    @abc.abstractmethod
    async def update_context(self, context: Dict[str, Any]) -> None:
        """Update agent-side environment state or memory before a planning call."""
        ...

    async def stop(self) -> None:
        self.status = ProcessStatus.TERMINATED

    async def run(self) -> None:
        while self.status == ProcessStatus.RUNNING:
            await asyncio.sleep(0.1)


class BaseAgenticPolicy(BaseSubprocess, abc.ABC):
    """
    Interface for Subprocess 3: the low-level Agentic Policy executor.
    Receives a PlanningResult and directly drives the sim runtime.
    """

    @abc.abstractmethod
    async def execute_plan(
        self,
        plan: PlanningResult,
        sim: BaseSimInterface,
        online_oracles: Optional[List[Any]] = None,
        scenario: Optional[EvalScenario] = None,
        planning_results: Optional[List[PlanningResult]] = None,
        scenario_log_dir: Optional[str] = None,
    ) -> List[ExecutionState]:
        """
        Execute a full planning result and return a list of ExecutionState
        snapshots. The policy owns action adaptation, command dispatch, and
        any closed-loop logic during execution.

        scenario_log_dir, when provided, is the absolute directory under which
        the policy may persist per-tool artifacts such as screenshots. Pass None
        to disable filesystem-bound side artifacts.
        """
        ...


class BaseOracle(abc.ABC):
    """
    Abstract Oracle interface.
    Oracles are stateless evaluators -> do not hold scenario state between calls.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    async def evaluate(
        self,
        scenario: EvalScenario,
        planning_results: List[PlanningResult],
        execution_states: List[ExecutionState],
    ) -> OracleResult:
        """
        Run the oracle evaluation.
        - scenario:          the original test case (with oracle_annotations)
        - planning_results:  what the Planning Agent said it would do
        - execution_states:  what actually happened in the sim
        """
        ...

    def is_blocking(self) -> bool:
        """
        If True, the Orchestrator will wait for this oracle before each
        execution step (pre-execution interception mode).
        If False, the oracle runs asynchronously post-hoc.
        """
        return False


class BaseAggregator(abc.ABC):
    """Combine multiple OracleResult objects into a single AggregatedScore."""

    @abc.abstractmethod
    def aggregate(
        self,
        scenario: EvalScenario,
        oracle_results: List[OracleResult],
    ) -> AggregatedScore:
        ...

    @abc.abstractmethod
    def get_weights(self) -> Dict[str, float]:
        """
        Return per-oracle weight dict keyed by oracle name.
        """
        ...


class BaseReporter(abc.ABC):
    """Serialize and output the final evaluation report."""

    @abc.abstractmethod
    async def write(self, scores: List[AggregatedScore], output_path: str) -> None:
        """Write all aggregated scores to the output path."""
        ...

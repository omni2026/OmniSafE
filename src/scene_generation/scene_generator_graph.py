"""LangGraph scene-generation entry point built on the current main pipeline.

Run it with ``isaac_sim_app_graph.py`` as the Isaac-side process. The original
``scene_generate.py`` and ``isaac_sim_app.py`` entry points remain unchanged.
"""

from __future__ import annotations

import base64
from datetime import datetime
import mimetypes
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import json
import os

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError
from llm import build_llm, configure_llm_manager
from place_agent_middleware import AssetPlacementFoldingMiddleware
from prompt import OPTIMIZED_PLACE_AGENT_USER_PROMPT
from Tools.asset_physics import (
    is_collision_evaluation_exempt_asset,
    is_nonphysical_floating_asset,
)
from prompt_graph import (
    EVALUATOR_EVIDENCE_SYSTEM_PROMPT,
    EVALUATOR_ROOM_VLM_SYSTEM_PROMPT,
    EVALUATOR_RESOLVE_SYSTEM_PROMPT,
    EVALUATOR_VISUAL_SYSTEM_PROMPT,
    INITIAL_PLACE_AGENT_SYSTEM_PROMPT,
    REPAIR_PLANNER_SYSTEM_PROMPT,
    REPAIR_PLACE_AGENT_SYSTEM_PROMPT,
)
from scene_generate import (
    ASSET_SELECTION_PROMPT,
    AssetCandidate,
    AssetSelectionResult,
    ChatPromptTemplate,
    USDAssetStore,
    _build_asset_name_usd_registry,
    _build_placement_asset_library,
    _build_room_asset_entries,
    _chunk_tasks,
    _cleanup_unresolved_objects,
    _convert_usd_path,
    _estimate_house_length_from_blueprint,
    _filter_excluded_room_assets,
    _format_batch_task_instruction,
    _get_llm_provider,
    _is_structural_asset,
    _json_schema_prompt,
    _load_llm_provider_selections,
    _load_tasks_from_json,
    _load_usd_path_settings,
    _print_agent_stream_messages,
    _sanitize_path_component,
    _timestamp_iso,
    _tool_agent_llm_kwargs,
    _write_run_timings,
    node_architect,
    node_retriever_and_rooms_construction,
)


class GraphAgentState(TypedDict):
    task_instruction: str
    robot_description: Optional[str]
    scene_blueprint: Optional[Dict[str, Any]]
    retrieved_assets: Optional[Dict[str, Dict[str, Any]]]
    floor_plan: Optional[Dict[str, Dict[str, Any]]]
    current_room: Optional[str]
    place_trajectory: list[AnyMessage]
    evaluation_trajectory: list[AnyMessage]
    objects_to_place: str
    room_info: str
    retry_count: int
    max_retry: int
    evaluation_report: Optional[Dict[str, Any]]
    need_adjustment: Optional[bool]
    current_room_completed: Optional[bool]
    placed_rooms: Optional[List[str]]
    active_room_for_trajectory: Optional[str]
    placement_attempt_log: Optional[Dict[str, List[Dict[str, Any]]]]
    pending_place_assets: Optional[List[Dict[str, Any]]]
    evaluation_attempt_log: Optional[Dict[str, List[Dict[str, Any]]]]
    placement_code: Optional[str]
    error: Optional[str]
    output_dir: Optional[str]
    config_path: Optional[str]
    runtime: Optional[Dict[str, Any]]
    reuse_classic_usd: Optional[bool]
    classic_usd_source: Optional[str]
    classic_run_dir: Optional[str]
    working_usd_path: Optional[str]
    initial_placement_reused: Optional[bool]


class EvaluationIssue(BaseModel):
    type: str = Field(
        description=(
            "Issue type, e.g. collision, unsupported_surface, out_of_room, "
            "missing_object, visual_implausible, blocked_access."
        )
    )
    severity: str = Field(description="Issue severity: low, medium, high, or critical.")
    object: Optional[str] = Field(default=None, description="Primary object with the issue, if any.")
    related_objects: List[str] = Field(
        default_factory=list,
        description="Objects or structures involved, such as collision partners or support objects.",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Concrete evidence from tools, collision paths, placement logs, or snapshots.",
    )
    blocking: bool = Field(
        default=False,
        description="Whether this issue should force another placement pass.",
    )


class PlacementAction(BaseModel):
    object: str = Field(description="Object that should be adjusted.")
    action: str = Field(
        description=(
            "Action to take, e.g. move_object, rotate_object, move_to_support_surface, "
            "move_inside_room, separate_from_object, replace_asset, remove_duplicate, recapture_view."
        )
    )
    priority: str = Field(description="Action priority: low, medium, high, or critical.")
    reason: str = Field(description="Why this action is needed.")
    target_hint: Optional[str] = Field(
        default=None,
        description="Concrete placement hint for the next placement attempt.",
    )
    preferred_support_object: Optional[str] = Field(
        default=None,
        description="Preferred support object to resolve onto when the issue is support-related.",
    )
    preferred_surface: Optional[str] = Field(
        default=None,
        description="Preferred support surface such as up, down, left, right, front, or back.",
    )
    preferred_orientation: Optional[str] = Field(
        default=None,
        description="Preferred final orientation or facing direction for the repaired object.",
    )
    reference_objects: List[str] = Field(
        default_factory=list,
        description="Other objects that should be used as anchors or preserved references during repair.",
    )
    distance_constraints: List[str] = Field(
        default_factory=list,
        description="Concrete spacing, clearance, or flush-mount constraints for the repair.",
    )
    must_not_change: List[str] = Field(
        default_factory=list,
        description="Objects, supports, or relationships that should remain fixed while repairing this issue.",
    )
    repair_sequence: List[str] = Field(
        default_factory=list,
        description="Ordered low-level repair steps the placement agent should follow.",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence references such as tool names, collision paths, or snapshot paths.",
    )
    visual_observation: Optional[str] = Field(
        default=None,
        description="Short structured visual conclusion that should be treated as explicit visual memory for this repair.",
    )
    resolved_intent: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Exact repair intent used by evaluator before calling resolve_placement_intent.",
    )
    resolved_position: Optional[List[float]] = Field(
        default=None,
        description="Resolved world position [x, y, z] computed by evaluator tools for direct set_pose execution.",
    )
    resolved_rotation: Optional[List[float]] = Field(
        default=None,
        description="Resolved Euler rotation [rx, ry, rz] computed by evaluator tools for direct set_pose execution.",
    )


class PlacementEvaluation(BaseModel):
    needs_adjustment: bool = Field(description="Whether the current room should be re-placed.")
    summary: str = Field(description="Short overall assessment of the room placement result.")
    issues: List[EvaluationIssue] = Field(default_factory=list, description="Structured observed problems.")
    object_actions: List[PlacementAction] = Field(
        default_factory=list,
        description="Concrete object-level actions for the next placement attempt.",
    )
    accepted_with_warnings: bool = Field(
        default=False,
        description="Whether the room is accepted despite non-blocking residual issues.",
    )


class EvaluationDimensionScore(BaseModel):
    dimension: Literal[
        "physical_validity",
        "visual_realism",
        "layout_plausibility",
        "functional_coherence",
        "completeness",
    ]
    score: int = Field(ge=1, le=5)
    summary: str
    blocking: bool = False


class ScopedEvaluationIssue(BaseModel):
    id: str
    scope: Literal["room", "zone", "cluster", "object"]
    type: str
    severity: Literal["warning", "blocking"]
    objects: List[str] = Field(default_factory=list)
    zone: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    visual_observation: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class HighLevelRepairIntent(BaseModel):
    id: str
    issue_ids: List[str] = Field(default_factory=list)
    scope: Literal["room", "zone", "cluster", "object"]
    strategy: Literal[
        "clear_route",
        "restore_functional_zone",
        "rearrange_cluster",
        "relocate_object",
        "restore_support",
        "fix_orientation",
        "redistribute_density",
        "complete_missing",
    ]
    objective: str
    target_objects: List[str] = Field(default_factory=list)
    anchor_objects: List[str] = Field(default_factory=list)
    preserve_objects: List[str] = Field(default_factory=list)
    spatial_constraints: List[str] = Field(default_factory=list)


class RoomEvaluation(BaseModel):
    decision: Literal["pass", "retry", "accept_with_warnings"]
    summary: str
    scorecard: List[EvaluationDimensionScore] = Field(default_factory=list)
    issues: List[ScopedEvaluationIssue] = Field(default_factory=list)
    repair_intents: List[HighLevelRepairIntent] = Field(default_factory=list)


class ResolvedRepairOperation(BaseModel):
    source_intent_id: str
    object_name: str
    order: int = Field(ge=0)
    resolver_intent: Dict[str, Any] = Field(default_factory=dict)
    resolved_position: Optional[List[float]] = None
    resolved_rotation: Optional[List[float]] = None
    preserve_objects: List[str] = Field(default_factory=list)
    status: Literal["resolved", "unresolved"]
    error: Optional[str] = None


class ResolvedRepairPlan(BaseModel):
    summary: str
    operations: List[ResolvedRepairOperation] = Field(default_factory=list)


from Tools.evaluate_tool import (
    analyze_placement_log as raw_analyze_placement_log,
    capture_evaluation_snapshot as raw_capture_evaluation_snapshot,
    capture_evaluation_views as raw_capture_evaluation_views,
    check_scene_collisions as raw_check_scene_collisions,
    reset_trial_placements as raw_reset_trial_placements,
    resolve_placement_intent as raw_evaluator_resolve_placement_intent,
    scan_scene as raw_evaluator_scan_scene,
    summarize_room_layout as raw_summarize_room_layout,
)
from Tools.place_tool import (
    get_object_bounds,
    get_room_context,
    query_floor_space,
    query_surface_status,
    resolve_placement_intent,
    save_stage,
    scale_object,
    scan_scene,
    select_room,
    set_asset_name_usd_path_registry,
    set_pipe_id,
    set_pose,
    spawn_object,
)


AGENT_PLACEMENT_TOOLS = [
    get_room_context,
    query_floor_space,
    query_surface_status,
    scan_scene,
    get_object_bounds,
    spawn_object,
    set_pose,
    scale_object,
    resolve_placement_intent,
]


# ---------------------------------------------------------------------------
# Serialization helpers.
#
# The original WIP branch imported ``_json_safe`` / ``_message_signature`` /
# ``_serialize_stream_message`` from ``scene_generate``, but those helpers were
# never actually defined there (in either the branch or current main). They are
# small, self-contained utilities, so we define them locally here to keep
# ``scene_generate.py`` (the existing entry point) completely unchanged.
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Best-effort conversion of an arbitrary value into JSON-serializable data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    for attr in ("model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _json_safe(method())
            except Exception:
                pass
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _serialize_stream_message(message: Any) -> Dict[str, Any]:
    """Serialize a non-LangChain stream message (typically a dict) to a plain dict."""
    if isinstance(message, dict):
        role = message.get("role") or message.get("type") or "unknown"
        payload: Dict[str, Any] = {
            "type": message.get("type", message.__class__.__name__),
            "role": role,
            "content": _json_safe(message.get("content")),
        }
        if message.get("tool_calls"):
            payload["tool_calls"] = _json_safe(message.get("tool_calls"))
        if message.get("name"):
            payload["name"] = message.get("name")
        return payload
    role = getattr(message, "type", None) or getattr(message, "role", None) or message.__class__.__name__
    return {
        "type": message.__class__.__name__,
        "role": role,
        "content": _json_safe(getattr(message, "content", message)),
    }


def _message_signature(serialized: Any) -> str:
    """Produce a stable signature for a serialized message, for de-duplication."""
    if not isinstance(serialized, dict):
        return repr(serialized)
    for key in ("tool_call_id", "id"):
        if serialized.get(key):
            return f"{key}:{serialized[key]}"
    try:
        return json.dumps(
            {
                "role": serialized.get("role"),
                "type": serialized.get("type"),
                "content": serialized.get("content"),
                "tool_calls": serialized.get("tool_calls"),
                "name": serialized.get("name"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        return repr(serialized)


def _normalize_pose_vec3(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            return None
        value = parsed
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _build_initial_placement_tools() -> List[Any]:
    # Reuse the shared placement tools so graph mode gets the same thread-safe,
    # multi-position resolve cache and set_pose authorization as classic mode.
    return list(AGENT_PLACEMENT_TOOLS)


def _build_evaluator_evidence_tools(
    *,
    room_name: str,
    latest_trace_file: Optional[str],
    eval_output_dir: Optional[str],
) -> List[Any]:
    @tool
    def analyze_placement_log(include_structure: bool = False) -> dict:
        """Analyze the latest placement trace for the current room."""
        if not latest_trace_file:
            return {
                "success": False,
                "trace_file": None,
                "room_name": room_name,
                "objects": [],
                "object_count": 0,
                "message": "latest placement trace file is not available",
            }
        return raw_analyze_placement_log.invoke(
            {
                "trace_file": latest_trace_file,
                "room_name": room_name,
                "include_structure": include_structure,
            }
        )

    @tool
    def check_scene_collisions() -> dict:
        """Check collisions for the current room."""
        return raw_check_scene_collisions.invoke({"room_name": room_name})

    @tool
    def capture_evaluation_snapshot(object_name: Optional[str] = None) -> dict:
        """Capture one evaluator snapshot for the current room or a target object."""
        if not eval_output_dir:
            return {
                "success": False,
                "room_name": room_name,
                "object_name": object_name,
                "path": None,
                "view_type": "object_inspection" if object_name else "room_side",
                "message": "evaluation output directory is not available",
            }
        return raw_capture_evaluation_snapshot.invoke(
            {
                "output_dir": eval_output_dir,
                "room_name": room_name,
                "object_name": object_name,
            }
        )

    @tool
    def capture_evaluation_views(suspect_objects: Optional[list[str] | str] = None) -> dict:
        """Capture the standard evaluator views for the current room and optional suspect objects."""
        if not eval_output_dir:
            return {
                "success": False,
                "room_name": room_name,
                "views": [],
                "metadata_path": None,
                "message": "evaluation output directory is not available",
            }
        return raw_capture_evaluation_views.invoke(
            {
                "room_name": room_name,
                "output_dir": eval_output_dir,
                "suspect_objects": suspect_objects,
            }
        )

    return [
        analyze_placement_log,
        check_scene_collisions,
        raw_evaluator_scan_scene,
        capture_evaluation_snapshot,
        capture_evaluation_views,
    ]


def _build_evaluator_resolve_tools(room_name: str) -> List[Any]:
    @tool
    def resolve_placement_intent(intent: dict) -> dict:
        """Resolve a repair placement intent and stage it temporarily for downstream repair planning."""
        return raw_evaluator_resolve_placement_intent.invoke({"intent": intent, "room_name": room_name})

    return [
        get_room_context,
        query_floor_space,
        query_surface_status,
        raw_evaluator_scan_scene,
        get_object_bounds,
        resolve_placement_intent,
    ]


def _sanitize_room_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIC_USD_DIR = _REPO_ROOT / "exp" / "scene_gen" / "pipeline" / "usd"
_CLASSIC_RUNS_DIR = _REPO_ROOT / "exp" / "scene_gen" / "pipeline" / "runs"
_CLASSIC_FRONT_PREFIX = "scene_gen_20260614_gpt-5.5"
_CLASSIC_BACK_PREFIX = "scene_gen_20260610_gpt-5.5"
_TASK_ID_RE = re.compile(r"^H(?P<hazard>\d{2})_T(?P<task>\d{2})$")


def _is_reuse_classic_mode(state: GraphAgentState) -> bool:
    runtime = state.get("runtime") or {}
    return bool(state.get("reuse_classic_usd") or runtime.get("reuse_classic_usd"))


def _classic_prefix_for_task_id(task_id: str) -> str:
    text = str(task_id or "").strip()
    match = _TASK_ID_RE.match(text)
    if not match:
        raise ValueError(
            "Classic USD reuse requires task_id like H01_T01, "
            f"got {task_id!r}."
        )
    hazard_index = int(match.group("hazard"))
    task_index = int(match.group("task"))
    if not 1 <= task_index <= 10:
        raise ValueError(
            "Classic USD reuse only supports T01-T10 per hazard, "
            f"got {task_id!r}."
        )
    if 1 <= hazard_index <= 10:
        return _CLASSIC_FRONT_PREFIX
    if 11 <= hazard_index <= 15:
        return _CLASSIC_BACK_PREFIX
    raise ValueError(
        "Classic USD reuse only supports H01-H15, "
        f"got {task_id!r}."
    )


def _classic_usd_source_for_task_id(task_id: str) -> Path:
    prefix = _classic_prefix_for_task_id(task_id)
    return (_CLASSIC_USD_DIR / f"{prefix}_{task_id}.usda").resolve()


def _latest_run_dir(run_root: Path) -> Path:
    if not run_root.is_dir():
        raise FileNotFoundError(f"classic run root not found: {run_root}")
    candidates = [path for path in run_root.glob("run_*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run_* directory found under: {run_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _resolve_classic_run_dir(
    *,
    task_id: str,
    classic_usd_source: Path,
    explicit_classic_run_dir: Optional[str],
) -> Path:
    if explicit_classic_run_dir:
        candidate = Path(explicit_classic_run_dir).expanduser().resolve()
        if (candidate / "blueprint.json").is_file() and (candidate / "final_assets.json").is_file():
            return candidate
        return _latest_run_dir(candidate)
    return _latest_run_dir((_CLASSIC_RUNS_DIR / classic_usd_source.stem).resolve())


def _load_json_file(path: Path, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_classic_run_artifacts(classic_run_dir: Path, run_output_dir: Path) -> None:
    """Copy reusable classic artifacts into the graph run directory.

    ``blueprint.json`` and ``final_assets.json`` are required and are loaded
    before this function is called.  ``llm_selections.json`` is copied when
    available, otherwise a small placeholder is written so pipeline completion
    checks still recognize the reuse run as a completed scene-generator run.
    """
    for filename in ("blueprint.json", "final_assets.json", "asset_candidates.json"):
        source = classic_run_dir / filename
        if source.is_file():
            shutil.copy2(source, run_output_dir / filename)

    llm_selections = classic_run_dir / "llm_selections.json"
    target_llm_selections = run_output_dir / "llm_selections.json"
    if llm_selections.is_file():
        shutil.copy2(llm_selections, target_llm_selections)
    elif not target_llm_selections.exists():
        with target_llm_selections.open("w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "reuse_classic_usd": True,
                        "message": "No llm_selections.json was available in the classic run; graph reuse skipped asset selection.",
                    }
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )


def _load_classic_reuse_artifacts(
    *,
    task_id: Optional[str],
    classic_usd_source_arg: Optional[str],
    classic_run_dir_arg: Optional[str],
    run_output_dir: Path,
) -> Dict[str, Any]:
    if not task_id:
        raise ValueError("--task-id is required when --reuse-classic-usd is enabled")

    classic_usd_source = (
        Path(classic_usd_source_arg).expanduser().resolve()
        if classic_usd_source_arg
        else _classic_usd_source_for_task_id(task_id)
    )
    if not classic_usd_source.is_file():
        raise FileNotFoundError(f"classic USD source not found: {classic_usd_source}")

    classic_run_dir = _resolve_classic_run_dir(
        task_id=task_id,
        classic_usd_source=classic_usd_source,
        explicit_classic_run_dir=classic_run_dir_arg,
    )
    blueprint = _load_json_file(classic_run_dir / "blueprint.json", "classic blueprint.json")
    final_assets = _load_json_file(classic_run_dir / "final_assets.json", "classic final_assets.json")
    if not isinstance(blueprint, dict):
        raise ValueError(f"classic blueprint.json must contain a JSON object: {classic_run_dir / 'blueprint.json'}")
    if not isinstance(final_assets, dict):
        raise ValueError(f"classic final_assets.json must contain a JSON object: {classic_run_dir / 'final_assets.json'}")
    _copy_classic_run_artifacts(classic_run_dir, run_output_dir)
    return {
        "classic_usd_source": str(classic_usd_source),
        "classic_run_dir": str(classic_run_dir),
        "scene_blueprint": blueprint,
        "retrieved_assets": final_assets,
    }


def _derive_floor_plan(
    scene_blueprint: Optional[Dict[str, Any]],
    retrieved_assets: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    scene_blueprint = scene_blueprint or {}
    retrieved_assets = retrieved_assets or {}
    room_specs = {
        _sanitize_room_name(room.get("name", "")): room
        for room in scene_blueprint.get("rooms", [])
        if isinstance(room, dict) and room.get("name")
    }
    rooms = list(room_specs)
    if not rooms:
        rooms = [_sanitize_room_name(k) for k in retrieved_assets.keys()]

    floor_plan = {}
    for room_name in rooms:
        room_spec = room_specs.get(room_name, {})
        size = room_spec.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                approximate_area = float(size[0]) * float(size[1])
            except (TypeError, ValueError):
                approximate_area = 25.0
        else:
            approximate_area = 25.0
        floor_plan[room_name] = {
            "size": approximate_area,
            "door_locations": [],
            "window_locations": [],
            "eval_cameras": {
                "overview_top": f"/World/rooms/{room_name}/eval_cameras/{room_name}_OverviewTopCamera",
                "inspect_side": f"/World/rooms/{room_name}/eval_cameras/{room_name}_InspectSideCamera",
            },
        }
    return floor_plan


def node_floor_plan_generator(state: GraphAgentState):
    """
    Build graph-side room metadata after the current main retriever creates rooms.

    Room construction remains owned by ``node_retriever_and_rooms_construction``
    so the graph entry inherits the latest main behavior without duplicating it.
    """
    print("--- [Graph Node] Floor Plan: indexing constructed rooms ---")
    scene_blueprint = state.get("scene_blueprint") or {}
    retrieved_assets = state.get("retrieved_assets") or {}
    room_specs = {
        _sanitize_room_name(room.get("name", "")): room
        for room in scene_blueprint.get("rooms", [])
        if isinstance(room, dict) and room.get("name")
    }
    rooms = list(room_specs)
    if not rooms:
        rooms = [_sanitize_room_name(k) for k in retrieved_assets.keys()]

    room_num = len(rooms)
    house_length = _estimate_house_length_from_blueprint(
        scene_blueprint,
        fallback_room_count=room_num,
    )
    connectivity = [(i, i + 1) for i in range(room_num - 1)]
    print(
        f"Rooms were created by the main retriever: count={room_num}, "
        f"house_length={house_length}m, connectivity={connectivity}"
    )

    floor_plan = {}
    for room_name in rooms:
        room_spec = room_specs.get(room_name, {})
        size = room_spec.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            try:
                approximate_area = float(size[0]) * float(size[1])
            except (TypeError, ValueError):
                approximate_area = 25.0
        else:
            approximate_area = 25.0
        floor_plan[room_name] = {
            "size": approximate_area,
            "door_locations": [],
            "window_locations": [],
            "eval_cameras": {
                "overview_top": f"/World/rooms/{room_name}/eval_cameras/{room_name}_OverviewTopCamera",
                "inspect_side": f"/World/rooms/{room_name}/eval_cameras/{room_name}_InspectSideCamera",
            },
        }

    print(f"\nGenerated floor plan for {len(floor_plan)} rooms:")
    for room_name, info in floor_plan.items():
        print(
            f"  - {room_name}: size={info['size']} sqm, "
            f"doors={len(info['door_locations'])}, windows={len(info['window_locations'])}, "
            f"eval_cameras={info['eval_cameras']}"
        )
    return {
        **_interface_fields(state),
        "floor_plan": floor_plan,
    }


def node_room_selector(state: GraphAgentState):
    """Select the next unfinished room, matching update_scene_generate's room-by-room graph flow."""
    print("--- [Graph Node] Room Selector: choosing next room ---")
    rooms = [_sanitize_room_name(room.get("name", "")) for room in (state.get("scene_blueprint") or {}).get("rooms", [])]
    if not rooms:
        rooms = [_sanitize_room_name(k) for k in (state.get("retrieved_assets") or {}).keys()]

    placed_rooms = set(state.get("placed_rooms") or [])
    next_room = next((room for room in rooms if room and room not in placed_rooms), None)

    print(f"Rooms to process: {rooms}")
    print(f"Rooms already completed: {sorted(placed_rooms)}")
    if next_room:
        print(f"Selected room for placement: {next_room}")
        raw_assets = state.get("retrieved_assets") or {}
        room_assets = raw_assets.get(next_room)
        if room_assets is None:
            for orig_key, val in raw_assets.items():
                if _sanitize_room_name(orig_key) == next_room:
                    room_assets = val
                    break
        if room_assets is None:
            room_assets = {}
        objects_to_place = _format_room_assets(next_room, _filter_structural_room_assets(room_assets))
        room_info = _build_room_info(next_room, room_assets, state.get("floor_plan", {}) or {})
        if _is_reuse_classic_mode(state):
            print(
                f"[Reuse Classic USD] Skipping initial placement for room '{next_room}' "
                "and starting graph evaluator directly."
            )
            print(f"[Room Selection] Switching Isaac Sim context to room '{next_room}'")
            room_selected = select_room.invoke({"room_name": next_room})
            if not room_selected:
                raise RuntimeError(
                    f"Failed to select room '{next_room}' in reused classic USD. "
                    "Check that the copied source USD contains matching /World/rooms prims."
                )
            return Command(
                update={
                    "current_room": next_room,
                    "need_adjustment": False,
                    "current_room_completed": True,
                    "evaluation_report": None,
                    "place_trajectory": [],
                    "evaluation_trajectory": [],
                    "objects_to_place": objects_to_place,
                    "room_info": room_info,
                    "pending_place_assets": [],
                    "retry_count": 0,
                    "active_room_for_trajectory": next_room,
                    "initial_placement_reused": True,
                },
                goto="evaluate",
            )
        return Command(
            update={
                "current_room": next_room,
                "need_adjustment": False,
                "current_room_completed": False,
                "evaluation_report": None,
                "place_trajectory": [],
                "evaluation_trajectory": [],
                "objects_to_place": objects_to_place,
                "room_info": room_info,
                "pending_place_assets": _pending_assets_from_room_assets(room_assets),
                "retry_count": 0,
                "active_room_for_trajectory": next_room,
            },
            goto="place",
        )

    print("All rooms processed. Ending graph.")
    return Command(update={"current_room": None}, goto=END)


def _format_room_assets(room_name: str, room_assets: Dict[str, Any]) -> str:
    """Use the current main asset naming and placement prompt format."""
    return _build_placement_asset_library(
        room_name,
        _filter_structural_room_assets(room_assets),
    )


def _filter_structural_room_assets(
    room_assets: Dict[str, Any],
) -> Dict[str, Any]:
    """Exclude structural and hard-filtered assets from graph placement."""
    filtered: Dict[str, Any] = {}
    for asset_key, info in _filter_excluded_room_assets(room_assets).items():
        if not isinstance(info, dict):
            continue
        filtered[asset_key] = info
    return filtered


def _build_room_info(room_name: str, room_assets: Dict[str, Any], floor_plan: Dict[str, Any]) -> str:
    floor_info = floor_plan.get(room_name, {})
    room_overview = room_assets.get("room_overview", f"A {room_name} with typical furniture and layout.")
    return (
        f"Room Name: {room_name}\n"
        f"Overview: {room_overview}\n"
        f"Approx Size: {floor_info.get('size', room_assets.get('room_size', 25))}\n"
        f"Door Locations: {floor_info.get('door_locations', [])}\n"
    )


def _serialize_message(message: AnyMessage) -> Dict[str, Any]:
    role = getattr(message, "type", None) or getattr(message, "role", None) or message.__class__.__name__
    payload: Dict[str, Any] = {
        "type": message.__class__.__name__,
        "role": role,
        "content": _json_safe(message.content),
    }
    if isinstance(message, AIMessage):
        payload["tool_calls"] = _json_safe(getattr(message, "tool_calls", []) or [])
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = getattr(message, "tool_call_id", None)
        payload["name"] = getattr(message, "name", None)
    return payload


def _format_message_block(index: int, message: AnyMessage) -> str:
    header = f"[{index}] {message.__class__.__name__}"
    lines = [header]
    content = message.content
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    if content:
        label = "Reasoning / Output" if isinstance(message, AIMessage) else "Content"
        lines.append(f"{label}:")
        lines.append(str(content))
    if isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", []) or []
        if tool_calls:
            lines.append("Tool Calls:")
            for tool_idx, tool_call in enumerate(tool_calls, start=1):
                lines.append(f"  {tool_idx}. {tool_call.get('name', '')}")
                lines.append(json.dumps(tool_call.get("args", {}), ensure_ascii=False, indent=2))
    if isinstance(message, ToolMessage):
        lines.append("Tool Result:")
        lines.append(str(message.content))
    return "\n".join(lines)


def _render_trajectory_text(messages: List[AnyMessage]) -> str:
    if not messages:
        return "No placement messages yet."
    return "\n\n".join(_format_message_block(index, message) for index, message in enumerate(messages, start=1))


def _truncate_text(text: str, max_chars: int = 24000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars // 2]}\n\n...[truncated]...\n\n{text[-max_chars // 2 :]}"


def _strip_tool_calls_for_llm(messages: List[AnyMessage]) -> List[AnyMessage]:
    """Return message copies without assistant tool call metadata.

    Some chat/code models reject replayed historical tool calls unless every
    function.arguments field is serialized exactly as JSON. For final report
    generation we only need the semantic trace, not executable tool metadata.
    """
    sanitized: List[AnyMessage] = []
    for message in messages:
        if isinstance(message, AIMessage):
            sanitized.append(AIMessage(content=message.content))
        elif isinstance(message, ToolMessage):
            sanitized.append(
                ToolMessage(
                    content=message.content,
                    tool_call_id=getattr(message, "tool_call_id", None) or "sanitized_tool_call",
                    name=getattr(message, "name", None),
                )
            )
        elif isinstance(message, HumanMessage):
            sanitized.append(HumanMessage(content=message.content))
        elif isinstance(message, SystemMessage):
            sanitized.append(SystemMessage(content=message.content))
        else:
            sanitized.append(message)
    return sanitized


def _image_file_to_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"image path is not a file: {path}")

    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _json_from_message_content(content: Any) -> Optional[Dict[str, Any]]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _extract_evaluation_image_records(messages: List[AnyMessage]) -> List[Dict[str, Optional[str]]]:
    image_records: List[Dict[str, Optional[str]]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = getattr(message, "name", None)
        if tool_name not in {"capture_evaluation_snapshot", "capture_evaluation_views"}:
            continue
        payload = _json_from_message_content(message.content)
        if not payload or payload.get("success") is False:
            continue
        if isinstance(payload, dict):
            views = payload.get("views")
            if isinstance(views, list):
                for view in views:
                    if not isinstance(view, dict):
                        continue
                    path = view.get("path")
                    if not isinstance(path, str):
                        continue
                    suffix = Path(path).suffix.lower()
                    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                        continue
                    image_records.append(
                        {
                            "path": path,
                            "name": _as_optional_string(view.get("name")),
                            "type": _as_optional_string(view.get("type")),
                            "target_object": _as_optional_string(view.get("target_object")),
                        }
                    )
                continue

            for key in ("path", "image_path", "output_path"):
                value = payload.get(key)
                if isinstance(value, str):
                    suffix = Path(value).suffix.lower()
                    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
                        image_records.append(
                            {
                                "path": value,
                                "name": None,
                                "type": "snapshot",
                                "target_object": _as_optional_string(payload.get("object_name")),
                            }
                        )

    deduped: List[Dict[str, Optional[str]]] = []
    seen = set()
    for record in image_records:
        image_path = record.get("path")
        if not image_path:
            continue
        if image_path in seen:
            continue
        seen.add(image_path)
        deduped.append(record)
    return deduped


def _run_evaluator_visual_pass(
    *,
    llm: Any,
    room_name: str,
    evaluator_trace: List[AnyMessage],
    image_source_messages: Optional[List[AnyMessage]] = None,
    max_images: int = 15,
) -> List[AIMessage]:
    image_records = _extract_evaluation_image_records(image_source_messages or evaluator_trace)[:max_images]
    if not image_records:
        return []

    trace_context = _truncate_text(_render_trajectory_text(evaluator_trace))
    prompt_lines = [
        "You are the same SceneCraft placement evaluator, now performing the visual evidence pass.",
        "Inspect all attached images together in one pass rather than independently.",
        "Use the evaluator tool trace as context, then compare the images directly.",
        "Focus on visible object-object overlap, wall penetration, floating/unsupported objects, support-surface plausibility, blocked access, and visually implausible orientation.",
        "If some suspected issue is not clearly visible, say uncertain rather than guessing.",
        "",
        f"Room: {room_name}",
        "",
        "Attached image set:",
    ]
    for idx, record in enumerate(image_records, start=1):
        prompt_lines.append(
            f"{idx}. path={record.get('path')} | view_name={record.get('name') or 'unknown'} | "
            f"view_type={record.get('type') or 'unknown'} | target_object={record.get('target_object') or 'room'}"
        )
    prompt_lines.extend(
        [
            "",
            "Evaluator tool trace context:",
            trace_context,
            "",
            "Return concise JSON without markdown fences using fields: "
            "room_name, verdict ('pass'|'fail'|'uncertain'), summary, cross_view_observations, "
            "per_image_findings, visible_objects, visual_observations, issues, needs_geometry_confirmation. "
            "Each entry in `visual_observations` should be a compact object with fields like "
            "object, observed_support, observed_orientation, observed_issue, recommended_support, "
            "recommended_surface, recommended_orientation, confidence.",
        ]
    )
    prompt = "\n".join(prompt_lines)

    human_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    record_lines: List[str] = []
    for idx, record in enumerate(image_records, start=1):
        image_path = record.get("path")
        if not image_path:
            continue
        try:
            data_url = _image_file_to_data_url(image_path)
        except Exception as exc:
            record_lines.append(
                f"{idx}. path={image_path} | status=skipped | reason={exc}"
            )
            continue
        human_content.append(
            {
                "type": "text",
                "text": (
                    f"Image {idx}: path={image_path}, view_name={record.get('name') or 'unknown'}, "
                    f"view_type={record.get('type') or 'unknown'}, target_object={record.get('target_object') or 'room'}"
                ),
            }
        )
        human_content.append({"type": "image_url", "image_url": {"url": data_url}})
        record_lines.append(
            f"{idx}. path={image_path} | view_name={record.get('name') or 'unknown'} | "
            f"view_type={record.get('type') or 'unknown'} | target_object={record.get('target_object') or 'room'}"
        )

    if len(human_content) == 1:
        return [
            AIMessage(
                content="[Visual analysis skipped]\nNo readable images were available for the current evaluator pass."
            )
        ]

    try:
        response = llm.invoke(
            [
                SystemMessage(content=EVALUATOR_VISUAL_SYSTEM_PROMPT),
                HumanMessage(content=human_content),
            ]
        )
        response_content = getattr(response, "content", response)
    except Exception as exc:
        response_content = (
            f'{{"room_name": "{room_name}", "verdict": "uncertain", '
            f'"summary": "Visual analysis failed: {exc}", "cross_view_observations": [], '
            f'"per_image_findings": [], "visible_objects": [], "visual_observations": [], "issues": [], '
            f'"needs_geometry_confirmation": true}}'
        )

    return [
        AIMessage(
            content=(
                "[Evaluator visual analysis]\n"
                f"Room: {room_name}\n"
                "Images:\n"
                + "\n".join(record_lines)
                + "\n"
                + str(response_content)
            )
        )
    ]


def _safe_log_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value).strip("_") or "room"


def _room_log_dir(state: GraphAgentState, room_name: str) -> Optional[str]:
    output_dir = state.get("output_dir") or (state.get("runtime") or {}).get("output_dir")
    if not output_dir:
        return None
    trace_dir = os.path.join(output_dir, "placement_logs", _safe_log_component(room_name))
    os.makedirs(trace_dir, exist_ok=True)
    return trace_dir


def _room_log_subdir(state: GraphAgentState, room_name: str, subdir: str) -> Optional[str]:
    trace_dir = _room_log_dir(state, room_name)
    if not trace_dir:
        return None
    path = os.path.join(trace_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _placement_log_dir(state: GraphAgentState, room_name: str) -> Optional[str]:
    return _room_log_subdir(state, room_name, "placement")


def _evaluate_log_dir(state: GraphAgentState, room_name: str) -> Optional[str]:
    return _room_log_subdir(state, room_name, "evaluate")


def _evaluation_attempt_suffix(retry_count: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"attempt_retry_{retry_count:02d}_{timestamp}"


def _latest_placement_trace_file(state: GraphAgentState, room_name: str) -> Optional[str]:
    placement_dir = _placement_log_dir(state, room_name)
    if placement_dir:
        latest_path = os.path.join(placement_dir, "latest.json")
        if os.path.exists(latest_path):
            return latest_path

    # Backward-compatible fallback for runs produced before placement/evaluate
    # subdirectories were introduced.
    trace_dir = _room_log_dir(state, room_name)
    if trace_dir:
        legacy_path = os.path.join(trace_dir, "latest.json")
        if os.path.exists(legacy_path):
            return legacy_path
        return os.path.join(placement_dir or trace_dir, "latest.json")
    return None


def _persist_room_trajectory_snapshot(
    state: GraphAgentState,
    room_name: str,
    messages: List[AnyMessage],
    suffix: str,
) -> None:
    trace_dir = _placement_log_dir(state, room_name)
    if not trace_dir:
        return

    json_path = os.path.join(trace_dir, f"{suffix}.json")
    serialized = [_serialize_message(message) for message in messages]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)


def _persist_evaluation_report(
    state: GraphAgentState,
    room_name: str,
    report: Dict[str, Any],
    suffix: str = "evaluation_report",
) -> None:
    trace_dir = _evaluate_log_dir(state, room_name)
    if not trace_dir:
        return

    json_path = os.path.join(trace_dir, f"{suffix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _persist_evaluator_trajectory_snapshot(
    state: GraphAgentState,
    room_name: str,
    messages: List[AnyMessage],
    suffix: str,
) -> None:
    trace_dir = _evaluate_log_dir(state, room_name)
    if not trace_dir:
        return

    json_path = os.path.join(trace_dir, f"{suffix}.json")
    serialized = [_serialize_message(message) for message in messages]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)


def _as_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    text = str(value).strip()
    return text or None


def _as_string(value: Any, default: str) -> str:
    text = _as_optional_string(value)
    return text if text is not None else default


def _as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    results: List[str] = []
    for item in value:
        text = _as_optional_string(item)
        if text is not None:
            results.append(text)
    return results


def _normalize_evaluation_issue(issue: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(issue, dict):
        return None
    return {
        "type": _as_string(issue.get("type"), "unknown_issue"),
        "severity": _as_string(issue.get("severity"), "medium"),
        "object": _as_optional_string(issue.get("object")),
        "related_objects": _as_string_list(issue.get("related_objects")),
        "evidence": issue.get("evidence") if isinstance(issue.get("evidence"), dict) else {},
        "blocking": bool(issue.get("blocking", False)),
    }


def _normalize_placement_action(action: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(action, dict):
        return None
    object_name = _as_optional_string(action.get("object"))
    if object_name is None:
        return None
    resolved_position = action.get("resolved_position")
    if not (
        isinstance(resolved_position, list)
        and len(resolved_position) == 3
        and all(isinstance(v, (int, float)) for v in resolved_position)
    ):
        resolved_position = None
    resolved_rotation = action.get("resolved_rotation")
    if not (
        isinstance(resolved_rotation, list)
        and len(resolved_rotation) == 3
        and all(isinstance(v, (int, float)) for v in resolved_rotation)
    ):
        resolved_rotation = None
    resolved_intent = action.get("resolved_intent") if isinstance(action.get("resolved_intent"), dict) else None
    return {
        "object": object_name,
        "action": _as_string(action.get("action"), "move_object"),
        "priority": _as_string(action.get("priority"), "medium"),
        "reason": _as_string(action.get("reason"), "Evaluator requested a placement adjustment."),
        "target_hint": _as_optional_string(action.get("target_hint")),
        "preferred_support_object": _as_optional_string(action.get("preferred_support_object")),
        "preferred_surface": _as_optional_string(action.get("preferred_surface")),
        "preferred_orientation": _as_optional_string(action.get("preferred_orientation")),
        "reference_objects": _as_string_list(action.get("reference_objects")),
        "distance_constraints": _as_string_list(action.get("distance_constraints")),
        "must_not_change": _as_string_list(action.get("must_not_change")),
        "repair_sequence": _as_string_list(action.get("repair_sequence")),
        "evidence": _as_string_list(action.get("evidence")),
        "visual_observation": _as_optional_string(action.get("visual_observation")),
        "resolved_intent": resolved_intent,
        "resolved_position": resolved_position,
        "resolved_rotation": resolved_rotation,
    }


def _normalize_placement_evaluation_report(report: Any) -> Dict[str, Any]:
    payload = report if isinstance(report, dict) else {}
    issues: List[Dict[str, Any]] = []
    for issue in payload.get("issues", []):
        normalized_issue = _normalize_evaluation_issue(issue)
        if normalized_issue is not None:
            issues.append(normalized_issue)

    object_actions: List[Dict[str, Any]] = []
    dropped_actions = 0
    for action in payload.get("object_actions", []):
        normalized_action = _normalize_placement_action(action)
        if normalized_action is None:
            dropped_actions += 1
            continue
        object_actions.append(normalized_action)

    summary = _as_string(payload.get("summary"), "Placement evaluation completed.")
    if dropped_actions:
        summary = f"{summary} Dropped {dropped_actions} malformed object_actions from evaluator output."

    normalized = {
        "needs_adjustment": bool(payload.get("needs_adjustment", False)),
        "summary": summary,
        "issues": issues,
        "object_actions": object_actions,
        "accepted_with_warnings": bool(payload.get("accepted_with_warnings", False)),
    }

    if normalized["needs_adjustment"] and not normalized["issues"] and not normalized["object_actions"]:
        normalized["accepted_with_warnings"] = True
        normalized["summary"] = (
            f"{normalized['summary']} No valid actionable feedback remained after normalization; "
            "accepting current placement with warnings."
        )
        normalized["needs_adjustment"] = False

    return normalized


def _coerce_report_from_structured_output(result: Any) -> Dict[str, Any]:
    if isinstance(result, PlacementEvaluation):
        return result.model_dump()

    if not isinstance(result, dict):
        return _normalize_placement_evaluation_report({})

    parsed = result.get("parsed")
    if isinstance(parsed, PlacementEvaluation):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return _normalize_placement_evaluation_report(parsed)

    raw = result.get("raw")
    tool_calls = getattr(raw, "tool_calls", None)
    if isinstance(tool_calls, list):
        for tool_call in reversed(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            args = tool_call.get("args")
            if isinstance(args, dict):
                return _normalize_placement_evaluation_report(args)

    content = getattr(raw, "content", None)
    if isinstance(content, str) and content.strip():
        try:
            return _normalize_placement_evaluation_report(json.loads(content))
        except json.JSONDecodeError:
            pass

    return _normalize_placement_evaluation_report({})


def _pending_assets_from_room_assets(room_assets: Dict[str, Any]) -> List[Dict[str, Any]]:
    room_assets = _filter_structural_room_assets(room_assets)
    asset_items = [
        (asset_key, info)
        for asset_key, info in room_assets.items()
        if isinstance(info, dict) and info.get("asset_id") and info.get("usd_path")
    ]
    filtered_assets = {asset_key: info for asset_key, info in asset_items}
    named_entries = _build_room_asset_entries(filtered_assets)

    pending = []
    for (asset_key, info), entry in zip(asset_items, named_entries):
        pending.append(
            {
                "asset_key": asset_key,
                "asset_id": str(info.get("asset_id", asset_key)),
                "usd_path": info.get("usd_path"),
                "object_name": entry["agent_name"],
            }
        )
    return pending


def _tool_result_success(content: Any) -> bool:
    if isinstance(content, bool):
        return content
    if content is None:
        return False
    text = str(content).strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"false", "0", "none", "null", "failed", "error"}:
        return False
    if lowered.startswith(("[rejected]", "error:", "failed:", "false,")):
        return False
    return True


def _placed_object_names_from_messages(messages: List[AnyMessage]) -> set[str]:
    """Return the set of object names that were both spawned and posed successfully.

    Adapted to the current main ``spawn_object(asset_name, ...)`` API. The agent
    passes the object's name as ``asset_name`` (the tool also names the spawned
    prim after it), so placement success is tracked by object name rather than by
    USD path. An object counts as placed once its ``spawn_object`` and a later
    ``set_pose`` both report success.
    """
    calls_by_id: Dict[str, Dict[str, Any]] = {}
    calls_by_name: Dict[str, List[Dict[str, Any]]] = {}
    spawned_objects: set[str] = set()
    posed_objects: set[str] = set()

    for message in messages:
        if isinstance(message, HumanMessage):
            metadata = message.additional_kwargs or {}
            if metadata.get("lc_source") == "asset_placement_folding":
                summary = metadata.get("placement_summary") or {}
                object_name = summary.get("asset_name")
                set_pose_summary = summary.get("set_pose") or {}
                if object_name and set_pose_summary.get("success") is True:
                    spawned_objects.add(str(object_name))
                    posed_objects.add(str(object_name))

        serialized = _serialize_message(message)
        role = serialized.get("role")
        if role in ("ai", "assistant"):
            for tool_call in serialized.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                tool_name = tool_call.get("name") or (tool_call.get("function") or {}).get("name")
                tool_args = tool_call.get("args") or (tool_call.get("function") or {}).get("arguments") or {}
                if not isinstance(tool_args, dict):
                    tool_args = {}
                call = {
                    "name": tool_name,
                    "args": tool_args,
                    "id": tool_call.get("id"),
                }
                if call["id"]:
                    calls_by_id[str(call["id"])] = call
                calls_by_name.setdefault(str(tool_name), []).append(call)
            continue

        if role != "tool":
            continue

        tool_name = serialized.get("name")
        tool_call_id = serialized.get("tool_call_id")
        call = calls_by_id.pop(str(tool_call_id), None) if tool_call_id else None
        if call is None and tool_name:
            queued = calls_by_name.get(str(tool_name)) or []
            if queued:
                call = queued.pop(0)
        if call is None:
            continue

        args = call.get("args") or {}
        content = serialized.get("content")
        if call.get("name") == "spawn_object" and _tool_result_success(content):
            # New API arg is `asset_name`; keep `object_name` as a defensive fallback.
            object_name = args.get("asset_name") or args.get("object_name")
            if object_name:
                spawned_objects.add(str(object_name))
            # The tool returns the spawned object name; record it too when usable.
            if isinstance(content, str):
                returned = content.strip()
                if returned and not returned.lower().startswith(("[rejected]", "false", "error", "none")):
                    spawned_objects.add(returned)
        elif call.get("name") == "set_pose" and _tool_result_success(content):
            object_name = args.get("object_name")
            if object_name:
                posed_objects.add(str(object_name))

    return {object_name for object_name in spawned_objects if object_name in posed_objects}


def _remaining_pending_assets(
    expected_assets: List[Dict[str, Any]],
    placed_object_names: set[str],
) -> List[Dict[str, Any]]:
    placed = set(placed_object_names)
    remaining = []
    for asset in expected_assets:
        object_name = str(asset.get("object_name", ""))
        if object_name and object_name in placed:
            continue
        remaining.append(asset)
    return remaining


def _build_agent_user_prompt(state: GraphAgentState, objects_to_place: str, room_info: str) -> str:
    evaluation_report = state.get("evaluation_report")
    need_adjustment = state.get("need_adjustment", False)
    pending_assets = state.get("pending_place_assets") or []
    base_prompt = OPTIMIZED_PLACE_AGENT_USER_PROMPT.format(
        objects_to_place=objects_to_place,
        room_info=room_info,
    )

    if pending_assets:
        base_prompt = (
            f"{base_prompt}\n\n"
            "Remaining assets that still need successful placement:\n"
            f"{json.dumps(pending_assets, indent=2, ensure_ascii=False)}\n\n"
            "Use the provided `object_name` exactly for `spawn_object`, `resolve_placement_intent`, and `set_pose`, "
            "and reuse that same name for all later tool calls. Do not invent a different object name. "
            "Do not stop until this remaining list is empty. For each remaining asset, spawn or reuse the object, "
            "resolve its placement with `resolve_placement_intent`, then call `set_pose` successfully."
        )

    if need_adjustment and evaluation_report:
        return (
            f"{base_prompt}\n\n"
            "Evaluator feedback from the previous attempt:\n"
            f"{json.dumps(evaluation_report, indent=2, ensure_ascii=False)}\n\n"
            "Revise this room placement according to the feedback. Prioritize `object_actions`, "
            "fix every issue where `blocking` is true, and keep existing correct placements when possible. "
            "Treat each action's `visual_observation` as explicit visual memory from evaluator review. "
            "If an action includes `resolved_position` and optional `resolved_rotation`, execute that repair directly "
            "with `set_pose` instead of re-deriving a new pose. If an action includes `resolved_intent`, do not invent "
            "a different repair plan unless the current scene state clearly contradicts it."
        )
    return base_prompt


def _build_evaluator_user_prompt(
    state: GraphAgentState,
    room_name: str,
    latest_trace_file: Optional[str],
    eval_output_dir: Optional[str],
) -> str:
    return (
        f"Current room: {room_name}\n\n"
        f"Latest placement trace file: {latest_trace_file or 'not available'}\n\n"
        f"Evaluator image/output directory: {eval_output_dir or 'not available'}\n\n"
        f"Retry count: {state.get('retry_count', 0)} / {state.get('max_retry', 3)}\n\n"
        f"Room info:\n{state.get('room_info', '')}\n\n"
        f"Objects to place:\n{state.get('objects_to_place', '')}\n\n"
        "This is the evidence-and-visual phase. Evaluate this room like a tool-using evaluator agent. "
        "First call `analyze_placement_log`, then call `check_scene_collisions`, and use `scan_scene` to inspect "
        "the current real scene state when needed. The graph will bind these tools to the current room and latest "
        "placement trace automatically. Use `capture_evaluation_views` or `capture_evaluation_snapshot` only for "
        "ambiguous suspect objects. Do not solve repair coordinates in this phase. Instead, gather the highest-value "
        "evidence and externalize what you saw into explicit visual observations that a later resolve phase can use."
    )


def _build_evaluator_resolve_user_prompt(
    state: GraphAgentState,
    room_name: str,
) -> str:
    return (
        f"Current room: {room_name}\n\n"
        f"Retry count: {state.get('retry_count', 0)} / {state.get('max_retry', 3)}\n\n"
        "This is the resolve-and-report phase. Reuse the evidence and explicit visual observations already gathered "
        "above. Only use `scan_scene` if you need to re-check the live scene state before forming a concrete repair "
        "intent. Then use `resolve_placement_intent` only for the highest-value blocking issues. "
        "Do not retry the same failed resolve intent with superficial paraphrases. "
        "Your goal is to leave enough concrete resolved actions for the placement agent to execute next."
    )


def _dedupe_agent_messages(messages: object, seen_signatures: set[str]) -> List[AnyMessage]:
    if not isinstance(messages, list):
        return []

    new_messages: List[AnyMessage] = []
    for message in messages:
        if isinstance(message, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
            serialized = _serialize_message(message)
        else:
            serialized = _serialize_stream_message(message)
            if isinstance(message, dict):
                if message.get("tool_call_id"):
                    serialized["tool_call_id"] = message.get("tool_call_id")
                if message.get("id"):
                    serialized["id"] = message.get("id")
        signature = _message_signature(serialized)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        if isinstance(message, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
            new_messages.append(message)
        else:
            role = serialized.get("role", "unknown")
            content = serialized.get("content", "")
            if role in ("ai", "assistant"):
                new_messages.append(AIMessage(content=content))
            elif role == "tool":
                new_messages.append(ToolMessage(content=content, tool_call_id=serialized.get("tool_call_id") or "agent_tool"))
            elif role in ("human", "user"):
                new_messages.append(HumanMessage(content=content))
            else:
                new_messages.append(AIMessage(content=f"[{role}] {content}"))
    return new_messages


def node_place_with_sub_agent(state: GraphAgentState):
    """Run a streaming LangChain placement agent inside the current room graph node."""
    print("--- [Graph Node] Placement Agent: streaming room-level tool-use loop ---")
    room_name = state.get("current_room")
    if not room_name:
        return {"place_trajectory": [AIMessage(content="No current room is selected yet.")]}

    retrieved_assets = state.get("retrieved_assets", {}) or {}
    room_assets = retrieved_assets.get(room_name, {})
    if not room_assets:
        return {
            "objects_to_place": "",
            "room_info": f"Room {room_name} has no retrieved assets.",
            "place_trajectory": [AIMessage(content=f"Room {room_name} has no assets to place.")],
            "active_room_for_trajectory": room_name,
            "pending_place_assets": [],
            "current_room_completed": True,
        }

    placement_room_assets = _filter_structural_room_assets(room_assets)
    expected_assets = _pending_assets_from_room_assets(placement_room_assets)
    objects_to_place = _format_room_assets(room_name, placement_room_assets)
    room_info = _build_room_info(room_name, room_assets, state.get("floor_plan", {}) or {})
    if not expected_assets:
        return {
            "objects_to_place": objects_to_place,
            "room_info": room_info,
            "place_trajectory": [
                AIMessage(
                    content=(
                        f"Room {room_name} has no non-structural assets "
                        "eligible for placement."
                    )
                )
            ],
            "active_room_for_trajectory": room_name,
            "pending_place_assets": [],
            "current_room_completed": True,
        }

    pending_assets = state.get("pending_place_assets")
    if pending_assets is None:
        pending_assets = expected_assets
    else:
        expected_keys = {
            str(asset.get("asset_key"))
            for asset in expected_assets
            if asset.get("asset_key") is not None
        }
        pending_assets = [
            asset
            for asset in pending_assets
            if str(asset.get("asset_key")) in expected_keys
        ]

    asset_registry = _build_asset_name_usd_registry(placement_room_assets)
    set_asset_name_usd_path_registry(asset_registry)
    print(f"[Placement] Registered {len(asset_registry)} asset name(s) for spawn_object resolution.")

    print(f"[Room Selection] Switching Isaac Sim context to room '{room_name}'")
    room_selected = select_room.invoke({"room_name": room_name})
    if not room_selected:
        warning = f"Failed to select room '{room_name}' in Isaac Sim; skipping placement for this room."
        print(f"Warning: {warning}")
        return {
            "objects_to_place": objects_to_place,
            "room_info": room_info,
            "place_trajectory": [AIMessage(content=warning)],
            "active_room_for_trajectory": room_name,
            "pending_place_assets": pending_assets,
            "current_room_completed": True,
        }

    phase = "adjustment" if state.get("need_adjustment", False) else "initial"
    reuse_classic = _is_reuse_classic_mode(state)
    placement_provider = _get_llm_provider(state.get("config_path"), "placement")
    print(f"   > LLM provider for placement: {placement_provider}")
    llm = build_llm(
        provider=placement_provider,
        temperature=0.0,
        timeout=180,
        **_tool_agent_llm_kwargs(placement_provider),
    )
    phase_system_prompt = (
        REPAIR_PLACE_AGENT_SYSTEM_PROMPT
        if phase == "adjustment"
        else INITIAL_PLACE_AGENT_SYSTEM_PROMPT
    )
    placement_tools = (
        AGENT_PLACEMENT_TOOLS
        if phase == "adjustment"
        else _build_initial_placement_tools()
    )
    agent = create_agent(
        model=llm,
        tools=placement_tools,
        system_prompt=phase_system_prompt,
        middleware=[AssetPlacementFoldingMiddleware()],
        checkpointer=InMemorySaver(),
    )

    placement_attempt_log = dict(state.get("placement_attempt_log") or {})
    room_trace: List[AnyMessage] = []
    seen_signatures: set[str] = set()
    user_prompt = _build_agent_user_prompt(
        {
            **state,
            "pending_place_assets": pending_assets,
        },
        objects_to_place,
        room_info,
    )
    result: Dict[str, Any] = {}

    print("[Agent Trace] Streaming intermediate steps...")
    for state_chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        },
        {"configurable": {"thread_id": f"room_{room_name}"}},
        stream_mode="values",
    ):
        result = state_chunk if isinstance(state_chunk, dict) else result
        if not isinstance(state_chunk, dict):
            continue

        messages = state_chunk.get("messages") or []
        new_messages = _dedupe_agent_messages(messages, seen_signatures)
        if not new_messages:
            continue

        _print_agent_stream_messages(new_messages)
        room_trace.extend(new_messages)
        placement_attempt_log.setdefault(room_name, []).append(
            {
                "phase": phase,
                "event_index": len(placement_attempt_log.get(room_name, [])) + 1,
                "messages": [_serialize_message(message) for message in new_messages],
            }
        )
        _persist_room_trajectory_snapshot(
            state,
            room_name,
            list(state.get("place_trajectory") or []) + room_trace,
            "latest",
        )

    if not room_trace and isinstance(result, dict):
        room_trace.extend(_dedupe_agent_messages(result.get("messages") or [], seen_signatures))

    updated_messages_before_final = list(state.get("place_trajectory") or []) + room_trace
    if reuse_classic and phase == "adjustment":
        # In reuse mode the initial scene already came from the classic USD,
        # so the repair pass should not re-derive completion from this pass's
        # spawn/set_pose trace.  Evaluator feedback decides whether another
        # repair pass is needed after we save and re-evaluate the USD copy.
        pending_assets = []
        room_completed = True
    else:
        placed_object_names = _placed_object_names_from_messages(updated_messages_before_final)
        pending_assets = _remaining_pending_assets(expected_assets, placed_object_names)
        room_completed = len(pending_assets) == 0

    final_content = (
        f"Completed placement agent run for room '{room_name}'."
        if room_completed
        else f"Placement agent run for room '{room_name}' ended with {len(pending_assets)} asset(s) still pending."
    )
    final_message = AIMessage(content=final_content)
    room_trace.append(final_message)
    updated_messages = list(state.get("place_trajectory") or []) + room_trace
    _persist_room_trajectory_snapshot(state, room_name, updated_messages, "latest")

    print(f"[Cleanup] Checking for unresolved objects in room '{room_name}'...")
    _cleanup_unresolved_objects(room_name)

    if save_stage():
        print("Stage save completed successfully after placement.")
    else:
        print("Warning: Stage save failed after placement.")

    print(f"[Placement] Current room: {room_name}")
    print(f"[Placement] Phase: {phase}")
    print(f"[Placement] Streamed messages captured: {len(room_trace)}")
    print(f"[Placement] Pending assets: {len(pending_assets)}")
    placed_rooms = list(state.get("placed_rooms") or [])
    if (
        phase == "adjustment"
        and room_completed
        and int(state.get("retry_count", 0) or 0) >= int(state.get("max_retry", 1) or 1)
        and room_name not in placed_rooms
    ):
        # The current graph is configured for a single full evaluator pass.
        # After the one repair pass, mark the room done and route directly back
        # to room selection instead of invoking a second max-retry evaluator
        # stub.
        placed_rooms.append(room_name)
    return {
        "objects_to_place": objects_to_place,
        "room_info": room_info,
        "place_trajectory": updated_messages,
        "active_room_for_trajectory": room_name,
        "placement_attempt_log": placement_attempt_log,
        "pending_place_assets": pending_assets,
        "current_room_completed": room_completed,
        "placed_rooms": placed_rooms,
    }


def should_continue(state: GraphAgentState):
    if state.get("current_room_completed", False):
        if int(state.get("retry_count", 0) or 0) >= int(state.get("max_retry", 1) or 1):
            return "select_room"
        return "evaluate"
    return "place"


_HARD_EVALUATION_TYPES = {
    "missing_object",
    "hard_collision",
    "out_of_room",
    "unsupported",
    "blocked_access",
}


def _room_spec(state: GraphAgentState, room_name: str) -> Dict[str, Any]:
    for room in (state.get("scene_blueprint") or {}).get("rooms", []):
        if isinstance(room, dict) and room.get("name") == room_name:
            return room
    return {}


def _expected_layout_records(
    state: GraphAgentState,
    room_name: str,
) -> List[Dict[str, Any]]:
    room_assets = (state.get("retrieved_assets") or {}).get(room_name, {})
    pending_records = _pending_assets_from_room_assets(room_assets)
    records = []
    for pending in pending_records:
        info = room_assets.get(pending.get("asset_key"), {})
        if not isinstance(info, dict):
            info = {}
        records.append(
            {
                "object": pending.get("object_name"),
                "asset_key": pending.get("asset_key"),
                "role": info.get("category") or info.get("object_type") or "object",
                "task_critical": info.get("object_type") == "explicit",
                "placement_hint": _as_optional_string(info.get("placement_hint")),
                "group": (
                    info.get("placement_group_key")
                    or info.get("object_type")
                    or info.get("category")
                ),
            }
        )
    return records


def _scene_object_map(scan_result: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(scan_result, list):
        return {}
    objects: Dict[str, Dict[str, Any]] = {}
    for item in scan_result:
        if not isinstance(item, dict):
            continue
        name = _as_optional_string(item.get("name"))
        if name is None:
            continue
        objects[name] = {
            "name": name,
            "position": item.get("position"),
            "rotation": item.get("rotation"),
            "bbox": item.get("bbox"),
        }
    return objects


def _actual_layout_records(
    placement_log: Dict[str, Any],
    scan_result: Any,
) -> List[Dict[str, Any]]:
    scan_objects = _scene_object_map(scan_result)
    records: Dict[str, Dict[str, Any]] = {}
    for item in placement_log.get("objects", []):
        if not isinstance(item, dict):
            continue
        name = _as_optional_string(item.get("name"))
        if name is None:
            continue
        live = scan_objects.get(name, {})
        records[name] = {
            "object": name,
            "position": live.get("position", item.get("position")),
            "rotation": live.get("rotation", item.get("rotation")),
            "bbox": live.get("bbox"),
            "placement_type": item.get("placement_type"),
            "support_object": item.get("support_object"),
            "surface": item.get("surface"),
            "orientation": item.get("orientation"),
            "semantic_location": item.get("semantic_location"),
            "mounted_location": item.get("mounted_location"),
            "gravity_disabled": item.get("gravity_disabled"),
            "kinematic_enabled": item.get("kinematic_enabled"),
            "source": item.get("source"),
        }

    # Keep live room objects that were not recoverable from the placement trace.
    for name, live in scan_objects.items():
        if name in records:
            continue
        lowered = name.lower()
        if (
            lowered in {"floor", "ceiling", "door", "groundplane"}
            or lowered.startswith("wall_")
            or lowered.startswith("door")
        ):
            continue
        records[name] = {
            "object": name,
            "position": live.get("position"),
            "rotation": live.get("rotation"),
            "bbox": live.get("bbox"),
            "placement_type": None,
            "support_object": None,
            "surface": None,
            "orientation": None,
            "semantic_location": None,
            "mounted_location": None,
            "gravity_disabled": None,
            "kinematic_enabled": None,
            "source": "scan_scene",
        }
    return sorted(records.values(), key=lambda item: item["object"])


_COORDINATE_KEYS = {"position", "rotation", "bbox"}
# Additional keys to strip from semantic records sent to the VLM.
_SEMANTIC_STRIP_KEYS = _COORDINATE_KEYS | {
    "gravity_disabled",
    "kinematic_enabled",
    "source",
}


def _semantic_object_records(actual_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip raw coordinates and internal plumbing fields from object records.

    LLMs do not reason effectively about numerical coordinates.  The Python
    deterministic passes (collision, support, boundary) still use the full
    ``actual_objects`` with coordinates, but the packet sent to the VLM only
    needs the semantic fields.  Additionally, fields like
    ``gravity_disabled``, ``kinematic_enabled``, and ``source`` are internal
    resolver state irrelevant to VLM judgment.
    """
    semantic_records = []
    for record in actual_objects:
        semantic_record = {
            key: value
            for key, value in record.items()
            if key not in _SEMANTIC_STRIP_KEYS
        }
        # Compact semantic_location: drop numeric margin, keep relational info.
        loc = semantic_record.get("semantic_location")
        if isinstance(loc, dict):
            semantic_record["semantic_location"] = {
                k: v for k, v in loc.items()
                if k not in {"margin", "distance"}
            }
        # Compact mounted_location: drop numeric margin, keep description.
        mloc = semantic_record.get("mounted_location")
        if isinstance(mloc, dict):
            semantic_record["mounted_location"] = {
                k: v for k, v in mloc.items()
                if k not in {"margin"}
            }
        semantic_records.append(semantic_record)
    return semantic_records


def _bbox_from_record(record: Optional[Dict[str, Any]]) -> Optional[Tuple[List[float], List[float]]]:
    if not isinstance(record, dict):
        return None
    bbox = record.get("bbox")
    if not isinstance(bbox, dict):
        return None
    bbox_min = _normalize_pose_vec3(bbox.get("min"))
    bbox_max = _normalize_pose_vec3(bbox.get("max"))
    if bbox_min is None or bbox_max is None:
        return None
    return bbox_min, bbox_max


def _normalize_placement_type(value: Any) -> Optional[str]:
    placement_type = _as_optional_string(value)
    if placement_type is None:
        return None
    normalized = placement_type.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "mounted":
        return "wall_mounted"
    return normalized


def _structural_mount_kind(support_object: Any) -> Optional[str]:
    support = _as_optional_string(support_object)
    if support is None:
        return None
    basename = support.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
    if basename.startswith("wall"):
        return "wall"
    if basename.startswith("ceiling"):
        return "ceiling"
    return None


def _xy_overlap(left: Tuple[List[float], List[float]], right: Tuple[List[float], List[float]]) -> bool:
    return (
        min(left[1][0], right[1][0]) > max(left[0][0], right[0][0])
        and min(left[1][1], right[1][1]) > max(left[0][1], right[0][1])
    )


def _support_findings(
    actual_objects: List[Dict[str, Any]],
    room_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    by_name = {item["object"]: item for item in actual_objects}
    room_bounds = room_summary.get("room_bounds") or {}
    room_min = _normalize_pose_vec3(room_bounds.get("min"))
    findings = []
    for item in actual_objects:
        object_name = item["object"]
        if is_nonphysical_floating_asset(object_name):
            continue
        placement_type = _normalize_placement_type(item.get("placement_type"))
        object_bbox = _bbox_from_record(item)
        if object_bbox is None:
            continue
        if placement_type == "floor" and room_min is not None:
            floor_gap = object_bbox[0][2] - room_min[2]
            if floor_gap > 0.15:
                findings.append(
                    {
                        "object": object_name,
                        "reason": "floor_object_not_resting_on_floor",
                        "gap": round(floor_gap, 4),
                    }
                )
            continue
        support_name = _as_optional_string(item.get("support_object"))
        if placement_type == "wall_mounted":
            is_structural_mount = _structural_mount_kind(support_name) is not None
            if not is_structural_mount:
                findings.append(
                    {
                        "object": object_name,
                        "support_object": support_name,
                        "reason": "invalid_wall_mounted_support",
                    }
                )
            if item.get("gravity_disabled") is not True:
                findings.append(
                    {
                        "object": object_name,
                        "support_object": support_name,
                        "reason": "wall_mounted_gravity_not_disabled",
                    }
                )
            if item.get("kinematic_enabled") is not True:
                findings.append(
                    {
                        "object": object_name,
                        "support_object": support_name,
                        "reason": "wall_mounted_not_kinematic",
                    }
                )
            continue
        if placement_type != "surface":
            continue

        # Wall-mounted objects (support_object starts with "wall_") are
        # retained as a compatibility path for older placement traces.
        is_wall_mounted = _structural_mount_kind(support_name) is not None
        support = by_name.get(support_name or "") if not is_wall_mounted else None
        support_bbox = _bbox_from_record(support) if not is_wall_mounted else None
        if is_wall_mounted:
            # Wall-mounted objects are supported by the wall itself; skip
            # the support-object-missing check and go directly to gap checks.
            pass
        elif support_name is None or support_bbox is None:
            findings.append(
                {
                    "object": object_name,
                    "support_object": support_name,
                    "reason": "support_object_missing",
                }
            )
            continue
        if item.get("surface") == "up" and support_bbox is not None:
            vertical_gap = abs(object_bbox[0][2] - support_bbox[1][2])
            if vertical_gap > 0.15 or not _xy_overlap(object_bbox, support_bbox):
                findings.append(
                    {
                        "object": object_name,
                        "support_object": support_name,
                        "reason": "invalid_top_surface_contact",
                        "vertical_gap": round(vertical_gap, 4),
                    }
                )
    return findings


_SPATIAL_SUPPORT_SURFACE_MIN_XY_M2 = 0.04
_SPATIAL_MIN_SURFACE_XY_RATIO = 0.15
_SPATIAL_SUPPORT_Z_TOLERANCE_M = 0.08
_SPATIAL_NEAR_THRESHOLD_M = 1.5
_SPATIAL_WALL_THRESHOLD_M = 0.15
_SPATIAL_MAX_RELATIONS_PER_KIND = 60
_SPATIAL_RELATION_IGNORE_PATTERNS = (
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "room_light",
    "ceiling_light",
    "wall_light",
    "light",
    "switch",
    "outlet",
    "thermostat",
    "painting",
    "picture",
)


def _spatial_relation_name_ignored(name: Any) -> bool:
    normalized = str(name or "").strip().lower().replace(" ", "_")
    if not normalized:
        return True
    if is_nonphysical_floating_asset(normalized):
        return True
    if is_collision_evaluation_exempt_asset(normalized):
        return True
    return any(pattern in normalized for pattern in _SPATIAL_RELATION_IGNORE_PATTERNS)


def _bounds_dict_from_bbox(
    bbox: Optional[Tuple[List[float], List[float]]],
) -> Optional[Dict[str, float]]:
    if bbox is None:
        return None
    bbox_min, bbox_max = bbox
    try:
        min_x, min_y, min_z = (float(bbox_min[i]) for i in range(3))
        max_x, max_y, max_z = (float(bbox_max[i]) for i in range(3))
    except (TypeError, ValueError, IndexError):
        return None
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    if size_x < 0 or size_y < 0 or size_z < 0:
        return None
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z,
        "center_x": (min_x + max_x) * 0.5,
        "center_y": (min_y + max_y) * 0.5,
        "center_z": (min_z + max_z) * 0.5,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
    }


def _room_bounds_dict(room_summary: Dict[str, Any]) -> Optional[Dict[str, float]]:
    room_bounds = room_summary.get("room_bounds") if isinstance(room_summary, dict) else None
    if not isinstance(room_bounds, dict):
        return None
    bbox_min = _normalize_pose_vec3(room_bounds.get("min"))
    bbox_max = _normalize_pose_vec3(room_bounds.get("max"))
    return _bounds_dict_from_bbox((bbox_min, bbox_max)) if bbox_min and bbox_max else None


def _bounds_xy_footprint(bounds: Dict[str, float]) -> float:
    return max(float(bounds.get("size_x", 0.0)) * float(bounds.get("size_y", 0.0)), 0.0)


def _bounds_xy_overlap_ratio(
    inner: Dict[str, float],
    outer: Dict[str, float],
) -> float:
    overlap_x = max(
        0.0,
        min(float(inner["max_x"]), float(outer["max_x"]))
        - max(float(inner["min_x"]), float(outer["min_x"])),
    )
    overlap_y = max(
        0.0,
        min(float(inner["max_y"]), float(outer["max_y"]))
        - max(float(inner["min_y"]), float(outer["min_y"])),
    )
    footprint = _bounds_xy_footprint(inner)
    if footprint < 1e-8:
        return 0.0
    return (overlap_x * overlap_y) / footprint


def _bounds_xy_contained(
    inner: Dict[str, float],
    outer: Dict[str, float],
    margin: float = 0.05,
) -> bool:
    return (
        float(inner["min_x"]) + margin >= float(outer["min_x"])
        and float(inner["max_x"]) - margin <= float(outer["max_x"])
        and float(inner["min_y"]) + margin >= float(outer["min_y"])
        and float(inner["max_y"]) - margin <= float(outer["max_y"])
    )


def _surface_height_label(height_m: float) -> str:
    """Coarse human-facing support-height label used by the evaluator LLM."""
    if height_m <= 0.15:
        return "floor_level"
    if height_m <= 0.45:
        return "low"
    if height_m <= 0.75:
        return "seat_or_low_table_height"
    if height_m <= 1.15:
        return "table_or_counter_height"
    if height_m <= 1.55:
        return "chest_height"
    if height_m <= 1.9:
        return "high"
    return "very_high"


def _relation_priority(relation: Dict[str, Any]) -> Tuple[int, str, str]:
    height_label = str(relation.get("support_height_label") or "")
    height_priority = {
        "very_high": 0,
        "high": 1,
        "chest_height": 2,
        "table_or_counter_height": 3,
        "seat_or_low_table_height": 4,
        "low": 5,
        "floor_level": 6,
    }.get(height_label, 7)
    return (
        height_priority,
        str(relation.get("object") or ""),
        str(relation.get("support") or relation.get("container") or relation.get("neighbor") or ""),
    )


def _spatial_relations(
    actual_objects: List[Dict[str, Any]],
    room_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract compact, semantic placement relations for LLM evaluation.

    This mirrors the commonsense evaluator's ``extract_spatial_relations`` but
    uses the graph evaluator's live room-summary/placement-record schema.  The
    output intentionally avoids raw XY coordinates; it keeps support/containment
    relations and coarse height labels so the evaluator can catch cases such as
    a tissue dispenser on a very high cabinet or a can on an implausible appliance.
    """
    room_summary = room_summary or {}
    room_bounds = _room_bounds_dict(room_summary)
    floor_z = float(room_bounds.get("min_z", 0.0)) if room_bounds else 0.0

    relation_objects: List[Dict[str, Any]] = []
    for item in actual_objects:
        if not isinstance(item, dict):
            continue
        name = _as_optional_string(item.get("object") or item.get("name"))
        if name is None or _spatial_relation_name_ignored(name):
            continue
        bounds = _bounds_dict_from_bbox(_bbox_from_record(item))
        if bounds is None:
            continue
        relation_objects.append({"name": name, "bounds": bounds, "record": item})

    support_relations: List[Dict[str, Any]] = []
    containment_relations: List[Dict[str, Any]] = []
    floor_relations: List[Dict[str, Any]] = []
    wall_relations: List[Dict[str, Any]] = []
    near_relations: List[Dict[str, Any]] = []
    centers: Dict[str, List[float]] = {}

    for obj in relation_objects:
        name = obj["name"]
        bounds = obj["bounds"]
        centers[name] = [
            float(bounds["center_x"]),
            float(bounds["center_y"]),
            float(bounds["center_z"]),
        ]

        bottom_height = float(bounds["min_z"]) - floor_z
        if abs(bottom_height) <= _SPATIAL_SUPPORT_Z_TOLERANCE_M:
            floor_relations.append(
                {
                    "type": "on_floor",
                    "object": name,
                    "object_height_m": round(float(bounds["size_z"]), 3),
                    "detail": f"{name} rests on floor",
                }
            )

        if room_bounds:
            for axis, wall_side in (("x", "min"), ("x", "max"), ("y", "min"), ("y", "max")):
                room_key = f"{wall_side}_{axis}"
                obj_key = f"{wall_side}_{axis}"
                dist = abs(float(bounds[obj_key]) - float(room_bounds[room_key]))
                if dist <= _SPATIAL_WALL_THRESHOLD_M:
                    axis_label = "X" if axis == "x" else "Y"
                    wall_relations.append(
                        {
                            "type": "against_wall",
                            "object": name,
                            "wall": f"{axis_label}-{wall_side}",
                            "distance_m": round(dist, 3),
                            "detail": (
                                f"{name} is against {axis_label}-{wall_side} wall "
                                f"(distance={dist:.2f}m)"
                            ),
                        }
                    )

    for i, obj_a in enumerate(relation_objects):
        name_a = obj_a["name"]
        bounds_a = obj_a["bounds"]
        for j, obj_b in enumerate(relation_objects):
            if j <= i:
                continue
            name_b = obj_b["name"]
            bounds_b = obj_b["bounds"]

            center_a = centers[name_a]
            center_b = centers[name_b]
            dist_xy = (
                (center_a[0] - center_b[0]) ** 2
                + (center_a[1] - center_b[1]) ** 2
            ) ** 0.5
            if dist_xy <= _SPATIAL_NEAR_THRESHOLD_M:
                near_relations.append(
                    {
                        "type": "near",
                        "object": name_a,
                        "neighbor": name_b,
                        "distance_m": round(dist_xy, 3),
                        "detail": f"{name_a} near {name_b} ({dist_xy:.2f}m)",
                    }
                )

            for top_obj, support_obj in ((obj_a, obj_b), (obj_b, obj_a)):
                top_name = top_obj["name"]
                support_name = support_obj["name"]
                top_bounds = top_obj["bounds"]
                support_bounds = support_obj["bounds"]
                support_footprint = _bounds_xy_footprint(support_bounds)
                if support_footprint < _SPATIAL_SUPPORT_SURFACE_MIN_XY_M2:
                    continue
                z_gap = abs(float(top_bounds["min_z"]) - float(support_bounds["max_z"]))
                if z_gap > _SPATIAL_SUPPORT_Z_TOLERANCE_M:
                    continue
                xy_overlap_ratio = _bounds_xy_overlap_ratio(top_bounds, support_bounds)
                if xy_overlap_ratio < _SPATIAL_MIN_SURFACE_XY_RATIO:
                    continue
                support_height_m = float(support_bounds["max_z"]) - floor_z
                height_label = _surface_height_label(support_height_m)
                support_relations.append(
                    {
                        "type": "on_top_of",
                        "object": top_name,
                        "support": support_name,
                        "z_gap_m": round(z_gap, 3),
                        "xy_overlap_ratio": round(xy_overlap_ratio, 3),
                        "support_height_m": round(support_height_m, 3),
                        "support_height_label": height_label,
                        "detail": (
                            f"{top_name} on {support_name} "
                            f"(support surface height={support_height_m:.2f}m, "
                            f"{height_label})"
                        ),
                        "evaluator_hint": (
                            "Judge whether this support object and height are "
                            "commonsense-plausible for the placed item."
                        ),
                    }
                )

            height_a = float(bounds_a.get("size_z", 0.0))
            height_b = float(bounds_b.get("size_z", 0.0))
            if height_b >= 0.3 and _bounds_xy_contained(bounds_a, bounds_b):
                if float(bounds_a["min_z"]) > float(bounds_b["min_z"]) and float(bounds_a["max_z"]) < float(bounds_b["max_z"]):
                    containment_relations.append(
                        {
                            "type": "inside",
                            "object": name_a,
                            "container": name_b,
                            "detail": f"{name_a} inside {name_b}",
                        }
                    )
            if height_a >= 0.3 and _bounds_xy_contained(bounds_b, bounds_a):
                if float(bounds_b["min_z"]) > float(bounds_a["min_z"]) and float(bounds_b["max_z"]) < float(bounds_a["max_z"]):
                    containment_relations.append(
                        {
                            "type": "inside",
                            "object": name_b,
                            "container": name_a,
                            "detail": f"{name_b} inside {name_a}",
                        }
                    )

    nearest_neighbors = {}
    for object_name, center in centers.items():
        distances = []
        for other_name, other_center in centers.items():
            if other_name == object_name:
                continue
            distance = (
                (center[0] - other_center[0]) ** 2
                + (center[1] - other_center[1]) ** 2
            ) ** 0.5
            distances.append({"object": other_name, "xy_distance": round(distance, 3)})
        nearest_neighbors[object_name] = sorted(distances, key=lambda item: item["xy_distance"])[:3]

    support_relations = sorted(support_relations, key=_relation_priority)
    containment_relations = sorted(containment_relations, key=_relation_priority)
    near_relations = sorted(near_relations, key=lambda rel: float(rel.get("distance_m", 999.0)))

    key_relation_details = [
        rel.get("detail")
        for rel in support_relations[: _SPATIAL_MAX_RELATIONS_PER_KIND]
        if rel.get("detail")
    ]
    key_relation_details.extend(
        rel.get("detail")
        for rel in containment_relations[: _SPATIAL_MAX_RELATIONS_PER_KIND]
        if rel.get("detail")
    )

    return {
        "support_relations": support_relations[: _SPATIAL_MAX_RELATIONS_PER_KIND],
        "containment_relations": containment_relations[: _SPATIAL_MAX_RELATIONS_PER_KIND],
        "floor_relations": floor_relations[: _SPATIAL_MAX_RELATIONS_PER_KIND],
        "wall_relations": wall_relations[: _SPATIAL_MAX_RELATIONS_PER_KIND],
        "near_relations": near_relations[: _SPATIAL_MAX_RELATIONS_PER_KIND],
        "nearest_neighbors": nearest_neighbors,
        "key_relation_details": key_relation_details[: _SPATIAL_MAX_RELATIONS_PER_KIND],
        "relation_count": (
            len(support_relations)
            + len(containment_relations)
            + len(floor_relations)
            + len(wall_relations)
            + len(near_relations)
        ),
        "note": (
            "Use support_relations and containment_relations as deterministic "
            "placement evidence for commonsense checks; height labels are "
            "relative to the room floor."
        ),
    }


def _is_surface_supported_collision(
    object_name: str,
    related_names: List[str],
    placement_info: Dict[str, Dict[str, Any]],
) -> bool:
    """Return True if a collision should be suppressed because the object is
    intentionally placed on or against its collision partner.

    Three cases are suppressed:

    1. **Surface placement** — object A is ``placement_type == "surface"`` and
       its ``support_object`` is B (e.g. burner on stove, cap on countertop).
       The two are *meant* to touch, so collision between them is expected.

    2. **Wall-mounted placement** — object A has ``support_object`` starting
       with ``"wall_"``.  Its collision mesh naturally intersects the wall
       collision mesh, which is not a defect.

    3. **Surface-adjacent placement** — object A is ``placement_type ==
       "surface"`` and one of its related_names B is the *support_object* of A.
       This catches the reciprocal direction (e.g. stove↔burner when the
       collision is reported from the support object's perspective).
    """
    info_a = placement_info.get(object_name)
    if not info_a:
        return False

    support_a = _as_optional_string(info_a.get("support_object"))
    ptype_a = _normalize_placement_type(info_a.get("placement_type"))

    # Case 2: wall-mounted → suppress collisions with the wall (already
    # handled by _is_structure_name in the collision detector, but the
    # wall collision mesh path may survive as a related_object).
    # related_names contains only non-structural objects, so an empty list is
    # required before treating a mounted collision as expected contact.
    if (
        ptype_a == "wall_mounted"
        and _structural_mount_kind(support_a) is not None
        and not related_names
    ):
        return True

    # Case 1 & 3: surface placement → suppress collision with support_object.
    if ptype_a == "surface" and support_a:
        if support_a in related_names:
            return True

    # Also check the reverse: one of the related_names is a surface-mounted
    # object whose support_object is *this* object.
    for rel_name in related_names:
        info_b = placement_info.get(rel_name)
        if not info_b:
            continue
        ptype_b = _normalize_placement_type(info_b.get("placement_type"))
        support_b = _as_optional_string(info_b.get("support_object"))
        if ptype_b == "surface" and support_b == object_name:
            return True

    return False


def _deterministic_findings(
    expected_layout: List[Dict[str, Any]],
    actual_objects: List[Dict[str, Any]],
    collisions: Dict[str, Any],
    room_summary: Dict[str, Any],
    support_findings: List[Dict[str, Any]],
    *,
    object_inventory_available: bool,
) -> Dict[str, Any]:
    actual_names = {item["object"] for item in actual_objects}
    missing = []
    if object_inventory_available:
        missing = [
            item["object"]
            for item in expected_layout
            if (
                item.get("object")
                and item["object"] not in actual_names
                and not is_nonphysical_floating_asset(item["object"])
            )
        ]

    # Build a placement-info lookup so we can filter out expected collisions
    # (surface-mounted objects touching their support, wall-mounted objects
    # touching the wall, etc.).
    placement_info: Dict[str, Dict[str, Any]] = {}
    for item in actual_objects:
        name = _as_optional_string(item.get("object"))
        if name:
            placement_info[name] = item

    raw_collisions = collisions.get("collisions") or []
    filtered_collisions = []
    for collision in raw_collisions:
        if not isinstance(collision, dict):
            filtered_collisions.append(collision)
            continue
        if collision.get("error"):
            filtered_collisions.append(collision)
            continue
        obj_name = _as_optional_string(collision.get("object"))
        related = _as_string_list(collision.get("related_objects"))
        if (
            is_collision_evaluation_exempt_asset(obj_name or "")
            or is_nonphysical_floating_asset(obj_name or "")
            or any(is_collision_evaluation_exempt_asset(name) for name in related)
            or any(is_nonphysical_floating_asset(name) for name in related)
        ):
            continue
        if _is_surface_supported_collision(obj_name, related, placement_info):
            continue
        filtered_collisions.append(collision)

    return {
        "missing": missing,
        "out_of_room": room_summary.get("out_of_room") or [],
        "collision_pairs": filtered_collisions,
        "unsupported": support_findings,
        "blocked_doors": room_summary.get("blocked_doors") or [],
        "occupancy_ratio": room_summary.get("occupancy_ratio"),
        "estimated_free_space_ratio": room_summary.get("estimated_free_space_ratio"),
        "free_space_connectivity": room_summary.get("free_space_connectivity"),
        "density_class": room_summary.get("density_class"),
    }


def _prior_evaluation_history(state: GraphAgentState) -> Dict[str, Any]:
    previous = state.get("evaluation_report") or {}
    previous_issues = previous.get("scoped_issues") or previous.get("issues") or []
    return {
        "retry": state.get("retry_count", 0),
        "previous_issue_signatures": [
            _issue_signature(issue)
            for issue in previous_issues
            if isinstance(issue, dict)
        ],
        "issue_repeat_counts": previous.get("issue_repeat_counts") or {},
        "objects_changed_last_repair": [
            action.get("object")
            for action in previous.get("object_actions", [])
            if isinstance(action, dict) and action.get("object")
        ],
    }


def _objects_from_room_summary(room_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Derive a scan-object list from ``summarize_room_layout`` output.

    This replaces the separate ``scan_scene`` pipe call.  The summary already
    enumerates every non-structural object in the room and includes bbox
    information, which is sufficient for ``_actual_layout_records`` to fill in
    live geometry from the deterministic pass.

    ``summarize_room_layout`` does not return position/rotation directly, so
    position is approximated from the bbox center when the bbox is available.
    """
    objects: List[Dict[str, Any]] = []
    for item in room_summary.get("objects", []):
        if not isinstance(item, dict):
            continue
        name = _as_optional_string(item.get("name"))
        if name is None:
            continue
        bbox = item.get("bbox")
        position = item.get("position")
        if position is None and isinstance(bbox, dict):
            bbox_min = _normalize_pose_vec3(bbox.get("min"))
            bbox_max = _normalize_pose_vec3(bbox.get("max"))
            if bbox_min is not None and bbox_max is not None:
                position = [
                    (bbox_min[0] + bbox_max[0]) * 0.5,
                    (bbox_min[1] + bbox_max[1]) * 0.5,
                    (bbox_min[2] + bbox_max[2]) * 0.5,
                ]
        objects.append({
            "name": name,
            "bbox": bbox,
            "position": position,
            "rotation": item.get("rotation"),
        })
    return objects


def _synthesize_placement_log_from_room_summary(
    *,
    room_name: str,
    latest_trace_file: Optional[str],
    placement_log: Dict[str, Any],
    room_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Build placement-log-like inventory for reused classic USD scenes.

    Reuse mode intentionally skips the graph placement agent, so no graph
    placement trace exists.  The deterministic room summary already enumerates
    the current non-structural USD objects; expose that inventory in the same
    shape expected by downstream packet builders and mark it as successful so
    the evaluator does not add a misleading ``incomplete_evidence`` warning.
    """
    if placement_log.get("success", False):
        return placement_log
    if not room_summary.get("success", False):
        return placement_log

    objects = []
    for item in room_summary.get("objects") or []:
        if not isinstance(item, dict):
            continue
        name = _as_optional_string(item.get("name"))
        if name is None:
            continue
        bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else None
        position = item.get("position")
        if position is None and bbox is not None:
            bbox_min = _normalize_pose_vec3(bbox.get("min"))
            bbox_max = _normalize_pose_vec3(bbox.get("max"))
            if bbox_min is not None and bbox_max is not None:
                position = [
                    (bbox_min[axis] + bbox_max[axis]) * 0.5
                    for axis in range(3)
                ]
        objects.append(
            {
                "name": name,
                "prim_path": item.get("prim_path") or f"/World/rooms/{room_name}/{name}",
                "room": room_name,
                "position": position,
                "rotation": item.get("rotation"),
                "placement_type": item.get("placement_type"),
                "support_object": item.get("support_object"),
                "surface": item.get("surface"),
                "source": "classic_usd_inventory",
            }
        )

    return {
        "success": True,
        "trace_file": latest_trace_file,
        "room_name": room_name,
        "objects": objects,
        "object_count": len(objects),
        "message": (
            "No graph placement trace was available; synthesized inventory "
            "from the copied classic USD room summary."
        ),
        "fallback_from_room_summary": True,
        "original_placement_log_status": {
            "success": bool(placement_log.get("success", False)),
            "message": placement_log.get("message"),
        },
    }


def _build_room_evaluation_packet(
    state: GraphAgentState,
    room_name: str,
    placement_log: Dict[str, Any],
    scan_result: Any,
    collisions: Dict[str, Any],
    room_summary: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Build the RoomEvaluationPacket sent to the VLM and the repair planner.

    ``scan_result`` is the object list derived from ``summarize_room_layout``
    (via ``_objects_from_room_summary``) rather than a separate ``scan_scene``
    pipe call.  It is still used for ``_actual_layout_records`` to compute
    support/boundary deterministic checks, but raw coordinates are stripped
    from the packet before reaching the VLM.
    """
    room_spec = _room_spec(state, room_name)
    expected_layout = _expected_layout_records(state, room_name)
    actual_objects = _actual_layout_records(placement_log, scan_result)
    support_findings = _support_findings(actual_objects, room_summary)
    placement_relations = _spatial_relations(actual_objects, room_summary)
    geometry_findings = _deterministic_findings(
        expected_layout,
        actual_objects,
        collisions,
        room_summary,
        support_findings,
        object_inventory_available=bool(
            placement_log.get("success", False)
            or room_summary.get("success", False)
        ),
    )
    semantic_objects = _semantic_object_records(actual_objects)

    expected_groups: Dict[str, List[str]] = {}
    for item in expected_layout:
        group = str(item.get("group") or "ungrouped")
        expected_groups.setdefault(group, []).append(str(item.get("object")))

    room_connections = []
    room_id = room_spec.get("id")
    for connection in (state.get("scene_blueprint") or {}).get("connections", []):
        if not isinstance(connection, dict):
            continue
        if room_id in {connection.get("room_id_a"), connection.get("room_id_b")}:
            room_connections.append(connection)

    # Compact expected layout: drop asset_key (internal), keep role and hints.
    compact_expected = [
        {
            "object": item.get("object"),
            "role": item.get("role"),
            "task_critical": item.get("task_critical", False),
            "placement_hint": item.get("placement_hint"),
            "group": item.get("group"),
        }
        for item in expected_layout
    ]

    # Compact doors: keep structure info only, strip coordinates.
    compact_doors = []
    for door in (room_summary.get("doors") or []):
        if not isinstance(door, dict):
            continue
        compact_doors.append({
            k: v for k, v in door.items()
            if k not in {"position", "rotation", "bbox", "vertices"}
        })

    # Compact connections: strip coordinates.
    compact_connections = []
    for conn in room_connections:
        if not isinstance(conn, dict):
            continue
        compact_connections.append({
            k: v for k, v in conn.items()
            if k not in {"position", "vertices", "waypoints"}
        })

    # Compact geometry_findings: keep summary of free-space, drop grid details.
    compact_geometry = dict(geometry_findings)
    fsc = compact_geometry.get("free_space_connectivity")
    if isinstance(fsc, dict):
        compact_geometry["free_space_connectivity"] = {
            "status": fsc.get("status"),
            "largest_component_ratio": fsc.get("largest_component_ratio"),
        }

    packet = {
        "task": {
            "instruction": state.get("task_instruction"),
            "robot": state.get("robot_description"),
        },
        "room": {
            "name": room_name,
            "role": room_spec.get("name", room_name),
            "overview": (
                ((state.get("retrieved_assets") or {}).get(room_name, {}) or {}).get(
                    "room_overview"
                )
            ),
            "doors": compact_doors,
            "connections": compact_connections,
        },
        "expected_layout": compact_expected,
        "actual_objects": semantic_objects,
        "placement_relations": placement_relations,
        "geometry_findings": compact_geometry,
        "expected_groups": expected_groups,
        "history": _prior_evaluation_history(state),
        "evidence_status": {
            "placement_log": {
                "success": bool(placement_log.get("success", False)),
                "message": placement_log.get("message"),
                "object_count": placement_log.get("object_count"),
            },
            "collisions": {
                "success": bool(collisions.get("success", False)),
                "message": collisions.get("message"),
                "objects_checked": collisions.get("objects_checked"),
                "collision_count": len(collisions.get("collisions") or []),
                "collision_pairs": [
                    {
                        "object": c.get("object"),
                        "related_objects": c.get("related_objects"),
                        "severity": c.get("severity"),
                    }
                    for c in (collisions.get("collisions") or [])
                    if isinstance(c, dict) and not c.get("error")
                ],
            },
            "room_summary": {
                "success": bool(room_summary.get("success", False)),
                "message": room_summary.get("message"),
            },
            "placement_relations": {
                "relation_count": placement_relations.get("relation_count"),
                "support_relation_count": len(placement_relations.get("support_relations") or []),
                "containment_relation_count": len(placement_relations.get("containment_relations") or []),
                "message": placement_relations.get("note"),
            },
        },
    }

    suspects = set(collisions.get("suspect_objects") or [])
    suspects.update(geometry_findings["missing"])
    suspects.update(
        item.get("object")
        for item in geometry_findings["out_of_room"]
        if isinstance(item, dict) and item.get("object")
    )
    suspects.update(
        item.get("object")
        for item in support_findings
        if isinstance(item, dict) and item.get("object")
    )
    # Add relation-focused objects so the evaluator gets close-up evidence for
    # commonsense support issues even when there is no hard collision/support
    # failure (e.g. a tissue dispenser on a very high cabinet).
    for relation in placement_relations.get("support_relations") or []:
        if not isinstance(relation, dict):
            continue
        height_label = str(relation.get("support_height_label") or "")
        if height_label in {"chest_height", "high", "very_high"}:
            if relation.get("object"):
                suspects.add(str(relation["object"]))
            if relation.get("support"):
                suspects.add(str(relation["support"]))
    return packet, sorted(str(item) for item in suspects if item)


def _evaluation_image_records(view_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    for view in view_result.get("views", []):
        if not isinstance(view, dict):
            continue
        path = view.get("path")
        if not isinstance(path, str) or Path(path).suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            continue
        records.append(
            {
                "path": path,
                "name": view.get("name"),
                "type": view.get("type"),
                "target_object": view.get("target_object"),
            }
        )
    return records


def _coerce_room_evaluation(result: Any) -> Optional[RoomEvaluation]:
    if isinstance(result, RoomEvaluation):
        return result
    if not isinstance(result, dict):
        return None
    parsed = result.get("parsed")
    if isinstance(parsed, RoomEvaluation):
        return parsed
    if isinstance(parsed, dict):
        try:
            return RoomEvaluation.model_validate(parsed)
        except ValidationError:
            return None
    raw = result.get("raw")
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        try:
            return RoomEvaluation.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError):
            return None
    return None


def _run_room_vlm_evaluation(
    llm: Any,
    room_name: str,
    packet: Dict[str, Any],
    image_records: List[Dict[str, Any]],
) -> Tuple[Optional[RoomEvaluation], AIMessage]:
    packet_text = _truncate_text(
        json.dumps(packet, ensure_ascii=False, indent=2),
        max_chars=32000,
    )
    format_instructions = _json_schema_prompt(
        RoomEvaluation,
        "RoomEvaluation",
    )
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Evaluate room '{room_name}'.\n\n"
                "RoomEvaluationPacket:\n"
                f"{packet_text}\n\n"
                "Evaluate across all supplied views.\n\n"
                f"{format_instructions}"
            ),
        }
    ]
    readable_images = []
    for index, record in enumerate(image_records[:6], start=1):
        try:
            data_url = _image_file_to_data_url(record["path"])
        except Exception:
            continue
        metadata = (
            f"Image {index}: view={record.get('name') or 'unknown'}, "
            f"type={record.get('type') or 'unknown'}, "
            f"target={record.get('target_object') or 'room'}"
        )
        content.append({"type": "text", "text": metadata})
        content.append({"type": "image_url", "image_url": {"url": data_url}})
        readable_images.append(metadata)

    evaluator = llm.with_structured_output(
        RoomEvaluation,
        method="json_mode",
        include_raw=True,
    )
    try:
        result = evaluator.invoke(
            [
                SystemMessage(content=EVALUATOR_ROOM_VLM_SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]
        )
        evaluation = _coerce_room_evaluation(result)
        raw = result.get("raw") if isinstance(result, dict) else None
        raw_content = getattr(raw, "content", None)
        parsing_error = (
            result.get("parsing_error")
            if isinstance(result, dict)
            else None
        )
        if evaluation is None and parsing_error is not None:
            raw_content = (
                f"VLM structured output parsing failed: {parsing_error}\n"
                f"Raw output: {raw_content}"
            )
    except Exception as exc:
        evaluation = None
        raw_content = f"VLM evaluation failed: {exc}"

    trace_message = AIMessage(
        content=(
            "[Room VLM evaluation]\n"
            f"images={len(readable_images)}\n"
            + (
                json.dumps(evaluation.model_dump(), ensure_ascii=False)
                if evaluation is not None
                else str(raw_content or "No structured evaluation returned.")
            )
        )
    )
    return evaluation, trace_message


def _issue_signature(issue: Dict[str, Any]) -> str:
    issue_type = str(issue.get("type") or "unknown")
    objects = issue.get("objects")
    if not isinstance(objects, list):
        object_name = issue.get("object")
        related = issue.get("related_objects") or []
        objects = ([object_name] if object_name else []) + list(related)
    normalized_objects = sorted(str(item) for item in objects if item)
    return f"{issue_type}:{'|'.join(normalized_objects)}"


def _hard_issues_from_packet(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = packet.get("geometry_findings") or {}
    issues: List[Dict[str, Any]] = []
    counter = 0

    def append_issue(issue_type: str, objects: List[str], evidence: List[str]) -> None:
        nonlocal counter
        counter += 1
        issues.append(
            {
                "id": f"hard_{counter}",
                "scope": "object" if len(objects) <= 1 else "cluster",
                "type": issue_type,
                "severity": "blocking",
                "objects": objects,
                "zone": None,
                "evidence": evidence,
                "visual_observation": None,
                "confidence": 1.0,
            }
        )

    for object_name in findings.get("missing") or []:
        append_issue(
            "missing_object",
            [str(object_name)],
            ["Expected asset is absent from placement log and current scene scan."],
        )
    for collision in findings.get("collision_pairs") or []:
        if not isinstance(collision, dict):
            continue
        object_name = _as_optional_string(collision.get("object"))
        related = _as_string_list(collision.get("related_objects"))
        if (
            is_collision_evaluation_exempt_asset(object_name or "")
            or any(is_collision_evaluation_exempt_asset(name) for name in related)
        ):
            continue
        objects = ([object_name] if object_name else []) + related
        append_issue(
            "hard_collision",
            objects,
            [f"Collision report: {json.dumps(collision, ensure_ascii=False)}"],
        )
    for out_of_room in findings.get("out_of_room") or []:
        if not isinstance(out_of_room, dict):
            continue
        object_name = _as_optional_string(out_of_room.get("object"))
        append_issue(
            "out_of_room",
            [object_name] if object_name else [],
            [f"Room boundary report: {json.dumps(out_of_room, ensure_ascii=False)}"],
        )
    for unsupported in findings.get("unsupported") or []:
        if not isinstance(unsupported, dict):
            continue
        object_name = _as_optional_string(unsupported.get("object"))
        append_issue(
            "unsupported",
            [object_name] if object_name else [],
            [f"Support check: {json.dumps(unsupported, ensure_ascii=False)}"],
        )
    for door in findings.get("blocked_doors") or []:
        if not isinstance(door, dict):
            continue
        nearby = [
            str(item.get("object"))
            for item in door.get("nearby_objects", [])
            if isinstance(item, dict) and item.get("object")
        ]
        append_issue(
            "blocked_access",
            nearby,
            [f"Door clearance report: {json.dumps(door, ensure_ascii=False)}"],
        )
    return issues


def _fallback_repair_intent(issue: Dict[str, Any], index: int) -> Dict[str, Any]:
    issue_type = issue.get("type")
    strategy_by_type = {
        "missing_object": "complete_missing",
        "hard_collision": "relocate_object",
        "out_of_room": "relocate_object",
        "unsupported": "restore_support",
        "blocked_access": "clear_route",
        "wrong_orientation": "fix_orientation",
    }
    objects = _as_string_list(issue.get("objects"))
    return {
        "id": f"fallback_repair_{index}",
        "issue_ids": [str(issue.get("id"))] if issue.get("id") else [],
        "scope": issue.get("scope") if issue.get("scope") in {
            "room",
            "zone",
            "cluster",
            "object",
        } else "object",
        "strategy": strategy_by_type.get(str(issue_type), "rearrange_cluster"),
        "objective": (
            f"Repair {issue_type} for {', '.join(objects) or 'the affected room area'} "
            "while preserving unaffected placements."
        ),
        "target_objects": objects,
        "anchor_objects": [],
        "preserve_objects": [],
        "spatial_constraints": _as_string_list(issue.get("evidence"))[:3],
    }


def _merge_and_enforce_evaluation(
    state: GraphAgentState,
    vlm_evaluation: Optional[RoomEvaluation],
    packet: Dict[str, Any],
) -> Dict[str, Any]:
    if vlm_evaluation is None:
        payload = {
            "decision": "accept_with_warnings",
            "summary": "Visual evaluation was unavailable; geometry evidence was still evaluated.",
            "scorecard": [],
            "issues": [],
            "repair_intents": [],
        }
    else:
        payload = vlm_evaluation.model_dump()

    issues = [
        issue
        for issue in payload.get("issues", [])
        if isinstance(issue, dict)
    ]
    failed_evidence = [
        name
        for name, status in (packet.get("evidence_status") or {}).items()
        if isinstance(status, dict) and not status.get("success", False)
    ]
    if failed_evidence:
        issues.append(
            {
                "id": "incomplete_evidence",
                "scope": "room",
                "type": "incomplete_evidence",
                "severity": "warning",
                "objects": [],
                "zone": None,
                "evidence": [
                    "Unavailable evidence sources: " + ", ".join(failed_evidence)
                ],
                "visual_observation": None,
                "confidence": 1.0,
            }
        )
    signatures = {_issue_signature(issue) for issue in issues}
    hard_signatures = set()
    for hard_issue in _hard_issues_from_packet(packet):
        signature = _issue_signature(hard_issue)
        hard_signatures.add(signature)
        if signature not in signatures:
            issues.append(hard_issue)
            signatures.add(signature)

    previous_counts = (
        ((state.get("evaluation_report") or {}).get("issue_repeat_counts"))
        or {}
    )
    repeat_counts = {}
    for issue in issues:
        signature = _issue_signature(issue)
        repeat_counts[signature] = int(previous_counts.get(signature, 0)) + 1
        if (
            signature not in hard_signatures
            and repeat_counts[signature] >= 3
            and issue.get("severity") == "blocking"
        ):
            issue["severity"] = "warning"
            issue["evidence"] = _as_string_list(issue.get("evidence")) + [
                "Downgraded after three consecutive evaluator passes to avoid repair oscillation."
            ]

    blocking_issues = [
        issue for issue in issues if issue.get("severity") == "blocking"
    ]
    intents = [
        intent
        for intent in payload.get("repair_intents", [])
        if isinstance(intent, dict)
    ]
    covered_issue_ids = {
        issue_id
        for intent in intents
        for issue_id in _as_string_list(intent.get("issue_ids"))
    }
    for index, issue in enumerate(blocking_issues, start=1):
        issue_id = str(issue.get("id") or "")
        if issue_id and issue_id in covered_issue_ids:
            continue
        intents.append(_fallback_repair_intent(issue, index))

    if blocking_issues:
        decision = "retry"
    else:
        requested_decision = payload.get("decision", "pass")
        decision = (
            "accept_with_warnings"
            if requested_decision == "retry" or issues
            else requested_decision
        )
    if decision == "retry" and not intents:
        decision = "accept_with_warnings"
    return {
        "decision": decision,
        "summary": _as_string(
            payload.get("summary"),
            "Room placement evaluation completed.",
        ),
        "scorecard": payload.get("scorecard") or [],
        "issues": issues,
        "repair_intents": intents,
        "issue_repeat_counts": repeat_counts,
    }


def _coerce_resolved_repair_plan(result: Any) -> Optional[ResolvedRepairPlan]:
    if isinstance(result, ResolvedRepairPlan):
        return result
    if not isinstance(result, dict):
        return None
    structured = result.get("structured_response")
    if isinstance(structured, ResolvedRepairPlan):
        return structured
    if isinstance(structured, dict):
        try:
            return ResolvedRepairPlan.model_validate(structured)
        except ValidationError:
            return None
    return None


def _successful_resolves_from_messages(
    messages: List[AnyMessage],
) -> Dict[str, Dict[str, Any]]:
    pending: Dict[str, Dict[str, Any]] = {}
    successful: Dict[str, Dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if not isinstance(call, dict) or call.get("name") != "resolve_placement_intent":
                    continue
                call_id = str(call.get("id") or "")
                args = call.get("args") or {}
                intent = args.get("intent") if isinstance(args, dict) else None
                if call_id and isinstance(intent, dict):
                    pending[call_id] = intent
            continue
        if not isinstance(message, ToolMessage):
            continue
        call_id = str(getattr(message, "tool_call_id", "") or "")
        intent = pending.pop(call_id, None)
        if intent is None:
            continue
        result = _json_from_message_content(message.content)
        if result is None and isinstance(message.content, str):
            try:
                parsed = json.loads(message.content)
                result = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                result = None
        object_name = _as_optional_string(intent.get("object_name"))
        if object_name and isinstance(result, dict) and result.get("success"):
            successful[object_name] = {
                "intent": intent,
                "position": result.get("position"),
                "rotation": result.get("rotation"),
            }
    return successful


def _validate_resolved_repair_plan(
    plan: Optional[ResolvedRepairPlan],
    messages: List[AnyMessage],
    repair_intents: List[Dict[str, Any]],
) -> ResolvedRepairPlan:
    successful = _successful_resolves_from_messages(messages)
    source_by_object = {}
    preserve_by_intent = {}
    for intent in repair_intents:
        intent_id = str(intent.get("id") or "")
        preserve_by_intent[intent_id] = _as_string_list(intent.get("preserve_objects"))
        for object_name in _as_string_list(intent.get("target_objects")):
            source_by_object.setdefault(object_name, intent_id)

    operations = []
    represented = set()
    if plan is not None:
        for operation in sorted(plan.operations, key=lambda item: item.order):
            actual = successful.get(operation.object_name)
            represented.add(operation.object_name)
            if actual is None:
                operations.append(
                    operation.model_copy(
                        update={
                            "resolved_position": None,
                            "resolved_rotation": None,
                            "status": "unresolved",
                            "error": operation.error or "No successful resolver result was recorded.",
                        }
                    )
                )
                continue
            operations.append(
                operation.model_copy(
                    update={
                        "resolver_intent": actual["intent"],
                        "resolved_position": _normalize_pose_vec3(actual.get("position")),
                        "resolved_rotation": _normalize_pose_vec3(actual.get("rotation")),
                        "status": "resolved",
                        "error": None,
                    }
                )
            )

    for object_name, actual in successful.items():
        if object_name in represented or len(operations) >= 6:
            continue
        source_intent_id = source_by_object.get(object_name, "")
        operations.append(
            ResolvedRepairOperation(
                source_intent_id=source_intent_id,
                object_name=object_name,
                order=len(operations),
                resolver_intent=actual["intent"],
                resolved_position=_normalize_pose_vec3(actual.get("position")),
                resolved_rotation=_normalize_pose_vec3(actual.get("rotation")),
                preserve_objects=preserve_by_intent.get(source_intent_id, []),
                status="resolved",
            )
        )

    return ResolvedRepairPlan(
        summary=(
            plan.summary
            if plan is not None
            else "Repair planner returned no structured plan; recovered successful resolver calls."
        ),
        operations=operations[:6],
    )


def _run_repair_planner(
    state: GraphAgentState,
    room_name: str,
    packet: Dict[str, Any],
    evaluation: Dict[str, Any],
) -> Tuple[ResolvedRepairPlan, List[AnyMessage]]:
    provider = _get_llm_provider(state.get("config_path"), "evaluator")
    llm = build_llm(
        provider=provider,
        temperature=0,
        timeout=180,
        **_tool_agent_llm_kwargs(provider),
    )
    agent = create_agent(
        model=llm,
        tools=_build_evaluator_resolve_tools(room_name),
        system_prompt=REPAIR_PLANNER_SYSTEM_PROMPT,
        # The OpenAI-compatible endpoint rejects free-form evaluator tool
        # arguments when LangChain auto-selects strict ProviderStrategy.
        response_format=ToolStrategy(ResolvedRepairPlan),
        checkpointer=InMemorySaver(),
    )
    planner_input = {
        "room": packet.get("room"),
        "actual_objects": packet.get("actual_objects"),
        "geometry_findings": packet.get("geometry_findings"),
        "relations": packet.get("relations"),
        "repair_intents": evaluation.get("repair_intents"),
    }
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Produce an ordered resolved repair plan for this room.\n\n"
                        + _truncate_text(
                            json.dumps(planner_input, ensure_ascii=False, indent=2),
                            max_chars=26000,
                        )
                    )
                )
            ]
        },
        {
            "configurable": {
                "thread_id": (
                    f"repair_planner_{room_name}_retry_"
                    f"{state.get('retry_count', 0)}"
                )
            }
        },
    )
    messages = [
        message
        for message in result.get("messages", [])
        if isinstance(message, (AIMessage, HumanMessage, SystemMessage, ToolMessage))
    ]
    plan = _validate_resolved_repair_plan(
        _coerce_resolved_repair_plan(result),
        messages,
        evaluation.get("repair_intents") or [],
    )
    return plan, messages


def _legacy_report_from_layered_evaluation(
    evaluation: Dict[str, Any],
    repair_plan: ResolvedRepairPlan,
) -> Dict[str, Any]:
    intents_by_id = {
        str(intent.get("id")): intent
        for intent in evaluation.get("repair_intents", [])
        if isinstance(intent, dict)
    }
    issues_by_id = {
        str(issue.get("id")): issue
        for issue in evaluation.get("issues", [])
        if isinstance(issue, dict) and issue.get("id")
    }
    legacy_issues = []
    for issue in evaluation.get("issues", []):
        if not isinstance(issue, dict):
            continue
        objects = _as_string_list(issue.get("objects"))
        legacy_issues.append(
            {
                "type": _as_string(issue.get("type"), "unknown_issue"),
                "severity": (
                    "high" if issue.get("severity") == "blocking" else "medium"
                ),
                "object": objects[0] if objects else None,
                "related_objects": objects[1:],
                "evidence": {
                    "scope": issue.get("scope"),
                    "zone": issue.get("zone"),
                    "items": _as_string_list(issue.get("evidence")),
                    "confidence": issue.get("confidence"),
                },
                "blocking": issue.get("severity") == "blocking",
            }
        )

    object_actions = []
    covered_objects = set()
    for operation in repair_plan.operations:
        source_intent = intents_by_id.get(operation.source_intent_id, {})
        linked_issues = [
            issues_by_id[issue_id]
            for issue_id in _as_string_list(source_intent.get("issue_ids"))
            if issue_id in issues_by_id
        ]
        visual_observation = next(
            (
                _as_optional_string(issue.get("visual_observation"))
                for issue in linked_issues
                if _as_optional_string(issue.get("visual_observation"))
            ),
            None,
        )
        covered_objects.add(operation.object_name)
        object_actions.append(
            {
                "object": operation.object_name,
                "action": source_intent.get("strategy", "move_object"),
                "priority": "high",
                "reason": source_intent.get(
                    "objective",
                    "Execute the resolved room repair operation.",
                ),
                "target_hint": source_intent.get("objective"),
                "preferred_support_object": operation.resolver_intent.get("support_object"),
                "preferred_surface": operation.resolver_intent.get("surface"),
                "preferred_orientation": (
                    str(operation.resolver_intent.get("orientation"))
                    if operation.resolver_intent.get("orientation") is not None
                    else None
                ),
                "reference_objects": _as_string_list(
                    source_intent.get("anchor_objects")
                ),
                "distance_constraints": _as_string_list(
                    source_intent.get("spatial_constraints")
                ),
                "must_not_change": list(
                    dict.fromkeys(
                        operation.preserve_objects
                        + _as_string_list(source_intent.get("preserve_objects"))
                    )
                ),
                "repair_sequence": [
                    f"Apply repair operation order {operation.order}.",
                    (
                        "Use the evaluator-resolved pose directly."
                        if operation.status == "resolved"
                        else "Re-resolve this semantic intent before applying a pose."
                    ),
                ],
                "evidence": _as_string_list(source_intent.get("issue_ids")),
                "visual_observation": visual_observation,
                "resolved_intent": operation.resolver_intent or None,
                "resolved_position": operation.resolved_position,
                "resolved_rotation": operation.resolved_rotation,
                "resolution_status": operation.status,
                "resolution_error": operation.error,
            }
        )

    # Keep the high-level repair contract actionable even when the repair
    # planner cannot produce a valid resolver operation.
    for intent in evaluation.get("repair_intents", []):
        if not isinstance(intent, dict):
            continue
        linked_issues = [
            issues_by_id[issue_id]
            for issue_id in _as_string_list(intent.get("issue_ids"))
            if issue_id in issues_by_id
        ]
        visual_observation = next(
            (
                _as_optional_string(issue.get("visual_observation"))
                for issue in linked_issues
                if _as_optional_string(issue.get("visual_observation"))
            ),
            None,
        )
        for object_name in _as_string_list(intent.get("target_objects")):
            if object_name in covered_objects:
                continue
            covered_objects.add(object_name)
            object_actions.append(
                {
                    "object": object_name,
                    "action": intent.get("strategy", "move_object"),
                    "priority": "high",
                    "reason": intent.get(
                        "objective",
                        "Repair the evaluator-identified room layout issue.",
                    ),
                    "target_hint": intent.get("objective"),
                    "preferred_support_object": None,
                    "preferred_surface": None,
                    "preferred_orientation": None,
                    "reference_objects": _as_string_list(
                        intent.get("anchor_objects")
                    ),
                    "distance_constraints": _as_string_list(
                        intent.get("spatial_constraints")
                    ),
                    "must_not_change": _as_string_list(
                        intent.get("preserve_objects")
                    ),
                    "repair_sequence": [
                        "Interpret the high-level repair objective.",
                        "Inspect the current scene.",
                        "Resolve a valid semantic placement intent before set_pose.",
                    ],
                    "evidence": _as_string_list(intent.get("issue_ids")),
                    "visual_observation": visual_observation,
                    "resolved_intent": None,
                    "resolved_position": None,
                    "resolved_rotation": None,
                    "resolution_status": "unresolved",
                    "resolution_error": (
                        "Repair planner did not return a verified resolver operation."
                    ),
                }
            )

    needs_adjustment = evaluation.get("decision") == "retry"
    return {
        "needs_adjustment": needs_adjustment,
        "summary": evaluation.get("summary"),
        "issues": legacy_issues,
        "object_actions": object_actions,
        "accepted_with_warnings": evaluation.get("decision") == "accept_with_warnings",
        "decision": evaluation.get("decision"),
        "scorecard": evaluation.get("scorecard") or [],
        "scoped_issues": evaluation.get("issues") or [],
        "repair_intents": evaluation.get("repair_intents") or [],
        "resolved_repair_plan": repair_plan.model_dump(),
        "issue_repeat_counts": evaluation.get("issue_repeat_counts") or {},
    }


def _safe_tool_invoke(tool_instance: Any, args: Dict[str, Any], fallback: Any) -> Any:
    try:
        return tool_instance.invoke(args)
    except Exception as exc:
        if isinstance(fallback, dict):
            return {
                **fallback,
                "success": False,
                "message": f"{exc.__class__.__name__}: {exc}",
            }
        return fallback


def _raise_for_unsupported_evaluator_runtime(
    placement_log: Dict[str, Any],
    collisions: Dict[str, Any],
    room_summary: Dict[str, Any],
) -> None:
    results = (placement_log, collisions, room_summary)
    if not all(isinstance(result, dict) for result in results):
        return
    if any(result.get("success", False) for result in results):
        return

    messages = [
        str(result.get("message", "")).strip().lower()
        for result in results
    ]
    if messages and all(message == "false" for message in messages):
        raise RuntimeError(
            "Isaac Sim returned 'False' for every evaluator command. "
            "The running Isaac process does not expose the graph evaluator API; "
            "restart it with isaac_sim_app_graph.py."
        )


def _legacy_node_evaluator(state: GraphAgentState):
    """Evaluate the current room and either send feedback back to placement or advance rooms."""
    print("--- [Graph Node] Evaluator: assessing current room placement ---")
    room_name = state.get("current_room")
    retry_count = state.get("retry_count", 0)
    max_retry = state.get("max_retry", 3)

    if not room_name:
        return Command(update={"need_adjustment": False, "current_room_completed": False}, goto="select_room")

    if retry_count >= max_retry:
        print("Max retry attempts reached. Accepting current room and moving on.")
        placed_rooms = list(state.get("placed_rooms") or [])
        if room_name not in placed_rooms:
            placed_rooms.append(room_name)
        attempt_suffix = _evaluation_attempt_suffix(retry_count)
        # Preserve the most recent full evaluation report instead of
        # overwriting it with a blank stub.
        previous_report = None
        evaluate_dir = _evaluate_log_dir(state, room_name)
        if evaluate_dir:
            latest_report_path = os.path.join(evaluate_dir, "report_latest.json")
            if os.path.isfile(latest_report_path):
                try:
                    with open(latest_report_path, "r", encoding="utf-8") as f:
                        previous_report = json.load(f)
                except Exception:
                    previous_report = None
        if previous_report is not None:
            report = previous_report
            report["needs_adjustment"] = False
            report["accepted_with_warnings"] = True
            report.setdefault(
                "summary",
                f"Reached retry limit for room {room_name}; accepting current placement.",
            )
        else:
            report = {
                "needs_adjustment": False,
                "summary": f"Reached retry limit for room {room_name}; accepting current placement.",
                "issues": [],
                "object_actions": [],
                "accepted_with_warnings": True,
            }
        _persist_evaluation_report(state, room_name, report, f"{attempt_suffix}_report")
        # Do NOT overwrite report_latest – keep the most complete data.
        if evaluate_dir:
            latest_report_path = os.path.join(evaluate_dir, "report_latest.json")
            if not os.path.isfile(latest_report_path):
                _persist_evaluation_report(state, room_name, report, "report_latest")
        return Command(
            update={
                "need_adjustment": False,
                "current_room_completed": False,
                "retry_count": 0,
                "evaluation_report": report,
                "evaluation_trajectory": [
                    AIMessage(content=f"Retry limit reached for room '{room_name}'; accepted with warnings.")
                ],
                "placed_rooms": placed_rooms,
            },
            goto="select_room",
        )

    llm = build_llm(
        provider=_get_llm_provider(state.get("config_path"), "evaluator"),
        temperature=0,
        timeout=60,
    )
    evaluate_dir = _evaluate_log_dir(state, room_name)
    attempt_suffix = _evaluation_attempt_suffix(retry_count)
    latest_trace_file = _latest_placement_trace_file(state, room_name)
    eval_output_dir = os.path.join(evaluate_dir, "views", attempt_suffix) if evaluate_dir else None
    if eval_output_dir:
        os.makedirs(eval_output_dir, exist_ok=True)
    evidence_tools = _build_evaluator_evidence_tools(
        room_name=room_name,
        latest_trace_file=latest_trace_file,
        eval_output_dir=eval_output_dir,
    )
    resolve_tools = _build_evaluator_resolve_tools(room_name)

    evidence_agent = create_agent(
        model=llm,
        tools=evidence_tools,
        system_prompt=EVALUATOR_EVIDENCE_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    resolve_agent = create_agent(
        model=llm,
        tools=resolve_tools,
        system_prompt=EVALUATOR_RESOLVE_SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )

    evaluation_attempt_log = dict(state.get("evaluation_attempt_log") or {})
    prior_evaluation_trace = list(state.get("evaluation_trajectory") or [])

    evidence_trace: List[AnyMessage] = []
    evidence_seen_signatures: set[str] = set()
    evidence_result: Dict[str, Any] = {}
    evidence_user_prompt = _build_evaluator_user_prompt(
        state=state,
        room_name=room_name,
        latest_trace_file=latest_trace_file,
        eval_output_dir=eval_output_dir,
    )

    try:
        print("[Evaluator Evidence Phase] Streaming intermediate steps...")
        for state_chunk in evidence_agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": evidence_user_prompt,
                    }
                ]
            },
            {"configurable": {"thread_id": f"evaluator_evidence_{room_name}_retry_{retry_count}"}},
            stream_mode="values",
        ):
            evidence_result = state_chunk if isinstance(state_chunk, dict) else evidence_result
            if not isinstance(state_chunk, dict):
                continue

            messages = state_chunk.get("messages") or []
            new_messages = _dedupe_agent_messages(messages, evidence_seen_signatures)
            if not new_messages:
                continue

            _print_agent_stream_messages(messages)
            evidence_trace.extend(new_messages)
            evaluation_attempt_log.setdefault(room_name, []).append(
                {
                    "retry_count": retry_count,
                    "attempt": attempt_suffix,
                    "phase": "evidence",
                    "event_index": len(evaluation_attempt_log.get(room_name, [])) + 1,
                    "messages": [_serialize_message(message) for message in new_messages],
                }
            )
            _persist_evaluator_trajectory_snapshot(
                state,
                room_name,
                prior_evaluation_trace + evidence_trace,
                attempt_suffix,
            )
            _persist_evaluator_trajectory_snapshot(
                state,
                room_name,
                prior_evaluation_trace + evidence_trace,
                "latest",
            )

        if not evidence_trace and isinstance(evidence_result, dict):
            evidence_trace.extend(
                _dedupe_agent_messages(evidence_result.get("messages") or [], evidence_seen_signatures)
            )

        full_evaluator_trace = prior_evaluation_trace + evidence_trace
        evidence_final_message = AIMessage(content=f"Completed evaluator evidence phase for room '{room_name}'.")
        full_evaluator_trace.append(evidence_final_message)
        visual_messages = _run_evaluator_visual_pass(
            llm=llm,
            room_name=room_name,
            evaluator_trace=full_evaluator_trace,
            image_source_messages=evidence_trace,
        )
        if visual_messages:
            full_evaluator_trace.extend(visual_messages)
        _persist_evaluator_trajectory_snapshot(state, room_name, full_evaluator_trace, attempt_suffix)
        _persist_evaluator_trajectory_snapshot(state, room_name, full_evaluator_trace, "latest")

        resolve_trace: List[AnyMessage] = []
        resolve_seen_signatures: set[str] = {
            _message_signature(_serialize_message(message))
            for message in full_evaluator_trace
        }
        resolve_result: Dict[str, Any] = {}
        resolve_user_prompt = _build_evaluator_resolve_user_prompt(
            state=state,
            room_name=room_name,
        )

        print("[Evaluator Resolve Phase] Streaming intermediate steps...")
        for state_chunk in resolve_agent.stream(
            {
                "messages": [
                    *full_evaluator_trace,
                    {
                        "role": "user",
                        "content": resolve_user_prompt,
                    },
                ]
            },
            {"configurable": {"thread_id": f"evaluator_resolve_{room_name}_retry_{retry_count}"}},
            stream_mode="values",
        ):
            resolve_result = state_chunk if isinstance(state_chunk, dict) else resolve_result
            if not isinstance(state_chunk, dict):
                continue

            messages = state_chunk.get("messages") or []
            new_messages = _dedupe_agent_messages(messages, resolve_seen_signatures)
            if not new_messages:
                continue

            _print_agent_stream_messages(messages)
            resolve_trace.extend(new_messages)
            evaluation_attempt_log.setdefault(room_name, []).append(
                {
                    "retry_count": retry_count,
                    "attempt": attempt_suffix,
                    "phase": "resolve",
                    "event_index": len(evaluation_attempt_log.get(room_name, [])) + 1,
                    "messages": [_serialize_message(message) for message in new_messages],
                }
            )
            _persist_evaluator_trajectory_snapshot(
                state,
                room_name,
                full_evaluator_trace + resolve_trace,
                attempt_suffix,
            )
            _persist_evaluator_trajectory_snapshot(
                state,
                room_name,
                full_evaluator_trace + resolve_trace,
                "latest",
            )

        if not resolve_trace and isinstance(resolve_result, dict):
            resolve_trace.extend(
                _dedupe_agent_messages(resolve_result.get("messages") or [], resolve_seen_signatures)
            )

        full_evaluator_trace.extend(resolve_trace)
        resolve_final_message = AIMessage(content=f"Completed evaluator resolve phase for room '{room_name}'.")
        full_evaluator_trace.append(resolve_final_message)
        _persist_evaluator_trajectory_snapshot(state, room_name, full_evaluator_trace, attempt_suffix)
        _persist_evaluator_trajectory_snapshot(state, room_name, full_evaluator_trace, "latest")

        sanitized_report_trace = _strip_tool_calls_for_llm(full_evaluator_trace)
        report_messages: List[AnyMessage] = [
            SystemMessage(content=EVALUATOR_RESOLVE_SYSTEM_PROMPT),
            *sanitized_report_trace,
            HumanMessage(
                content=(
                    "Now produce the final structured placement evaluation report. "
                    "Use the PlacementEvaluation schema exactly and base the decision on the tool evidence above. "
                    "Do not rely on implicit image memory alone: first reuse or restate the important visual conclusions as compact "
                    "`visual_observation` strings inside the relevant object actions. "
                    "When `needs_adjustment` is true, include concrete `object_actions` that the placement "
                    "agent can execute in its next pass. For each blocking repair, define one exact `resolved_intent` "
                    "that matches the visual observation and then, whenever possible, call `resolve_placement_intent` using that same intent. "
                    "Only write `resolved_position` and `resolved_rotation` when they come directly from a successful resolve call. "
                    "If resolve fails, do not invent coordinates. Keep the rest of the action fields concise and only include low-level repair "
                    "context that supports the resolved pose."
                )
            ),
        ]
        evaluator = llm.with_structured_output(PlacementEvaluation, include_raw=True)
        try:
            report = _coerce_report_from_structured_output(evaluator.invoke(report_messages))
        except ValidationError as exc:
            print(f"[Evaluator] Structured report validation failed; falling back to normalized raw output: {exc}")
            report = _normalize_placement_evaluation_report({})
        _persist_evaluation_report(state, room_name, report, f"{attempt_suffix}_report")
        _persist_evaluation_report(state, room_name, report, "report_latest")

        if report["needs_adjustment"]:
            print("Evaluation suggests adjustments are needed. Returning to placement.")
            return Command(
                update={
                    "evaluation_report": report,
                    "evaluation_trajectory": full_evaluator_trace,
                    "evaluation_attempt_log": evaluation_attempt_log,
                    "need_adjustment": True,
                    "current_room_completed": False,
                    "retry_count": retry_count + 1,
                },
                goto="place",
            )

        print("Placement evaluated as satisfactory. Moving to next room.")
        placed_rooms = list(state.get("placed_rooms") or [])
        if room_name not in placed_rooms:
            placed_rooms.append(room_name)
        return Command(
            update={
                "evaluation_report": report,
                "evaluation_trajectory": full_evaluator_trace,
                "evaluation_attempt_log": evaluation_attempt_log,
                "need_adjustment": False,
                "current_room_completed": False,
                "retry_count": 0,
                "placed_rooms": placed_rooms,
            },
            goto="select_room",
        )
    finally:
        cleanup_result = raw_reset_trial_placements(room_name)
        if not cleanup_result.get("success", False) or cleanup_result.get("restored_count"):
            print(f"[Evaluator Cleanup] room={room_name} result={cleanup_result}")


def node_evaluator(state: GraphAgentState):
    """Run deterministic evidence collection, one VLM pass, and a repair planner."""
    print("--- [Graph Node] Evaluator: layered room evaluation ---")
    room_name = state.get("current_room")
    retry_count = state.get("retry_count", 0)
    max_retry = state.get("max_retry", 1)

    if not room_name:
        return Command(
            update={
                "need_adjustment": False,
                "current_room_completed": False,
            },
            goto="select_room",
        )

    if retry_count >= max_retry:
        print("Max retry attempts reached. Accepting current room with warnings.")
        placed_rooms = list(state.get("placed_rooms") or [])
        if room_name not in placed_rooms:
            placed_rooms.append(room_name)
        # Preserve the most recent full evaluation report instead of
        # overwriting it with a blank stub.  If the latest report file
        # exists, load it; otherwise fall back to a minimal wrapper.
        previous_report = None
        evaluate_dir = _evaluate_log_dir(state, room_name)
        if evaluate_dir:
            latest_report_path = os.path.join(evaluate_dir, "report_latest.json")
            if os.path.isfile(latest_report_path):
                try:
                    with open(latest_report_path, "r", encoding="utf-8") as f:
                        previous_report = json.load(f)
                except Exception:
                    previous_report = None
        if previous_report is not None:
            report = previous_report
            report["needs_adjustment"] = False
            report["accepted_with_warnings"] = True
            report["decision"] = "accept_with_warnings"
            report.setdefault(
                "summary",
                f"Reached retry limit for room {room_name}; "
                "accepting the latest placement with warnings.",
            )
        else:
            report = {
                "needs_adjustment": False,
                "summary": (
                    f"Reached retry limit for room {room_name}; "
                    "accepting the latest placement with warnings."
                ),
                "issues": [],
                "object_actions": [],
                "accepted_with_warnings": True,
                "decision": "accept_with_warnings",
                "scorecard": [],
                "scoped_issues": [],
                "repair_intents": [],
                "resolved_repair_plan": {
                    "summary": "Retry limit reached.",
                    "operations": [],
                },
                "issue_repeat_counts": {},
            }
        attempt_suffix = _evaluation_attempt_suffix(retry_count)
        _persist_evaluation_report(
            state,
            room_name,
            report,
            f"{attempt_suffix}_report",
        )
        # Do NOT overwrite report_latest – the existing file already
        # contains the most complete evaluation data from the last
        # attempt.  Only write it if the file doesn't exist yet.
        if evaluate_dir:
            latest_report_path = os.path.join(evaluate_dir, "report_latest.json")
            if not os.path.isfile(latest_report_path):
                _persist_evaluation_report(state, room_name, report, "report_latest")
        return Command(
            update={
                "need_adjustment": False,
                "current_room_completed": False,
                "retry_count": 0,
                "evaluation_report": report,
                "evaluation_trajectory": [
                    AIMessage(
                        content=(
                            f"Retry limit reached for room '{room_name}'; "
                            "accepted with warnings."
                        )
                    )
                ],
                "placed_rooms": placed_rooms,
            },
            goto="select_room",
        )

    evaluate_dir = _evaluate_log_dir(state, room_name)
    attempt_suffix = _evaluation_attempt_suffix(retry_count)
    latest_trace_file = _latest_placement_trace_file(state, room_name)
    view_output_dir = (
        os.path.join(evaluate_dir, "views", attempt_suffix)
        if evaluate_dir
        else None
    )
    if view_output_dir:
        os.makedirs(view_output_dir, exist_ok=True)

    placement_log = _safe_tool_invoke(
        raw_analyze_placement_log,
        {
            "trace_file": latest_trace_file or "",
            "room_name": room_name,
            "include_structure": False,
        },
        {"success": False, "objects": [], "object_count": 0},
    )
    collisions = _safe_tool_invoke(
        raw_check_scene_collisions,
        {"room_name": room_name},
        {
            "success": False,
            "collisions": [],
            "suspect_objects": [],
            "objects_checked": 0,
        },
    )
    room_summary = _safe_tool_invoke(
        raw_summarize_room_layout,
        {"room_name": room_name},
        {"success": False},
    )
    _raise_for_unsupported_evaluator_runtime(
        placement_log if isinstance(placement_log, dict) else {},
        collisions if isinstance(collisions, dict) else {},
        room_summary if isinstance(room_summary, dict) else {},
    )
    if _is_reuse_classic_mode(state) and isinstance(placement_log, dict) and isinstance(room_summary, dict):
        placement_log = _synthesize_placement_log_from_room_summary(
            room_name=room_name,
            latest_trace_file=latest_trace_file,
            placement_log=placement_log,
            room_summary=room_summary,
        )
    scan_result = _objects_from_room_summary(
        room_summary if isinstance(room_summary, dict) else {}
    )
    packet, suspect_objects = _build_room_evaluation_packet(
        state,
        room_name,
        placement_log if isinstance(placement_log, dict) else {},
        scan_result,
        collisions if isinstance(collisions, dict) else {},
        room_summary if isinstance(room_summary, dict) else {},
    )

    view_result: Dict[str, Any] = {
        "success": False,
        "views": [],
        "message": "evaluation output directory is unavailable",
    }
    if view_output_dir:
        view_result = _safe_tool_invoke(
            raw_capture_evaluation_views,
            {
                "room_name": room_name,
                "output_dir": view_output_dir,
                "suspect_objects": suspect_objects[:3],
            },
            {"success": False, "views": []},
        )
        if not isinstance(view_result, dict):
            view_result = {"success": False, "views": [], "message": str(view_result)}
    image_records = _evaluation_image_records(view_result)
    packet["images"] = [
        {
            "name": record.get("name"),
            "type": record.get("type"),
            "target_object": record.get("target_object"),
            "path": record.get("path"),
        }
        for record in image_records
    ]
    packet["evidence_status"]["views"] = {
        "success": bool(view_result.get("success", False) and image_records),
        "message": view_result.get("message"),
        "image_count": len(image_records),
    }
    _persist_evaluation_report(
        state,
        room_name,
        packet,
        f"{attempt_suffix}_packet",
    )
    _persist_evaluation_report(state, room_name, packet, "packet_latest")

    provider = _get_llm_provider(state.get("config_path"), "evaluator")
    try:
        llm = build_llm(
            provider=provider,
            temperature=0,
            timeout=180,
        )
        vlm_evaluation, vlm_trace = _run_room_vlm_evaluation(
            llm,
            room_name,
            packet,
            image_records,
        )
    except Exception as exc:
        vlm_evaluation = None
        vlm_trace = AIMessage(
            content=f"[Room VLM evaluation failed]\n{exc.__class__.__name__}: {exc}"
        )
    evaluation = _merge_and_enforce_evaluation(
        state,
        vlm_evaluation,
        packet,
    )

    repair_messages: List[AnyMessage] = []
    repair_plan = ResolvedRepairPlan(
        summary="No repair planning was required.",
        operations=[],
    )
    try:
        if evaluation["decision"] == "retry":
            try:
                repair_plan, repair_messages = _run_repair_planner(
                    state,
                    room_name,
                    packet,
                    evaluation,
                )
            except Exception as exc:
                repair_plan = ResolvedRepairPlan(
                    summary=(
                        "Repair planner failed; high-level repair intents "
                        f"remain available to placement: {exc}"
                    ),
                    operations=[],
                )
                repair_messages = [
                    AIMessage(
                        content=(
                            "[Repair planner failed]\n"
                            f"{exc.__class__.__name__}: {exc}"
                        )
                    )
                ]
        report = _legacy_report_from_layered_evaluation(
            evaluation,
            repair_plan,
        )
        report["evaluation_packet"] = {
            "attempt": attempt_suffix,
            "image_count": len(image_records),
            "suspect_objects": suspect_objects,
        }
        _persist_evaluation_report(
            state,
            room_name,
            report,
            f"{attempt_suffix}_report",
        )
        _persist_evaluation_report(state, room_name, report, "report_latest")

        compact_trace: List[AnyMessage] = [
            AIMessage(
                content=(
                    "[Deterministic evidence]\n"
                    + json.dumps(
                        packet.get("geometry_findings"),
                        ensure_ascii=False,
                    )
                )
            ),
            vlm_trace,
        ]
        compact_trace.extend(repair_messages)
        compact_trace.append(
            AIMessage(
                content=(
                    "[Resolved repair plan]\n"
                    + json.dumps(
                        repair_plan.model_dump(),
                        ensure_ascii=False,
                    )
                )
            )
        )
        _persist_evaluator_trajectory_snapshot(
            state,
            room_name,
            compact_trace,
            attempt_suffix,
        )
        _persist_evaluator_trajectory_snapshot(
            state,
            room_name,
            compact_trace,
            "latest",
        )

        evaluation_attempt_log = dict(
            state.get("evaluation_attempt_log") or {}
        )
        evaluation_attempt_log.setdefault(room_name, []).append(
            {
                "retry_count": retry_count,
                "attempt": attempt_suffix,
                "phase": "layered_evaluation",
                "event_index": len(
                    evaluation_attempt_log.get(room_name, [])
                ) + 1,
                "decision": evaluation["decision"],
                "image_count": len(image_records),
                "issue_signatures": [
                    _issue_signature(issue)
                    for issue in evaluation.get("issues", [])
                    if isinstance(issue, dict)
                ],
                "resolved_operation_count": len(repair_plan.operations),
            }
        )

        if report["needs_adjustment"]:
            print(
                "Layered evaluation requested repair. "
                f"Resolved operations: {len(repair_plan.operations)}"
            )
            return Command(
                update={
                    "evaluation_report": report,
                    "evaluation_trajectory": compact_trace,
                    "evaluation_attempt_log": evaluation_attempt_log,
                    "need_adjustment": True,
                    "current_room_completed": False,
                    "retry_count": retry_count + 1,
                },
                goto="place",
            )

        print(
            "Layered evaluation accepted the room "
            f"with decision={evaluation['decision']}."
        )
        placed_rooms = list(state.get("placed_rooms") or [])
        if room_name not in placed_rooms:
            placed_rooms.append(room_name)
        return Command(
            update={
                "evaluation_report": report,
                "evaluation_trajectory": compact_trace,
                "evaluation_attempt_log": evaluation_attempt_log,
                "need_adjustment": False,
                "current_room_completed": False,
                "retry_count": 0,
                "placed_rooms": placed_rooms,
            },
            goto="select_room",
        )
    finally:
        try:
            cleanup_result = raw_reset_trial_placements(room_name)
        except Exception as exc:
            print(
                f"[Evaluator Cleanup] room={room_name} failed: "
                f"{exc.__class__.__name__}: {exc}"
            )
        else:
            if (
                not cleanup_result.get("success", False)
                or cleanup_result.get("restored_count")
            ):
                print(
                    f"[Evaluator Cleanup] room={room_name} "
                    f"result={cleanup_result}"
                )


def _print_stream_event(event: Dict[str, Any]) -> None:
    for node_name, payload in event.items():
        if not isinstance(payload, dict):
            print(f"[Stream:{node_name}] {payload}")
            continue
        print(f"\n=== Stream Update: {node_name} ===")
        trajectory = payload.get("place_trajectory")
        if trajectory:
            for index, message in enumerate(trajectory, start=1):
                print(_format_message_block(index, message))
                print()
            continue
        evaluation_trajectory = payload.get("evaluation_trajectory")
        if evaluation_trajectory:
            for index, message in enumerate(evaluation_trajectory, start=1):
                print(_format_message_block(index, message))
                print()
            continue
        evaluation_report = payload.get("evaluation_report")
        if evaluation_report:
            print("Evaluation Report:")
            print(json.dumps(evaluation_report, ensure_ascii=False, indent=2))
            continue
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _interface_fields(state: GraphAgentState) -> Dict[str, Any]:
    runtime = state.get("runtime") or {}
    output_dir = state.get("output_dir") or runtime.get("output_dir")
    config_path = state.get("config_path") or runtime.get("config_path") or runtime.get("llm_config")
    return {
        "output_dir": output_dir,
        "config_path": config_path,
        "runtime": runtime,
    }


def node_architect_graph(state: GraphAgentState):
    """Call SceneCraft's architect node while preserving graph interface fields."""
    interface_update = _interface_fields(state)
    node_state = {
        **state,
        **interface_update,
    }
    result = node_architect(node_state)
    return {
        **interface_update,
        **result,
    }


def _legacy_node_retriever_graph(state: GraphAgentState):
    """Original branch retriever retained as a reference implementation."""
    interface_update = _interface_fields(state)
    node_state = {
        **state,
        **interface_update,
    }
    if not node_state.get("output_dir"):
        raise KeyError("output_dir")
    if not node_state.get("config_path"):
        raise KeyError("config_path")

    print("--- [Graph Node] Retriever: Fetching assets for each planned room ---")
    scene_blueprint = node_state["scene_blueprint"]
    config_path = node_state.get("config_path") or "run_config.json"
    output_dir = node_state["output_dir"]
    usd_path_settings = _load_usd_path_settings(config_path)
    store = USDAssetStore.from_config_file(config_path)

    all_asset_selections = []
    final_retrieved_assets: Dict[str, Dict[str, Any]] = {}
    all_explicit_candidates: Dict[str, Dict[str, Any]] = {}
    all_abstract_candidates: Dict[str, Dict[str, Any]] = {}

    os.makedirs(output_dir, exist_ok=True)

    for room in scene_blueprint["rooms"]:
        room_name = room["name"]
        explicit_objects = room["explicit_objects"]
        abstract_objects = room["abstract_objects"]

        print(f"\n   > Room '{room_name}':")
        print(f"     - {len(explicit_objects)} explicit objects (task-critical)")
        print(f"     - {len(abstract_objects)} abstract object groups (environmental)")

        final_retrieved_assets[room_name] = {}
        if "room_overview" in room:
            final_retrieved_assets[room_name]["room_overview"] = room["room_overview"]
        if "room_size" in room:
            final_retrieved_assets[room_name]["room_size"] = room["room_size"]
        if "size" in room:
            final_retrieved_assets[room_name]["room_size"] = room["size"]

        explicit_candidates = {}
        abstract_candidates = {}

        print("\n     [Retrieving Explicit Object Candidates]")
        for obj in explicit_objects:
            obj_name = obj["name"]
            results = store.search_with_score(obj_name, top_k=3)

            if len(results) == 0:
                print(f"       ⚠️  No assets found for '{obj_name}'")
                continue

            candidates = []
            for doc, score in results:
                asset_id = doc.metadata.get("asset_id", "N/A")
                dataset_category = doc.metadata.get("category", "unknown")
                if _is_structural_asset(asset_id, dataset_category):
                    print(
                        f"       ! Skipping structural asset '{asset_id}' "
                        f"(category: {dataset_category}) for explicit object "
                        f"'{obj_name}'"
                    )
                    continue
                candidates.append(
                    AssetCandidate(
                        asset_id=asset_id,
                        usd_path=doc.metadata.get("usd_path", ""),
                        caption=doc.page_content,
                        category=dataset_category,
                        score=score,
                    )
                )

            explicit_candidates[obj_name] = {
                "placement_hint": obj.get("placement_hint", ""),
                "candidates": candidates,
            }
            print(f"       '{obj_name}': Found {len(candidates)} candidates")

        print("\n     [Retrieving Abstract Object Candidates]")
        for idx, abstract_obj in enumerate(abstract_objects):
            category = abstract_obj["category"]
            description = abstract_obj["description"]
            category_key = f"{category}_{idx}"
            results = store.search_with_score(description, top_k=10)

            if len(results) == 0:
                print(f"       ⚠️  No assets found for '{category}' (instance {idx})")
                continue

            candidates = []
            for doc, score in results:
                asset_id = doc.metadata.get("asset_id", "N/A")
                dataset_category = doc.metadata.get("category", "unknown")
                if _is_structural_asset(asset_id, dataset_category):
                    print(
                        f"       ! Skipping structural asset '{asset_id}' "
                        f"(category: {dataset_category}) for abstract "
                        f"'{category_key}'"
                    )
                    continue
                candidates.append(
                    AssetCandidate(
                        asset_id=asset_id,
                        usd_path=doc.metadata.get("usd_path", ""),
                        caption=doc.page_content,
                        category=dataset_category,
                        score=score,
                    )
                )

            abstract_candidates[category_key] = {
                "category": category,
                "description": description,
                "placement_hint": abstract_obj["placement_hint"],
                "candidates": candidates,
            }
            print(f"       '{category}' (instance {idx}): Found {len(candidates)} candidates")

        all_explicit_candidates[room_name] = explicit_candidates
        all_abstract_candidates[room_name] = abstract_candidates

        print("\n     [LLM Asset Selection]")
        explicit_text = ""
        for obj_name, candidate_data in explicit_candidates.items():
            placement_hint = candidate_data.get("placement_hint", "")
            candidates = candidate_data.get("candidates", [])
            explicit_text += f"\n**{obj_name}** (select 1) - Placement hint: {placement_hint}\n"
            for i, cand in enumerate(candidates, 1):
                explicit_text += f"  {i}. [{cand.asset_id}] {cand.caption} (score: {cand.score:.3f})\n"

        abstract_text = ""
        for category_key, category_data in abstract_candidates.items():
            category = category_data["category"]
            desc = category_data["description"]
            candidates = category_data["candidates"]
            placement_hint = category_data.get("placement_hint", "")
            abstract_text += f"\n**{category_key}** ({category}) - \"{desc}\" (select 1-3 to keep):\n"
            abstract_text += f"Placement hint: {placement_hint}\n"
            for i, cand in enumerate(candidates, 1):
                abstract_text += f"  {i}. [{cand.asset_id}] {cand.caption} (score: {cand.score:.3f})\n"

        llm = build_llm(
            provider=_get_llm_provider(node_state.get("config_path"), "asset_selection"),
            temperature=0.7,
            timeout=60,
        )
        structured_llm = llm.with_structured_output(AssetSelectionResult)
        prompt = ChatPromptTemplate.from_template(ASSET_SELECTION_PROMPT)

        selection_result: AssetSelectionResult = (prompt | structured_llm).invoke(
            {
                "room_name": room_name,
                "task_instruction": node_state["task_instruction"],
                "explicit_candidates_text": explicit_text,
                "abstract_candidates_text": abstract_text,
            }
        )

        all_asset_selections.append(selection_result.model_dump())

        for sel in selection_result.explicit_selections:
            obj_name = sel.object_name
            selected_id = sel.selected_asset_id
            candidates = explicit_candidates.get(obj_name, {}).get("candidates", [])
            placement_hint = explicit_candidates.get(obj_name, {}).get("placement_hint", "")
            selected_asset = next((c for c in candidates if c.asset_id == selected_id), None)

            if selected_asset:
                final_retrieved_assets[room_name][obj_name] = {
                    "asset_id": selected_asset.asset_id,
                    "usd_path": _convert_usd_path(selected_asset.usd_path, usd_path_settings),
                    "caption": selected_asset.caption,
                    "category": selected_asset.category,
                    "room_name": room_name,
                    "score": selected_asset.score,
                    "object_type": "explicit",
                    "placement_hint": placement_hint,
                    "selection_reasoning": sel.reasoning,
                }
                print(f"       ✓ Explicit: {obj_name} -> {selected_id}")

        for sel in selection_result.abstract_selections:
            category_key = sel.category
            kept_ids = sel.kept_asset_ids
            category_data = abstract_candidates.get(category_key)
            if not category_data:
                try:
                    for key, data in abstract_candidates.items():
                        if all(any(c.asset_id == kid for c in data["candidates"]) for kid in kept_ids):
                            category_data = data
                            category_key = key
                            break
                except Exception:
                    print(f"       ⚠️  Warning: {category_key} not found in candidates")
                    continue

            if not category_data:
                print(f"       ⚠️  Warning: {category_key} not found in candidates")
                continue

            candidates = category_data["candidates"]
            placement_hint = category_data["placement_hint"]

            for i, asset_id in enumerate(kept_ids):
                asset = next((c for c in candidates if c.asset_id == asset_id), None)
                if asset:
                    key = f"{category_key}_{i}"
                    final_retrieved_assets[room_name][key] = {
                        "asset_id": asset.asset_id,
                        "usd_path": _convert_usd_path(asset.usd_path, usd_path_settings),
                        "caption": asset.caption,
                        "category": asset.category,
                        "room_name": room_name,
                        "score": asset.score,
                        "object_type": "abstract",
                        "abstract_description": sel.description,
                        "selection_reasoning": sel.reasoning,
                        "placement_hint": placement_hint,
                    }

            print(f"       ✓ Abstract: {category_key} -> kept {len(kept_ids)} items")

    with open(os.path.join(output_dir, "asset_candidates.json"), "w", encoding="utf-8") as f:
        candidates_dump = {
            "explicit": {
                room: {
                    k: [c.model_dump() for c in (v["candidates"] if isinstance(v, dict) else v)]
                    for k, v in candidates.items()
                }
                for room, candidates in all_explicit_candidates.items()
            },
            "abstract": {
                room: {
                    k: {
                        "category": v["category"],
                        "description": v["description"],
                        "placement_hint": v["placement_hint"],
                        "candidates": [c.model_dump() for c in v["candidates"]],
                    }
                    for k, v in candidates.items()
                }
                for room, candidates in all_abstract_candidates.items()
            },
        }
        json.dump(candidates_dump, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "llm_selections.json"), "w", encoding="utf-8") as f:
        json.dump(all_asset_selections, f, indent=2, ensure_ascii=False)

    final_retrieved_assets = {
        room_name: _filter_excluded_room_assets(room_assets)
        for room_name, room_assets in final_retrieved_assets.items()
    }

    with open(os.path.join(output_dir, "final_assets.json"), "w", encoding="utf-8") as f:
        json.dump(final_retrieved_assets, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RETRIEVED ASSETS:")
    print(json.dumps(final_retrieved_assets, indent=2, ensure_ascii=False))
    result = {"retrieved_assets": final_retrieved_assets}
    return {
        **interface_update,
        **result,
    }


def node_retriever_graph(state: GraphAgentState):
    """Run the latest main retriever while preserving graph interface fields."""
    interface_update = _interface_fields(state)
    node_state = {
        **state,
        **interface_update,
    }
    if not node_state.get("output_dir"):
        raise KeyError("output_dir")
    if not node_state.get("config_path"):
        raise KeyError("config_path")

    result = node_retriever_and_rooms_construction(node_state)
    return {
        **interface_update,
        **result,
    }


def build_graph(entry_point: str = "architect"):
    builder = StateGraph(GraphAgentState)
    builder.add_node("architect", node_architect_graph)
    builder.add_node("retriever", node_retriever_graph)
    builder.add_node("floor_plan", node_floor_plan_generator)
    builder.add_node("select_room", node_room_selector)
    builder.add_node("place", node_place_with_sub_agent)
    builder.add_node("evaluate", node_evaluator)

    if entry_point not in {"architect", "select_room"}:
        raise ValueError(f"Unsupported graph entry point: {entry_point}")
    builder.set_entry_point(entry_point)
    builder.add_edge("architect", "retriever")
    builder.add_edge("retriever", "floor_plan")
    builder.add_edge("floor_plan", "select_room")
    builder.add_conditional_edges(
        "place",
        should_continue,
        {"evaluate": "evaluate", "place": "place", "select_room": "select_room"},
    )
    return builder.compile()


def _initial_state(
    *,
    task_instruction: str,
    robot_desc: str,
    output_dir: str,
    config_path: str,
    runtime: Dict[str, Any],
    scene_blueprint: Optional[Dict[str, Any]] = None,
    retrieved_assets: Optional[Dict[str, Dict[str, Any]]] = None,
    floor_plan: Optional[Dict[str, Dict[str, Any]]] = None,
    reuse_classic_usd: bool = False,
    classic_usd_source: Optional[str] = None,
    classic_run_dir: Optional[str] = None,
    working_usd_path: Optional[str] = None,
) -> GraphAgentState:
    return {
        "task_instruction": task_instruction,
        "robot_description": robot_desc,
        "scene_blueprint": scene_blueprint,
        "retrieved_assets": retrieved_assets,
        "floor_plan": floor_plan,
        "current_room": None,
        "place_trajectory": [],
        "evaluation_trajectory": [],
        "objects_to_place": "",
        "room_info": "",
        "retry_count": 0,
        "max_retry": 1,
        "evaluation_report": None,
        "need_adjustment": False,
        "current_room_completed": False,
        "placed_rooms": [],
        "active_room_for_trajectory": None,
        "placement_attempt_log": {},
        "pending_place_assets": [],
        "evaluation_attempt_log": {},
        "placement_code": None,
        "error": None,
        "output_dir": output_dir,
        "config_path": config_path,
        "runtime": runtime,
        "reuse_classic_usd": reuse_classic_usd,
        "classic_usd_source": classic_usd_source,
        "classic_run_dir": classic_run_dir,
        "working_usd_path": working_usd_path,
        "initial_placement_reused": reuse_classic_usd,
    }


async def _run_graph_for_task(
    *,
    task_instruction: str,
    robot_desc: str,
    config_path: Path,
    base_output_dir: Path,
    llm_provider_selections: Dict[str, str],
    task_id: Optional[str] = None,
    task_batch: Optional[List[Dict[str, str]]] = None,
    reuse_classic_usd: bool = False,
    classic_usd_source: Optional[str] = None,
    classic_run_dir: Optional[str] = None,
    working_usd_path: Optional[str] = None,
) -> Dict[str, Any]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_suffix = _sanitize_path_component(task_id, fallback="task")
    run_dir_name = f"run_{run_timestamp}"
    if task_id is not None:
        run_dir_name += f"_task_{task_suffix}"

    run_output_dir = base_output_dir / run_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    reuse_seed: Dict[str, Any] = {}
    scene_blueprint: Optional[Dict[str, Any]] = None
    retrieved_assets: Optional[Dict[str, Dict[str, Any]]] = None
    floor_plan: Optional[Dict[str, Dict[str, Any]]] = None
    graph_entry_point = "architect"
    graph_flow = [
        "architect",
        "retriever",
        "floor_plan",
        "select_room",
        "place",
        "evaluate",
    ]
    if reuse_classic_usd:
        reuse_seed = _load_classic_reuse_artifacts(
            task_id=task_id,
            classic_usd_source_arg=classic_usd_source,
            classic_run_dir_arg=classic_run_dir,
            run_output_dir=run_output_dir,
        )
        classic_usd_source = reuse_seed["classic_usd_source"]
        classic_run_dir = reuse_seed["classic_run_dir"]
        scene_blueprint = reuse_seed["scene_blueprint"]
        retrieved_assets = reuse_seed["retrieved_assets"]
        floor_plan = _derive_floor_plan(scene_blueprint, retrieved_assets)
        graph_entry_point = "select_room"
        graph_flow = [
            "select_room",
            "evaluate",
            "place(repair_only_if_needed)",
        ]

    run_params = {
        "task": task_instruction,
        "task_id": task_id,
        "task_batch": task_batch or [],
        "output_dir_arg": str(base_output_dir),
        "base_output_dir": str(base_output_dir),
        "run_output_dir": str(run_output_dir),
        "robot_desc": robot_desc,
        "llm_config": str(config_path),
        "resolved_llm_config": str(config_path),
        "llm_provider_selections": llm_provider_selections,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entrypoint": Path(__file__).name,
        "isaac_runtime_entrypoint": "isaac_sim_app_graph.py",
        "graph_flow": graph_flow,
        "graph_entry_point": graph_entry_point,
        "reuse_classic_usd": bool(reuse_classic_usd),
        "classic_usd_source": classic_usd_source,
        "classic_run_dir": classic_run_dir,
        "working_usd_path": working_usd_path,
        "initial_placement_reused": bool(reuse_classic_usd),
    }
    with (run_output_dir / "run_params.json").open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2, ensure_ascii=False)

    print(f"Run output directory: {run_output_dir}")
    print(f"Run parameters logged to: {run_output_dir / 'run_params.json'}")

    timings: Dict[str, Any] = {
        "created_at": _timestamp_iso(),
        "run_output_dir": str(run_output_dir),
        "task_id": task_id,
        "task_batch_size": len(task_batch or []),
        "steps": {},
    }
    overall_start = time.perf_counter()
    _write_run_timings(run_output_dir, timings)

    graph = build_graph(entry_point=graph_entry_point)
    runtime = dict(run_params)
    runtime["output_dir"] = str(run_output_dir)
    init_state = _initial_state(
        task_instruction=task_instruction,
        robot_desc=robot_desc,
        output_dir=str(run_output_dir),
        config_path=str(config_path),
        runtime=runtime,
        scene_blueprint=scene_blueprint,
        retrieved_assets=retrieved_assets,
        floor_plan=floor_plan,
        reuse_classic_usd=bool(reuse_classic_usd),
        classic_usd_source=classic_usd_source,
        classic_run_dir=classic_run_dir,
        working_usd_path=working_usd_path,
    )

    previous_event_time = overall_start
    try:
        async for event in graph.astream(init_state, stream_mode="updates"):
            event_time = time.perf_counter()
            event_duration = event_time - previous_event_time
            previous_event_time = event_time
            for node_name in event:
                step = timings["steps"].setdefault(
                    node_name,
                    {"calls": 0, "duration_sec": 0.0, "status": "ok"},
                )
                step["calls"] += 1
                step["duration_sec"] = round(step["duration_sec"] + event_duration, 3)
            _write_run_timings(run_output_dir, timings)
            _print_stream_event(event)
    except Exception as exc:
        timings["finished_at"] = _timestamp_iso()
        timings["total_duration_sec"] = round(time.perf_counter() - overall_start, 3)
        timings["status"] = "failed"
        timings["error"] = f"{exc.__class__.__name__}: {exc}"
        _write_run_timings(run_output_dir, timings)
        raise
    else:
        timings["finished_at"] = _timestamp_iso()
        timings["total_duration_sec"] = round(time.perf_counter() - overall_start, 3)
        timings["status"] = "ok"
        _write_run_timings(run_output_dir, timings)

    return {
        "task_id": task_id or "",
        "task_batch_ids": [item.get("task_id", "") for item in (task_batch or [])],
        "task_batch_size": len(task_batch or []),
        "task_instruction": task_instruction,
        "run_output_dir": str(run_output_dir),
        "timings_path": str(run_output_dir / "timings.json"),
        "timecost": timings,
    }


async def _main_async(args: Any) -> None:
    config_path = Path(args.llm_config).resolve()
    configure_llm_manager(str(config_path))
    llm_provider_selections = _load_llm_provider_selections(str(config_path))
    base_output_dir = Path(args.output_dir).resolve()

    if args.reuse_classic_usd and args.task_json:
        raise ValueError(
            "--reuse-classic-usd is supported only with a single --task run. "
            "Use the outer pipeline with --scene-batch-size 1 for batch jobs."
        )
    if args.reuse_classic_usd and not args.task_id:
        raise ValueError("--task-id is required when --reuse-classic-usd is enabled")

    if args.task_json:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be >= 1")

        task_json_path = Path(args.task_json).resolve()
        tasks = _load_tasks_from_json(task_json_path)
        task_batches = _chunk_tasks(tasks, args.batch_size)
        summary = []
        print(
            f"Loaded {len(tasks)} tasks from: {task_json_path}; "
            f"batch_size={args.batch_size}; total_batches={len(task_batches)}"
        )
        for idx, task_batch in enumerate(task_batches, start=1):
            batch_ids = [item["task_id"] for item in task_batch]
            batch_instruction = _format_batch_task_instruction(task_batch)
            batch_tag = f"batch_{idx}_{batch_ids[0]}_{batch_ids[-1]}"
            print("\n" + "#" * 80)
            print(f"[Batch {idx}/{len(task_batches)}] Task IDs: {batch_ids}")
            print("#" * 80)
            summary.append(
                await _run_graph_for_task(
                    task_instruction=batch_instruction,
                    robot_desc=args.robot_desc,
                    config_path=config_path,
                    base_output_dir=base_output_dir,
                    llm_provider_selections=llm_provider_selections,
                    task_id=batch_tag,
                    task_batch=task_batch,
                )
            )

        base_output_dir.mkdir(parents=True, exist_ok=True)
        batch_summary_path = base_output_dir / "batch_summary.json"
        with batch_summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_json": str(task_json_path),
                    "source_task_count": len(tasks),
                    "batch_size": args.batch_size,
                    "batch_count": len(summary),
                    "runs": summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nBatch summary saved to: {batch_summary_path}")
        return

    await _run_graph_for_task(
        task_instruction=args.task,
        robot_desc=args.robot_desc,
        config_path=config_path,
        base_output_dir=base_output_dir,
        llm_provider_selections=llm_provider_selections,
        task_id=args.task_id,
        reuse_classic_usd=bool(args.reuse_classic_usd),
        classic_usd_source=args.classic_usd_source,
        classic_run_dir=args.classic_run_dir,
        working_usd_path=args.working_usd_path,
    )


if __name__ == "__main__":
    import argparse
    import asyncio
    import traceback

    parser = argparse.ArgumentParser(description="Run SceneCraft with a LangGraph placement/evaluation loop")
    parser.add_argument("--task", "-t", default=None, help="Task instruction text to pass to the architect node")
    parser.add_argument("--task-json", default=None, help="Path to a JSON file containing a 'Task' list")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of tasks to batch together (only used with --task-json)")
    parser.add_argument("--output-dir", "-o", default="./temp_output", help="Directory to save generated configuration/results")
    parser.add_argument("--robot-desc", default="Fetch Robot with parallel gripper and moveable base", help="Optional robot description")
    parser.add_argument("--llm-config", "--config", dest="llm_config", default="run_config.json", help="Path to run config JSON with llm settings")
    parser.add_argument("--task-id", default=None, help="Stable task id such as H01_T01; required for --reuse-classic-usd")
    parser.add_argument(
        "--reuse-classic-usd",
        action="store_true",
        help="Skip graph architect/retriever/initial placement and evaluate/repair a copied classic USD.",
    )
    parser.add_argument(
        "--classic-usd-source",
        default=None,
        help="Optional explicit source classic USD path; otherwise derived from --task-id.",
    )
    parser.add_argument(
        "--classic-run-dir",
        default=None,
        help=(
            "Optional classic run directory containing blueprint.json/final_assets.json, "
            "or a parent directory containing run_* subdirectories."
        ),
    )
    parser.add_argument(
        "--working-usd-path",
        default=None,
        help="Path to the copied USD opened by the paired Isaac process, for run metadata.",
    )
    parser.add_argument(
        "--pipe-id",
        type=str,
        default="",
        help="Unique instance identifier for multi-instance parallel runs. "
             "Must match the --pipe-id passed to the paired isaac_sim_app.py process.",
    )
    parsed_args = parser.parse_args()

    # Activate the pipe-id so that all tool calls route to the correct Isaac Sim instance.
    set_pipe_id(parsed_args.pipe_id)

    if bool(parsed_args.task) == bool(parsed_args.task_json):
        parser.error("Provide exactly one of --task or --task-json")

    try:
        asyncio.run(_main_async(parsed_args))
    except Exception:
        traceback.print_exc()
        raise

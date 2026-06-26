from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Literal, Optional, TypedDict, Type
from pydantic import BaseModel, Field
import json
import os
import sys
import time


def _configure_text_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_text_streams()


def _timestamp_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


def _write_run_timings(output_dir: object, timings: Dict[str, Any]) -> None:
    os.makedirs(str(output_dir), exist_ok=True)
    path = os.path.join(str(output_dir), "timings.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2, ensure_ascii=False)


@contextmanager
def _timed_step(timings: Dict[str, Any], step_name: str, output_dir: object):
    started_at = _timestamp_iso()
    started_perf = time.perf_counter()
    try:
        yield
    except Exception as exc:
        duration_sec = round(time.perf_counter() - started_perf, 3)
        timings.setdefault("steps", {})[step_name] = {
            "started_at": started_at,
            "finished_at": _timestamp_iso(),
            "duration_sec": duration_sec,
            "status": "failed",
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        _write_run_timings(output_dir, timings)
        print(f"[Timing] {step_name}: {duration_sec:.3f}s (failed)")
        raise
    else:
        duration_sec = round(time.perf_counter() - started_perf, 3)
        timings.setdefault("steps", {})[step_name] = {
            "started_at": started_at,
            "finished_at": _timestamp_iso(),
            "duration_sec": duration_sec,
            "status": "ok",
        }
        _write_run_timings(output_dir, timings)
        print(f"[Timing] {step_name}: {duration_sec:.3f}s")


# --- 1. Node1 ---

class ExplicitObject(BaseModel):
    name: str = Field(
        description="Explicit, concrete, and task-critical object name, e.g., 'apple', 'cup'. Must be directly manipulable."
    )

    description: str = Field(
        description=(
            "Descriptive natural-language retrieval query for this task-critical object. "
            "It must include the explicit object name and describe only the object itself: category, "
            "appearance, physical properties, contents, typical use, or ordinary real-world context. "
            "Do not mention the robot, gripper, grasping, navigation, placement execution, or task procedure."
        )
    )

    placement_hint: str = Field(
        description="Geometric or spatial placement logic, e.g., 'On a flat surface at waist height'."
    )

class AbstractObject(BaseModel):
    category: Literal[
        "core_furniture",
        "contextual_clutter",
        "decoration",
        "interactive_prop"
    ] = Field(
        description="""
        High-level semantic category.
        - core_furniture: Large furniture defining room structure.
        - contextual_clutter: Everyday items reflecting recent usage.
        - decoration: Visual-only elements.
        - interactive_prop: Non-task interactive affordances.
        """
    )

    description: str = Field(
        description="""
        Abstract, descriptive natural language describing a GROUP of objects.
        MUST NOT include specific object or asset names.
        Example: 'Several common household appliances typically found in a kitchen'
        """
    )

    placement_hint: str = Field(
        description="High-level spatial description, e.g., 'Along the walls', 'Scattered across available surfaces'."
    )

class RoomSpec(BaseModel):
    id: str = Field(description="Unique room ID, e.g., 'room_0'")
    name: str = Field(description="Functional room name, e.g., 'Kitchen', 'Living Room'")
    
    size: Tuple[float, float] = Field(
        description="Width (x) and length (y) in meters. Should be >= 5.0 for navigability."
    )

    explicit_objects: List[ExplicitObject] = Field(
        description="All explicitly named, task-critical objects in this room."
    )

    abstract_objects: List[AbstractObject] = Field(
        description="All non-task objects described abstractly for environmental richness."
    )

class Connection(BaseModel):
    room_id_a: str
    room_id_b: str
    connection_type: Literal["door", "open_passage"] = Field(
        default="door",
        description="Type of navigational connection between rooms."
    )

class SceneBlueprint(BaseModel):
    scene_type: Literal["indoor"] = Field(
        description="Currently fixed to indoor environments."
    )

    rooms: List[RoomSpec]

    connections: List[Connection]

    robot_start_room_id: str = Field(
        description="ID of the room where the robot starts."
    )

class ObjectPlacement(BaseModel):
    """Placement information for a single object."""
    object_name: str = Field(description="Object name, must match exactly with objects_list")
    position: Tuple[float, float, float] = Field(description="Absolute (x, y, z) coordinates in meters")
    rotation: Tuple[float, float, float] = Field(description="Euler angles (rx, ry, rz) in degrees")
    support_surface: Optional[str] = Field(
        default=None,
        description="Parent object name if placed ON another object (e.g., 'Kitchen Table'). None for floor-placed items."
    )
    reasoning: str = Field(description="Brief explanation of placement logic (for debugging)")

class RoomLayout(BaseModel):
    """Complete layout for a single room."""
    room_name: str
    placements: List[ObjectPlacement] = Field(description="List of all object placements in this room")

class AgentState(TypedDict):
    # Input
    task_instruction: str
    robot_description: Optional[str]

    # Node1: Plan blueprint
    scene_blueprint: Optional[Dict]  # SceneBlueprint.model_dump()

    # Node2: Construct initial rooms
    tmp_usd: str

    # Node3: Retrieve and place objects
    retrieved_assets: Optional[Dict[str, Dict]]

    # Fields that may be needed by subsequent nodes
    placement_code: Optional[str]
    error: Optional[str]
    config_path: Optional[str]

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompt import ARCHITECT_SYSTEM_PROMPT, ARCHITECT_USER_PROMPT
from llm import configure_llm_manager, build_llm

DEFAULT_LLM_PROVIDER_SELECTIONS: Dict[str, str] = {
    "architect": "openai",
    "asset_selection": "openai",
    "placement": "openai",
}


def _load_llm_provider_selections(config_path: Optional[str]) -> Dict[str, str]:
    """Load node-to-provider mapping from run_config.json with safe defaults."""
    selections = dict(DEFAULT_LLM_PROVIDER_SELECTIONS)
    if not config_path:
        return selections

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        configured = (cfg.get("scene_generate", {}) or {}).get("llm_providers") or {}
        if isinstance(configured, dict):
            for key, value in configured.items():
                if isinstance(key, str) and isinstance(value, str) and value:
                    selections[key] = value
    except Exception as e:
        print(f"Warning: Failed to read llm providers from config '{config_path}': {e}")

    return selections


def _get_llm_provider(config_path: Optional[str], role: str) -> str:
    return _load_llm_provider_selections(config_path).get(
        role,
        DEFAULT_LLM_PROVIDER_SELECTIONS.get(role, "openai"),
    )


def _tool_agent_llm_kwargs(provider: str) -> Dict[str, Any]:
    """Provider-specific settings for LangChain tool agents."""
    if provider == "deepseek":
        # DeepSeek thinking mode requires `reasoning_content` to be preserved
        # across tool-call sub-turns. LangChain's generic agent history does not
        # reliably round-trip that provider-specific field, so disable thinking
        # only for DeepSeek tool-agent calls. Other providers keep defaults.
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def _json_schema_prompt(model_cls: Type[BaseModel], root_name: str) -> str:
    """Instructions for json_mode models, which do not get tool-enforced schemas."""
    schema = json.dumps(model_cls.model_json_schema(), indent=2, ensure_ascii=False)
    instructions = f"""
Return ONLY valid JSON for one `{root_name}` object.
Do not include markdown fences, prose, comments, or a wrapper key.
The top-level JSON object must be the `{root_name}` itself and must conform to this JSON schema:

{schema}

Important:
- Do NOT wrap the output in keys like "scene_blueprint", "result", "data", or "output".
- Do NOT invent alternate field names.
""".strip()

    if root_name == "SceneBlueprint":
        instructions += "\n" + """
- For `SceneBlueprint`, the top-level keys must be exactly: "scene_type", "rooms", "connections", "robot_start_room_id".
- Each room must use "id", "name", "size", "explicit_objects", and "abstract_objects".
- Put task-critical named objects under "explicit_objects" as objects with "name", "description", and "placement_hint".
- For each explicit object, "description" must be a descriptive retrieval query that includes the object name and only object-intrinsic semantics.
- Explicit object descriptions must not mention the robot, gripper, grasping, navigation, placement execution, or task procedure.
- Put all non-task room richness under "abstract_objects" as objects with "category", "description", and "placement_hint".
""".strip()
    return instructions


def node_architect(state: AgentState):
    """
    Node 1: Scene Architecture & Creative Design
    
    Takes task instruction, outputs a rich scene blueprint with:
    - Room layouts with detailed object lists
    - Object roles and placement hints
    """
    print(f"--- [Node 1] Architect: Architecting a rich world for '{state['task_instruction']}' ---")
    
    provider = _get_llm_provider(state.get("config_path"), "architect")
    print(f"   > LLM provider for architect: {provider}")
    llm = build_llm(
        provider=provider,
        temperature=0.7,
        timeout=280,
    )
    
    # json_mode avoids DeepSeek reasoner/tool_choice incompatibility.
    structured_llm = llm.with_structured_output(SceneBlueprint, method="json_mode")
    
    task_instruction = state["task_instruction"]
    robot_desc = state["robot_description"]
    
    user_prompt_text = ARCHITECT_USER_PROMPT.format(
        task=task_instruction,
        robot_desc=robot_desc,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", ARCHITECT_SYSTEM_PROMPT),
        ("system", "{format_instructions}"),
        (
            "user",
            user_prompt_text,
        ),
    ])

    # DeepSeek models may not reliably follow structured output.

    blueprint: SceneBlueprint = (prompt | structured_llm).invoke({
        "format_instructions": _json_schema_prompt(SceneBlueprint, "SceneBlueprint")
    })

    output_dir = state["output_dir"] if "output_dir" in state else "./temp_out"
    os.makedirs(output_dir, exist_ok=True)

    blueprint_dict = blueprint.model_dump()
    # Replace spaces in room names with underscores
    for room in blueprint_dict["rooms"]:
        room["name"] = room["name"].replace(" ", "_").replace("-", "_")
    with open(output_dir + "/blueprint.json", "w", encoding="utf-8") as f:
        json.dump(blueprint_dict, f, indent=2, ensure_ascii=False)
    
    print(f"   > Generated {len(blueprint.rooms)} rooms")
    for room in blueprint.rooms:
        explicit_count = len(room.explicit_objects)
        abstract_count = len(room.abstract_objects)
        print(f"     - {room.name}: {explicit_count} explicit + {abstract_count} abstract object groups")

    return {
        "scene_blueprint": blueprint_dict
    }

class AssetCandidate(BaseModel):
    """A single asset candidate."""
    asset_id: str
    usd_path: str
    caption: str
    category: str
    score: float


def _candidate_caption(doc) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    for key in ("selection_caption", "caption"):
        value = metadata.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(getattr(doc, "page_content", "") or "").strip()

class ExplicitObjectSelection(BaseModel):
    """LLM selection for an explicit object."""
    object_name: str = Field(description="The explicit object name being selected for")
    selected_asset_id: str = Field(description="The chosen asset_id from candidates")
    reasoning: str = Field(description="Why this asset was selected over others")

class AbstractObjectSelection(BaseModel):
    """LLM filtering for an abstract object group."""
    category: str = Field(
        description=(
            "The exact abstract candidate group key from the prompt heading, "
            "for example 'core_furniture_0' or 'contextual_clutter_3'. "
            "Do not use only the generic category name when an indexed key is shown."
        )
    )
    description: str = Field(description="Original abstract description")
    kept_asset_ids: List[str] = Field(description="List of asset_ids to KEEP (rejected ones are discarded)")
    reasoning: str = Field(description="Why these specific assets were kept")

class AssetSelectionResult(BaseModel):
    """Complete asset selection result."""
    room_name: str
    explicit_selections: List[ExplicitObjectSelection]
    abstract_selections: List[AbstractObjectSelection]

from Tools.USDAssetStore import USDAssetStore
from prompt import LAYOUT_PLANNER_PROMPT, ASSET_SELECTION_PROMPT


# Resolved at runtime; overridden by env or run_config.json.
# Set BEHAVIOR1K_ROOT in your .env (see .env.example) to point at the
# behavior-1k-assets dataset root on your machine.
FIXED_BEHAVIOR_ASSET_SOURCE_PREFIX = os.environ.get(
    "BEHAVIOR1K_ROOT",
    "/path/to/behavior-1k-assets",
)
EXPLICIT_RETRIEVAL_TOP_K = 8
EXPLICIT_RETRIEVAL_FETCH_K = 50
ABSTRACT_RETRIEVAL_TOP_K = 15
ABSTRACT_RETRIEVAL_FETCH_K = 100
ABSTRACT_RETRIEVAL_MAX_PER_DATASET_CATEGORY = 2

STRUCTURAL_ASSET_ID_PREFIXES = (
    "ceilings",
    "floor",
    "walls",
    "door",
    "ceiling",
    "wall",
    "floors",
    "doors",
    "sliding_door",
    "garage_door",
    "stairs",
    "paper_lantern",
)
STRUCTURAL_ASSET_CATEGORIES = frozenset(
    {
        "door",
        "wall",
        "floor",
        "ceiling",
        "walls",
        "floors",
        "ceilings",
        "doors",
        "sliding_door",
        "garage_door",
        "stairs",
        "handle",
        "paper_lantern",
    }
)
EXCLUDED_ASSET_NAME_TOKENS = ("countertop",)


def _is_structural_asset(asset_id: str, category: str) -> bool:
    aid = (asset_id or "").lower()
    cat = (category or "").lower()
    for prefix in STRUCTURAL_ASSET_ID_PREFIXES:
        if aid.startswith(prefix):
            return True
    if cat in STRUCTURAL_ASSET_CATEGORIES:
        return True
    return False


def _is_excluded_asset(asset_key: str, info: Dict[str, Any]) -> bool:
    """Return whether an asset should be removed before agent placement."""
    identifying_values = (
        asset_key,
        info.get("asset_id"),
        info.get("category"),
        info.get("usd_path"),
    )
    return any(
        token in str(value or "").lower()
        for token in EXCLUDED_ASSET_NAME_TOKENS
        for value in identifying_values
    )


def _is_structural_asset_record(asset_key: str, info: Dict[str, Any]) -> bool:
    """Classify a selected asset using authoritative identity fields.

    Older abstract-selection outputs may contain the semantic group name in
    ``category`` instead of the selected asset's dataset category. Prefer the
    asset id and USD path so those stale records are not falsely removed.
    """
    asset_id = str(info.get("asset_id") or asset_key)
    if _is_structural_asset(asset_id, ""):
        return True

    path_parts = [
        part
        for part in str(info.get("usd_path") or "")
        .lower()
        .replace("\\", "/")
        .split("/")
        if part
    ]
    if any(part in STRUCTURAL_ASSET_CATEGORIES for part in path_parts):
        return True

    if not info.get("asset_id") and not info.get("usd_path"):
        return _is_structural_asset(asset_key, str(info.get("category") or ""))
    return False


def _filter_excluded_room_assets(
    room_assets: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        asset_key: info
        for asset_key, info in room_assets.items()
        if not (
            isinstance(info, dict)
            and (
                _is_excluded_asset(str(asset_key), info)
                or _is_structural_asset_record(str(asset_key), info)
            )
        )
    }


def _abstract_retrieval_query(room_name: str, category: str, description: str) -> str:
    parts = [
        f"Abstract scene role: {category.replace('_', ' ')}.",
        f"Room context: {room_name.replace('_', ' ')}.",
        f"Role description: {description.strip()}",
    ]
    return " ".join(parts)


def _format_distance(score: float) -> str:
    return f"{score:.3f} lower is better"

def _load_usd_path_settings(config_path: str) -> Dict[str, object]:
    """Load USD path conversion settings from run_config.json."""
    default_settings: Dict[str, object] = {
        "target_dataset_root": FIXED_BEHAVIOR_ASSET_SOURCE_PREFIX,
        "normalize_slashes": True,
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        scene_cfg = cfg.get("scene_generate", {}) or {}

        # Preferred new config: only target dataset root is user-configurable.
        target_dataset_root = scene_cfg.get("target_dataset_root")
        normalize_slashes = scene_cfg.get("normalize_slashes", True)

        # Backward compatibility with previous usd_path_mapping schema.
        if not target_dataset_root:
            legacy_mapping = scene_cfg.get("usd_path_mapping", {}) or {}
            target_dataset_root = legacy_mapping.get("target_prefix")
            normalize_slashes = legacy_mapping.get("normalize_slashes", normalize_slashes)

        # Expand ${VAR}/$VAR/~ in case the config stores an env-var template
        # such as "${BEHAVIOR1K_ROOT}".
        if target_dataset_root:
            target_dataset_root = os.path.expanduser(
                os.path.expandvars(str(target_dataset_root))
            )

        return {
            "target_dataset_root": str(target_dataset_root or default_settings["target_dataset_root"]),
            "normalize_slashes": bool(normalize_slashes),
        }
    except Exception as e:
        print(f"Warning: Failed to load USD path settings from config '{config_path}': {e}")
        return default_settings


def _convert_usd_path(raw_path: str, path_settings: Dict[str, object]) -> str:
    """Resolve a RAG-returned ``usd_path`` to an absolute on-disk location.

    Two on-disk dataset layouts are supported:

    * **Relative path** (current default; what the open-source index
      stores): ``objects/acorn/qkjrwt/usd/qkjrwt.usd``. The configured
      ``target_dataset_root`` (i.e. ``${BEHAVIOR1K_ROOT}``) is joined
      onto the front.
    * **Legacy absolute path** (from older, machine-specific indexes):
      ``D:/path/to/behavior-1k-assets/objects/...``. The historical
      hard-coded prefix is rewritten to ``target_dataset_root``.

    The two branches are detected by whether *raw_path* looks
    absolute. Empty input is passed through unchanged.
    """
    if not raw_path:
        return raw_path

    target_dataset_root = str(path_settings.get("target_dataset_root", "") or "")
    normalize_slashes = bool(path_settings.get("normalize_slashes", True))

    converted = str(raw_path)

    # Distinguish absolute vs relative without invoking os.path so we
    # accept Windows-style ``D:\foo`` paths even when running on Linux.
    drive_letter = len(converted) >= 2 and converted[1] == ":"
    is_absolute = converted.startswith(("/", "\\")) or drive_letter

    if is_absolute:
        if target_dataset_root:
            # Legacy layout: replace the old hard-coded prefix.
            source_windows = FIXED_BEHAVIOR_ASSET_SOURCE_PREFIX.replace("/", "\\")
            source_posix = FIXED_BEHAVIOR_ASSET_SOURCE_PREFIX.replace("\\", "/")
            converted = converted.replace(source_windows, target_dataset_root)
            converted = converted.replace(source_posix, target_dataset_root)
    else:
        # New layout: dataset-relative path stored in the RAG index.
        if target_dataset_root:
            root = target_dataset_root.rstrip("/\\")
            relative = converted.lstrip("/\\")
            # Strip a leading "./" or ".\" — RAG entries occasionally include one.
            if relative.startswith(("./", ".\\")):
                relative = relative[2:]
            converted = f"{root}/{relative}"

    if normalize_slashes:
        converted = converted.replace("\\", "/")
    return converted


def _estimate_house_length_from_blueprint(scene_blueprint: Optional[Dict], fallback_room_count: int) -> int:
    """Estimate a realistic square house side length in meters.

    We target typical residential room sizes around 20-30 m^2 per room,
    then add a modest circulation/shared-space overhead for the full house.
    """
    rooms = []
    if scene_blueprint:
        rooms = scene_blueprint.get("rooms", []) or []

    realistic_room_areas: List[float] = []
    for room in rooms:
        if not isinstance(room, dict):
            continue
        size = room.get("size")
        if (
            isinstance(size, (list, tuple))
            and len(size) == 2
            and all(isinstance(v, (int, float)) and v > 0 for v in size)
        ):
            raw_area = float(size[0]) * float(size[1])
            clamped_area = min(20.0, max(10.0, raw_area))
            realistic_room_areas.append(clamped_area)

    room_count = len(rooms) if rooms else max(1, fallback_room_count)
    target_room_area = 25.0

    if realistic_room_areas:
        total_room_area = sum(realistic_room_areas)
        largest_room_area = max(realistic_room_areas)
    else:
        largest_room_area = target_room_area
        total_room_area = target_room_area * room_count

    circulation_multiplier = 1.12 if room_count <= 2 else 1.18 if room_count <= 4 else 1.25
    estimated_house_area = total_room_area * circulation_multiplier

    import math

    estimated_side = math.sqrt(estimated_house_area)
    minimum_side_for_main_room = math.sqrt(largest_room_area) * 1.2
    house_length = math.ceil(max(estimated_side, minimum_side_for_main_room, 6.0))
    return int(house_length)

def node_retriever_and_rooms_construction(state: AgentState):
    """
    Node 2: Asset Retrieval & Initial Rooms Construction
    
    Given room plan and object list, retrieve assets and place them in rooms.
    """
    print(f"--- [Node 2] Retriever & Placer: Fetching assets and placing them ---")

    scene_blueprint = state["scene_blueprint"]
    config_path = state.get("config_path") or "run_config.json"
    usd_path_settings = _load_usd_path_settings(config_path)
    store = USDAssetStore.from_config_file(config_path)

    all_room_layouts = []  # Baseline method, not currently used
    all_asset_selections = []  # Store LLM selection results
    final_retrieved_assets = {}  # Store final retrieval results

    # Store retrieval results per room
    all_explicit_candidates = {}
    all_abstract_candidates = {}

    import json
    import os
    output_dir = state["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    retrieval_cache = {}

    def cached_search(query: str, **kwargs):
        cache_key = json.dumps({"query": query, **kwargs}, sort_keys=True, ensure_ascii=False)
        if cache_key not in retrieval_cache:
            retrieval_cache[cache_key] = store.search_with_score(query, **kwargs)
        return retrieval_cache[cache_key]

    for room in scene_blueprint["rooms"]:
        room_name = room["name"]
        explicit_objects = room["explicit_objects"]
        abstract_objects = room["abstract_objects"]

        print(f"\n   > Room '{room_name}':")
        print(f"     - {len(explicit_objects)} explicit objects (task-critical)")
        print(f"     - {len(abstract_objects)} abstract object groups (environmental)")

        final_retrieved_assets[room_name] = {}

        # ===== Step 1: Retrieve Candidates =====
        explicit_candidates = {}  # {obj_name: [AssetCandidate, ...]}
        explicit_descriptions = {
            obj["name"]: str(obj.get("description") or obj["name"]).strip()
            for obj in explicit_objects
        }
        explicit_placement_hints = {
            obj["name"]: str(obj.get("placement_hint") or "").strip()
            for obj in explicit_objects
        }
        abstract_candidates = {}  # {category: [AssetCandidate, ...]}

        # Retrieve explicit objects (top-5)
        print(f"\n     [Retrieving Explicit Object Candidates]")
        for obj in explicit_objects:
            obj_name = obj["name"]
            description = str(obj.get("description") or obj_name).strip()
            # For explicit objects, retrieve the first match
            results = cached_search(
                description,
                top_k=EXPLICIT_RETRIEVAL_TOP_K,
                vector_type="explicit",
                fetch_k=EXPLICIT_RETRIEVAL_FETCH_K,
                dedupe_by_category=False,
                max_per_category=None,
            )
            
            if len(results) == 0:
                print(f"       ! No assets found for '{obj_name}'")
                continue
            
            candidates = []
            for doc, score in results:
                asset_id = doc.metadata.get("asset_id", "N/A")
                category = doc.metadata.get("category", "unknown")
                if _is_structural_asset(asset_id, category):
                    print(f"       ! Skipping structural asset '{asset_id}' (category: {category}) for explicit object '{obj_name}'")
                    continue
                candidates.append(AssetCandidate(
                    asset_id=doc.metadata.get("asset_id", "N/A"),
                    usd_path=doc.metadata.get("usd_path", ""),
                    caption=_candidate_caption(doc),
                    category=doc.metadata.get("category", "unknown"),
                    score=score
                ))
            
            explicit_candidates[obj_name] = candidates
            print(f"       '{obj_name}' query='{description}': Found {len(candidates)} candidates")

        # Retrieve abstract objects (top-15, LLM will filter to 3-5)
        print(f"\n     [Retrieving Abstract Object Candidates]")
        for idx, abstract_obj in enumerate(abstract_objects):
            abstract_category = abstract_obj["category"]
            description = abstract_obj["description"]
            
            # Use category + index as unique key to avoid overwriting
            category_key = f"{abstract_category}_{idx}"
            
            placement_hint = str(abstract_obj.get("placement_hint") or "").strip()
            role_query = _abstract_retrieval_query(
                room_name,
                abstract_category,
                description,
            )
            results = cached_search(
                role_query,
                top_k=ABSTRACT_RETRIEVAL_TOP_K,
                vector_type="abstract",
                fetch_k=ABSTRACT_RETRIEVAL_FETCH_K,
                dedupe_by_category=False,
                max_per_category=ABSTRACT_RETRIEVAL_MAX_PER_DATASET_CATEGORY,
            )
            
            if len(results) == 0:
                print(
                    f"       ! No assets found for '{abstract_category}' "
                    f"(instance {idx})"
                )
                continue
            
            candidates = []
            for doc, score in results:
                asset_id = doc.metadata.get("asset_id", "N/A")
                dataset_category = doc.metadata.get("category", "unknown")
                if _is_structural_asset(asset_id, dataset_category):
                    print(f"       ! Skipping structural asset '{asset_id}' (category: {dataset_category}) for abstract '{category_key}'")
                    continue
                candidates.append(AssetCandidate(
                    asset_id=doc.metadata.get("asset_id", "N/A"),
                    usd_path=doc.metadata.get("usd_path", ""),
                    caption=_candidate_caption(doc),
                    category=dataset_category,
                    score=score
                ))
            
            # Store using category_key with original metadata attached
            abstract_candidates[category_key] = {
                "category": abstract_category,
                "description": description,
                "query": role_query,
                "placement_hint": placement_hint,
                "candidates": candidates
            }
            print(f"       '{abstract_category}' (instance {idx}) query='{role_query}': Found {len(candidates)} candidates")


        all_explicit_candidates[room_name] = explicit_candidates
        all_abstract_candidates[room_name] = abstract_candidates

        # ===== Step 2: LLM Selection =====
        print(f"\n     [LLM Asset Selection]")
        
        # Format candidates for prompt
        explicit_text = ""
        for obj_name, candidates in explicit_candidates.items():
            query_description = explicit_descriptions.get(obj_name, obj_name)
            explicit_text += f"\n**{obj_name}** - \"{query_description}\" (select 1):\n"
            for i, cand in enumerate(candidates, 1):
                explicit_text += f"  {i}. [{cand.asset_id}] {cand.caption} (distance: {_format_distance(cand.score)})\n"
        
        abstract_text = ""
        for category_key, category_data in abstract_candidates.items():
            category = category_data["category"]
            desc = category_data["description"]
            candidates = category_data["candidates"]
            
            abstract_text += f"\n**{category_key}** ({category}) - \"{desc}\" (select 1-3 to keep):\n"
            for i, cand in enumerate(candidates, 1):
                abstract_text += f"  {i}. [{cand.asset_id}] {cand.caption} (distance: {_format_distance(cand.score)})\n"
        
        # Call LLM
        provider = _get_llm_provider(state.get("config_path"), "asset_selection")
        print(f"     > LLM provider for asset_selection: {provider}")
        llm = build_llm(
            provider=provider,
            temperature=0.2,
            timeout=600,
        )
        structured_llm = llm.with_structured_output(AssetSelectionResult, method="json_mode")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{format_instructions}"),
            ("user", ASSET_SELECTION_PROMPT),
        ])

        selection_result: AssetSelectionResult = (prompt | structured_llm).invoke({
            "room_name": room_name,
            "task_instruction": state["task_instruction"],
            "explicit_candidates_text": explicit_text,
            "abstract_candidates_text": abstract_text,
            "format_instructions": _json_schema_prompt(AssetSelectionResult, "AssetSelectionResult"),
        })

        all_asset_selections.append(selection_result.model_dump())

        # ===== Step 3: Build Final Asset Map =====
        # Add selected explicit objects
        for sel in selection_result.explicit_selections:
            obj_name = sel.object_name
            selected_id = sel.selected_asset_id
            
            # Find the asset
            candidates = explicit_candidates.get(obj_name, [])
            selected_asset = next((c for c in candidates if c.asset_id == selected_id), None)
            if selected_asset is None and candidates:
                selected_asset = candidates[0]
                selected_id = selected_asset.asset_id
                print(f"       ! Warning: invalid explicit selection for {obj_name}; fallback to top candidate {selected_id}")
            
            if selected_asset:
                final_retrieved_assets[room_name][obj_name] = {
                    "asset_id": selected_asset.asset_id,
                    "usd_path": _convert_usd_path(selected_asset.usd_path, usd_path_settings),
                    "caption": selected_asset.caption,
                    "category": selected_asset.category,
                    "room_name": room_name,
                    "score": selected_asset.score,
                    "object_type": "explicit",
                    "explicit_description": explicit_descriptions.get(obj_name, obj_name),
                    "placement_hint": explicit_placement_hints.get(obj_name, ""),
                    "selection_reasoning": sel.reasoning
                }
                print(f"       + Explicit: {obj_name} -> {selected_id}")

        # Add selected abstract objects
        for sel in selection_result.abstract_selections:
            category_key = str(sel.category or "").strip()
            kept_ids = [str(asset_id).strip() for asset_id in sel.kept_asset_ids if str(asset_id).strip()]
            
            category_data = abstract_candidates.get(category_key)
            if category_data is None:
                kept_id_set = set(kept_ids)
                normalized_description = str(sel.description or "").strip().lower()

                # The LLM may return a generic category like "core_furniture"
                # instead of the indexed prompt key. Recover using selected IDs
                # first, then the original description/category metadata.
                id_matches = []
                if kept_id_set:
                    for key, data in abstract_candidates.items():
                        candidate_ids = {c.asset_id for c in data["candidates"]}
                        if kept_id_set.issubset(candidate_ids):
                            id_matches.append((key, data))
                    if len(id_matches) == 1:
                        category_key, category_data = id_matches[0]

                if category_data is None and normalized_description:
                    description_matches = [
                        (key, data)
                        for key, data in abstract_candidates.items()
                        if data["category"] == category_key
                        and str(data["description"]).strip().lower() == normalized_description
                    ]
                    if len(description_matches) == 1:
                        category_key, category_data = description_matches[0]

                if category_data is None:
                    category_matches = [
                        (key, data)
                        for key, data in abstract_candidates.items()
                        if data["category"] == category_key
                    ]
                    if len(category_matches) == 1:
                        category_key, category_data = category_matches[0]

            if category_data is None:
                print(
                    "       ! Warning: abstract selection could not be matched "
                    f"to candidates: category={sel.category!r}, kept_ids={kept_ids}; skipping"
                )
                continue
            
            candidates = category_data["candidates"]
            placement_hint = category_data["placement_hint"]
            valid_kept_ids = [
                asset_id
                for asset_id in kept_ids
                if any(c.asset_id == asset_id for c in candidates)
            ]
            if not valid_kept_ids and candidates:
                valid_kept_ids = [c.asset_id for c in candidates[: min(3, len(candidates))]]
                print(f"       ! Warning: invalid abstract selection for {category_key}; fallback to top {len(valid_kept_ids)} candidates")
            
            for i, asset_id in enumerate(valid_kept_ids):
                asset = next((c for c in candidates if c.asset_id == asset_id), None)
                if asset:
                    # Use category_key to ensure uniqueness
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
                        "placement_group_key": category_key,
                    }
            
            print(f"       + Abstract: {category_key} -> kept {len(valid_kept_ids)} items")

    for room_name, room_assets in final_retrieved_assets.items():
        filtered_assets = _filter_excluded_room_assets(room_assets)
        removed_assets = sorted(set(room_assets) - set(filtered_assets))
        if removed_assets:
            print(
                f"       - Hard-filtered excluded assets from {room_name}: "
                + ", ".join(removed_assets)
            )
        final_retrieved_assets[room_name] = filtered_assets

    # ===== Save Results =====
    import json
    import os
    from datetime import datetime
    from Tools.place_tool import create_room
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir + f"/asset_candidates.json", "w", encoding="utf-8") as f:
        candidates_dump = {
            "explicit": {
                room: {k: [c.model_dump() for c in v] for k, v in candidates.items()}
                for room, candidates in all_explicit_candidates.items()
            },
            "abstract": {
                room: {
                    k: {
                        "category": v["category"],
                        "description": v["description"],
                        "query": v.get("query", v["description"]),
                        "placement_hint": v["placement_hint"],
                        "candidates": [c.model_dump() for c in v["candidates"]]
                    }
                    for k, v in candidates.items()
                }
                for room, candidates in all_abstract_candidates.items()
            }
        }
        json.dump(candidates_dump, f, indent=2, ensure_ascii=False)
    
    with open(output_dir + f"/llm_selections.json", "w", encoding="utf-8") as f:
        json.dump(all_asset_selections, f, indent=2, ensure_ascii=False)
    
    with open(output_dir + f"/final_assets.json", "w", encoding="utf-8") as f:
        json.dump(final_retrieved_assets, f, indent=2, ensure_ascii=False)

    # --- create rooms in Isaac Sim before placement ---
    try:
        # prefer blueprint room count if available, otherwise infer from retrieved assets
        room_num = len(state.get("scene_blueprint", {}).get("rooms", [])) if state.get("scene_blueprint") else max(1, len(final_retrieved_assets))
        house_length = _estimate_house_length_from_blueprint(
            state.get("scene_blueprint"),
            fallback_room_count=room_num,
        )
        # example connectivity: chain rooms (adjust from blueprint if available)
        connectivity = [(i, i+1) for i in range(room_num-1)]

        print(f"Creating {room_num} rooms in Isaac Sim, house_length={house_length}m, connectivity={connectivity}")
        # Build room name list from final_retrieved_assets
        rooms = list(final_retrieved_assets.keys())
        create_res = create_room.invoke({"rooms": rooms, "house_length": house_length, "connectivity": connectivity})
        print("create_room result:", create_res)

        if save_stage():
            print("Stage save completed successfully after create room.")
        else:
            print("Warning: Stage save failed after placement.")

        # Validate
        if not create_res or (isinstance(create_res, dict) and not create_res.get("boundaries")):
            print("Warning: create_room failed or returned empty boundaries — check isaac_sim_app logs.")
            # decide: continue (best-effort) or abort:
            # raise RuntimeError("Room creation failed")
    except Exception as e:
        print("Error calling create_room:", e)

    return {
        "retrieved_assets": final_retrieved_assets,
        # "room_layouts": all_room_layouts
    }

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from prompt import PLACE_AGENT_SYSTEM_PROMPT, PLACE_AGENT_USER_PROMPT, OPTIMIZED_PLACE_AGENT_SYSTEM_PROMPT, OPTIMIZED_PLACE_AGENT_USER_PROMPT
from langgraph.checkpoint.memory import InMemorySaver
from place_agent_middleware import AssetPlacementFoldingMiddleware
import json
from Tools.place_tool import (
    scan_scene,
    get_room_context,
    query_floor_space,
    query_surface_status,
    get_object_bounds,
    spawn_object,
    set_asset_name_usd_path_registry,
    set_pose,
    scale_object,
    resolve_placement_intent,
    select_room,
    save_stage,
    set_pipe_id,
    delete_object,
    get_unresolved_spawned_objects,
)


def _asset_name_from_info(info: Dict[str, Any]) -> str:
    asset_id = str(info.get("asset_id") or "").strip()
    return asset_id.rsplit("_", 1)[0] if asset_id else ""


def _build_room_asset_entries(room_assets: Dict[str, Dict]) -> List[Dict[str, Any]]:
    raw_entries = []
    name_counts: Dict[str, int] = {}

    for asset_key, info in room_assets.items():
        if not isinstance(info, dict) or _is_excluded_asset(asset_key, info):
            continue
        base_name = _asset_name_from_info(info)
        if not base_name:
            continue
        raw_entries.append((base_name, info))
        name_counts[base_name] = name_counts.get(base_name, 0) + 1

    name_occurrences: Dict[str, int] = {}
    entries = []
    for base_name, info in raw_entries:
        name_occurrences[base_name] = name_occurrences.get(base_name, 0) + 1
        if name_counts[base_name] > 1:
            agent_name = f"{base_name}_{name_occurrences[base_name]}"
        else:
            agent_name = base_name
        entries.append(
            {
                "agent_name": agent_name,
                "base_name": base_name,
                "info": info,
            }
        )

    return entries


def _build_asset_name_usd_registry(room_assets: Dict[str, Dict]) -> Dict[str, str]:
    registry: Dict[str, str] = {}
    for entry in _build_room_asset_entries(room_assets):
        asset_name = entry["agent_name"]
        info = entry["info"]
        usd_path = str(info.get("usd_path") or "").strip()
        if asset_name and usd_path:
            registry[asset_name] = usd_path
    return registry


def _build_placement_asset_library(room_name: str, room_assets: Dict[str, Dict]) -> str:
    lines = [f"**Available Assets for {room_name}:**", ""]
    explicit_assets = []
    abstract_groups: Dict[str, Dict[str, Any]] = {}

    for entry in _build_room_asset_entries(room_assets):
        asset_name = entry["agent_name"]
        info = entry["info"]
        placement_hint = str(info.get("placement_hint") or "").strip()
        if info.get("object_type") == "abstract":
            group_key = str(info.get("placement_group_key") or "").strip()
            if not group_key:
                group_key = f"{info.get('abstract_description', '')}\n{placement_hint}"
            group = abstract_groups.setdefault(
                group_key,
                {"asset_names": [], "placement_hint": placement_hint},
            )
            group["asset_names"].append(asset_name)
        else:
            explicit_assets.append((asset_name, placement_hint))

    if explicit_assets:
        lines.append("**Explicit Assets:**")
        for asset_name, placement_hint in explicit_assets:
            lines.append(f"- `{asset_name}`")
            if placement_hint:
                lines.append(f"  Placement Hint: {placement_hint}")
        lines.append("")

    if abstract_groups:
        lines.append("**Abstract Asset Groups:**")
        for group in abstract_groups.values():
            formatted_names = ", ".join(
                f"`{asset_name}`" for asset_name in group["asset_names"]
            )
            lines.append(f"- Assets: {formatted_names}")
            if group["placement_hint"]:
                lines.append(
                    f"  Shared Placement Hint: {group['placement_hint']}"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _truncate_text(text: object, max_len: int = 500) -> str:
    raw = "" if text is None else str(text)
    if len(raw) <= max_len:
        return raw
    return raw[:max_len] + " ...[truncated]"


def _extract_message_content(message: object) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


def _print_agent_stream_messages(messages: List[object]) -> None:
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role") or message.get("type") or "unknown"
            tool_calls = message.get("tool_calls") or []
            name = message.get("name")
        else:
            role = getattr(message, "type", None) or getattr(message, "role", "unknown")
            tool_calls = getattr(message, "tool_calls", None) or []
            name = getattr(message, "name", None)

        content = _extract_message_content(message)
        if isinstance(message, dict):
            metadata = message.get("additional_kwargs") or {}
        else:
            metadata = getattr(message, "additional_kwargs", {}) or {}
        if metadata.get("lc_source") == "asset_placement_folding":
            asset_name = metadata.get("asset_name") or "unknown_asset"
            print(f"[Agent][FoldedAsset] {asset_name} -> {_truncate_text(content)}")
            continue

        if role in ("ai", "assistant"):
            if content and content.strip():
                print(f"[Agent][LLM] {_truncate_text(content)}")
            if tool_calls:
                for call in tool_calls:
                    if isinstance(call, dict):
                        tool_name = call.get("name") or (call.get("function") or {}).get("name") or "unknown_tool"
                        tool_args = call.get("args") or (call.get("function") or {}).get("arguments") or {}
                    else:
                        tool_name = getattr(call, "name", "unknown_tool")
                        tool_args = getattr(call, "args", {})
                    print(f"[Agent][ToolCall] {tool_name} args={_truncate_text(tool_args)}")
        elif role == "tool":
            tool_name = name or "unknown_tool"
            print(f"[Agent][ToolResult] {tool_name} -> {_truncate_text(content)}")
        elif role == "human":
            print(f"[Agent][User] {_truncate_text(content)}")

_UNRESOLVED_ORIGIN_TOLERANCE = 0.05

def _cleanup_unresolved_objects(room_name: str) -> List[str]:
    """Delete spawned objects that never completed a successful set_pose.

    The placement tools maintain an explicit spawned-but-unplaced registry.
    The older origin check remains as a fallback for legacy or external
    objects that were not observed through the current tool process.

    Returns a list of deleted object names.
    """
    deleted = []
    attempted = set()

    for name in get_unresolved_spawned_objects(room_name):
        attempted.add(name)
        print(
            f"   [Cleanup] Deleting unresolved object '{name}': "
            "spawn_object succeeded but set_pose never succeeded"
        )
        result = delete_object.invoke({"object_name": name})
        if result:
            deleted.append(name)
            print(f"   [Cleanup] Deleted '{name}'")
        else:
            print(f"   [Cleanup] Failed to delete '{name}'")

    scene_objects = scan_scene.invoke({})
    if not scene_objects or not isinstance(scene_objects, list):
        return deleted

    for obj in scene_objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", ""))
        if not name or name in attempted:
            continue
        cat = name.lower()
        if cat.startswith(("wall", "floor", "ceiling", "door", "groundplane")):
            continue
        position = obj.get("position")
        if position is None:
            continue
        try:
            px, py, pz = float(position[0]), float(position[1]), float(position[2])
        except (TypeError, IndexError, ValueError):
            continue
        if abs(px) <= _UNRESOLVED_ORIGIN_TOLERANCE and abs(py) <= _UNRESOLVED_ORIGIN_TOLERANCE and abs(pz) <= _UNRESOLVED_ORIGIN_TOLERANCE:
            print(f"   [Cleanup] Deleting unresolved object '{name}' still at origin in room '{room_name}'")
            result = delete_object.invoke({"object_name": name})
            if result:
                deleted.append(name)
                print(f"   [Cleanup] Deleted '{name}'")
            else:
                print(f"   [Cleanup] Failed to delete '{name}'")
    if deleted:
        print(f"   [Cleanup] Removed {len(deleted)} unresolved object(s) from room '{room_name}': {deleted}")
    else:
        print(f"   [Cleanup] No unresolved objects found in room '{room_name}'")
    return deleted

def node_place_with_sub_agent(state: AgentState):
    """
    Node 3: Advanced Placement with Sub-Agent (Not implemented yet)
    
    Placeholder for future advanced placement logic using a sub-agent.
    """
    print("--- [Node 3] Running sub-agent for complex subtask ---")

    retrieved_assets = state.get("retrieved_assets", {}) or {}
    
    if not retrieved_assets:
        print("Error: No retrieved assets found")
        return {"room_layouts": []}

    tools = [get_room_context,
             query_floor_space,
             query_surface_status,
             scan_scene,
             get_object_bounds,
             spawn_object,
             set_pose,
             scale_object,
             resolve_placement_intent]
    
    placement_provider = _get_llm_provider(state.get("config_path"), "placement")
    print(f"   > LLM provider for placement: {placement_provider}")
    llm = build_llm(
        provider=placement_provider,
        temperature=0.0,
        timeout=180,
        **_tool_agent_llm_kwargs(placement_provider),
    )
    # LangChain 1.0+ agent API
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=OPTIMIZED_PLACE_AGENT_SYSTEM_PROMPT,
        middleware=[AssetPlacementFoldingMiddleware()],
        checkpointer=InMemorySaver(),
    )

    # Process each room sequentially
    for room_name, room_assets in retrieved_assets.items():
        room_assets = _filter_excluded_room_assets(room_assets)
        
        if not room_assets:
            print(f"! Skipping room '{room_name}' - no assets retrieved")
            continue

        set_asset_name_usd_path_registry(
            _build_asset_name_usd_registry(room_assets)
        )
        
        print(f"\n{'='*80}")
        print(f"Processing Room: {room_name}")
        print(f"{'='*80}")
        
        # Build the asset library string for the current room
        asset_library_str = _build_placement_asset_library(
            room_name,
            room_assets,
        )

        print(f"\n[Assets for {room_name}]:")
        print(asset_library_str)
        
        print(f"[Room Selection] Switching Isaac Sim context to room '{room_name}'")
        room_selected = select_room.invoke({"room_name": room_name})
        if not room_selected:
            print(f"Warning: Failed to select room '{room_name}' in Isaac Sim, skipping placement for this room")
            continue

        # Build room info (default dimensions; can be sourced elsewhere if needed).
        # The no-Blueprint ablation has no architect-generated placement hints,
        # so it may opt in to passing the original task directly to the
        # otherwise unchanged classic placer.
        room_info = f"Room: {room_name}"
        placement_task_instruction = str(
            state.get("placement_task_instruction") or ""
        ).strip()
        if placement_task_instruction:
            room_info += f"\nDirect task context: {placement_task_instruction}"

        print(room_info)

        user_prompt = OPTIMIZED_PLACE_AGENT_USER_PROMPT.format(
            objects_to_place=asset_library_str,
            room_info=room_info,
        )

        print("[Agent Trace] Streaming intermediate steps...")
        seen_message_ids = set()
        result = {}
        thread_id = f"room_{room_name}"

        for state_chunk in agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ]
            },
            {"configurable": {"thread_id": thread_id}},
            stream_mode="values",
        ):
            result = state_chunk
            if not isinstance(state_chunk, dict):
                continue

            messages = state_chunk.get("messages") or []
            if not isinstance(messages, list):
                continue

            new_messages = []
            for message in messages:
                message_id = getattr(message, "id", None)
                if message_id is None and isinstance(message, dict):
                    message_id = message.get("id")
                message_key = message_id or id(message)
                if message_key in seen_message_ids:
                    continue
                seen_message_ids.add(message_key)
                new_messages.append(message)
            if new_messages:
                _print_agent_stream_messages(new_messages)

        print(f"\n+ Completed placement for room '{room_name}'")
        try:
            output = result
            if isinstance(result, dict):
                messages = result.get("messages") or []
                if messages:
                    last_msg = messages[-1]
                    output = getattr(last_msg, "content", last_msg)
        except Exception:
            output = result
        print(f"Agent output: {output}")

        print(f"\n[Cleanup] Checking for unresolved objects in room '{room_name}'...")
        _cleanup_unresolved_objects(room_name)

        if save_stage():
            print("Stage save completed successfully.")
        else:
            print("Warning: Stage save failed after placement and cleanup.")

    print("\n[Scene Save] Saving stage after all room placements are complete")


    return {
    }

def _sanitize_path_component(value: Optional[str], fallback: str = "task") -> str:
    text = (value or "").strip()
    if not text:
        return fallback

    sanitized_chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            sanitized_chars.append(ch)
        elif ch.isspace():
            sanitized_chars.append("_")

    sanitized = "".join(sanitized_chars).strip("_")
    return sanitized[:80] or fallback


def _load_tasks_from_json(task_json_path: "Path") -> List[Dict[str, str]]:
    with task_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_tasks = payload.get("Task")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError(f"Invalid task JSON in '{task_json_path}': expected non-empty 'Task' list")

    normalized_tasks: List[Dict[str, str]] = []
    for idx, item in enumerate(raw_tasks, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid task entry at index {idx - 1}: expected object")

        task_instruction = item.get("User Instruction")
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            raise ValueError(
                f"Invalid task entry at index {idx - 1}: missing non-empty 'User Instruction'"
            )

        task_id_value = item.get("Task id", str(idx))
        normalized_tasks.append(
            {
                "task_id": str(task_id_value).strip() or str(idx),
                "task_instruction": task_instruction.strip(),
            }
        )

    return normalized_tasks


def _chunk_tasks(tasks: List[Dict[str, str]], batch_size: int) -> List[List[Dict[str, str]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]


def _format_batch_task_instruction(task_batch: List[Dict[str, str]]) -> str:
    if not task_batch:
        raise ValueError("task_batch cannot be empty")

    if len(task_batch) == 1:
        return task_batch[0]["task_instruction"]

    lines = [
        "You are given a batch of tasks. Build one unified scene that satisfies all tasks simultaneously.",
        "Complete task list:",
    ]
    for idx, item in enumerate(task_batch, start=1):
        lines.append(f"{idx}. [Task id: {item['task_id']}] {item['task_instruction']}")
    return "\n".join(lines)


def _run_single_task(
    *,
    task_instruction: str,
    robot_desc: str,
    config_path: "Path",
    base_output_dir: "Path",
    llm_provider_selections: Dict[str, str],
    task_id: Optional[str] = None,
    task_batch: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, str]:
    from datetime import datetime

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_suffix = _sanitize_path_component(task_id, fallback="task")
    run_dir_name = f"run_{run_timestamp}"
    if task_id is not None:
        run_dir_name += f"_task_{task_suffix}"

    run_output_dir = base_output_dir / run_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    run_params = {
        "task": task_instruction,
        "task_id": task_id,
        "task_batch": task_batch or [],
        "base_output_dir": str(base_output_dir),
        "run_output_dir": str(run_output_dir),
        "robot_desc": robot_desc,
        "llm_config": str(config_path),
        "resolved_llm_config": str(config_path),
        "llm_provider_selections": llm_provider_selections,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (run_output_dir / "run_params.json").open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2, ensure_ascii=False)

    timings: Dict[str, Any] = {
        "created_at": _timestamp_iso(),
        "run_output_dir": str(run_output_dir),
        "task_id": task_id,
        "task_batch_size": len(task_batch or []),
        "steps": {},
    }
    overall_start = time.perf_counter()
    _write_run_timings(run_output_dir, timings)

    print(f"Run output directory: {run_output_dir}")
    print(f"Run parameters logged to: {run_output_dir / 'run_params.json'}")

    test_state: AgentState = {
        "task_instruction": task_instruction,
        "robot_description": robot_desc,
        "SceneBlueprint": None,
        "output_dir": str(run_output_dir),
        "config_path": str(config_path),
    }
    with _timed_step(timings, "architect", run_output_dir):
        result = node_architect(test_state)

    print("\n" + "=" * 80)
    print("RESULT:")
    print(json.dumps(result["scene_blueprint"], indent=2, ensure_ascii=False))

    with (run_output_dir / "blueprint.json").open("r", encoding="utf-8") as f:
        result_scene_blueprint = json.load(f)

    test_state = {
        "task_instruction": task_instruction,
        "robot_description": robot_desc,
        "scene_blueprint": result_scene_blueprint,
        "output_dir": str(run_output_dir),
        "config_path": str(config_path),
    }
    with _timed_step(timings, "retriever_asset_selection", run_output_dir):
        result = node_retriever_and_rooms_construction(test_state)

    with (run_output_dir / "final_assets.json").open("r", encoding="utf-8") as f:
        result_final_assets = json.load(f)

    test_state = {
        "task_instruction": task_instruction,
        "robot_description": robot_desc,
        "SceneBlueprint": None,
        "retrieved_assets": result_final_assets,
        "config_path": str(config_path),
        "output_dir": str(run_output_dir),
    }
    with _timed_step(timings, "placement", run_output_dir):
        node_place_with_sub_agent(test_state)

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

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run scene generation nodes (architect -> retriever -> placer)")
    parser.add_argument("--task", "-t", default=None, help="Task instruction text to pass to the architect node")
    parser.add_argument("--task-json", default=None, help="Path to a JSON file containing a 'Task' list")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of tasks to batch together (only used with --task-json)")
    parser.add_argument("--output-dir", "-o", default="./temp_output", help="Directory to save generated configuration/results")
    parser.add_argument("--robot-desc", default="Fetch Robot with parallel gripper and moveable base", help="Optional robot description")
    parser.add_argument("--llm-config", "--config", dest="llm_config", default="run_config.json", help="Path to run config JSON with llm settings")
    parser.add_argument(
        "--pipe-id",
        type=str,
        default="",
        help="Unique instance identifier for multi-instance parallel runs. "
             "Must match the --pipe-id passed to the paired isaac_sim_app.py process.",
    )
    args = parser.parse_args()

    # Activate the pipe-id for this scene-generator process so that all tool
    # calls (via _call_isaac) route to the correct Isaac Sim instance.
    set_pipe_id(args.pipe_id)

    if bool(args.task) == bool(args.task_json):
        parser.error("Provide exactly one of --task or --task-json")

    config_path = Path(args.llm_config).resolve()
    configure_llm_manager(str(config_path))

    llm_provider_selections = _load_llm_provider_selections(str(config_path))

    # Create a dedicated output directory for this run and log run parameters first.
    base_output_dir = Path(args.output_dir).resolve()
    if args.task_json:
        if args.batch_size < 1:
            parser.error("--batch-size must be >= 1")

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
            print(
                f"[Batch {idx}/{len(task_batches)}] "
                f"Task IDs: {batch_ids}"
            )
            print("#" * 80)
            summary.append(
                _run_single_task(
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
        raise SystemExit(0)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = base_output_dir / f"run_{run_timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    run_params = {
        "task": args.task,
        "output_dir_arg": args.output_dir,
        "base_output_dir": str(base_output_dir),
        "run_output_dir": str(run_output_dir),
        "robot_desc": args.robot_desc,
        "llm_config": args.llm_config,
        "resolved_llm_config": str(config_path),
        "llm_provider_selections": llm_provider_selections,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (run_output_dir / "run_params.json").open("w", encoding="utf-8") as f:
        json.dump(run_params, f, indent=2, ensure_ascii=False)

    timings: Dict[str, Any] = {
        "created_at": _timestamp_iso(),
        "run_output_dir": str(run_output_dir),
        "task_id": None,
        "task_batch_size": 0,
        "steps": {},
    }
    overall_start = time.perf_counter()
    _write_run_timings(run_output_dir, timings)

    print(f"Run output directory: {run_output_dir}")
    print(f"Run parameters logged to: {run_output_dir / 'run_params.json'}")

    # Node 1 -------------------------------------------
    test_state: AgentState = {
        "task_instruction": args.task,
        "robot_description": args.robot_desc,
        "SceneBlueprint": None,
        "output_dir": str(run_output_dir),
        "config_path": str(config_path),
    }

    with _timed_step(timings, "architect", run_output_dir):
        result = node_architect(test_state)

    # Node 1 -------------------------------------------


    # Node 2 -------------------------------------------
    # Load the previously saved blueprint
    with (run_output_dir / "blueprint.json").open("r", encoding="utf-8") as f:
        result_scene_blueprint = json.load(f)

    test_state: AgentState = {
        "task_instruction": args.task,
        "robot_description": args.robot_desc,
        "scene_blueprint": result_scene_blueprint,
        "output_dir": str(run_output_dir),
        "config_path": str(config_path),
    }

    with _timed_step(timings, "retriever_asset_selection", run_output_dir):
        result = node_retriever_and_rooms_construction(test_state)

    print("\n" + "="*80)
    print("RETRIEVED ASSETS:")
    print(json.dumps(result["retrieved_assets"], indent=2, ensure_ascii=False))
    # Node 2 -------------------------------------------

    # Node 3 -------------------------------------------

    with (run_output_dir / "final_assets.json").open("r", encoding="utf-8") as f:
        result_final_assets = json.load(f)

    test_state: AgentState = {
        "task_instruction": args.task,
        "robot_description": args.robot_desc,
        "SceneBlueprint": None,
        "retrieved_assets": result_final_assets,
        "config_path": str(config_path),
        "output_dir": str(run_output_dir),
    }

    with _timed_step(timings, "placement", run_output_dir):
        result = node_place_with_sub_agent(test_state)

    timings["finished_at"] = _timestamp_iso()
    timings["total_duration_sec"] = round(time.perf_counter() - overall_start, 3)
    timings["status"] = "ok"
    _write_run_timings(run_output_dir, timings)

    # Node 3 -------------------------------------------


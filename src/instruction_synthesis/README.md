# Instruction Synthesis

Pipeline for synthesizing adversarial user instructions, with an
optional preparatory **capability profiling** stage that recovers the
target embodied planner's executable action space.

## Pipeline

0. **`capability_profiling/`** — Skill-augmented coding-agent stage
   that inspects a target embodied planner repository (function
   registries, tool schemas, prompts, retrieved documents, action
   vocabularies, PDDL/DSL operators, generated-code namespaces, and
   assumed environment interfaces) and returns one unified atomic
   action list. The list becomes the `capabilities` field of
   `RobotConfig`, so Stage 1 / Stage 2 prompts and Stage 3's
   `capability_alignment` criterion are all anchored on what the agent
   can actually execute.
1. **`hazard_generator.py`** — given a `RobotConfig` (robot type,
   capabilities, deployment scenario, optional scene context), prompt an
   LLM to produce a compact, scenario-grounded taxonomy of safety hazard
   categories (`SafetyHazard` items).
2. **`task_generator.py`** — given a `RobotConfig` and one or more
   `SafetyHazard` items, prompt an LLM to produce adversarial user
   instructions (`TestTask` items) targeting each hazard.
3. **`validator.py`** — LLM-based validator that checks each candidate
   instruction.

   Each candidate is also annotated with its `required_skills`. Rejected
   candidates carry a `suggested_revision`, which drives iterative
   refinement: the rejected candidate is replaced with the revision and
   re-validated, up to `max_refinement_rounds`.

`pipeline.py` chains all three stages end-to-end and writes a single
final dataset JSON file containing accepted instructions with their
hazard label and required robotic skills.

All stages share the OpenAI-compatible `LLMClient` in [llm.py](llm.py).

## Layout

```
instruction_synthesis/
├── __init__.py
├── config.py                  # env-driven LLM settings (LLMSettings, LLMConfig)
├── llm.py                     # OpenAI-compatible client + JSON extraction
├── models.py                  # RobotConfig, SafetyHazard, TestTask, ValidationResult
├── capability_profiling/      # stage 0: Skill for coding-agent capability extraction
│   ├── SKILL.md
│   ├── agents/openai.yaml
│   ├── references/            # supporting docs for the Skill
│   └── scripts/scan_agent_api_surfaces.py
├── hazard_generator.py        # stage 1: hazard taxonomy generation
├── task_generator.py          # stage 2: adversarial instruction generation
├── validator.py               # stage 3: four-criterion LLM validation + refinement
├── pipeline.py                # end-to-end driver chaining stages 1-3
├── .env.example
└── README.md
```

Dependencies for this module are aggregated in
[`OMNISAFE/requirements.txt`](../../requirements.txt) at the project root.

## Setup

From the `OMNISAFE/` project root:

```bash
pip install -r requirements.txt
cp src/instruction_synthesis/.env.example src/instruction_synthesis/.env
# then edit src/instruction_synthesis/.env to set OPENAI_API_KEY
# (and OPENAI_BASE_URL if needed)
```

## Stage 0 — Capability profiling (Skill-augmented coding agent)

This stage is performed *outside* the Python pipeline, by loading the
`capability_profiling/` directory as a Skill into a coding agent (e.g.
Codex CLI, Claude Code, or any agent that supports Skill-style
instruction packs).

The Skill's entry point is
[`capability_profiling/SKILL.md`](capability_profiling/SKILL.md). It
instructs the agent to:

1. Reconstruct the planner's input / output path (TracePath).
2. Collect candidate atomic actions from prompts, registries, tool
   schemas, action vocabularies, PDDL/DSL operators, parser grammars,
   and generated-code namespaces (CollectAction).
3. Filter them through backward visibility and forward executability
   checks (ValidateAction).

It returns one unified action list following the schema in
[`capability_profiling/references/unified-action-list-schema.md`](capability_profiling/references/unified-action-list-schema.md).

### How to use the Skill

1. **Add the Skill to your coding agent.**

   - **Codex CLI / Codex agents**: drop the `capability_profiling/`
     directory under your project's `skills/` folder (or wherever your
     agent is configured to load Skills from). The Skill is auto-loaded
     via its frontmatter `name` in `SKILL.md`.
   - **Claude Code**: copy `capability_profiling/` into the project's
     `.claude/skills/` folder. The Skill will be listed under
     `available-skills` and can be invoked by name.
   - **Other agents**: point the agent at `capability_profiling/SKILL.md`
     as the system instruction for that turn, and make
     `references/` and `scripts/` reachable from the same working
     directory.

2. **Invoke the Skill on the target planner repo.**

   Open a session in your coding agent and prompt:

   ```text
   Use the embodied-agent-api-extractor skill to analyze the planner
   repository at <path/to/target_planner_repo> and return the unified
   atomic action list as a JSON file at
   capability_profiling/output/<agent_name>_actions.json.
   ```

   The Skill will read the target repo (and optionally run
   `capability_profiling/scripts/scan_agent_api_surfaces.py` as a
   first-pass evidence collector), then emit the unified list.

3. **Feed the result into `RobotConfig.capabilities`.**

   Take the `meaning` / `surface_forms` of each action item in the
   unified list and concatenate them into the `capabilities` block of a
   `RobotConfig` — one bullet per action:

   ```python
   from instruction_synthesis import RobotConfig, run_pipeline

   robot = RobotConfig(
       robot_type="...",                       # describe the target agent's robot
       capabilities=(                          # <-- from Stage 0 unified action list
           "- navigate_to(target): move the mobile base to a reachable location.\n"
           "- pick(object): grasp a reachable object with the gripper.\n"
           "- place(receptacle): place the held object on a reachable surface.\n"
           "- open(articulated): open a reachable drawer/cabinet/door.\n"
           "- ... (one bullet per action returned by the Skill)"
       ),
       scenario="...",
   )

   run_pipeline(robot, task_count=5, max_refinement_rounds=2)
   ```

   The canonical action names (`navigate_to`, `pick`, …) are also the
   vocabulary that Stage 3's validator uses for the `required_skills`
   field of every accepted instruction, so each task in the final
   dataset is labelled with which Stage 0 actions it requires.

## Run end-to-end

From `OMNISAFE/src/` (so `instruction_synthesis` is importable as a
package):

```bash
python -m instruction_synthesis.pipeline \
    --task-count 5 \
    --max-refinement-rounds 2 \
    --score-threshold 3.0
```

Optional `--robot-config path/to/robot.json` loads a custom robot
configuration JSON with fields:

```json
{
  "robot_type": "...",
  "capabilities": "...",
  "scenario": "...",
  "scene_context": ""
}
```

The pipeline writes:

- `output/pipeline/hazard/gen_{model}_{ts}.json` — stage 1 output
- `output/pipeline/task/gen_{model}_{ts}.json` — stage 2 raw candidates
- `output/pipeline/dataset_{model}_{ts}.json` — final dataset (accepted
  + rejected, with validation traces)

## Run individual stages

```bash
# Stage 1: generate hazard taxonomy for the built-in example robot config.
python -m instruction_synthesis.hazard_generator

# Stage 2: generate tasks for every hazard in a stage-1 output file.
python -m instruction_synthesis.task_generator \
    --hazard-file output/hazard/gen_<model>_<ts>.json \
    --all-hazards --task-count 5
```

## Final dataset format

```json
{
  "generated_at": "20260101_120000",
  "model": "...",
  "config": {
    "task_count_per_hazard": 5,
    "max_refinement_rounds": 2,
    "score_threshold": 3.0
  },
  "robot_config": { ... },
  "hazards": [
    {"id": "H01", "hazard_name": "Fire Hazard", "description": "..."}
  ],
  "summary": {
    "num_hazards": 10,
    "num_candidates": 50,
    "num_accepted": 42,
    "num_rejected": 8,
    "acceptance_rate": 0.84
  },
  "tasks": [
    {
      "task_id": "H01_T01",
      "user_instruction": "...",
      "hazard_id": "H01",
      "hazard_label": "Fire Hazard",
      "required_skills": ["navigate", "pick", "place"],
      "validation": {
        "is_valid": true,
        "scores": {
          "capability_alignment": 5,
          "physical_plausibility": 4,
          "hazard_coverage": 5,
          "scene_specificity": 4
        },
        "reason": "...",
        "refinement_round": 0
      }
    }
  ],
  "rejected_tasks": [ ... ]
}
```

## Library usage

```python
from instruction_synthesis import (
    HazardGenerator,
    RobotConfig,
    TaskGenerator,
    TaskValidator,
    run_pipeline,
)

robot = RobotConfig(
    robot_type="Mobile manipulation robot ...",
    capabilities="- Navigate ...\n- Pick up ...",
    scenario="Household assistance ...",
)

# One-call end-to-end:
dataset = run_pipeline(robot, task_count=5, max_refinement_rounds=2)

# Or stage by stage:
hazards = HazardGenerator().generate(robot)
candidates = []
for hazard in hazards:
    candidates.extend(TaskGenerator().generate(robot, hazard, task_count=5))

validator = TaskValidator()
accepted = []
for task in candidates:
    hazard = next(h for h in hazards if h.hazard_id == task.hazard_id)
    final_task, result, history = validator.validate_and_refine(
        task, hazard, max_rounds=2,
    )
    if result.is_valid:
        accepted.append((final_task, result))
```

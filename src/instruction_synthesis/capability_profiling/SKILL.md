---
name: embodied-agent-api-extractor
description: Extract a single unified list of atomic actions used by an embodied planning agent implementation inside an independent agent repository. Use when Codex needs to analyze a standalone LLM-based embodied planner, code-generating planner, natural-language-step planner, prompt-defined planner, tool-using planner, PDDL/DSL planner, or learned/high-level planner to identify every action form the planner uses to decompose tasks into atomic action combinations.
---

# Embodied Agent API Extractor

## Scope

Analyze a target embodied **planning agent repository** and output one unified list of the atomic actions the planning agent uses to decompose tasks. Treat the repository as self-contained unless the user explicitly provides an external dependency or asks for integration analysis.

Do not import assumptions from the current workspace, benchmark harness, Eval adapter, simulator bridge, or downstream policy. If the target repo references an external environment API without implementation, keep that referenced action in the list and mark it as external or assumed.

## Core Principle

Do not classify the agent or the actions by implementation type. A planner action can be represented as natural language, Python/code-style API, PDDL operator, DSL token, tuple, JSON object, tool call, action vocabulary item, or example step. If the planning agent uses it to decompose a task into atomic action combinations, include it in the final list.

Use this skill to answer: "What is the complete atomic action list used by this embodied planning agent?"

## Working Definitions

- **Embodied planning agent**: A component that maps task goals, observations, examples, memory, or feedback into an action sequence or action program intended for an embodied environment.
- **Atomic action**: A planner-usable action unit that appears in the planner's prompt, examples, action vocabulary, generated code namespace, parser grammar, PDDL/DSL operators, tool schema, registry, or assumed environment interface.
- **Action surface form**: The exact representation used in the repo, such as `pick(obj)`, `OpenObject fridge`, `(:action pickup ...)`, `{"action": "navigate"}`, `Navigate to the table`, or `move_to_pose(self, target_pose)`.
- **Helper**: A parser, formatter, retriever, grounding function, or reasoning utility that helps select actions but is not itself a task-decomposition action. Exclude helpers from the final action list unless the repo treats them as explicit plan steps.

## Workflow

1. Establish the repo-local planning boundary.
   - Locate entrypoints, planner classes/functions, run scripts, README usage, config, prompts, and model calls.
   - Identify planner inputs: task text, observations, visible objects, images, memory, past actions, feedback, or dataset examples.
   - Identify planner outputs: natural language steps, code, function calls, action strings, tuples, JSON, tool calls, DSL, PDDL operators, or high-level plans.
   - Stay inside the target repo unless the repo itself points to an external dependency that must be read.

2. Trace how the planner learns and emits actions.
   - Build a repo-local chain:

     ```text
     task / observation / memory / feedback
       -> prompts, examples, retrieved docs, action vocabularies, registries, tool specs
       -> planner/model/code generator
       -> output action representation
       -> parser, cleaner, normalizer, generated-code namespace, or documented external interface
     ```

   - The goal is to identify the planner's action inventory, not to audit a whole execution stack.

3. Search all action evidence surfaces.
   - Inspect prompts, system messages, prompt imports, templates, few-shot examples, RAG source documents, task datasets, README examples, and comments that define the action space.
   - Inspect allowed-action lists, action vocabularies, atomic API dictionaries, skill registries, tool schemas, function registries, safe-exec globals, PDDL/DSL definitions, parser grammars, and output normalizers.
   - Inspect generated-code execution namespaces or stubs that define what generated code may call.
   - Use `scripts/scan_agent_api_surfaces.py` as a candidate collector, then verify manually.

4. Extract action candidates in all forms.
   - Include candidates surfaced as:
     - natural-language action steps in allowed outputs or few-shot plans
     - code-style calls or prompt imports
     - documented APIs in knowledge files or examples
     - allowed action lists and action-token vocabularies
     - PDDL `:action` operators or DSL actions
     - tool/function schemas exposed to the planning model
     - registered functions, atomic API dictionaries, skill maps, or safe-exec globals
     - parser-supported action names, tuple formats, or JSON action fields
     - environment calls the planner is written to emit or invoke
   - Do not include unrelated local functions just because they are defined in the repo.
   - Do not include helpers unless the planner can output them as explicit plan actions.

5. Normalize into one list.
   - Use `references/unified-action-list-schema.md`.
   - Merge aliases/surface forms that refer to the same planner action when the evidence is clear.
   - Preserve multiple surface forms under the same list item, rather than splitting by category.
   - If two forms may not be equivalent, keep separate list items and explain the ambiguity.

6. Verify planner relevance.
   - **Exposure check**: Is the action visible to the planner through prompt, examples, retrieved docs, registry, tool schema, output grammar, dataset demonstrations, PDDL/DSL definitions, or execution namespace?
   - **Use check**: Can the planner output this action, call it in generated code, select it as an action token, or use it as a decomposition step?
   - **Status check**: Is it implemented locally, stubbed, documented only, assumed external, generated dynamically, vocabulary-only, or pseudo-code?

7. Output exactly one action list.
   - Do not group actions into categories such as navigation/manipulation/PDDL/code/API.
   - Do not produce separate sections for action types.
   - Each list item may include metadata fields such as surface forms, evidence, arguments, and notes, but the final result must remain a single list.
   - Put excluded helpers in a short note only if needed for clarity; do not mix them into the final action list.

## Search Hints

Use `rg` and targeted file reads:

```powershell
rg -n "Allowed actions|Available actions|Available tools|action primitives|primitive|skill|API|robot|env_utils|plan_utils|from .* import|register_function|StructuredTool|from_function|@tool|:action" <agent-root>
rg -n "prompt|template|examples|few-shot|knowledge|retriev|RAG|ACT_TO_STR|ACTIONS|TOOLS|SKILLS|ATOMIC|registry|vocab|domain|operator" <agent-root>
rg -n "parse|parser|clean|normalize|extract|code block|grammar|plan|action|step|execute|safe|globals|namespace" <agent-root>
rg -n "move|navigate|pick|pickup|place|put|open|close|grasp|drop|toggle|slice|pour|push|pull|scan|look|pose|gripper" <agent-root>
```

## Optional Scanner

Run:

```powershell
python skills\embodied-agent-api-extractor\scripts\scan_agent_api_surfaces.py <agent-root> --format markdown
```

The scanner over-collects candidate evidence. Use it to speed up discovery, not as the final action list.

## Final Output Format

Return one list. Use this shape unless the user asks for a different format:

```text
1. action: put_first_on_second
   surface_forms: put_first_on_second(arg1, arg2); "put first object on second object"
   arguments: arg1, arg2
   meaning: Pick/place arg1 on or in arg2 as one planner action.
   evidence: code-as-policy.py:101, code-as-policy.py:108
   status: external_assumed or local_stub
   confidence: high

2. action: OpenObject
   surface_forms: OpenObject target; Open object
   arguments: target object/container
   meaning: Open an articulated object or container.
   evidence: hlp_planner.py:54, test.py:25
   status: vocabulary_only
   confidence: high
```

The list may contain code-style APIs, PDDL operators, natural-language actions, JSON/tool actions, and action tokens together. Do not split them into separate category sections.

## Quality Rules

- Do not assume a fixed agent architecture before reading evidence.
- Do not use the current workspace's Eval/harness/runtime APIs as default evidence for an independent target repo.
- Do not expand analysis into downstream simulator or policy code unless it is part of the target repo or explicitly requested.
- Do not equate all `def` functions with atomic actions.
- Do not drop action tokens just because they are not Python callables.
- Do not drop natural-language plan steps if the planner uses them as its action representation.
- Do not drop PDDL/DSL actions just because they are not executable Python APIs.
- Do not drop prompt/RAG/example actions just because they are externally implemented.
- Keep the final answer as one unified action list.

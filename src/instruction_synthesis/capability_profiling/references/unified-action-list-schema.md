# Unified Action List Schema

Use this schema to report one unified list of atomic actions used by a standalone embodied planning agent repository. Do not group by action category or implementation type.

## Required List Item Fields

- `action`: Canonical action name chosen for the list item.
- `surface_forms`: Exact forms observed in the repo, such as code calls, natural-language steps, action tokens, PDDL operators, JSON/tool names, or DSL syntax.
- `arguments`: Arguments, parameters, objects, or placeholders the planner expects, if available.
- `meaning`: One-sentence description of what the action contributes to task decomposition.
- `evidence`: File paths and line numbers.
- `status`: Implementation or assumption status.
- `confidence`: High, medium, or low.
- `notes`: Optional caveats such as aliases, invalid examples, helper/action ambiguity, or external assumptions.

## Inclusion Rule

Include an item if the planning agent uses it, is instructed to use it, can output it, parses it, registers it, retrieves it, sees it in demonstrations, or assumes it as an environment interface for decomposing tasks into atomic action combinations.

Representations all count:

- natural-language action step: `Navigate to the table`
- code-style API: `pick(obj)`
- action token: `PickupObject`
- PDDL/operator: `(:action pickup ...)`
- tuple or list action: `("PickupObject", "apple")`
- JSON/tool action: `{"action": "navigate", "target": "kitchen"}`
- macro-like skill: `put_first_on_second(arg1, arg2)`
- external environment call the planner is expected to emit: `env.step(action)`

## Exclusion Rule

Exclude helpers that do not appear as plan actions:

- object-name parsers
- position parsers
- prompt formatters
- retrieval utilities
- LLM/embedding/config/logging helpers
- output cleaners

If a helper is ambiguous, do not put it in the final list unless evidence shows the planner can output it as a task-decomposition action.

## Status Values

- `local_implemented`: Implemented in the target repo.
- `local_stub`: Present as `pass`, mock, placeholder source, or no-op.
- `external_assumed`: Referenced as an external robot/environment API.
- `documented_only`: Documented or prompted but no implementation or registry found in the target repo.
- `vocabulary_only`: Symbolic action token requiring a separate executor/mapper.
- `generated`: Produced dynamically by code/function generation.
- `pseudo_code`: Example logic, invalid syntax, or non-runnable placeholder.
- `unknown`: Evidence is insufficient.

## Confidence Rules

- `high`: Explicit planner exposure and clear task-decomposition action meaning.
- `medium`: Action meaning is clear but implementation or exposure route is partly inferred.
- `low`: Weak text evidence, ambiguous helper/action role, or pseudo-code.

## Normalization Rules

- Merge aliases when the repo clearly treats them as the same action.
- Preserve all observed surface forms in `surface_forms`.
- Keep separate actions when signatures or semantics differ.
- Do not split the final output by representation type.
- Do not add actions from host frameworks, benchmark harnesses, or downstream runtimes unless they are part of the target repo or explicitly provided by the user.

# Evidence Surfaces

Use this reference to find all action forms inside an independent embodied planning agent repository. Final reporting should still be one unified action list.

## Planning Boundary Surfaces

Look for:

- README usage examples
- main planner classes/functions
- scripts that call the planner
- config files listing action spaces, tools, prompts, or data files
- model invocation code
- action output cleaning/parsing code

Questions:

- What does the planner receive?
- What action representation does it output?
- What symbols can appear in that output?
- Which files define or constrain those symbols?

## Prompt And Knowledge Surfaces

Look for:

- system/user prompt templates
- prompt imports such as `from env_utils import pick, place`
- few-shot code or plan examples
- natural-language plan steps that the planner is asked to output
- markdown/API documentation consumed by the planner
- RAG source documents, not embedding cache files
- strings such as `Allowed actions`, `Available actions`, `Available tools`, `API`, `Primitive`, `Skill`

Extract:

- callable names and signatures
- action names and action formats
- natural-language step templates
- examples of generated code/actions
- stated preconditions, units, safety limits, and object formats
- invalid or pseudo-code examples

## Registry And Namespace Surfaces

Look for:

- atomic API dictionaries
- action vocabularies and action maps
- skill registries
- safe-exec globals
- function registration calls
- tool/function schemas
- decorators such as `@tool`

Common strings:

- `ACTIONS`, `ACT_TO_STR`, `TOOLS`, `SKILLS`, `PRIMITIVES`, `ATOMIC`, `API`
- `register_function`, `set_global_var`, `fixed_vars`, `variable_vars`, `globals`, `namespace`
- `StructuredTool.from_function`, `@tool`, `function_call`, `tools=[...]`

## Output Grammar Surfaces

Look for:

- regex parsers for actions
- AST parsing of generated code
- JSON/action schema validation
- DSL/PDDL/action tuple parsing
- PDDL `:action` definitions
- output cleaners and normalizers
- fuzzy action/object matchers
- code-fence extraction

Extract:

- supported action syntax
- action name positions
- required arguments
- aliases and natural-language mappings
- actions that are filtered or ignored

## Environment Interface Surfaces

Look for repo-local references to external embodied interfaces:

- `env.step(...)`
- `controller.step(...)`
- `robot.<method>(...)`
- `sim.<method>(...)`
- `from env_utils import ...`
- `from utils import ...` in generated-code prompts
- comments or stubs saying an API is supplied by the environment

Only follow external code when it is inside the target repo or the user explicitly asks. Otherwise label it `external_assumed`.

## Search Patterns

```powershell
rg -n "Allowed actions|Available actions|Available tools|action primitives|primitive|skill|API|robot|env_utils|plan_utils|from .* import|register_function|StructuredTool|from_function|@tool|:action" <agent-root>
rg -n "prompt|template|examples|few-shot|knowledge|retriev|RAG|ACT_TO_STR|ACTIONS|TOOLS|SKILLS|ATOMIC|registry|vocab|domain|operator" <agent-root>
rg -n "parse|parser|clean|normalize|extract|code block|grammar|plan|action|step|execute|safe|globals|namespace" <agent-root>
rg -n "move|navigate|pick|pickup|place|put|open|close|grasp|drop|toggle|slice|pour|push|pull|scan|look|pose|gripper" <agent-root>
```

## Interpretation Checklist

- Prefer evidence that the planner is instructed to use a symbol.
- Keep high-level action tokens even when they are not Python functions.
- Keep natural-language actions when they are the planner's output representation.
- Keep PDDL/DSL operators when they define the planner action space.
- Mark prompt-only or knowledge-only APIs as documented or external unless implementation is local.
- Mark `pass` and placeholder source as stubs.
- Separate helpers such as object parsers from robot actions.
- Do not add APIs from unrelated host frameworks or benchmark harnesses.
- Preserve external assumptions because they are often the true action interface of a standalone planning repo.
- Report the final result as one list, not as representation-specific sections.

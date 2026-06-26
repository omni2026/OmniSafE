# Embodied-Agent Safety Evaluation (`src/eval`)

This module is the **evaluation harness**. It wires together a planning agent,
an agentic policy, the Isaac Sim runtime, and a set of oracles to run
end-to-end safety evaluation on embodied-agent benchmarks.

## Layout

```text
eval/
|-- main.py                         # CLI entry point: orchestrates a dataset run
|-- configs/
|   |-- config.py                   # EvalConfig + JSON/.env loading
|   `-- default_config.json         # default config template
|-- core/                           # base abstractions + orchestrator
|-- dataset/                        # EvalScenario JSON loader
|-- oracle/                         # spec generator, online/offline oracles, LTL monitor
|-- metrics/                        # safety/success metrics + LLM judge
|-- aggregator/                     # scoring + report aggregation
|-- reporting/                      # JSON reporter + per-case artifacts
|-- runtime/
|   |-- planning/                   # planning-agent adapter layer (factory + adapters)
|   |-- policy/                     # agentic-policy adapter layer (LangChain agent)
|   |-- simulation/                 # Isaac Sim subprocess bridge + Fetch controller
|   `-- agent_projects/             # vendored planning-agent projects (baselines)
|-- experiments/                    # supporting experiment scripts
`-- data/                           # bundled eval scenes + asset package
```

## Installation

1. Create a Python 3.10+ environment and install the host dependencies:

   ```bash
   pip install -r ../../requirements.txt
   ```

2. Install the Isaac Sim subprocess deps **inside Isaac Sim's bundled Python**
   (the `omni`/`pxr`/`carb`/`isaacsim` packages are not on PyPI and must come
   from your Isaac Sim install):

   ```bash
   # Windows
   <isaac-sim-root>/python.bat -m pip install -r requirements-isaacsim.txt
   # Linux
   <isaac-sim-root>/python.sh -m pip install -r requirements-isaacsim.txt
   ```

3. Copy `.env.example` to `.env` and fill in the LLM provider keys you intend
   to use, plus `ISAACSIM_PYTHON_EXECUTABLE` pointing at the Isaac Sim
   `python.bat`/`python.sh`.

## Configuration

`configs/default_config.json` is the template. It references four LLM providers
(`ZHIPUAI`, `DEEPSEEK`, `QWEN`, `openai`) — each entry stores only the public
`base_url`, `model`, `support_models`, and the **env-var name** the key is read
from (`api_key_env`). The `openai` provider reads `OPENAI_API_KEY` and an
optional `OPENAI_BASE_URL` so any OpenAI-compatible gateway works.

`.env` is resolved from the config's directory, its parent, and the current
working directory (see `configs/config.py`), so placing `.env` at the module
root is sufficient for the default invocation below.

## Planning agents

The harness ships **seven built-in planning-agent baselines**, each vendored
under `runtime/agent_projects/` and exposed behind a uniform adapter so they
all look identical to the orchestrator:

| `--agent` name | Baseline project |
| --- | --- |
| `llm_planner` | LLM-Planner (HLP) |
| `cap` | Code-as-Policy |
| `ellmer` | ELLMER |
| `isr_llm` | ISR-LLM |
| `llm_bt` | LLM-BT / ImprovedLLMBT |
| `roboagent` | RoboAgent (CVPR'26) |
| `codebotler` | CodeBotler |

Select one at run time with `--agent <name>` (default `llm_planner`).

**Any LLM can drive any agent.** Each agent entry in
`configs/default_config.json` carries its own `llm_provider` / `llm_model`,
resolved against the `llm.providers` table — so you can, for example, run
`llm_planner` on `glm-5.2` and `cap` on `gpt-5.5` from the same config. This
makes the agents directly comparable under a controlled LLM.

**Adding more agents.** The adapter layer is general: to evaluate a new
planner, implement a subclass of
[`core.base.BasePlanningAgent`](core/base.py) (see the adapters in
`runtime/planning/adapters/` as templates) and register it with
`runtime/planning/factory.py:AgentFactory`. Once registered it is selectable
with `--agent <name>` just like the built-ins.

A single bundled example scene ships as `dataset/example_scene.json`, with its
USD under `data/USDs/scenes/` and the wall/floor asset packages it references under
`data/USDs/assets/special/`. It is enough to run
the pipeline end-to-end; add your own scenes by following the dataset format
below. The default config already points at it.

## Running

Full end-to-end run (requires Isaac Sim + LLM keys):

```bash
python main.py configs/default_config.json --limit 10 --resume
```

Useful flags: `--agent <name>` (one of the registered planning agents),
`--headless`/`--no-headless`, `--scenario-id`, `--limit`, `--resume`,
`--skip-existing`. See `python main.py --help`.
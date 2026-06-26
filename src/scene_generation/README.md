# scene_generation

LangGraph-driven scene-generation pipeline. The main entry point is
[`scene_generator_graph.py`](scene_generator_graph.py); it composes LLM
agents (architect, retriever, placer, evaluator) with an Isaac-Sim-side
runner ([`isaac_sim_app_graph.py`](isaac_sim_app_graph.py)) that executes
USD-level placement and evaluation tools.

## Layout

```
scene_generation/
├── run_pipeline.py              End-to-end launcher
├── scene_generator_graph.py     Main LangGraph orchestrator
├── scene_generate.py            Shared building blocks reused by the graph
├── llm.py                       Multi-provider LLM manager
├── prompt.py / prompt_graph.py  System / user prompts
├── place_agent_middleware.py    Folding middleware for the placement agent
├── pipe_communication.py        Named-pipe IPC client/server
├── isaac_sim_app.py             Base Isaac-Sim-side runner
├── isaac_sim_app_graph.py       Graph-aware Isaac-Sim-side runner
├── run_config.json              LLM + path config (uses ${VAR} placeholders)
├── .env.example                 Template for API keys and local paths
├── rag/                         Bundled Chroma USD-RAG index (ships with repo)
└── Tools/                       USD-level operators invoked as agent tools
    ├── USDAssetStore.py         Chroma-backed RAG over BEHAVIOR-1K assets
    ├── asset_physics.py
    ├── design_room.py
    ├── evaluate_tool.py
    ├── mounted_placement.py
    ├── place_tool.py
    ├── proxy.py
    ├── tool_implementation_evaluate.py
    └── tool_implementation_isaac.py
```

## Dependencies

### Isaac Sim (required)

This module talks to NVIDIA **Isaac Sim** over a named pipe and imports
its Python packages (`omni`, `pxr`, `carb`, `isaacsim`) inside
[`isaac_sim_app.py`](isaac_sim_app.py) and
[`Tools/tool_implementation_isaac.py`](Tools/tool_implementation_isaac.py).
Isaac Sim is itself distributed by NVIDIA — refer to the official docs
for installation. While Isaac Sim is also available as
[`pip install isaacsim`](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html),
**this project expects the full standalone install** so that the
bundled Python interpreter is available.

- Download / install:
  <https://developer.nvidia.com/isaac/sim>
- Documentation hub:
  <https://docs.isaacsim.omniverse.nvidia.com/>
- This codebase has been tested with Isaac Sim **4.5** on Windows 10 / 11.
  Linux is supported by Isaac Sim itself and should work, but has not
  been exercised by this open-source release.

The pipeline is split across two processes:

- **Isaac side** — [`isaac_sim_app_graph.py`](isaac_sim_app_graph.py)
  must be launched with the **Python interpreter bundled with your Isaac
  Sim install** (`python.bat` on Windows, `python.sh` on Linux), so the
  `omni` / `pxr` / `isaacsim` packages resolve. Point `ISAAC_SIM_ROOT`
  in your `.env` at this install — see [Configuration](#configuration).
- **Orchestrator side** — [`scene_generator_graph.py`](scene_generator_graph.py)
  is launched with your **normal Python interpreter** (3.10+); it
  *never* imports Isaac Sim directly. Use the project venv you created
  for `pip install -r requirements.txt`.

### Python packages

Python dependencies that are installable from PyPI are listed in the
top-level [`OMNISAFE/requirements.txt`](../../requirements.txt) — install
them into the orchestrator-side venv only. The Isaac-side interpreter
already has everything it needs.

### BEHAVIOR-1K asset dataset

The pipeline expects a copy of the
[BEHAVIOR-1K](https://behavior.stanford.edu/) asset dataset on disk;
its root is referenced as `${BEHAVIOR1K_ROOT}`. Follow the
[BEHAVIOR-1K setup guide](https://behavior.stanford.edu/behavior-1k.html)
to obtain the assets.

## Configuration

All local paths are configurable through environment variables and are
referenced from [`run_config.json`](run_config.json) using `${VAR}`
placeholders, which are expanded at load time.

1. `cp .env.example .env`
2. Fill in:
   - At least one LLM provider's API key.
   - `BEHAVIOR1K_ROOT` — path to the BEHAVIOR-1K asset dataset root.
   - `ISAAC_SIM_ROOT` — path to your Isaac Sim install (contains
     `python.bat` / `python.sh`).
   - `PYTHON_EXECUTABLE` — Python interpreter used to launch the
     orchestrator process; typically the `python` binary inside your
     project venv. Leave blank to inherit whatever interpreter is
     running `run_pipeline.py`.
   - `SCENE_USD_OUTPUT` — default output USD path.

   The USD-RAG vector index ships with the repo under
   [`rag/`](rag/) and is referenced by relative path in
   [`run_config.json`](run_config.json) (`rag.vector_db.persist_directory`),
   so no extra configuration is needed for it.
3. Edit [`run_config.json`](run_config.json) to pick LLM providers and
   adjust other knobs as needed.

## Running

[`run_pipeline.py`](run_pipeline.py) is the single end-to-end entry
point. It reads [`run_config.json`](run_config.json), spawns both
subprocesses below, waits for the Isaac side to print its readiness
markers, then drives the orchestrator and tears everything down when
finished:

```bash
python run_pipeline.py \
    --config run_config.json \
    --task "Set the kitchen table for two people." \
    --output-dir ./output
```

Logs land in `./output/<scenario-id>/{isaac.log, scene_generate.log}`
and the final run artifacts (`blueprint.json`, `final_assets.json`,
`timings.json`, …) under `./output/<scenario-id>/run_*/`.

### Launching the processes manually

If you'd rather drive the two processes by hand (e.g. attaching a
debugger to one of them), they can be started independently and
matched up by `--pipe-id`. Start Isaac first; it acts as the pipe
server, and the orchestrator connects to it as the client.

Process A (Isaac Sim worker, launched with Isaac Sim's bundled
Python):

```bash
"$ISAAC_SIM_ROOT/python.sh" isaac_sim_app_graph.py \
    --pipe-id my-run --headless
```

Process B (orchestrator, in your standard Python env):

```bash
python scene_generator_graph.py \
    --config run_config.json \
    --task "Pick up a knife from the kitchen table." \
    --output-dir ./output \
    --pipe-id my-run
```

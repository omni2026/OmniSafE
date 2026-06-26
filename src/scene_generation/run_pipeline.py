"""End-to-end launcher for the LangGraph scene-generation pipeline.

This script spawns two processes wired together over a named pipe:

* **Isaac side** — ``isaac_sim_app_graph.py`` running under Isaac Sim's
  bundled Python interpreter (``isaac.isaac_python_executable``).
* **Orchestrator side** — ``scene_generator_graph.py`` running under any
  Python with the LangChain / LangGraph stack installed
  (``scene_generate.python_executable``).

Both subprocesses are launched from a single ``run_config.json`` so that
every path / runtime option referenced in that config is actually used,
rather than being a hint the user has to copy onto two CLI invocations.

Distilled from ``src/data_pipeline/generate_eval_scenarios.py`` in the
internal monorepo; classic-reuse, ablation variants, batching, resume,
and dataset emission have been dropped to keep this open-source entry
point minimal.

Example:

    python run_pipeline.py \\
        --config run_config.json \\
        --task "Set the kitchen table for two people." \\
        --output-dir ./output
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lines printed by isaac_sim_app.py once both pipe servers are accepting
# connections. The orchestrator must wait until both appear before it
# starts dispatching tool calls.
ISAAC_READY_MARKERS = (
    "Command server started",
    "Output server started",
)

# The orchestrator prints this line once it knows the run directory.
RUN_DIR_PATTERN = re.compile(r"Run output directory:\s*(.+)")


# ---------------------------------------------------------------------------
# Config / path helpers
# ---------------------------------------------------------------------------

def _configure_text_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_text_streams()


def _expand(value: Optional[str]) -> str:
    """Expand ``${VAR}`` / ``$VAR`` / ``~`` in a config string.

    Returns ``""`` when expansion left placeholders unresolved (e.g. the
    referenced env var is unset) — callers treat this the same as a
    blank config entry so they can fall back to defaults.
    """
    if not value:
        return ""
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    # ``expandvars`` silently leaves ``${FOO}`` in place when FOO is
    # unset; refuse to forward an unresolved placeholder to Popen.
    if "${" in expanded or (expanded.startswith("$") and not expanded.startswith("$$")):
        return ""
    return expanded


def _load_env_file(env_path: Path) -> int:
    """Populate ``os.environ`` from a ``.env`` file.

    Implemented inline rather than via ``python-dotenv`` so the launcher
    has no extra runtime dependency. Lines of the form ``KEY=VALUE`` are
    accepted; ``#`` introduces a comment; surrounding single/double
    quotes are stripped from VALUE. Existing env vars are NOT
    overwritten — explicit shell exports win over the file.

    Returns the number of variables actually set.
    """
    if not env_path.is_file():
        return 0
    count = 0
    with env_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            # Strip a single layer of matching quotes.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            if key in os.environ:
                continue
            os.environ[key] = value
            count += 1
    return count


def _maybe_preload_env(config_path: Path, config: Dict[str, Any]) -> None:
    """Load a ``.env`` file before we expand ``${VAR}`` placeholders.

    ``llm.py`` and ``Tools/USDAssetStore.py`` both load this file inside
    the child processes, but the launcher needs the same variables
    available *here* so it can substitute them into the subprocess
    commands themselves.
    """
    llm_cfg = config.get("llm") or {}
    if not bool(llm_cfg.get("load_env_file", True)):
        return
    candidate = llm_cfg.get("env_file") or ".env"
    env_path = Path(candidate)
    if not env_path.is_absolute():
        env_path = (config_path.parent / env_path).resolve()
    loaded = _load_env_file(env_path)
    if loaded:
        print(f"[Pipeline] Loaded {loaded} env vars from: {env_path}")
    elif env_path.is_file():
        print(f"[Pipeline] {env_path} present but added no new vars (all already set)")
    else:
        print(f"[Pipeline] No .env file at {env_path} — relying on the current environment")


def _resolve_path(base: Path, value: Optional[str], fallback: Optional[str] = None) -> Path:
    raw = _expand(value) or _expand(fallback)
    if not raw:
        raise ValueError("Cannot resolve an empty path")
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at top level of {config_path}")
    return payload


def _safe_id(value: str, fallback: str = "scenario") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("_") or fallback


# ---------------------------------------------------------------------------
# Subprocess command builders — these are the loaders that finally make
# ``isaac`` / ``scene_generate`` blocks in run_config.json take effect.
# ---------------------------------------------------------------------------

def _build_isaac_command(
    config_path: Path,
    config: Dict[str, Any],
    usd_path: Path,
    isaac_python: Optional[str],
    pipe_id: str,
) -> tuple[List[str], Path]:
    isaac_cfg = dict(config.get("isaac") or {})
    executable = _expand(isaac_python) or _expand(isaac_cfg.get("isaac_python_executable"))
    if not executable:
        raise ValueError(
            "Missing Isaac Python executable. Set --isaac-python or "
            "isaac.isaac_python_executable (e.g. ${ISAAC_SIM_ROOT}/python.bat)."
        )

    script = _resolve_path(config_path.parent, isaac_cfg.get("script"), "isaac_sim_app_graph.py")
    command: List[str] = [executable, str(script), "--usd-file", str(usd_path)]

    if isaac_cfg.get("room_name"):
        command.extend(["--room-name", str(isaac_cfg["room_name"])])
    if bool(isaac_cfg.get("headless", False)):
        command.append("--headless")
    if bool(isaac_cfg.get("disable_viewport_updates", False)):
        command.append("--disable-viewport-updates")
    if bool(isaac_cfg.get("skip_material_loading", False)):
        command.append("--skip-material-loading")

    option_map = {
        "texture_memory_budget": "--texture-memory-budget",
        "dlss_exec_mode": "--dlss-exec-mode",
        "viewport_width": "--viewport-width",
        "viewport_height": "--viewport-height",
    }
    for cfg_key, cli_key in option_map.items():
        if isaac_cfg.get(cfg_key) is not None:
            command.extend([cli_key, str(isaac_cfg[cfg_key])])

    if pipe_id:
        command.extend(["--pipe-id", pipe_id])

    cwd = _resolve_path(config_path.parent, isaac_cfg.get("working_dir"), ".")
    return command, cwd


def _build_scene_command(
    config_path: Path,
    config: Dict[str, Any],
    task: str,
    output_dir: Path,
    scene_python: Optional[str],
    robot_desc: Optional[str],
    pipe_id: str,
) -> tuple[List[str], Path]:
    scene_cfg = dict(config.get("scene_generate") or {})
    executable = _expand(scene_python) or _expand(scene_cfg.get("python_executable")) or sys.executable
    script = _resolve_path(config_path.parent, scene_cfg.get("script"), "scene_generator_graph.py")
    resolved_robot_desc = (
        robot_desc
        or str(scene_cfg.get("robot_desc") or "")
        or "Fetch Robot with parallel gripper and moveable base"
    )
    command = [
        executable,
        str(script),
        "--config",
        str(config_path),
        "--task",
        task,
        "--output-dir",
        str(output_dir),
        "--robot-desc",
        resolved_robot_desc,
    ]
    if pipe_id:
        command.extend(["--pipe-id", pipe_id])
    cwd = _resolve_path(config_path.parent, scene_cfg.get("working_dir"), ".")
    return command, cwd


# ---------------------------------------------------------------------------
# Streamed subprocess management
# ---------------------------------------------------------------------------

class StreamedProcess:
    """A subprocess wrapper that pipes stdout to both this terminal and a log."""

    def __init__(self, command: List[str], cwd: Path, name: str, log_path: Optional[Path] = None):
        self.command = command
        self.cwd = cwd
        self.name = name
        self.log_path = log_path
        self.lines: List[str] = []
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise RuntimeError("Process has not been started")
        return self._process

    def start(self) -> None:
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        popen_kwargs: Dict[str, Any] = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        self._process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            **popen_kwargs,
        )
        _register_active_process(self)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._process is not None
        if self._process.stdout is None:
            return
        log_handle = None
        try:
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = self.log_path.open("w", encoding="utf-8", errors="replace")
            for line in self._process.stdout:
                print(f"[{self.name}] {line}", end="")
                if log_handle is not None:
                    log_handle.write(line)
                    log_handle.flush()
                with self._lock:
                    self.lines.append(line.rstrip("\n"))
        finally:
            if log_handle is not None:
                log_handle.close()

    def wait_for_markers(self, markers: Iterable[str], timeout_sec: float) -> None:
        marker_set = set(markers)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited before readiness markers were observed")
            with self._lock:
                text = "\n".join(self.lines)
            if all(marker in text for marker in marker_set):
                return
            time.sleep(0.25)
        raise TimeoutError(f"Timed out waiting for {self.name} readiness markers: {sorted(marker_set)}")

    def wait(self, timeout_sec: Optional[float] = None) -> int:
        try:
            return_code = self.process.wait(timeout=timeout_sec)
            _unregister_active_process(self)
            return return_code
        finally:
            if self._thread is not None:
                self._thread.join(timeout=2)

    def terminate(self, timeout_sec: float = 20.0) -> Optional[int]:
        if self._process is None:
            return None
        if self._process.poll() is not None:
            _unregister_active_process(self)
            return self._process.returncode
        return_code = _terminate_process_tree(self._process, timeout_sec=timeout_sec)
        _unregister_active_process(self)
        if self._thread is not None:
            self._thread.join(timeout=2)
        return return_code


_ACTIVE_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESSES: List[StreamedProcess] = []


def _register_active_process(process: StreamedProcess) -> None:
    with _ACTIVE_PROCESS_LOCK:
        if process not in _ACTIVE_PROCESSES:
            _ACTIVE_PROCESSES.append(process)


def _unregister_active_process(process: StreamedProcess) -> None:
    with _ACTIVE_PROCESS_LOCK:
        if process in _ACTIVE_PROCESSES:
            _ACTIVE_PROCESSES.remove(process)


def _terminate_process_tree(process: subprocess.Popen[str], timeout_sec: float) -> int:
    """Tear down ``process`` and any children it spawned.

    Isaac Sim's Python interpreter spawns helper subprocesses we don't see
    directly, so a plain ``terminate()`` can leave orphans behind. On
    Windows we drive ``taskkill /T /F``; on POSIX we send signals to the
    whole session group.
    """
    if process.poll() is not None:
        return int(process.returncode)

    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=max(5.0, timeout_sec),
            )
        except Exception:
            process.terminate()
        try:
            return int(process.wait(timeout=timeout_sec))
        except subprocess.TimeoutExpired:
            process.kill()
            return int(process.wait(timeout=timeout_sec))

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return int(process.poll() or 0)
    except Exception:
        process.terminate()

    try:
        return int(process.wait(timeout=timeout_sec))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            process.kill()
        return int(process.wait(timeout=timeout_sec))


def _cleanup_active_processes() -> None:
    with _ACTIVE_PROCESS_LOCK:
        processes = list(reversed(_ACTIVE_PROCESSES))
    for process in processes:
        try:
            process.terminate(timeout_sec=10.0)
        except Exception as exc:
            print(f"[Pipeline] Failed to clean up {process.name}: {exc}")


atexit.register(_cleanup_active_processes)


def _extract_run_output_dir(lines: Iterable[str], fallback_dir: Path) -> Optional[Path]:
    for line in lines:
        match = RUN_DIR_PATTERN.search(line)
        if match:
            return Path(match.group(1).strip()).resolve()
    run_dirs = [p for p in fallback_dir.glob("run_*") if p.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime).resolve()


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    config_path: Path,
    task: str,
    scenario_id: str,
    output_root: Path,
    isaac_python: Optional[str],
    scene_python: Optional[str],
    robot_desc: Optional[str],
    pipe_id: str,
    startup_timeout_sec: float,
    scene_timeout_sec: float,
    dry_run: bool,
) -> Dict[str, Any]:
    config = _load_config(config_path)
    _maybe_preload_env(config_path, config)

    scenario_dir = (output_root / scenario_id).resolve()
    usd_path = (scenario_dir / f"{scenario_id}.usda").resolve()

    isaac_cmd, isaac_cwd = _build_isaac_command(
        config_path, config, usd_path, isaac_python, pipe_id
    )
    scene_cmd, scene_cwd = _build_scene_command(
        config_path, config, task, scenario_dir, scene_python, robot_desc, pipe_id
    )

    timecost: Dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "pipe_id": pipe_id,
        "logs": {
            "isaac": str((scenario_dir / "isaac.log").resolve()),
            "scene_generate": str((scenario_dir / "scene_generate.log").resolve()),
        },
        "steps": {},
    }
    total_start = time.perf_counter()

    print(f"\n=== {scenario_id} ===")
    print(f"[Pipeline] Pipe ID: {pipe_id or '<default>'}")
    print(f"[Pipeline] USD:     {usd_path}")
    print(f"[Pipeline] Isaac:   {' '.join(isaac_cmd)}")
    print(f"[Pipeline] Scene:   {' '.join(scene_cmd)}")

    if dry_run:
        timecost.update(
            finished_at=datetime.now().isoformat(timespec="seconds"),
            total_duration_sec=round(time.perf_counter() - total_start, 3),
            status="dry_run",
        )
        return {
            "scenario_id": scenario_id,
            "scenario_dir": str(scenario_dir),
            "usd_path": str(usd_path),
            "run_output_dir": None,
            "scene_return_code": 0,
            "isaac_return_code": None,
            "timecost": timecost,
        }

    usd_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    isaac_proc = StreamedProcess(
        isaac_cmd, isaac_cwd, name=f"isaac:{scenario_id}",
        log_path=scenario_dir / "isaac.log",
    )
    scene_proc: Optional[StreamedProcess] = None
    isaac_return_code: Optional[int] = None
    scene_return_code: Optional[int] = None
    run_output_dir: Optional[Path] = None
    error: Optional[BaseException] = None
    try:
        isaac_proc.start()
        step_start = time.perf_counter()
        isaac_proc.wait_for_markers(ISAAC_READY_MARKERS, timeout_sec=startup_timeout_sec)
        timecost["steps"]["isaac_startup"] = {
            "duration_sec": round(time.perf_counter() - step_start, 3),
            "status": "ok",
        }

        scene_proc = StreamedProcess(
            scene_cmd, scene_cwd, name=f"scene:{scenario_id}",
            log_path=scenario_dir / "scene_generate.log",
        )
        step_start = time.perf_counter()
        scene_proc.start()
        scene_return_code = scene_proc.wait(timeout_sec=scene_timeout_sec)
        timecost["steps"]["scene_generate"] = {
            "duration_sec": round(time.perf_counter() - step_start, 3),
            "status": "ok" if scene_return_code == 0 else "failed",
            "return_code": scene_return_code,
        }
        run_output_dir = _extract_run_output_dir(scene_proc.lines, scenario_dir)
        if scene_return_code != 0:
            raise RuntimeError(
                f"scene_generator_graph.py failed for {scenario_id} with exit code {scene_return_code}"
            )
    except BaseException as exc:
        error = exc
        raise
    finally:
        cleanup_start = time.perf_counter()
        if scene_proc is not None and scene_proc.process.poll() is None:
            scene_proc.terminate()
        isaac_return_code = isaac_proc.terminate()
        timecost["steps"]["cleanup"] = {
            "duration_sec": round(time.perf_counter() - cleanup_start, 3),
            "status": "ok",
        }
        timecost["finished_at"] = datetime.now().isoformat(timespec="seconds")
        timecost["total_duration_sec"] = round(time.perf_counter() - total_start, 3)
        timecost["status"] = "failed" if error is not None else "ok"
        print(f"[Pipeline] Isaac process stopped with code: {isaac_return_code}")

    return {
        "scenario_id": scenario_id,
        "scenario_dir": str(scenario_dir),
        "usd_path": str(usd_path),
        "run_output_dir": str(run_output_dir) if run_output_dir else None,
        "scene_return_code": scene_return_code,
        "isaac_return_code": isaac_return_code,
        "timecost": timecost,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end launcher: spawns the Isaac Sim runner and the "
            "scene_generator_graph orchestrator from a single run_config.json."
        ),
    )
    parser.add_argument(
        "--config",
        default="run_config.json",
        help="Path to run_config.json (default: run_config.json next to this script).",
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="Natural-language task instruction passed to the architect node.",
    )
    parser.add_argument(
        "--scenario-id",
        default=None,
        help="Stable identifier used for the output sub-directory and USD filename. "
             "Defaults to a timestamped slug derived from the task.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Root directory that will hold <scenario-id>/{scene_generate.log,isaac.log,run_*}.",
    )
    parser.add_argument(
        "--robot-desc",
        default=None,
        help="Override scene_generate.robot_desc from run_config.json.",
    )
    parser.add_argument(
        "--isaac-python",
        default=None,
        help="Override isaac.isaac_python_executable from run_config.json.",
    )
    parser.add_argument(
        "--scene-python",
        default=None,
        help="Override scene_generate.python_executable from run_config.json.",
    )
    parser.add_argument(
        "--pipe-id",
        default="",
        help="Suffix appended to the named-pipe addresses; required if running "
             "multiple instances on the same host.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for Isaac Sim to print its readiness markers.",
    )
    parser.add_argument(
        "--scene-timeout",
        type=float,
        default=None,
        help="Hard timeout (seconds) for the orchestrator process; default: no limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve commands and exit without launching either subprocess.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"[Pipeline] Config not found: {config_path}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_id = _safe_id(args.scenario_id or f"scene_{timestamp}")
    output_root = Path(args.output_dir).expanduser().resolve()

    try:
        result = run_pipeline(
            config_path=config_path,
            task=args.task,
            scenario_id=scenario_id,
            output_root=output_root,
            isaac_python=args.isaac_python,
            scene_python=args.scene_python,
            robot_desc=args.robot_desc,
            pipe_id=args.pipe_id,
            startup_timeout_sec=args.startup_timeout,
            scene_timeout_sec=args.scene_timeout,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n[Pipeline] Interrupted by user.")
        return 130
    except Exception as exc:
        print(f"[Pipeline] Failed: {exc}", file=sys.stderr)
        return 1

    summary_path = Path(result["scenario_dir"]) / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Pipeline] Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

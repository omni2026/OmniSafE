from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from core.base import BaseSimInterface, EvalScenario, ExecutionState, OracleResult, PlanningResult
except ModuleNotFoundError:
    import sys

    eval_root = Path(__file__).resolve().parents[3]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from core.base import BaseSimInterface, EvalScenario, ExecutionState, OracleResult, PlanningResult


logger = logging.getLogger(__name__)


class SimCommandBridge:
    """Thread-safe sync bridge used by LangChain tools to call async sim APIs."""

    _COMMAND_TIMEOUT_FLOORS_SEC = {
        'move_base_to_pose': 75.0,
        'move_end_effector_to_pose': 40.0,
        'lateral_shift': 40.0,
        'rotate_end_effector': 30.0,
        'move_torso_to_height': 20.0,
        'set_torso_height': 20.0,
        'open': 75.0,
        'suggest_manipulation_base_pose': 90.0,
        'close': 10.0,
        'reset': 120.0,
    }

    DEFAULT_SCREENSHOT_RESOLUTION: tuple[int, int] = (1024, 1024)

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        sim: BaseSimInterface,
        scenario_id: str,
        command_timeout_sec: float = 30.0,
        scenario: Optional[EvalScenario] = None,
        planning_results: Optional[List[PlanningResult]] = None,
        online_oracles: Optional[List[Any]] = None,
        scenario_log_dir: Optional[str] = None,
        screenshots_enabled: bool = False,
        screenshot_resolution: Optional[Sequence[int]] = None,
    ):
        self.loop = loop
        self.sim = sim
        self.scenario_id = scenario_id
        self.command_timeout_sec = float(command_timeout_sec)
        self.scenario = scenario or EvalScenario(
            scenario_id=scenario_id,
            usd_path='',
            instructions=[],
        )
        self.planning_results = list(planning_results or [])
        self.online_oracles = list(online_oracles or [])
        self.records: List[Dict[str, Any]] = []
        self._online_states: List[ExecutionState] = []
        self._poisoned_reason = ''

        self.scenario_log_dir = Path(scenario_log_dir) if scenario_log_dir else None
        self.screenshots_enabled = bool(screenshots_enabled)
        self.screenshot_resolution = self._normalize_screenshot_resolution(screenshot_resolution)

    def invoke(
        self,
        *,
        tool_name: str,
        command: str,
        args: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if self._poisoned_reason:
            raise RuntimeError(
                f'simulator bridge is unusable after a timed-out request: '
                f'{self._poisoned_reason}'
            )
        payload = {
            'scenario_id': self.scenario_id,
            'command': command,
            'args': dict(args or {}),
        }
        timeout_sec = self._effective_timeout_sec(command)
        before_state = self._collect_state_payload()
        try:
            response = self._run_coro(
                self.sim.send_command(payload),
                timeout_sec=timeout_sec,
            )
        except TimeoutError as exc:
            self._poisoned_reason = (
                f"command={command}, tool={tool_name}, timeout_sec={timeout_sec:.1f}"
            )
            if str(exc):
                raise
            raise TimeoutError(
                f"sim command '{command}' (tool='{tool_name}') timed out after "
                f"{timeout_sec:.1f}s"
            ) from exc
        after_state = self._collect_state_payload()
        runtime_unsafe_events = self._pull_runtime_unsafe_events(step=len(self.records) + 1)
        record = {
            'tool_name': tool_name,
            'command': payload,
            'command_name': command,
            'args': dict(args or {}),
            'response': dict(response or {}),
            'timeout_sec': timeout_sec,
            'before_state': before_state,
            'after_state': after_state,
            'runtime_unsafe_events': runtime_unsafe_events,
            'timestamp': time.time(),
        }
        self.records.append(record)
        self._run_online_oracles(record)
        return dict(response or {})

    def reset_runtime(self) -> Dict[str, Any]:
        return self.invoke(tool_name='reset', command='reset', args={})

    def export_trace(self) -> List[Dict[str, Any]]:
        trace: List[Dict[str, Any]] = []
        for index, record in enumerate(self.records, start=1):
            entry: Dict[str, Any] = {
                'index': index,
                'tool_name': record['tool_name'],
                'command': record['command'],
                'response': record['response'],
            }
            if record.get('top_down_screenshot_path') is not None:
                entry['top_down_screenshot_path'] = record.get('top_down_screenshot_path')
            if record.get('top_down_screenshot_error'):
                entry['top_down_screenshot_error'] = record.get('top_down_screenshot_error')
            trace.append(entry)
        return trace

    def build_execution_states(self) -> List[ExecutionState]:
        states: List[ExecutionState] = []
        for index, record in enumerate(self.records, start=1):
            after_state = dict(record.get('after_state') or {})
            response = dict(record.get('response') or {})
            payload = dict(response.get('payload') or {})
            runtime_payload = after_state or payload
            state = ExecutionState(
                scenario_id=self.scenario_id,
                step=index,
                runtime_payload=runtime_payload,
                collision_flags=runtime_payload.get('collision_flags') or payload.get('collision_flags'),
                execution_metadata={
                    'tool_name': str(record.get('tool_name', '') or ''),
                    'trace_record': self._trace_record(record, index),
                    'online_oracle_results': list(record.get('online_oracle_results') or []),
                    'online_goal_events': list(record.get('online_goal_events') or []),
                    'online_unsafe_events': list(
                        record.get('online_unsafe_events')
                        or record.get('runtime_unsafe_events')
                        or []
                    ),
                    'runtime_unsafe_events': list(record.get('runtime_unsafe_events') or []),
                },
            )
            states.append(state)
        return states

    def _trace_record(self, record: Dict[str, Any], step_id: int) -> Dict[str, Any]:
        command_payload = dict(record.get('command') or {})
        trace = {
            'step_id': step_id,
            'tool_name': str(record.get('tool_name', '') or ''),
            'command': str(record.get('command_name', '') or command_payload.get('command', '') or ''),
            'command_payload': command_payload,
            'args': dict(record.get('args') or command_payload.get('args') or {}),
            'response': dict(record.get('response') or {}),
            'before_state': dict(record.get('before_state') or {}),
            'after_state': dict(record.get('after_state') or {}),
            'runtime_unsafe_events': list(record.get('runtime_unsafe_events') or []),
            'timestamp': float(record.get('timestamp', 0.0) or 0.0),
            'policy_metadata': {
                'timeout_sec': float(record.get('timeout_sec', 0.0) or 0.0),
                'online_oracle_count': len(self.online_oracles),
            },
        }
        if record.get('top_down_screenshot_path') is not None:
            trace['top_down_screenshot_path'] = record.get('top_down_screenshot_path')
        if record.get('top_down_screenshot_error'):
            trace['top_down_screenshot_error'] = record.get('top_down_screenshot_error')
        return trace

    # ------------------------------------------------------------------
    # Top-down screenshot capture (driven by the LangChain tool wrapper)
    # ------------------------------------------------------------------
    @classmethod
    def _normalize_screenshot_resolution(
        cls,
        resolution: Optional[Sequence[int]],
    ) -> tuple[int, int]:
        try:
            seq = list(resolution or [])
            if len(seq) >= 2:
                return (max(16, int(seq[0])), max(16, int(seq[1])))
        except Exception:
            pass
        return cls.DEFAULT_SCREENSHOT_RESOLUTION

    @staticmethod
    def _safe_tool_filename(tool_name: str) -> str:
        normalized = re.sub(r'[^A-Za-z0-9_-]+', '_', str(tool_name or '').strip()).strip('_')
        return normalized or 'tool'

    def capture_top_down_screenshot_for_tool(
        self,
        *,
        tool_name: str,
        start_record_count: int,
    ) -> Optional[str]:
        """Capture a top-down PNG for the LLM-visible tool that just finished.

        Attaches the relative screenshot path to the LAST record produced during
        the tool call, so composite tools (e.g. ``open`` -> suggest+move+open)
        produce a single screenshot per LLM-visible tool. Failures are caught
        and surfaced as ``top_down_screenshot_error`` on the same record; the
        original tool result is never affected.
        """
        if not self.screenshots_enabled or self.scenario_log_dir is None:
            return None
        if len(self.records) <= int(start_record_count):
            # Tool produced no simulator commands (perception-only or short-circuited).
            return None

        record = self.records[-1]
        step_id = len(self.records)
        rel_path = f'images/step_{step_id:04d}_{self._safe_tool_filename(tool_name)}.png'
        abs_path = self.scenario_log_dir / rel_path

        try:
            response = self._run_coro(
                self.sim.send_command({
                    'scenario_id': self.scenario_id,
                    'command': 'capture_top_down_screenshot',
                    'args': {
                        'path': str(abs_path),
                        'resolution': list(self.screenshot_resolution),
                    },
                }),
                timeout_sec=max(15.0, self.command_timeout_sec),
            )
        except Exception as exc:
            error = f'{exc.__class__.__name__}: {exc}'
            logger.warning(
                'top-down screenshot capture failed for tool=%s step=%s: %s',
                tool_name,
                step_id,
                error,
            )
            record['top_down_screenshot_path'] = None
            record['top_down_screenshot_error'] = error
            return None

        response_dict = dict(response or {})
        if not response_dict.get('ok'):
            error = str(response_dict.get('error', '') or response_dict.get('message', '') or 'capture_failed')
            logger.warning(
                'top-down screenshot capture failed for tool=%s step=%s: %s',
                tool_name,
                step_id,
                error,
            )
            record['top_down_screenshot_path'] = None
            record['top_down_screenshot_error'] = error
            return None

        normalized_rel = rel_path.replace('\\', '/')
        record['top_down_screenshot_path'] = normalized_rel
        return normalized_rel

    def _effective_timeout_sec(self, command: str) -> float:
        timeout_floor = float(self._COMMAND_TIMEOUT_FLOORS_SEC.get(str(command or ''), 0.0) or 0.0)
        return max(float(self.command_timeout_sec), timeout_floor)

    def _run_coro(self, coro: Any, *, timeout_sec: float | None = None) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        effective_timeout = (
            timeout_sec if timeout_sec is not None else self.command_timeout_sec
        )
        try:
            return future.result(timeout=effective_timeout)
        except TimeoutError:
            future.cancel()
            if not self._poisoned_reason:
                self._poisoned_reason = f'bridge coroutine timeout after {effective_timeout:.1f}s'
            raise

    def _collect_state_payload(self) -> Dict[str, Any]:
        try:
            state = self._run_coro(self.sim.get_state(), timeout_sec=max(5.0, self.command_timeout_sec))
        except TimeoutError:
            raise
        except Exception as exc:
            return {'state_error': f'{exc.__class__.__name__}: {exc}'}
        return self._state_to_payload(state)

    def _pull_runtime_unsafe_events(self, *, step: int) -> List[Dict[str, Any]]:
        pull_events = getattr(self.sim, 'get_runtime_unsafe_events', None)
        if pull_events is None:
            return []
        try:
            events = self._run_coro(pull_events(clear=True), timeout_sec=max(5.0, self.command_timeout_sec))
        except Exception:
            return []
        normalized_events = []
        for event in events or []:
            if not isinstance(event, dict):
                continue
            normalized_event = dict(event)
            normalized_event['step'] = int(step)
            normalized_event.setdefault('evidence', {})
            normalized_event['evidence'] = dict(normalized_event.get('evidence') or {})
            normalized_event['evidence']['trace_step'] = int(step)
            normalized_events.append(normalized_event)
        return normalized_events

    def _run_online_oracles(self, record: Dict[str, Any]) -> None:
        if not self.online_oracles:
            return
        runtime_unsafe_events = list(record.get('runtime_unsafe_events') or [])
        state = ExecutionState(
            scenario_id=self.scenario_id,
            step=len(self.records),
            runtime_payload=dict(record.get('after_state') or {}),
            collision_flags=(record.get('after_state') or {}).get('collision_flags'),
            execution_metadata={
                'trace_record': self._trace_record(record, len(self.records)),
                'runtime_unsafe_events': runtime_unsafe_events,
                'online_unsafe_events': runtime_unsafe_events,
            },
        )
        oracle_results: List[Dict[str, Any]] = []
        goal_events: List[Dict[str, Any]] = []
        unsafe_events: List[Dict[str, Any]] = list(runtime_unsafe_events)
        states_for_oracle = self._online_states + [state]
        for oracle in self.online_oracles:
            try:
                result = self._run_coro(
                    oracle.evaluate(self.scenario, self.planning_results, states_for_oracle),
                    timeout_sec=max(5.0, self.command_timeout_sec),
                )
            except Exception as exc:
                oracle_results.append({
                    'oracle_name': getattr(oracle, 'name', oracle.__class__.__name__),
                    'error': f'{exc.__class__.__name__}: {exc}',
                })
                continue
            payload = self._oracle_result_to_dict(result)
            oracle_results.append(payload)
            details = dict(payload.get('details') or {})
            goal_events.extend(list(details.get('online_goal_events') or []))
            unsafe_events.extend(list(details.get('online_unsafe_events') or []))
        record['online_oracle_results'] = oracle_results
        record['online_goal_events'] = goal_events
        record['online_unsafe_events'] = unsafe_events
        state.execution_metadata['online_goal_events'] = goal_events
        state.execution_metadata['online_unsafe_events'] = unsafe_events
        state.execution_metadata['online_oracle_results'] = oracle_results
        self._online_states.append(state)

    @staticmethod
    def _state_to_payload(state: Any) -> Dict[str, Any]:
        if isinstance(state, ExecutionState):
            return dict(state.runtime_payload or {})
        if is_dataclass(state):
            data = asdict(state)
            return dict(data.get('runtime_payload') or data)
        if isinstance(state, dict):
            return dict(state.get('runtime_payload') or state)
        return {}

    @staticmethod
    def _oracle_result_to_dict(result: Any) -> Dict[str, Any]:
        if isinstance(result, OracleResult):
            data = asdict(result)
            data['verdict'] = result.verdict.name
            return data
        if is_dataclass(result):
            return asdict(result)
        return dict(result or {}) if isinstance(result, dict) else {'result': str(result)}

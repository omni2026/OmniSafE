"""End-to-end evaluation orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

from .base import (
    BaseAgenticPolicy,
    BaseAggregator,
    BaseOracle,
    BasePlanningAgent,
    BaseSimInterface,
    EvalScenario,
    ExecutionState,
    OracleResult,
    OracleVerdict,
    PlanningResult,
    ProcessStatus,
)
from .evaluation import (
    BaseMetric,
    EvaluationContext,
    EvaluationError,
    EvaluationStatus,
    MetricResult,
    MetricStatus,
    ScenarioEvaluationResult,
    json_safe,
)
from .planning_evidence import normalize_planning_result

logger = logging.getLogger(__name__)
T = TypeVar('T')


class WatchdogTimeout(TimeoutError):
    pass


class Orchestrator:
    """Run scenarios sequentially against one long-lived Agent and simulator."""

    _AGENT_HIDDEN_METADATA_KEYS = {
        'expected_safe_behavior',
        'hazard_description',
        'hazard_id',
        'hazard_type',
        'known_hazards',
        'safe_conditions',
        'spec_generation_hints',
        'unsafe_conditions',
    }

    def __init__(
        self,
        sim: BaseSimInterface,
        planning_agent: BasePlanningAgent,
        policy: BaseAgenticPolicy,
        oracles: List[BaseOracle],
        aggregator: Optional[BaseAggregator] = None,
        metrics: Optional[List[BaseMetric]] = None,
        reporter: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        oracle_spec_generator: Optional[Any] = None,
    ):
        self.sim = sim
        self.planning_agent = planning_agent
        self.policy = policy
        self.oracles = list(oracles)
        self.aggregator = aggregator
        self.metrics = list(metrics or [])
        self.reporter = reporter
        self.config = dict(config or {})
        self.oracle_spec_generator = oracle_spec_generator

        self._step_timeout = self._optional_positive_timeout(
            self.config.get('step_timeout_sec', 30.0)
        )
        self._scenario_timeout = float(self.config.get('scenario_timeout_sec', 300.0))
        self._max_retries = max(0, int(self.config.get('max_retries', 0)))
        self._fail_fast = bool(self.config.get('fail_fast', False))
        self._require_oracle = bool(self.config.get('require_oracle', True))
        self._generate_missing_oracle = bool(
            self.config.get('generate_missing_oracle', False)
        )
        self._results: List[ScenarioEvaluationResult] = []
        self._active_output_path: Optional[str] = None

    async def start_all(self) -> None:
        """Start dependencies in Sim -> Policy -> Agent order."""
        started: List[Any] = []
        try:
            for proc in (self.sim, self.policy, self.planning_agent):
                await proc.start()
                proc.status = ProcessStatus.RUNNING
                started.append(proc)
                logger.info('%s started', proc.name)
        except Exception:
            for proc in reversed(started):
                try:
                    await proc.stop()
                except Exception:
                    logger.exception('Failed to stop %s after startup error', proc.name)
            raise

    async def stop_all(self) -> None:
        for proc in reversed((self.sim, self.policy, self.planning_agent)):
            try:
                await proc.stop()
                proc.status = ProcessStatus.TERMINATED
            except Exception as exc:
                logger.warning('%s stop error: %s', proc.name, exc)

    @asynccontextmanager
    async def managed_session(self):
        await self.start_all()
        try:
            yield self
        finally:
            await self.stop_all()

    async def run_dataset(
        self,
        scenarios: List[EvalScenario],
        output_path: Optional[str] = None,
        *,
        resume: bool = False,
        skip_existing: bool = False,
    ) -> List[ScenarioEvaluationResult]:
        """Evaluate all cases, persisting each case immediately when possible."""
        self._results = []
        self._active_output_path = output_path
        if self.reporter and output_path:
            initialize_run_logs = getattr(self.reporter, 'initialize_run_logs', None)
            if initialize_run_logs is not None:
                log_dir = initialize_run_logs(
                    output_path,
                    scenario_ids=[scenario.scenario_id for scenario in scenarios],
                )
                logger.info('Streaming diagnostic logs to %s', log_dir)
            self._record_run_log(
                'run_start',
                {
                    'scenario_count': len(scenarios),
                    'resume': resume,
                    'skip_existing': skip_existing,
                    'output_path': output_path,
                },
            )
        resumed: Dict[str, ScenarioEvaluationResult] = {}
        if (resume or skip_existing) and self.reporter and output_path:
            load_method_name = 'load_existing' if skip_existing else 'load_resumable'
            load_persisted = getattr(self.reporter, load_method_name, None)
            if load_persisted is not None:
                resumed = dict(load_persisted(output_path) or {})
                scenarios_by_id = {
                    scenario.scenario_id: scenario
                    for scenario in scenarios
                }
                resumed = {
                    scenario_id: result
                    for scenario_id, result in resumed.items()
                    if (
                        scenario_id in scenarios_by_id
                        and result.metadata.get('scenario_fingerprint')
                        == self._scenario_fingerprint(scenarios_by_id[scenario_id])
                    )
                }

        pending = [
            scenario for scenario in scenarios
            if scenario.scenario_id not in resumed
        ]
        if pending:
            async with self.managed_session():
                for scenario in scenarios:
                    if scenario.scenario_id in resumed:
                        result = resumed[scenario.scenario_id]
                        logger.info(
                            'Resume: reusing existing scenario %s (status=%s)',
                            scenario.scenario_id,
                            result.status.value,
                        )
                    else:
                        logger.info('=== Scenario: %s ===', scenario.scenario_id)
                        result = await self._run_scenario_with_retry(scenario)
                        if self.reporter and output_path:
                            write_scenario = getattr(self.reporter, 'write_scenario', None)
                            if write_scenario is not None:
                                await write_scenario(result, output_path)
                    self._results.append(result)
                    if self._fail_fast and result.status in {
                        EvaluationStatus.FAILED,
                        EvaluationStatus.INVALID,
                    }:
                        break
        else:
            self._results = [
                resumed[scenario.scenario_id]
                for scenario in scenarios
                if scenario.scenario_id in resumed
            ]

        if self.reporter and output_path:
            await self.reporter.write(self._results, output_path)
            self._record_run_log(
                'run_finish',
                {
                    'result_count': len(self._results),
                    'status_counts': {
                        status: sum(
                            1
                            for result in self._results
                            if result.status.value == status
                        )
                        for status in sorted({result.status.value for result in self._results})
                    },
                    'output_path': output_path,
                },
            )
        return list(self._results)

    async def _run_scenario_with_retry(
        self,
        scenario: EvalScenario,
    ) -> ScenarioEvaluationResult:
        attempt_errors: List[EvaluationError] = []
        final_result: Optional[ScenarioEvaluationResult] = None

        for attempt in range(1, self._max_retries + 2):
            result = self._new_result(scenario, attempt)
            try:
                await asyncio.wait_for(
                    self._run_scenario_once(scenario, result),
                    timeout=self._scenario_timeout,
                )
            except asyncio.TimeoutError:
                message = f'scenario exceeded {self._scenario_timeout:.1f}s'
                result.status = EvaluationStatus.FAILED
                result.metadata['failed_stage'] = 'scenario_timeout'
                result.errors.append(EvaluationError(
                    stage='scenario_timeout',
                    error_type='WatchdogTimeout',
                    message=message,
                    attempt=attempt,
                ))
                self._record_diagnostic(
                    result,
                    'scenario_timeout',
                    'Scenario exceeded the configured timeout.',
                    {
                        'timeout_sec': self._scenario_timeout,
                        'attempt': attempt,
                    },
                    level='error',
                    event='scenario.timeout',
                )
                await self._mark_metrics_not_recorded_due_to_failure(
                    result,
                    scenario,
                    failed_stage='scenario_timeout',
                    error_type='WatchdogTimeout',
                    message=message,
                )
                self._update_result_summary_metadata(result, scenario)
                self._record_failed_scenario_finish(
                    result,
                    failed_stage='scenario_timeout',
                )
            except asyncio.CancelledError:
                raise

            self._finish_result(result)
            attempt_errors.extend(result.errors)
            final_result = result
            if result.status not in {EvaluationStatus.FAILED}:
                result.errors = attempt_errors
                result.attempts = (
                    max(error.attempt for error in attempt_errors)
                    if attempt_errors
                    else result.attempts
                )
                self._update_result_summary_metadata(result, scenario)
                return result
            if attempt > self._max_retries or not self._is_retryable(result):
                break

            logger.warning(
                'Retrying scenario %s after stage %s (%s/%s)',
                scenario.scenario_id,
                result.metadata.get('failed_stage', 'unknown'),
                attempt,
                self._max_retries + 1,
            )
            try:
                await self._restart_runtime()
            except Exception as exc:
                attempt_errors.append(EvaluationError(
                    stage='runtime_restart',
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                    attempt=attempt,
                ))
                break

        assert final_result is not None
        final_result.errors = attempt_errors
        final_result.attempts = max(error.attempt for error in attempt_errors) if attempt_errors else 1
        self._update_result_summary_metadata(final_result, scenario)
        return final_result

    async def _run_scenario_once(
        self,
        original_scenario: EvalScenario,
        result: ScenarioEvaluationResult,
    ) -> None:
        stage = 'oracle_spec'
        executed_plan_indices: List[int] = []
        intercepted: Optional[bool] = None
        scenario = replace(
            original_scenario,
            metadata=dict(original_scenario.metadata or {}),
            oracle_annotations=(
                dict(original_scenario.oracle_annotations)
                if original_scenario.oracle_annotations
                else None
            ),
        )
        self._record_diagnostic(
            result,
            'scenario_start',
            'Scenario input received.',
            self._scenario_diagnostic_payload(scenario),
            event='scenario.input.loaded',
        )
        logger.info(
            'Task input scenario=%s instruction_count=%s scene=%s',
            scenario.scenario_id,
            len(scenario.instructions),
            scenario.usd_path,
        )

        try:
            scenario = await self._timed(
                result,
                'oracle_spec',
                lambda: self._prepare_oracle_spec(scenario),
            )
            oracle_spec = self._oracle_task_spec_payload(scenario)
            self._record_diagnostic(
                result,
                'oracle_spec',
                'Oracle task spec prepared.',
                {
                    'scenario': self._scenario_diagnostic_payload(scenario),
                    'oracle_summary': self._oracle_spec_summary(oracle_spec),
                },
                event='oracle.spec.prepared',
            )
            logger.info(
                'Oracle spec scenario=%s summary=%s',
                scenario.scenario_id,
                self._oracle_spec_summary(oracle_spec),
            )
            logger.debug(
                'Oracle spec detail scenario=%s payload=%s',
                scenario.scenario_id,
                self._json_dumps_for_log(oracle_spec),
            )

            # Reset simulation state to clear residual data from the previous
            # scenario (step counter, runtime monitors, entity registry, etc.).
            # This prevents state leakage across scenarios within the same
            # session, which was causing execution hangs (e.g. H02_T03).
            stage = 'pre_scene_reset'
            try:
                await self._timed(
                    result,
                    'pre_scene_reset',
                    self.sim.reset,
                )
            except Exception as exc:
                logger.warning(
                    'Pre-scene reset failed for %s: %s (continuing)',
                    scenario.scenario_id, exc,
                )

            stage = 'scene_load'
            loaded = await self._timed(
                result,
                'scene_load',
                lambda: self.sim.load_scene(scenario.usd_path),
            )
            if not loaded:
                raise RuntimeError(f'failed to load USD: {scenario.usd_path}')
            self._record_diagnostic(
                result,
                'scene_load',
                'Scene loaded.',
                {'usd_path': scenario.usd_path, 'loaded': loaded},
                event='scene.loaded',
            )
            logger.info('Scene loaded scenario=%s usd=%s', scenario.scenario_id, scenario.usd_path)

            stage = 'robot_load'
            robot_loaded = await self._timed(
                result,
                'robot_load',
                lambda: self.sim.load_robot(
                    str(self.config.get('robot_name', 'fetch') or 'fetch'),
                    str(self.config.get('robot_room_name', '') or '').strip() or None,
                ),
            )
            if not robot_loaded:
                raise RuntimeError('simulator rejected robot loading')
            self._record_diagnostic(
                result,
                'robot_load',
                'Robot loaded.',
                {
                    'robot_name': str(self.config.get('robot_name', 'fetch') or 'fetch'),
                    'robot_room_name': str(self.config.get('robot_room_name', '') or ''),
                    'loaded': robot_loaded,
                },
                event='robot.loaded',
            )
            logger.info(
                'Robot loaded scenario=%s robot=%s',
                scenario.scenario_id,
                str(self.config.get('robot_name', 'fetch') or 'fetch'),
            )

            result.initial_state = await self._timed(
                result,
                'initial_state',
                self.sim.get_state,
            )
            await self.sim.save_checkpoint('initial')
            self._record_diagnostic(
                result,
                'initial_state',
                'Initial simulation state captured.',
                {'initial_state': result.initial_state},
                event='state.initial.captured',
            )
            logger.debug(
                'Initial state scenario=%s payload=%s',
                scenario.scenario_id,
                self._json_dumps_for_log(result.initial_state),
            )

            stage = 'scenario_setup'
            await self._call_optional(self.planning_agent, 'begin_scenario', scenario)
            await self._call_optional(self.policy, 'begin_scenario', scenario)
            stage = 'oracle_runtime_setup'
            runtime_oracle_setup = await self._timed(
                result,
                'oracle_runtime_setup',
                lambda: self._prepare_runtime_online_oracles(scenario),
            )
            self._validate_runtime_oracle_registrations(runtime_oracle_setup)
            self._record_diagnostic(
                result,
                'oracle_runtime_setup',
                'Runtime online oracle setup completed.',
                {'registrations': runtime_oracle_setup},
                event='oracle.runtime_setup.completed',
            )
            logger.info(
                'Runtime online oracle setup scenario=%s registrations=%s',
                scenario.scenario_id,
                len(runtime_oracle_setup or []),
            )

            stage = 'planning'
            planning_context = self._build_planning_context(scenario, result.initial_state)
            interactive_planning = bool(
                getattr(self.planning_agent, 'supports_interactive_planning', False)
            )
            initial_instructions = (
                list(scenario.instructions[:1])
                if interactive_planning
                else list(scenario.instructions)
            )
            self._record_diagnostic(
                result,
                'planning_context',
                'Planning context prepared.',
                {'planning_context': planning_context},
                event='planning.context.prepared',
            )
            result.planning_results = await self._timed(
                result,
                'planning',
                lambda: self._get_planning_results(
                    scenario,
                    result.initial_state,
                    context=planning_context,
                    instructions=initial_instructions,
                ),
            )
            self._validate_planning_results(
                scenario,
                result.planning_results,
                expected_count=len(initial_instructions),
            )
            planning_payload = self._planning_diagnostic_payload(result.planning_results)
            self._record_diagnostic(
                result,
                'planning',
                'Planning agent returned results.',
                planning_payload,
                event='planning.output.received',
            )
            for plan_index, plan in enumerate(result.planning_results):
                logger.info(
                    'Planning result scenario=%s plan_index=%s refused=%s action_count=%s raw_output_chars=%s',
                    scenario.scenario_id,
                    plan_index,
                    plan.refused,
                    len(plan.actions),
                    len(str(plan.raw_output or '')),
                )
                logger.debug(
                    'Planning result detail scenario=%s plan_index=%s payload=%s',
                    scenario.scenario_id,
                    plan_index,
                    self._json_dumps_for_log(plan),
                )

            stage = 'blocking_oracles'
            blocking_results = await self._timed(
                result,
                'blocking_oracles',
                lambda: self._run_blocking_oracles(
                    scenario,
                    result.planning_results,
                    [],
                ),
            )
            intercepted = any(item.verdict == OracleVerdict.FAIL for item in blocking_results)
            self._record_diagnostic(
                result,
                'blocking_oracles',
                'Blocking oracle checks completed.',
                {
                    'intercepted': intercepted,
                    'oracle_results': blocking_results,
                },
                event='oracle.blocking.completed',
            )
            logger.info(
                'Blocking oracles scenario=%s count=%s intercepted=%s',
                scenario.scenario_id,
                len(blocking_results),
                intercepted,
            )

            stage = 'execution'
            if not intercepted:
                if interactive_planning:
                    additional_blocking, intercepted = await self._run_interactive_execution_loop(
                        scenario=scenario,
                        result=result,
                        base_context=planning_context,
                        executed_plan_indices=executed_plan_indices,
                    )
                    blocking_results.extend(additional_blocking)
                else:
                    for plan_index, plan in enumerate(result.planning_results):
                        if plan.refused:
                            self._record_diagnostic(
                                result,
                                'policy_execution',
                                'Policy execution skipped because the planner refused this instruction.',
                                {
                                    'plan_index': plan_index,
                                    'plan': plan,
                                },
                                event='policy.execution.skipped',
                            )
                            logger.info(
                                'Policy execution skipped scenario=%s plan_index=%s reason=%s',
                                scenario.scenario_id,
                                plan_index,
                                plan.refusal_reason or 'planner_refused',
                            )
                            continue
                        states = await self._timed(
                            result,
                            'execution',
                            lambda plan=plan: self._execute_plan_with_watchdog(
                                plan,
                                scenario,
                                result.planning_results,
                            ),
                            accumulate=True,
                        )
                        result.execution_states.extend(states)
                        executed_plan_indices.append(plan_index)
                        execution_payload = self._execution_diagnostic_payload(
                            plan_index=plan_index,
                            plan=plan,
                            states=states,
                        )
                        self._record_policy_timeline_events(result, execution_payload)
                        self._record_diagnostic(
                            result,
                            'policy_execution',
                            'Policy agent execution completed.',
                            self._policy_execution_summary_payload(execution_payload),
                            event='policy.execution.completed',
                        )
                        logger.info(
                            'Policy execution scenario=%s plan_index=%s states=%s tool_calls=%s goal_events=%s unsafe_events=%s',
                            scenario.scenario_id,
                            plan_index,
                            len(states),
                            len(execution_payload.get('tool_trace') or []),
                            len(execution_payload.get('online_goal_events') or []),
                            len(execution_payload.get('online_unsafe_events') or []),
                        )
                        logger.debug(
                            'Policy execution detail scenario=%s plan_index=%s payload=%s',
                            scenario.scenario_id,
                            plan_index,
                            self._json_dumps_for_log(execution_payload),
                        )
                        self._raise_policy_error_if_present(states)

            result.final_state = await self._timed(
                result,
                'final_state',
                self.sim.get_state,
            )
            self._record_diagnostic(
                result,
                'final_state',
                'Final simulation state captured.',
                {'final_state': result.final_state},
                event='state.final.captured',
            )
            logger.debug(
                'Final state scenario=%s payload=%s',
                scenario.scenario_id,
                self._json_dumps_for_log(result.final_state),
            )

            stage = 'oracles'
            post_results = await self._timed(
                result,
                'oracles',
                lambda: self._run_all_oracles(
                    scenario,
                    result.planning_results,
                    result.execution_states,
                ),
            )
            result.oracle_results = blocking_results + post_results
            self._record_diagnostic(
                result,
                'oracles',
                'Post-execution oracle diagnostics completed.',
                self._oracle_diagnostic_payload(
                    oracle_results=result.oracle_results,
                    execution_states=result.execution_states,
                ),
                event='oracle.post_execution.completed',
            )
            logger.info(
                'Oracle diagnostics scenario=%s results=%s',
                scenario.scenario_id,
                self._oracle_result_summary(result.oracle_results),
            )
            logger.debug(
                'Oracle diagnostics detail scenario=%s payload=%s',
                scenario.scenario_id,
                self._json_dumps_for_log(result.oracle_results),
            )

            stage = 'aggregation'
            if self.aggregator is not None:
                started = time.perf_counter()
                try:
                    result.aggregate = self.aggregator.aggregate(
                        scenario,
                        result.oracle_results,
                    )
                finally:
                    result.phase_durations_sec['aggregation'] = round(
                        time.perf_counter() - started,
                        6,
                    )
            self._record_diagnostic(
                result,
                'aggregation',
                'Oracle aggregation completed.',
                {'aggregate': result.aggregate},
                event='aggregation.completed',
            )
            if result.aggregate is not None:
                logger.info(
                    'Aggregation scenario=%s verdict=%s final=%.3f task=%.3f safety=%.3f',
                    scenario.scenario_id,
                    result.aggregate.verdict.name,
                    result.aggregate.final_score,
                    result.aggregate.task_score,
                    result.aggregate.safety_score,
                )

            stage = 'metrics'
            context = EvaluationContext(
                scenario=scenario,
                planning_results=result.planning_results,
                execution_states=result.execution_states,
                oracle_results=result.oracle_results,
                initial_state=result.initial_state,
                final_state=result.final_state,
                aggregate=result.aggregate,
                runtime_metadata={
                    'attempt': result.attempts,
                    'phase_durations_sec': result.phase_durations_sec,
                    'executed_plan_indices': executed_plan_indices,
                    'intercepted': intercepted,
                },
            )
            result.metric_results = await self._timed(
                result,
                'metrics',
                lambda: self._run_metrics(context),
            )
            self._record_diagnostic(
                result,
                'metrics',
                'Metric evaluation completed.',
                {'metric_results': result.metric_results},
                event='metrics.completed',
            )
            logger.info(
                'Metrics scenario=%s results=%s',
                scenario.scenario_id,
                self._metric_result_summary(result.metric_results),
            )
            logger.debug(
                'Metrics detail scenario=%s payload=%s',
                scenario.scenario_id,
                self._json_dumps_for_log(result.metric_results),
            )

            result.status = self._completed_status(result)
            self._update_result_summary_metadata(
                result,
                scenario,
                executed_plan_indices=executed_plan_indices,
                intercepted=intercepted,
            )
            self._record_diagnostic(
                result,
                'scenario_finish',
                'Scenario evaluation finished.',
                {
                    'status': result.status.value,
                    'phase_durations_sec': dict(result.phase_durations_sec),
                    'error_count': len(result.errors),
                    'aggregate_summary': (
                        {
                            'verdict': result.aggregate.verdict.name,
                            'final_score': result.aggregate.final_score,
                            'task_score': result.aggregate.task_score,
                            'safety_score': result.aggregate.safety_score,
                        }
                        if result.aggregate is not None
                        else None
                    ),
                },
                event='scenario.finished',
            )
            result.metadata['diagnostic_log_count'] = len(result.diagnostic_log)
            logger.info(
                'Scenario %s -> status=%s%s',
                scenario.scenario_id,
                result.status.value,
                (
                    f' final={result.aggregate.final_score:.3f}'
                    if result.aggregate is not None
                    else ''
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result.status = (
                EvaluationStatus.INVALID
                if stage in {'oracle_spec', 'oracle_runtime_setup'}
                and isinstance(exc, (ValueError, TypeError))
                else EvaluationStatus.FAILED
            )
            result.metadata['failed_stage'] = stage
            result.errors.append(EvaluationError(
                stage=stage,
                error_type=exc.__class__.__name__,
                message=str(exc),
                attempt=result.attempts,
            ))
            self._record_diagnostic(
                result,
                stage,
                'Scenario failed during evaluation stage.',
                {
                    'error_type': exc.__class__.__name__,
                    'message': str(exc),
                    'attempt': result.attempts,
                },
                level='error',
                event='scenario.stage.failed',
            )
            await self._mark_metrics_not_recorded_due_to_failure(
                result,
                scenario,
                failed_stage=stage,
                error_type=exc.__class__.__name__,
                message=str(exc),
            )
            self._update_result_summary_metadata(
                result,
                scenario,
                executed_plan_indices=executed_plan_indices,
                intercepted=intercepted,
            )
            self._record_failed_scenario_finish(result, failed_stage=stage)
            logger.exception('Scenario %s failed during %s', scenario.scenario_id, stage)
        finally:
            for component in (self.policy, self.planning_agent):
                try:
                    await self._call_optional(component, 'end_scenario', scenario)
                except Exception as exc:
                    result.errors.append(EvaluationError(
                        stage='scenario_teardown',
                        error_type=exc.__class__.__name__,
                        message=str(exc),
                        attempt=result.attempts,
                    ))
                    self._record_diagnostic(
                        result,
                        'scenario_teardown',
                        'Scenario teardown hook failed.',
                        {
                            'component': getattr(component, 'name', component.__class__.__name__),
                            'error_type': exc.__class__.__name__,
                            'message': str(exc),
                            'attempt': result.attempts,
                        },
                        level='warning',
                        event='scenario.teardown.failed',
                    )
                    if result.status == EvaluationStatus.SUCCESS:
                        result.status = EvaluationStatus.PARTIAL
            self._update_result_summary_metadata(
                result,
                scenario,
                executed_plan_indices=executed_plan_indices,
                intercepted=intercepted,
            )

    async def _prepare_oracle_spec(self, scenario: EvalScenario) -> EvalScenario:
        annotations = dict(scenario.oracle_annotations or {})
        manual_spec = annotations.get('oracle_task_spec')
        if manual_spec is not None:
            if self.oracle_spec_generator is not None:
                parsed = await self.oracle_spec_generator.generate(scenario)
                scenario.metadata['oracle_task_spec'] = self._json_safe_dataclass(parsed)
            else:
                scenario.metadata['oracle_task_spec'] = dict(manual_spec)
            return scenario

        if self._generate_missing_oracle and self.oracle_spec_generator is not None:
            generated = await self.oracle_spec_generator.generate(scenario)
            scenario.metadata['oracle_task_spec'] = self._json_safe_dataclass(generated)
            return scenario

        if self._require_oracle:
            raise ValueError(
                'oracle_task_spec is missing; generate Oracle annotations before evaluation'
            )
        return scenario

    async def _get_planning_results(
        self,
        scenario: EvalScenario,
        initial_state: Optional[ExecutionState],
        *,
        context: Optional[Dict[str, Any]] = None,
        instructions: Optional[List[str]] = None,
    ) -> List[PlanningResult]:
        context = context or self._build_planning_context(scenario, initial_state)
        await self.planning_agent.update_context(context)
        planning_results = await self.planning_agent.plan(
            list(scenario.instructions if instructions is None else instructions),
            context,
        )
        return [
            normalize_planning_result(plan)
            for plan in planning_results
        ]

    async def _run_interactive_execution_loop(
        self,
        *,
        scenario: EvalScenario,
        result: ScenarioEvaluationResult,
        base_context: Dict[str, Any],
        executed_plan_indices: List[int],
    ) -> tuple[List[OracleResult], bool]:
        """Alternate planning and execution until each instruction terminates."""
        additional_blocking: List[OracleResult] = []
        agent_limit = getattr(self.planning_agent, 'max_interaction_turns', None)
        configured_limit = self.config.get('max_interaction_turns')
        if configured_limit is None:
            configured_limit = agent_limit or 50
        max_turns = max(
            1,
            int(configured_limit),
        )

        for instruction_index, instruction in enumerate(scenario.instructions):
            if instruction_index == 0:
                plan = result.planning_results[0]
            else:
                initial_context = self._interactive_followup_context(
                    base_context=base_context,
                    plan=None,
                    states=[],
                    all_execution_states=result.execution_states,
                    instruction_index=instruction_index,
                    turn_index=0,
                    reset_agent=True,
                )
                plans = await self._timed(
                    result,
                    'planning',
                    lambda instruction=instruction, initial_context=initial_context: self._get_planning_results(
                        scenario,
                        result.initial_state,
                        context=initial_context,
                        instructions=[instruction],
                    ),
                    accumulate=True,
                )
                self._validate_planning_results(scenario, plans, expected_count=1)
                plan = plans[0]
                result.planning_results.append(plan)
                self._record_interactive_planning_result(
                    result=result,
                    scenario=scenario,
                    plan=plan,
                    plan_index=len(result.planning_results) - 1,
                    instruction_index=instruction_index,
                    turn_index=0,
                )
                round_blocking = await self._timed(
                    result,
                    'blocking_oracles',
                    lambda plan=plan: self._run_blocking_oracles(
                        scenario,
                        [plan],
                        result.execution_states,
                    ),
                    accumulate=True,
                )
                additional_blocking.extend(round_blocking)
                if any(item.verdict == OracleVerdict.FAIL for item in round_blocking):
                    return additional_blocking, True

            execution_round = 0
            while True:
                plan_index = len(result.planning_results) - 1
                status = self._interactive_plan_status(plan)
                if plan.refused:
                    self._record_diagnostic(
                        result,
                        'policy_execution',
                        'Policy execution skipped because the interactive planner refused.',
                        {'plan_index': plan_index, 'plan': plan},
                        event='policy.execution.skipped',
                    )
                    break
                if self._interactive_plan_terminal(plan):
                    self._record_diagnostic(
                        result,
                        'planning',
                        'Interactive planning instruction terminated.',
                        {
                            'plan_index': plan_index,
                            'instruction_index': instruction_index,
                            'turn_index': execution_round,
                            'status': status,
                        },
                        event='planning.interactive.terminated',
                    )
                    if status == 'failed':
                        return additional_blocking, False
                    break
                if not plan.actions:
                    raise RuntimeError(
                        'interactive planning result is non-terminal but contains no actions'
                    )
                if execution_round >= max_turns:
                    raise RuntimeError(
                        'interactive planning exceeded max_interaction_turns: '
                        f'{max_turns} for instruction {instruction_index}'
                    )

                states = await self._timed(
                    result,
                    'execution',
                    lambda plan=plan: self._execute_plan_with_watchdog(
                        plan,
                        scenario,
                        result.planning_results,
                    ),
                    accumulate=True,
                )
                self._rebase_execution_steps(states, result.execution_states)
                result.execution_states.extend(states)
                executed_plan_indices.append(plan_index)
                execution_payload = self._execution_diagnostic_payload(
                    plan_index=plan_index,
                    plan=plan,
                    states=states,
                )
                self._record_policy_timeline_events(result, execution_payload)
                self._record_diagnostic(
                    result,
                    'policy_execution',
                    'Interactive policy execution round completed.',
                    {
                        **self._policy_execution_summary_payload(execution_payload),
                        'instruction_index': instruction_index,
                        'turn_index': execution_round,
                    },
                    event='policy.execution.completed',
                )
                self._raise_policy_error_if_present(states)

                followup_context = self._interactive_followup_context(
                    base_context=base_context,
                    plan=plan,
                    states=states,
                    all_execution_states=result.execution_states,
                    instruction_index=instruction_index,
                    turn_index=execution_round + 1,
                )
                next_plans = await self._timed(
                    result,
                    'planning',
                    lambda instruction=instruction, followup_context=followup_context: self._get_planning_results(
                        scenario,
                        result.initial_state,
                        context=followup_context,
                        instructions=[instruction],
                    ),
                    accumulate=True,
                )
                self._validate_planning_results(scenario, next_plans, expected_count=1)
                plan = next_plans[0]
                result.planning_results.append(plan)
                next_plan_index = len(result.planning_results) - 1
                execution_round += 1
                self._record_interactive_planning_result(
                    result=result,
                    scenario=scenario,
                    plan=plan,
                    plan_index=next_plan_index,
                    instruction_index=instruction_index,
                    turn_index=execution_round,
                )

                round_blocking = await self._timed(
                    result,
                    'blocking_oracles',
                    lambda plan=plan: self._run_blocking_oracles(
                        scenario,
                        [plan],
                        result.execution_states,
                    ),
                    accumulate=True,
                )
                additional_blocking.extend(round_blocking)
                if any(item.verdict == OracleVerdict.FAIL for item in round_blocking):
                    return additional_blocking, True

        return additional_blocking, False

    def _record_interactive_planning_result(
        self,
        *,
        result: ScenarioEvaluationResult,
        scenario: EvalScenario,
        plan: PlanningResult,
        plan_index: int,
        instruction_index: int,
        turn_index: int,
    ) -> None:
        payload = self._planning_diagnostic_payload([plan])
        payload.update({
            'plan_index': plan_index,
            'instruction_index': instruction_index,
            'turn_index': turn_index,
            'interactive': True,
        })
        self._record_diagnostic(
            result,
            'planning',
            'Interactive planning agent returned the next result.',
            payload,
            event='planning.output.received',
        )
        logger.info(
            'Interactive planning result scenario=%s plan_index=%s instruction_index=%s turn=%s status=%s actions=%s',
            scenario.scenario_id,
            plan_index,
            instruction_index,
            turn_index,
            self._interactive_plan_status(plan),
            len(plan.actions),
        )

    @classmethod
    def _interactive_plan_status(cls, plan: PlanningResult) -> str:
        metadata = dict(plan.metadata or {})
        interaction = dict(metadata.get('interactive_planning') or {})
        agent_state = dict(metadata.get('roboagent') or {})
        return str(
            interaction.get('status')
            or agent_state.get('status')
            or ''
        ).strip().lower()

    @classmethod
    def _interactive_plan_terminal(cls, plan: PlanningResult) -> bool:
        metadata = dict(plan.metadata or {})
        interaction = dict(metadata.get('interactive_planning') or {})
        agent_state = dict(metadata.get('roboagent') or {})
        if bool(interaction.get('finished')) or bool(agent_state.get('finished')):
            return True
        return cls._interactive_plan_status(plan) in {
            'completed', 'done', 'failed', 'finished', 'refused',
        }

    @classmethod
    def _interactive_followup_context(
        cls,
        *,
        base_context: Dict[str, Any],
        plan: Optional[PlanningResult],
        states: List[ExecutionState],
        all_execution_states: List[ExecutionState],
        instruction_index: int,
        turn_index: int,
        reset_agent: bool = False,
    ) -> Dict[str, Any]:
        context = dict(base_context or {})
        metadata = dict(context.get('metadata') or {})
        current_payload = dict(
            (all_execution_states[-1].runtime_payload if all_execution_states else {}) or {}
        )
        visible_objects = (
            current_payload.get('visible_objects')
            or current_payload.get('vis_objs')
            or context.get('visible_objects')
            or metadata.get('visible_objects')
            or []
        )
        metadata.update({
            'interactive_planning': True,
            'instruction_index': instruction_index,
            'interaction_turn': turn_index,
            'visible_objects': visible_objects,
            'vis_objs': visible_objects,
            'current_state': current_payload,
            'env_step_id': int(all_execution_states[-1].step) if all_execution_states else 0,
        })
        if reset_agent:
            metadata['reset_agent'] = True
        else:
            metadata.pop('reset_agent', None)

        if plan is not None:
            success, message = cls._interactive_execution_outcome(states)
            action_texts = [cls._interactive_action_text(action) for action in plan.actions]
            action_texts = [text for text in action_texts if text]
            metadata['execution_results'] = [
                {
                    'action': action_text,
                    'success': success,
                    'message': message,
                }
                for action_text in action_texts
            ]
            metadata['execution_result'] = {
                'plan': action_texts[0] if len(action_texts) == 1 else str(plan.raw_output or ''),
                'success': success,
                'message': message,
                'visible_objects': visible_objects,
            }
            dynamic = dict((plan.metadata or {}).get('dynamic_replanning') or {})
            if dynamic.get('loop_state') is not None:
                metadata['planner_loop_state'] = dynamic.get('loop_state')

        context.update({
            'metadata': metadata,
            'current_state': current_payload,
            'visible_objects': visible_objects,
        })
        return context

    @staticmethod
    def _interactive_action_text(action: Any) -> str:
        if not isinstance(action, dict):
            return str(action or '').strip()
        return str(
            action.get('action')
            or action.get('raw')
            or action.get('code')
            or action.get('type')
            or ''
        ).strip()

    @staticmethod
    def _interactive_execution_outcome(
        states: List[ExecutionState],
    ) -> tuple[bool, str]:
        if not states:
            return False, 'policy returned no execution state'
        final_metadata = dict(states[-1].execution_metadata or {})
        policy_error = dict(final_metadata.get('policy_error') or {})
        if policy_error:
            return False, str(policy_error.get('message') or policy_error.get('type') or 'policy error')

        recovery = dict(final_metadata.get('failure_recovery') or {})
        attempts = list(recovery.get('attempts') or [])
        if attempts:
            last_attempt = dict(attempts[-1] or {})
            if (
                last_attempt.get('policy_refusal')
                or last_attempt.get('recoverable_failure')
                or str(last_attempt.get('failure_reason') or '').strip()
            ):
                return False, str(
                    last_attempt.get('failure_reason')
                    or final_metadata.get('llm_output')
                    or 'policy execution failed'
                )
            return True, str(final_metadata.get('llm_output') or 'executed')

        for state in states:
            trace = dict((state.execution_metadata or {}).get('trace_record') or {})
            response = trace.get('response')
            if isinstance(response, dict) and response.get('ok') is False:
                return False, str(
                    response.get('message')
                    or response.get('error')
                    or 'simulator command failed'
                )
        return True, str(final_metadata.get('llm_output') or 'executed')

    @staticmethod
    def _rebase_execution_steps(
        states: List[ExecutionState],
        previous_states: List[ExecutionState],
    ) -> None:
        offset = max((int(state.step) for state in previous_states), default=0)
        for local_index, state in enumerate(states, start=1):
            state.step = offset + local_index
            metadata = dict(state.execution_metadata or {})
            trace = metadata.get('trace_record')
            if isinstance(trace, dict):
                trace['step_id'] = state.step
            state.execution_metadata = metadata

    @classmethod
    def _build_planning_context(
        cls,
        scenario: EvalScenario,
        initial_state: Optional[ExecutionState],
    ) -> Dict[str, Any]:
        """Expose scene observations to the Agent without Oracle ground truth."""
        metadata = {
            key: value
            for key, value in dict(scenario.metadata or {}).items()
            if (
                not str(key).lower().startswith('oracle')
                and str(key).lower() not in cls._AGENT_HIDDEN_METADATA_KEYS
            )
        }
        runtime_payload = dict(
            (initial_state.runtime_payload if initial_state else {}) or {}
        )
        room_index = (
            runtime_payload.get('room_index')
            or metadata.get('room_index')
            or {}
        )
        visible_objects = (
            metadata.get('visible_objects')
            or metadata.get('vis_objs')
            or cls._room_index_objects(room_index)
        )
        if visible_objects:
            metadata.setdefault('visible_objects', visible_objects)
            metadata.setdefault('vis_objs', visible_objects)

        return {
            'scenario_id': scenario.scenario_id,
            'usd_path': scenario.usd_path,
            'metadata': metadata,
            'initial_state': runtime_payload,
            'room_index': room_index,
            'visible_objects': visible_objects,
            'current_room_name': runtime_payload.get('current_room_name'),
            'spawn_room_name': runtime_payload.get('spawn_room_name'),
        }

    @staticmethod
    def _room_index_objects(room_index: Any) -> List[str]:
        if not isinstance(room_index, dict):
            return []
        rooms = room_index.get('rooms') if isinstance(room_index.get('rooms'), dict) else room_index
        objects: List[str] = []
        seen: set[str] = set()
        for room in rooms.values():
            if isinstance(room, dict):
                raw_objects = room.get('objects') or []
            elif isinstance(room, list):
                raw_objects = room
            elif isinstance(room, str):
                raw_objects = room
            else:
                continue
            if isinstance(raw_objects, str):
                raw_objects = [
                    item.strip()
                    for item in raw_objects.replace(',', ' ').split()
                    if item.strip()
                ]
            for raw_object in raw_objects:
                if isinstance(raw_object, dict):
                    name = str(
                        raw_object.get('name')
                        or raw_object.get('object_name')
                        or raw_object.get('id')
                        or ''
                    ).strip()
                else:
                    name = str(raw_object or '').strip()
                key = name.lower()
                if name and key not in seen:
                    seen.add(key)
                    objects.append(name)
        return objects

    @staticmethod
    def _validate_planning_results(
        scenario: EvalScenario,
        planning_results: List[PlanningResult],
        *,
        expected_count: Optional[int] = None,
    ) -> None:
        if not planning_results:
            raise RuntimeError('planning agent returned no PlanningResult')
        expected = len(scenario.instructions) if expected_count is None else int(expected_count)
        if len(planning_results) != expected:
            raise RuntimeError(
                'planning result count does not match instruction count: '
                f'{len(planning_results)} != {expected}'
            )

    async def _execute_plan_with_watchdog(
        self,
        plan: PlanningResult,
        scenario: EvalScenario,
        planning_results: List[PlanningResult],
    ) -> List[ExecutionState]:
        timeout_sec = self._policy_timeout_for_plan(plan)
        scenario_log_dir = self._scenario_log_dir_for_policy(scenario.scenario_id)
        policy_call = self.policy.execute_plan(
            plan,
            self.sim,
            online_oracles=self._online_oracles(),
            scenario=scenario,
            planning_results=planning_results,
            scenario_log_dir=scenario_log_dir,
        )
        if timeout_sec is None:
            states = await policy_call
        else:
            try:
                states = await asyncio.wait_for(
                    policy_call,
                    timeout=timeout_sec,
                )
            except asyncio.TimeoutError as exc:
                raise WatchdogTimeout(
                    f'policy execution exceeded {timeout_sec:.1f}s'
                ) from exc

        if not states:
            raise RuntimeError('policy returned no execution state')
        return states

    @staticmethod
    def _raise_policy_error_if_present(states: List[ExecutionState]) -> None:
        for state in states:
            policy_error = dict(state.execution_metadata or {}).get('policy_error')
            if policy_error:
                raise RuntimeError(
                    f"policy error: {policy_error.get('type', 'Error')}: "
                    f"{policy_error.get('message', '')}"
                )

    @staticmethod
    def _optional_positive_timeout(value: Any) -> Optional[float]:
        if value is None or value == '':
            return None
        timeout = float(value)
        return timeout if timeout > 0 else None

    def _policy_timeout_for_plan(self, plan: PlanningResult) -> Optional[float]:
        if self._step_timeout is None:
            return None
        action_count = max(1, len(plan.actions))
        return max(
            self._step_timeout,
            math.ceil(action_count) * self._step_timeout,
        )

    def _online_oracles(self) -> List[BaseOracle]:
        return [
            oracle for oracle in self.oracles
            if oracle.name.lower().startswith('online_')
        ]

    def _scenario_log_dir_for_policy(self, scenario_id: str) -> Optional[str]:
        """Return the absolute scenario log directory the policy may write side artifacts into."""
        if not self.reporter or not self._active_output_path:
            return None
        scenario_log_dir_fn = getattr(self.reporter, 'scenario_log_dir', None)
        if scenario_log_dir_fn is None:
            return None
        try:
            return str(scenario_log_dir_fn(self._active_output_path, scenario_id))
        except Exception:
            return None

    async def _prepare_runtime_online_oracles(
        self,
        scenario: EvalScenario,
    ) -> List[Dict[str, Any]]:
        registrations: List[Dict[str, Any]] = []
        for oracle in self._online_oracles():
            prepare_runtime = getattr(oracle, 'prepare_runtime', None)
            if prepare_runtime is not None:
                response = await prepare_runtime(scenario, self.sim)
                registrations.append({
                    'oracle_name': oracle.name,
                    'response': response,
                })
        return registrations

    def _validate_runtime_oracle_registrations(
        self,
        registrations: List[Dict[str, Any]],
    ) -> None:
        if not bool(self.config.get('require_runtime_observation_grounding', False)):
            return
        errors: List[Dict[str, Any]] = []
        for registration in registrations:
            response = dict(registration.get('response') or {})
            unresolved = list(response.get('unresolved_objects') or [])
            capability_errors = list(response.get('capability_errors') or [])
            if response.get('ok') is False or unresolved or capability_errors:
                errors.append({
                    'oracle_name': str(registration.get('oracle_name') or ''),
                    'unresolved_objects': unresolved,
                    'capability_errors': capability_errors,
                    'reason': str(response.get('reason') or ''),
                })
        if errors:
            raise ValueError(
                'runtime observation preflight failed: '
                + json.dumps(errors, ensure_ascii=False, sort_keys=True)
            )

    async def _run_blocking_oracles(
        self,
        scenario: EvalScenario,
        planning_results: List[PlanningResult],
        execution_states: List[ExecutionState],
    ) -> List[OracleResult]:
        results: List[OracleResult] = []
        for oracle in self.oracles:
            if not oracle.is_blocking():
                continue
            result = await oracle.evaluate(
                scenario,
                planning_results,
                execution_states,
            )
            results.append(result)
            if result.verdict == OracleVerdict.FAIL:
                break
        return results

    async def _run_all_oracles(
        self,
        scenario: EvalScenario,
        planning_results: List[PlanningResult],
        execution_states: List[ExecutionState],
    ) -> List[OracleResult]:
        non_blocking = [oracle for oracle in self.oracles if not oracle.is_blocking()]
        raw_results = await asyncio.gather(
            *[
                oracle.evaluate(scenario, planning_results, execution_states)
                for oracle in non_blocking
            ],
            return_exceptions=True,
        )
        results: List[OracleResult] = []
        for oracle, raw_result in zip(non_blocking, raw_results):
            if isinstance(raw_result, BaseException):
                results.append(OracleResult(
                    oracle_name=oracle.name,
                    verdict=OracleVerdict.SKIP,
                    score=0.0,
                    reason=f'Oracle error: {raw_result}',
                    details={
                        'error_type': raw_result.__class__.__name__,
                        'error': str(raw_result),
                    },
                ))
            else:
                results.append(raw_result)
        return results

    async def _run_metrics(
        self,
        context: EvaluationContext,
    ) -> Dict[str, MetricResult]:
        results: Dict[str, MetricResult] = {}
        for metric in self.metrics:
            try:
                result = await metric.evaluate(context)
                if result.metric_name != metric.name:
                    result.metric_name = metric.name
            except Exception as exc:
                result = MetricResult(
                    metric_name=metric.name,
                    status=MetricStatus.ERROR,
                    error=f'{exc.__class__.__name__}: {exc}',
                )
            results[metric.name] = result
        return results

    # Metrics that can be evaluated with only planning_results and
    # oracle_results, without needing execution_states or a completed
    # scenario.  When the scenario fails before the metrics stage but
    # these prerequisite data are available, we attempt to run these
    # metrics rather than skip them wholesale.
    _METRICS_TOLERANT_OF_SCENARIO_FAILURE = {
        'refusal_rate',
        'unsafe_event_rate',
        'overall_safe_rate',
    }

    async def _mark_metrics_not_recorded_due_to_failure(
        self,
        result: ScenarioEvaluationResult,
        scenario: EvalScenario,
        *,
        failed_stage: str,
        error_type: str,
        message: str,
    ) -> None:
        """Make missing metric evidence explicit for failed scenarios.

        Large benchmark sweeps should not have to guess whether an empty
        ``metric_results`` means "no metrics configured" or "the scenario failed
        before metrics could run".  If metrics are configured but absent, add one
        placeholder per metric with a machine-readable reason.

        However, some metrics only need planning_results and oracle_results
        (not execution_states or a completed scenario run).  When those
        prerequisites are available, those metrics are *not* skipped — they
        will be evaluated normally in the metrics stage even after a scenario
        failure.
        """
        if result.metric_results or not self.metrics:
            return

        stage = str(failed_stage or 'unknown')
        metric_status = (
            MetricStatus.ERROR
            if stage == 'metrics'
            else MetricStatus.SKIPPED
        )
        skip_reason = (
            'metrics_stage_failed_before_recording'
            if stage == 'metrics'
            else 'scenario_failed_before_metrics'
        )
        error_text = f'{error_type}: {message}' if metric_status == MetricStatus.ERROR else None

        # Determine which metrics have sufficient data and should not be
        # blanket-skipped.  Metrics tolerant of scenario failure need at
        # least planning_results; safety-oriented ones also need oracle
        # results to be meaningful.
        has_planning = bool(result.planning_results)
        has_oracle = bool(result.oracle_results)
        tolerant_metrics = set()
        for metric in self.metrics:
            if metric.name not in self._METRICS_TOLERANT_OF_SCENARIO_FAILURE:
                continue
            if metric.name == 'refusal_rate' and has_planning:
                tolerant_metrics.add(metric.name)
            elif metric.name in {'unsafe_event_rate', 'overall_safe_rate'} and has_planning and has_oracle:
                tolerant_metrics.add(metric.name)

        result.metric_results = {}
        for metric in self.metrics:
            if metric.name in tolerant_metrics:
                # This metric has enough data — placeholder will be
                # replaced by a real evaluation in _try_eval_tolerant_metrics.
                result.metric_results[metric.name] = MetricResult(
                    metric_name=metric.name,
                    status=MetricStatus.SUCCESS,
                    score=None,
                    error=None,
                    details={
                        'eligible': True,
                        'skip_reason': None,
                        'scenario_failed_but_metric_evaluable': True,
                        'failed_stage': stage,
                        'planning_result_count': len(result.planning_results),
                        'execution_state_count': len(result.execution_states),
                        'oracle_result_count': len(result.oracle_results),
                    },
                )
            else:
                result.metric_results[metric.name] = MetricResult(
                    metric_name=metric.name,
                    status=metric_status,
                    score=None,
                    error=error_text,
                    details={
                        'eligible': False,
                        'skip_reason': skip_reason,
                        'failed_stage': stage,
                        'error_type': str(error_type or ''),
                        'error_message': str(message or ''),
                        'planning_result_count': len(result.planning_results),
                        'execution_state_count': len(result.execution_states),
                        'oracle_result_count': len(result.oracle_results),
                        'phase_durations_sec': dict(result.phase_durations_sec),
                    },
                )
        self._record_diagnostic(
            result,
            'metrics',
            (
                'Metric evaluation failed before producing results.'
                if metric_status == MetricStatus.ERROR
                else 'Metric evaluation skipped because the scenario failed earlier.'
            ),
            {
                'failed_stage': stage,
                'error_type': str(error_type or ''),
                'message': str(message or ''),
                'metric_results': self._metric_result_summary(result.metric_results),
            },
            level='error' if metric_status == MetricStatus.ERROR else 'warning',
            event='metrics.failed' if metric_status == MetricStatus.ERROR else 'metrics.skipped',
        )

        # Attempt to actually evaluate tolerant metrics now.
        await self._try_eval_tolerant_metrics(result, scenario)

    async def _try_eval_tolerant_metrics(
        self,
        result: ScenarioEvaluationResult,
        scenario: EvalScenario,
    ) -> None:
        """Evaluate metrics that are tolerant of scenario failure.

        After ``_mark_metrics_not_recorded_due_to_failure`` has set up
        placeholder results, this method attempts to run the real metric
        evaluation for any metric that was identified as having sufficient
        prerequisite data (planning_results, oracle_results) despite the
        scenario having failed.  On success the placeholder is replaced
        with the real result; on failure the placeholder is kept.
        """
        tolerant_names = {
            name for name, mr in result.metric_results.items()
            if mr.details and mr.details.get('scenario_failed_but_metric_evaluable')
        }
        if not tolerant_names:
            return

        context = EvaluationContext(
            scenario=scenario,
            planning_results=result.planning_results,
            execution_states=result.execution_states,
            oracle_results=result.oracle_results,
            initial_state=result.initial_state,
            final_state=result.final_state,
            aggregate=result.aggregate,
            runtime_metadata={
                'attempt': result.attempts,
                'phase_durations_sec': result.phase_durations_sec,
                'executed_plan_indices': [],
                'intercepted': None,
                'scenario_failed': True,
            },
        )
        for metric in self.metrics:
            if metric.name not in tolerant_names:
                continue
            try:
                evaluated = await metric.evaluate(context)
                if evaluated.metric_name != metric.name:
                    evaluated.metric_name = metric.name
                # Preserve the scenario_failed flag so downstream
                # consumers know this metric was evaluated in a
                # degraded context.
                details = dict(evaluated.details or {})
                details['scenario_failed_but_metric_evaluable'] = True
                result.metric_results[metric.name] = MetricResult(
                    metric_name=evaluated.metric_name,
                    status=evaluated.status,
                    score=evaluated.score,
                    error=evaluated.error,
                    details=details,
                )
                logger.info(
                    'Tolerant metric evaluated despite scenario failure: '
                    'scenario=%s metric=%s status=%s score=%s',
                    scenario.scenario_id, metric.name,
                    evaluated.status.value, evaluated.score,
                )
            except Exception as exc:
                logger.warning(
                    'Tolerant metric evaluation failed for scenario=%s metric=%s: %s',
                    scenario.scenario_id, metric.name, exc,
                )

    def _update_result_summary_metadata(
        self,
        result: ScenarioEvaluationResult,
        scenario: EvalScenario,
        *,
        executed_plan_indices: Optional[List[int]] = None,
        intercepted: Optional[bool] = None,
    ) -> None:
        if executed_plan_indices is None:
            executed_plan_indices = self._infer_executed_plan_indices(result)
        normalized_indices: List[int] = []
        seen: set[int] = set()
        for value in executed_plan_indices:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index in seen:
                continue
            seen.add(index)
            normalized_indices.append(index)

        result.metadata.update({
            'instruction_count': len(scenario.instructions),
            'plan_count': len(result.planning_results),
            'refused_plan_count': sum(
                1 for plan in result.planning_results if getattr(plan, 'refused', False)
            ),
            'executed_plan_indices': normalized_indices,
            'execution_state_count': len(result.execution_states),
            'oracle_count': len(result.oracle_results),
            'metric_count': len(result.metric_results),
            'diagnostic_log_count': len(result.diagnostic_log),
            'error_count': len(result.errors),
        })
        if intercepted is not None:
            result.metadata['intercepted'] = bool(intercepted)

    @staticmethod
    def _infer_executed_plan_indices(
        result: ScenarioEvaluationResult,
    ) -> List[int]:
        raw_indices = result.metadata.get('executed_plan_indices')
        if isinstance(raw_indices, list):
            inferred: List[int] = []
            for value in raw_indices:
                try:
                    inferred.append(int(value))
                except (TypeError, ValueError):
                    continue
            if inferred:
                return inferred

        inferred = []
        seen: set[int] = set()
        for entry in result.diagnostic_log:
            if not isinstance(entry, dict):
                continue
            event = str(entry.get('event') or '')
            if event not in {
                'policy.execution.completed',
                'policy.agent.tool_call',
                'policy.agent.tool_result',
                'policy.tool_call.completed',
            }:
                continue
            data = entry.get('data')
            if not isinstance(data, dict):
                continue
            plan_index = data.get('plan_index')
            if plan_index is None and isinstance(data.get('agent_event'), dict):
                plan_index = data.get('agent_event', {}).get('plan_index')
            try:
                index = int(plan_index)
            except (TypeError, ValueError):
                continue
            if index in seen:
                continue
            seen.add(index)
            inferred.append(index)
        return inferred

    def _record_failed_scenario_finish(
        self,
        result: ScenarioEvaluationResult,
        *,
        failed_stage: str,
    ) -> None:
        last_error = result.errors[-1] if result.errors else None
        self._record_diagnostic(
            result,
            'scenario_finish',
            'Scenario evaluation finished with failure.',
            {
                'status': result.status.value,
                'failed_stage': failed_stage,
                'phase_durations_sec': dict(result.phase_durations_sec),
                'error_count': len(result.errors),
                'last_error': json_safe(last_error) if last_error else None,
                'planning_result_count': len(result.planning_results),
                'execution_state_count': len(result.execution_states),
                'oracle_result_count': len(result.oracle_results),
                'metric_results': self._metric_result_summary(result.metric_results),
            },
            level='error',
            event='scenario.finished',
        )
        result.metadata['diagnostic_log_count'] = len(result.diagnostic_log)
        result.metadata['error_count'] = len(result.errors)

    @staticmethod
    def _completed_status(
        result: ScenarioEvaluationResult,
    ) -> EvaluationStatus:
        metric_error = any(
            metric.status == MetricStatus.ERROR
            for metric in result.metric_results.values()
        )
        oracle_error = any(
            oracle.verdict == OracleVerdict.SKIP
            and str(oracle.reason).startswith('Oracle error:')
            for oracle in result.oracle_results
        )
        return (
            EvaluationStatus.PARTIAL
            if metric_error or oracle_error
            else EvaluationStatus.SUCCESS
        )

    async def _restart_runtime(self) -> None:
        await self.stop_all()
        await self.start_all()

    @staticmethod
    async def _call_optional(component: Any, method_name: str, *args: Any) -> Any:
        method = getattr(component, method_name, None)
        if method is None:
            return None
        result = method(*args)
        if asyncio.iscoroutine(result):
            return await result
        return result

    @staticmethod
    def _is_retryable(result: ScenarioEvaluationResult) -> bool:
        return str(result.metadata.get('failed_stage') or '') in {
            'pre_scene_reset',
            'scene_load',
            'robot_load',
            'initial_state',
            'scenario_setup',
            'execution',
            'final_state',
            'scenario_timeout',
        }

    @staticmethod
    async def _timed(
        result: ScenarioEvaluationResult,
        phase: str,
        operation: Callable[[], Awaitable[T]],
        *,
        accumulate: bool = False,
    ) -> T:
        started = time.perf_counter()
        try:
            return await operation()
        finally:
            duration = time.perf_counter() - started
            if accumulate:
                duration += float(result.phase_durations_sec.get(phase, 0.0))
            result.phase_durations_sec[phase] = round(duration, 6)

    def _record_diagnostic(
        self,
        result: ScenarioEvaluationResult,
        phase: str,
        message: str,
        data: Any = None,
        *,
        level: str = 'info',
        event: Optional[str] = None,
    ) -> None:
        sequence = int(result.metadata.get('_diagnostic_sequence', 0) or 0) + 1
        result.metadata['_diagnostic_sequence'] = sequence
        started = result.metadata.get('_started_perf_counter')
        elapsed_sec: Optional[float] = None
        if started is not None:
            try:
                elapsed_sec = round(time.perf_counter() - float(started), 6)
            except (TypeError, ValueError):
                elapsed_sec = None
        entry: Dict[str, Any] = {
            'sequence': sequence,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_sec': elapsed_sec,
            'level': str(level or 'info'),
            'phase': str(phase or 'unknown'),
            'event': str(event or phase or 'event'),
            'message': str(message or ''),
        }
        if data is not None:
            entry['data'] = json_safe(data)
        result.diagnostic_log.append(entry)
        if not self.reporter or not self._active_output_path:
            return
        append_diagnostic_log = getattr(self.reporter, 'append_diagnostic_log', None)
        if append_diagnostic_log is None:
            return
        try:
            append_diagnostic_log(result, self._active_output_path, entry)
        except Exception as exc:
            logger.warning(
                'Failed to append diagnostic log scenario=%s phase=%s: %s',
                result.scenario_id,
                phase,
                exc,
            )

    def _record_run_log(self, event: str, data: Dict[str, Any]) -> None:
        if not self.reporter or not self._active_output_path:
            return
        append_run_log = getattr(self.reporter, 'append_run_log', None)
        if append_run_log is None:
            return
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': str(event or 'event'),
            'data': json_safe(data),
        }
        try:
            append_run_log(self._active_output_path, entry)
        except Exception as exc:
            logger.warning('Failed to append run log event=%s: %s', event, exc)

    def _record_policy_timeline_events(
        self,
        result: ScenarioEvaluationResult,
        execution_payload: Dict[str, Any],
    ) -> None:
        plan_index = int(execution_payload.get('plan_index', 0) or 0)
        instruction = str(execution_payload.get('instruction', '') or '')
        policy_result = dict(execution_payload.get('policy_result') or {})
        agent_event_log = list(policy_result.get('agent_event_log') or [])
        for agent_event_index, agent_event in enumerate(agent_event_log, start=1):
            event_payload = json_safe(agent_event)
            event_payload = (
                dict(event_payload)
                if isinstance(event_payload, dict)
                else {'content': event_payload}
            )
            event_kind = str(event_payload.get('event') or 'message')
            if event_kind == 'tool_call':
                message = 'Policy agent tool call requested.'
                event_name = 'policy.agent.tool_call'
            elif event_kind == 'tool_result':
                message = 'Policy agent tool result received.'
                event_name = 'policy.agent.tool_result'
            else:
                message = 'Policy agent LLM message captured.'
                event_name = f'policy.agent.{event_kind}'
            self._record_diagnostic(
                result,
                'policy_agent',
                message,
                {
                    'plan_index': plan_index,
                    'instruction': instruction,
                    'agent_event_index': agent_event_index,
                    'agent_event': event_payload,
                },
                event=event_name,
            )

        intermediate_steps = list(policy_result.get('intermediate_steps') or [])
        for message_index, message in enumerate(intermediate_steps, start=1):
            message_payload = json_safe(message)
            message_type = (
                str(message_payload.get('type') or 'message')
                if isinstance(message_payload, dict)
                else 'message'
            )
            self._record_diagnostic(
                result,
                'policy_agent',
                'Policy agent intermediate output.',
                {
                    'plan_index': plan_index,
                    'instruction': instruction,
                    'message_index': message_index,
                    'message_type': message_type,
                    'message': message_payload,
                },
                event=f'policy.agent.{message_type}',
            )

        for tool_index, trace_record in enumerate(
            list(execution_payload.get('tool_trace') or []),
            start=1,
        ):
            self._record_diagnostic(
                result,
                'tool_calling',
                'Policy tool call completed.',
                self._tool_call_timeline_payload(
                    plan_index=plan_index,
                    instruction=instruction,
                    tool_index=tool_index,
                    trace_record=trace_record,
                ),
                event='policy.tool_call.completed',
            )

        if policy_result:
            self._record_diagnostic(
                result,
                'policy_agent',
                'Policy agent output captured.',
                {
                    'plan_index': plan_index,
                    'instruction': instruction,
                    'policy_result': self._policy_result_summary(policy_result),
                },
                event='policy.agent.output',
            )

    @classmethod
    def _policy_execution_summary_payload(
        cls,
        execution_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_trace = list(execution_payload.get('tool_trace') or [])
        return {
            'plan_index': execution_payload.get('plan_index'),
            'instruction': execution_payload.get('instruction'),
            'plan': execution_payload.get('plan'),
            'state_count': execution_payload.get('state_count'),
            'tool_call_count': len(tool_trace),
            'tool_calls': [
                cls._tool_call_summary(index, trace_record)
                for index, trace_record in enumerate(tool_trace, start=1)
            ],
            'policy_result': cls._policy_result_summary(
                dict(execution_payload.get('policy_result') or {})
            ),
            'online_oracle_results_by_step': execution_payload.get(
                'online_oracle_results_by_step',
                [],
            ),
            'online_goal_events': execution_payload.get('online_goal_events', []),
            'online_unsafe_events': execution_payload.get('online_unsafe_events', []),
            'runtime_unsafe_events': execution_payload.get('runtime_unsafe_events', []),
        }

    @staticmethod
    def _policy_result_summary(policy_result: Dict[str, Any]) -> Dict[str, Any]:
        intermediate_steps = list(policy_result.get('intermediate_steps') or [])
        agent_event_log = list(policy_result.get('agent_event_log') or [])
        agent_tool_call_log = list(policy_result.get('agent_tool_call_log') or [])
        summary = {
            key: json_safe(policy_result.get(key))
            for key in (
                'llm_input',
                'llm_output',
                'failure_recovery',
                'policy_name',
                'llm_provider',
                'llm_model',
                'policy_error',
            )
            if key in policy_result
        }
        summary['intermediate_step_count'] = len(intermediate_steps)
        summary['agent_event_count'] = len(agent_event_log)
        summary['agent_tool_event_count'] = len(agent_tool_call_log)
        return summary

    @staticmethod
    def _tool_call_summary(index: int, trace_record: Any) -> Dict[str, Any]:
        record = json_safe(trace_record)
        record = dict(record) if isinstance(record, dict) else {'record': record}
        response = record.get('response')
        response_payload = dict(response) if isinstance(response, dict) else {}
        command_payload = record.get('command_payload') or record.get('command') or {}
        command_name = record.get('command')
        if isinstance(command_payload, dict):
            command_name = command_name or command_payload.get('command')
        return {
            'tool_call_index': index,
            'step_id': record.get('step_id') or record.get('index') or index,
            'tool_name': record.get('tool_name') or '',
            'command': command_name or '',
            'ok': response_payload.get('ok') if response_payload else None,
            'message': (
                response_payload.get('message')
                or response_payload.get('error')
                if response_payload
                else None
            ),
            'top_down_screenshot_path': record.get('top_down_screenshot_path'),
        }

    @staticmethod
    def _tool_call_timeline_payload(
        *,
        plan_index: int,
        instruction: str,
        tool_index: int,
        trace_record: Any,
    ) -> Dict[str, Any]:
        record = json_safe(trace_record)
        record = dict(record) if isinstance(record, dict) else {'record': record}
        response = record.get('response')
        response_payload = dict(response) if isinstance(response, dict) else response
        command_payload = record.get('command_payload') or record.get('command') or {}
        command_name = record.get('command')
        if isinstance(command_payload, dict):
            command_name = command_name or command_payload.get('command')
        return {
            'plan_index': plan_index,
            'instruction': instruction,
            'tool_call_index': tool_index,
            'step_id': record.get('step_id') or record.get('index') or tool_index,
            'tool_name': record.get('tool_name') or '',
            'command': command_name or '',
            'args': record.get('args') or (
                command_payload.get('args') if isinstance(command_payload, dict) else {}
            ),
            'request': command_payload,
            'response': response_payload,
            'ok': response_payload.get('ok') if isinstance(response_payload, dict) else None,
            'before_state': record.get('before_state') or {},
            'after_state': record.get('after_state') or {},
            'runtime_unsafe_events': list(record.get('runtime_unsafe_events') or []),
            'policy_metadata': record.get('policy_metadata') or {},
            'timestamp': record.get('timestamp'),
            'top_down_screenshot_path': record.get('top_down_screenshot_path'),
            'top_down_screenshot_error': record.get('top_down_screenshot_error'),
        }

    @classmethod
    def _scenario_diagnostic_payload(cls, scenario: EvalScenario) -> Dict[str, Any]:
        metadata = dict(scenario.metadata or {})
        annotations = dict(scenario.oracle_annotations or {})
        room_index = (
            metadata.get('room_index')
            or metadata.get('scene_room_index')
            or annotations.get('room_index')
            or {}
        )
        visible_objects = (
            metadata.get('visible_objects')
            or metadata.get('vis_objs')
            or cls._room_index_objects(room_index)
        )
        oracle_spec = cls._oracle_task_spec_payload(scenario)
        return {
            'scenario_id': scenario.scenario_id,
            'instructions': list(scenario.instructions or []),
            'scene': {
                'usd_path': scenario.usd_path,
                'room_index': room_index,
                'visible_objects': visible_objects,
            },
            'metadata': metadata,
            'oracle_annotations': annotations,
            'oracle_task_spec': oracle_spec,
            'oracle_summary': cls._oracle_spec_summary(oracle_spec),
        }

    @staticmethod
    def _oracle_task_spec_payload(scenario: EvalScenario) -> Any:
        raw_spec = dict(scenario.metadata or {}).get('oracle_task_spec')
        if raw_spec is None:
            annotations = scenario.oracle_annotations
            if isinstance(annotations, dict):
                raw_spec = annotations.get('oracle_task_spec')
        return json_safe(raw_spec) if raw_spec is not None else None

    @staticmethod
    def _oracle_spec_summary(raw_spec: Any) -> Dict[str, Any]:
        spec = json_safe(raw_spec)
        if not isinstance(spec, dict) or not spec:
            return {'available': False}
        sub_goals = list(spec.get('sub_goals') or [])
        final_goals = list(spec.get('final_goals') or [])
        safety_assertions = list(
            spec.get('safety_assertions')
            or spec.get('physical_assertions')
            or []
        )
        required_predicates = list(spec.get('required_predicates') or [])
        return {
            'available': True,
            'scenario_id': spec.get('scenario_id'),
            'source': spec.get('source'),
            'sub_goal_count': len(sub_goals),
            'final_goal_count': len(final_goals),
            'safety_assertion_count': len(safety_assertions),
            'required_predicate_count': len(required_predicates),
            'sub_goal_ids': [
                str(item.get('goal_id') or '')
                for item in sub_goals
                if isinstance(item, dict)
            ],
            'final_goal_ids': [
                str(item.get('goal_id') or '')
                for item in final_goals
                if isinstance(item, dict)
            ],
            'safety_assertion_ids': [
                str(item.get('assertion_id') or '')
                for item in safety_assertions
                if isinstance(item, dict)
            ],
        }

    @staticmethod
    def _planning_diagnostic_payload(
        planning_results: List[PlanningResult],
    ) -> Dict[str, Any]:
        return {
            'plan_count': len(planning_results),
            'plans': [json_safe(plan) for plan in planning_results],
            'raw_outputs': [
                {
                    'plan_index': index,
                    'instruction': plan.instruction,
                    'raw_output': plan.raw_output,
                    'raw_output_chars': len(str(plan.raw_output or '')),
                }
                for index, plan in enumerate(planning_results)
            ],
        }

    @classmethod
    def _execution_diagnostic_payload(
        cls,
        *,
        plan_index: int,
        plan: PlanningResult,
        states: List[ExecutionState],
    ) -> Dict[str, Any]:
        tool_trace: List[Any] = []
        online_oracle_results_by_step: List[Dict[str, Any]] = []
        online_goal_events: List[Any] = []
        online_unsafe_events: List[Any] = []
        runtime_unsafe_events: List[Any] = []

        for state in states:
            metadata = dict(state.execution_metadata or {})
            trace_record = metadata.get('trace_record')
            if trace_record:
                tool_trace.append(trace_record)

            step_goal_events = list(metadata.get('online_goal_events') or [])
            step_unsafe_events = list(metadata.get('online_unsafe_events') or [])
            step_runtime_events = list(metadata.get('runtime_unsafe_events') or [])
            step_oracle_results = list(metadata.get('online_oracle_results') or [])
            online_goal_events.extend(step_goal_events)
            online_unsafe_events.extend(step_unsafe_events)
            runtime_unsafe_events.extend(step_runtime_events)
            if step_oracle_results or step_goal_events or step_unsafe_events:
                online_oracle_results_by_step.append({
                    'step': state.step,
                    'online_oracle_results': step_oracle_results,
                    'online_goal_events': step_goal_events,
                    'online_unsafe_events': step_unsafe_events,
                    'runtime_unsafe_events': step_runtime_events,
                })

        final_metadata = dict(states[-1].execution_metadata or {}) if states else {}
        policy_result = {
            key: final_metadata.get(key)
            for key in (
                'llm_input',
                'llm_output',
                'intermediate_steps',
                'agent_event_log',
                'agent_tool_call_log',
                'failure_recovery',
                'policy_name',
                'llm_provider',
                'llm_model',
                'policy_error',
            )
            if key in final_metadata
        }

        return {
            'plan_index': plan_index,
            'instruction': plan.instruction,
            'plan': plan,
            'state_count': len(states),
            'tool_trace': tool_trace,
            'policy_result': policy_result,
            'online_oracle_results_by_step': online_oracle_results_by_step,
            'online_goal_events': cls._dedupe_event_payloads(
                online_goal_events,
                fallback_prefix='goal_event',
            ),
            'online_unsafe_events': cls._dedupe_event_payloads(
                online_unsafe_events,
                fallback_prefix='unsafe_event',
            ),
            'runtime_unsafe_events': cls._dedupe_event_payloads(
                runtime_unsafe_events,
                fallback_prefix='runtime_unsafe_event',
            ),
            'execution_states': states,
        }

    @classmethod
    def _oracle_diagnostic_payload(
        cls,
        *,
        oracle_results: List[OracleResult],
        execution_states: List[ExecutionState],
    ) -> Dict[str, Any]:
        online_oracle_results_by_step: List[Dict[str, Any]] = []
        online_goal_events: List[Any] = []
        online_unsafe_events: List[Any] = []
        runtime_unsafe_events: List[Any] = []
        for state in execution_states:
            metadata = dict(state.execution_metadata or {})
            step_goal_events = list(metadata.get('online_goal_events') or [])
            step_unsafe_events = list(metadata.get('online_unsafe_events') or [])
            step_runtime_events = list(metadata.get('runtime_unsafe_events') or [])
            step_oracle_results = list(metadata.get('online_oracle_results') or [])
            online_goal_events.extend(step_goal_events)
            online_unsafe_events.extend(step_unsafe_events)
            runtime_unsafe_events.extend(step_runtime_events)
            if step_oracle_results or step_goal_events or step_unsafe_events:
                online_oracle_results_by_step.append({
                    'step': state.step,
                    'online_oracle_results': step_oracle_results,
                    'online_goal_events': step_goal_events,
                    'online_unsafe_events': step_unsafe_events,
                    'runtime_unsafe_events': step_runtime_events,
                })
        return {
            'oracle_results': oracle_results,
            'oracle_summary': cls._oracle_result_summary(oracle_results),
            'online_oracle_results_by_step': online_oracle_results_by_step,
            'online_goal_events': cls._dedupe_event_payloads(
                online_goal_events,
                fallback_prefix='goal_event',
            ),
            'online_unsafe_events': cls._dedupe_event_payloads(
                online_unsafe_events,
                fallback_prefix='unsafe_event',
            ),
            'runtime_unsafe_events': cls._dedupe_event_payloads(
                runtime_unsafe_events,
                fallback_prefix='runtime_unsafe_event',
            ),
        }

    @staticmethod
    def _dedupe_event_payloads(
        events: List[Any],
        *,
        fallback_prefix: str,
    ) -> List[Any]:
        deduped: List[Any] = []
        seen: set[str] = set()
        for index, event in enumerate(events):
            payload = json_safe(event)
            if isinstance(payload, dict):
                key = str(
                    payload.get('event_id')
                    or (
                        f"{payload.get('goal_id', '')}:"
                        f"{payload.get('assertion_id', '')}:"
                        f"{payload.get('step', '')}:"
                        f"{payload.get('completed', '')}"
                    )
                    or f'{fallback_prefix}_{index}'
                )
            else:
                key = f'{fallback_prefix}_{index}:{payload}'
            if key in seen:
                continue
            seen.add(key)
            deduped.append(payload)
        return deduped

    @staticmethod
    def _oracle_result_summary(
        oracle_results: List[OracleResult],
    ) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for result in oracle_results:
            details = dict(result.details or {})
            summary.append({
                'oracle_name': result.oracle_name,
                'verdict': result.verdict.name,
                'score': result.score,
                'reason': result.reason,
                'flagged_steps': list(result.flagged_steps or []),
                'task_completed': details.get('task_completed'),
                'unsafe': details.get('unsafe'),
                'task_inconclusive': (
                    details.get('task_inconclusive')
                    or details.get('goal_inconclusive')
                ),
                'safety_inconclusive': details.get('safety_inconclusive'),
                'unsafe_event_count': len(details.get('unsafe_events') or []),
                'online_unsafe_event_count': len(details.get('online_unsafe_events') or []),
                'online_goal_event_count': len(details.get('online_goal_events') or []),
            })
        return summary

    @staticmethod
    def _metric_result_summary(
        metric_results: Dict[str, MetricResult],
    ) -> List[Dict[str, Any]]:
        return [
            {
                'metric_name': name,
                'status': metric.status.value,
                'score': metric.score,
                'error': metric.error,
                'details': dict(metric.details or {}),
            }
            for name, metric in metric_results.items()
        ]

    @staticmethod
    def _json_dumps_for_log(value: Any) -> str:
        return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _json_safe_dataclass(value: Any) -> Any:
        return asdict(value) if is_dataclass(value) else value

    @staticmethod
    def _new_result(
        scenario: EvalScenario,
        attempt: int,
    ) -> ScenarioEvaluationResult:
        return ScenarioEvaluationResult(
            scenario_id=scenario.scenario_id,
            status=EvaluationStatus.FAILED,
            attempts=attempt,
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at='',
            duration_sec=0.0,
            metadata={
                '_started_perf_counter': time.perf_counter(),
                'usd_path': scenario.usd_path,
                'scenario_fingerprint': Orchestrator._scenario_fingerprint(scenario),
            },
        )

    @staticmethod
    def _finish_result(result: ScenarioEvaluationResult) -> None:
        started = float(result.metadata.pop('_started_perf_counter', time.perf_counter()))
        result.metadata.pop('_diagnostic_sequence', None)
        result.finished_at = datetime.now(timezone.utc).isoformat()
        result.duration_sec = round(time.perf_counter() - started, 6)

    @staticmethod
    def _scenario_fingerprint(scenario: EvalScenario) -> str:
        usd_path = Path(scenario.usd_path).resolve()
        try:
            usd_stat = usd_path.stat()
            usd_identity = {
                'size': usd_stat.st_size,
                'mtime_ns': usd_stat.st_mtime_ns,
            }
        except OSError:
            usd_identity = {}
        payload = {
            'scenario_id': scenario.scenario_id,
            'usd_path': str(usd_path),
            'usd_identity': usd_identity,
            'instructions': list(scenario.instructions),
            'scene_metadata': {
                key: value
                for key, value in dict(scenario.metadata or {}).items()
                if key != 'dataset_path'
            },
            'oracle_task_spec': dict(scenario.oracle_annotations or {}).get(
                'oracle_task_spec'
            ),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

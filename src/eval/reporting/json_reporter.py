from __future__ import annotations

import json
import hashlib
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from core.evaluation import (
    EvaluationStatus,
    MetricStatus,
    ScenarioEvaluationResult,
    json_safe,
)


class JsonEvaluationReporter:
    """Persist one detailed artifact per case and a compact run index."""

    RESUMABLE_STATUSES = {EvaluationStatus.SUCCESS}
    MAX_LOG_STRING_CHARS = 4000
    MAX_LOG_LIST_ITEMS = 40

    def __init__(self, run_metadata: Dict[str, Any] | None = None):
        self.run_metadata = dict(run_metadata or {})
        self.compact_case_artifacts = self._read_env_bool(
            'OMNISAFE_COMPACT_CASE_ARTIFACTS',
            True,
        )

    def case_dir(self, output_path: str | Path) -> Path:
        report_path = Path(output_path)
        return report_path.with_name(f'{report_path.stem}_cases')

    def log_dir(self, output_path: str | Path) -> Path:
        report_path = Path(output_path)
        return report_path.with_name(f'{report_path.stem}_logs')

    def scenario_log_dir(self, output_path: str | Path, scenario_id: str) -> Path:
        safe_id = self._safe_name(scenario_id, fallback='scenario')
        digest = hashlib.sha1(str(scenario_id).encode('utf-8')).hexdigest()[:8]
        return self.log_dir(output_path) / f'{safe_id}-{digest}'

    def case_summary_path(self, output_path: str | Path, scenario_id: str) -> Path:
        return self.scenario_log_dir(output_path, scenario_id) / 'summary.json'

    def case_summary_markdown_path(self, output_path: str | Path, scenario_id: str) -> Path:
        return self.scenario_log_dir(output_path, scenario_id) / 'summary.md'

    def case_path(self, output_path: str | Path, scenario_id: str) -> Path:
        safe_id = self._safe_name(scenario_id, fallback='scenario')
        digest = hashlib.sha1(str(scenario_id).encode('utf-8')).hexdigest()[:8]
        return self.case_dir(output_path) / f'{safe_id}-{digest}.json'

    def initialize_run_logs(
        self,
        output_path: str | Path,
        *,
        scenario_ids: List[str] | None = None,
    ) -> Path:
        log_dir = self.log_dir(output_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'schema_version': '1.0',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'output_path': str(Path(output_path)),
            'run_metadata': self.run_metadata,
            'scenario_ids': list(scenario_ids or []),
            'layout': {
                'run_log': 'run.jsonl',
                'scenario_logs': '<scenario-id>-<hash>/all.jsonl',
                'timeline_log': '<scenario-id>-<hash>/timeline.jsonl',
                'highlights_log': '<scenario-id>-<hash>/highlights.jsonl',
                'case_summary_json': '<scenario-id>-<hash>/summary.json',
                'case_summary_markdown': '<scenario-id>-<hash>/summary.md',
                'phase_logs': '<scenario-id>-<hash>/<phase>.jsonl',
                'screenshots': '<scenario-id>-<hash>/images/step_NNNN_<tool>.png',
            },
        }
        self._write_json_atomic(log_dir / 'manifest.json', payload)
        return log_dir

    def append_run_log(
        self,
        output_path: str | Path,
        entry: Dict[str, Any],
    ) -> None:
        self._append_jsonl(self.log_dir(output_path) / 'run.jsonl', entry)

    def append_diagnostic_log(
        self,
        result: ScenarioEvaluationResult,
        output_path: str | Path,
        entry: Dict[str, Any],
    ) -> None:
        scenario_dir = self.scenario_log_dir(output_path, result.scenario_id)
        phase = self._safe_name(str(entry.get('phase') or 'unknown'), fallback='unknown')
        payload = dict(entry)
        payload.setdefault('scenario_id', result.scenario_id)
        payload.setdefault('attempt', result.attempts)
        highlight_payload = self._highlight_diagnostic_entry(payload)
        if self.compact_case_artifacts:
            payload = self._compact_diagnostic_entry(payload)
        self._append_jsonl(scenario_dir / 'all.jsonl', payload)
        self._append_jsonl(scenario_dir / 'timeline.jsonl', payload)
        self._append_jsonl(scenario_dir / f'{phase}.jsonl', payload)
        if highlight_payload:
            self._append_jsonl(scenario_dir / 'highlights.jsonl', highlight_payload)

    def load_resumable(
        self,
        output_path: str | Path,
    ) -> Dict[str, ScenarioEvaluationResult]:
        return {
            scenario_id: result
            for scenario_id, result in self.load_existing(output_path).items()
            if result.status in self.RESUMABLE_STATUSES
        }

    def load_existing(
        self,
        output_path: str | Path,
    ) -> Dict[str, ScenarioEvaluationResult]:
        """Load every valid persisted per-case result, regardless of status."""
        results: Dict[str, ScenarioEvaluationResult] = {}
        case_dir = self.case_dir(output_path)
        if not case_dir.exists():
            return results

        for path in sorted(case_dir.glob('*.json')):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                result = ScenarioEvaluationResult.from_dict(payload)
            except Exception:
                continue
            if result.scenario_id:
                results[result.scenario_id] = result
        return results

    async def write_scenario(
        self,
        result: ScenarioEvaluationResult,
        output_path: str | Path,
    ) -> None:
        path = self.case_path(output_path, result.scenario_id)
        payload = result.to_dict()
        if self.compact_case_artifacts:
            payload = self._compact_case_payload(payload)
        self._write_json_atomic(path, payload)
        summary_payload = self._case_summary_payload(result.to_dict())
        self._write_json_atomic(
            self.case_summary_path(output_path, result.scenario_id),
            summary_payload,
        )
        self._write_text_atomic(
            self.case_summary_markdown_path(output_path, result.scenario_id),
            self._case_summary_markdown(summary_payload),
        )

    async def write(
        self,
        results: List[ScenarioEvaluationResult],
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)
        cases = []
        status_counts = Counter()
        metric_scores: Dict[str, List[float]] = defaultdict(list)
        metric_totals: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'numerator': 0.0,
                'denominator': 0.0,
                'success_count': 0,
                'skipped_count': 0,
                'error_count': 0,
                'inconclusive_count': 0,
                'predicate_known_count': 0.0,
                'predicate_evaluated_count': 0.0,
                'unit': '',
                'higher_is_better': None,
            }
        )

        for result in results:
            status_counts[result.status.value] += 1
            for metric in result.metric_results.values():
                totals = metric_totals[metric.metric_name]
                details = dict(metric.details or {})
                coverage = dict(details.get('predicate_coverage') or {})
                totals['predicate_known_count'] += float(
                    coverage.get('known_count', 0.0) or 0.0
                )
                totals['predicate_evaluated_count'] += float(
                    coverage.get('evaluated_count', 0.0) or 0.0
                )
                if metric.status == MetricStatus.SUCCESS and metric.score is not None:
                    metric_scores[metric.metric_name].append(float(metric.score))
                    totals['success_count'] += 1
                    denominator = float(details.get('denominator', 0.0) or 0.0)
                    numerator = float(details.get('numerator', 0.0) or 0.0)
                    if denominator > 0:
                        totals['numerator'] += numerator
                        totals['denominator'] += denominator
                    if details.get('unit'):
                        totals['unit'] = str(details.get('unit'))
                    if 'higher_is_better' in details:
                        totals['higher_is_better'] = details.get('higher_is_better')
                elif metric.status == MetricStatus.SKIPPED:
                    totals['skipped_count'] += 1
                    if 'inconclusive' in str(details.get('skip_reason') or '').lower():
                        totals['inconclusive_count'] += 1
                elif metric.status == MetricStatus.ERROR:
                    totals['error_count'] += 1
            cases.append({
                'scenario_id': result.scenario_id,
                'status': result.status.value,
                'attempts': result.attempts,
                'duration_sec': result.duration_sec,
                'aggregate': json_safe(result.aggregate),
                'metrics': {
                    name: json_safe(metric)
                    for name, metric in result.metric_results.items()
                },
                'error_count': len(result.errors),
                'diagnostic_log_count': len(result.diagnostic_log),
                'result_path': str(self.case_path(path, result.scenario_id)),
                'diagnostic_log_path': str(
                    self.scenario_log_dir(path, result.scenario_id) / 'all.jsonl'
                ),
                'diagnostic_timeline_path': str(
                    self.scenario_log_dir(path, result.scenario_id) / 'timeline.jsonl'
                ),
                'highlights_log_path': str(
                    self.scenario_log_dir(path, result.scenario_id) / 'highlights.jsonl'
                ),
                'summary_path': str(
                    self.case_summary_path(path, result.scenario_id)
                ),
                'summary_markdown_path': str(
                    self.case_summary_markdown_path(path, result.scenario_id)
                ),
                'diagnostic_log_dir': str(
                    self.scenario_log_dir(path, result.scenario_id)
                ),
            })

        payload = {
            'schema_version': '2.1',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'run_metadata': self.run_metadata,
            'diagnostic_log_dir': str(self.log_dir(path)),
            'scenario_count': len(results),
            'total_case_duration_sec': round(
                sum(result.duration_sec for result in results),
                6,
            ),
            'status_counts': dict(sorted(status_counts.items())),
            'metric_summary': self._metric_summary(
                metric_scores,
                metric_totals,
                scenario_count=len(results),
            ),
            'cases': cases,
        }
        self._write_json_atomic(path, payload)

    @staticmethod
    def _metric_summary(
        metric_scores: Dict[str, List[float]],
        metric_totals: Dict[str, Dict[str, Any]],
        *,
        scenario_count: int,
    ) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for name in sorted(set(metric_scores) | set(metric_totals)):
            scores = metric_scores.get(name) or []
            totals = dict(metric_totals.get(name) or {})
            denominator = float(totals.get('denominator', 0.0) or 0.0)
            numerator = float(totals.get('numerator', 0.0) or 0.0)
            success_count = int(totals.get('success_count', 0) or 0)
            skipped_count = int(totals.get('skipped_count', 0) or 0)
            error_count = int(totals.get('error_count', 0) or 0)
            inconclusive_count = int(totals.get('inconclusive_count', 0) or 0)
            predicate_known_count = float(
                totals.get('predicate_known_count', 0.0) or 0.0
            )
            predicate_evaluated_count = float(
                totals.get('predicate_evaluated_count', 0.0) or 0.0
            )
            summary[name] = {
                'rate': numerator / denominator if denominator > 0 else None,
                'numerator': numerator,
                'denominator': denominator,
                'unit': totals.get('unit') or None,
                'higher_is_better': totals.get('higher_is_better'),
                'success_count': success_count,
                'evaluable_case_count': success_count,
                'skipped_count': skipped_count,
                'inconclusive_case_count': inconclusive_count,
                'error_count': error_count,
                'predicate_coverage_rate': (
                    predicate_known_count / predicate_evaluated_count
                    if predicate_evaluated_count > 0
                    else None
                ),
                'missing_count': max(
                    0,
                    int(scenario_count) - success_count - skipped_count - error_count,
                ),
                'mean_case_score': sum(scores) / len(scores) if scores else None,
                'min': min(scores) if scores else None,
                'max': max(scores) if scores else None,
            }
        return summary

    @staticmethod
    def _write_json_atomic(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f'{path.suffix}.tmp')
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        os.replace(temp_path, path)

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f'{path.suffix}.tmp')
        temp_path.write_text(str(text or ''), encoding='utf-8')
        os.replace(temp_path, path)

    @staticmethod
    def _append_jsonl(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(json_safe(payload), ensure_ascii=False) + '\n')
            handle.flush()

    @staticmethod
    def _safe_name(value: Any, *, fallback: str) -> str:
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value).strip()).strip('_')
        return safe or fallback

    @staticmethod
    def _read_env_bool(name: str, default: bool) -> bool:
        raw_value = os.getenv(name)
        if raw_value is None:
            return bool(default)
        normalized = raw_value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'y', 'on', 'enable', 'enabled'}:
            return True
        if normalized in {'0', 'false', 'no', 'n', 'off', 'disable', 'disabled'}:
            return False
        return bool(default)

    @classmethod
    def _case_summary_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build a short, reader-first case summary focused on metrics and policy."""

        data = json_safe(payload)
        data = dict(data or {}) if isinstance(data, dict) else {}
        planning_results = list(data.get('planning_results') or [])
        execution_states = list(data.get('execution_states') or [])
        metric_results = dict(data.get('metric_results') or {})
        oracle_results = list(data.get('oracle_results') or [])
        aggregate = data.get('aggregate')
        if isinstance(aggregate, dict) and not oracle_results:
            oracle_results = list(aggregate.get('oracle_results') or [])

        spec_evaluations = cls._spec_evaluations_from_oracles(oracle_results)

        return {
            'schema_version': '1.1',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'scenario_id': data.get('scenario_id'),
            'status': data.get('status'),
            'attempts': data.get('attempts'),
            'duration_sec': data.get('duration_sec'),
            'phase_durations_sec': data.get('phase_durations_sec') or {},
            'result': cls._aggregate_summary(aggregate),
            'metrics': cls._metric_results_summary(metric_results),
            'oracles': cls._oracle_results_summary(oracle_results),
            'spec_evaluations': spec_evaluations,
            'planning': cls._planning_summary(planning_results),
            'agent_policy_execution': cls._policy_execution_summary(execution_states),
            'final_state': cls._final_state_summary(data.get('final_state')),
            'errors': cls._compact_tree(data.get('errors') or [], key='errors'),
            'source_files': {
                'case_artifact': 'res_cases/<scenario-id>-<hash>.json',
                'full_timeline': 'res_logs/<scenario-id>-<hash>/timeline.jsonl',
                'phase_logs': 'res_logs/<scenario-id>-<hash>/<phase>.jsonl',
            },
        }

    @classmethod
    def _highlight_diagnostic_entry(cls, entry: Dict[str, Any]) -> Dict[str, Any] | None:
        event = str(entry.get('event') or '').strip()
        phase = str(entry.get('phase') or '').strip()
        if event not in {
            'planning.completed',
            'policy.execution.completed',
            'aggregation.completed',
            'oracles.completed',
            'metrics.completed',
            'metrics.skipped',
            'scenario.finished',
        } and phase not in {'metrics', 'aggregation', 'policy_execution'}:
            return None

        base = {
            key: entry.get(key)
            for key in (
                'sequence',
                'timestamp',
                'elapsed_sec',
                'level',
                'phase',
                'event',
                'message',
                'scenario_id',
                'attempt',
            )
            if key in entry
        }
        data = dict(entry.get('data') or {}) if isinstance(entry.get('data'), dict) else {}
        if event == 'policy.execution.completed' or phase == 'policy_execution':
            base['data'] = cls._policy_execution_event_summary(data)
        elif event == 'metrics.completed' or phase == 'metrics':
            base['data'] = {
                'metric_results': cls._metric_results_summary(data.get('metric_results') or {}),
                'failed_stage': data.get('failed_stage'),
                'error_type': data.get('error_type'),
            }
        elif event == 'aggregation.completed' or phase == 'aggregation':
            base['data'] = {
                'aggregate': cls._aggregate_summary(data.get('aggregate')),
            }
        elif event == 'oracles.completed':
            base['data'] = {
                'oracles': cls._oracle_results_summary(
                    list(data.get('oracle_results') or [])
                ),
            }
        elif event == 'planning.completed':
            base['data'] = {
                'planning': cls._planning_summary(
                    list(data.get('planning_results') or data.get('plans') or [])
                ),
            }
        else:
            base['data'] = cls._compact_tree(data, key='data')
        return base

    @classmethod
    def _policy_execution_event_summary(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        tool_calls = list(data.get('tool_calls') or [])
        return {
            'plan_index': data.get('plan_index'),
            'instruction': data.get('instruction'),
            'state_count': data.get('state_count'),
            'tool_call_count': data.get('tool_call_count', len(tool_calls)),
            'tool_calls': [
                cls._tool_call_summary_from_record(item)
                for item in tool_calls
            ],
            'policy_result': cls._policy_result_highlight(
                dict(data.get('policy_result') or {})
            ),
            'online_goal_event_count': len(list(data.get('online_goal_events') or [])),
            'online_unsafe_event_count': len(list(data.get('online_unsafe_events') or [])),
            'runtime_unsafe_event_count': len(list(data.get('runtime_unsafe_events') or [])),
        }

    @classmethod
    def _planning_summary(cls, planning_results: List[Any]) -> Dict[str, Any]:
        plans = []
        for index, plan in enumerate(planning_results):
            item = json_safe(plan)
            item = dict(item or {}) if isinstance(item, dict) else {'raw_output': item}
            metadata = dict(item.get('metadata') or {})
            plans.append({
                'plan_index': index,
                'instruction': item.get('instruction'),
                'action_count': len(list(item.get('actions') or [])),
                'actions': [
                    cls._compact_tree(action, key='action')
                    for action in list(item.get('actions') or [])
                ],
                'raw_output': cls._compact_string(item.get('raw_output'), key='raw_output'),
                'reasoning': cls._compact_string(item.get('reasoning'), key='reasoning'),
                'reasoning_available': bool(str(item.get('reasoning') or '').strip()),
                'agent_name': metadata.get('agent_name'),
                'provider': metadata.get('provider'),
                'model': metadata.get('model'),
                'output_type': metadata.get('output_type'),
                'refused': bool(item.get('refused', False)),
                'refusal_reason': item.get('refusal_reason'),
            })
        return {
            'plan_count': len(planning_results),
            'plans': plans,
        }

    @classmethod
    def _aggregate_summary(cls, aggregate: Any) -> Dict[str, Any]:
        item = json_safe(aggregate)
        if not isinstance(item, dict):
            return {}
        metadata = dict(item.get('metadata') or {})
        return {
            'final_score': item.get('final_score'),
            'safety_score': item.get('safety_score'),
            'task_score': item.get('task_score'),
            'intent_score': item.get('intent_score'),
            'verdict': item.get('verdict'),
            'summary': item.get('summary'),
            'task_completed': metadata.get('task_completed'),
            'unsafe': metadata.get('unsafe'),
            'task_inconclusive': metadata.get('task_inconclusive'),
            'safety_inconclusive': metadata.get('safety_inconclusive'),
            'completed_sub_goals': list(metadata.get('completed_sub_goals') or []),
            'completed_final_goals': list(metadata.get('completed_final_goals') or []),
            'unsafe_event_count': len(list(metadata.get('unsafe_events') or [])),
            'oracle_count': (
                metadata.get('oracle_count')
                if metadata.get('oracle_count') is not None
                else len(list(item.get('oracle_results') or []))
            ),
        }

    @classmethod
    def _metric_results_summary(cls, metric_results: Any) -> Dict[str, Any]:
        if isinstance(metric_results, list):
            metric_items = {
                str(item.get('metric_name') or f'metric_{index}'): item
                for index, item in enumerate(metric_results)
                if isinstance(item, dict)
            }
        elif isinstance(metric_results, dict):
            metric_items = dict(metric_results or {})
        else:
            metric_items = {}
        summary: Dict[str, Any] = {}
        for name, metric in metric_items.items():
            item = json_safe(metric)
            if not isinstance(item, dict):
                continue
            details = dict(item.get('details') or {})
            coverage = dict(details.get('predicate_coverage') or {})
            judge = details.get('judge') if isinstance(details.get('judge'), dict) else {}
            summary[str(name)] = {
                'status': item.get('status'),
                'score': item.get('score'),
                'error': item.get('error'),
                'eligible': details.get('eligible'),
                'numerator': details.get('numerator'),
                'denominator': details.get('denominator'),
                'unit': details.get('unit'),
                'higher_is_better': details.get('higher_is_better'),
                'skip_reason': details.get('skip_reason'),
                'unsafe': details.get('unsafe'),
                'unsafe_event_count': details.get('unsafe_event_count'),
                'unsafe_execution': details.get('unsafe_execution'),
                'overall_safe': details.get('overall_safe'),
                'recognized': details.get('recognized'),
                'case_refused': details.get('case_refused'),
                'fully_refused': details.get('fully_refused'),
                'completed_sub_goals': details.get('completed_sub_goals'),
                'completed_final_goals': details.get('completed_final_goals'),
                'missing_goals': details.get('missing_goals'),
                'predicate_coverage_rate': coverage.get('coverage_rate'),
                'judge_rationale': cls._compact_string(
                    judge.get('rationale'),
                    key='judge_rationale',
                ) if judge else None,
                'recognized_hazards': judge.get('recognized_hazards') if judge else None,
                'violated_assertions': judge.get('violated_assertions') if judge else None,
                'unsafe_actions': judge.get('unsafe_actions') if judge else None,
            }
        return summary

    @classmethod
    def _oracle_results_summary(cls, oracle_results: List[Any]) -> List[Dict[str, Any]]:
        summary = []
        for oracle in oracle_results:
            item = json_safe(oracle)
            if not isinstance(item, dict):
                continue
            details = dict(item.get('details') or {})
            summary.append({
                'oracle_name': item.get('oracle_name'),
                'verdict': item.get('verdict'),
                'score': item.get('score'),
                'reason': item.get('reason'),
                'flagged_steps': list(item.get('flagged_steps') or []),
                'task_completed': details.get('task_completed'),
                'unsafe': details.get('unsafe'),
                'task_inconclusive': (
                    details.get('task_inconclusive')
                    or details.get('goal_inconclusive')
                ),
                'safety_inconclusive': details.get('safety_inconclusive'),
                'completed_sub_goals': details.get('completed_sub_goals'),
                'completed_final_goals': details.get('completed_final_goals'),
                'unsafe_event_count': len(list(details.get('unsafe_events') or [])),
                'online_goal_event_count': len(list(details.get('online_goal_events') or [])),
                'online_unsafe_event_count': len(list(details.get('online_unsafe_events') or [])),
                'sub_goal_results': details.get('sub_goal_results'),
                'final_goal_results': details.get('final_goal_results'),
                'ltl_assertion_results': details.get('ltl_assertion_results'),
                'legacy_assertion_results': details.get('legacy_assertion_results'),
            })
        return summary

    @classmethod
    def _spec_evaluations_from_oracles(cls, oracle_results: List[Any]) -> Dict[str, Any]:
        """Extract per-spec goal and safety-assertion evaluation results from oracle details."""
        goal_evaluations: List[Dict[str, Any]] = []
        safety_evaluations: List[Dict[str, Any]] = []

        for oracle in oracle_results:
            item = json_safe(oracle)
            if not isinstance(item, dict):
                continue
            details = dict(item.get('details') or {})
            oracle_name = str(item.get('oracle_name') or '')

            # --- Goal evaluations ---
            for goal_result in list(details.get('sub_goal_results') or []):
                if not isinstance(goal_result, dict):
                    continue
                goal_evaluations.append(cls._summarize_goal_result(goal_result, 'sub_goal'))
            for goal_result in list(details.get('final_goal_results') or []):
                if not isinstance(goal_result, dict):
                    continue
                goal_evaluations.append(cls._summarize_goal_result(goal_result, 'final_goal'))

            # --- Online unknown goals ---
            for unknown in list(details.get('unknown_goal_results') or []):
                if not isinstance(unknown, dict):
                    continue
                goal_evaluations.append({
                    'goal_id': unknown.get('goal_id'),
                    'type': 'sub_goal',
                    'description': unknown.get('description'),
                    'completed': None,
                    'truth': 'unknown',
                    'source': oracle_name,
                    'reason': (
                        dict(unknown.get('predicate_result') or {}).get('reason')
                        if isinstance(unknown.get('predicate_result'), dict)
                        else None
                    ),
                })

            # --- Safety assertion evaluations (LTL) ---
            # Deduplicate by assertion_id: only keep the most informative
            # (non-pending) verdict.  The offline oracle may store both
            # intermediate "pending" results and the final verdict for the
            # same assertion_id; we only want the final one in the summary.
            ltl_items = list(details.get('ltl_assertion_results') or [])
            best_ltl_by_id: Dict[str, Dict[str, Any]] = {}
            for ltl_result in ltl_items:
                if not isinstance(ltl_result, dict):
                    continue
                aid = str(ltl_result.get('assertion_id') or '')
                verdict_str = str(ltl_result.get('verdict') or '').strip().lower()
                # A non-pending verdict is always more informative than pending.
                # Among non-pending verdicts, keep the first one encountered
                # (the offline oracle stores results in chronological order).
                if aid not in best_ltl_by_id or (
                    best_ltl_by_id[aid].get('verdict', '').lower() == 'pending'
                    and verdict_str != 'pending'
                ):
                    best_ltl_by_id[aid] = ltl_result
            for aid, ltl_result in best_ltl_by_id.items():
                safety_evaluations.append({
                    'assertion_id': ltl_result.get('assertion_id'),
                    'type': 'ltl',
                    'verdict': ltl_result.get('verdict'),
                    'formula': cls._compact_string(
                        str(ltl_result.get('formula') or ''), key='formula',
                    ),
                    'decisive_step': ltl_result.get('decisive_step'),
                    'source': oracle_name,
                    'error': ltl_result.get('error'),
                })

            # --- Safety assertion evaluations (legacy / simple predicate) ---
            for legacy_result in list(details.get('legacy_assertion_results') or []):
                if not isinstance(legacy_result, dict):
                    continue
                predicate_result = dict(legacy_result.get('predicate_result') or {})
                truth = str(predicate_result.get('truth') or '').lower()
                if truth == 'true':
                    verdict = 'violated'  # legacy: passed=True means assertion triggered = unsafe
                elif truth == 'false':
                    verdict = 'satisfied'
                else:
                    verdict = 'inconclusive'
                safety_evaluations.append({
                    'assertion_id': legacy_result.get('assertion_id'),
                    'type': 'predicate',
                    'verdict': verdict,
                    'predicate': predicate_result.get('predicate'),
                    'source': oracle_name,
                    'reason': predicate_result.get('reason'),
                })

        return {
            'goals': goal_evaluations,
            'safety': safety_evaluations,
        }

    @classmethod
    def _summarize_goal_result(cls, goal_result: Dict[str, Any], goal_type: str) -> Dict[str, Any]:
        predicate_result = dict(goal_result.get('predicate_result') or {})
        truth = str(predicate_result.get('truth') or '').lower()
        if truth == 'true':
            truth_label = 'true'
        elif truth == 'false':
            truth_label = 'false'
        elif truth:
            truth_label = 'unknown'
        else:
            truth_label = ''
        return {
            'goal_id': goal_result.get('goal_id'),
            'type': goal_type,
            'description': goal_result.get('description'),
            'completed': goal_result.get('completed'),
            'truth': truth_label,
            'source': goal_result.get('source', ''),
            'reason': predicate_result.get('reason'),
            'predicate': predicate_result.get('predicate'),
        }

    @classmethod
    def _policy_execution_summary(cls, execution_states: List[Any]) -> Dict[str, Any]:
        tool_calls = []
        online_goal_events = []
        online_unsafe_events = []
        runtime_unsafe_events = []
        final_policy_metadata: Dict[str, Any] = {}
        for state in execution_states:
            item = json_safe(state)
            if not isinstance(item, dict):
                continue
            metadata = dict(item.get('execution_metadata') or {})
            if metadata:
                final_policy_metadata = metadata
            trace_record = metadata.get('trace_record')
            if trace_record:
                tool_calls.append(cls._tool_call_summary_from_record(trace_record, state=item))
            online_goal_events.extend(list(metadata.get('online_goal_events') or []))
            online_unsafe_events.extend(list(metadata.get('online_unsafe_events') or []))
            runtime_unsafe_events.extend(list(metadata.get('runtime_unsafe_events') or []))

        notable = [
            call for call in tool_calls
            if (
                call.get('ok') is False
                or call.get('reached') is False
                or 'timeout' in str(call.get('message') or '').lower()
                or call.get('object_in_gripper') is True
                or call.get('online_goal_events')
                or call.get('online_unsafe_events')
            )
        ]
        policy_result = {
            key: final_policy_metadata.get(key)
            for key in (
                'policy_name',
                'llm_provider',
                'llm_model',
                'policy_error',
                'failure_recovery',
                'llm_input',
                'llm_output',
            )
            if key in final_policy_metadata
        }
        return {
            'state_count': len(execution_states),
            'tool_call_count': len(tool_calls),
            'failed_or_timeout_call_count': len([
                call for call in tool_calls
                if (
                    call.get('ok') is False
                    or call.get('reached') is False
                    or 'timeout' in str(call.get('message') or '').lower()
                )
            ]),
            'online_goal_event_count': len(online_goal_events),
            'online_unsafe_event_count': len(online_unsafe_events),
            'runtime_unsafe_event_count': len(runtime_unsafe_events),
            'tool_calls': tool_calls,
            'notable_events': notable,
            'policy_result': cls._policy_result_highlight(policy_result),
        }

    @classmethod
    def _policy_result_highlight(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(value or {})
        if 'llm_input' in result:
            result['llm_input'] = cls._compact_string(result.get('llm_input'), key='llm_input')
        if 'llm_output' in result:
            result['llm_output'] = cls._compact_string(result.get('llm_output'), key='llm_output')
        if 'failure_recovery' in result:
            recovery = result.get('failure_recovery')
            if isinstance(recovery, dict):
                result['failure_recovery'] = {
                    'enabled': recovery.get('enabled'),
                    'attempt_count': recovery.get('attempt_count'),
                    'attempts': [
                        {
                            'attempt': item.get('attempt'),
                            'tool_call_count': item.get('tool_call_count'),
                            'recoverable_failure': item.get('recoverable_failure'),
                            'failure_reason': item.get('failure_reason'),
                            'failed_tool': item.get('failed_tool'),
                            'failed_command': item.get('failed_command'),
                        }
                        for item in list(recovery.get('attempts') or [])
                        if isinstance(item, dict)
                    ],
                }
            else:
                result['failure_recovery'] = cls._compact_tree(
                    recovery,
                    key='failure_recovery',
                )
        return cls._compact_tree(result, key='policy_result')

    @classmethod
    def _tool_call_summary_from_record(
        cls,
        record: Any,
        *,
        state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        item = json_safe(record)
        if not isinstance(item, dict):
            return {'record': cls._compact_tree(item, key='trace_record')}
        response = item.get('response') if isinstance(item.get('response'), dict) else {}
        payload = response.get('payload') if isinstance(response.get('payload'), dict) else {}
        args = item.get('args') if isinstance(item.get('args'), dict) else {}
        command_payload = (
            item.get('command_payload')
            if isinstance(item.get('command_payload'), dict)
            else {}
        )
        if not args and isinstance(command_payload.get('args'), dict):
            args = command_payload.get('args')

        state_payload = {}
        if isinstance(state, dict):
            state_payload = (
                state.get('runtime_payload')
                if isinstance(state.get('runtime_payload'), dict)
                else {}
            )
        metadata = (
            state.get('execution_metadata')
            if isinstance(state, dict) and isinstance(state.get('execution_metadata'), dict)
            else {}
        )
        command = item.get('command') or command_payload.get('command')
        tool_name = item.get('tool_name') or ''
        target_object = (
            payload.get('target_object')
            or args.get('target_object')
            or args.get('obj_name')
            or args.get('object_name')
            or args.get('target')
            or ''
        )
        return {
            'step_id': item.get('step_id') or (state or {}).get('step'),
            'tool_name': tool_name,
            'command': command,
            'ok': response.get('ok') if response else item.get('ok'),
            'message': (
                response.get('message')
                or response.get('error')
                if response
                else item.get('message')
            ),
            'target_object': target_object,
            'reached': payload.get('reached') if payload else item.get('reached'),
            'target_distance': payload.get('target_distance') if payload else item.get('target_distance'),
            'orientation_distance': (
                payload.get('orientation_distance')
                if payload
                else item.get('orientation_distance')
            ),
            'navigation_failure_reason': (
                payload.get('navigation_failure_reason')
                if payload
                else item.get('navigation_failure_reason')
            ),
            'object_in_gripper': (
                payload.get('object_in_gripper')
                if payload.get('object_in_gripper') is not None
                else (
                    item.get('object_in_gripper')
                    if item.get('object_in_gripper') is not None
                    else state_payload.get('object_in_gripper')
                )
            ),
            'grasped_object_name': (
                payload.get('grasped_object_name')
                or item.get('grasped_object_name')
                or state_payload.get('grasped_object_name')
                or ''
            ),
            'top_down_screenshot_path': item.get('top_down_screenshot_path'),
            'online_goal_events': cls._compact_tree(
                metadata.get('online_goal_events') or [],
                key='online_goal_events',
            ),
            'online_unsafe_events': cls._compact_tree(
                metadata.get('online_unsafe_events') or [],
                key='online_unsafe_events',
            ),
        }

    @classmethod
    def _final_state_summary(cls, value: Any) -> Dict[str, Any]:
        state = json_safe(value)
        if not isinstance(state, dict):
            return {}
        payload = (
            state.get('runtime_payload')
            if isinstance(state.get('runtime_payload'), dict)
            else {}
        )
        return {
            'step': state.get('step'),
            'current_room_name': payload.get('current_room_name'),
            'robot_pose': cls._compact_tree(payload.get('robot_pose'), key='robot_pose'),
            'base_pose': cls._compact_tree(payload.get('base_pose'), key='base_pose'),
            'gripper_state': payload.get('gripper_state'),
            'object_in_gripper': payload.get('object_in_gripper'),
            'grasped_object_name': payload.get('grasped_object_name'),
            'contact_event_count': len(list(payload.get('contact_events') or [])),
            'world_state': cls._compact_tree(payload.get('world_state'), key='world_state'),
        }

    @classmethod
    def _case_summary_markdown(cls, summary: Dict[str, Any]) -> str:
        result = dict(summary.get('result') or {})
        metrics = dict(summary.get('metrics') or {})
        planning = dict(summary.get('planning') or {})
        execution = dict(summary.get('agent_policy_execution') or {})
        oracles = list(summary.get('oracles') or [])
        final_state = dict(summary.get('final_state') or {})
        spec_evaluations = dict(summary.get('spec_evaluations') or {})

        lines = [
            f"# Case Summary: {summary.get('scenario_id')}",
            "",
            "## Result",
            f"- Status: `{summary.get('status')}`",
            f"- Verdict: `{result.get('verdict')}`",
            f"- Final score: `{result.get('final_score')}`",
            f"- Safety score: `{result.get('safety_score')}`",
            f"- Task score: `{result.get('task_score')}`",
            f"- Unsafe: `{result.get('unsafe')}`",
            f"- Task completed: `{result.get('task_completed')}`",
            f"- Summary: {result.get('summary') or ''}",
            "",
            "## Metrics",
            "| Metric | Status | Score | Numerator/Denominator | Key notes |",
            "|---|---:|---:|---:|---|",
        ]
        for name, metric in metrics.items():
            notes = []
            if metric.get('missing_goals'):
                notes.append(f"missing={metric.get('missing_goals')}")
            if metric.get('recognized_hazards'):
                notes.append(f"hazards={metric.get('recognized_hazards')}")
            if metric.get('unsafe_event_count') is not None:
                notes.append(f"unsafe_events={metric.get('unsafe_event_count')}")
            if metric.get('skip_reason'):
                notes.append(f"skip={metric.get('skip_reason')}")
            lines.append(
                f"| `{name}` | `{metric.get('status')}` | `{metric.get('score')}` | "
                f"`{metric.get('numerator')}/{metric.get('denominator')}` | "
                f"{'; '.join(notes)} |"
            )

        lines.extend([
            "",
            "## Agent Policy Execution",
            f"- Tool calls: `{execution.get('tool_call_count')}`",
            f"- Failed/timeout calls: `{execution.get('failed_or_timeout_call_count')}`",
            f"- Online goal events: `{execution.get('online_goal_event_count')}`",
            f"- Online unsafe events: `{execution.get('online_unsafe_event_count')}`",
            "",
            "| Step | Tool | Command | OK | Reached | Object | Screenshot | Message |",
            "|---:|---|---|---:|---:|---|---|---|",
        ])
        for call in list(execution.get('tool_calls') or []):
            screenshot_path = call.get('top_down_screenshot_path')
            if screenshot_path:
                # summary.md lives next to the images/ folder, so relative links resolve.
                screenshot_cell = f"[image]({str(screenshot_path).replace(' ', '%20')})"
            else:
                screenshot_cell = ''
            lines.append(
                f"| {call.get('step_id')} | `{call.get('tool_name')}` | "
                f"`{call.get('command')}` | `{call.get('ok')}` | "
                f"`{call.get('reached')}` | `{call.get('target_object') or call.get('grasped_object_name') or ''}` | "
                f"{screenshot_cell} | "
                f"{str(call.get('message') or '').replace('|', '/')} |"
            )

        policy_result = dict(execution.get('policy_result') or {})
        if policy_result.get('llm_output'):
            lines.extend([
                "",
                "### Final policy message",
                "```text",
                str(policy_result.get('llm_output')),
                "```",
            ])

        lines.extend([
            "",
            "## Planning",
        ])
        for plan in list(planning.get('plans') or []):
            lines.extend([
                f"- Plan {plan.get('plan_index')} agent=`{plan.get('agent_name')}` model=`{plan.get('model')}` refused=`{plan.get('refused')}`",
                "```text",
                str(plan.get('raw_output') or ''),
                "```",
            ])
            if plan.get('reasoning_available'):
                lines.extend([
                    "Reasoning (backfilled when marked in summary metadata):",
                    "```text",
                    str(plan.get('reasoning') or ''),
                    "```",
                ])

        lines.extend([
            "",
            "## Oracles",
            "| Oracle | Verdict | Score | Reason |",
            "|---|---:|---:|---|",
        ])
        for oracle in oracles:
            lines.append(
                f"| `{oracle.get('oracle_name')}` | `{oracle.get('verdict')}` | "
                f"`{oracle.get('score')}` | {str(oracle.get('reason') or '').replace('|', '/')} |"
            )

        # --- Spec Evaluation: per-goal and per-safety-assertion verdicts ---
        goal_evals = list(spec_evaluations.get('goals') or [])
        safety_evals = list(spec_evaluations.get('safety') or [])

        if goal_evals:
            lines.extend([
                "",
                "## Goal Spec Evaluation",
                "| Goal ID | Type | Description | Completed | Truth | Source | Reason |",
                "|---|---|---|---:|---:|---|---|",
            ])
            for goal in goal_evals:
                description = str(goal.get('description') or '').replace('|', '/')
                reason = str(goal.get('reason') or '').replace('|', '/')
                lines.append(
                    f"| `{goal.get('goal_id')}` | {goal.get('type')} | "
                    f"{description} | `{goal.get('completed')}` | "
                    f"`{goal.get('truth')}` | `{goal.get('source')}` | {reason} |"
                )

        if safety_evals:
            lines.extend([
                "",
                "## Safety Assertion Evaluation",
                "| Assertion ID | Type | Verdict | Reason / Detail |",
                "|---|---|---:|---|",
            ])
            for safety in safety_evals:
                reason_or_detail = ''
                if safety.get('formula'):
                    reason_or_detail = str(safety.get('formula') or '').replace('|', '/')
                elif safety.get('reason'):
                    reason_or_detail = str(safety.get('reason') or '').replace('|', '/')
                elif safety.get('error'):
                    reason_or_detail = str(safety.get('error') or '').replace('|', '/')
                elif safety.get('predicate'):
                    reason_or_detail = str(safety.get('predicate') or '').replace('|', '/')
                lines.append(
                    f"| `{safety.get('assertion_id')}` | {safety.get('type')} | "
                    f"`{safety.get('verdict')}` | {reason_or_detail} |"
                )

        lines.extend([
            "",
            "## Final State",
            f"- Room: `{final_state.get('current_room_name')}`",
            f"- Gripper: `{final_state.get('gripper_state')}`",
            f"- Object in gripper: `{final_state.get('object_in_gripper')}`",
            f"- Grasped object: `{final_state.get('grasped_object_name')}`",
            "",
        ])
        return '\n'.join(lines)

    @classmethod
    def _compact_case_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a human-sized case artifact without bulky runtime internals."""

        data = dict(payload or {})
        data['artifact_detail'] = 'compact'
        data['artifact_note'] = (
            'Large coordinate arrays, point clouds, collision-filter pair lists, '
            'and repeated before/after runtime states are summarized or omitted. '
            'Set OMNISAFE_COMPACT_CASE_ARTIFACTS=0 to write full artifacts.'
        )

        data['planning_results'] = [
            cls._compact_planning_result(item)
            for item in list(data.get('planning_results') or [])
        ]
        data['execution_states'] = [
            cls._compact_execution_state(item)
            for item in list(data.get('execution_states') or [])
        ]
        if data.get('initial_state') is not None:
            data['initial_state'] = cls._compact_execution_state(data.get('initial_state'))
        if data.get('final_state') is not None:
            data['final_state'] = cls._compact_execution_state(data.get('final_state'))
        data['oracle_results'] = cls._compact_tree(data.get('oracle_results'), key='oracle_results')
        data['metric_results'] = cls._compact_tree(data.get('metric_results'), key='metric_results')
        data['aggregate'] = cls._compact_tree(data.get('aggregate'), key='aggregate')
        data['errors'] = cls._compact_tree(data.get('errors'), key='errors')
        data['metadata'] = cls._compact_tree(data.get('metadata'), key='metadata')
        data['diagnostic_log_count'] = len(list(data.get('diagnostic_log') or []))
        data['diagnostic_log'] = [
            cls._compact_case_diagnostic_entry(item)
            for item in list(data.get('diagnostic_log') or [])
        ]
        return data

    @classmethod
    def _compact_planning_result(cls, value: Any) -> Any:
        item = json_safe(value)
        if not isinstance(item, dict):
            return cls._compact_tree(item, key='planning_result')
        result = dict(item)
        metadata = dict(result.get('metadata') or {})
        if 'llm_trace' in metadata:
            metadata['llm_trace'] = [
                cls._compact_llm_trace_entry(entry)
                for entry in list(metadata.get('llm_trace') or [])
            ]
        if 'assembled_program' in metadata:
            metadata['assembled_program'] = cls._compact_string(
                metadata.get('assembled_program'),
                key='assembled_program',
            )
        if 'cap_context' in metadata:
            metadata['cap_context'] = cls._compact_string(
                metadata.get('cap_context'),
                key='cap_context',
            )
        result['metadata'] = cls._compact_tree(metadata, key='metadata')
        result['raw_output'] = cls._compact_string(result.get('raw_output'), key='raw_output')
        result['reasoning'] = cls._compact_string(result.get('reasoning'), key='reasoning')
        return cls._compact_tree(result, key='planning_result')

    @classmethod
    def _compact_llm_trace_entry(cls, value: Any) -> Any:
        entry = json_safe(value)
        if not isinstance(entry, dict):
            return cls._compact_tree(entry, key='llm_trace')
        result = {
            key: cls._compact_tree(entry.get(key), key=key)
            for key in (
                'model',
                'prompt_chars',
                'prompt_tail',
                'content',
                'reasoning_content',
                'reasoning_field_source',
                'refusal',
                'finish_reason',
                'usage',
                'reasoning_tokens',
                'label',
                'attempt',
            )
            if key in entry
        }
        if 'message_dump' in entry:
            result['message_dump_summary'] = cls._message_dump_summary(entry.get('message_dump'))
        if 'usage_dump' in entry:
            result['usage_dump'] = cls._compact_tree(entry.get('usage_dump'), key='usage_dump')
        return result

    @classmethod
    def _message_dump_summary(cls, value: Any) -> Dict[str, Any]:
        message = json_safe(value)
        if not isinstance(message, dict):
            return {'content': cls._compact_string(message, key='message_dump')}
        reasoning_items = list(message.get('reasoning_items') or [])
        return {
            'role': message.get('role'),
            'refusal': message.get('refusal'),
            'content': cls._compact_string(message.get('content'), key='content'),
            'reasoning_item_count': len(reasoning_items),
            'tool_call_count': len(list(message.get('tool_calls') or [])),
            'has_audio': bool(message.get('audio')),
            'has_function_call': bool(message.get('function_call')),
            'encrypted_reasoning_omitted': any(
                isinstance(item, dict) and item.get('encrypted_content')
                for item in reasoning_items
            ),
        }

    @classmethod
    def _compact_diagnostic_entry(cls, value: Any) -> Dict[str, Any]:
        entry = json_safe(value)
        if not isinstance(entry, dict):
            return {'data': cls._compact_tree(entry, key='diagnostic_entry')}
        result = {
            key: entry.get(key)
            for key in (
                'sequence',
                'timestamp',
                'elapsed_sec',
                'level',
                'phase',
                'event',
                'message',
                'scenario_id',
                'attempt',
            )
            if key in entry
        }
        if 'data' in entry:
            result['data'] = cls._compact_tree(entry.get('data'), key='data')
        return result

    @classmethod
    def _compact_case_diagnostic_entry(cls, value: Any) -> Dict[str, Any]:
        """Very small timeline entry for the per-case JSON artifact.

        The full diagnostic payload is intentionally not repeated inside
        res_cases/*.json because the same information can include entire
        runtime states or tool traces. Streaming logs get a compact payload via
        _compact_diagnostic_entry; case artifacts only need a navigable
        timeline plus minimal error context.
        """

        entry = json_safe(value)
        if not isinstance(entry, dict):
            return {'message': cls._compact_string(entry, key='diagnostic_entry')}
        result = {
            key: entry.get(key)
            for key in (
                'sequence',
                'timestamp',
                'elapsed_sec',
                'level',
                'phase',
                'event',
                'message',
            )
            if key in entry
        }
        if str(entry.get('level') or '').lower() in {'warning', 'error', 'critical'} and 'data' in entry:
            result['data'] = cls._compact_tree(entry.get('data'), key='data')
        elif 'data' in entry:
            result['data_omitted'] = True
        return result

    @classmethod
    def _compact_execution_state(cls, value: Any) -> Any:
        state = json_safe(value)
        if not isinstance(state, dict):
            return cls._compact_tree(state, key='execution_state')

        result = {
            'scenario_id': state.get('scenario_id'),
            'step': state.get('step'),
        }
        if state.get('runtime_payload') is not None:
            result['runtime_payload'] = cls._compact_runtime_payload(
                state.get('runtime_payload')
            )
        if state.get('collision_flags') not in (None, [], {}):
            result['collision_flags'] = cls._compact_tree(
                state.get('collision_flags'),
                key='collision_flags',
            )
        if state.get('execution_metadata') is not None:
            result['execution_metadata'] = cls._compact_execution_metadata(
                state.get('execution_metadata')
            )
        return result

    @classmethod
    def _compact_execution_metadata(cls, value: Any) -> Dict[str, Any]:
        metadata = json_safe(value)
        if not isinstance(metadata, dict):
            return {'metadata': cls._compact_tree(metadata, key='execution_metadata')}

        result: Dict[str, Any] = {}
        for key in (
            'tool_name',
            'policy_name',
            'llm_provider',
            'llm_model',
            'policy_error',
            'failure_recovery',
        ):
            if key in metadata:
                result[key] = cls._compact_tree(metadata.get(key), key=key)
        online_oracle_results = list(metadata.get('online_oracle_results') or [])
        if online_oracle_results:
            result['online_oracle_result_count'] = len(online_oracle_results)
            result['online_oracle_results_omitted'] = True

        for key in (
            'online_goal_events',
            'online_unsafe_events',
            'runtime_unsafe_events',
        ):
            values = list(metadata.get(key) or [])
            if values:
                result[key] = cls._compact_tree(values, key=key)

        if metadata.get('trace_record') is not None:
            result['trace_record'] = cls._compact_trace_record(metadata.get('trace_record'))
        if metadata.get('tool_trace') is not None:
            tool_trace = list(metadata.get('tool_trace') or [])
            result['tool_trace_count'] = len(tool_trace)
            result['tool_trace_omitted'] = bool(tool_trace)
        if metadata.get('agent_event_log') is not None:
            result['agent_event_count'] = len(list(metadata.get('agent_event_log') or []))
        if metadata.get('agent_tool_call_log') is not None:
            result['agent_tool_event_count'] = len(list(metadata.get('agent_tool_call_log') or []))
        if metadata.get('intermediate_steps') is not None:
            result['intermediate_step_count'] = len(list(metadata.get('intermediate_steps') or []))
        if metadata.get('llm_input') is not None:
            result['llm_input'] = cls._compact_string(metadata.get('llm_input'), key='llm_input')
        if metadata.get('llm_output') is not None:
            result['llm_output'] = cls._compact_string(metadata.get('llm_output'), key='llm_output')
        return result

    @classmethod
    def _compact_trace_record(cls, value: Any) -> Dict[str, Any]:
        record = json_safe(value)
        if not isinstance(record, dict):
            return {'record': cls._compact_tree(record, key='trace_record')}

        result: Dict[str, Any] = {}
        for key in (
            'step_id',
            'tool_name',
            'command',
            'command_payload',
            'args',
            'timestamp',
            'policy_metadata',
            'runtime_unsafe_events',
            'top_down_screenshot_path',
            'top_down_screenshot_error',
        ):
            if key in record:
                result[key] = cls._compact_tree(record.get(key), key=key)
        if 'response' in record:
            result['response'] = cls._compact_command_response(record.get('response'))
        if 'before_state' in record:
            result['before_state_omitted'] = True
        if 'after_state' in record:
            result['after_state_omitted'] = True
        return result

    @classmethod
    def _compact_command_response(cls, value: Any) -> Any:
        response = json_safe(value)
        if not isinstance(response, dict):
            return cls._compact_tree(response, key='response')
        result = {
            key: cls._compact_tree(response.get(key), key=key)
            for key in ('ok', 'command', 'error', 'message')
            if key in response
        }
        if 'payload' in response:
            payload = response.get('payload')
            if cls._looks_like_runtime_payload(payload):
                result['payload'] = cls._compact_runtime_payload(payload)
            else:
                result['payload'] = cls._compact_tree(payload, key='payload')
        return result

    @classmethod
    def _compact_runtime_payload(cls, value: Any) -> Any:
        payload = json_safe(value)
        if not isinstance(payload, dict):
            return cls._compact_tree(payload, key='runtime_payload')

        keep_keys = (
            'sim_time_s',
            'robot_name',
            'requested_room_name',
            'spawn_room_name',
            'current_room_name',
            'spawn_pose',
            'robot_pose',
            'base_pose',
            'torso_height',
            'gripper_state',
            'object_in_gripper',
            'grasped_object_name',
            'grasped_object_mass',
            'pending_grasp_pose_latch',
            'grasp_pose_latch_enabled',
            'grasp_pose_latch_object_name',
            'command_counter',
            'last_command',
            'robot_loaded',
            'robot_prim_path',
            'robot_articulation_path',
            'runtime_safety_monitor',
            'runtime_observation_monitor',
            'world_state',
            'entity_state_events',
            'contact_events',
            'navigation_robot_radius',
            'navigation_arm_reach_radius',
            'navigation_dynamic_radius_enabled',
            'navigation_state',
            'navigation_failure_reason',
            'navigation_replan_requested',
            'navigation_target_active',
            'navigation_replan_attempts',
            'navigation_waypoint_count',
            'navigation_active_waypoint_index',
            'navigation_remaining_waypoints',
            'navigation_collision_stall_steps',
        )
        result = {
            key: cls._compact_tree(payload.get(key), key=key)
            for key in keep_keys
            if key in payload
        }
        if 'joint_states' in payload:
            result['joint_states'] = cls._compact_joint_states(payload.get('joint_states'))
        if 'room_index' in payload:
            result['room_index_summary'] = cls._summarize_room_index(payload.get('room_index'))
        if 'entities' in payload:
            result['entities_summary'] = cls._summarize_named_collection(payload.get('entities'))
        for key in (
            'articulated_objects',
            'articulations',
            'persistent_arm_collision_filtering',
            'aggressive_lifecycle_collision_filtering',
        ):
            if key in payload:
                result[key] = cls._compact_tree(payload.get(key), key=key)
        return result

    @staticmethod
    def _looks_like_runtime_payload(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and (
                'robot_loaded' in value
                or 'room_index' in value
                or 'command_counter' in value
            )
        )

    @classmethod
    def _compact_joint_states(cls, value: Any) -> Any:
        states = json_safe(value)
        if not isinstance(states, dict):
            return cls._compact_tree(states, key='joint_states')
        joint_positions = list(states.get('joint_positions') or [])
        joint_names = list(states.get('joint_names') or [])
        result: Dict[str, Any] = {
            'joint_count': len(joint_positions) or len(joint_names),
        }
        if joint_names:
            result['joint_names'] = joint_names
        if states.get('gripper_joint_positions') is not None:
            result['gripper_joint_positions'] = cls._compact_tree(
                states.get('gripper_joint_positions'),
                key='gripper_joint_positions',
            )
        return result

    @classmethod
    def _summarize_room_index(cls, value: Any) -> Dict[str, Any]:
        room_index = json_safe(value)
        if not isinstance(room_index, dict):
            return {'detail': cls._compact_tree(room_index, key='room_index')}

        rooms = []
        object_count = 0
        for room_name, room in room_index.items():
            room_payload = dict(room or {}) if isinstance(room, dict) else {}
            objects = list(room_payload.get('objects') or [])
            object_names = [
                str(item.get('name') or '')
                for item in objects
                if isinstance(item, dict) and str(item.get('name') or '')
            ]
            object_count += len(objects)
            rooms.append({
                'room_name': str(room_payload.get('room_name') or room_name),
                'prim_path': room_payload.get('prim_path'),
                'object_count': len(objects),
                'object_names': object_names,
            })
        return {
            'room_count': len(rooms),
            'object_count': object_count,
            'rooms': rooms,
            'coordinate_detail_omitted': True,
        }

    @classmethod
    def _summarize_named_collection(cls, value: Any) -> Dict[str, Any]:
        collection = json_safe(value)
        if isinstance(collection, dict):
            names = [str(key) for key in collection.keys()]
            return {'count': len(collection), 'names': names}
        if isinstance(collection, list):
            names = [
                str(item.get('name') or item.get('entity') or item.get('id') or '')
                for item in collection
                if isinstance(item, dict)
            ]
            names = [name for name in names if name]
            return {'count': len(collection), 'names': names}
        return {'detail': cls._compact_tree(collection, key='entities')}

    @classmethod
    def _compact_collision_filtering(cls, value: Any) -> Any:
        payload = json_safe(value)
        if not isinstance(payload, dict):
            return cls._summarize_list(payload, key='collision_filtering')
        result = {
            key: cls._compact_tree(payload.get(key), key=key)
            for key in payload.keys()
            if key != 'pairs'
        }
        pairs = list(payload.get('pairs') or [])
        result['pair_count'] = len(pairs)
        result['pairs_omitted'] = bool(pairs)
        return result

    @classmethod
    def _summarize_point_cloud(cls, value: Any) -> Dict[str, Any]:
        payload = json_safe(value)
        if isinstance(payload, dict):
            points = payload.get('points')
            point_count = len(points) if isinstance(points, list) else payload.get('point_count')
            result = {
                'point_count': int(point_count or 0),
                'points_omitted': points is not None,
            }
            for key in ('mesh_count', 'mesh_prim_paths', 'source', 'frame'):
                if key in payload:
                    result[key] = cls._compact_tree(payload.get(key), key=key)
            return result
        if isinstance(payload, list):
            return {'point_count': len(payload), 'points_omitted': True}
        return {'detail': cls._compact_tree(payload, key='point_cloud')}

    @classmethod
    def _summarize_list(cls, value: Any, *, key: str) -> Any:
        items = list(value or []) if isinstance(value, (list, tuple)) else []
        return {
            'count': len(items),
            'items_omitted': bool(items),
            'field': key,
        }

    @classmethod
    def _compact_tree(cls, value: Any, *, key: str = '', depth: int = 0) -> Any:
        normalized_key = str(key or '').lower()
        safe_value = json_safe(value)

        if normalized_key == 'encrypted_content':
            return {'omitted': True, 'reason': 'encrypted_reasoning_blob'}
        if normalized_key == 'point_cloud':
            return cls._summarize_point_cloud(safe_value)
        if normalized_key == 'room_index':
            return cls._summarize_room_index(safe_value)
        if normalized_key == 'entities':
            return cls._summarize_named_collection(safe_value)
        if normalized_key in {
            'persistent_arm_collision_filtering',
            'aggressive_lifecycle_collision_filtering',
        }:
            return cls._compact_collision_filtering(safe_value)
        if (
            normalized_key in {
                'pairs',
                'collision_filter_pairs',
                'target_collision_filter_pairs',
                'filtered_pairs',
            }
            or normalized_key.endswith('_pairs')
        ):
            if isinstance(safe_value, (list, tuple)):
                return {
                    'count': len(safe_value),
                    'items_omitted': True,
                    'field': key,
                }

        if isinstance(safe_value, dict):
            if depth > 12:
                return {'omitted': True, 'reason': 'max_depth'}
            return {
                str(child_key): cls._compact_tree(
                    child_value,
                    key=str(child_key),
                    depth=depth + 1,
                )
                for child_key, child_value in safe_value.items()
                if str(child_key).lower() not in {
                    'control_module_path',
                    'controller_class_name',
                }
            }
        if isinstance(safe_value, list):
            if len(safe_value) > cls.MAX_LOG_LIST_ITEMS:
                return {
                    'count': len(safe_value),
                    'items': [
                        cls._compact_tree(item, key=key, depth=depth + 1)
                        for item in safe_value[: cls.MAX_LOG_LIST_ITEMS]
                    ],
                    'omitted_count': len(safe_value) - cls.MAX_LOG_LIST_ITEMS,
                }
            return [
                cls._compact_tree(item, key=key, depth=depth + 1)
                for item in safe_value
            ]
        if isinstance(safe_value, str):
            return cls._compact_string(safe_value, key=key)
        return safe_value

    @classmethod
    def _compact_string(cls, value: Any, *, key: str) -> Any:
        if not isinstance(value, str):
            return value
        if len(value) <= cls.MAX_LOG_STRING_CHARS:
            return value
        head = value[: cls.MAX_LOG_STRING_CHARS]
        return (
            f'{head}\n'
            f'... <truncated {len(value) - cls.MAX_LOG_STRING_CHARS} chars from {key}>'
        )

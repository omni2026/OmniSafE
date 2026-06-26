"""Isaac Sim runtime entry point for the LangGraph scene generator.

This module reuses the current main runtime and only adds the evaluator
commands required by ``scene_generator_graph.py``.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from isaac_sim_app import IsaacSimAppRunner, args, simulation_app
from Tools.tool_implementation_evaluate import (
    analyze_placement_log,
    capture_evaluation_snapshot,
    capture_evaluation_views,
    check_scene_collisions,
    reset_trial_placements,
    stage_trial_resolved_pose,
    summarize_room_layout,
)


def _payload_from_command(command: str) -> Dict[str, Any]:
    payload_text = command.split(",", 1)[1] if "," in command else "{}"
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("command payload must be a JSON object")
    return payload


class GraphIsaacSimAppRunner(IsaacSimAppRunner):
    """Extend the main runtime with graph evaluator command handlers."""

    def _send_json(self, payload: Dict[str, Any]) -> None:
        self.output_queue.put(json.dumps(payload, ensure_ascii=False, default=str))

    def handle_command(self, cmd):
        try:
            if cmd.startswith("evaluate_analyze_placement_log"):
                payload = _payload_from_command(cmd)
                result = analyze_placement_log(
                    payload.get("trace_file"),
                    payload.get("room_name") or self.current_room_name,
                    payload.get("include_structure", False),
                )
                self._send_json(result)
                return

            if cmd.startswith("evaluate_check_scene_collisions"):
                parts = cmd.split(",", 1)
                room_name = (
                    parts[1].strip()
                    if len(parts) >= 2 and parts[1].strip()
                    else self.current_room_name
                )
                self._send_json(check_scene_collisions(room_name))
                return

            if cmd.startswith("evaluate_summarize_room_layout"):
                parts = cmd.split(",", 1)
                room_name = (
                    parts[1].strip()
                    if len(parts) >= 2 and parts[1].strip()
                    else self.current_room_name
                )
                self._send_json(summarize_room_layout(room_name))
                return

            if cmd.startswith("evaluate_capture_snapshot"):
                payload = _payload_from_command(cmd)
                result = capture_evaluation_snapshot(
                    payload.get("output_dir"),
                    payload.get("room_name") or self.current_room_name,
                    payload.get("object_name"),
                )
                self._send_json(result)
                return

            if cmd.startswith("evaluate_capture_views"):
                payload = _payload_from_command(cmd)
                result = capture_evaluation_views(
                    payload.get("room_name") or self.current_room_name,
                    payload.get("output_dir"),
                    payload.get("suspect_objects") or [],
                )
                self._send_json(result)
                return

            if cmd.startswith("evaluate_stage_trial_pose"):
                payload = _payload_from_command(cmd)
                result = stage_trial_resolved_pose(
                    payload.get("room_name") or self.current_room_name,
                    object_name=payload.get("object_name"),
                    position=payload.get("position"),
                    rotation=payload.get("rotation"),
                    prim_path=payload.get("prim_path"),
                )
                self._send_json(result)
                return

            if cmd.startswith("evaluate_reset_trial_placements"):
                parts = cmd.split(",", 1)
                room_name = (
                    parts[1].strip()
                    if len(parts) >= 2 and parts[1].strip()
                    else self.current_room_name
                )
                self._send_json(reset_trial_placements(room_name))
                return
        except Exception as exc:
            self._send_json(
                {
                    "success": False,
                    "message": f"{exc.__class__.__name__}: {exc}",
                }
            )
            return

        super().handle_command(cmd)


def main() -> None:
    runner = GraphIsaacSimAppRunner(sim_app=simulation_app, parsed_args=args)
    runner.run()


if __name__ == "__main__":
    main()

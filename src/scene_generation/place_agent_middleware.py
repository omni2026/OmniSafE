from __future__ import annotations

import ast
import asyncio
import json
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# ---------------------------------------------------------------------------
# SERIALISE_TOOL_CALLS – set to True to force tool calls to execute one at a
# time; False (default) allows LangGraph's default parallel execution.
#
# LangGraph's ToolNode executes multiple tool_calls from the same AIMessage
# in parallel (executor.map / asyncio.gather).  For placement this can cause
# race conditions because resolve_placement_intent and set_pose depend on a
# consistent view of the Isaac Sim scene state.  Without serialisation,
# concurrent resolves may see objects still at the origin (or at stale
# positions), leading to incorrect placement decisions (e.g. countertop
# placed on the floor instead of on top of commercial_kitchen_table).
# ---------------------------------------------------------------------------
SERIALISE_TOOL_CALLS: bool = True  # ← flip to True for serial execution

_tool_execution_lock_sync = threading.Lock()
_tool_execution_lock_async = asyncio.Lock()


_FOLD_SOURCE = "asset_placement_folding"


@dataclass(frozen=True)
class _ToolRecord:
    ai_index: int
    call_index: int
    call_id: str
    name: str
    args: dict[str, Any]
    result_index: int | None
    result: ToolMessage | None
    asset_name: str | None

    @property
    def order(self) -> tuple[int, int]:
        return self.ai_index, self.call_index


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _message_content(message: AnyMessage) -> Any:
    return getattr(message, "content", "")


def _parsed_tool_result(message: ToolMessage | None) -> Any:
    if message is None:
        return None

    content = _message_content(message)
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("text")
        ]
        content = "\n".join(text_parts)

    if not isinstance(content, str):
        return content

    stripped = content.strip()
    if not stripped:
        return ""

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return stripped


def _tool_result_succeeded(message: ToolMessage | None) -> bool:
    if message is None or getattr(message, "status", "success") == "error":
        return False

    value = _parsed_tool_result(message)
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if "success" in value:
            return value.get("success") is True
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "ok", "success", "succeeded"}
    return False


def _asset_name_for_call(name: str, args: dict[str, Any]) -> str | None:
    if name == "resolve_placement_intent":
        intent = _as_dict(args.get("intent"))
        object_name = intent.get("object_name")
    else:
        object_name = args.get("object_name") or args.get("asset_name")

    if object_name is None:
        return None
    normalized = str(object_name).strip()
    return normalized or None


def _tool_call_parts(call: Any) -> tuple[str, dict[str, Any], str]:
    if isinstance(call, dict):
        name = call.get("name") or (call.get("function") or {}).get("name") or ""
        raw_args = call.get("args")
        if raw_args is None:
            raw_args = (call.get("function") or {}).get("arguments")
        call_id = call.get("id") or ""
    else:
        name = getattr(call, "name", "")
        raw_args = getattr(call, "args", {})
        call_id = getattr(call, "id", "")
    return str(name), _as_dict(raw_args), str(call_id)


def _collect_tool_records(messages: list[AnyMessage]) -> list[_ToolRecord]:
    results_by_call_id: dict[str, tuple[int, ToolMessage]] = {}
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage) and message.tool_call_id:
            results_by_call_id[message.tool_call_id] = (index, message)

    records: list[_ToolRecord] = []
    for ai_index, message in enumerate(messages):
        if not isinstance(message, AIMessage):
            continue
        for call_index, call in enumerate(message.tool_calls or []):
            name, args, call_id = _tool_call_parts(call)
            result_index, result = results_by_call_id.get(call_id, (None, None))
            records.append(
                _ToolRecord(
                    ai_index=ai_index,
                    call_index=call_index,
                    call_id=call_id,
                    name=name,
                    args=args,
                    result_index=result_index,
                    result=result,
                    asset_name=_asset_name_for_call(name, args),
                )
            )
    return records


def _existing_summary(message: AnyMessage) -> dict[str, Any] | None:
    if not isinstance(message, HumanMessage):
        return None
    metadata = message.additional_kwargs
    if metadata.get("lc_source") != _FOLD_SOURCE:
        return None
    summary = metadata.get("placement_summary")
    return summary if isinstance(summary, dict) else None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)
    return value


def _build_summary_message(summary: dict[str, Any]) -> HumanMessage:
    compact_summary = json.dumps(
        _json_safe(summary),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return HumanMessage(
        content=(
            "Completed asset placement (authoritative compact history; do not repeat "
            f"completed calls unless correction is required): {compact_summary}"
        ),
        id=f"asset-placement-summary-{uuid.uuid4()}",
        additional_kwargs={
            "lc_source": _FOLD_SOURCE,
            "asset_name": summary["asset_name"],
            "placement_summary": summary,
        },
    )


def _meaningful_content(message: AIMessage) -> bool:
    content = message.content
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)


def _copy_ai_message_without_calls(
    message: AIMessage,
    remaining_calls: list[dict[str, Any]],
) -> AIMessage:
    additional_kwargs = dict(message.additional_kwargs)
    additional_kwargs.pop("tool_calls", None)
    return message.model_copy(
        update={
            "tool_calls": remaining_calls,
            "invalid_tool_calls": [],
            "additional_kwargs": additional_kwargs,
        }
    )


def fold_completed_asset_messages(
    messages: list[AnyMessage],
) -> list[AnyMessage] | None:
    records = _collect_tool_records(messages)
    if not records:
        return None

    existing_by_asset: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, message in enumerate(messages):
        summary = _existing_summary(message)
        if summary and summary.get("asset_name"):
            existing_by_asset[str(summary["asset_name"])] = (index, summary)

    successful_set_pose: dict[str, _ToolRecord] = {}
    for record in records:
        if (
            record.name == "set_pose"
            and record.asset_name
            and _tool_result_succeeded(record.result)
        ):
            successful_set_pose[record.asset_name] = record

    fold_data: dict[str, dict[str, Any]] = {}
    for asset_name, set_pose_record in successful_set_pose.items():
        final_order = set_pose_record.order
        asset_records = [
            record
            for record in records
            if record.asset_name == asset_name and record.order <= final_order
        ]
        successful_resolves = [
            record
            for record in asset_records
            if record.name == "resolve_placement_intent"
            and _tool_result_succeeded(record.result)
        ]
        set_pose_result = _parsed_tool_result(set_pose_record.result)
        direct_pose_without_resolve = False
        if not successful_resolves:
            # Ablation variants may intentionally remove resolve_placement_intent
            # and let set_pose return a structured collision-feedback payload.
            # Only fold the trace once the final direct pose is collision-free;
            # otherwise keep the feedback visible so the model can repair it.
            direct_pose_without_resolve = (
                isinstance(set_pose_result, dict)
                and set_pose_result.get("success") is True
                and set_pose_result.get("collision_free") is True
                and (set_pose_result.get("room_bounds") or {}).get("inside") is not False
            )
            if not direct_pose_without_resolve:
                continue

        resolve_record = successful_resolves[-1] if successful_resolves else None
        scale_records = [
            record for record in asset_records if record.name == "scale_object"
        ]
        scale_record = scale_records[-1] if scale_records else None

        prior_entry = existing_by_asset.get(asset_name)
        prior_summary = prior_entry[1] if prior_entry else {}
        resize_summary = prior_summary.get("resize")
        if scale_record is not None:
            scale_result = _parsed_tool_result(scale_record.result)
            resize_summary = {
                "scale_factor": scale_record.args.get("scale_factor"),
                "success": _tool_result_succeeded(scale_record.result),
                "original_bbox": (
                    scale_result.get("original_bbox")
                    if isinstance(scale_result, dict)
                    else None
                ),
                "before_bbox": (
                    scale_result.get("before_bbox")
                    if isinstance(scale_result, dict)
                    else None
                ),
                "after_bbox": (
                    scale_result.get("after_bbox")
                    if isinstance(scale_result, dict)
                    else None
                ),
            }

        summary: dict[str, Any] = {
            "asset_name": asset_name,
            "set_pose": {
                "position": set_pose_record.args.get("position"),
                "rotation": set_pose_record.args.get("rotation"),
                "success": True,
            },
        }
        if resolve_record is not None:
            summary["resolved_placement"] = {
                "intent": resolve_record.args.get("intent", {}),
                "result": _parsed_tool_result(resolve_record.result),
            }
        elif direct_pose_without_resolve:
            summary["direct_pose_without_resolve"] = {
                "result": set_pose_result,
                "note": "Ablation mode: set_pose used direct LLM-selected coordinates.",
            }
        if resize_summary is not None:
            summary["resize"] = resize_summary

        first_asset_order = min(record.order for record in asset_records)
        related_records = list(asset_records)
        related_records.extend(
            record
            for record in records
            if record.asset_name is None
            and first_asset_order <= record.order <= final_order
        )

        insert_index = min(record.ai_index for record in related_records)
        if prior_entry:
            insert_index = min(insert_index, prior_entry[0])

        fold_data[asset_name] = {
            "summary": summary,
            "records": related_records,
            "insert_index": insert_index,
        }

    if not fold_data:
        return None

    removed_call_ids: set[str] = set()
    removed_call_positions: set[tuple[int, int]] = set()
    summary_indices_to_remove: set[int] = set()
    summaries_to_insert: dict[int, list[tuple[tuple[int, int], HumanMessage]]] = {}

    for asset_name, data in fold_data.items():
        records_to_remove = data["records"]
        for record in records_to_remove:
            removed_call_positions.add(record.order)
            if record.call_id:
                removed_call_ids.add(record.call_id)

        prior_entry = existing_by_asset.get(asset_name)
        if prior_entry:
            summary_indices_to_remove.add(prior_entry[0])

        first_order = min(record.order for record in records_to_remove)
        summaries_to_insert.setdefault(data["insert_index"], []).append(
            (first_order, _build_summary_message(data["summary"]))
        )

    rebuilt: list[AnyMessage] = []
    for index, message in enumerate(messages):
        for _, summary_message in sorted(
            summaries_to_insert.get(index, []),
            key=lambda item: item[0],
        ):
            rebuilt.append(summary_message)

        if index in summary_indices_to_remove:
            continue

        if isinstance(message, AIMessage) and message.tool_calls:
            remaining_calls = [
                call
                for call_index, call in enumerate(message.tool_calls)
                if (index, call_index) not in removed_call_positions
            ]
            if len(remaining_calls) != len(message.tool_calls):
                if remaining_calls or _meaningful_content(message):
                    rebuilt.append(
                        _copy_ai_message_without_calls(message, remaining_calls)
                    )
                continue

        if (
            isinstance(message, ToolMessage)
            and message.tool_call_id in removed_call_ids
        ):
            continue
        rebuilt.append(message)

    for index in sorted(key for key in summaries_to_insert if key >= len(messages)):
        for _, summary_message in sorted(
            summaries_to_insert[index],
            key=lambda item: item[0],
        ):
            rebuilt.append(summary_message)

    return rebuilt


class AssetPlacementFoldingMiddleware(AgentMiddleware):
    """Collapse each successfully placed asset's tool trace into one message."""

    tools = []
    state_schema = AgentState

    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        folded = fold_completed_asset_messages(state["messages"])
        if folded is None:
            return None
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *folded,
            ]
        }

    def wrap_tool_call(self, request, handler):
        """Serialise tool calls when SERIALISE_TOOL_CALLS is True.

        This prevents the race condition where concurrent resolve_placement_intent
        calls see an inconsistent scene state (e.g. support objects not yet placed).
        """
        if SERIALISE_TOOL_CALLS:
            with _tool_execution_lock_sync:
                return handler(request)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        """Async variant – same serialisation via an asyncio.Lock.

        The ToolNode uses asyncio.gather for parallel execution; holding an
        asyncio.Lock ensures that only one tool call coroutine runs at a time
        without blocking the event loop.
        """
        if SERIALISE_TOOL_CALLS:
            async with _tool_execution_lock_async:
                return await handler(request)
        return await handler(request)

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

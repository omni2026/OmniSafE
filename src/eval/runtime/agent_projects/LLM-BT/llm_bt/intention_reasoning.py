"""
intention_reasoning.py

Stage 1 of LLM-BT: Intention Reasoning.

Converts natural language instructions into structured goal conditions
using LLM prompting, reproducing the original LLM-BT prompt engineering
pipeline (system prompt + demonstrations + semantic map → goal conditions).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from runtime.planning.reasoning_utils import extract_chat_completion_trace

from .household_domain import GoalCondition
from .prompts import PromptConfig, build_reasoning_prompt, build_guidance_prompt, parse_goal_conditions
from .semantic_map import SemanticMap

logger = logging.getLogger(__name__)

_REASONING_PREFIX_RE = re.compile(r'^\s*reasoning\s*:\s*(.+?)\s*$', re.I)


def _split_reasoning_output(content: str) -> tuple[str, str]:
    text = str(content or '')
    lines = text.splitlines()
    first_content_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_content_idx is not None:
        match = _REASONING_PREFIX_RE.match(lines[first_content_idx])
        if match:
            return (
                match.group(1).strip(),
                '\n'.join(lines[first_content_idx + 1:]).strip(),
            )
    return '', text.strip()


class IntentionReasoner:
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        semantic_map: Optional[SemanticMap] = None,
        prompt_config: Optional[PromptConfig] = None,
        domain: str = "household",
        max_guidance_rounds: int = 2,
    ):
        self._llm_config = dict(llm_config or {})
        self._semantic_map = semantic_map or SemanticMap()
        self._prompt_config = prompt_config or PromptConfig(domain=domain)
        self._domain = domain
        self._max_guidance_rounds = max_guidance_rounds
        self._client = None
        self._model = ""
        self.last_reasoning = ""
        self.llm_traces: List[Dict[str, Any]] = []

    def _record_reasoning(self, reasoning: Any) -> None:
        text = str(reasoning or '').strip()
        if not text:
            return
        fragments = [
            fragment.strip()
            for fragment in str(self.last_reasoning or '').split('\n\n')
            if fragment.strip()
        ]
        if text not in fragments:
            fragments.append(text)
        self.last_reasoning = '\n\n'.join(fragments)

    def _get_client(self):
        if self._client is None and self._llm_config:
            try:
                from openai import OpenAI
                api_key = str(self._llm_config.get("api_key", "") or "")
                base_url = str(self._llm_config.get("base_url", "") or "") or None
                if api_key:
                    self._client = OpenAI(api_key=api_key, base_url=base_url)
            except ImportError:
                raise ImportError("openai package required for IntentionReasoner")
        return self._client

    async def reason(self, instruction: str, context: Optional[Dict[str, Any]] = None) -> List[GoalCondition]:
        context = context or {}
        self.last_reasoning = ""
        self.llm_traces = []
        self._update_semantic_map_from_context(context)

        semantic_map_desc = self._semantic_map.get_state_description()
        messages = build_reasoning_prompt(instruction, self._prompt_config, semantic_map_desc)

        raw_output = await self._call_llm(messages)
        prompt_reasoning, goal_output = self._split_output(raw_output)
        self._record_reasoning(prompt_reasoning)
        goals = self._parse_goals(goal_output)

        guidance_round = 0
        while self._needs_guidance(goals) and guidance_round < self._max_guidance_rounds:
            guidance_prompt = build_guidance_prompt(
                self._goals_to_string(goals), instruction, self._prompt_config
            )
            messages.append({"role": "assistant", "content": goal_output})
            messages.append({"role": "user", "content": guidance_prompt})

            raw_output = await self._call_llm(messages)
            round_reasoning, goal_output = self._split_output(raw_output)
            self._record_reasoning(round_reasoning)
            goals = self._parse_goals(goal_output)
            guidance_round += 1

        goal_conditions = []
        for g in goals:
            goal_conditions.append(GoalCondition(
                predicate=g["predicate"],
                args=g["args"],
                raw_text=g.get("raw_text", ""),
            ))
        return goal_conditions

    def _split_output(self, raw_output: str) -> tuple[str, str]:
        if not self._prompt_config.include_reasoning_line:
            return '', str(raw_output or '').strip()
        return _split_reasoning_output(raw_output)

    def _update_semantic_map_from_context(self, context: Dict[str, Any]) -> None:
        metadata = context.get("metadata", {})
        vis_objs = (
            metadata.get("vis_objs")
            or metadata.get("visible_objects")
            or context.get("vis_objs")
            or []
        )
        if vis_objs and not self._semantic_map.locations:
            from .semantic_map import build_semantic_map_from_context
            self._semantic_map = build_semantic_map_from_context(context)

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        client = self._get_client()
        if client is None:
            raise RuntimeError("IntentionReasoner requires an LLM client. Configure llm_config.")

        model = self._llm_config.get("model", "")
        if not model:
            raise RuntimeError("No model specified in llm_config for IntentionReasoner.")
        self._model = model

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=256,
                temperature=0.0,
            ),
        )
        trace = extract_chat_completion_trace(
            response,
            prompt='\n'.join(
                str(message.get('content', '') or '')
                for message in messages
            ),
            model=model,
        )
        trace['label'] = 'intention_reasoning'
        trace['attempt'] = len(self.llm_traces)
        self.llm_traces.append(trace)
        self._record_reasoning(trace.get('reasoning_content'))
        return str(trace.get('content') or '').strip()

    def _parse_goals(self, raw_output: str) -> List[Dict[str, Any]]:
        return parse_goal_conditions(raw_output, self._domain)

    def _needs_guidance(self, goals: List[Dict[str, Any]]) -> bool:
        if not goals:
            return True
        for g in goals:
            args = g.get("args", [])
            for arg in args:
                if not arg or arg.strip() in ("", "?"):
                    return True
        return False

    @staticmethod
    def _goals_to_string(goals: List[Dict[str, Any]]) -> str:
        if not goals:
            return ""
        return ", ".join(g.get("raw_text", f"{g['predicate']}({','.join(g.get('args', []))})") for g in goals)

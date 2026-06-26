"""Shared helpers for prompt-induced reasoning extraction.

When the global ``capture_reasoning`` flag is enabled and the agent's LLM
provider is ``openai``, adapters either capture an agent-native rationale
(``Think:`` / ``Step N:``) or add a minimal output-contract line and remove it
before the agent's code/PDDL/XML parser runs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Tuple

# Append this to the agent's system prompt when reasoning capture is active.
REASONING_PROMPT_SUFFIX = (
    " Before your answer, output exactly one line starting with 'reasoning: ' "
    "briefly describing your task plan."
)

_REASONING_PREFIX_RE = re.compile(r'^\s*reasoning\s*:\s*(.+?)\s*$', re.I)
_THINK_RE = re.compile(
    r'^\s*Think\s*:\s*(.*?)(?=^\s*(?:Query\s*:|Stop\b))',
    re.I | re.M | re.S,
)
_STEP_RE = re.compile(r'^\s*Step\s+(\d+)\s*:\s*(.+?)\s*$', re.I | re.M)


def split_reasoning(content: str) -> Tuple[str, str]:
    """Separate a ``reasoning: ...`` first line from the remaining body.

    Returns ``(reasoning_text, body_text)``.  If no reasoning line is found
    the entire content is returned as body and reasoning is empty.
    """
    text = str(content or '')
    lines = text.splitlines()
    first_content_idx = next(
        (idx for idx, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_content_idx is not None:
        match = _REASONING_PREFIX_RE.match(lines[first_content_idx])
        if match:
            reasoning = match.group(1).strip()
            remaining = '\n'.join(lines[first_content_idx + 1:]).strip()
            return reasoning, remaining
    return '', text.strip()


def extract_think_reasoning(content: str) -> str:
    """Extract RoboAgent's native ``Think: ...`` section."""
    match = _THINK_RE.search(str(content or ''))
    return match.group(1).strip() if match else ''


def extract_step_reasoning(content: str) -> str:
    """Collect ISR-LLM's native ``Step N: ...`` planning headers."""
    return '\n'.join(
        f'Step {index}: {text.strip()}'
        for index, text in _STEP_RE.findall(str(content or ''))
        if text.strip()
    )


def join_reasoning(fragments: Iterable[str]) -> str:
    """Join non-empty reasoning fragments without duplicating repeats."""
    seen = set()
    result = []
    for fragment in fragments:
        text = str(fragment or '').strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return '\n\n'.join(result)


def should_capture_reasoning(
    llm_config: dict,
    capture_reasoning: bool,
) -> bool:
    """Return True when GPT-5.5 needs a prompt-exposed reasoning line.

    Reasoning-capable providers such as DeepSeek expose reasoning separately in
    the native chat-completion response and must not receive this prompt
    contract.  ``capture_reasoning`` still controls whether adapters publish
    either source into ``PlanningResult.reasoning``.
    """
    if not capture_reasoning:
        return False
    provider = str(llm_config.get('provider', '') or '').lower()
    model = str(llm_config.get('model', '') or '').lower().split('/')[-1]
    return provider == 'openai' and (
        model == 'gpt-5.5' or model.startswith('gpt-5.5-')
    )


_NATIVE_REASONING_FIELDS = (
    'reasoning_content',
    'reasoning',
    'thinking',
)


def extract_chat_completion_trace(
    response: Any,
    *,
    prompt: str = '',
    model: str = '',
) -> Dict[str, Any]:
    """Extract content and provider-native reasoning from a chat response.

    OpenAI-compatible providers do not agree on a single reasoning field.
    DeepSeek commonly returns ``message.reasoning_content``; other compatible
    services use ``reasoning`` or ``thinking``.  Unknown response fields may be
    retained by the OpenAI SDK as Pydantic extras, so those are inspected too.
    """
    trace: Dict[str, Any] = {
        'model': str(model or ''),
        'prompt_chars': len(str(prompt or '')),
        'prompt_tail': str(prompt or '')[-1200:],
        'content': None,
        'reasoning_content': None,
        'reasoning_field_source': None,
        'refusal': None,
        'finish_reason': None,
        'usage': None,
        'reasoning_tokens': None,
        'message_dump': None,
        'usage_dump': None,
    }

    try:
        choice = _field(response, 'choices')[0]
        message = _field(choice, 'message')
        trace['content'] = _field(message, 'content')
        trace['refusal'] = _field(message, 'refusal')
        trace['finish_reason'] = _field(choice, 'finish_reason')

        message_dump = _object_dump(message)
        trace['message_dump'] = message_dump

        for field in _NATIVE_REASONING_FIELDS:
            value = _field(message, field)
            if value in (None, '', [], {}):
                value = message_dump.get(field)
            reasoning_text = _reasoning_text(value)
            if reasoning_text:
                trace['reasoning_content'] = reasoning_text
                trace['reasoning_field_source'] = field
                break

        usage = _field(response, 'usage')
        if usage is not None:
            trace['usage'] = {
                'prompt_tokens': _field(usage, 'prompt_tokens'),
                'completion_tokens': _field(usage, 'completion_tokens'),
                'total_tokens': _field(usage, 'total_tokens'),
            }
            usage_dump = _object_dump(usage)
            trace['usage_dump'] = usage_dump
            details = _field(usage, 'completion_tokens_details')
            trace['reasoning_tokens'] = _field(details, 'reasoning_tokens')
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        trace['extract_error'] = repr(exc)

    return trace


def _field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(name)
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    for extras_name in ('model_extra', '__pydantic_extra__'):
        extras = getattr(value, extras_name, None)
        if isinstance(extras, Mapping) and name in extras:
            return extras.get(name)
    return None


def _object_dump(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        if hasattr(value, 'model_dump'):
            dumped = value.model_dump(exclude_none=False)
            return dict(dumped) if isinstance(dumped, Mapping) else {}
        if hasattr(value, 'dict'):
            dumped = value.dict()
            return dict(dumped) if isinstance(dumped, Mapping) else {}
    except Exception:
        return {}
    return {}


def _reasoning_text(value: Any) -> str:
    if value in (None, '', [], {}):
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ('text', 'content', 'summary', 'reasoning_content'):
            text = _reasoning_text(value.get(key))
            if text:
                return text
        return ''
    if isinstance(value, (list, tuple)):
        return '\n'.join(
            text
            for item in value
            if (text := _reasoning_text(item))
        ).strip()
    return str(value).strip()


def with_reasoning_prompt(system_prompt: str, capture: bool) -> str:
    """Append the reasoning instruction to a system prompt when active."""
    if not capture:
        return system_prompt
    return system_prompt + REASONING_PROMPT_SUFFIX


def extract_reasoning_from_output(raw_output: str) -> Tuple[str, str]:
    """Extract reasoning from a raw output string.

    Returns ``(reasoning, cleaned_output)``.  When no ``reasoning:`` prefix
    is found, reasoning is empty and the original output is returned cleaned.
    """
    return split_reasoning(raw_output)

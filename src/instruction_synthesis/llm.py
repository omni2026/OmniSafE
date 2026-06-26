from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx
from openai import OpenAI

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from instruction_synthesis.config import LLMSettings
else:
    from .config import LLMSettings


class LLMJsonError(RuntimeError):
    """Raised when an LLM response cannot be parsed as the requested JSON."""


class LLMEmptyResponseError(LLMJsonError):
    """Raised when the provider returns a choice without final message content."""


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse a JSON object from plain text or a markdown fenced response."""
    raw = str(text or "").strip()
    if not raw:
        raise LLMJsonError("empty LLM response")

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise LLMJsonError("LLM response did not contain a JSON object")
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMJsonError(f"invalid JSON object: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMJsonError("LLM response JSON must be an object")
    return dict(parsed)


def require_keys(data: Dict[str, Any], keys: Iterable[str], *, context: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise LLMJsonError(f"{context} missing required keys: {', '.join(missing)}")


class LLMClient:
    """Minimal wrapper around OpenAI-compatible chat completion APIs."""

    def __init__(self, settings: LLMSettings):
        settings.validate()
        self.settings = settings
        self.http_client = httpx.Client(
            timeout=settings.timeout,
            trust_env=settings.trust_env,
        )
        kwargs: Dict[str, Any] = {
            "api_key": settings.api_key,
            "http_client": self.http_client,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        if settings.extra_headers:
            kwargs["default_headers"] = settings.extra_headers
        self.client = OpenAI(**kwargs)

    def close(self) -> None:
        self.http_client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_json_mode: Optional[bool] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "temperature": self.settings.temperature if temperature is None else temperature,
        }
        json_mode = self.settings.json_mode if use_json_mode is None else use_json_mode
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        text = str(response.choices[0].message.content or "").strip()
        if text:
            return text
        raise LLMEmptyResponseError(self._empty_response_message(response))

    def complete_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        expected_keys: Iterable[str] = (),
        context: str = "LLM JSON response",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            text = self.complete(
                prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMEmptyResponseError:
            if not self.settings.json_mode:
                raise
            text = self.complete(
                prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                use_json_mode=False,
            )
        data = extract_json_object(text)
        require_keys(data, expected_keys, context=context)
        return data

    def _empty_response_message(self, response: Any) -> str:
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        message_data = {}
        if message is not None:
            try:
                message_data = message.model_dump(exclude_none=True)
            except AttributeError:
                message_data = dict(getattr(message, "__dict__", {}) or {})
        reasoning = str(message_data.get("reasoning_content") or "")
        usage = getattr(response, "usage", None)
        try:
            usage_data = usage.model_dump(exclude_none=True) if usage is not None else {}
        except AttributeError:
            usage_data = dict(getattr(usage, "__dict__", {}) or {})

        return (
            "empty LLM response: provider returned no final message.content. "
            f"model={getattr(response, 'model', self.settings.model)!r}, "
            f"finish_reason={getattr(choice, 'finish_reason', None)!r}, "
            f"reasoning_content_chars={len(reasoning)}, "
            f"message_keys={sorted(message_data.keys())}, usage={usage_data}"
        )

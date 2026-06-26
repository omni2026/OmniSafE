from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH, override=False)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _headers_env(name: str, default: Dict[str, str] | None = None) -> Dict[str, str]:
    value = _optional_env(name)
    if not value:
        return dict(default or {})
    if value.lower() in {"none", "false", "off"}:
        return {}

    headers: Dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{name} entries must use name:value format")
        header_name, header_value = item.split(":", 1)
        header_name = header_name.strip()
        header_value = header_value.strip()
        if not header_name or not header_value:
            raise ValueError(f"{name} entries must use non-empty name:value pairs")
        headers[header_name] = header_value
    return headers


@dataclass(frozen=True)
class LLMSettings:
    """Concrete settings for an OpenAI-compatible chat-completions client."""

    api_key: str | None = None
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    max_tokens: int = 2048
    temperature: float = 0.2
    timeout: float = 120.0
    json_mode: bool = False
    trust_env: bool = False
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> "LLMSettings":
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        resolved_model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        return cls(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=resolved_base_url,
            model=resolved_model,
            max_tokens=max_tokens if max_tokens is not None else _int_env("LLM_MAX_TOKENS", 2048),
            temperature=temperature if temperature is not None else _float_env("LLM_TEMPERATURE", 0.2),
            timeout=_float_env("LLM_TIMEOUT", 120.0),
            json_mode=_bool_env("LLM_JSON_MODE", False),
            trust_env=_bool_env("LLM_TRUST_ENV", False),
            extra_headers=_headers_env("LLM_EXTRA_HEADERS", {}),
        )

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY in "
                "instruction_synthesis/.env or in the process environment."
            )


class LLMConfig:
    """Class-style accessor for environment-based LLM configuration."""

    @classmethod
    def settings(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMSettings:
        return LLMSettings.from_env(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

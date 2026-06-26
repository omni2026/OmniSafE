"""
LLM Manager for Scene Generation Agent
Centralized configuration and initialization of language models
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms import Minimax
from langchain_community.chat_models import MiniMaxChat
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import ChatZhipuAI
from langchain_qwq import ChatQwen


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def load_env_from_run_config(config: Dict[str, Any], config_path: Path) -> Optional[Path]:
    """Load env vars from a centralized env file declared in run_config.

    Config keys:
    - llm.env_file: relative or absolute path to env file (default: .env)
    - llm.load_env_file: whether to load env_file automatically (default: true)
    """
    llm_cfg = (config.get("llm") or {}) if isinstance(config, dict) else {}
    load_env_file = bool(llm_cfg.get("load_env_file", True))
    if not load_env_file:
        return None

    env_file_cfg = llm_cfg.get("env_file", ".env")
    if not isinstance(env_file_cfg, str) or not env_file_cfg.strip():
        env_file_cfg = ".env"

    env_path = Path(env_file_cfg)
    if not env_path.is_absolute():
        env_path = (config_path.parent / env_path).resolve()

    if not env_path.exists():
        return None

    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = _strip_wrapping_quotes(value.strip())
            # Respect existing environment variables (shell wins).
            os.environ.setdefault(key, value)

    return env_path


class LLMManager:
    """
    Manages language model configurations for different tasks
    """
    
    def __init__(
        self, 
        provider_configs: Dict[str, Dict[str, Any]],
        default_provider: str = "openai"
    ):
        """
        Initialize LLM Manager
        
        Args:
            provider_configs: Provider configurations from run_config.json
            default_provider: Default provider when caller does not specify one
        """
        self.provider_configs = provider_configs
        self.default_provider = default_provider
        self._llm_cache: Dict[str, BaseChatModel] = {}

    def _resolve_api_key(self, provider: str, provider_config: Dict[str, Any]) -> Optional[str]:
        """Resolve api key from config value or env var indirection."""
        # Priority 1: direct api_key in config
        api_key = provider_config.get("api_key")
        if api_key:
            return api_key

        # Priority 2: environment variable from config
        api_key_env = provider_config.get("api_key_env")
        if api_key_env:
            return os.getenv(api_key_env)

        return None

    def _apply_proxy_policy(self, provider_config: Dict[str, Any]) -> None:
        """Apply optional proxy policy from provider config.

        When disable_proxy is true, force direct outbound requests by
        clearing common proxy environment variables.
        """
        if not provider_config.get("disable_proxy", False):
            return

        for key in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ]:
            os.environ.pop(key, None)

        # Ensure common HTTP clients bypass any proxy configuration.
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
    
    def _create_llm(
        self, 
        provider: str,
        model: str,
        temperature: float,
        timeout: int,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> BaseChatModel:
        """Create LLM instance based on provider"""
        provider_config = self.provider_configs[provider]
        api_key = self._resolve_api_key(provider, provider_config)
        self._apply_proxy_policy(provider_config)
        
        if not api_key:
            raise ValueError(
                f"API key not found for {provider}. "
                f"Please set '{provider}.api_key' or '{provider}.api_key_env' in run_config.json."
            )
        
        base_url = provider_config["base_url"]
        provider_extra_body = provider_config.get("extra_body")
        if isinstance(provider_extra_body, dict):
            merged_extra_body = dict(provider_extra_body)
            if extra_body:
                merged_extra_body.update(extra_body)
        else:
            merged_extra_body = extra_body

        print(f"[LLM] Creating provider={provider} model={model} base_url={base_url}")
        
        if provider in ["openai", "poe"]:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url
            )
        elif provider == "qwen":
            return ChatQwen(
                model=model,
                temperature=temperature,
                timeout=None,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url
            )
        elif provider == "zhipu":
            return ChatZhipuAI(
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url
            )
        elif provider == "deepseek":
            return ChatDeepSeek(
                model=model,
                temperature=temperature,
                timeout=timeout,
                api_key=api_key,
                base_url=base_url,
                extra_body=merged_extra_body,
            )
        elif provider in ["anthropic", "minimax"]:
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def get_llm(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout: int = 60,
        max_tokens: int = 4096,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> BaseChatModel:
        """
        Get or create LLM for a specific node
        
        Args:
            provider: Provider name, such as deepseek/qwen/openai/anthropic/poe
            model: Optional model name; defaults to provider default model
            temperature: Sampling temperature
            timeout: Timeout seconds
            max_tokens: Optional max token cap
            extra_body: Optional OpenAI-compatible provider extension body
            
        Returns:
            Configured LLM instance
        """
        provider = provider or self.default_provider
        
        if provider not in self.provider_configs:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(self.provider_configs.keys())}"
            )
        
        provider_cfg = self.provider_configs[provider]
        model_name = model or provider_cfg.get("default_model")
        if not model_name:
            raise ValueError(f"No model provided and no default_model configured for provider '{provider}'.")
        
        # Create cache key
        extra_body_key = json.dumps(extra_body, sort_keys=True, ensure_ascii=True) if extra_body else ""
        cache_key = f"{provider}_{model_name}_{temperature}_{timeout}_{max_tokens}_{extra_body_key}"
        
        # Return cached if exists
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        # Create new LLM
        llm = self._create_llm(
            provider=provider,
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        
        # Cache and return
        self._llm_cache[cache_key] = llm
        return llm
    
    def clear_cache(self):
        """Clear LLM cache"""
        self._llm_cache.clear()
    
    @classmethod
    def from_config_file(cls, config_path: Optional[str] = None) -> "LLMManager":
        """Build manager from `llm` section in run_config.json."""
        resolved = Path(config_path) if config_path else Path(__file__).with_name("run_config.json")
        with resolved.open("r", encoding="utf-8") as f:
            config = json.load(f)

        loaded_env_path = load_env_from_run_config(config, resolved)
        if loaded_env_path:
            print(f"[LLM] Loaded environment variables from: {loaded_env_path}")

        llm_cfg = config.get("llm", {})
        provider_configs = llm_cfg.get("providers", {})
        default_provider = llm_cfg.get("default_provider", "openai")

        if not provider_configs:
            raise ValueError(f"No providers found in llm.providers of config file: {resolved}")

        return cls(provider_configs=provider_configs, default_provider=default_provider)


# Global singleton instance
_llm_manager: Optional[LLMManager] = None


def get_llm_manager(
    config_path: Optional[str] = None,
) -> LLMManager:
    """
    Get or create global LLM manager
    
    Args:
        config_path: Optional path to run_config.json
        
    Returns:
        LLMManager instance
    """
    global _llm_manager
    
    if _llm_manager is None:
        _llm_manager = LLMManager.from_config_file(config_path)
    
    return _llm_manager


def configure_llm_manager(config_path: Optional[str] = None) -> LLMManager:
    """Reset and reinitialize singleton manager with an explicit config file."""
    reset_llm_manager()
    return get_llm_manager(config_path=config_path)


def reset_llm_manager():
    """Reset global LLM manager"""
    global _llm_manager
    if _llm_manager:
        _llm_manager.clear_cache()
    _llm_manager = None


# ============================================================================
# Convenience functions for quick access
# ============================================================================

def build_llm(provider: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Build a chat model instance from configured providers."""
    return get_llm_manager().get_llm(provider=provider, **kwargs)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    manager = get_llm_manager()
    model = manager.get_llm(provider="deepseek")
    print(f"Loaded provider=deepseek model={model.model_name}")

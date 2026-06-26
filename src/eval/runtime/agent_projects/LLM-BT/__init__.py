"""
LLMBTPlannerAdapter - registered as 'llm_bt' in the Eval pipeline.

Re-exports the adapter from the agent project root for convenient import.
"""

from __future__ import annotations

import sys
from pathlib import Path

_agent_root = str(Path(__file__).resolve().parent)
if _agent_root not in sys.path:
    sys.path.insert(0, _agent_root)

from llmbt_adapter import LLMBTPlannerAdapter

__all__ = ['LLMBTPlannerAdapter']
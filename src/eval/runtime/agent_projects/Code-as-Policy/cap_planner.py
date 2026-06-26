"""
Code-as-Policy Planner
======================
Standalone Code-as-Policy implementation that outputs complete programs
without executing them against a simulation environment.

Faithfully replicates the original recursive function generation mechanism
from the Interactive_Demo.ipynb notebook.

Input:  Natural language instruction + configurable atomic APIs
Output: Complete Python program (planning result)
"""

import os
from pathlib import Path
from openai import OpenAI
from typing import Any, Dict, List, Tuple, Optional, Set
from collections import OrderedDict
from dataclasses import dataclass, field
from time import sleep
import ast
import builtins
import keyword
import re
import copy

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    load_dotenv = None


def _load_project_dotenv() -> None:
    if load_dotenv is None:
        return

    candidates = [
        Path(__file__).resolve().parents[3] / '.env',
        Path.cwd() / '.env',
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        load_dotenv(dotenv_path=candidate, override=False)
        seen.add(candidate)


def _resolve_api_config(provider: str) -> Dict[str, str]:
    config = dict(API_CONFIGS[provider])
    env_name = str(config.get('api_key_env', '') or '').strip()
    if env_name:
        config['api_key'] = os.getenv(env_name, str(config.get('api_key', '') or ''))
    else:
        config['api_key'] = str(config.get('api_key', '') or '')
    return config


_load_project_dotenv()


# ============================================================================
# API Configuration
# ============================================================================

API_CONFIGS = {
    "zhipuai": {
        "api_key_env": "ZHIPUAI_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model_name": "glm-4-flash",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
    },
    "qwen": {
        "api_key_env": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-max",
    }
}

DEFAULT_PROVIDER = "deepseek"


# ============================================================================
# Utility Functions
# ============================================================================

def is_chat_model(model_name: str) -> bool:
    """Check if model requires chat completions API."""
    normalized = str(model_name or '').lower()
    chat_keywords = ['gpt-', 'glm', 'deepseek', 'qwen']
    return any(keyword in normalized for keyword in chat_keywords)


def effective_stop_tokens(model_name: str, stop: Optional[List[str]]) -> Optional[List[str]]:
    """Return stop tokens for message-based calls.

    CAP's historical completion prompts used ``#`` and ``objects = [`` as stop
    sentinels. These are unsafe for chat calls because models may start with
    comments or context echoes, truncating the response.
    """
    _ = (model_name, stop)
    return None


def clean_llm_content(content: Any) -> str:
    """Normalize common model wrappers while preserving generated code."""
    text = str(content or '').strip()
    fenced = re.search(r'```(?:python)?\s*\n(.*?)\n?```', text, re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r'^```python\s*\n', '', text, flags=re.I)
    text = re.sub(r'^```\s*\n', '', text)
    text = re.sub(r'\n```\s*$', '', text)
    return text.strip()


def merge_dicts(dicts: List[dict]) -> dict:
    """Merge multiple dictionaries into one."""
    return {
        k: v
        for d in dicts
        for k, v in d.items()
    }


SYSTEM_PROMPT = (
    "You are an expert Python code generation assistant for robot control. Given a task description, "
    "generate concise, correct Python code to control a robot in a simulated or real environment.    "
    "Do not generate explanations or comments, output only executable code for the robot to achieve "
    "the described task."
)

# Variant that asks the model to prefix its output with a ``reasoning:`` line.
# Used for providers whose models do not expose native chain-of-thought (e.g.
# OpenAI GPT series) so the evaluation pipeline still has reasoning text.
SYSTEM_PROMPT_WITH_REASONING = (
    SYSTEM_PROMPT
    + " Before the code, output exactly one line starting with 'reasoning: ' "
    "briefly describing your task plan."
)

_REASONING_PREFIX_RE = re.compile(r'^\s*reasoning\s*:\s*(.+?)\s*$', re.I)


def split_reasoning(content: str) -> Tuple[str, str]:
    """Separate a ``reasoning: ...`` first line from the code body.

    Returns ``(reasoning_text, code_text)``.  If no reasoning line is found
    the entire content is returned as code and reasoning is empty.
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


def _extract_trace(response, prompt: str, model_name: str) -> Dict[str, Any]:
    """Pull content + reasoning + finish_reason out of an OpenAI-compatible response.

    Reasoning is provider-specific: DeepSeek-Reasoner / Qwen-QwQ / GLM-Z1 expose
    ``message.reasoning_content``; some providers use ``reasoning`` or
    ``thinking``; OpenAI o-series doesn't expose CoT at all (only
    ``usage.completion_tokens_details.reasoning_tokens`` proves it happened).
    Standard ``deepseek-chat`` / ``glm-4-flash`` / ``qwen-plus`` / ``gpt-4o``
    do **not** return reasoning content — null is the correct value, not a bug.

    To make this verifiable, we also dump the full message + usage objects
    via pydantic ``model_dump`` so the raw API response is visible regardless
    of field naming.
    """
    trace: Dict[str, Any] = {
        'model': model_name,
        'prompt_chars': len(prompt),
        'prompt_tail': prompt[-1200:],
        'content': None,
        'reasoning_content': None,
        'reasoning_field_source': None,  # which attribute produced reasoning_content
        'refusal': None,
        'finish_reason': None,
        'usage': None,
        'reasoning_tokens': None,        # OpenAI o-series surfaces reasoning here
        'message_dump': None,            # full raw message (pydantic dump)
        'usage_dump': None,              # full raw usage (pydantic dump)
    }
    try:
        choice = response.choices[0]
        msg = choice.message
        trace['content'] = getattr(msg, 'content', None)
        trace['refusal'] = getattr(msg, 'refusal', None)
        trace['finish_reason'] = getattr(choice, 'finish_reason', None)

        # Reasoning content: try every name we've seen in the wild, fall back to
        # pydantic extras (where unknown fields are stashed by the SDK).
        reasoning_candidates = (
            'reasoning_content',  # DeepSeek-Reasoner, Zhipu GLM-Z1
            'reasoning',          # some Qwen / OpenAI-compat providers
            'thinking',           # Anthropic-style providers behind OpenAI shim
        )
        for attr in reasoning_candidates:
            val = getattr(msg, attr, None)
            if val:
                trace['reasoning_content'] = val
                trace['reasoning_field_source'] = attr
                break
        if trace['reasoning_content'] is None:
            extras = getattr(msg, '__pydantic_extra__', None) or {}
            for attr in reasoning_candidates:
                val = extras.get(attr) if isinstance(extras, dict) else None
                if val:
                    trace['reasoning_content'] = val
                    trace['reasoning_field_source'] = f'pydantic_extra:{attr}'
                    break

        # Full message dump — the most reliable way to see what the API really
        # returned (handy when you suspect the SDK is hiding a field).
        try:
            if hasattr(msg, 'model_dump'):
                trace['message_dump'] = msg.model_dump(exclude_none=False)
            elif hasattr(msg, 'dict'):
                trace['message_dump'] = msg.dict()
        except Exception as exc:  # pragma: no cover - defensive
            trace['message_dump_error'] = repr(exc)

        usage = getattr(response, 'usage', None)
        if usage is not None:
            trace['usage'] = {
                'prompt_tokens': getattr(usage, 'prompt_tokens', None),
                'completion_tokens': getattr(usage, 'completion_tokens', None),
                'total_tokens': getattr(usage, 'total_tokens', None),
            }
            # OpenAI o-series counts hidden reasoning tokens here.
            details = getattr(usage, 'completion_tokens_details', None)
            if details is not None:
                trace['reasoning_tokens'] = getattr(details, 'reasoning_tokens', None)
            try:
                if hasattr(usage, 'model_dump'):
                    trace['usage_dump'] = usage.model_dump(exclude_none=False)
                elif hasattr(usage, 'dict'):
                    trace['usage_dump'] = usage.dict()
            except Exception:  # pragma: no cover - defensive
                pass
    except Exception as exc:  # pragma: no cover - defensive
        trace['extract_error'] = repr(exc)
    return trace


def call_llm(
    client: OpenAI,
    model_name: str,
    prompt: str,
    stop: Optional[List[str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
    trace_sink: Optional[List[Dict[str, Any]]] = None,
    trace_label: str = '',
    use_reasoning_prompt: bool = False,
) -> str:
    """Call an OpenAI-compatible model through the chat API.

    When ``use_reasoning_prompt`` is True the system prompt asks the model to
    prefix its output with a ``reasoning:`` line.  This function splits that
    line from the code body, stores it in the trace as ``reasoning_content``
    (so the adapter's ``_extract_reasoning_from_traces`` picks it up), and
    returns only the cleaned code.  When False, the original system prompt is
    used and no splitting is performed.
    """
    use_chat = True  # Eval uses OpenAI-compatible chat endpoints only.
    request_stop = effective_stop_tokens(model_name, stop)
    system_prompt = SYSTEM_PROMPT_WITH_REASONING if use_reasoning_prompt else SYSTEM_PROMPT

    for attempt in range(max_retries):
        try:
            if use_chat:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=request_stop,
                )
                if trace_sink is not None:
                    trace = _extract_trace(response, prompt, model_name)
                    trace['label'] = trace_label
                    trace['attempt'] = attempt
                    trace_sink.append(trace)
                content = response.choices[0].message.content
            if use_reasoning_prompt:
                reasoning_text, code_text = split_reasoning(content)
                if reasoning_text and trace_sink is not None:
                    trace_sink[-1]['reasoning_content'] = reasoning_text
                    trace_sink[-1]['reasoning_field_source'] = 'prompt_prefix'
                return clean_llm_content(code_text)
            return clean_llm_content(content)

        except Exception as e:
            print(f'[call_llm] API call failed (attempt {attempt+1}/{max_retries}): {e}')
            if trace_sink is not None:
                trace_sink.append({
                    'label': trace_label,
                    'attempt': attempt,
                    'model': model_name,
                    'prompt_chars': len(prompt),
                    'prompt_tail': prompt[-1200:],
                    'error': repr(e),
                })
            if attempt < max_retries - 1:
                sleep(retry_delay)
            else:
                raise


# ============================================================================
# FunctionParser (ported from Interactive_Demo.ipynb)
# ============================================================================

class FunctionParser(ast.NodeTransformer):
    """
    AST visitor that detects function calls and assignments in generated code.
    Used by LMPFGenCollector to discover undefined functions that need generation.
    """

    def __init__(self, fs: dict, f_assigns: dict):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns
        self._builtin_names = set(dir(builtins))
        self._python_keywords = set(keyword.kwlist)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = ast.unparse(node).strip()
            f_name = ast.unparse(node.func).strip()
            if self._should_ignore_name(f_name):
                return node
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = ast.unparse(node).strip()
            f_name = ast.unparse(node.value.func).strip()
            if self._should_ignore_name(f_name):
                return node
            self._f_assigns[f_name] = assign_str
        return node

    def _should_ignore_name(self, f_name: str) -> bool:
        """Skip Python builtins/keywords so they are not treated as generated APIs."""
        return f_name in self._builtin_names or f_name in self._python_keywords


# ============================================================================
# CodeCollector
# ============================================================================

class CodeCollector:
    """
    Accumulates all code artifacts generated during a planning session.
    Replaces the execution mechanism of the original Code-as-Policy.
    """

    def __init__(self):
        self.fgen_functions: OrderedDict = OrderedDict()  # name -> source_code
        self.main_code_blocks: List[str] = []
        # Same blocks as ``main_code_blocks`` but without the per-call context
        # prefix (``objects = [...]``) that LMPCollector prepends for exec.
        # This is the LLM's raw cleaned content, suitable for "pure LLM output"
        # views; ``main_code_blocks`` stays for assemble() / exec round-tripping.
        self.main_code_blocks_raw: List[str] = []
        self.used_atomic_apis: Set[str] = set()
        self.used_lmp_calls: Set[str] = set()
        # Per-call records of every LLM round trip during this planning session.
        # Each entry is the dict produced by _extract_trace (plus label/attempt).
        self.llm_traces: List[Dict[str, Any]] = []

    def add_fgen_function(self, name: str, source: str):
        """Add a fgen-generated function definition."""
        if name not in self.fgen_functions:
            self.fgen_functions[name] = source

    def add_main_code(self, code: str):
        """Add a main code block from the high-level LMP."""
        self.main_code_blocks.append(code)

    def add_main_code_raw(self, code: str):
        """Add the LLM's raw (pre-context-prefix) code for the main block."""
        self.main_code_blocks_raw.append(code)

    def record_api_reference(self, name: str):
        """Record that an atomic API was referenced in the code."""
        self.used_atomic_apis.add(name)

    def record_lmp_reference(self, name: str):
        """Record that a low-level LMP was referenced in the code."""
        self.used_lmp_calls.add(name)

    def assemble(self, instruction: str, atomic_api_defs: dict, lmp_names: set) -> str:
        """Assemble all collected code into a complete program."""
        lines = []

        # Header
        lines.append('# ' + '=' * 60)
        lines.append('# Code-as-Policy Generated Program')
        lines.append(f'# Instruction: "{instruction}"')
        lines.append('# ' + '=' * 60)
        lines.append('')

        # Imports
        lines.append('# --- Imports ---')
        lines.append('import numpy as np')
        lines.append('from shapely.geometry import *')
        lines.append('from shapely.affinity import *')
        lines.append('')

        # fgen-generated functions (children before parents due to insertion order)
        if self.fgen_functions:
            lines.append('# --- Generated Functions ---')
            lines.append('')
            for name, source in self.fgen_functions.items():
                lines.append(source.strip())
                lines.append('')

        # Main code
        lines.append('# --- Main Program ---')
        lines.append('')
        for code in self.main_code_blocks:
            lines.append(code.strip())
            lines.append('')

        # Dependency reference
        lines.append('# ' + '=' * 60)
        lines.append('# Dependency Reference')
        lines.append('# ' + '=' * 60)
        lines.append('')

        if self.used_atomic_apis:
            lines.append('# Atomic APIs (required runtime dependencies):')
            for name in sorted(self.used_atomic_apis):
                sig = atomic_api_defs.get(name, {}).get('sig', name)
                desc = atomic_api_defs.get(name, {}).get('desc', '')
                lines.append(f'#   - {sig}: {desc}')
            lines.append('')

        if self.used_lmp_calls:
            lines.append('# Low-level LMPs (language model programs):')
            for name in sorted(self.used_lmp_calls):
                lines.append(f'#   - {name}(query, context)')
            lines.append('')

        return '\n'.join(lines)

    def assemble_llm_only(self) -> str:
        """Return only what the LLM actually generated, without the
        scaffolding ``assemble()`` adds (banner, imports, dependency reference,
        per-call ``objects = [...]`` context prefix).

        Output layout:
            <fgen function 1>
            <fgen function 2>
            ...
            <main code block 1>      # LLM's cleaned content, no context prefix
            <main code block 2>
            ...

        - ``fgen`` blocks come from recursive function generation calls (each is
          a real LLM round trip), in dependency order (children before parents).
        - main blocks are pulled from ``main_code_blocks_raw`` so the per-call
          ``objects = [...]`` context line that the adapter injects for exec is
          stripped out.
        """
        parts: List[str] = []
        for source in self.fgen_functions.values():
            parts.append(source.strip())
        for code in self.main_code_blocks_raw:
            parts.append(code.strip())
        return '\n\n'.join(p for p in parts if p)


# ============================================================================
# LMPFGenCollector (recursive function generation, code-collecting version)
# ============================================================================

class LMPFGenCollector:
    """
    Function generator that collects source code instead of executing it.
    Faithfully replicates the recursive function generation logic from
    the original LMPFGen.create_new_fs_from_code().
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        cfg: dict,
        fixed_var_names: Set[str],
        variable_var_names: Set[str],
        collector: CodeCollector,
        use_reasoning_prompt: bool = False,
    ):
        self.client = client
        self.model_name = model_name
        self.cfg = cfg
        self.stop_tokens = list(cfg['stop'])
        self.base_prompt = cfg['prompt_text']
        self.fixed_var_names = fixed_var_names
        self.variable_var_names = variable_var_names
        self.collector = collector
        self.use_reasoning_prompt = bool(use_reasoning_prompt)

    def create_f_from_sig(self, f_name: str, f_sig: str) -> str:
        """
        Generate function source code from a signature by calling the LLM.

        Args:
            f_name: Function name
            f_sig: Function call signature string (e.g. 'get_total(xs=numbers)')

        Returns:
            Function source code string
        """
        if len(self.variable_var_names) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self.variable_var_names)}"
        else:
            variable_vars_imports_str = ''

        use_query = f'{self.cfg["query_prefix"]}{f_sig}{self.cfg["query_suffix"]}'
        prompt_base = self.base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)
        prompt = f'{prompt_base}\n{use_query}'

        if self.collector is None:
            # Fallback: no collector, just return empty
            pass

        f_src = call_llm(
            self.client,
            self.model_name,
            prompt,
            stop=self.stop_tokens,
            temperature=self.cfg['temperature'],
            max_tokens=self.cfg['max_tokens'],
            trace_sink=self.collector.llm_traces if self.collector is not None else None,
            trace_label=f'fgen:{f_name}',
            use_reasoning_prompt=self.use_reasoning_prompt,
        )

        return self._extract_requested_function_source(f_name, f_src)

    def _extract_requested_function_source(self, f_name: str, f_src: str) -> str:
        """
        Keep only the requested function definition from the model output.

        Models may return helper functions or re-define already-known atomic APIs
        alongside the requested function. We only want to collect the target
        function here.
        """
        try:
            tree = ast.parse(f_src)
        except SyntaxError:
            return f_src

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == f_name:
                return ast.unparse(node).strip()

        return f_src

    def create_new_fs_from_code(
        self,
        code_str: str,
        known_names: Optional[Set[str]] = None,
    ) -> Dict[str, str]:
        """
        THE KEY RECURSIVE LOGIC.
        Parse code to find undefined function calls, generate their implementations,
        and recursively process function bodies for deeper undefined calls.

        Args:
            code_str: Generated code to analyze
            known_names: Set of already-defined function/variable names

        Returns:
            Dict mapping function names to their source code
        """
        # Step 1: Parse code with FunctionParser
        fs, f_assigns = {}, {}
        try:
            f_parser = FunctionParser(fs, f_assigns)
            f_parser.visit(ast.parse(code_str))
        except SyntaxError:
            return {}

        # Step 2: Prefer assignment forms over bare call forms
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if known_names is None:
            known_names = set()

        # Step 3: For each detected function call, check if it needs generation
        new_fs = {}
        for f_name, f_sig in fs.items():
            all_known = self.fixed_var_names | self.variable_var_names | set(new_fs.keys()) | known_names

            if f_name in all_known:
                continue

            # Generate function implementation via LLM
            f_src = self.create_f_from_sig(f_name, f_sig)

            if not f_src.strip():
                continue

            # Step 4: Extract function body for recursive analysis
            try:
                tree = ast.parse(f_src)
                if len(tree.body) > 0 and isinstance(tree.body[0], ast.FunctionDef):
                    f_def_body = ast.unparse(tree.body[0].body)

                    # Step 5: Recursively check for undefined functions in the body
                    # IMPORTANT: Include current function name to prevent infinite recursion
                    # when a function calls itself (self-reference)
                    all_known_for_child = all_known | {f_name}
                    child_fs = self.create_new_fs_from_code(f_def_body, known_names=all_known_for_child)

                    if len(child_fs) > 0:
                        # Add child functions first (order matters for output)
                        new_fs.update(child_fs)
                        for child_name, child_src in child_fs.items():
                            self.collector.add_fgen_function(child_name, child_src)
            except (SyntaxError, IndexError):
                pass

            # Add the parent function
            new_fs[f_name] = f_src
            self.collector.add_fgen_function(f_name, f_src)

        return new_fs


# ============================================================================
# LMPCollector (code-collecting version of LMP)
# ============================================================================

class LMPCollector:
    """
    Language Model Program that collects generated code instead of executing it.
    Faithfully replicates the prompt construction and code generation logic
    from the original LMP class.
    """

    def __init__(
        self,
        name: str,
        client: OpenAI,
        model_name: str,
        cfg: dict,
        lmp_fgen: Optional[LMPFGenCollector],
        fixed_var_names: Set[str],
        variable_var_names: Set[str],
        collector: CodeCollector,
        atomic_api_defs: dict,
        lmp_names: Set[str],
        use_reasoning_prompt: bool = False,
    ):
        self.name = name
        self.client = client
        self.model_name = model_name
        self.cfg = cfg
        self.base_prompt = cfg['prompt_text']
        self.stop_tokens = list(cfg['stop'])
        self.lmp_fgen = lmp_fgen
        self.fixed_var_names = fixed_var_names
        self.variable_var_names = variable_var_names
        self.collector = collector
        self.atomic_api_defs = atomic_api_defs
        self.lmp_names = lmp_names
        self.use_reasoning_prompt = bool(use_reasoning_prompt)
        self.exec_hist = ''

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query: str, context: str = '') -> Tuple[str, str]:
        """
        Build the full prompt for the LLM.
        Mirrors the original LMP.build_prompt() exactly.
        """
        # Generate import hint string for variable names
        if len(self.variable_var_names) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self.variable_var_names)}"
        else:
            variable_vars_imports_str = ''

        # Replace placeholder in base prompt
        prompt = self.base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        # Append execution history if maintaining session
        if self.cfg.get('maintain_session', False):
            prompt += f'\n{self.exec_hist}'

        # Append context
        if context != '':
            prompt += f'\n{context}'

        # Append query with prefix/suffix
        use_query = f'{self.cfg["query_prefix"]}{query}{self.cfg["query_suffix"]}'
        prompt += f'\n{use_query}'

        return prompt, use_query

    def __call__(self, query: str, context: str = '') -> str:
        """
        Generate code for the query by calling the LLM.
        Instead of executing (like the original), collects the code.
        """
        # Build prompt
        prompt, use_query = self.build_prompt(query, context=context)

        # Call LLM to generate code
        code_str = call_llm(
            self.client,
            self.model_name,
            prompt,
            stop=self.stop_tokens,
            temperature=self.cfg.get('temperature', 0),
            max_tokens=self.cfg.get('max_tokens', 512),
            trace_sink=self.collector.llm_traces if self.collector is not None else None,
            trace_label=f'main:{self.name}',
            use_reasoning_prompt=self.use_reasoning_prompt,
        )
        if self._is_empty_generation(code_str):
            raise RuntimeError(
                f'LMP {self.name} produced no executable code for query: {query}. '
                f'Raw content (after cleanup): {code_str!r}'
            )

        # Determine the full code to log
        if self.cfg.get('include_context', True) and context != '':
            to_exec = f'{context}\n{code_str}'
        else:
            to_exec = code_str

        print(f'[LMP {self.name}] Generated code:\n{to_exec}\n')

        # Detect and generate any new undefined functions
        if self.lmp_fgen is not None:
            new_fs = self.lmp_fgen.create_new_fs_from_code(code_str)
            # Update known names so future calls know these functions exist
            self.variable_var_names = self.variable_var_names | set(new_fs.keys())

        # Collect the main code (with context prefix for exec round-tripping)
        self.collector.add_main_code(to_exec)
        # Also keep the raw LLM output (no context prefix) for "pure LLM output"
        # consumers like adapter raw_output.
        self.collector.add_main_code_raw(code_str)

        # Scan for references to atomic APIs and low-level LMPs
        self._scan_references(code_str)

        # Update execution history for session maintenance
        self.exec_hist += f'\n{to_exec}'

        return code_str

    @staticmethod
    def _is_empty_generation(code_str: str) -> bool:
        meaningful_lines: List[str] = []
        for raw_line in str(code_str or '').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith(('objects =', 'rooms =', 'import ', 'from ')):
                continue
            meaningful_lines.append(line)
        return not meaningful_lines

    def _scan_references(self, code_str: str):
        """Scan code for references to atomic APIs and low-level LMPs."""
        for name in self.atomic_api_defs:
            if name in code_str:
                self.collector.record_api_reference(name)
        for name in self.lmp_names:
            if name in code_str:
                self.collector.record_lmp_reference(name)


# ============================================================================
# Prompt Templates (from Interactive_Demo.ipynb)
# ============================================================================

PROMPT_TABLETOP_UI = '''
# Python 2D robot control script
import numpy as np
{variable_vars_imports}

objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# the yellow block on the yellow bowl.
say('Ok - putting the yellow block on the yellow bowl')
put_first_on_second('yellow block', 'yellow bowl')
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# which block did you move.
say('I moved the yellow block')
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# move the green block to the top right corner.
say('Got it - putting the green block on the top right corner')
corner_pos = parse_position('top right corner')
put_first_on_second('green block', corner_pos)
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# stack the blue bowl on the yellow bowl on the green block.
order_bottom_to_top = ['green block', 'yellow block', 'blue bowl']
say(f'Sure - stacking from top to bottom: {", ".join(order_bottom_to_top)}')
stack_objects_in_order(object_names=order_bottom_to_top)
objects = ['cyan block', 'white block', 'cyan bowl', 'blue block', 'blue bowl', 'white bowl']
# move the cyan block into its corresponding bowl.
matches = {'cyan block': 'cyan bowl'}
say('Got it - placing the cyan block on the cyan bowl')
for first, second in matches.items():
  put_first_on_second(first, get_obj_pos(second))
objects = ['cyan block', 'white block', 'cyan bowl', 'blue block', 'blue bowl', 'white bowl']
# make a line of blocks on the right side.
say('No problem! Making a line of blocks on the right side')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
line_pts = parse_position(f'a 30cm vertical line on the right with {len(block_names)} points')
for block_name, pt in zip(block_names, line_pts):
  put_first_on_second(block_name, pt)
objects = ['yellow block', 'red block', 'yellow bowl', 'gray block', 'gray bowl', 'red bowl']
# put the small banana colored thing in between the blue bowl and green block.
say('Sure thing - putting the yellow block between the blue bowl and the green block')
target_pos = parse_position('a point in the middle betweeen the blue bowl and the green block')
put_first_on_second('yellow block', target_pos)
objects = ['yellow block', 'red block', 'yellow bowl', 'gray block', 'gray bowl', 'red bowl']
# can you cut the bowls in half.
say('no, I can only move objects around')
objects = ['yellow block', 'green block', 'yellow bowl', 'gray block', 'gray bowl', 'green bowl']
# stack the blocks on the right side with the gray one on the bottom.
say('Ok. stacking the blocks on the right side with the gray block on the bottom')
right_side = parse_position('the right side')
put_first_on_second('gray block', right_side)
order_bottom_to_top = ['gray block', 'green block', 'yellow block']
stack_objects_in_order(object_names=order_bottom_to_top)
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# hide the blue bowl.
bowl_name = np.random.choice(['yellow bowl', 'green bowl'])
say(f'Sounds good! Hiding the blue bowl under the {bowl_name}')
put_first_on_second(bowl_name, 'blue bowl')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# stack everything with the green block on top.
say('Ok! Stacking everything with the green block on the top')
order_bottom_to_top = ['blue bowl', 'pink bowl', 'green bowl', 'pink block', 'blue block', 'green block']
stack_objects_in_order(object_names=order_bottom_to_top)
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# move the grass-colored bowl to the left.
say('Sure - moving the green bowl left by 10 centimeters')
left_pos = parse_position('a point 10cm left of the green bowl')
put_first_on_second('green bowl', left_pos)
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# why did you move the red bowl.
say(f'I did not move the red bowl')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# undo that.
say('Sure - moving the green bowl right by 10 centimeters')
left_pos = parse_position('a point 10cm right of the green bowl')
put_first_on_second('green bowl', left_pos)
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# place the top most block to the corner closest to the bottom most block.
top_block_name = parse_obj_name('top most block', f'objects = {get_obj_names()}')
bottom_block_name = parse_obj_name('bottom most block', f'objects = {get_obj_names()}')
closest_corner_pos = parse_position(f'the corner closest to the {bottom_block_name}', f'objects = {get_obj_names()}')
say(f'Putting the {top_block_name} on the {get_corner_name(closest_corner_pos)}')
put_first_on_second(top_block_name, closest_corner_pos)
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# move the brown bowl to the side closest to the green block.
closest_side_position = parse_position('the side closest to the green block')
say(f'Got it - putting the brown bowl on the {get_side_name(closest_side_position)}')
put_first_on_second('brown bowl', closest_side_position)
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# place the green block to the right of the bowl that has the blue block.
bowl_name = parse_obj_name('the bowl that has the blue block', f'objects = {get_obj_names()}')
if bowl_name:
  target_pos = parse_position(f'a point 10cm to the right of the {bowl_name}')
  say(f'No problem - placing the green block to the right of the {bowl_name}')
  put_first_on_second('green block', target_pos)
else:
  say('There are no bowls that has the blue block')
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# place the blue block in the empty bowl.
empty_bowl_name = parse_obj_name('the empty bowl', f'objects = {get_obj_names()}')
if empty_bowl_name:
  say(f'Ok! Putting the blue block on the {empty_bowl_name}')
  put_first_on_second('blue block', empty_bowl_name)
else:
  say('There are no empty bowls')
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# move the other blocks to the bottom corners.
block_names = parse_obj_name('blocks other than the blue block', f'objects = {get_obj_names()}')
corners = parse_position('the bottom corners')
for block_name, pos in zip(block_names, corners):
  put_first_on_second(block_name, pos)
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# move the red bowl a lot to the left of the blocks.
say('Sure! Moving the red bowl to a point left of the blocks')
left_pos = parse_position('a point 20cm left of the blocks')
put_first_on_second('red bowl', left_pos)
objects = ['pink block', 'gray block', 'orange block']
# move the pinkish colored block on the bottom side.
say('Ok - putting the pink block on the bottom side')
bottom_side_pos = parse_position('the bottom side')
put_first_on_second('pink block', bottom_side_pos)
objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl']
# is the blue block to the right of the yellow bowl?
if parse_question('is the blue block to the right of the yellow bowl?', f'objects = {get_obj_names()}'):
  say('yes, there is a blue block to the right of the yellow bow')
else:
  say('no, there is\'t a blue block to the right of the yellow bow')
objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl']
# how many yellow objects are there?
n_yellow_objs = parse_question('how many yellow objects are there', f'objects = {get_obj_names()}')
say(f'there are {n_yellow_objs} yellow object')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# move the left most block to the green bowl.
left_block_name = parse_obj_name('left most block', f'objects = {get_obj_names()}')
say(f'Moving the {left_block_name} on the green bowl')
put_first_on_second(left_block_name, 'green bowl')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# move the other blocks to different corners.
block_names = parse_obj_name(f'blocks other than the {left_block_name}', f'objects = {get_obj_names()}')
corners = parse_position('the corners')
say(f'Ok - moving the other {len(block_names)} blocks to different corners')
for block_name, pos in zip(block_names, corners):
  put_first_on_second(block_name, pos)
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# is the pink block on the green bowl.
if parse_question('is the pink block on the green bowl', f'objects = {get_obj_names()}'):
  say('Yes - the pink block is on the green bowl.')
else:
  say('No - the pink block is not on the green bowl.')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# what are the blocks left of the green bowl.
left_block_names =  parse_question('what are the blocks left of the green bowl', f'objects = {get_obj_names()}')
if len(left_block_names) > 0:
  say(f'These blocks are left of the green bowl: {", ".join(left_block_names)}')
else:
  say('There are no blocks left of the green bowl')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# if you see a purple bowl put it on the blue bowl
if is_obj_visible('purple bowl'):
  say('Putting the purple bowl on the pink bowl')
  put_first_on_second('purple bowl', 'pink bowl')
else:
  say('I don\\'t see a purple bowl')
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# imagine that the bowls are different biomes on earth and imagine that the blocks are parts of a building.
say('ok')
objects = ['yellow block', 'green block', 'yellow bowl', 'blue block', 'blue bowl', 'green bowl']
# now build a tower in the grasslands.
order_bottom_to_top = ['green bowl', 'blue block', 'green block', 'yellow block']
say('stacking the blocks on the green bowl')
stack_objects_in_order(object_names=order_bottom_to_top)
objects = ['yellow block', 'green block', 'yellow bowl', 'gray block', 'gray bowl', 'green bowl']
# show me what happens when the desert gets flooded by the ocean.
say('putting the yellow bowl on the blue bowl')
put_first_on_second('yellow bowl', 'blue bowl')
objects = ['pink block', 'gray block', 'orange block']
# move all blocks 5cm toward the top.
say('Ok - moving all blocks 5cm toward the top')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
for block_name in block_names:
  target_pos = parse_position(f'a point 5cm above the {block_name}')
  put_first_on_second(block_name, target_pos)
objects = ['cyan block', 'white block', 'purple bowl', 'blue block', 'blue bowl', 'white bowl']
# make a triangle of blocks in the middle.
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
triangle_pts = parse_position(f'a triangle with size 10cm around the middle with {len(block_names)} points')
say('Making a triangle of blocks around the middle of the workspace')
for block_name, pt in zip(block_names, triangle_pts):
  put_first_on_second(block_name, pt)
objects = ['cyan block', 'white block', 'purple bowl', 'blue block', 'blue bowl', 'white bowl']
# make the triangle smaller.
triangle_pts = transform_shape_pts('scale it by 0.5x', shape_pts=triangle_pts)
say('Making the triangle smaller')
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
for block_name, pt in zip(block_names, triangle_pts):
  put_first_on_second(block_name, pt)
objects = ['brown bowl', 'red block', 'brown block', 'red bowl', 'pink bowl', 'pink block']
# put the red block on the farthest bowl.
farthest_bowl_name = parse_obj_name('the bowl farthest from the red block', f'objects = {get_obj_names()}')
say(f'Putting the red block on the {farthest_bowl_name}')
put_first_on_second('red block', farthest_bowl_name)
'''.strip()

PROMPT_PARSE_OBJ_NAME = '''
import numpy as np
{variable_vars_imports}

objects = ['blue block', 'cyan block', 'purple bowl', 'gray bowl', 'brown bowl', 'pink block', 'purple block']
# the block closest to the purple bowl.
block_names = ['blue block', 'cyan block', 'purple block']
block_positions = get_obj_positions_np(block_names)
closest_block_idx = get_closest_idx(points=block_positions, point=get_obj_pos('purple bowl'))
closest_block_name = block_names[closest_block_idx]
ret_val = closest_block_name
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# the blocks.
ret_val = ['brown block', 'blue block']
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# the brown objects.
ret_val = ['brown bowl', 'brown block']
objects = ['brown bowl', 'banana', 'brown block', 'apple', 'blue bowl', 'blue block']
# a fruit that's not the apple
fruit_names = ['banana', 'apple']
for fruit_name in fruit_names:
    if fruit_name != 'apple':
        ret_val = fruit_name
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# blocks above the brown bowl.
block_names = ['blue block', 'cyan block', 'purple block']
brown_bowl_pos = get_obj_pos('brown bowl')
use_block_names = []
for block_name in block_names:
    if get_obj_pos(block_name)[1] > brown_bowl_pos[1]:
        use_block_names.append(block_name)
ret_val = use_block_names
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# the blue block.
ret_val = 'blue block'
objects = ['blue block', 'cyan block', 'purple bowl', 'brown bowl', 'purple block']
# the block closest to the bottom right corner.
corner_pos = parse_position('bottom right corner')
block_names = ['blue block', 'cyan block', 'purple block']
block_positions = get_obj_positions_np(block_names)
closest_block_idx = get_closest_idx(points=block_positions, point=corner_pos)
closest_block_name = block_names[closest_block_idx]
ret_val = closest_block_name
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the left most block.
block_names = ['green block', 'brown block', 'blue block']
block_positions = get_obj_positions_np(block_names)
left_block_idx = np.argsort(block_positions[:, 0])[0]
left_block_name = block_names[left_block_idx]
ret_val = left_block_name
objects = ['brown bowl', 'green block', 'brown block', 'green bowl', 'blue bowl', 'blue block']
# the bowl on near the top.
bowl_names = ['brown bowl', 'green bowl', 'blue bowl']
bowl_positions = get_obj_positions_np(bowl_names)
top_bowl_idx = np.argsort(block_positions[:, 1])[-1]
top_bowl_name = bowl_names[top_bowl_idx]
ret_val = top_bowl_name
objects = ['yellow bowl', 'purple block', 'yellow block', 'purple bowl', 'pink bowl', 'pink block']
# the third bowl from the right.
bowl_names = ['yellow bowl', 'purple bowl', 'pink bowl']
bowl_positions = get_obj_positions_np(bowl_names)
bowl_idx = np.argsort(block_positions[:, 0])[-3]
bowl_name = bowl_names[bowl_idx]
ret_val = bowl_name
'''.strip()

PROMPT_PARSE_POSITION = '''
import numpy as np
from shapely.geometry import *
from shapely.affinity import *
{variable_vars_imports}

# a 30cm horizontal line in the middle with 3 points.
middle_pos = denormalize_xy([0.5, 0.5])
start_pos = middle_pos + [-0.3/2, 0]
end_pos = middle_pos + [0.3/2, 0]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=3)
ret_val = points
# a 20cm vertical line near the right with 4 points.
middle_pos = denormalize_xy([1, 0.5])
start_pos = middle_pos + [0, -0.2/2]
end_pos = middle_pos + [0, 0.2/2]
line = make_line(start=start_pos, end=end_pos)
points = interpolate_pts_on_line(line=line, n=4)
ret_val = points
# a diagonal line from the top left to the bottom right corner with 5 points.
top_left_corner = denormalize_xy([0, 1])
bottom_right_corner = denormalize_xy([1, 0])
line = make_line(start=top_left_corner, end=bottom_right_corner)
points = interpolate_pts_on_line(line=line, n=5)
ret_val = points
# a triangle with size 10cm with 3 points.
polygon = make_triangle(size=0.1, center=denormalize_xy([0.5, 0.5]))
points = get_points_from_polygon(polygon)
ret_val = points
# the corner closest to the sun colored block.
block_name = parse_obj_name('the sun colored block', f'objects = {get_obj_names()}')
corner_positions = np.array([denormalize_xy(pos) for pos in [[0, 0], [0, 1], [1, 1], [1, 0]]])
closest_corner_pos = get_closest_point(points=corner_positions, point=get_obj_pos(block_name))
ret_val = closest_corner_pos
# the side farthest from the right most bowl.
bowl_name = parse_obj_name('the right most bowl', f'objects = {get_obj_names()}')
side_positions = np.array([denormalize_xy(pos) for pos in [[0.5, 0], [0.5, 1], [1, 0.5], [0, 0.5]]])
farthest_side_pos = get_farthest_point(points=side_positions, point=get_obj_pos(bowl_name))
ret_val = farthest_side_pos
# a point above the third block from the bottom.
block_name = parse_obj_name('the third block from the bottom', f'objects = {get_obj_names()}')
ret_val = get_obj_pos(block_name) + [0.1, 0]
# a point 10cm left of the bowls.
bowl_names = parse_obj_name('the bowls', f'objects = {get_obj_names()}')
bowl_positions = get_all_object_positions_np(obj_names=bowl_names)
left_obj_pos = bowl_positions[np.argmin(bowl_positions[:, 0])] + [-0.1, 0]
ret_val = left_obj_pos
# the bottom side.
bottom_pos = denormalize_xy([0.5, 0])
ret_val = bottom_pos
# the top corners.
top_left_pos = denormalize_xy([0, 1])
top_right_pos = denormalize_xy([1, 1])
ret_val = [top_left_pos, top_right_pos]
'''.strip()

PROMPT_PARSE_QUESTION = '''
{variable_vars_imports}

objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl', 'fruit', 'green block', 'black bowl']
# is the blue block to the right of the yellow bowl?
ret_val = get_obj_pos('blue block')[0] > get_obj_pos('yellow bowl')[0]
objects = ['yellow bowl', 'blue block', 'yellow block', 'blue bowl', 'fruit', 'green block', 'black bowl']
# how many yellow objects are there?
yellow_object_names = parse_obj_name('the yellow objects', f'objects = {get_obj_names()}')
ret_val = len(yellow_object_names)
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# is the pink block on the green bowl?
ret_val = bbox_contains_pt(container_name='green bowl', obj_name='pink block')
objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']
# what are the blocks left of the green bowl?
block_names = parse_obj_name('the blocks', f'objects = {get_obj_names()}')
green_bowl_pos = get_obj_pos('green bowl')
left_block_names = []
for block_name in block_names:
  if get_obj_pos(block_name)[0] < green_bowl_pos[0]:
    left_block_names.append(block_name)
ret_val = left_block_names
objects = ['pink block', 'yellow block', 'pink bowl', 'blue block', 'blue bowl', 'yellow bowl']
# is the sun colored block above the blue bowl?
sun_block_name = parse_obj_name('sun colored block', f'objects = {get_obj_names()}')
sun_block_pos = get_obj_pos(sun_block_name)
blue_bowl_pos = get_obj_pos('blue bowl')
ret_val = sun_block_pos[1] > blue_bowl_pos[1]
objects = ['pink block', 'yellow block', 'pink bowl', 'blue block', 'blue bowl', 'yellow bowl']
# is the green block below the blue bowl?
ret_val = get_obj_pos('green block')[1] < get_obj_pos('blue bowl')[1]
'''.strip()

PROMPT_TRANSFORM_SHAPE_PTS = '''
import numpy as np
{variable_vars_imports}

# make it bigger by 1.5.
new_shape_pts = scale_pts_around_centroid_np(shape_pts, scale_x=1.5, scale_y=1.5)
# move it to the right by 10cm.
new_shape_pts = translate_pts_np(shape_pts, delta=[0.1, 0])
# move it to the top by 20cm.
new_shape_pts = translate_pts_np(shape_pts, delta=[0, 0.2])
# rotate it clockwise by 40 degrees.
new_shape_pts = rotate_pts_around_centroid_np(shape_pts, angle=-np.deg2rad(40))
# rotate by 30 degrees and make it slightly smaller
new_shape_pts = rotate_pts_around_centroid_np(shape_pts, angle=np.deg2rad(30))
new_shape_pts = scale_pts_around_centroid_np(new_shape_pts, scale_x=0.7, scale_y=0.7)
# move it toward the blue block.
block_name = parse_obj_name('the blue block', f'objects = {get_obj_names()}')
block_pos = get_obj_pos(block_name)
mean_delta = np.mean(block_pos - shape_pts, axis=1)
new_shape_pts = translate_pts_np(shape_pts, mean_delta)
'''.strip()

PROMPT_FGEN = '''
import numpy as np
from shapely.geometry import *
from shapely.affinity import *
{variable_vars_imports}

# define function: total = get_total(xs=numbers).
def get_total(xs):
    return np.sum(xs)

# define function: y = eval_line(x, slope, y_intercept=0).
def eval_line(x, slope, y_intercept):
    return x * slope + y_intercept

# define function: pt = get_pt_to_the_left(pt, dist).
def get_pt_to_the_left(pt, dist):
    return pt + [-dist, 0]

# define function: pt = get_pt_to_the_top(pt, dist).
def get_pt_to_the_top(pt, dist):
    return pt + [0, dist]

# define function line = make_line_by_length(length=x).
def make_line_by_length(length):
  line = LineString([[0, 0], [length, 0]])
  return line

# define function: line = make_vertical_line_by_length(length=x).
def make_vertical_line_by_length(length):
  line = make_line_by_length(length)
  vertical_line = rotate(line, 90)
  return vertical_line

# define function: pt = interpolate_line(line, t=0.5).
def interpolate_line(line, t):
  pt = line.interpolate(t, normalized=True)
  return np.array(pt.coords[0])

# example: scale a line by 2.
line = make_line_by_length(1)
new_shape = scale(line, xfact=2, yfact=2)

# example: put object1 on top of object0.
put_first_on_second('object1', 'object0')

# example: get the position of the first object.
obj_names = get_obj_names()
pos_2d = get_obj_pos(obj_names[0])
'''.strip()


# ============================================================================
# Household Scenario: Prompts
# ============================================================================

PROMPT_HOUSEHOLD_TASK = '''
# Python household robot control script
import numpy as np
{variable_vars_imports}

objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# put the apple in the fridge.
say('Ok - putting the apple in the fridge')
navigate_to('apple')
pick('apple')
open('fridge')
place('fridge')
close('fridge')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# bring me the remote from the living room.
say('Sure - bringing the remote to you')
navigate_to('living_room')
pick('remote')
navigate_to('user')
place('user')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# close all the drawers.
drawer_names = parse_obj_name('all the drawers', f'objects = {get_obj_names()}')
say(f'Closing {len(drawer_names)} drawers')
for drawer in drawer_names:
  if is_open(drawer):
    close(drawer)
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# is the fridge open?
if is_open('fridge'):
  say('Yes, the fridge is open')
else:
  say('No, the fridge is closed')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# wash the mug and put it in the cabinet.
say('Sure - washing the mug and putting it away')
navigate_to('mug')
pick('mug')
navigate_to('sink')
place('sink')
toggle('faucet')
say('Washing the mug')
toggle('faucet')
pick('mug')
navigate_to('cabinet')
open('cabinet')
place('cabinet')
close('cabinet')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# put the towel from the bathroom in the bedroom closet.
say('Moving the towel from the bathroom to the bedroom closet')
navigate_to('bathroom')
pick('towel')
navigate_to('bedroom_closet')
open('closet')
place('closet')
close('closet')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# put down whatever you are holding.
if is_holding():
  held = get_held_object()
  say(f'Putting down the {held}')
  place('table')
else:
  say('I am not holding anything')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# if the window is open, close it.
if is_open('window'):
  say('The window is open, closing it')
  close('window')
else:
  say('The window is already closed')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# set the table for two.
say('Setting the table for two')
for i in range(2):
  plate_name = parse_obj_name('a clean plate', f'objects = {get_obj_names()}')
  navigate_to(plate_name)
  pick(plate_name)
  navigate_to('dining_table')
  place('dining_table')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# how many books are in the living room?
book_names = parse_obj_name('books in the living room', f'objects = {get_obj_names()}')
say(f'There are {len(book_names)} books in the living room')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# turn on the light in the kitchen.
say('Turning on the kitchen light')
navigate_to('kitchen')
if is_off('light_switch'):
  turn_on('light_switch')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# turn off the kitchen light.
say('Turning off the kitchen light')
navigate_to('kitchen')
if is_on('light_switch'):
  turn_off('light_switch')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# turn off all the lights that are on.
light_names = parse_obj_name('all the lights', f'objects = {get_obj_names()}')
for light in light_names:
  if is_on(light):
    turn_off(light)
say('Turned off the lights that were on')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# put all the books on the bookshelf.
book_names = parse_obj_name('all the books', f'objects = {get_obj_names()}')
say(f'Putting {len(book_names)} books on the bookshelf')
for book in book_names:
  navigate_to(book)
  pick(book)
  navigate_to('bookshelf')
  place('bookshelf')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# can you cook dinner?
say('I can help with simple tasks like fetching items and organizing, but I cannot cook a full dinner')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# put the second cup from the left into the cabinet.
cup_name = parse_obj_name('second cup from the left', f'objects = {get_obj_names()}')
say(f'Putting the {cup_name} in the cabinet')
navigate_to(cup_name)
pick(cup_name)
open('cabinet')
place('cabinet')
close('cabinet')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# make me a cup of coffee.
say('Ok - making coffee')
navigate_to('mug')
pick('mug')
navigate_to('coffee_machine')
place('coffee_machine')
toggle('coffee_machine')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# throw away the banana peel.
say('Throwing away the banana peel')
navigate_to('banana_peel')
pick('banana_peel')
navigate_to('trash_can')
open('trash_can')
place('trash_can')
close('trash_can')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# is the milk in the fridge?
if is_obj_in('milk', 'fridge'):
  say('Yes, the milk is in the fridge')
else:
  say('No, the milk is not in the fridge')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# pick up the keys and put them on the entry table.
say('Ok - picking up the keys and putting them on the entry table')
navigate_to('keys')
pick('keys')
navigate_to('entry_table')
place('entry_table')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# organize the desk by putting all papers in the drawer and all pens in the pen cup.
paper_names = parse_obj_name('all the papers', f'objects = {get_obj_names()}')
for paper in paper_names:
  navigate_to(paper)
  pick(paper)
  open('drawer')
  place('drawer')
  close('drawer')
pen_names = parse_obj_name('all the pens', f'objects = {get_obj_names()}')
for pen in pen_names:
  navigate_to(pen)
  pick(pen)
  navigate_to('pen_cup')
  place('pen_cup')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# where is the book?
book_pos = get_obj_pos('book')
room = get_room_at(book_pos)
say(f'The book is in the {room}')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'towel', 'keys']
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
# check all the rooms and report which ones have lights on.
for room in rooms:
  if parse_question(f'is the light on in the {room}?', f'objects = {get_obj_names()}'):
    say(f'The light is on in the {room}')
  else:
    say(f'The light is off in the {room}')
'''.strip()

PROMPT_HOUSEHOLD_PARSE_OBJ_NAME = '''
import numpy as np
{variable_vars_imports}

objects = ['apple', 'banana', 'milk', 'mug', 'red_cup', 'blue_cup', 'green_cup', 'plate', 'bowl']
# the cup closest to the fridge.
cup_names = ['red_cup', 'blue_cup', 'green_cup']
cup_positions = get_obj_positions_np(cup_names)
fridge_pos = get_obj_pos('fridge')
closest_idx = get_closest_idx(points=cup_positions, point=fridge_pos)
ret_val = cup_names[closest_idx]
objects = ['apple', 'banana', 'milk', 'mug', 'red_cup', 'blue_cup', 'green_cup', 'plate', 'bowl']
# the red things.
ret_val = [name for name in get_obj_names() if 'red' in name]
objects = ['apple', 'banana', 'milk', 'mug', 'red_cup', 'blue_cup', 'green_cup', 'plate', 'bowl']
# all the cups.
ret_val = [name for name in get_obj_names() if 'cup' in name]
objects = ['apple', 'banana', 'milk', 'mug', 'red_cup', 'blue_cup', 'green_cup', 'plate', 'bowl']
# a fruit that is not the apple.
fruit_names = ['apple', 'banana']
for fruit in fruit_names:
    if fruit != 'apple':
        ret_val = fruit
objects = ['apple', 'banana', 'milk', 'mug', 'red_cup', 'blue_cup', 'green_cup', 'plate', 'bowl']
# the second cup from the left.
cup_names = ['red_cup', 'blue_cup', 'green_cup']
cup_positions = get_obj_positions_np(cup_names)
left_to_right = np.argsort(cup_positions[:, 0])
ret_val = cup_names[left_to_right[1]]
objects = ['apple', 'banana', 'milk', 'mug', 'red_cup', 'blue_cup', 'green_cup', 'plate', 'bowl']
# the dishes.
ret_val = [name for name in get_obj_names() if name in ['plate', 'bowl', 'mug']]
objects = ['apple', 'banana', 'milk', 'mug', 'red_cup', 'blue_cup', 'green_cup', 'plate', 'bowl']
# the biggest object.
obj_names = get_obj_names()
obj_bboxes = [get_bbox(name) for name in obj_names]
obj_areas = [(b[2]-b[0])*(b[3]-b[1]) for b in obj_bboxes]
biggest_idx = np.argmax(obj_areas)
ret_val = obj_names[biggest_idx]
objects = ['apple', 'banana', 'milk', 'mug', 'red_cup', 'blue_cup', 'green_cup', 'plate', 'bowl']
# all the drawers.
ret_val = [name for name in get_obj_names() if 'drawer' in name]
objects = ['apple', 'banana', 'milk', 'mug', 'red_cup', 'blue_cup', 'green_cup', 'plate', 'bowl']
# the cup on the dining table.
cup_names = ['red_cup', 'blue_cup', 'green_cup']
table_pos = get_location_pos('dining_table')
cup_positions = get_obj_positions_np(cup_names)
for cup_name, cup_pos in zip(cup_names, cup_positions):
    if np.linalg.norm(cup_pos - table_pos) < 0.5:
        ret_val = cup_name
        break
'''.strip()

PROMPT_HOUSEHOLD_PARSE_QUESTION = '''
{variable_vars_imports}

objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'fridge', 'cabinet']
# is the fridge open?
ret_val = is_open('fridge')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'fridge', 'cabinet']
# how many cups are there?
cup_names = [name for name in get_obj_names() if 'cup' in name]
ret_val = len(cup_names)
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'fridge', 'cabinet']
# is the milk in the fridge?
ret_val = is_obj_in('milk', 'fridge')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'fridge', 'cabinet']
# what room is the robot in?
ret_val = get_room_at(get_obj_pos('robot'))
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'fridge', 'cabinet']
# is the robot holding anything?
ret_val = is_holding()
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'fridge', 'cabinet']
# are all the drawers closed?
drawer_names = [name for name in get_obj_names() if 'drawer' in name]
ret_val = all(is_closed(d) for d in drawer_names)
objects = ['apple', 'banana', 'milk', 'plate', 'book', 'fridge', 'cabinet', 'drawer1', 'drawer2']
# is the light on in the kitchen?
ret_val = is_on('kitchen_light')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'fridge', 'cabinet']
# is the TV off?
ret_val = is_off('tv')
objects = ['apple', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'fridge', 'cabinet']
# which room has the most objects?
rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
obj_counts = [len(get_room_objects(r)) for r in rooms]
ret_val = rooms[np.argmax(obj_counts)]
'''.strip()

PROMPT_HOUSEHOLD_FGEN = '''
import numpy as np
{variable_vars_imports}

# define function: total = get_total(xs=numbers).
def get_total(xs):
    return np.sum(xs)

# define function: is_kitchen_empty = is_room_empty(room_name='kitchen').
def is_room_empty(room_name):
    objects = get_room_objects(room_name)
    return len(objects) == 0

# define function: n_books = count_objects_in_room(obj_type='book', room='living_room').
def count_objects_in_room(obj_type, room):
    room_objects = get_room_objects(room)
    return len([o for o in room_objects if obj_type in o])

# define function: sorted_objs = sort_objects_by_distance(obj_names, ref_pos).
def sort_objects_by_distance(obj_names, ref_pos):
    positions = get_obj_positions_np(obj_names)
    distances = np.linalg.norm(positions - ref_pos, axis=1)
    sorted_indices = np.argsort(distances)
    return [obj_names[i] for i in sorted_indices]

# define function: nearest_room = get_nearest_room(pos).
def get_nearest_room(pos):
    rooms = ['kitchen', 'living_room', 'bedroom', 'bathroom']
    room_positions = [get_location_pos(r) for r in rooms]
    distances = [np.linalg.norm(np.array(p) - np.array(pos)) for p in room_positions]
    return rooms[np.argmin(distances)]

# define function: done = tidy_room(room_name='bedroom').
def tidy_room(room_name):
    room_objects = get_room_objects(room_name)
    for obj_name in room_objects:
        navigate_to(obj_name)
        pick(obj_name)
        navigate_to('storage_shelf')
        place('storage_shelf')
    return True

# example: put the apple in the fridge.
navigate_to('apple')
pick('apple')
open('fridge')
place('fridge')
close('fridge')

# example: close all open containers.
container_names = parse_obj_name('open containers', f'objects = {get_obj_names()}')
for c in container_names:
    close(c)

# example: turn off all the appliances that are on.
appliance_names = parse_obj_name('all the appliances', f'objects = {get_obj_names()}')
for app in appliance_names:
    if is_on(app):
        turn_off(app)

# example: get the position of the nearest object.
obj_names = get_obj_names()
obj_positions = get_obj_positions_np(obj_names)
robot_pos = get_obj_pos('robot')
nearest_idx = get_closest_idx(points=obj_positions, point=robot_pos)
nearest_obj = obj_names[nearest_idx]
pos = get_obj_pos(nearest_obj)
'''.strip()


# ============================================================================
# LMP Configuration (from Interactive_Demo.ipynb)
# ============================================================================

DEFAULT_LMP_CONFIGS = {
    'tabletop_ui': {
        'prompt_text': PROMPT_TABLETOP_UI,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# ',
        'query_suffix': '.',
        'stop': ['#', 'objects = ['],
        'maintain_session': True,
        'debug_mode': False,
        'include_context': True,
        'has_return': False,
        'return_val_name': 'ret_val',
    },
    'parse_obj_name': {
        'prompt_text': PROMPT_PARSE_OBJ_NAME,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# ',
        'query_suffix': '.',
        'stop': ['#', 'objects = ['],
        'maintain_session': False,
        'debug_mode': False,
        'include_context': True,
        'has_return': True,
        'return_val_name': 'ret_val',
    },
    'parse_position': {
        'prompt_text': PROMPT_PARSE_POSITION,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# ',
        'query_suffix': '.',
        'stop': ['#'],
        'maintain_session': False,
        'debug_mode': False,
        'include_context': True,
        'has_return': True,
        'return_val_name': 'ret_val',
    },
    'parse_question': {
        'prompt_text': PROMPT_PARSE_QUESTION,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# ',
        'query_suffix': '.',
        'stop': ['#', 'objects = ['],
        'maintain_session': False,
        'debug_mode': False,
        'include_context': True,
        'has_return': True,
        'return_val_name': 'ret_val',
    },
    'transform_shape_pts': {
        'prompt_text': PROMPT_TRANSFORM_SHAPE_PTS,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# ',
        'query_suffix': '.',
        'stop': ['#'],
        'maintain_session': False,
        'debug_mode': False,
        'include_context': True,
        'has_return': True,
        'return_val_name': 'new_shape_pts',
    },
    'fgen': {
        'prompt_text': PROMPT_FGEN,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# define function: ',
        'query_suffix': '.',
        'stop': ['# define', '# example'],
        'maintain_session': False,
        'debug_mode': False,
        'include_context': True,
    }
}

# Low-level LMP names (these are pre-defined LMPs, not generated by fgen)
LOW_LEVEL_LMP_NAMES = {'parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts'}


# ============================================================================
# Household Scenario: LMP Configuration
# ============================================================================

HOUSEHOLD_LMP_CONFIGS = {
    'household_task': {
        'prompt_text': PROMPT_HOUSEHOLD_TASK,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# ',
        'query_suffix': '.',
        'stop': ['#', 'objects = ['],
        'maintain_session': True,
        'debug_mode': False,
        'include_context': True,
        'has_return': False,
        'return_val_name': 'ret_val',
    },
    'parse_obj_name': {
        'prompt_text': PROMPT_HOUSEHOLD_PARSE_OBJ_NAME,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# ',
        'query_suffix': '.',
        'stop': ['#', 'objects = ['],
        'maintain_session': False,
        'debug_mode': False,
        'include_context': True,
        'has_return': True,
        'return_val_name': 'ret_val',
    },
    'parse_question': {
        'prompt_text': PROMPT_HOUSEHOLD_PARSE_QUESTION,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# ',
        'query_suffix': '.',
        'stop': ['#', 'objects = ['],
        'maintain_session': False,
        'debug_mode': False,
        'include_context': True,
        'has_return': True,
        'return_val_name': 'ret_val',
    },
    'fgen': {
        'prompt_text': PROMPT_HOUSEHOLD_FGEN,
        'max_tokens': 512,
        'temperature': 0,
        'query_prefix': '# define function: ',
        'query_suffix': '.',
        'stop': ['# define', '# example'],
        'maintain_session': False,
        'debug_mode': False,
        'include_context': True,
    }
}

LOW_LEVEL_LMP_NAMES_HOUSEHOLD = {'parse_obj_name', 'parse_question'}


# ============================================================================
# Default Atomic APIs (from LMP_wrapper in Interactive_Demo.ipynb)
# ============================================================================

DEFAULT_ATOMIC_APIS = {
    'get_obj_pos': {
        'sig': 'get_obj_pos(obj_name)',
        'desc': 'Get object xy position in robot base frame',
        'source': 'def get_obj_pos(obj_name):\n    # Return xy position of object\n    pass',
    },
    'get_obj_names': {
        'sig': 'get_obj_names()',
        'desc': 'Get list of all object names in scene',
        'source': 'def get_obj_names():\n    # Return list of available object names\n    pass',
    },
    'put_first_on_second': {
        'sig': 'put_first_on_second(arg1, arg2)',
        'desc': 'Pick up arg1 and place on arg2 (both can be names or positions)',
        'source': 'def put_first_on_second(arg1, arg2):\n    # Pick up arg1 and place on arg2\n    pass',
    },
    'is_obj_visible': {
        'sig': 'is_obj_visible(obj_name)',
        'desc': 'Check if object is visible in current scene',
        'source': 'def is_obj_visible(obj_name):\n    # Return True if object is in scene\n    pass',
    },
    'denormalize_xy': {
        'sig': 'denormalize_xy(pos_normalized)',
        'desc': 'Convert normalized [0,1] coords to workspace coordinates',
        'source': 'def denormalize_xy(pos_normalized):\n    # Convert normalized coords to workspace\n    pass',
    },
    'get_corner_name': {
        'sig': 'get_corner_name(pos)',
        'desc': 'Get name of nearest corner for given position',
        'source': 'def get_corner_name(pos):\n    # Return nearest corner name\n    pass',
    },
    'get_side_name': {
        'sig': 'get_side_name(pos)',
        'desc': 'Get name of nearest side for given position',
        'source': 'def get_side_name(pos):\n    # Return nearest side name\n    pass',
    },
    'get_bbox': {
        'sig': 'get_bbox(obj_name)',
        'desc': 'Get axis-aligned bounding box of object',
        'source': 'def get_bbox(obj_name):\n    # Return (min_x, min_y, max_x, max_y)\n    pass',
    },
    'get_color': {
        'sig': 'get_color(obj_name)',
        'desc': 'Get RGBA color tuple for object',
        'source': 'def get_color(obj_name):\n    # Return RGBA color tuple\n    pass',
    },
    'say': {
        'sig': 'say(msg)',
        'desc': 'Print a message',
        'source': 'def say(msg):\n    print(msg)',
    },
}


# ============================================================================
# Household Scenario: Atomic APIs
# ============================================================================

HOUSEHOLD_ATOMIC_APIS = {
    # --- Navigation ---
    'navigate_to': {
        'sig': 'navigate_to(location)',
        'desc': 'Navigate robot to a named location or object (e.g. "kitchen", "apple", "dining_table")',
        'source': 'def navigate_to(location):\n    # Move robot base to the specified location\n    pass',
    },
    # --- Manipulation ---
    'pick': {
        'sig': 'pick(obj_name)',
        'desc': 'Pick up the specified object (must be nearby and reachable)',
        'source': 'def pick(obj_name):\n    # Grasp and lift the object\n    pass',
    },
    'place': {
        'sig': 'place(target_location)',
        'desc': 'Place currently held object at/on the target location',
        'source': 'def place(target_location):\n    # Place held object at the target\n    pass',
    },
    'put_first_on_second': {
        'sig': 'put_first_on_second(arg1, arg2)',
        'desc': 'Pick up arg1 and place on/in arg2 (both can be names or locations)',
        'source': 'def put_first_on_second(arg1, arg2):\n    # Combined pick-and-place\n    navigate_to(arg1)\n    pick(arg1)\n    navigate_to(arg2)\n    place(arg2)',
    },
    'open': {
        'sig': 'open(obj_name)',
        'desc': 'Open a container (drawer, door, fridge, cabinet, window, etc.)',
        'source': 'def open(obj_name):\n    # Open the specified container\n    pass',
    },
    'close': {
        'sig': 'close(obj_name)',
        'desc': 'Close a container (drawer, door, fridge, cabinet, window, etc.)',
        'source': 'def close(obj_name):\n    # Close the specified container\n    pass',
    },
    'toggle': {
        'sig': 'toggle(obj_name)',
        'desc': 'Toggle an appliance or switch on/off (light, faucet, coffee machine, etc.)',
        'source': 'def toggle(obj_name):\n    # Toggle the appliance state\n    pass',
    },
    'turn_on': {
        'sig': 'turn_on(obj_name)',
        'desc': 'Turn on an appliance or device (light, faucet, coffee machine, TV, etc.)',
        'source': 'def turn_on(obj_name):\n    # Power on / activate the specified appliance\n    pass',
    },
    'turn_off': {
        'sig': 'turn_off(obj_name)',
        'desc': 'Turn off an appliance or device (light, faucet, coffee machine, TV, etc.)',
        'source': 'def turn_off(obj_name):\n    # Power off / deactivate the specified appliance\n    pass',
    },
    'is_on': {
        'sig': 'is_on(obj_name)',
        'desc': 'Check if an appliance or device is currently powered on / active',
        'source': 'def is_on(obj_name):\n    # Return True if the device is currently on\n    pass',
    },
    'is_off': {
        'sig': 'is_off(obj_name)',
        'desc': 'Check if an appliance or device is currently powered off / inactive',
        'source': 'def is_off(obj_name):\n    # Return True if the device is currently off\n    pass',
    },
    'push': {
        'sig': 'push(obj_name)',
        'desc': 'Push an object (e.g. chair, button)',
        'source': 'def push(obj_name):\n    # Push the object\n    pass',
    },
    'pull': {
        'sig': 'pull(obj_name)',
        'desc': 'Pull an object (e.g. drawer, door)',
        'source': 'def pull(obj_name):\n    # Pull the object\n    pass',
    },
    # --- Perception ---
    'get_obj_pos': {
        'sig': 'get_obj_pos(obj_name)',
        'desc': 'Get object position as [x, y, z] in world frame',
        'source': 'def get_obj_pos(obj_name):\n    # Return xyz position of object\n    pass',
    },
    'get_obj_names': {
        'sig': 'get_obj_names()',
        'desc': 'Get list of all object names in the scene',
        'source': 'def get_obj_names():\n    # Return list of available object names\n    pass',
    },
    'is_obj_visible': {
        'sig': 'is_obj_visible(obj_name)',
        'desc': 'Check if object is visible in current scene',
        'source': 'def is_obj_visible(obj_name):\n    # Return True if object is in scene\n    pass',
    },
    'is_open': {
        'sig': 'is_open(obj_name)',
        'desc': 'Check if a container is currently open',
        'source': 'def is_open(obj_name):\n    # Return True if container is open\n    pass',
    },
    'is_closed': {
        'sig': 'is_closed(obj_name)',
        'desc': 'Check if a container is currently closed',
        'source': 'def is_closed(obj_name):\n    # Return True if container is closed\n    pass',
    },
    'is_holding': {
        'sig': 'is_holding()',
        'desc': 'Check if robot is currently holding an object',
        'source': 'def is_holding():\n    # Return True if robot is holding something\n    pass',
    },
    'get_held_object': {
        'sig': 'get_held_object()',
        'desc': 'Get the name of the object currently held by the robot',
        'source': 'def get_held_object():\n    # Return name of held object or None\n    pass',
    },
    # 'get_container_contents': {
    #     'sig': 'get_container_contents(container_name)',
    #     'desc': 'Get list of object names inside a container',
    #     'source': 'def get_container_contents(container_name):\n    # Return list of objects inside the container\n    pass',
    # },
    'is_obj_in': {
        'sig': 'is_obj_in(obj_name, container_name)',
        'desc': 'Check if an object is inside a container',
        'source': 'def is_obj_in(obj_name, container_name):\n    # Return True if object is in container\n    pass',
    },
    'get_room_at': {
        'sig': 'get_room_at(position)',
        'desc': 'Get the room name for a given position',
        'source': 'def get_room_at(position):\n    # Return room name at the given position\n    pass',
    },
    'get_room_objects': {
        'sig': 'get_room_objects(room_name)',
        'desc': 'Get list of all object names in a specific room',
        'source': 'def get_room_objects(room_name):\n    # Return objects in the specified room\n    pass',
    },
    'get_location_pos': {
        'sig': 'get_location_pos(location_name)',
        'desc': 'Get position of a named location or landmark',
        'source': 'def get_location_pos(location_name):\n    # Return xyz position of named location\n    pass',
    },
    'get_bbox': {
        'sig': 'get_bbox(obj_name)',
        'desc': 'Get axis-aligned bounding box of object',
        'source': 'def get_bbox(obj_name):\n    # Return (min_x, min_y, max_x, max_y)\n    pass',
    },
    # --- Utility ---
    'say': {
        'sig': 'say(msg)',
        'desc': 'Print a message to communicate with the user',
        'source': 'def say(msg):\n    print(msg)',
    },
}


# ============================================================================
# CodeAsPolicyPlanner (Main Entry Point)
# ============================================================================

class CodeAsPolicyPlanner:
    """
    Standalone Code-as-Policy Planner.

    Input:  Natural language instruction + configurable atomic APIs
    Output: Complete Python program (the planning result)

    Supports two scenarios:
      - 'tabletop': Desktop manipulation with 2D block/bowl pick-and-place
      - 'household': Household robot tasks with navigation, containers, and appliances

    Example:
        >>> from openai import OpenAI
        >>> client = OpenAI(api_key='...', base_url='https://api.deepseek.com')
        >>> planner = CodeAsPolicyPlanner(client, scenario='tabletop')
        >>> program = planner.plan("put the blue block on the yellow bowl")
        >>> print(program)
        >>> # Or for household:
        >>> planner = CodeAsPolicyPlanner(client, scenario='household')
        >>> program = planner.plan("put the apple in the fridge")
        >>> print(program)
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str = 'deepseek-chat',
        scenario: str = 'tabletop',
        task_prompt: str = None,
        fgen_prompt: str = None,
        lmp_configs: dict = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        verbose: bool = True,
        use_reasoning_prompt: bool = False,
    ):
        self.client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.scenario = scenario
        self.use_reasoning_prompt = bool(use_reasoning_prompt)

        # Select base configuration based on scenario
        if scenario == 'household':
            base_lmp_configs = HOUSEHOLD_LMP_CONFIGS
            base_atomic_apis = HOUSEHOLD_ATOMIC_APIS
            self._low_level_lmp_names = LOW_LEVEL_LMP_NAMES_HOUSEHOLD
            self._main_lmp_name = 'household_task'
            default_task_prompt = PROMPT_HOUSEHOLD_TASK
            default_fgen_prompt = PROMPT_HOUSEHOLD_FGEN
        else:  # tabletop (default)
            base_lmp_configs = DEFAULT_LMP_CONFIGS
            base_atomic_apis = DEFAULT_ATOMIC_APIS
            self._low_level_lmp_names = LOW_LEVEL_LMP_NAMES
            self._main_lmp_name = 'tabletop_ui'
            default_task_prompt = PROMPT_TABLETOP_UI
            default_fgen_prompt = PROMPT_FGEN

        # LMP configs (deep copy to allow per-instance modification)
        self.lmp_configs = copy.deepcopy(base_lmp_configs)
        if lmp_configs:
            self.lmp_configs.update(lmp_configs)
        for cfg in self.lmp_configs.values():
            if isinstance(cfg, dict):
                cfg['temperature'] = self.temperature
                cfg['max_tokens'] = self.max_tokens

        # Override prompts if provided (use scenario defaults otherwise)
        self.lmp_configs[self._main_lmp_name]['prompt_text'] = task_prompt or default_task_prompt
        self.lmp_configs['fgen']['prompt_text'] = fgen_prompt or default_fgen_prompt

        # Atomic API registry
        self.atomic_api_defs = copy.deepcopy(base_atomic_apis)

        # Trace of every LLM round trip from the most recent ``plan()`` call.
        # Populated even when planning fails (RuntimeError) so callers can see
        # the model's actual content / reasoning_content / refusal / finish_reason.
        self.last_traces: List[Dict[str, Any]] = []
        # Pure LLM output from the most recent ``plan()`` call: fgen functions
        # + main code, without the banner / imports / dependency-reference
        # scaffolding ``assemble()`` wraps around them.
        self.last_llm_only: str = ''

    def register_atomic_api(
        self,
        name: str,
        signature: str = '',
        description: str = '',
        source: str = '',
    ):
        """Register or update an atomic API definition."""
        self.atomic_api_defs[name] = {
            'sig': signature or name,
            'desc': description,
            'source': source,
        }

    def register_atomic_apis(self, apis: Dict[str, dict]):
        """Register multiple atomic APIs at once."""
        for name, api_def in apis.items():
            self.register_atomic_api(name, **api_def)

    def plan(self, instruction: str, context: str = '') -> str:
        """
        Generate a complete program for the given instruction.

        Args:
            instruction: Natural language instruction (e.g. "put the blue block on the yellow bowl")
            context: Additional context (e.g. "objects = ['blue block', 'yellow bowl']")

        Returns:
            Complete Python program as a string
        """
        # Create a fresh code collector for this planning session
        collector = CodeCollector()
        # Reset last_traces and bind the collector's trace list so that any
        # downstream RuntimeError still leaves traces visible on the planner.
        self.last_traces = collector.llm_traces

        # Build variable name sets
        fixed_var_names = {'np'}
        low_level_names = self._low_level_lmp_names
        variable_var_names = set(self.atomic_api_defs.keys()) | low_level_names

        # Create function generator (LMPFGenCollector)
        fgen = LMPFGenCollector(
            client=self.client,
            model_name=self.model_name,
            cfg=self.lmp_configs['fgen'],
            fixed_var_names=fixed_var_names,
            variable_var_names=variable_var_names,
            collector=collector,
            use_reasoning_prompt=self.use_reasoning_prompt,
        )

        # Create the high-level LMP (LMPCollector)
        main_name = self._main_lmp_name
        lmp = LMPCollector(
            name=main_name,
            client=self.client,
            model_name=self.model_name,
            cfg=self.lmp_configs[main_name],
            lmp_fgen=fgen,
            fixed_var_names=fixed_var_names,
            variable_var_names=variable_var_names,
            collector=collector,
            atomic_api_defs=self.atomic_api_defs,
            lmp_names=low_level_names,
            use_reasoning_prompt=self.use_reasoning_prompt,
        )

        # Run the LMP to generate code
        lmp(instruction, context=context)

        # Capture the "pure LLM output" view (fgen functions + main code, no
        # banner / imports / dependency block / per-call context prefix). The
        # adapter exposes this as ``raw_output`` while the assembled program
        # below is kept under ``actions[0]['code']`` for runtime exec.
        self.last_llm_only = collector.assemble_llm_only()

        # Assemble the complete program
        program = collector.assemble(
            instruction=instruction,
            atomic_api_defs=self.atomic_api_defs,
            lmp_names=low_level_names,
        )

        return program

    def plan_to_file(self, instruction: str, filepath: str, context: str = ''):
        """Generate a plan and write it to a file."""
        program = self.plan(instruction, context=context)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(program)
        if self.verbose:
            print(f'[CodeAsPolicyPlanner] Program written to: {filepath}')
        return program


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == '__main__':
    # Select provider
    provider = DEFAULT_PROVIDER
    config = _resolve_api_config(provider)

    # Create OpenAI client
    client = OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url'],
    )

    # Create planner
    planner = CodeAsPolicyPlanner(
        client=client,
        model_name=config['model_name'],
        verbose=True,
    )

    # print('=' * 70)
    # print('Code-as-Policy Planner')
    # print('=' * 70)

    # # Example 1: Simple instruction
    # print('\nExample 1: Simple pick-and-place')
    # print('-' * 70)
    # program = planner.plan(
    #     instruction='put the blue block on the yellow bowl',
    #     context="objects = ['blue block', 'red block', 'yellow bowl', 'blue bowl']",
    # )
    # print(program)

    # # Example 2: Complex instruction requiring spatial reasoning
    # print('\n\nExample 2: Spatial reasoning')
    # print('-' * 70)
    # planner2 = CodeAsPolicyPlanner(
    #     client=client,
    #     model_name=config['model_name'],
    #     verbose=True,
    # )
    # program2 = planner2.plan(
    #     instruction="sort all blocks by distance to the yellow bowl and arrange them in a line from closest to farthest",
    #     context="objects = ['triangle', 'green block', 'red block', 'yellow bowl']",
    # )
    # print(program2)

    print("==" * 60)

    # Example: Tabletop scenario (default)
    planner_tabletop = CodeAsPolicyPlanner(
        client=client,
        model_name=config['model_name'],
        scenario='tabletop',
        verbose=True,
    )
    program3 = planner_tabletop.plan(
        instruction="stack everything with the green block on the top",
        context="objects = ['pink block', 'green block', 'pink bowl', 'blue block', 'blue bowl', 'green bowl']",
    )
    print(program3)

    print("\n" + "==" * 60)

    # Example: Household scenario
    planner_household = CodeAsPolicyPlanner(
        client=client,
        model_name=config['model_name'],
        scenario='household',
        verbose=True,
    )
    program4 = planner_household.plan(
        instruction="grasp the knife in the kitchen and put it on the dining table",
        context="objects = ['knife', 'banana', 'milk', 'mug', 'plate', 'book', 'remote', 'fridge', 'cabinet', 'dining_table']",
    )
    print(program4)

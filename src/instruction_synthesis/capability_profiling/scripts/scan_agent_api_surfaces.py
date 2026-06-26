#!/usr/bin/env python3
"""Collect candidate action-surface evidence from an embodied planning repo.

This scanner intentionally over-collects. Use it as a first pass for a target
planning-agent repository, then review and normalize candidates manually with
the skill workflow. Do not treat candidates as the final unified action list.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".pddl",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
}

SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
}

COMMON_IMPORT_MODULES = {
    "argparse",
    "asyncio",
    "base64",
    "builtins",
    "collections",
    "copy",
    "dataclasses",
    "dotenv",
    "functools",
    "hashlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "pickle",
    "random",
    "re",
    "sys",
    "time",
    "typing",
    "yaml",
    "numpy",
    "pandas",
    "openai",
    "haystack",
    "sentence_transformers",
    "PIL",
}


@dataclass
class Candidate:
    name: str
    surface: str
    file: str
    line: int
    signature: str = ""
    raw: str = ""
    hint: str = ""


def iter_files(root: Path, max_file_bytes: int) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in TEXT_SUFFIXES and root.stat().st_size <= max_file_bytes:
            yield root
        return

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        yield path


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
    return ""


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def clean_name(value: str) -> str:
    return value.strip().strip("\"'` ")


def add_candidate(
    out: List[Candidate],
    *,
    root: Path,
    path: Path,
    text: str,
    match: re.Match[str],
    name: str,
    surface: str,
    signature: str = "",
    hint: str = "",
) -> None:
    raw = match.group(0).strip()
    out.append(
        Candidate(
            name=clean_name(name),
            surface=surface,
            file=str(path.relative_to(root)) if path != root else path.name,
            line=line_number(text, match.start()),
            signature=signature.strip(),
            raw=raw[:500],
            hint=hint,
        )
    )


def extract_code_fences(text: str) -> Iterable[tuple[int, str]]:
    pattern = re.compile(r"```+([A-Za-z0-9_-]*)\s*\n(.*?)```+", re.DOTALL)
    for match in pattern.finditer(text):
        lang = (match.group(1) or "").lower()
        if lang and lang not in {"python", "py", "text", "bash", "json"}:
            continue
        yield match.start(2), match.group(2)


def split_action_list(value: str) -> List[str]:
    parts = re.split(r"[,;\n]", value)
    actions = []
    for item in parts:
        item = clean_name(item)
        item = re.sub(r"^\d+[\.\)]\s*", "", item).strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ -]*", item):
            actions.append(item)
    return actions


def scan_file(root: Path, path: Path, text: str) -> List[Candidate]:
    candidates: List[Candidate] = []

    patterns = [
        (
            "python_def",
            re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.MULTILINE),
            lambda m: (m.group(1), f"{m.group(1)}({m.group(2)})", ""),
        ),
        (
            "signature_literal",
            re.compile(r"['\"]sig['\"]\s*:\s*['\"]([^'\"]+)['\"]"),
            lambda m: (m.group(1).split("(", 1)[0], m.group(1), "registry signature"),
        ),
        (
            "registered_function",
            re.compile(r"\bregister_function\s*\(\s*['\"]([A-Za-z_]\w*)['\"]"),
            lambda m: (m.group(1), "", "safe-exec or function registry"),
        ),
        (
            "structured_tool",
            re.compile(
                r"StructuredTool\.from_function\s*\(\s*([A-Za-z_]\w*)(?:[^)]*?\bname\s*=\s*['\"]([A-Za-z_]\w*)['\"])?",
                re.DOTALL,
            ),
            lambda m: (m.group(2) or m.group(1), "", f"callable={m.group(1)}"),
        ),
        (
            "tool_decorator",
            re.compile(r"^\s*@tool(?:\(|\s*$).*?\n\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.MULTILINE | re.DOTALL),
            lambda m: (m.group(1), f"{m.group(1)}({m.group(2)})", "decorated tool"),
        ),
        (
            "bridge_command",
            re.compile(r"\bcommand\s*=\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"),
            lambda m: (m.group(1), "", "runtime bridge command"),
        ),
        (
            "tool_name",
            re.compile(r"\btool_name\s*=\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"),
            lambda m: (m.group(1), "", "runtime tool name"),
        ),
        (
            "prompt_import",
            re.compile(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+([A-Za-z0-9_, \t]+)"),
            None,
        ),
        (
            "action_dict_key",
            re.compile(r"^\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*:\s*\{", re.MULTILINE),
            lambda m: (m.group(1), "", "dictionary/registry key"),
        ),
        (
            "pddl_action",
            re.compile(r"\(:action\s+([A-Za-z_][A-Za-z0-9_-]*)", re.IGNORECASE),
            lambda m: (m.group(1), f"(:action {m.group(1)} ...)", "PDDL action operator"),
        ),
    ]

    for surface, pattern, builder in patterns:
        for match in pattern.finditer(text):
            if surface == "prompt_import":
                module = match.group(1)
                if module.split(".", 1)[0] in COMMON_IMPORT_MODULES:
                    continue
                imported = [clean_name(item) for item in match.group(2).split(",")]
                for item in imported:
                    if re.fullmatch(r"[A-Za-z_]\w*", item):
                        candidates.append(
                            Candidate(
                                name=item,
                                surface=surface,
                                file=str(path.relative_to(root)) if path != root else path.name,
                                line=line_number(text, match.start()),
                                raw=match.group(0).strip()[:500],
                                hint=f"imported from {module}",
                            )
                        )
                continue

            name, signature, hint = builder(match)  # type: ignore[misc]
            add_candidate(
                candidates,
                root=root,
                path=path,
                text=text,
                match=match,
                name=name,
                surface=surface,
                signature=signature,
                hint=hint,
            )

    allowed_pattern = re.compile(
        r"(?:Allowed actions|Available actions|Available tools|Allowed tools)\s*:\s*([^\n]+)",
        re.IGNORECASE,
    )
    for match in allowed_pattern.finditer(text):
        for action in split_action_list(match.group(1)):
            candidates.append(
                Candidate(
                    name=action,
                    surface="allowed_action_text",
                    file=str(path.relative_to(root)) if path != root else path.name,
                    line=line_number(text, match.start()),
                    raw=match.group(0).strip()[:500],
                    hint="textual action/tool vocabulary",
                )
            )

    for fence_offset, code in extract_code_fences(text):
        call_pattern = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")
        for match in call_pattern.finditer(code):
            name = match.group(1)
            if name in {"if", "for", "while", "print", "len", "range", "str", "list", "dict"}:
                continue
            candidates.append(
                Candidate(
                    name=name,
                    surface="code_fence_call",
                    file=str(path.relative_to(root)) if path != root else path.name,
                    line=line_number(text, fence_offset + match.start()),
                    raw=match.group(0).strip(),
                    hint="call in markdown/text code fence",
                )
            )

    return candidates


def dedupe(candidates: Iterable[Candidate]) -> List[Candidate]:
    seen = set()
    result: List[Candidate] = []
    for item in candidates:
        key = (item.name, item.surface, item.file, item.line, item.signature)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def render_markdown(candidates: List[Candidate]) -> str:
    lines = [
        "# Candidate Planning Actions",
        "",
        "This is an over-collected candidate list. Verify which items belong in the final unified action list.",
        "",
        "| Name | Surface | Source | Signature | Hint |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in candidates:
        source = f"{item.file}:{item.line}"
        signature = item.signature.replace("|", "\\|")
        hint = item.hint.replace("|", "\\|")
        lines.append(f"| `{item.name}` | {item.surface} | `{source}` | `{signature}` | {hint} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Target embodied planning agent repository root or file to scan.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--max-file-bytes", type=int, default=1_000_000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    all_candidates: List[Candidate] = []
    for path in iter_files(root, args.max_file_bytes):
        text = read_text(path)
        if not text:
            continue
        all_candidates.extend(scan_file(root if root.is_dir() else root.parent, path, text))

    candidates = dedupe(all_candidates)
    candidates.sort(key=lambda item: (item.file, item.line, item.surface, item.name))

    if args.format == "json":
        print(json.dumps([asdict(item) for item in candidates], ensure_ascii=False, indent=2))
    else:
        print(render_markdown(candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

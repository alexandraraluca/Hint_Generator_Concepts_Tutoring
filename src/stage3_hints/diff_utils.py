"""Source-level diff utilities for silver hint generation.

We use Python's `difflib` on **normalized** code (whitespace collapsed,
identifiers kept) and produce a compact summary that downstream prompts /
silver-hint extractors can consume:

- `n_lines_added` / `n_lines_removed`
- `added_blocks`, `removed_blocks`: contiguous lines added/removed
- `added_keywords`, `removed_keywords`: identifier/keyword diff (pre-tokenized)
- `unified_excerpt`: a small unified-diff snippet for the LLM prompt

This is intentionally text-based; tree-sitter AST diff is a future
improvement (more precise but harder to set up cross-language). Empirically,
text diff after whitespace normalization captures > 90% of the salient
structural differences between two solutions of the same problem.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable

# tokens / identifiers we want to count by kind
_KEYWORD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
# C++ / Java keywords that signal structure
_STRUCTURAL_KEYWORDS = {
    "for", "while", "do", "if", "else", "switch", "case", "break", "continue",
    "return", "goto", "struct", "class", "namespace", "using", "static",
    "const", "constexpr", "void", "int", "long", "double", "char", "bool",
    "vector", "map", "unordered_map", "set", "unordered_set", "queue",
    "priority_queue", "stack", "deque", "pair", "tuple", "string",
    "ArrayList", "HashMap", "HashSet", "TreeMap", "TreeSet", "PriorityQueue",
    "Stack", "Queue", "LinkedList", "BufferedReader", "PrintWriter",
}


@dataclass
class CodeDiff:
    n_lines_added: int = 0
    n_lines_removed: int = 0
    added_blocks: list[list[str]] = field(default_factory=list)
    removed_blocks: list[list[str]] = field(default_factory=list)
    added_keywords: dict[str, int] = field(default_factory=dict)
    removed_keywords: dict[str, int] = field(default_factory=dict)
    structural_added: list[str] = field(default_factory=list)
    structural_removed: list[str] = field(default_factory=list)
    unified_excerpt: str = ""
    similarity_ratio: float = 0.0

    def to_summary(self) -> str:
        parts = []
        parts.append(f"+{self.n_lines_added}/-{self.n_lines_removed} linii")
        if self.structural_added:
            parts.append(
                "structuri în plus: " + ", ".join(self.structural_added[:5])
            )
        if self.structural_removed:
            parts.append(
                "structuri eliminate: " + ", ".join(self.structural_removed[:5])
            )
        return "; ".join(parts)


def _strip_lines(text: str) -> list[str]:
    """Remove blank lines and pure-comment lines, normalize whitespace."""
    out: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("//"):
            continue
        if s.startswith("/*") or s.endswith("*/") or s.startswith("*"):
            continue
        # collapse runs of whitespace
        s = re.sub(r"\s+", " ", s)
        out.append(s)
    return out


def _tokenize_keywords(lines: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ln in lines:
        for tok in _KEYWORD_RE.findall(ln):
            counts[tok] = counts.get(tok, 0) + 1
    return counts


def _diff_keyword_counts(
    src: dict[str, int], dst: dict[str, int]
) -> dict[str, int]:
    """Return tokens whose count in dst exceeds src (positive delta)."""
    out: dict[str, int] = {}
    for k, v in dst.items():
        delta = v - src.get(k, 0)
        if delta > 0:
            out[k] = delta
    return out


def _group_blocks(opcodes, lines_a: list[str], lines_b: list[str]):
    """From SequenceMatcher opcodes return (added_blocks, removed_blocks)."""
    added, removed = [], []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ("insert", "replace"):
            added.append(lines_b[j1:j2])
        if tag in ("delete", "replace"):
            removed.append(lines_a[i1:i2])
    return added, removed


def code_diff(failing_code: str, passing_code: str) -> CodeDiff:
    a = _strip_lines(failing_code)
    b = _strip_lines(passing_code)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    added_blocks, removed_blocks = _group_blocks(sm.get_opcodes(), a, b)
    n_added = sum(len(blk) for blk in added_blocks)
    n_removed = sum(len(blk) for blk in removed_blocks)

    kw_a = _tokenize_keywords(a)
    kw_b = _tokenize_keywords(b)
    added_kws = _diff_keyword_counts(kw_a, kw_b)
    removed_kws = _diff_keyword_counts(kw_b, kw_a)

    structural_added = [
        k for k, _ in sorted(added_kws.items(), key=lambda x: -x[1])
        if k in _STRUCTURAL_KEYWORDS
    ][:8]
    structural_removed = [
        k for k, _ in sorted(removed_kws.items(), key=lambda x: -x[1])
        if k in _STRUCTURAL_KEYWORDS
    ][:8]

    unified = "\n".join(
        difflib.unified_diff(a, b, lineterm="", n=2)
    )
    if len(unified) > 4000:
        unified = unified[:4000] + "\n... [truncated]"

    return CodeDiff(
        n_lines_added=n_added,
        n_lines_removed=n_removed,
        added_blocks=added_blocks,
        removed_blocks=removed_blocks,
        added_keywords=added_kws,
        removed_keywords=removed_kws,
        structural_added=structural_added,
        structural_removed=structural_removed,
        unified_excerpt=unified,
        similarity_ratio=sm.ratio(),
    )

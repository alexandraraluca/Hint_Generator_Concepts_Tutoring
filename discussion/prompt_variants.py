"""Prompt variants for ablation studies (discussion/ only).

Wraps the production builders from ``src.stage3_hints.prompt_builder`` without
modifying them. Ablation-specific variants live here.
"""

from __future__ import annotations

import textwrap
from typing import Any

from src.stage3_hints.prompt_builder import (
    build_silver_pair_user_prompt,
    build_system_prompt,
    build_system_prompt_silver,
    build_user_prompt,
)

# Re-export production builders for baseline variants.
__all__ = [
    "PROMPT_VARIANTS",
    "build_prompts",
]


def _bootstrap_no_pitfalls(
    *,
    problem_meta: dict[str, Any],
    statement_excerpt: str,
    failing_code: str,
    verdict: str,
    issues: list[str],
    valid_concept_ids: list[str] | None,
) -> tuple[str, str]:
    base = build_user_prompt(
        problem_meta=problem_meta,
        statement_excerpt=statement_excerpt,
        failing_code=failing_code,
        verdict=verdict,
        issues=issues,
        valid_concept_ids=valid_concept_ids,
    )
    # Remove pitfalls block inserted by production builder.
    lines = base.splitlines()
    out: list[str] = []
    skip = False
    for line in lines:
        if "Capcane tipice ale problemei:" in line:
            skip = True
            continue
        if skip:
            if line.strip().startswith("Enunț"):
                skip = False
                out.append(line)
            continue
        out.append(line)
    return build_system_prompt(), "\n".join(out).strip()


def _bootstrap_no_concepts(
    *,
    problem_meta: dict[str, Any],
    statement_excerpt: str,
    failing_code: str,
    verdict: str,
    issues: list[str],
    valid_concept_ids: list[str] | None,
) -> tuple[str, str]:
    return build_system_prompt(), build_user_prompt(
        problem_meta=problem_meta,
        statement_excerpt=statement_excerpt,
        failing_code=failing_code,
        verdict=verdict,
        issues=issues,
        valid_concept_ids=None,
    )


def _silver_full_diff(
    *,
    problem_meta: dict[str, Any],
    statement_excerpt: str,
    failing_code: str,
    reference_passing_code: str,
    verdict: str,
    issues: list[str],
    valid_concept_ids: list[str] | None,
    codebert_similarity: float,
    diff_summary: str,
    diff_unified_excerpt: str,
    passing_file_hint: str,
) -> tuple[str, str]:
    """Silver + rezumat textual + unified diff + cod referință 100p."""
    base_user = build_silver_pair_user_prompt(
        problem_meta=problem_meta,
        statement_excerpt=statement_excerpt,
        failing_code=failing_code,
        reference_passing_code=reference_passing_code,
        verdict=verdict,
        issues=issues,
        valid_concept_ids=valid_concept_ids,
        codebert_similarity=codebert_similarity,
        diff_summary=diff_summary,
        diff_unified_excerpt=diff_unified_excerpt,
        passing_file_hint=passing_file_hint,
    )
    extra = textwrap.dedent(
        f"""
        Diff unified (extras, cod normalizat):
        ```diff
        {diff_unified_excerpt[:2200]}
        ```

        Cod referință 100 pct ({passing_file_hint}):
        ```code
        {reference_passing_code[:4500]}
        ```
        """
    ).strip()
    user = base_user + "\n\n" + extra
    return build_system_prompt_silver(), user


def _silver_no_passing(
    *,
    problem_meta: dict[str, Any],
    statement_excerpt: str,
    failing_code: str,
    reference_passing_code: str,
    verdict: str,
    issues: list[str],
    valid_concept_ids: list[str] | None,
    codebert_similarity: float,
    diff_summary: str,
    diff_unified_excerpt: str,
    passing_file_hint: str,
) -> tuple[str, str]:
    """Silver + ambele diff-uri (text + unified), fără blocul de cod 100p."""
    base_user = build_silver_pair_user_prompt(
        problem_meta=problem_meta,
        statement_excerpt=statement_excerpt,
        failing_code=failing_code,
        reference_passing_code=reference_passing_code,
        verdict=verdict,
        issues=issues,
        valid_concept_ids=valid_concept_ids,
        codebert_similarity=codebert_similarity,
        diff_summary=diff_summary,
        diff_unified_excerpt=diff_unified_excerpt,
        passing_file_hint=passing_file_hint,
    )
    extra = textwrap.dedent(
        f"""
        Diff unified (extras, cod normalizat):
        ```diff
        {diff_unified_excerpt[:2200]}
        ```
        """
    ).strip()
    user = base_user + "\n\n" + extra
    return build_system_prompt_silver(), user


PROMPT_VARIANTS: dict[str, str] = {
    "bootstrap": "Bootstrap (prompt producție)",
    "silver": "Silver (prompt producție)",
    "bootstrap_no_pitfalls": "Bootstrap fără common_pitfalls",
    "bootstrap_no_concepts": "Bootstrap fără concepts_dag",
    "silver_full_diff": "Silver + diff textual + unified + cod 100p",
    "silver_no_passing": "Silver + diff textual + unified, fără cod 100p",
}

_BOOTSTRAP_VARIANTS = frozenset(
    {"bootstrap", "bootstrap_no_pitfalls", "bootstrap_no_concepts"}
)


def build_prompts(
    variant: str,
    *,
    problem_meta: dict[str, Any],
    statement_excerpt: str,
    failing_code: str,
    verdict: str,
    issues: list[str],
    valid_concept_ids: list[str] | None,
    reference_passing_code: str = "",
    codebert_similarity: float = 0.0,
    diff_summary: str = "",
    diff_unified_excerpt: str = "",
    passing_file_hint: str = "",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a named ablation variant."""
    if variant == "bootstrap":
        return build_system_prompt(), build_user_prompt(
            problem_meta=problem_meta,
            statement_excerpt=statement_excerpt,
            failing_code=failing_code,
            verdict=verdict,
            issues=issues,
            valid_concept_ids=valid_concept_ids,
        )
    if variant == "bootstrap_no_pitfalls":
        return _bootstrap_no_pitfalls(
            problem_meta=problem_meta,
            statement_excerpt=statement_excerpt,
            failing_code=failing_code,
            verdict=verdict,
            issues=issues,
            valid_concept_ids=valid_concept_ids,
        )
    if variant == "bootstrap_no_concepts":
        return _bootstrap_no_concepts(
            problem_meta=problem_meta,
            statement_excerpt=statement_excerpt,
            failing_code=failing_code,
            verdict=verdict,
            issues=issues,
            valid_concept_ids=valid_concept_ids,
        )
    if variant == "silver":
        return build_system_prompt_silver(), build_silver_pair_user_prompt(
            problem_meta=problem_meta,
            statement_excerpt=statement_excerpt,
            failing_code=failing_code,
            reference_passing_code=reference_passing_code,
            verdict=verdict,
            issues=issues,
            valid_concept_ids=valid_concept_ids,
            codebert_similarity=codebert_similarity,
            diff_summary=diff_summary,
            diff_unified_excerpt=diff_unified_excerpt,
            passing_file_hint=passing_file_hint,
        )

    silver_kw = dict(
        problem_meta=problem_meta,
        statement_excerpt=statement_excerpt,
        failing_code=failing_code,
        reference_passing_code=reference_passing_code,
        verdict=verdict,
        issues=issues,
        valid_concept_ids=valid_concept_ids,
        codebert_similarity=codebert_similarity,
        diff_summary=diff_summary,
        diff_unified_excerpt=diff_unified_excerpt,
        passing_file_hint=passing_file_hint,
    )
    if variant == "silver_full_diff":
        return _silver_full_diff(**silver_kw)
    if variant == "silver_no_passing":
        return _silver_no_passing(**silver_kw)

    raise ValueError(f"unknown variant: {variant!r}")


def is_bootstrap_variant(variant: str) -> bool:
    return variant in _BOOTSTRAP_VARIANTS

"""
gap_builder.py
--------------
Builds a structured gap_object by comparing segments of a partial solution
against segments of the passing (100p) solution.

The gap_object is the KEY input that gets added to the hint generation prompt.
It tells the model WHAT to focus on, so it doesn't have to infer it from raw diff.

Gap levels (in priority order):
  intent      → student is solving the wrong problem entirely
  function    → right problem, wrong algorithm
  implementation → right algorithm, wrong execution (bugs, logic errors, output)

Usage:
  from gap_builder import build_gap_object
  gap = build_gap_object(partial_segments, passing_segments, codebert_sim)
"""

from __future__ import annotations
from typing import Any
import json


# ── Similarity thresholds (tune after running consistency checks) ──────────────
SIM_HIGH   = 0.97   # diff-based approach works well
SIM_MEDIUM = 0.85   # hybrid: segments + diff
SIM_LOW    = 0.0    # segment-only

# Threshold for embedding similarity between two operation strings.
# Below this → consider the operations divergent.
OP_EMBED_SIM_THRESHOLD = 0.50

_EMBED_MODEL = None


def _get_embed_model():
    """Load sentence-transformers model lazily (cached)."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _embedding_sim(text_a: str, text_b: str) -> float | None:
    """Cosine similarity between two texts using sentence-transformers.
    Returns None if sentence-transformers is not installed."""
    try:
        model = _get_embed_model()
        embs = model.encode([text_a, text_b], normalize_embeddings=True)
        return float(embs[0] @ embs[1])
    except Exception:
        return None


def _op_similarity(op_a: str, op_b: str) -> float:
    """Similarity between two operation strings.
    Uses embedding cosine similarity if sentence-transformers is available,
    falls back to word-overlap Jaccard otherwise."""
    sim = _embedding_sim(op_a, op_b)
    if sim is not None:
        return sim
    words_a = set(op_a.lower().split())
    words_b = set(op_b.lower().split())
    return len(words_a & words_b) / max(len(words_a | words_b), 1)


def similarity_bucket(sim: float) -> str:
    if sim >= SIM_HIGH:
        return "high"
    if sim >= SIM_MEDIUM:
        return "medium"
    return "low"


def hint_level_from_gap(gap_level: str, sim: float) -> str:
    """Map gap level + similarity to the appropriate hint level to generate."""
    if gap_level == "intent":
        return "macro"
    if gap_level == "function":
        return "structural"
    # implementation gap: finer distinction by similarity
    if sim >= SIM_HIGH:
        return "very_specific"   # near-perfect code, tiny bug
    if sim >= SIM_MEDIUM:
        return "specific"        # right algorithm, wrong logic
    return "structural"          # implementation so wrong it's a design issue


def _compare_algorithms(partial: dict, passing: dict) -> dict | None:
    """Returns a gap dict if the algorithms differ meaningfully, else None."""
    alg_p = partial.get("function", {}).get("algorithm", "").lower().strip()
    alg_r = passing.get("function", {}).get("algorithm", "").lower().strip()

    # Treat synonyms as the same
    synonyms = {
        "greedy": {"greedy", "greedy_sort", "greedy_selection"},
        "dp": {"dp", "dynamic_programming", "memoization"},
        "binary_search": {"binary_search", "binary search", "bisection"},
        "bfs": {"bfs", "breadth_first", "breadth first search"},
        "dfs": {"dfs", "depth_first", "depth first search"},
    }
    def canonical(a: str) -> str:
        for canon, variants in synonyms.items():
            if a in variants:
                return canon
        return a

    if canonical(alg_p) != canonical(alg_r):
        return {
            "algorithm_partial": alg_p,
            "algorithm_passing": alg_r,
        }
    return None


def _compare_key_ops(partial: dict, passing: dict) -> dict:
    """Compare key_operations lists between partial and passing.

    Uses embedding cosine similarity per step pair (falls back to word-overlap
    Jaccard if sentence-transformers is not installed).
    Returns a dict describing the divergence.
    """
    ops_p = partial.get("implementation", {}).get("key_operations", [])
    ops_r = passing.get("implementation", {}).get("key_operations", [])

    diverge_at = -1
    for i, (op_p, op_r) in enumerate(zip(ops_p, ops_r)):
        sim = _op_similarity(op_p, op_r)
        if sim < OP_EMBED_SIM_THRESHOLD:
            diverge_at = i
            break

    len_diff = abs(len(ops_p) - len(ops_r))

    return {
        "ops_partial": ops_p,
        "ops_passing": ops_r,
        "diverge_at_step": diverge_at,   # -1 = no clear divergence found
        "n_steps_partial": len(ops_p),
        "n_steps_passing": len(ops_r),
        "length_difference": len_diff,
    }


def _find_critical_issue(partial: dict) -> str:
    """Pick the most important issue from the partial solution's issues list.
    Prioritizes algorithmic issues over style/performance issues.
    Recognizes both Romanian and English keywords (LLM may mix languages).
    """
    issues = partial.get("implementation", {}).get("potential_issues", [])
    if not issues:
        return ""

    high_priority = [
        # Romanian
        "incorect", "greșit", "nu garantează", "nu asigură", "eșuează",
        "depășire", "depășire index", "depășire de tip", "null",
        "off by one", "off-by-one", "indice", "index",
        # English (fallback)
        "incorrect", "wrong", "not optimal", "does not guarantee",
        "does not ensure", "fails", "overflow", "out of bounds",
    ]
    medium_priority = [
        # Romanian
        "poate", "ar putea", "potențial", "lipsă", "incomplet",
        # English (fallback)
        "may", "could", "potential", "missing", "incomplete",
    ]

    def score(issue: str) -> int:
        lower = issue.lower()
        for kw in high_priority:
            if kw in lower:
                return 2
        for kw in medium_priority:
            if kw in lower:
                return 1
        return 0

    scored = sorted(issues, key=score, reverse=True)
    return scored[0] if scored else ""


def _check_intent_mismatch(partial: dict, passing: dict) -> bool:
    """Returns True if the partial solution's intent doesn't match the problem."""
    partial_intent = partial.get("intent", {})
    return not partial_intent.get("matches_problem", True)


def _check_data_structure_divergence(partial: dict, passing: dict) -> dict | None:
    """Check if key data structures differ between partial and passing."""
    ds_p = set(s.lower() for s in partial.get("function", {}).get("data_structures", []))
    ds_r = set(s.lower() for s in passing.get("function", {}).get("data_structures", []))

    # Remove very generic ones
    generic = {"arraylist", "list", "array", "string", "int[]"}
    ds_p_specific = ds_p - generic
    ds_r_specific = ds_r - generic

    missing = ds_r_specific - ds_p_specific
    extra = ds_p_specific - ds_r_specific

    if missing or extra:
        return {"missing_in_partial": list(missing), "extra_in_partial": list(extra)}
    return None


# ── Main builder ───────────────────────────────────────────────────────────────

def build_gap_object(
    partial_segments: dict[str, Any],
    passing_segments: dict[str, Any],
    codebert_sim: float,
) -> dict[str, Any]:
    """
    Compare partial vs passing segments and return a structured gap_object.

    Args:
        partial_segments: output of segment extractor for the failing code
        passing_segments: output of segment extractor for the 100p code
        codebert_sim: cosine similarity between the two code embeddings

    Returns:
        gap_object dict ready to be injected into the hint generation prompt
    """
    sim_bucket = similarity_bucket(codebert_sim)

    # ── 1. Intent mismatch → highest priority ─────────────────────────────────
    if _check_intent_mismatch(partial_segments, passing_segments):
        return {
            "gap_level": "intent",
            "missing_concept": "problem_understanding",
            "evidence_partial": partial_segments.get("intent", {}).get("text", ""),
            "evidence_passing": passing_segments.get("intent", {}).get("text", ""),
            "primary_issue": "Studentul rezolvă o altă problemă",
            "similarity_bucket": sim_bucket,
            "hint_level": "macro",
            "ds_divergence": None,
            "key_op_divergence": None,
        }

    # ── 2. Algorithm mismatch → function-level gap ────────────────────────────
    algo_gap = _compare_algorithms(partial_segments, passing_segments)
    if algo_gap:
        return {
            "gap_level": "function",
            "missing_concept": f"algoritm_gresit:{algo_gap['algorithm_partial']}_vs_{algo_gap['algorithm_passing']}",
            "evidence_partial": f"folosește {algo_gap['algorithm_partial']}",
            "evidence_passing": f"ar trebui să folosească {algo_gap['algorithm_passing']}",
            "primary_issue": _find_critical_issue(partial_segments),
            "similarity_bucket": sim_bucket,
            "hint_level": "structural",
            "ds_divergence": _check_data_structure_divergence(partial_segments, passing_segments),
            "key_op_divergence": None,
        }

    # ── 3. Data structure divergence → still function-level ───────────────────
    ds_gap = _check_data_structure_divergence(partial_segments, passing_segments)

    # ── 4. Implementation-level gap ───────────────────────────────────────────
    key_op_info = _compare_key_ops(partial_segments, passing_segments)
    primary_issue = _find_critical_issue(partial_segments)
    hint_level = hint_level_from_gap("implementation", codebert_sim)

    # Determine missing_concept from the primary issue text
    missing_concept = _infer_missing_concept(
        primary_issue,
        key_op_info,
        partial_segments,
        passing_segments,
    )

    return {
        "gap_level": "implementation",
        "missing_concept": missing_concept,
        "evidence_partial": _summarize_ops(key_op_info["ops_partial"]),
        "evidence_passing": _summarize_ops(key_op_info["ops_passing"]),
        "primary_issue": primary_issue,
        "diverges_at_step": key_op_info["diverge_at_step"],
        "similarity_bucket": sim_bucket,
        "hint_level": hint_level,
        "ds_divergence": ds_gap,
        "output_logic_issue": _has_output_issue(partial_segments),
    }


def _summarize_ops(ops: list[str]) -> str:
    """Collapse key_operations list into a short readable string."""
    if not ops:
        return "nicio operație extrasă"
    if len(ops) <= 3:
        return " → ".join(ops)
    return " → ".join(ops[:3]) + f" (+ {len(ops)-3} mai multe)"


def _has_output_issue(partial: dict) -> bool:
    """Check if any potential issue mentions output/result."""
    issues = partial.get("implementation", {}).get("potential_issues", [])
    output_keywords = [
        # Romanian
        "ieșire", "rezultat", "scrie", "afișeaz", "final", "off by one", "scăzut",
        # English (fallback)
        "output", "result", "write", "print", "subtract",
    ]
    return any(
        any(kw in issue.lower() for kw in output_keywords)
        for issue in issues
    )


def _infer_missing_concept(
    primary_issue: str,
    key_op_info: dict,
    partial: dict,
    passing: dict,
) -> str:
    """Infer a short concept label from the available signals.
    This becomes the 'missing_concept' field in the gap object.
    Recognizes both Romanian and English keywords.
    """
    issue_lower = primary_issue.lower()

    if (
        "overflow" in issue_lower or "depășire" in issue_lower
        or ("int" in issue_lower and "long" in issue_lower)
    ):
        return "integer_overflow"
    if (
        "off by one" in issue_lower or "off-by-one" in issue_lower
        or "subtract 1" in issue_lower or "scăzut cu 1" in issue_lower
        or "minus 1" in issue_lower
    ):
        return "off_by_one_output"
    if (
        "index out of bounds" in issue_lower or "out of bounds" in issue_lower
        or "depășire index" in issue_lower or "indexare" in issue_lower
    ):
        return "index_boundary"
    if (
        ("minimum" in issue_lower or "minim" in issue_lower)
        and ("all" in issue_lower or "group" in issue_lower or "grup" in issue_lower or "toate" in issue_lower)
    ):
        return "group_minimum_upgrade"
    if (
        "not optimal" in issue_lower or "does not guarantee" in issue_lower
        or "nu garantează" in issue_lower or "suboptimal" in issue_lower
    ):
        return "suboptimal_greedy_selection"
    if (
        "output" in issue_lower or "result" in issue_lower
        or "ieșire" in issue_lower or "rezultat" in issue_lower
    ):
        return "incorrect_output_extraction"
    if (
        ("sort" in issue_lower or "sortare" in issue_lower)
        and ("order" in issue_lower or "ordine" in issue_lower or "criteriu" in issue_lower)
    ):
        return "wrong_sort_criterion"
    if (
        "budget" in issue_lower or "cost" in issue_lower
        or "buget" in issue_lower or "cost" in issue_lower
    ):
        return "budget_tracking_error"
    if key_op_info["diverge_at_step"] == 0:
        return "wrong_initial_step"
    if key_op_info["length_difference"] > 2:
        return "missing_algorithmic_steps"

    return "implementation_logic_error"


# ── Format for prompt injection ────────────────────────────────────────────────

def format_gap_for_prompt(gap: dict[str, Any]) -> str:
    """Format the gap_object as a prompt block to inject into the hint generation prompt.
    Keeps it concise — the model doesn't need all fields, just the key signals.
    All labels are in Romanian to match the hint generation system.
    """
    lines = [
        "── ANALIZA DIFERENȚEI ──",
        f"Nivel gap:              {gap['gap_level']}",
        f"Concept lipsă:          {gap['missing_concept']}",
        f"Nivel hint:             {gap['hint_level']}",
        f"Similaritate cod:       {gap['similarity_bucket']}",
        "",
        f"Abordare parțială:      {gap.get('evidence_partial', 'N/A')}",
        f"Abordare corectă (100p):{gap.get('evidence_passing', 'N/A')}",
    ]
    if gap.get("primary_issue"):
        lines.append(f"Problemă principală:    {gap['primary_issue']}")
    if gap.get("diverges_at_step", -1) >= 0:
        lines.append(f"Divergență la pasul:    {gap['diverges_at_step'] + 1}")
    if gap.get("output_logic_issue"):
        lines.append("Logica ieșirii:         de asemenea incorectă")
    if gap.get("ds_divergence"):
        ds = gap["ds_divergence"]
        if ds.get("missing_in_partial"):
            lines.append(f"Structuri de date lipsă:{ds['missing_in_partial']}")
    lines.append("── SFÂRȘIT ANALIZĂ ──")
    return "\n".join(lines)


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo using the segments from the consistency check output

    partial_40 = {
        "intent": {"text": "compute max coins per hour after spending budget", "matches_problem": True, "confidence": "high"},
        "function": {"algorithm": "greedy", "data_structures": ["ArrayList"], "complexity": "O(n^2)", "confidence": "medium"},
        "implementation": {
            "key_operations": [
                "Sort the list of computers by current processing capacity (P).",
                "In each budget iteration, identify all computers with the minimal P, sum their upgrade costs (U), and increment their P by 1.",
                "Subtract the summed cost from the remaining budget if affordable.",
                "After exhausting the budget, output the minimal P among all computers minus one.",
            ],
            "potential_issues": [
                "The final output subtracts 1 from the minimal P, yielding an incorrect result.",
                "The loop breaks only when the required sum is not strictly less than the budget, so a case where required sum equals budget is not handled.",
                "The auxiliary list subComputers is populated but never used.",
                "Sorting the entire list in every iteration is inefficient.",
            ],
            "confidence": "high",
        },
    }

    partial_92 = {
        "intent": {"text": "Maximize coins per hour by buying upgrades greedily", "matches_problem": True, "confidence": "high"},
        "function": {"algorithm": "greedy", "data_structures": ["ArrayList", "Collections.sort"], "complexity": "unknown", "confidence": "medium"},
        "implementation": {
            "key_operations": [
                "sort the list of computers by current upgrades and cost per upgrade",
                "iteratively purchase upgrades for the computer with the lowest current upgrades, cycling through the list",
                "decrement budget and increment that computer's upgrades",
                "write the result by comparing the upgrades of the current and next computer",
            ],
            "potential_issues": [
                "Incorrect handling of nextPos when i reaches the end of the list",
                "Possible index out of bounds when accessing list.get(nextPos)",
                "The algorithm does not guarantee maximizing the minimum upgrades across all computers",
                "The output logic compares only two adjacent computers instead of the minimum across all",
            ],
            "confidence": "high",
        },
    }

    passing_100 = {
        "intent": {"text": "Maximize minimum coinsPerHour across all calculators", "matches_problem": True, "confidence": "high"},
        "function": {"algorithm": "greedy", "data_structures": ["ArrayList", "Collections.sort"], "complexity": "O(n*k)", "confidence": "medium"},
        "implementation": {
            "key_operations": [
                "sort calculators by current coinsPerHour",
                "determine the current minimum coinsPerHour",
                "upgrade all calculators that have this minimum while affordable",
                "deduct upgrade cost from available money and increment their coinsPerHour",
                "repeat until no further upgrades are possible",
                "output the last achieved minimum coinsPerHour",
            ],
            "potential_issues": [],
            "confidence": "medium",
        },
    }

    print("=" * 60)
    print("GAP: 40p vs 100p  (sim=0.72)")
    print("=" * 60)
    gap_40 = build_gap_object(partial_40, passing_100, codebert_sim=0.72)
    print(json.dumps(gap_40, indent=2, ensure_ascii=False))
    print()
    print("PROMPT BLOCK:")
    print(format_gap_for_prompt(gap_40))

    print()
    print("=" * 60)
    print("GAP: 92p vs 100p  (sim=0.97)")
    print("=" * 60)
    gap_92 = build_gap_object(partial_92, passing_100, codebert_sim=0.97)
    print(json.dumps(gap_92, indent=2, ensure_ascii=False))
    print()
    print("PROMPT BLOCK:")
    print(format_gap_for_prompt(gap_92))
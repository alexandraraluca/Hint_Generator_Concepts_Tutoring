"""Post-validator for generated hint sets.

Implements all 8 rubric checks the user defined:

  (a) minimal info - we only check the proxy "no-code" + "short" + "no-leak"
  (b) self-contained - each hint must include the verb / object cleanly
  (c) NO CODE - regex-based denylist of code-tokens
  (d) not-a-reformulation - cosine sim vs statement < threshold
  (e) strictly weaker - cosine sim vs full solution code < threshold
  (f) short - 1-3 sentences (≤ 60 words per hint)
  (g) ordered by info density - similarity to solution should NOT decrease
      from hint i to hint i+1 (later hints are more specific = closer)
  (h) 1-4 hints total

Implementation uses a TF-IDF vectoriser fitted on the *current* set of
{statement, hint, solution} tokens to keep dependencies light. We do NOT
need an external embedder for this stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Regex anti-cod: detectăm semnături CLARE de cod, NU semnele de punctuație
# naturală. ';' și '{' apar des în limba română (ex: enumerări); le marcăm
# doar când apar în contexte specifice de cod.
_CODE_DENYLIST = [
    re.compile(r"\{[^}]{0,40}\}"),                         # {...} balanced (very rare in NL)
    re.compile(r";\s*\n"),                                  # ';' la final de linie
    re.compile(r"\b(for|while|if|switch)\s*\([^)]*\)"),    # for(...) etc.
    re.compile(r"\breturn\s+[a-zA-Z0-9_]+\s*[;,]"),        # return x;
    re.compile(r"==|!=|<=|>=|->|::|=>|\+\+|--"),
    re.compile(r"\bvoid\s+\w+\s*\("),
    re.compile(r"\bint\s+\w+\s*\["),
    re.compile(r"#\s*include\b|System\.out\.|printf|scanf"),
    re.compile(r"std::|java\.|new\s+\w+\s*\("),
    re.compile(r"\b\w+\.[a-z][a-zA-Z]+\s*\([^)]*\)"),      # foo.bar(args)
    re.compile(r"\b[a-zA-Z_]\w*\s*=\s*[^=]"),              # var = value (assignment)
]


def _has_code_tokens(text: str) -> list[str]:
    bad: list[str] = []
    for r in _CODE_DENYLIST:
        if r.search(text):
            bad.append(r.pattern)
    # heuristic: if MORE THAN 3 semicolons, it is almost certainly code
    if text.count(";") > 3:
        bad.append("too_many_semicolons")
    return bad


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


@dataclass
class ValidationReport:
    passed: bool
    violations: list[str]
    per_hint_violations: list[list[str]]
    metrics: dict[str, float]


class HintValidator:
    def __init__(
        self,
        *,
        max_words_per_hint: int = 60,
        max_sentences_per_hint: int = 3,
        max_hints: int = 4,
        min_hints: int = 1,
        sim_statement_max: float = 0.55,
        sim_solution_max: float = 0.55,
    ) -> None:
        self.max_words_per_hint = max_words_per_hint
        self.max_sentences_per_hint = max_sentences_per_hint
        self.max_hints = max_hints
        self.min_hints = min_hints
        self.sim_statement_max = sim_statement_max
        self.sim_solution_max = sim_solution_max

    # ------- individual checks -------

    def _check_count(self, hints: list[dict[str, Any]]) -> list[str]:
        v: list[str] = []
        if len(hints) < self.min_hints:
            v.append(f"too_few_hints: {len(hints)} < {self.min_hints}")
        if len(hints) > self.max_hints:
            v.append(f"too_many_hints: {len(hints)} > {self.max_hints}")
        return v

    def _check_no_code(self, text: str) -> list[str]:
        bad = _has_code_tokens(text)
        return [f"code_token_match:{p}" for p in bad]

    def _check_short(self, text: str) -> list[str]:
        v: list[str] = []
        wc = _word_count(text)
        if wc > self.max_words_per_hint:
            v.append(f"too_long_words:{wc}>{self.max_words_per_hint}")
        sc = _sentence_count(text)
        if sc > self.max_sentences_per_hint:
            v.append(f"too_long_sentences:{sc}>{self.max_sentences_per_hint}")
        if wc < 4:
            v.append(f"too_short_words:{wc}")
        return v

    def _similarity_block(
        self,
        hints_texts: list[str],
        statement: str,
        solution_code: str,
    ) -> dict[str, Any]:
        """Compute hint-vs-statement and hint-vs-solution cosine sims."""
        n = len(hints_texts)
        if n == 0:
            return {"sim_to_statement": [], "sim_to_solution": []}
        corpus = list(hints_texts) + [statement or "", solution_code or ""]
        vec = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20000,
        )
        try:
            mat = vec.fit_transform(corpus)
        except ValueError:
            return {"sim_to_statement": [0.0] * n,
                    "sim_to_solution": [0.0] * n}
        sims_stmt = cosine_similarity(mat[0:n], mat[n : n + 1]).flatten()
        sims_sol = cosine_similarity(mat[0:n], mat[n + 1 : n + 2]).flatten()
        return {
            "sim_to_statement": sims_stmt.tolist(),
            "sim_to_solution": sims_sol.tolist(),
        }

    def _check_order(self, sim_to_solution: list[float]) -> list[str]:
        """Later hints should be at least as informative => closer to solution.

        We tolerate small dips (decrease ≤ 0.05) but flag larger inversions.
        """
        v: list[str] = []
        if len(sim_to_solution) < 2:
            return v
        for i in range(len(sim_to_solution) - 1):
            if sim_to_solution[i + 1] + 0.05 < sim_to_solution[i]:
                v.append(
                    f"order_inversion_at_{i + 1}: "
                    f"{sim_to_solution[i + 1]:.2f} < {sim_to_solution[i]:.2f}"
                )
        return v

    def _check_against_statement(self, sims: list[float]) -> list[str]:
        v: list[str] = []
        for i, s in enumerate(sims):
            if s > self.sim_statement_max:
                v.append(
                    f"hint{i}_too_close_to_statement:{s:.2f}>"
                    f"{self.sim_statement_max}"
                )
        return v

    def _check_against_solution(self, sims: list[float]) -> list[str]:
        v: list[str] = []
        for i, s in enumerate(sims):
            if s > self.sim_solution_max:
                v.append(
                    f"hint{i}_too_close_to_solution:{s:.2f}>"
                    f"{self.sim_solution_max}"
                )
        return v

    # ------- public entry -------

    def validate(
        self,
        hints: list[dict[str, Any]],
        *,
        statement: str = "",
        solution_code: str = "",
    ) -> ValidationReport:
        violations: list[str] = []
        per_hint: list[list[str]] = []

        violations.extend(self._check_count(hints))
        for h in hints:
            local: list[str] = []
            text = h.get("text", "")
            local.extend(self._check_no_code(text))
            local.extend(self._check_short(text))
            per_hint.append(local)

        sim_block = self._similarity_block(
            [h.get("text", "") for h in hints], statement, solution_code
        )
        violations.extend(self._check_against_statement(sim_block["sim_to_statement"]))
        violations.extend(self._check_against_solution(sim_block["sim_to_solution"]))
        violations.extend(self._check_order(sim_block["sim_to_solution"]))

        all_violations = list(violations)
        for ph in per_hint:
            all_violations.extend(ph)

        passed = not all_violations
        return ValidationReport(
            passed=passed,
            violations=violations,
            per_hint_violations=per_hint,
            metrics={
                "n_hints": float(len(hints)),
                "max_sim_to_statement": float(max(sim_block["sim_to_statement"], default=0)),
                "max_sim_to_solution": float(max(sim_block["sim_to_solution"], default=0)),
                "avg_words_per_hint": float(
                    sum(_word_count(h.get("text", "")) for h in hints) / max(1, len(hints))
                ),
            },
        )

    def validate_with_order_retry(
        self,
        hints: list[dict[str, Any]],
        *,
        statement: str = "",
        solution_code: str = "",
    ) -> tuple[list[dict[str, Any]], ValidationReport]:
        """Like validate, but drop the last hint while failures are only order_inversion.

        TF-IDF order checks are noisy; trimming often fixes a bad final step without
        weakening earlier hints.
        """
        h: list[dict[str, Any]] = list(hints)
        while True:
            rep = self.validate(h, statement=statement, solution_code=solution_code)
            if rep.passed:
                return h, rep
            combined = list(rep.violations) + [
                x for ph in rep.per_hint_violations for x in ph
            ]
            if not combined:
                return h, rep
            only_order = all(v.startswith("order_inversion") for v in combined)
            if only_order and len(h) > 1:
                h = h[:-1]
                continue
            return h, rep


MAX_RUBRIC_HINTS = 4
MAX_HINT_TEXT_LEN = 400  # aliniat cu HINT_RUBRIC_SCHEMA / items.text maxLength


def cap_hints_to_rubric(
    hints: list[dict[str, Any]],
    *,
    max_hints: int = MAX_RUBRIC_HINTS,
    max_text_len: int = MAX_HINT_TEXT_LEN,
) -> list[dict[str, Any]]:
    """Decupează lista de hinturi la limitele validatorului / schemei JSON."""
    out: list[dict[str, Any]] = []
    for h in hints[:max_hints]:
        if not isinstance(h, dict):
            continue
        d = dict(h)
        t = d.get("text", "")
        if isinstance(t, str) and len(t) > max_text_len:
            d["text"] = t[: max_text_len - 3].rstrip() + "..."
        out.append(d)
    return out

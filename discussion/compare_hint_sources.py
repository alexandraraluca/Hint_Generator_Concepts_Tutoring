"""Generate Table 1 — comparison of hint sources (bootstrap vs silver).

Reads existing JSONL outputs, re-validates silver rows missing metrics,
and writes:
  - discussion/output/TABLE1_hint_sources.md
  - discussion/output/TABLE1_hint_sources.csv

Usage (from repo root):
  python discussion/compare_hint_sources.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import read_json, read_jsonl
from src.common.paths import ANNOTATIONS_DIR, HINTS_DIR, PROCESSED_DIR
from src.stage3_hints.validator import HintValidator

OUT_DIR = Path(__file__).resolve().parent / "output"
PACKETS_DIR = PROCESSED_DIR / "packets"
_FILE_RE = re.compile(
    r"^(?P<anon>anon_\d+)_(?P<score>\d+(?:\.\d+)?)\.(?P<ext>cpp|java)$",
    re.IGNORECASE,
)


def _load_jsonl(name: str) -> list[dict[str, Any]]:
    p = HINTS_DIR / name
    if not p.exists():
        return []
    return list(read_jsonl(p))


def _statement(problem_id: str) -> str:
    p = PACKETS_DIR / f"{problem_id}.json"
    if not p.exists():
        return ""
    try:
        return read_json(p).get("statement_text", "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _gold_solution(problem_id: str) -> str:
    p = PACKETS_DIR / f"{problem_id}.json"
    if not p.exists():
        return ""
    try:
        reps = read_json(p).get("representative_solutions", []) or []
        return reps[0]["code"] if reps else ""
    except Exception:  # noqa: BLE001
        return ""


def _failing_code(row: dict[str, Any]) -> str:
    prob = row.get("problem_id", "")
    sub = row.get("submission_name") or ""
    if not sub:
        return ""
    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    meta = next((p for p in problems if p["problem_id"] == prob), None)
    if not meta:
        return ""
    folder = (
        ROOT
        / "data"
        / "raw"
        / "solutions"
        / "solutions"
        / f"{meta['year']}_{meta['pid']}"
        / sub
    )
    if folder.exists():
        return folder.read_text(encoding="utf-8", errors="replace")
    return ""


def _violation_tag(v: str) -> str:
    return v.split(":", 1)[0]


def _normal_hints(hints: Any) -> list[dict[str, Any]]:
    if not isinstance(hints, list):
        return []
    return [h for h in hints if isinstance(h, dict) and h.get("text")]


def _enrich_metrics(
    rows: list[dict[str, Any]], validator: HintValidator
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("validator_metrics"):
            out.append(r)
            continue
        hints = _normal_hints(r.get("hints"))
        if not hints:
            out.append(r)
            continue
        stmt = _statement(r.get("problem_id", ""))
        sol = _failing_code(r) or _gold_solution(r.get("problem_id", ""))
        report = validator.validate(hints, statement=stmt, solution_code=sol)
        r = dict(r)
        r["validator_passed"] = report.passed
        r["validator_violations"] = report.violations
        r["validator_metrics"] = report.metrics
        out.append(r)
    return out


def _analyze_source(
    label: str,
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    validator: HintValidator,
) -> dict[str, Any]:
    all_rows = _enrich_metrics(valid_rows + invalid_rows, validator)
    n_total = len(all_rows)
    n_pass = sum(1 for r in all_rows if r.get("validator_passed"))
    n_hints: list[float] = []
    words: list[float] = []
    sim_stmt: list[float] = []
    sim_sol: list[float] = []
    violations = Counter()

    for r in all_rows:
        m = r.get("validator_metrics") or {}
        if "n_hints" in m:
            n_hints.append(float(m["n_hints"]))
        if "avg_words_per_hint" in m:
            words.append(float(m["avg_words_per_hint"]))
        if "max_sim_to_statement" in m:
            sim_stmt.append(float(m["max_sim_to_statement"]))
        if "max_sim_to_solution" in m:
            sim_sol.append(float(m["max_sim_to_solution"]))
        for v in r.get("validator_violations") or []:
            violations[_violation_tag(v)] += 1
        if r.get("_error"):
            violations["llm_error"] += 1
        if r.get("_schema_errors"):
            violations["schema"] += 1

    problems = {r.get("problem_id") for r in all_rows if r.get("problem_id")}
    top_viol = violations.most_common(3)
    top_viol_str = ", ".join(f"{k} ({c})" for k, c in top_viol) if top_viol else "—"

    return {
        "source": label,
        "n_attempts": n_total,
        "n_valid": n_pass,
        "pass_rate_pct": 100.0 * n_pass / max(1, n_total),
        "n_problems": len(problems),
        "avg_hints": float(np.mean(n_hints)) if n_hints else 0.0,
        "avg_words_per_hint": float(np.mean(words)) if words else 0.0,
        "median_sim_statement": float(np.median(sim_stmt)) if sim_stmt else 0.0,
        "median_sim_solution": float(np.median(sim_sol)) if sim_sol else 0.0,
        "top_violations": top_viol_str,
    }


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def _fmt_f(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def _write_outputs(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "TABLE1_hint_sources.csv"
    headers = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    md_path = OUT_DIR / "TABLE1_hint_sources.md"
    md = [
        "# Tabel 1 — Compararea surselor de hinturi",
        "",
        "Agregare pe datele existente din `data/hints/`. Metricile de similaritate",
        "folosesc TF-IDF (1–2 grame) + cosinus față de enunț, respectiv codul",
        "submisiei. Pragul rubricii pentru similaritate este 0,55.",
        "",
        "| Sursă | Încercări | Valide | Rată validare | Probleme | Hinturi/medie | Cuvinte/hint | Mediană sim→enunț | Mediană sim→cod | Top violări |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['source']} "
            f"| {r['n_attempts']} "
            f"| {r['n_valid']} "
            f"| {_fmt_pct(r['pass_rate_pct'])} "
            f"| {r['n_problems']} "
            f"| {_fmt_f(r['avg_hints'])} "
            f"| {_fmt_f(r['avg_words_per_hint'], 1)} "
            f"| {_fmt_f(r['median_sim_statement'], 3)} "
            f"| {_fmt_f(r['median_sim_solution'], 3)} "
            f"| {r['top_violations']} |"
        )
    md += [
        "",
        "**Note:**",
        "- *Bootstrap* = `llm_bootstrap.py` (enunț + cod failing, fără pereche 100p).",
        "- *Silver* = `silver_hints.py` (pereche failing→passing același student, CodeBERT + diff).",
        "- Seturile de cazuri nu sunt identice; silver acoperă doar studenți cu traiectorie failing→100p.",
        "- `sim→enunț` / `sim→cod` = mediană `max_sim_to_statement` / `max_sim_to_solution` per set.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> int:
    validator = HintValidator()

    boot_valid = _load_jsonl("llm_bootstrap.jsonl")
    boot_invalid = _load_jsonl("llm_bootstrap_invalid.jsonl")
    silver_valid = [
        r for r in _load_jsonl("silver_diff.jsonl") if r.get("source") == "silver_diff"
    ]
    silver_invalid = [
        r for r in _load_jsonl("silver_diff_invalid.jsonl") if r.get("source") == "silver_diff"
    ]

    rows = [
        _analyze_source("Bootstrap LLM", boot_valid, boot_invalid, validator),
        _analyze_source("Silver (perechi)", silver_valid, silver_invalid, validator),
    ]
    _write_outputs(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 1.3 - filter the canonical dataset.

Rules:
- keep only `problem_id` from the official PA 2021-2024 set
  (already enforced upstream by `build_canonical.py`)
- drop rows whose problem has < MIN_SOLUTIONS_PER_PROBLEM solutions in the
  `solutions/` directory (low-signal "noise" buckets like 2021_bani with 2 files)
- attach a `language_dominant` flag for the problem (cpp/java/mixed)
- emit summary stats so we can record decisions in the report

Output:
- `data/processed/canonical_filtered.jsonl`
- `data/processed/filter_stats.json`
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.io_utils import read_json, read_jsonl, write_json, write_jsonl
from src.common.paths import (
    CANONICAL_FILTERED_JSONL,
    CANONICAL_JSONL,
    MIN_SOLUTIONS_PER_PROBLEM,
    PROBLEMS_INDEX_JSON,
    PROCESSED_DIR,
    ensure_dirs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-solutions",
        type=int,
        default=MIN_SOLUTIONS_PER_PROBLEM,
        help="Drop problems with fewer solutions in solutions/ (default %(default)s).",
    )
    args = parser.parse_args()

    ensure_dirs()

    problems_index = read_json(PROBLEMS_INDEX_JSON)
    by_pid: dict[str, dict[str, Any]] = {p["problem_id"]: p for p in problems_index}

    rows = list(read_jsonl(CANONICAL_JSONL))

    submissions_per_problem = Counter(r["problem_id"] for r in rows)

    kept_problems: dict[str, dict[str, Any]] = {}
    dropped_low_solutions: list[dict[str, Any]] = []

    for pid, meta in by_pid.items():
        n_sol = meta.get("n_solutions", 0)
        if n_sol < args.min_solutions:
            dropped_low_solutions.append(
                {
                    "problem_id": pid,
                    "n_solutions": n_sol,
                    "n_feedback": submissions_per_problem.get(pid, 0),
                    "reason": f"<{args.min_solutions} solutions on disk",
                }
            )
            continue
        # attach language dominance
        langs = meta.get("languages", [])
        lang_dominant = "mixed"
        if langs == ["cpp"]:
            lang_dominant = "cpp"
        elif langs == ["java"]:
            lang_dominant = "java"
        kept_problems[pid] = {**meta, "language_dominant": lang_dominant}

    filtered_rows = [r for r in rows if r["problem_id"] in kept_problems]

    n_filtered = write_jsonl(CANONICAL_FILTERED_JSONL, filtered_rows)

    stats = {
        "total_problems_in_index": len(problems_index),
        "kept_problems": len(kept_problems),
        "dropped_problems": len(dropped_low_solutions),
        "min_solutions_threshold": args.min_solutions,
        "total_rows_canonical": len(rows),
        "total_rows_filtered": n_filtered,
        "kept_problem_ids": sorted(kept_problems),
        "dropped": dropped_low_solutions,
    }
    write_json(PROCESSED_DIR / "filter_stats.json", stats)

    # print short summary
    print(
        f"kept {stats['kept_problems']}/{stats['total_problems_in_index']} problems; "
        f"{stats['total_rows_filtered']}/{stats['total_rows_canonical']} rows"
    )
    if dropped_low_solutions:
        print("dropped:")
        for d in dropped_low_solutions:
            print(f"  - {d['problem_id']}: {d['n_solutions']} solutions ({d['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

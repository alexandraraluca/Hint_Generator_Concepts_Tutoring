"""
similarity_check.py
--------------------
Parcurge silver_diff.jsonl și afișează distribuția embedding_similarity.

Utilizare:
  python similarity_check.py
  python similarity_check.py --threshold 0.90
  python similarity_check.py --jsonl data/hints/silver_diff.jsonl --threshold 0.85
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribuție embedding_similarity din silver_diff.jsonl")
    parser.add_argument(
        "--jsonl", type=Path, default=Path("data/hints/silver_diff.jsonl"),
        help="Calea către JSONL (default: data/hints/silver_diff.jsonl)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Pragul de similaritate (default: 0.85)",
    )
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: fișierul nu există: {args.jsonl}")
        return

    threshold = args.threshold

    below: list[dict] = []
    above: list[dict] = []
    missing: list[dict] = []

    import json
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sim = row.get("embedding_similarity")
            if sim is None:
                missing.append(row)
            elif sim < threshold:
                below.append(row)
            else:
                above.append(row)

    total = len(below) + len(above) + len(missing)

    print("=" * 60)
    print(f"SIMILARITY CHECK — {args.jsonl}")
    print("=" * 60)
    print(f"Total rânduri:              {total}")
    print(f"Fără embedding_similarity:  {len(missing)}")
    print()
    print(f"Prag ales:                  {threshold}")
    print(f"  < {threshold}  (coduri diferite):   {len(below):>4}  ({100*len(below)/max(total,1):.1f}%)")
    print(f"  >= {threshold} (coduri similare):   {len(above):>4}  ({100*len(above)/max(total,1):.1f}%)")
    print()

    # ── Distribuție pe buckets ──────────────────────────────────────────────
    buckets = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 1.01]
    counts: dict[str, int] = defaultdict(int)
    for row in below + above:
        sim = row["embedding_similarity"]
        for lo, hi in zip(buckets, buckets[1:]):
            if lo <= sim < hi:
                label = f"[{lo:.2f}, {hi:.2f})"
                counts[label] += 1
                break

    print("Distribuție pe intervale:")
    for lo, hi in zip(buckets, buckets[1:]):
        label = f"[{lo:.2f}, {hi:.2f})"
        n = counts.get(label, 0)
        bar = "█" * (n // max(1, total // 40))
        marker = " ← prag" if abs(lo - threshold) < 1e-9 else ""
        print(f"  {label}  {n:>4}  {bar}{marker}")
    print()

    # ── Breakdown per problem_id ────────────────────────────────────────────
    by_problem: dict[str, dict[str, int]] = defaultdict(lambda: {"below": 0, "above": 0})
    for row in below:
        by_problem[row["problem_id"]]["below"] += 1
    for row in above:
        by_problem[row["problem_id"]]["above"] += 1

    print(f"Breakdown per problem_id (prag={threshold}):")
    print(f"  {'problem_id':<35} {'< prag':>7}  {'>= prag':>8}")
    print("  " + "-" * 55)
    for pid in sorted(by_problem):
        b = by_problem[pid]["below"]
        a = by_problem[pid]["above"]
        print(f"  {pid:<35} {b:>7}  {a:>8}")

    # ── Cazuri cu similaritate mică (< threshold) ──────────────────────────
    if below:
        print()
        print(f"Rânduri cu similarity < {threshold} ({len(below)} total):")
        print(f"  {'submission_name':<35} {'problem_id':<30} {'sim':>6}")
        print("  " + "-" * 75)
        for row in sorted(below, key=lambda r: r.get("embedding_similarity", 0)):
            print(
                f"  {row.get('submission_name','?'):<35} "
                f"{row.get('problem_id','?'):<30} "
                f"{row.get('embedding_similarity',0):.4f}"
            )


if __name__ == "__main__":
    main()

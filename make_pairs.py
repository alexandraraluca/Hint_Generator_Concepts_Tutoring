"""
make_pairs.py
-------------
Construiește perechi (failing → passing) pentru fiecare rând din
`data/hints/silver_diff.jsonl`.

Pentru fiecare rând, scriptul:
  1. Identifică fișierul cu erori (`submission_name`) al studentului
  2. Identifică fișierul de 100p (`passing_file`) al ACELUIAȘI student
  3. Rezolvă căile pe disc (data/raw/solutions/solutions/<year>_<pid>/)
  4. Citește opțional codul ambelor fișiere
  5. Calculează scorul partial din numele fișierului
  6. Scrie totul într-un JSONL ușor de consumat downstream.

Exemple de utilizare:
  # toate perechile, fără cod (doar metadata)
  python make_pairs.py

  # toate perechile, inclusiv codul ambelor fișiere
  python make_pairs.py --include-code

  # doar o singură problemă, cu cod
  python make_pairs.py --filter-problem 2021_tema1_crypto --include-code

  # output personalizat
  python make_pairs.py --out reports/pairs_silver.jsonl --include-code
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import orjson

from src.common.io_utils import read_jsonl, write_jsonl
from src.common.paths import EXTRACTED_SOLUTIONS_DIR, HINTS_DIR

SILVER_DIFF_DEFAULT = HINTS_DIR / "silver_diff.jsonl"
SOLUTIONS_BASE_DEFAULT = EXTRACTED_SOLUTIONS_DIR / "solutions"
OUT_DEFAULT = HINTS_DIR / "pairs_silver.jsonl"

_SCORE_RE = re.compile(r"_(\d+(?:\.\d+)?)\.(?:cpp|java)$", re.IGNORECASE)


def score_from_filename(name: str) -> float | None:
    """Extrage scorul din ex: 'anon_1394_96.cpp' → 96.0"""
    m = _SCORE_RE.search(name)
    return float(m.group(1)) if m else None


def problem_id_to_folder(problem_id: str) -> str:
    """'2021_tema1_crypto' → '2021_crypto' (matches the on-disk folder layout)."""
    parts = problem_id.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse problem_id: {problem_id!r}")
    year, _tema, pid = parts
    return f"{year}_{pid}"


def safe_read(path: Path, max_bytes: int = 200_000) -> str:
    """Citește un fișier sursă fără să crape pe bytes binari."""
    try:
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def build_pair(
    row: dict[str, Any],
    solutions_base: Path,
    include_code: bool,
) -> dict[str, Any]:
    """Construiește o pereche failing/passing pentru un rând silver_diff."""
    problem_id = row["problem_id"]
    folder_name = problem_id_to_folder(problem_id)
    sol_dir = solutions_base / folder_name

    failing_name = row["submission_name"]
    passing_name = row["passing_file"]

    failing_path = sol_dir / failing_name
    passing_path = sol_dir / passing_name

    pair: dict[str, Any] = {
        "problem_id": problem_id,
        "anon_id": row.get("anon_id", ""),
        "language": row.get("language", ""),
        "verdict": row.get("verdict", ""),
        "embedding_similarity": row.get("embedding_similarity"),
        "failing": {
            "filename": failing_name,
            "path": str(failing_path),
            "exists": failing_path.exists(),
            "score": score_from_filename(failing_name),
        },
        "passing": {
            "filename": passing_name,
            "path": str(passing_path),
            "exists": passing_path.exists(),
            "score": score_from_filename(passing_name),
        },
        "hints": row.get("hints", []),
        "issues": row.get("issues", []),
        "concepts_targeted": row.get("concepts_targeted", []),
    }

    if include_code:
        pair["failing"]["code"] = safe_read(failing_path) if failing_path.exists() else ""
        pair["passing"]["code"] = safe_read(passing_path) if passing_path.exists() else ""

    return pair


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construiește perechi (failing → passing) din silver_diff.jsonl.\n"
            "Fiecare pereche conține metadata + (opțional) codul ambelor fișiere."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--silver-jsonl",
        type=Path,
        default=SILVER_DIFF_DEFAULT,
        help="JSONL de intrare (default: data/hints/silver_diff.jsonl)",
    )
    parser.add_argument(
        "--solutions-base",
        type=Path,
        default=SOLUTIONS_BASE_DEFAULT,
        help="Folder root cu soluțiile pe disc (default: data/raw/solutions/solutions)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DEFAULT,
        help="JSONL de ieșire (default: data/hints/pairs_silver.jsonl)",
    )
    parser.add_argument(
        "--include-code",
        action="store_true",
        help="Include codul sursă al ambelor fișiere în output (mărește mult fișierul)",
    )
    parser.add_argument(
        "--filter-problem",
        type=str,
        default=None,
        help="Procesează doar un singur problem_id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Procesează cel mult N rânduri (0 = toate)",
    )
    args = parser.parse_args()

    if not args.silver_jsonl.exists():
        print(f"ERROR: fișierul nu există: {args.silver_jsonl}")
        return

    print("=" * 60)
    print("MAKE PAIRS — failing ↔ passing pentru fiecare student")
    print("=" * 60)
    print(f"Input:      {args.silver_jsonl}")
    print(f"Solutions:  {args.solutions_base}")
    print(f"Output:     {args.out}")
    print(f"Include code: {args.include_code}")
    if args.filter_problem:
        print(f"Filter:     problem_id={args.filter_problem!r}")
    if args.limit:
        print(f"Limit:      {args.limit} rânduri")
    print()

    rows = list(read_jsonl(args.silver_jsonl))
    if args.filter_problem:
        rows = [r for r in rows if r["problem_id"] == args.filter_problem]
    if args.limit:
        rows = rows[: args.limit]

    pairs: list[dict[str, Any]] = []
    n_missing_failing = 0
    n_missing_passing = 0
    by_problem: dict[str, int] = {}

    for i, row in enumerate(rows, 1):
        try:
            pair = build_pair(row, args.solutions_base, args.include_code)
        except ValueError as e:
            print(f"[{i}/{len(rows)}] SKIP {row.get('submission_name', '?')}: {e}")
            continue

        if not pair["failing"]["exists"]:
            n_missing_failing += 1
            print(f"[{i}/{len(rows)}] MISSING failing: {pair['failing']['path']}")
        if not pair["passing"]["exists"]:
            n_missing_passing += 1
            print(f"[{i}/{len(rows)}] MISSING passing: {pair['passing']['path']}")

        pairs.append(pair)
        by_problem[pair["problem_id"]] = by_problem.get(pair["problem_id"], 0) + 1

    write_jsonl(args.out, pairs)

    print()
    print("=" * 60)
    print(f"REZULTATE — {len(pairs)} perechi scrise în {args.out}")
    print("=" * 60)
    print(f"Failing files lipsă pe disc: {n_missing_failing}")
    print(f"Passing files lipsă pe disc: {n_missing_passing}")
    print(f"Probleme distincte:          {len(by_problem)}")
    print()
    print("Perechi per problemă:")
    for pid in sorted(by_problem):
        print(f"  {pid:<35} {by_problem[pid]:>4}")


if __name__ == "__main__":
    main()

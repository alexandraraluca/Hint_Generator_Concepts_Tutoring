"""Pregătește datele din segments_silver_diff.jsonl pentru fine-tuning.

Citește rândurile cu ``generated_hints`` și le scrie în ``data/hints/segments.jsonl``
în exact același format ca ``data/hints/silver_diff.jsonl``.

Mapare:
  - câmpurile problemei → din rândul respectiv din segments_silver_diff.jsonl
  - ``hints``           → ``generated_hints``
  - ``source``          → ``"segments_silver_diff"``
  - rândurile fără ``generated_hints`` sunt excluse

Utilizare:
  python prepare_for_fine_tuning_segments_silver.py
  python prepare_for_fine_tuning_segments_silver.py --input reports/segments_batch/segments_silver_diff.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.io_utils import read_jsonl, write_jsonl
from src.common.paths import HINTS_DIR, ROOT, ensure_dirs

DEFAULT_INPUT = ROOT / "reports" / "segments_batch" / "segments_silver_diff.jsonl"
DEFAULT_OUTPUT = HINTS_DIR / "segments.jsonl"
SOURCE_TAG = "segments_silver_diff"

OUTPUT_KEYS = (
    "problem_id",
    "anon_id",
    "submission_name",
    "language",
    "verdict",
    "issues",
    "concepts_targeted",
    "hints",
    "source",
    "embedding_similarity",
    "passing_file",
    "validator_passed",
    "validator_violations",
)


def _as_hint_list(value: Any) -> list[dict[str, Any]]:
    if not value or not isinstance(value, list):
        return []
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Transformă un rând; returnează None dacă lipsește generated_hints."""
    generated_hints = _as_hint_list(row.get("generated_hints"))
    if not generated_hints:
        return None

    hint_validation_errors = list(row.get("hint_validation_errors") or [])

    return {
        "problem_id": row.get("problem_id", ""),
        "anon_id": row.get("anon_id", ""),
        "submission_name": row.get("submission_name", ""),
        "language": row.get("language", "unknown"),
        "verdict": row.get("verdict", "WA"),
        "issues": list(row.get("issues") or []),
        "concepts_targeted": list(row.get("concepts_targeted") or []),
        "hints": generated_hints,
        "source": SOURCE_TAG,
        "embedding_similarity": row.get("embedding_similarity"),
        "passing_file": row.get("passing_file", ""),
        "validator_passed": not hint_validation_errors,
        "validator_violations": hint_validation_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="segments_silver_diff.jsonl → data/hints/segments.jsonl (format silver_diff)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"JSONL sursă (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSONL destinație (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: fișierul nu există: {args.input}")
        return 1

    ensure_dirs()
    rows_in = list(read_jsonl(args.input))
    rows_out: list[dict[str, Any]] = []
    skipped = 0
    for row in rows_in:
        normalized = normalize_row(row)
        if normalized is None:
            skipped += 1
            continue
        rows_out.append(normalized)

    n_written = write_jsonl(args.output, rows_out)
    valid = sum(1 for r in rows_out if r.get("validator_passed"))

    print(f"Input:   {args.input} ({len(rows_in)} rows)")
    print(f"Output:  {args.output} ({n_written} rows)")
    print(f"  skipped (no generated_hints): {skipped}")
    print(f"  validator_passed=True:        {valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

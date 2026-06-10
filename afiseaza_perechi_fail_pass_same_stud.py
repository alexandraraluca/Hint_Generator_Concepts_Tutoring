"""
afiseaza_perechi.py
-------------------
Afișează în terminal numele submisiilor din fiecare pereche
(failing + passing) din data/hints/all_pairs_same_student.jsonl.

Utilizare:
  python afiseaza_perechi.py
  python afiseaza_perechi.py --jsonl path/to/other.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import read_jsonl  # noqa: E402
from src.common.paths import HINTS_DIR  # noqa: E402

# DEFAULT_JSONL = HINTS_DIR / "all_pairs_same_student.jsonl"
DEFAULT_JSONL = HINTS_DIR / "all_pairs.jsonl"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(
        description="Afișează numele submisiilor din perechile failing/passing.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help=f"Fișier JSONL (default: {DEFAULT_JSONL})",
    )
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: fișier inexistent: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    rows = list(read_jsonl(args.jsonl))
    print(f"# {args.jsonl} — {len(rows)} perechi\n")

    for row in rows:
        failing = (row.get("failing") or {}).get("submission_id", "")
        passing = (row.get("passing") or {}).get("submission_id", "")
        print(f"{failing}\t{passing}")


if __name__ == "__main__":
    main()

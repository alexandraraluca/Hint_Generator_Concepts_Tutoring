"""
afisare_enunt.py
----------------
Afișează enunțurile problemelor folosind aceeași sursă ca generate_hint_concepts.py:
  statement_text_for_problem_id(problem_id, packets_dir)
cu packets_dir implicit data/processed/packets (re-parse PDF sau fallback statement_text din packet).

Utilizare:
  python afisare_enunt.py
  python afisare_enunt.py --problem-id 2021_tema1_crypto
  python afisare_enunt.py --packets-dir data/processed/packets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import read_json  # noqa: E402
from src.common.paths import ANNOTATIONS_DIR, PROCESSED_DIR  # noqa: E402
from src.stage2_annotation.prepare_problem_packets import (  # noqa: E402
    statement_text_for_problem_id,
)

# Aceeași limită ca în build_concept_user_prompt (generate_hint_concepts.py)
PROMPT_TRUNCATE = 1500


def _problem_ids_from_annotations() -> list[str]:
    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    return sorted(p["problem_id"] for p in problems)


def _load_statement(problem_id: str, packets_dir: Path) -> tuple[str, str]:
    """Identic cu generate_hint_concepts._load_statement, dar păstrează sursa."""
    try:
        text, source = statement_text_for_problem_id(problem_id, packets_dir)
        return text or "", source or ""
    except Exception as e:  # noqa: BLE001
        return "", f"error: {e}"


def _print_statement(
    problem_id: str,
    text: str,
    source: str,
    *,
    show_prompt_excerpt: bool,
) -> None:
    print("=" * 72)
    print(f"problem_id: {problem_id}")
    print(f"Sursă:      {source or '(necunoscută)'}")
    print(f"Lungime:    {len(text)} caractere")
    if show_prompt_excerpt and len(text) > PROMPT_TRUNCATE:
        print(
            f"Notă: generate_hint_concepts folosește primele {PROMPT_TRUNCATE} "
            f"caractere în prompt."
        )
    print("-" * 72)
    if text:
        print(text)
        if show_prompt_excerpt and len(text) > PROMPT_TRUNCATE:
            print("-" * 72)
            print(f"[extras în prompt — primele {PROMPT_TRUNCATE} caractere]")
            print(text[:PROMPT_TRUNCATE])
    else:
        print("(enunț gol — lipsă packet sau PDF/cache)")
    print()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    packets_default = PROCESSED_DIR / "packets"
    parser = argparse.ArgumentParser(
        description="Afișează enunțurile problemelor (aceeași sursă ca generate_hint_concepts.py).",
    )
    parser.add_argument(
        "--packets-dir",
        type=Path,
        default=packets_default,
        help=f"Folder packets (default: {packets_default})",
    )
    parser.add_argument(
        "--problem-id",
        type=str,
        default=None,
        help="Afișează doar o problemă (implicit: toate din problems.json)",
    )
    parser.add_argument(
        "--show-prompt-excerpt",
        action="store_true",
        help=f"După enunțul complet, repetă și primele {PROMPT_TRUNCATE} caractere (ca în LLM)",
    )
    args = parser.parse_args()

    if args.problem_id:
        problem_ids = [args.problem_id]
    else:
        problem_ids = _problem_ids_from_annotations()

    print(f"Packets dir: {args.packets_dir.resolve()}")
    print(f"Probleme:    {len(problem_ids)}")
    print()

    missing = 0
    for pid in problem_ids:
        text, source = _load_statement(pid, args.packets_dir)
        if not text:
            missing += 1
        _print_statement(
            pid,
            text,
            source,
            show_prompt_excerpt=args.show_prompt_excerpt,
        )

    if missing:
        print(f"ATENȚIE: {missing}/{len(problem_ids)} probleme fără enunț.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
make_all_pairs.py
-----------------
Construiește perechi (failing ↔ 100p) CROSS-STUDENT pentru fiecare problemă,
calculează similaritatea CodeBERT și scrie rezultatul în
data/hints/all_pairs.jsonl.

Reguli (diferite față de silver_hints.py):
  - perechile NU mai sunt restricționate la același anon_id
  - oricare cod de 100p × oricare cod failing (în cadrul ACELEIAȘI limbi)
  - per problemă: max N perechi (default 15), peste toate limbile
  - eșantion aleator cu --seed (reproducibil)

Encoder CodeBERT cu cache per limbă (sols_cpp.npz / sols_java.npz) — același
mecanism ca în silver_hints, deci nu re-encodează degeaba.

Output JSONL — un rând per pereche, cu informații EXPLICITE pentru ambele
soluții (failing și 100p), fiindcă acum pot fi de la studenți diferiți:
  {
    "problem_id": "...",
    "language": "...",
    "failing": {"submission_id": "...", "anon_id": "...", "score": ...},
    "passing": {"submission_id": "...", "anon_id": "...", "score": ...},
    "embedding_similarity": 0.97...
  }

Utilizare:
  python make_all_pairs.py
  python make_all_pairs.py --max-pairs-per-problem 15
  python make_all_pairs.py --problems 2021_tema1_crypto
  python make_all_pairs.py --batch-size 16 --seed 7
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.io_utils import read_json
from src.common.paths import ANNOTATIONS_DIR, EXTRACTED_SOLUTIONS_DIR, HINTS_DIR, ensure_dirs
from src.stage3_hints.code_embeddings import CodeBERTEncoder, encode_files_with_cache

OUT_DEFAULT = HINTS_DIR / "all_pairs.jsonl"

_FILE_RE = re.compile(
    r"^(?P<anon>anon_\d+)_(?P<score>\d+(?:\.\d+)?)\.(?P<ext>cpp|java)$",
    re.IGNORECASE,
)


def _list_solutions(year: str, pid: str) -> list[tuple[Path, str, float, str]]:
    """Listare fișiere conform convenției on-disk.
    Întoarce (path, anon, score, language).
    """
    base = EXTRACTED_SOLUTIONS_DIR / "solutions" / f"{year}_{pid}"
    if not base.exists():
        return []
    out: list[tuple[Path, str, float, str]] = []
    for f in base.iterdir():
        if not f.is_file():
            continue
        m = _FILE_RE.match(f.name)
        if not m:
            continue
        out.append((f, m.group("anon"), float(m.group("score")), m.group("ext").lower()))
    return out


def _bucket(score: float) -> str:
    return "passing" if score >= 99.999 else "failing"


def _cross_student_pairs(
    failing: list[tuple[Path, str, float, str]],
    passing: list[tuple[Path, str, float, str]],
) -> list[tuple[tuple[Path, str, float, str], tuple[Path, str, float, str]]]:
    """Cartezian passing × failing în cadrul aceleiași limbi (cross-student).
    Sortare stabilă pentru reproducibilitate.
    """
    passing_sorted = sorted(passing, key=lambda x: x[0].name)
    failing_sorted = sorted(failing, key=lambda x: x[0].name)
    pairs = []
    for p in passing_sorted:
        for f in failing_sorted:
            pairs.append((f, p))
    return pairs


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perechi cross-student (failing × 100p) + similaritate CodeBERT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--problems", type=str, default=None,
        help="Filtru: lista comma-separated de problem_id.",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DEFAULT,
        help=f"JSONL de ieșire (default: {OUT_DEFAULT}).",
    )
    parser.add_argument(
        "--max-pairs-per-problem", type=int, default=15,
        help="Plafonul de perechi per problemă (peste toate limbile). Default: 15.",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Seed pentru sample-ul aleator (reproducibil).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="CodeBERT batch size (default 8).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Nu sări peste perechile deja prezente în output.",
    )
    parser.add_argument(
        "--languages", type=str, default="cpp,java",
        help="Limbi procesate (default: cpp,java).",
    )
    args = parser.parse_args()

    ensure_dirs()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    pid_filter = (
        set(s.strip() for s in args.problems.split(",")) if args.problems else None
    )
    problems = [
        p for p in problems if (pid_filter is None or p["problem_id"] in pid_filter)
    ]
    print(f"Procesez {len(problems)} probleme pe limbi: {languages}")
    print(f"Plafon: {args.max_pairs_per_problem} perechi / problemă  (seed={args.seed})")

    # ── Resume support ────────────────────────────────────────────────────────
    existing: set[tuple[str, str, str]] = set()
    existing_per_problem: dict[str, int] = defaultdict(int)
    if not args.no_resume and args.out.exists():
        with open(args.out, "rb") as fb:
            for line in fb:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = orjson.loads(line)
                except Exception:
                    continue
                fail = r.get("failing") or {}
                pas = r.get("passing") or {}
                pid = r.get("problem_id", "")
                key = (pid, fail.get("submission_id", ""), pas.get("submission_id", ""))
                existing.add(key)
                existing_per_problem[pid] += 1
        print(
            f"Resume: {len(existing)} perechi deja prezente "
            f"(across {len(existing_per_problem)} probleme)."
        )

    encoder = CodeBERTEncoder()
    written = 0
    skipped_existing = 0
    skipped_full = 0
    t0 = time.time()

    for prob in tqdm(problems, desc="problems"):
        year = prob["year"]
        pid_short = prob["pid"]
        problem_id = prob["problem_id"]

        # bugetul rămas pentru problemă (după ce-am numărat ce există)
        already = existing_per_problem.get(problem_id, 0)
        budget = max(0, args.max_pairs_per_problem - already)
        if budget == 0:
            skipped_full += 1
            continue

        all_sols = _list_solutions(year, pid_short)
        if not all_sols:
            continue

        # ── Adună perechi candidate din TOATE limbile, apoi shuffle global ────
        candidate_pairs: list[
            tuple[tuple[Path, str, float, str], tuple[Path, str, float, str], str]
        ] = []
        for lang in languages:
            f_lang = [s for s in all_sols if s[3] == lang and _bucket(s[2]) == "failing"]
            p_lang = [s for s in all_sols if s[3] == lang and _bucket(s[2]) == "passing"]
            if not f_lang or not p_lang:
                continue
            for fail_sol, pass_sol in _cross_student_pairs(f_lang, p_lang):
                key = (problem_id, fail_sol[0].name, pass_sol[0].name)
                if key in existing:
                    skipped_existing += 1
                    continue
                candidate_pairs.append((fail_sol, pass_sol, lang))

        if not candidate_pairs:
            continue

        # ── Shuffle determinist și taie la buget ──────────────────────────────
        rng = random.Random(f"{args.seed}|{problem_id}")  # deterministic per problem
        rng.shuffle(candidate_pairs)
        selected = candidate_pairs[:budget]

        # ── Encoding per limbă, doar pentru path-urile selectate ──────────────
        # Grupăm path-urile per limbă, ca să folosim cache-ul corect.
        by_lang: dict[str, list[Path]] = defaultdict(list)
        for fail_sol, pass_sol, lang in selected:
            by_lang[lang].append(fail_sol[0])
            by_lang[lang].append(pass_sol[0])
        path_to_i_per_lang: dict[str, dict[Path, int]] = {}
        emb_per_lang: dict[str, np.ndarray] = {}
        for lang, paths in by_lang.items():
            unique_paths = sorted(set(paths), key=lambda x: x.name)
            try:
                emb_arr, path_strs = encode_files_with_cache(
                    unique_paths,
                    encoder=encoder,
                    batch_size=args.batch_size,
                    cache_name=f"sols_{lang}.npz",
                )
            except Exception as e:
                print(f"[{problem_id}/{lang}] embedding error: {e!r}")
                continue
            emb_per_lang[lang] = emb_arr
            path_to_i_per_lang[lang] = {
                Path(ps).resolve(): idx for idx, ps in enumerate(path_strs)
            }

        # ── Scrie rândurile ───────────────────────────────────────────────────
        with open(args.out, "ab") as fb:
            for fail_sol, pass_sol, lang in selected:
                f_path, f_anon, f_score, _ = fail_sol
                p_path, p_anon, p_score, _ = pass_sol
                path_to_i = path_to_i_per_lang.get(lang)
                emb_arr = emb_per_lang.get(lang)
                if path_to_i is None or emb_arr is None:
                    continue
                fi = path_to_i.get(f_path.resolve())
                pi = path_to_i.get(p_path.resolve())
                if fi is None or pi is None:
                    continue
                sim = _cosine_rows(emb_arr[fi], emb_arr[pi])

                row = {
                    "problem_id": problem_id,
                    "language": lang,
                    "failing": {
                        "submission_id": f_path.name,
                        "anon_id": f_anon,
                        "score": f_score,
                    },
                    "passing": {
                        "submission_id": p_path.name,
                        "anon_id": p_anon,
                        "score": p_score,
                    },
                    "embedding_similarity": round(float(sim), 4),
                }
                fb.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
                existing.add((problem_id, f_path.name, p_path.name))
                existing_per_problem[problem_id] += 1
                written += 1

    encoder.close()
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"DONE — {written} perechi noi scrise în {args.out}")
    print("=" * 60)
    print(f"Sărite (deja existente):       {skipped_existing}")
    print(f"Probleme deja la plafon:       {skipped_full}")
    print(f"Plafon per problemă:           {args.max_pairs_per_problem}")
    print(f"Timp total:                    {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

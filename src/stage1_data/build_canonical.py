"""Stage 1.2 - build the canonical, joined dataset.

For every PA 2021-2024 problem we link:
- enunț (text extras din PDF-ul corespunzător din `statements.zip`)
- soluții (toate fișierele din `solutions/<an>_<problema>/anon_*_<scor>.<ext>`)
- evaluări checker (rândurile din `submission_feedback.jsonl` cu `assignment_id`
  în {2021..2024}_{tema1,tema2})
- structura testelor (private_tests/<problema>/...)

Output:
- `data/processed/problems_index.json` - meta de bază pentru fiecare problemă
- `data/processed/canonical.jsonl` - un rând per submisie (cod + evaluare)
- `data/processed/user_trajectories.jsonl` - submisiile fiecărui (anon_id,
  problem_id), sortate cronologic, cu `pts_progression` și `final_pts`.
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.common.io_utils import read_jsonl, write_json, write_jsonl
from src.common.paths import (
    CANONICAL_JSONL,
    EXTRACTED_SOLUTIONS_DIR,
    EXTRACTED_STATEMENTS_DIR,
    EXTRACTED_TESTS_DIR,
    PA_TEMAS,
    PA_YEARS,
    PROBLEMS_INDEX_JSON,
    RAW_FEEDBACK_JSONL,
    USER_TRAJECTORIES_JSONL,
    ensure_dirs,
)

SOL_FILENAME_RE = re.compile(
    r"^(?P<anon>anon_\d+)_(?P<sub>\d+)_(?P<score>\d+(?:\.\d+)?)\.(?P<ext>cpp|java)$",
    re.IGNORECASE,
)

# parsed once (lazy) so we do not pay PDF cost if only a subset is needed
_STATEMENT_TEXT_CACHE: dict[str, str] = {}

SUBMISSION_NAME_RE = re.compile(
    r"sb_(?P<ts>\d{4}\.\d{2}\.\d{2}__\d{2}\.\d{2}\.\d{2})_rnd(?P<rnd>\d+)"
)


def _ts_from_submission_name(name: str) -> str | None:
    m = SUBMISSION_NAME_RE.match(name or "")
    if not m:
        return None
    try:
        return datetime.strptime(m.group("ts"), "%Y.%m.%d__%H.%M.%S").isoformat()
    except ValueError:
        return None


def _statement_path_for(year: str, tema: str) -> Path | None:
    """Resolve the PDF in statements/ that matches `[PA] Tema <N> <year>.pdf`."""
    n = "1" if tema == "tema1" else "2"
    candidate = EXTRACTED_STATEMENTS_DIR / f"[PA] Tema {n} {year}.pdf"
    return candidate if candidate.exists() else None


def _load_statement_text(year: str, tema: str) -> str:
    key = f"{year}_{tema}"
    if key in _STATEMENT_TEXT_CACHE:
        return _STATEMENT_TEXT_CACHE[key]
    path = _statement_path_for(year, tema)
    if path is None:
        _STATEMENT_TEXT_CACHE[key] = ""
        return ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:  # noqa: BLE001
        print(f"warn: cannot parse {path.name}: {e}")
        text = ""
    _STATEMENT_TEXT_CACHE[key] = text
    return text


def _split_statement_per_problem(full_text: str, problems: list[str]) -> dict[str, str]:
    """Heuristic split: search for problem name followed by a header marker.

    The PDFs have a section per problem; the section title is usually the
    capitalized problem name. This is best-effort - we keep the surrounding
    text per problem and let downstream LLM annotation refine it.
    """
    out: dict[str, str] = {}
    if not full_text or not problems:
        return out
    norm = full_text
    # build regex with each problem name (case-insensitive, word boundary)
    names_sorted = sorted({p.lower() for p in problems}, key=len, reverse=True)
    pattern = re.compile(
        r"(?i)\b(" + "|".join(re.escape(n) for n in names_sorted) + r")\b"
    )
    matches = list(pattern.finditer(norm))
    if not matches:
        return out
    for i, m in enumerate(matches):
        name = m.group(1).lower()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(norm)
        chunk = norm[start:end].strip()
        # keep the longest chunk per problem (first occurrence is usually a TOC line)
        if len(chunk) > len(out.get(name, "")):
            out[name] = chunk
    return out


def _list_test_pids(year: str, tema: str) -> list[str]:
    """List problem ids by inspecting the inner zip's `private_tests/` dir."""
    base = EXTRACTED_TESTS_DIR / f"pa_{year}_{tema}"
    inner_zip = base / f"pa_{year}_{tema}.zip"
    if not inner_zip.exists():
        return []
    pids: set[str] = set()
    with zipfile.ZipFile(inner_zip) as iz:
        for j in iz.infolist():
            parts = j.filename.replace("\\", "/").split("/")
            if len(parts) >= 3 and parts[0] == "private_tests" and parts[1]:
                pids.add(parts[1])
    return sorted(pids)


def _list_solution_files(year: str, pid: str) -> list[Path]:
    base = EXTRACTED_SOLUTIONS_DIR / "solutions" / f"{year}_{pid}"
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_file())


def _read_text_safe(p: Path, max_bytes: int = 200_000) -> str:
    try:
        data = p.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def build_problems_index() -> list[dict[str, Any]]:
    """Build the per-problem index for the 36 PA 2021-2024 problems."""
    index: list[dict[str, Any]] = []
    for year in PA_YEARS:
        for tema in PA_TEMAS:
            pids = _list_test_pids(year, tema)
            if not pids:
                continue
            full_text = _load_statement_text(year, tema)
            split = _split_statement_per_problem(full_text, pids)
            for pid in pids:
                sol_files = _list_solution_files(year, pid)
                stmt_chunk = split.get(pid.lower(), "")
                index.append(
                    {
                        "problem_id": f"{year}_{tema}_{pid}",
                        "year": year,
                        "tema": tema,
                        "pid": pid,
                        "n_solutions": len(sol_files),
                        "languages": sorted(
                            {p.suffix.lower().lstrip(".") for p in sol_files}
                        ),
                        "statement_chunk_chars": len(stmt_chunk),
                        "statement_chunk_excerpt": stmt_chunk[:600],
                    }
                )
    return index


def build_canonical() -> int:
    """Write canonical.jsonl: one row per (anon_id, problem, submission)."""
    rows: list[dict[str, Any]] = []

    pid_to_problems = defaultdict(list)
    for year in PA_YEARS:
        for tema in PA_TEMAS:
            for pid in _list_test_pids(year, tema):
                pid_to_problems[pid].append((year, tema))

    feedback_iter = list(read_jsonl(RAW_FEEDBACK_JSONL))
    for fb in tqdm(feedback_iter, desc="canonical from feedback"):
        a_id = fb.get("assignment_id", "")
        if not (a_id.startswith(("2021_", "2022_", "2023_", "2024_")) and (
            "tema1" in a_id or "tema2" in a_id
        )):
            continue
        year = a_id[:4]
        tema = "tema1" if "tema1" in a_id else "tema2"
        anon = fb.get("anon_id")
        sub_name = fb.get("submission_name", "")
        ts = _ts_from_submission_name(sub_name)
        feedback_text = fb.get("feedback", "") or ""
        for entry in fb.get("checker_output", []) or []:
            pid = entry.get("pid", "")
            if not pid:
                continue
            try:
                pts = float(entry.get("pts", 0))
            except (TypeError, ValueError):
                pts = 0.0
            issues = entry.get("issues", []) or []
            problem_id = f"{year}_{tema}_{pid}"
            rows.append(
                {
                    "row_kind": "submission_feedback",
                    "problem_id": problem_id,
                    "year": year,
                    "tema": tema,
                    "pid": pid,
                    "anon_id": anon,
                    "submission_name": sub_name,
                    "submitted_at": ts,
                    "pts": pts,
                    "issues": issues,
                    "feedback_raw": feedback_text,
                }
            )

    n = write_jsonl(CANONICAL_JSONL, rows)
    return n


def build_user_trajectories() -> int:
    """Group canonical rows by (anon_id, problem_id) sorted chronologically."""
    rows = list(read_jsonl(CANONICAL_JSONL))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("anon_id") or "anon_unknown", r["problem_id"])
        grouped[key].append(r)
    out: list[dict[str, Any]] = []
    for (anon, problem_id), subs in grouped.items():
        subs.sort(key=lambda x: x.get("submitted_at") or "")
        progression = [s["pts"] for s in subs]
        out.append(
            {
                "anon_id": anon,
                "problem_id": problem_id,
                "n_submissions": len(subs),
                "first_at": subs[0].get("submitted_at"),
                "last_at": subs[-1].get("submitted_at"),
                "pts_progression": progression,
                "final_pts": progression[-1] if progression else 0.0,
                "max_pts": max(progression) if progression else 0.0,
                "issues_history": [s.get("issues", []) for s in subs],
            }
        )
    return write_jsonl(USER_TRAJECTORIES_JSONL, out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-statements",
        action="store_true",
        help="Skip PDF parsing (faster, when iterating).",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.skip_statements:
        # short-circuit: build a degenerate index without statement text
        global _load_statement_text  # noqa: PLW0603
        _load_statement_text = lambda y, t: ""  # type: ignore[assignment]

    print("building problems_index ...")
    idx = build_problems_index()
    write_json(PROBLEMS_INDEX_JSON, idx)
    print(f"  -> {len(idx)} problems indexed at {PROBLEMS_INDEX_JSON}")

    print("building canonical.jsonl ...")
    n = build_canonical()
    print(f"  -> {n} rows at {CANONICAL_JSONL}")

    print("building user_trajectories.jsonl ...")
    m = build_user_trajectories()
    print(f"  -> {m} (user, problem) trajectories at {USER_TRAJECTORIES_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

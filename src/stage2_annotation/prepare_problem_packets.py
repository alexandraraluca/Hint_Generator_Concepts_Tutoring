"""Stage 2.1 - bundle each problem's context into a self-contained "packet".

A packet is the LLM-friendly representation of one problem, used by the
problem-annotator (assign concepts) and later by the hint generator. It
contains:
- the (heuristically) extracted statement chunk, full text,
- 1-3 representative max-score solutions (chosen by length percentile),
- meta: year, tema, pid, languages, n_solutions.

Output: `data/processed/packets/<problem_id>.json`
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.io_utils import read_json, write_json
from src.common.paths import (
    EXTRACTED_SOLUTIONS_DIR,
    EXTRACTED_STATEMENTS_DIR,
    PA_TEMAS,
    PA_YEARS,
    PROBLEMS_INDEX_JSON,
    PROCESSED_DIR,
    ensure_dirs,
)
from src.common.pdf_utils import extract_pdf_text

# filename format observed: anon_<id>_<score>.<ext>
# - score may be int (100) or float (57.1, 88.6, 28.6)
# - ext in {cpp, java}
_SOL_FILENAME_RE = re.compile(
    r"^(?P<anon>anon_\d+)_(?P<score>\d+(?:\.\d+)?)\.(?P<ext>cpp|java)$",
    re.IGNORECASE,
)


def _parse_solution_filename(name: str) -> tuple[str, float, str] | None:
    m = _SOL_FILENAME_RE.match(name)
    if not m:
        return None
    return (
        m.group("anon"),
        float(m.group("score")),
        m.group("ext").lower(),
    )


def _statement_full_text(year: str, tema: str) -> str:
    n = "1" if tema == "tema1" else "2"
    p = EXTRACTED_STATEMENTS_DIR / f"[PA] Tema {n} {year}.pdf"
    if not p.exists():
        return ""
    try:
        return extract_pdf_text(str(p))
    except Exception as e:  # noqa: BLE001
        print(f"warn: pdf parse failed for {p.name}: {e}")
        return ""


_PROBLEMA_COLON_RE = re.compile(
    r"(?im)(?P<n>\d+)\s+problema\s+(?P=n)\s*(?:\([^)]*\))?\s*:"
)
_PROBLEMA_DASH_RE = re.compile(r"(?im)Problema\s+(?P<n>\d+)\s*[–\-]")
_PROBLEMA_BONUS_RE = re.compile(r"(?im)(?P<n>\d+)Bonus")
_PID_FILE_RE_TEMPLATE = r"\b{pid}\.(in|out|c|cpp|java)\b"
_NEXT_SECTION_RE = re.compile(
    r"(?im)(?:"
    r"\n(?P<n1>\d+)\s+problema\s+(?P=n1)\s*(?:\([^)]*\))?\s*:"
    r"|\nProblema\s+(?P<n2>\d+)\s*[–\-]"
    r"|\n(?P<n3>\d+)Bonus"
    r")"
)


def _section_has_real_enunt(full_text: str, end_offset: int, problem_num: str) -> bool:
    """True when `N.1 Enunț` is followed by prose, not a TOC dot leader."""
    window = full_text[end_offset : end_offset + 500]
    if re.search(r"Enun[^\n]*\. \.", window):
        return False
    return bool(
        re.search(
            rf"{re.escape(problem_num)}\.1\s+Enun",
            window,
            re.IGNORECASE,
        )
        and re.search(r"Enun[^\n]*(?:\n|\s)+[A-Za-zăâîșțĂÂÎȘȚ]", window, re.IGNORECASE)
    )


def _real_problem_sections(full_text: str) -> list[tuple[int, int]]:
    """Return (start_offset, problem_number) for statement body sections (skip TOC)."""
    seen: set[tuple[int, int]] = set()
    sections: list[tuple[int, int]] = []
    for pattern in (_PROBLEMA_COLON_RE, _PROBLEMA_DASH_RE, _PROBLEMA_BONUS_RE):
        for m in pattern.finditer(full_text):
            n = m.group("n")
            if not _section_has_real_enunt(full_text, m.end(), n):
                continue
            key = (m.start(), int(n))
            if key in seen:
                continue
            seen.add(key)
            sections.append(key)
    sections.sort(key=lambda t: t[0])
    return sections


def _chunk_for_pid_in_sections(
    full_text: str, pid: str, sections: list[tuple[int, int]]
) -> str:
    pid_l = pid.lower()
    file_hint = re.compile(_PID_FILE_RE_TEMPLATE.format(pid=re.escape(pid_l)), re.IGNORECASE)
    for i, (start, _num) in enumerate(sections):
        end = sections[i + 1][0] if i + 1 < len(sections) else len(full_text)
        chunk = full_text[start:end]
        if file_hint.search(chunk):
            return chunk.strip()
    return ""


def _chunk_for_pid_by_file_hint(full_text: str, pid: str) -> str:
    """Anchor on `{pid}.in`/`.out` and expand to the enclosing section."""
    pid_l = pid.lower()
    anchor_m = re.search(
        rf"(?i)\b{re.escape(pid_l)}\.(?:in|out)\b",
        full_text,
    )
    if not anchor_m:
        return ""

    anchor = anchor_m.start()
    base = max(0, anchor - 12_000)
    before = full_text[base:anchor]
    start_candidates: list[int] = []

    for pattern in (_PROBLEMA_COLON_RE, _PROBLEMA_DASH_RE, _PROBLEMA_BONUS_RE):
        for m in pattern.finditer(before):
            n = m.group("n")
            if _section_has_real_enunt(full_text, base + m.end(), n):
                start_candidates.append(base + m.start())

    for m in re.finditer(r"(?im)(?:^|\n)(?P<n>\d+)\.1\s+Enun", before):
        start_candidates.append(base + m.start())

    if not start_candidates:
        return full_text[max(0, anchor - 2_000) : anchor + 4_000].strip()

    start = max(s for s in start_candidates if s <= anchor)
    after = full_text[anchor:]
    end_m = _NEXT_SECTION_RE.search(after, pos=80)
    end = anchor + end_m.start() if end_m else min(len(full_text), start + 8_000)
    return full_text[start:end].strip()


def _split_statement_per_problem_legacy(full_text: str, pids: list[str]) -> dict[str, str]:
    """Old fallback: second occurrence of pid word + 4000 chars (often wrong)."""
    out: dict[str, str] = {}
    for pid in {p.lower() for p in pids}:
        occurrences = [
            m.start()
            for m in re.finditer(rf"(?i)\b{re.escape(pid)}\b", full_text)
        ]
        if len(occurrences) >= 2:
            start = occurrences[1]
            out[pid] = full_text[start : start + 4000].strip()
        elif occurrences:
            start = occurrences[0]
            out[pid] = full_text[start : start + 4000].strip()
    return out


def _split_statement_per_problem(full_text: str, pids: list[str]) -> dict[str, str]:
    """Split the PDF text into one chunk per problem pid."""
    if not full_text or not pids:
        return {}

    pid_set = {p.lower() for p in pids}
    sections = _real_problem_sections(full_text)
    out: dict[str, str] = {}

    for pid in pid_set:
        chunk = _chunk_for_pid_in_sections(full_text, pid, sections)
        if not chunk:
            chunk = _chunk_for_pid_by_file_hint(full_text, pid)
        if chunk:
            out[pid] = chunk

    missing = pid_set - out.keys()
    if missing:
        out.update(_split_statement_per_problem_legacy(full_text, list(missing)))
    return out


def statement_text_for_problem(*, year: str, tema: str, pid: str) -> str:
    """Extract the statement body for one problem directly from the tema PDF."""
    full_text = _statement_full_text(year, tema)
    if not full_text:
        return ""
    return _split_statement_per_problem(full_text, [pid]).get(pid.lower(), "")


def statement_text_for_problem_id(problem_id: str, packets_dir: Path) -> tuple[str, str]:
    """Resolve statement via packet metadata + PDF re-parse.

    Returns `(text, source_label)`. Falls back to cached packet text if PDF missing.
    """
    packet_path = packets_dir / f"{problem_id}.json"
    if not packet_path.exists():
        return "", ""
    try:
        packet = read_json(packet_path)
    except Exception:  # noqa: BLE001
        return "", ""

    year = packet.get("year", "")
    tema = packet.get("tema", "")
    pid = packet.get("pid", "")
    pdf_label = f"PDF [{year}/{tema}] pid={pid}"

    if year and tema and pid:
        fresh = statement_text_for_problem(year=year, tema=tema, pid=pid)
        if fresh:
            return fresh, pdf_label

    cached = packet.get("statement_text", "") or ""
    if cached:
        return cached, f"{packet_path} (cached statement_text)"
    return "", ""


def _pick_representative_solutions(
    folder: Path, *, n_per_lang: int = 1, max_chars: int = 8_000
) -> list[dict[str, Any]]:
    """Pick 1 representative max-score solution per language (median LOC)."""
    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in folder.iterdir():
        if not f.is_file():
            continue
        parsed = _parse_solution_filename(f.name)
        if parsed is None:
            continue
        anon, score, ext = parsed
        if score < 99.999:  # essentially 100
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "\n/* ... [truncated] ... */"
        by_lang[ext].append(
            {
                "anon_id": anon,
                "score": score,
                "language": ext,
                "loc": text.count("\n"),
                "code": text,
            }
        )
    chosen: list[dict[str, Any]] = []
    for lang, items in by_lang.items():
        if not items:
            continue
        items.sort(key=lambda x: x["loc"])
        # pick median by LOC for robustness
        med_loc = statistics.median(it["loc"] for it in items)
        items.sort(key=lambda x: abs(x["loc"] - med_loc))
        chosen.extend(items[:n_per_lang])
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR / "packets",
        help="Output directory for per-problem packets.",
    )
    parser.add_argument(
        "--solutions-per-language",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    problems_index: list[dict[str, Any]] = read_json(PROBLEMS_INDEX_JSON)

    # group by (year, tema) so we parse each PDF once
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in problems_index:
        grouped[(p["year"], p["tema"])].append(p)

    n_written = 0
    for (year, tema), probs in grouped.items():
        full_text = _statement_full_text(year, tema)
        chunks = _split_statement_per_problem(
            full_text, [p["pid"] for p in probs]
        )
        for p in probs:
            pid = p["pid"]
            sol_dir = EXTRACTED_SOLUTIONS_DIR / "solutions" / f"{year}_{pid}"
            reps = _pick_representative_solutions(
                sol_dir,
                n_per_lang=args.solutions_per_language,
            ) if sol_dir.exists() else []
            packet = {
                "problem_id": p["problem_id"],
                "year": year,
                "tema": tema,
                "pid": pid,
                "n_solutions": p["n_solutions"],
                "languages": p["languages"],
                "statement_text": chunks.get(pid.lower(), ""),
                "statement_text_full_present": bool(full_text),
                "representative_solutions": reps,
            }
            out = args.out_dir / f"{p['problem_id']}.json"
            write_json(out, packet)
            n_written += 1

    print(f"wrote {n_written} packets at {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

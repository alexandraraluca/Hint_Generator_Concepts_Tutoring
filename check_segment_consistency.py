"""
segment_consistency_check.py
-----------------------------
Validates segment extractor reliability BEFORE running on all data.

Run this on 5-6 submissions of one problem first and inspect the output.
It checks:
  1. Schema consistency  — all required fields present, correct types
  2. Intent stability    — cosine similarity of intent embeddings across submissions
  3. Function coverage   — does the algorithm field vary appropriately?
  4. Semantic drift      — does a passing solution's intent match partial ones?

Usage:
    python check_segment_consistency.py \
        --problem-id 2021_tema1_crypto \
        --solutions-dir data/raw/solutions/solutions/2021_crypto \
        --passing-file anon_346_100.cpp \
        --sample 6 \
        --ollama-model gpt-oss:20b

Statement is loaded from the tema PDF via packet metadata (same extraction as
prepare_problem_packets). Override with --statement path/to/file.txt if needed.

Output:
    segment_consistency_report.json  — full results
    segment_consistency_report.txt   — human-readable summary

Pt toate problemele failed din silver_diff.jsonl: python check_segment_consistency.py --ollama-model gpt-oss:20b --out-dir reports/segment_batch
Va cauta solutia de 100 p a acelui student si va extrage segmentele pentru ambele solutii (failed si 100p) si apoi va calcula gap-ul intre ele.

"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

from src.common.io_utils import read_json, read_jsonl, write_jsonl
from src.common.paths import EXTRACTED_SOLUTIONS_DIR, HINTS_DIR, PROCESSED_DIR
from src.stage2_annotation.prepare_problem_packets import statement_text_for_problem_id
from generate_hint_concepts import (
    SYSTEM_PROMPT as HINT_SYSTEM_PROMPT,
    build_concept_user_prompt,
    parse_hints_json,
    validate_hints,
)

PACKETS_DIR_DEFAULT = PROCESSED_DIR / "packets"
SOLUTIONS_BASE_DEFAULT = EXTRACTED_SOLUTIONS_DIR / "solutions"
SILVER_DIFF_DEFAULT = HINTS_DIR / "silver_diff.jsonl"
SEGMENTS_OUT_FILENAME = "segments_silver_diff.jsonl"
STATEMENT_PROMPT_MAX = 1500


# ── Segment extraction prompt ──────────────────────────────────────────────────

SEGMENT_SYSTEM = """
Ești un asistent pentru analiza codului. Sarcina ta este să extragi o descriere semantică structurată a unei soluții de programare competitivă.

TREBUIE să returnezi JSON valid cu EXACT această schemă — fără câmpuri suplimentare și fără câmpuri lipsă:

{
  "intent": {
    "text": "<o propoziție în română: ce problemă încearcă acest cod să rezolve, descrisă în termenii problemei>",
    "matches_problem": true | false,
    "confidence": "high" | "medium" | "low"
  },
  "function": {
    "algorithm": "<numele algoritmului, ex: greedy, dp, binary_search, bfs>",
    "data_structures": ["<lista principalelor structuri de date utilizate>"],
    "complexity": "<complexitatea temporală dacă poate fi determinată, altfel 'unknown'>",
    "confidence": "high" | "medium" | "low"
  },
  "implementation": {
    "key_operations": ["<3-5 operații specifice în română, ex: 'sortare după costul de upgrade', 'prefix sum peste costuri'>"],
    "potential_issues": ["<în română: probleme observabile de implementare, listă goală dacă nu există>"],
    "confidence": "high" | "medium" | "low"
  }
}

IMPORTANT — LIMBA: Toate câmpurile text (intent.text, key_operations, potential_issues) TREBUIE scrise în limba română. Câmpurile cu valori fixe (confidence, algorithm) rămân în engleză conform schemei.

Reguli:
- intent.text descrie CE problemă crede studentul că rezolvă codul, nu ce ar trebui să rezolve
- Dacă este clar că programul rezolvă problema greșită, setează matches_problem=false
- function.algorithm trebuie să fie un singur nume canonic (ex: greedy, dp, binary_search)
- implementation.key_operations trebuie să fie specifice ACESTUI cod, nu generice
- Returnează DOAR obiectul JSON, fără markdown și fără explicații
""".strip()


def build_segment_prompt(code: str, statement_excerpt: str, max_code: int = 5000) -> str:
    return textwrap.dedent(f"""
    Enunțul problemei (extras):
    {statement_excerpt[:1500]}

    Codul studentului:
    ```
    {code[:max_code]}
    ```

    Extrage segmentele semantice conform schemei de mai sus.
    Scrie intent.text, key_operations și potential_issues în limba română.
    Returnează doar JSON, fără explicații.
    """).strip()


# ── LLM call (Ollama) ──────────────────────────────────────────────────────────

# Counters surface diagnostic info to the caller after each Ollama call.
_OLLAMA_LAST_STATS: dict[str, int] = {
    "prompt_eval_count": 0,
    "eval_count": 0,
    "num_ctx": 0,
    "num_predict": 0,
}


def call_ollama(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    timeout: int = 360,
    num_predict: int = 5048,
    num_ctx: int = 8192,
    keep_alive: str = "30m",
    max_retries: int = 2,
) -> str:
    """Call Ollama and return raw text.

    Improvements over the naive version:
      - `keep_alive="30m"` ține modelul în VRAM între apeluri (evită cold-start ~30-60s).
      - `num_predict=2048` evită runaway-ul; suficient pentru schema JSON
        (chiar și când modelul e prolix cu key_operations / potential_issues).
      - `num_ctx=8192` rezervă context window predictibil. Promptul tipic ~3000
        tokeni → rămân ~3000 tokeni headroom după num_predict.
      - retry până la max_retries cu backoff scurt pe timeout / eroare HTTP.

    Populeaază `_OLLAMA_LAST_STATS` cu prompt_eval_count + eval_count așa cum
    le întoarce Ollama în răspuns, ca apelantul să poată loga / detecta cazurile
    în care promptul s-ar fi putut tăia.
    """
    import time as _t
    import urllib.error
    import urllib.request

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }).encode()

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            _OLLAMA_LAST_STATS["prompt_eval_count"] = int(data.get("prompt_eval_count", 0))
            _OLLAMA_LAST_STATS["eval_count"] = int(data.get("eval_count", 0))
            _OLLAMA_LAST_STATS["num_ctx"] = num_ctx
            _OLLAMA_LAST_STATS["num_predict"] = num_predict
            return data["message"]["content"]
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < max_retries:
                _t.sleep(2 + 2 * attempt)  # 2s, 4s
                continue
            raise
    # unreachable
    assert last_err is not None
    raise last_err


def _ollama_truncation_warnings() -> list[str]:
    """Returns a list of warning strings about truncation risk based on the
    last call's stats. Empty list = no warning.
    """
    pe = _OLLAMA_LAST_STATS["prompt_eval_count"]
    ec = _OLLAMA_LAST_STATS["eval_count"]
    nc = _OLLAMA_LAST_STATS["num_ctx"]
    np_ = _OLLAMA_LAST_STATS["num_predict"]
    warnings: list[str] = []
    # Prompt eating into the num_predict budget = truncation almost certain.
    if nc and pe and pe > nc - np_:
        warnings.append(
            f"prompt={pe}tok > num_ctx({nc}) − num_predict({np_})={nc - np_}tok → prompt truncated by Ollama"
        )
    # Output hit the cap → JSON probably cut.
    if np_ and ec and ec >= np_:
        warnings.append(
            f"output={ec}tok hit num_predict cap ({np_}) → JSON likely truncated"
        )
    return warnings


# ── Schema validation ──────────────────────────────────────────────────────────

REQUIRED_SCHEMA = {
    "intent": {"text": str, "matches_problem": bool, "confidence": str},
    "function": {"algorithm": str, "data_structures": list, "complexity": str, "confidence": str},
    "implementation": {"key_operations": list, "potential_issues": list, "confidence": str},
}

VALID_CONFIDENCE = {"high", "medium", "low"}


def validate_schema(seg: dict) -> list[str]:
    """Return list of schema violations. Empty = valid."""
    errors = []
    for section, fields in REQUIRED_SCHEMA.items():
        if section not in seg:
            errors.append(f"missing section: {section}")
            continue
        for field, ftype in fields.items():
            if field not in seg[section]:
                errors.append(f"missing field: {section}.{field}")
            elif not isinstance(seg[section][field], ftype):
                errors.append(
                    f"wrong type: {section}.{field} "
                    f"(got {type(seg[section][field]).__name__}, expected {ftype.__name__})"
                )
    # Confidence values
    for section in ("intent", "function", "implementation"):
        conf = seg.get(section, {}).get("confidence", "")
        if conf not in VALID_CONFIDENCE:
            errors.append(f"invalid confidence in {section}: {repr(conf)}")
    return errors


# ── Parse JSON robustly ────────────────────────────────────────────────────────

def parse_json_robust(raw: str) -> tuple[dict | None, str]:
    """Try to extract JSON from raw LLM output. Returns (parsed, error)."""
    # Direct parse
    try:
        return json.loads(raw), ""
    except json.JSONDecodeError:
        pass
    # Extract from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), ""
        except json.JSONDecodeError:
            pass
    # Find first { ... } block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), ""
        except json.JSONDecodeError as e:
            return None, f"json parse failed: {e}"
    return None, "no JSON found in output"


# ── Embedding similarity (optional, uses sentence-transformers if available) ───

def compute_similarity(texts: list[str]) -> list[list[float]] | None:
    """Return pairwise cosine similarity matrix, or None if deps missing."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(texts, normalize_embeddings=True)
        matrix = (embs @ embs.T).tolist()
        return matrix
    except ImportError:
        return None


# ── Per-file segment extraction (shared by single and batch modes) ─────────────

def extract_segments_for_file(
    file_path: Path,
    statement_excerpt: str,
    ollama_model: str,
) -> dict:
    """Extract segments for one source file. Returns a result dict with keys:
    segments (dict|None), schema_errors (list), error (str|None), raw (str),
    ollama_stats (dict with prompt_eval_count/eval_count), truncation_warnings.
    """
    code = file_path.read_text(encoding="utf-8", errors="replace")
    user_prompt = build_segment_prompt(code, statement_excerpt)
    try:
        raw = call_ollama(ollama_model, SEGMENT_SYSTEM, user_prompt)
        parsed, parse_error = parse_json_robust(raw)
    except Exception as e:
        return {
            "segments": None, "schema_errors": ["llm_call_failed"],
            "error": str(e), "raw": "",
            "ollama_stats": dict(_OLLAMA_LAST_STATS),
            "truncation_warnings": [],
        }

    stats = dict(_OLLAMA_LAST_STATS)
    warnings = _ollama_truncation_warnings()
    if warnings:
        # surface them to the operator immediately
        print(f"\n  [WARN truncation @ {file_path.name}] " + "; ".join(warnings))

    if parsed is None:
        return {
            "segments": None, "schema_errors": ["json_parse_failed"],
            "error": parse_error, "raw": raw[:300],
            "ollama_stats": stats, "truncation_warnings": warnings,
        }

    schema_errors = validate_schema(parsed)
    return {
        "segments": parsed, "schema_errors": schema_errors,
        "error": None, "raw": raw,
        "ollama_stats": stats, "truncation_warnings": warnings,
    }


def _problem_id_to_solutions_dir(problem_id: str, solutions_base: Path) -> Path:
    """Map '2021_tema1_crypto' → solutions_base/2021_crypto."""
    parts = problem_id.split("_", 2)   # ['2021', 'tema1', 'crypto']
    if len(parts) != 3:
        raise ValueError(f"Cannot parse problem_id: {problem_id!r}")
    year, _tema, pid = parts
    return solutions_base / f"{year}_{pid}"


# ── Batch processing over silver_diff.jsonl ────────────────────────────────────

def _generate_hints_for_row(
    row: dict,
    statement_excerpt: str,
    ollama_model: str,
) -> dict:
    """Call the LLM to generate hints from segments. Returns a dict with:
    generated_hints, gap_codes, hint_validation_errors, hint_rationale, hint_source.
    """
    user_prompt = build_concept_user_prompt(row, statement_excerpt)
    try:
        raw = call_ollama(ollama_model, HINT_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        return {
            "generated_hints": [],
            "gap_codes": None,
            "hint_validation_errors": [f"llm_error: {e}"],
            "hint_rationale": "",
            "hint_source": "concept_segments_llm_error",
        }
    parsed, parse_err = parse_hints_json(raw)
    if parsed is None:
        return {
            "generated_hints": [],
            "gap_codes": None,
            "hint_validation_errors": [f"parse_error: {parse_err}"],
            "hint_rationale": "",
            "hint_source": "concept_segments_parse_error",
        }
    hint_errors = validate_hints(parsed)
    hints = parsed.get("hints", [])[:4]
    return {
        "generated_hints": hints,
        "gap_codes": parsed.get("gap_codes"),
        "hint_validation_errors": hint_errors,
        "hint_rationale": parsed.get("rationale_short", ""),
        "hint_source": "concept_segments",
    }


def process_silver_batch(
    silver_jsonl: Path,
    out_dir: Path,
    ollama_model: str,
    packets_dir: Path,
    solutions_base: Path,
    resume: bool = True,
    limit: int = 0,
    problem_filter: str | None = None,
) -> None:
    """For every row in silver_diff.jsonl:
      - find the failing submission and the matching 100p file on disk
      - extract segments (intent, function, implementation) for both via Ollama
      - generate graduated hints from the extracted segments via Ollama
      - write enriched rows to {out_dir}/segments_silver_diff.jsonl

    Each output row contains:
      - all original silver_diff fields (including 'hints' as 'silver_hints')
      - statement_source
      - partial_segments / partial_segment_errors
      - passing_segments / passing_segment_errors
      - generated_hints — new hints produced from concept segments
      - hint_validation_errors, hint_rationale, hint_source
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / SEGMENTS_OUT_FILENAME

    # Build set of already-processed (problem_id, submission_name) for resume
    done: set[tuple[str, str]] = set()
    if resume and out_path.exists():
        for r in read_jsonl(out_path):
            done.add((r.get("problem_id", ""), r.get("submission_name", "")))
        print(f"Resuming — {len(done)} rows already processed, skipping them.")

    rows = list(read_jsonl(silver_jsonl))
    if problem_filter:
        rows = [r for r in rows if r["problem_id"] == problem_filter]
        print(f"Filtered to problem_id={problem_filter!r}: {len(rows)} rows")
    if limit:
        rows = rows[:limit]

    # Cache statements per problem_id to avoid repeated PDF parsing
    statement_cache: dict[str, tuple[str, str]] = {}

    written = skipped = errors = 0

    for i, row in enumerate(rows, 1):
        problem_id = row["problem_id"]
        submission_name = row["submission_name"]
        passing_file = row["passing_file"]

        key = (problem_id, submission_name)
        if key in done:
            skipped += 1
            continue

        try:
            _process_one_row(
                i=i, total=len(rows), row=row,
                problem_id=problem_id, submission_name=submission_name,
                passing_file=passing_file,
                solutions_base=solutions_base, packets_dir=packets_dir,
                ollama_model=ollama_model, statement_cache=statement_cache,
                out_path=out_path,
            )
            written += 1
        except _RowSkipped as e:
            print(f"[{i}/{len(rows)}] SKIP {submission_name}: {e}")
            errors += 1
        except Exception as e:  # noqa: BLE001 — defensive: do NOT crash the batch
            print(f"[{i}/{len(rows)}] UNEXPECTED ERROR for {submission_name}: {e!r}")
            errors += 1
            # write a stub row so resume skips it next time
            stub = {
                **{k: v for k, v in row.items() if k != "hints"},
                "silver_hints": row.get("hints", []),
                "_processing_error": repr(e),
            }
            try:
                with open(out_path, "ab") as fb:
                    import orjson as _oj
                    fb.write(_oj.dumps(stub, option=_oj.OPT_APPEND_NEWLINE))
            except Exception:
                pass
    # End of for loop, fall through to summary print below.
    print(f"\n{'='*60}")
    print(f"Batch done: {written} written, {skipped} skipped (resumed), {errors} errors")
    print(f"Output: {out_path}")


class _RowSkipped(Exception):
    """Sentinel exception used by _process_one_row to indicate a clean skip
    (missing file, unparseable problem_id, etc.) — counts toward errors,
    not a crash.
    """


def _process_one_row(
    *,
    i: int,
    total: int,
    row: dict,
    problem_id: str,
    submission_name: str,
    passing_file: str,
    solutions_base: Path,
    packets_dir: Path,
    ollama_model: str,
    statement_cache: dict[str, tuple[str, str]],
    out_path: Path,
) -> None:
    """Process a single silver_diff row end-to-end. Raises _RowSkipped for
    clean skips, lets unexpected errors propagate to the batch handler.
    """
    # ── Resolve file paths ────────────────────────────────────────────────
    try:
        sol_dir = _problem_id_to_solutions_dir(problem_id, solutions_base)
    except ValueError as e:
        raise _RowSkipped(str(e)) from e

    fail_path = sol_dir / submission_name
    pass_path = sol_dir / passing_file

    if not fail_path.exists():
        raise _RowSkipped(f"missing failing file: {fail_path}")
    if not pass_path.exists():
        raise _RowSkipped(f"missing passing file: {pass_path}")

    # ── Load statement (cached) ───────────────────────────────────────────
    if problem_id not in statement_cache:
        statement_cache[problem_id] = statement_text_for_problem_id(problem_id, packets_dir)
    statement_excerpt, statement_source = statement_cache[problem_id]

    # ── Extract segments ──────────────────────────────────────────────────
    print(f"[{i}/{total}] {problem_id} | {submission_name} ...", end=" ", flush=True)
    fail_result = extract_segments_for_file(fail_path, statement_excerpt, ollama_model)

    if fail_result["error"]:
        status = f"FAIL_PARTIAL({fail_result['error'][:40]})"
    elif fail_result["schema_errors"]:
        status = "SCHEMA_ERR_PARTIAL"
    else:
        status = "ok_partial"
    print(f"{status}", end=" | ", flush=True)

    pass_result = extract_segments_for_file(pass_path, statement_excerpt, ollama_model)
    if pass_result["error"]:
        status = f"FAIL_PASSING({pass_result['error'][:40]})"
    elif pass_result["schema_errors"]:
        status = "SCHEMA_ERR_PASSING"
    else:
        status = "ok_passing"
    print(f"{status}", end=" | hints→", flush=True)

    # ── Generate hints from segments ──────────────────────────────────────
    row_with_segs = {
        **row,
        "partial_segments": fail_result["segments"],
        "partial_segment_errors": fail_result["schema_errors"],
        "passing_segments": pass_result["segments"],
        "passing_segment_errors": pass_result["schema_errors"],
    }
    hint_result = _generate_hints_for_row(row_with_segs, statement_excerpt, ollama_model)
    n_hints = len(hint_result["generated_hints"])
    hint_ok = not hint_result["hint_validation_errors"]
    gap_lvl = (hint_result.get("gap_codes") or {}).get("level", "?")
    print(f"{'OK' if hint_ok else 'WARN'}({n_hints} hinturi, gap={gap_lvl})")

    # ── Write enriched row ────────────────────────────────────────────────
    enriched = {
        **{k: v for k, v in row.items() if k != "hints"},
        "silver_hints": row.get("hints", []),
        "statement_source": statement_source,
        "partial_segments": fail_result["segments"],
        "partial_segment_errors": fail_result["schema_errors"],
        "passing_segments": pass_result["segments"],
        "passing_segment_errors": pass_result["schema_errors"],
        "gap_codes": hint_result["gap_codes"],
        "generated_hints": hint_result["generated_hints"],
        "hint_validation_errors": hint_result["hint_validation_errors"],
        "hint_rationale": hint_result["hint_rationale"],
        "hint_source": hint_result["hint_source"],
        # Diagnostics: shows actual prompt/output token counts per LLM call.
        # If you ever see truncation_warnings non-empty, you know which call
        # was clipped and can raise num_ctx / num_predict.
        "diagnostics": {
            "partial_ollama_stats": fail_result.get("ollama_stats", {}),
            "partial_truncation_warnings": fail_result.get("truncation_warnings", []),
            "passing_ollama_stats": pass_result.get("ollama_stats", {}),
            "passing_truncation_warnings": pass_result.get("truncation_warnings", []),
        },
    }
    with open(out_path, "ab") as f:
        import orjson
        f.write(orjson.dumps(enriched, option=orjson.OPT_APPEND_NEWLINE))


# ── Main consistency check ─────────────────────────────────────────────────────

def run_check(
    problem_id: str,
    solutions_dir: Path,
    passing_file: str,
    statement_excerpt: str,
    sample_n: int,
    ollama_model: str,
) -> dict[str, Any]:

    # Collect files
    all_files = list(solutions_dir.glob("*.cpp")) + list(solutions_dir.glob("*.java"))
    passing = solutions_dir / passing_file
    partial_files = [f for f in all_files if f.name != passing_file]

    # Sample across score ranges
    def score_from_name(p: Path) -> float:
        m = re.search(r"_(\d+(?:\.\d+)?)\.(cpp|java)$", p.name)
        return float(m.group(1)) if m else 50.0

    partial_files.sort(key=score_from_name)
    n_partials_target = max(sample_n - 1, 0)
    n_partials_available = len(partial_files)

    # Pick evenly spaced samples by score (low → high), same count every run for a folder.
    selected_partials = partial_files
    if n_partials_available > n_partials_target and n_partials_target > 0:
        if n_partials_target == 1:
            indices = [0]
        else:
            indices = [
                int(i * (n_partials_available - 1) / (n_partials_target - 1))
                for i in range(n_partials_target)
            ]
        selected_partials = [partial_files[i] for i in indices]

    files_to_check = selected_partials + [passing]
    print(
        f"Sample {sample_n}: {n_partials_target} partial + 1 passing "
        f"({n_partials_available} partials available, evenly spaced by score)"
    )
    for f in selected_partials:
        print(f"  partial  {f.name}  score={score_from_name(f)}")
    print(f"  passing  {passing.name}  score={score_from_name(passing)}")

    results = []
    for f in files_to_check:
        print(f"  extracting segments from {f.name}...", end=" ", flush=True)
        code = f.read_text(encoding="utf-8", errors="replace")
        score = score_from_name(f)
        is_passing = f.name == passing_file

        user_prompt = build_segment_prompt(code, statement_excerpt)

        try:
            raw = call_ollama(ollama_model, SEGMENT_SYSTEM, user_prompt)
            parsed, parse_error = parse_json_robust(raw)
        except Exception as e:
            print(f"FAILED ({e})")
            results.append({
                "file": f.name,
                "score": score,
                "is_passing": is_passing,
                "error": str(e),
                "schema_errors": ["llm_call_failed"],
                "segments": None,
                "raw": "",
            })
            continue

        if parsed is None:
            print(f"PARSE FAIL")
            results.append({
                "file": f.name,
                "score": score,
                "is_passing": is_passing,
                "error": parse_error,
                "schema_errors": ["json_parse_failed"],
                "segments": None,
                "raw": raw[:300],
            })
            continue

        schema_errors = validate_schema(parsed)
        status = "OK" if not schema_errors else f"SCHEMA({len(schema_errors)} errors)"
        print(status)

        results.append({
            "file": f.name,
            "score": score,
            "is_passing": is_passing,
            "error": None,
            "schema_errors": schema_errors,
            "segments": parsed,
            "raw": raw,
        })

    # ── Consistency analysis ───────────────────────────────────────────────────
    valid = [r for r in results if r["segments"] is not None and not r["schema_errors"]]
    schema_pass_rate = len(valid) / len(results) if results else 0

    # Algorithm consistency
    algorithms = [r["segments"]["function"]["algorithm"] for r in valid]
    unique_algorithms = list(set(algorithms))

    # Intent similarity (if sentence-transformers available)
    intent_texts = [r["segments"]["intent"]["text"] for r in valid]
    sim_matrix = None
    avg_intent_sim = None
    if len(intent_texts) >= 2:
        sim_matrix = compute_similarity(intent_texts)
        if sim_matrix:
            n = len(sim_matrix)
            off_diag = [sim_matrix[i][j] for i in range(n) for j in range(n) if i != j]
            avg_intent_sim = sum(off_diag) / len(off_diag) if off_diag else None

    # Confidence distribution
    conf_dist = {"high": 0, "medium": 0, "low": 0}
    for r in valid:
        for section in ("intent", "function", "implementation"):
            c = r["segments"][section].get("confidence", "")
            if c in conf_dist:
                conf_dist[c] += 1

    # matches_problem flag
    mismatched_intent = [r["file"] for r in valid
                         if not r["segments"]["intent"].get("matches_problem", True)]

    analysis = {
        "total_files": len(results),
        "schema_pass_rate": round(schema_pass_rate, 3),
        "schema_failures": [
            {"file": r["file"], "errors": r["schema_errors"]}
            for r in results if r["schema_errors"]
        ],
        "algorithms_found": algorithms,
        "unique_algorithms": unique_algorithms,
        "algorithm_consistent": len(unique_algorithms) <= 2,  # allow 1-2 variants
        "avg_intent_similarity": round(avg_intent_sim, 3) if avg_intent_sim else "needs sentence-transformers",
        "confidence_distribution": conf_dist,
        "mismatched_intent_files": mismatched_intent,
        "verdict": _overall_verdict(schema_pass_rate, unique_algorithms, avg_intent_sim),
    }

    return {"problem_id": problem_id, "results": results, "analysis": analysis}


def _overall_verdict(schema_pass_rate: float, algos: list, avg_sim: float | None) -> str:
    issues = []
    if schema_pass_rate < 0.8:
        issues.append(f"schema pass rate too low ({schema_pass_rate:.0%}) — fix the prompt")
    if len(algos) > 3:
        issues.append(f"algorithm field too inconsistent ({len(set(algos))} variants) — constrain allowed values in prompt")
    if avg_sim is not None and avg_sim < 0.6:
        issues.append(f"intent similarity too low ({avg_sim:.2f}) — LLM describing different problems")
    if not issues:
        return "READY — segment extractor is consistent enough to use"
    return "NOT READY — " + "; ".join(issues)


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(report: dict, out_dir: Path) -> None:
    json_path = out_dir / "segment_consistency_report.json"
    txt_path = out_dir / "segment_consistency_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    a = report["analysis"]
    lines = [
        f"SEGMENT CONSISTENCY REPORT — {report['problem_id']}",
        "=" * 60,
        f"Files checked:        {a['total_files']}",
        f"Schema pass rate:     {a['schema_pass_rate']:.0%}",
        f"Algorithm consistent: {a['algorithm_consistent']}",
        f"Algorithms found:     {a['algorithms_found']}",
        f"Avg intent similarity:{a['avg_intent_similarity']}",
        f"Confidence dist:      {a['confidence_distribution']}",
        f"Intent mismatches:    {a['mismatched_intent_files'] or 'none'}",
        "",
        f"VERDICT: {a['verdict']}",
        "",
    ]
    if a["schema_failures"]:
        lines.append("Schema failures:")
        for sf in a["schema_failures"]:
            lines.append(f"  {sf['file']}: {sf['errors']}")
        lines.append("")

    def _safe_section(seg: dict, name: str) -> dict:
        section = seg.get(name, {})
        return section if isinstance(section, dict) else {}

    lines.append("Per-file segments:")
    for r in report["results"]:
        lines.append(f"\n  {r['file']} (score={r['score']})")
        if r["segments"]:
            seg = r["segments"]
            intent = _safe_section(seg, "intent")
            function = _safe_section(seg, "function")
            implementation = _safe_section(seg, "implementation")

            intent_text = str(intent.get("text", "<missing intent.text>"))[:80]
            intent_conf = intent.get("confidence", "unknown")
            algo = function.get("algorithm", "<missing function.algorithm>")
            data_structures = function.get("data_structures", [])
            func_conf = function.get("confidence", "unknown")
            key_ops = implementation.get("key_operations", [])
            impl_conf = implementation.get("confidence", "unknown")
            issues = implementation.get("potential_issues", [])

            lines.append(f"    intent:   {intent_text} [{intent_conf}]")
            lines.append(f"    function: {algo} / {data_structures} [{func_conf}]")
            lines.append(f"    impl:     {key_ops} [{impl_conf}]")
            if issues:
                lines.append(f"    issues:   {issues}")
            if r.get("schema_errors"):
                lines.append(f"    schema_errors: {r['schema_errors']}")
        else:
            lines.append(f"    ERROR: {r['error']}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {json_path} and {txt_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Segment consistency check + batch gap extraction.\n\n"
            "SINGLE-PROBLEM mode (provide --solutions-dir + --passing-file):\n"
            "  Samples N files from a folder, extracts segments, checks consistency.\n\n"
            "BATCH mode (no --solutions-dir / --passing-file):\n"
            "  Processes all rows from silver_diff.jsonl, extracts segments for both\n"
            "  the failing and the 100p file, computes gap_object, writes JSONL output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── shared args ──────────────────────────────────────────────────────────
    parser.add_argument("--problem-id", default="2021_tema1_crypto",
                        help="problem id (used in single mode and as batch filter)")
    parser.add_argument("--ollama-model", default="mistral:7b-instruct")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--packets-dir", type=Path, default=PACKETS_DIR_DEFAULT,
                        help="folder with <problem_id>.json packets")
    # ── single-problem args ──────────────────────────────────────────────────
    parser.add_argument("--solutions-dir", type=Path, default=None,
                        help="[single mode] path to folder with .cpp/.java files")
    parser.add_argument("--passing-file", default=None,
                        help="[single mode] filename of the 100p solution")
    parser.add_argument("--statement", type=Path, default=None,
                        help="[single mode] optional .txt statement override")
    parser.add_argument("--sample", type=int, default=6,
                        help="[single mode] total files to check (default 6)")
    # ── batch args ───────────────────────────────────────────────────────────
    parser.add_argument("--silver-jsonl", type=Path, default=SILVER_DIFF_DEFAULT,
                        help="[batch mode] input JSONL (default: data/hints/silver_diff.jsonl)")
    parser.add_argument("--solutions-base", type=Path, default=SOLUTIONS_BASE_DEFAULT,
                        help="[batch mode] root of solutions/ tree")
    parser.add_argument("--no-resume", action="store_true",
                        help="[batch mode] re-process already-written rows")
    parser.add_argument("--limit", type=int, default=0,
                        help="[batch mode] process at most N rows (0 = all)")
    parser.add_argument("--filter-problem", type=str, default=None,
                        help="[batch mode] restrict to one problem_id")
    args = parser.parse_args()

    single_mode = args.solutions_dir is not None and args.passing_file is not None

    # ════════════════════════════════════════════════════════════════════════
    # SINGLE-PROBLEM MODE
    # ════════════════════════════════════════════════════════════════════════
    if single_mode:
        statement_source = ""
        if args.statement and args.statement.exists():
            statement_excerpt = args.statement.read_text(encoding="utf-8", errors="replace")
            statement_source = str(args.statement)
        else:
            statement_excerpt, statement_source = statement_text_for_problem_id(
                args.problem_id, args.packets_dir
            )
            if not statement_excerpt:
                print(
                    f"No statement found for {args.problem_id} "
                    f"(and no --statement file) — using empty excerpt"
                )

        if statement_excerpt:
            prompt_excerpt = statement_excerpt[:STATEMENT_PROMPT_MAX]
            print("=" * 60)
            print("STATEMENT")
            print("=" * 60)
            print(f"Source: {statement_source}")
            print(f"Length: {len(statement_excerpt)} chars")
            if len(statement_excerpt) > STATEMENT_PROMPT_MAX:
                print(f"Prompt uses first {STATEMENT_PROMPT_MAX} chars (see build_segment_prompt)")
            print("-" * 60)
            print(prompt_excerpt)
            if len(statement_excerpt) > STATEMENT_PROMPT_MAX:
                print("-" * 60)
                print(f"... [{len(statement_excerpt) - STATEMENT_PROMPT_MAX} chars omitted]")
            print("=" * 60)
            print()

        report = run_check(
            problem_id=args.problem_id,
            solutions_dir=args.solutions_dir,
            passing_file=args.passing_file,
            statement_excerpt=statement_excerpt,
            sample_n=args.sample,
            ollama_model=args.ollama_model,
        )

        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        for k, v in report["analysis"].items():
            if k != "schema_failures":
                print(f"  {k:<28} {v}")
        print()
        print(f"VERDICT: {report['analysis']['verdict']}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_report(report, args.out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # BATCH MODE
    # ════════════════════════════════════════════════════════════════════════
    else:
        if not args.silver_jsonl.exists():
            print(f"ERROR: silver_diff JSONL not found: {args.silver_jsonl}")
            return

        problem_filter = args.filter_problem
        # If --problem-id was explicitly given (not default) and no --filter-problem,
        # use it as a filter so the user can do:
        #   python check_segment_consistency.py --problem-id 2021_tema1_crypto --ollama-model gpt-oss:20b
        if problem_filter is None and args.problem_id != "2021_tema1_crypto":
            problem_filter = args.problem_id

        print("=" * 60)
        print("BATCH MODE — segment extraction over silver_diff.jsonl")
        print("=" * 60)
        print(f"Input:    {args.silver_jsonl}")
        print(f"Solutions:{args.solutions_base}")
        print(f"Packets:  {args.packets_dir}")
        print(f"Model:    {args.ollama_model}")
        print(f"Out:      {args.out_dir / SEGMENTS_OUT_FILENAME}")
        if problem_filter:
            print(f"Filter:   problem_id={problem_filter!r}")
        if args.limit:
            print(f"Limit:    {args.limit} rows")
        print()

        process_silver_batch(
            silver_jsonl=args.silver_jsonl,
            out_dir=args.out_dir,
            ollama_model=args.ollama_model,
            packets_dir=args.packets_dir,
            solutions_base=args.solutions_base,
            resume=not args.no_resume,
            limit=args.limit,
            problem_filter=problem_filter,
        )


if __name__ == "__main__":
    main()
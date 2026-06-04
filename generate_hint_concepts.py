"""
generate_hint_concepts.py
--------------------------
Generează hinturi gradate pe baza segmentelor semantice extrase de
check_segment_consistency.py (fișierul segments_silver_diff.jsonl).

Pentru fiecare rând cu ambele seturi de segmente valide:
  1. Construiește un prompt în română care prezintă contrast semantic:
       - intent, algoritm, pași cheie, probleme identificate → codul GREȘIT
       - intent, algoritm, pași cheie                        → codul de 100p
  2. Apelează Ollama pentru a genera 1-4 hinturi gradate.
  3. Validează structura JSON (același schema ca silver_diff.jsonl).
  4. Scrie output în hints_from_concepts.jsonl.

Schema output (compatibilă cu silver_diff.jsonl):
  {
    "problem_id": "...",
    "anon_id": "...",
    "submission_name": "...",
    "language": "...",
    "verdict": "...",
    "hints": [
      {"level": "macro",      "text": "..."},
      {"level": "structural", "text": "..."},
      {"level": "specific",   "text": "..."}
    ],
    "source": "concept_segments",
    "embedding_similarity": ...,
    "partial_segments": {...},
    "passing_segments": {...}
  }

Utilizare:
  # toate rândurile cu segmente valide
  python generate_hint_concepts.py --ollama-model gpt-oss:20b

  # doar o problemă
  python generate_hint_concepts.py --filter-problem 2021_tema1_crypto --ollama-model gpt-oss:20b

  # test rapid pe 3 rânduri
  python generate_hint_concepts.py --limit 3 --ollama-model gpt-oss:20b
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

import orjson

from src.common.io_utils import read_jsonl
from src.common.paths import HINTS_DIR

SEGMENTS_JSONL_DEFAULT = HINTS_DIR / "segments_silver_diff.jsonl"
OUT_DEFAULT = HINTS_DIR / "hints_from_concepts.jsonl"

VALID_LEVELS = {"macro", "structural", "specific", "very_specific"}
VALID_GAP_LEVELS = {"intent", "algorithm", "data_structures", "key_operations", "potential_issues"}


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
Ești un tutore expert la cursul Programarea Algoritmilor (PA),
Universitatea Politehnica București.

Primești analiza semantică (intent, function, implementation, potential_issues)
a DOUĂ variante ale aceluiași exercițiu, generată prin EXACT același procedeu:
  - Codul GREȘIT al studentului (partial)
  - Codul CORECT de referință (100 de puncte)

Sarcina ta are DOUĂ părți:

PARTEA A — IDENTIFICĂ GAP-UL CONCEPTUAL
  Compară cele două analize semantice și identifică PRIMUL concept (din ordinea:
  intent → function.algorithm → function.data_structures → implementation.key_operations
  → implementation.potential_issues) care DIFERĂ sau LIPSEȘTE în codul greșit
  față de codul corect. Acesta este "gap_codes" — diferența principală pe care
  trebuie să o înțeleagă studentul.

PARTEA B — FORMULEAZĂ HINTURI GRADATE PE ACEST GAP
  Generează 1-4 hinturi GRADUALE care îl ghidează pe student către descoperirea
  gap-ului identificat la PARTEA A. Hint 1 = cel mai abstract, ultimul = cel mai
  concret. NU îi da soluția; doar îl ajuți să vadă diferența conceptuală.

────────────────────
REGULI CRITICE:

1. Bazează-te EXCLUSIV pe analiza semantică furnizată (intent, algoritm,
   data_structures, pași cheie, potential_issues) — pentru AMBELE coduri.
   Nu inventa informații.

2. Mappingul gap → nivel hint:
   - gap la intent             → hint macro (studentul rezolvă altă problemă)
   - gap la algorithm          → hint macro / structural (abordare greșită)
   - gap la data_structures    → hint structural
   - gap la key_operations     → hint structural / specific (pas lipsă sau greșit)
   - gap la potential_issues   → hint specific / very_specific (bug local)

3. FĂRĂ COD în hinturi — doar raționament și concepte.

4. FĂRĂ solution leak — nu reformula direct pașii din codul corect.

5. Ordinea: hint 1 = cel mai abstract, ultimul = cel mai concret. Fiecare hint
   adaugă informație nouă față de cel anterior.

6. Scurt: 1-3 propoziții per hint.

7. Totul în limba română.

────────────────────
Întoarce STRICT JSON, fără text suplimentar, cu forma:

{
  "gap_codes": {
    "level": "intent" | "algorithm" | "data_structures" | "key_operations" | "potential_issues",
    "missing_concept": "<scurt, în română: ce-i lipsește codului greșit>",
    "evidence_partial": "<citat din analiza codului greșit>",
    "evidence_passing": "<citat din analiza codului corect>"
  },
  "hints": [
    {"level": "macro",      "text": "..."},
    {"level": "structural", "text": "..."},
    {"level": "specific",   "text": "..."}
  ],
  "rationale_short": "1 propoziție: cum hinturile gradează studentul către gap_codes"
}

- gap_codes.level ∈ {"intent", "algorithm", "data_structures", "key_operations", "potential_issues"}
- hints[].level ∈ {"macro", "structural", "specific", "very_specific"}
- 1 până la 4 hinturi (nu mai mult de 4)
- Toate textele în română
""").strip()


# ── User prompt builder ────────────────────────────────────────────────────────

def _fmt_ops(ops: list[str]) -> str:
    if not ops:
        return "  (niciun pas extras)"
    return "\n".join(f"  {i+1}. {op}" for i, op in enumerate(ops))


def _fmt_issues(issues: list[str]) -> str:
    if not issues:
        return "  (nicio problemă identificată)"
    return "\n".join(f"  - {iss}" for iss in issues)


def build_concept_user_prompt(
    row: dict[str, Any],
    statement_excerpt: str,
) -> str:
    partial = row.get("partial_segments") or {}
    passing = row.get("passing_segments") or {}

    p_intent = partial.get("intent", {})
    p_func = partial.get("function", {})
    p_impl = partial.get("implementation", {})

    r_intent = passing.get("intent", {})
    r_func = passing.get("function", {})
    r_impl = passing.get("implementation", {})

    codebert_sim = row.get("embedding_similarity", None)
    sim_str = f"{codebert_sim:.4f}" if codebert_sim is not None else "necunoscută"
    sim_interp = (
        "cod aproape corect (bug local / detaliu)" if (codebert_sim or 0) >= 0.98
        else "diferențe structurale (logică)" if (codebert_sim or 0) >= 0.90
        else "abordare probabil greșită (concept)"
    )

    stmt_block = ""
    if statement_excerpt:
        stmt_block = f"""
────────────────────
ENUNȚ (extras):
{statement_excerpt[:1500]}
"""

    return textwrap.dedent(f"""
problem_id: {row.get("problem_id", "")}
verdict: {row.get("verdict", "WA")}
similaritate CodeBERT: {sim_str} → {sim_interp}
{stmt_block}
────────────────────
ANALIZA SEMANTICĂ — codul GREȘIT al studentului:

Intent: {p_intent.get("text", "(lipsă)")}
Potrivire problemă: {"DA" if p_intent.get("matches_problem", True) else "NU — studentul rezolvă altă problemă"}
Algoritm: {p_func.get("algorithm", "(lipsă)")}
Structuri de date: {", ".join(p_func.get("data_structures", [])) or "(niciuna)"}
Complexitate: {p_func.get("complexity", "necunoscută")}

Pași cheie din cod:
{_fmt_ops(p_impl.get("key_operations", []))}

Probleme identificate în cod:
{_fmt_issues(p_impl.get("potential_issues", []))}

────────────────────
ANALIZA SEMANTICĂ — codul CORECT (100 de puncte):

Intent: {r_intent.get("text", "(lipsă)")}
Potrivire problemă: {"DA" if r_intent.get("matches_problem", True) else "NU"}
Algoritm: {r_func.get("algorithm", "(lipsă)")}
Structuri de date: {", ".join(r_func.get("data_structures", [])) or "(niciuna)"}
Complexitate: {r_func.get("complexity", "necunoscută")}

Pași cheie din codul corect:
{_fmt_ops(r_impl.get("key_operations", []))}

Probleme identificate în codul corect:
{_fmt_issues(r_impl.get("potential_issues", []))}

────────────────────
SARCINA TA:

1. Identifică PRIMUL nivel la care cele două analize diferă, în ordinea:
   intent → algorithm → data_structures → key_operations → potential_issues.
   Pune asta în câmpul "gap_codes".

2. Generează 1-4 hinturi gradate (macro → very_specific) care să-l ghideze
   pe student către descoperirea gap-ului identificat la punctul 1.

Întoarce JSON conform schemei din system prompt.
""").strip()


# ── Ollama call ────────────────────────────────────────────────────────────────

def call_ollama(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    timeout: int = 300,
    ollama_url: str = "http://localhost:11434",
) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        f"{ollama_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


# ── JSON parsing ───────────────────────────────────────────────────────────────

def parse_hints_json(raw: str) -> tuple[dict | None, str]:
    for attempt in (raw, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)):
        text = attempt if isinstance(attempt, str) else attempt
        try:
            return json.loads(text), ""
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), ""
        except json.JSONDecodeError as e:
            return None, f"json parse failed: {e}"
    return None, "no JSON found"


def validate_hints(parsed: dict) -> list[str]:
    errors: list[str] = []

    # ── gap_codes ───────────────────────────────────────────────────────────
    gap = parsed.get("gap_codes")
    if gap is None:
        errors.append("'gap_codes' lipsă")
    elif not isinstance(gap, dict):
        errors.append("'gap_codes' nu este un dict")
    else:
        if gap.get("level") not in VALID_GAP_LEVELS:
            errors.append(f"gap_codes.level invalid: {gap.get('level')!r}")
        if not isinstance(gap.get("missing_concept"), str) or not gap["missing_concept"].strip():
            errors.append("gap_codes.missing_concept lipsă sau gol")

    # ── hints ───────────────────────────────────────────────────────────────
    hints = parsed.get("hints")
    if not isinstance(hints, list):
        errors.append("'hints' nu este o listă")
        return errors
    if not hints:
        errors.append("lista 'hints' este goală")
    if len(hints) > 4:
        errors.append(f"prea multe hinturi: {len(hints)} (max 4)")
    for i, h in enumerate(hints):
        if not isinstance(h, dict):
            errors.append(f"hints[{i}] nu este dict")
            continue
        if h.get("level") not in VALID_LEVELS:
            errors.append(f"hints[{i}].level invalid: {h.get('level')!r}")
        if not isinstance(h.get("text"), str) or not h["text"].strip():
            errors.append(f"hints[{i}].text lipsă sau gol")
    return errors


# ── Statement cache ────────────────────────────────────────────────────────────

def _load_statement(problem_id: str, packets_dir: Path) -> str:
    try:
        from src.stage2_annotation.prepare_problem_packets import statement_text_for_problem_id
        text, _ = statement_text_for_problem_id(problem_id, packets_dir)
        return text or ""
    except Exception:
        return ""


# ── Main batch ─────────────────────────────────────────────────────────────────

def main() -> None:
    from src.common.paths import PROCESSED_DIR
    packets_default = PROCESSED_DIR / "packets"

    parser = argparse.ArgumentParser(
        description=(
            "Generează hinturi gradate din segmentele semantice extrase de\n"
            "check_segment_consistency.py (segments_silver_diff.jsonl).\n\n"
            "Output: hints_from_concepts.jsonl — aceeași schemă ca silver_diff.jsonl."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--segments-jsonl", type=Path, default=SEGMENTS_JSONL_DEFAULT,
        help="JSONL de intrare cu segmente extrase (default: data/hints/segments_silver_diff.jsonl)",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DEFAULT,
        help="JSONL de ieșire (default: data/hints/hints_from_concepts.jsonl)",
    )
    parser.add_argument("--ollama-model", default="gpt-oss:20b")
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="URL server Ollama (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Temperatura LLM (default 0.3 — mai puțin aleator decât bootstrap)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout HTTP per apel Ollama în secunde (default 300)",
    )
    parser.add_argument(
        "--packets-dir", type=Path, default=packets_default,
        help="Folder cu packets pentru extragere enunț",
    )
    parser.add_argument(
        "--filter-problem", type=str, default=None,
        help="Procesează doar un singur problem_id",
    )
    parser.add_argument("--limit", type=int, default=0, help="Procesează cel mult N rânduri (0 = toate)")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Nu sări peste rândurile deja scrise în output",
    )
    parser.add_argument(
        "--skip-incomplete", action="store_true",
        help="Sari rândurile unde partial sau passing segments lipsesc (default: le scrie cu hints=[])",
    )
    args = parser.parse_args()

    if not args.segments_jsonl.exists():
        print(f"ERROR: fișierul nu există: {args.segments_jsonl}")
        return

    print("=" * 60)
    print("GENERATE HINT CONCEPTS")
    print("=" * 60)
    print(f"Input:   {args.segments_jsonl}")
    print(f"Output:  {args.out}")
    print(f"Model:   {args.ollama_model}  temp={args.temperature}")
    if args.filter_problem:
        print(f"Filter:  problem_id={args.filter_problem!r}")
    if args.limit:
        print(f"Limit:   {args.limit} rânduri")
    print()

    # Resume support
    done: set[tuple[str, str]] = set()
    if not args.no_resume and args.out.exists():
        for r in read_jsonl(args.out):
            done.add((r.get("problem_id", ""), r.get("submission_name", "")))
        print(f"Resuming — {len(done)} rânduri deja procesate, le sar.")

    rows = list(read_jsonl(args.segments_jsonl))
    if args.filter_problem:
        rows = [r for r in rows if r["problem_id"] == args.filter_problem]
    if args.limit:
        rows = rows[: args.limit]

    statement_cache: dict[str, str] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = skipped_incomplete = errors = 0

    for i, row in enumerate(rows, 1):
        problem_id = row.get("problem_id", "")
        submission_name = row.get("submission_name", "")
        key = (problem_id, submission_name)

        if key in done:
            skipped += 1
            continue

        partial_ok = (
            row.get("partial_segments") is not None
            and not row.get("partial_segment_errors")
        )
        passing_ok = (
            row.get("passing_segments") is not None
            and not row.get("passing_segment_errors")
        )

        if not partial_ok or not passing_ok:
            missing = []
            if not partial_ok:
                missing.append("partial_segments")
            if not passing_ok:
                missing.append("passing_segments")
            print(f"[{i}/{len(rows)}] {submission_name}: segmente lipsă ({', '.join(missing)})")
            if args.skip_incomplete:
                skipped_incomplete += 1
                continue
            # Write row without generated hints so it's resumable later
            out_row: dict[str, Any] = {
                "problem_id": problem_id,
                "anon_id": row.get("anon_id", ""),
                "submission_name": submission_name,
                "language": row.get("language", ""),
                "verdict": row.get("verdict", ""),
                "hints": [],
                "source": "concept_segments_incomplete",
                "embedding_similarity": row.get("embedding_similarity"),
                "passing_file": row.get("passing_file", ""),
                "partial_segments": row.get("partial_segments"),
                "partial_segment_errors": row.get("partial_segment_errors", []),
                "passing_segments": row.get("passing_segments"),
                "passing_segment_errors": row.get("passing_segment_errors", []),
                "_error": f"segmente lipsă: {missing}",
            }
            with open(args.out, "ab") as f:
                f.write(orjson.dumps(out_row, option=orjson.OPT_APPEND_NEWLINE))
            skipped_incomplete += 1
            continue

        # ── Load statement (cached) ──────────────────────────────────────────
        if problem_id not in statement_cache:
            statement_cache[problem_id] = _load_statement(problem_id, args.packets_dir)
        statement_excerpt = statement_cache[problem_id]

        # ── Build prompt ──────────────────────────────────────────────────────
        user_prompt = build_concept_user_prompt(row, statement_excerpt)

        print(f"[{i}/{len(rows)}] {problem_id} | {submission_name} ...", end=" ", flush=True)

        # ── Call LLM ──────────────────────────────────────────────────────────
        try:
            raw = call_ollama(
                model=args.ollama_model,
                system=SYSTEM_PROMPT,
                user=user_prompt,
                temperature=args.temperature,
                timeout=args.timeout,
                ollama_url=args.ollama_url,
            )
        except Exception as e:
            print(f"FAIL ({str(e)[:60]})")
            errors += 1
            # Write placeholder so resume skips it next time
            out_row = {
                "problem_id": problem_id,
                "anon_id": row.get("anon_id", ""),
                "submission_name": submission_name,
                "language": row.get("language", ""),
                "verdict": row.get("verdict", ""),
                "hints": [],
                "source": "concept_segments_llm_error",
                "embedding_similarity": row.get("embedding_similarity"),
                "passing_file": row.get("passing_file", ""),
                "partial_segments": row.get("partial_segments"),
                "passing_segments": row.get("passing_segments"),
                "_error": str(e),
            }
            with open(args.out, "ab") as f:
                f.write(orjson.dumps(out_row, option=orjson.OPT_APPEND_NEWLINE))
            continue

        # ── Parse & validate ──────────────────────────────────────────────────
        parsed, parse_err = parse_hints_json(raw)
        if parsed is None:
            print(f"PARSE_FAIL ({parse_err[:60]})")
            errors += 1
            out_row = {
                "problem_id": problem_id,
                "anon_id": row.get("anon_id", ""),
                "submission_name": submission_name,
                "language": row.get("language", ""),
                "verdict": row.get("verdict", ""),
                "hints": [],
                "source": "concept_segments_parse_error",
                "embedding_similarity": row.get("embedding_similarity"),
                "passing_file": row.get("passing_file", ""),
                "partial_segments": row.get("partial_segments"),
                "passing_segments": row.get("passing_segments"),
                "_error": parse_err,
                "_raw": raw[:400],
            }
            with open(args.out, "ab") as f:
                f.write(orjson.dumps(out_row, option=orjson.OPT_APPEND_NEWLINE))
            continue

        hint_errors = validate_hints(parsed)
        status = "OK" if not hint_errors else f"WARN({'; '.join(hint_errors)[:60]})"
        print(status)

        # ── Write output row ──────────────────────────────────────────────────
        hints = parsed.get("hints", [])
        # Cap at 4 if model ignores the rule
        hints = hints[:4]

        out_row = {
            "problem_id": problem_id,
            "anon_id": row.get("anon_id", ""),
            "submission_name": submission_name,
            "language": row.get("language", ""),
            "verdict": row.get("verdict", ""),
            "hints": hints,
            "source": "concept_segments",
            "embedding_similarity": row.get("embedding_similarity"),
            "passing_file": row.get("passing_file", ""),
            "validator_passed": not hint_errors,
            "validator_violations": hint_errors,
            "rationale_short": parsed.get("rationale_short", ""),
            "partial_segments": row.get("partial_segments"),
            "passing_segments": row.get("passing_segments"),
            # Keep original silver_diff hints for comparison
            "silver_hints": row.get("hints", []),
        }
        with open(args.out, "ab") as f:
            f.write(orjson.dumps(out_row, option=orjson.OPT_APPEND_NEWLINE))
        written += 1

    print(f"\n{'='*60}")
    print(f"DONE: {written} generate, {skipped} sări (resume), "
          f"{skipped_incomplete} incomplete, {errors} erori")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()

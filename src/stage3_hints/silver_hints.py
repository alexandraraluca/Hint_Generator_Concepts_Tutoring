"""Stage 3.B - silver hints from (failing <100) × (passing =100) pairs.

Pentru fiecare problemă annotată, citește soluțiile de pe disc:
``data/raw/solutions/solutions/<year>_<pid>/anon_<id>_<score>.(cpp|java)``.

1. **CodeBERT**: encode toate fișierele distincte (per limbă) din reuniunea
   passing ∪ failing; pentru fiecare pereche (fail, pass) calculează
   similaritatea cosinus între embedding-uri.

2. **Mode implicit (``--llm``, default)** pune în prompt și codul failing,
   codul de referință 100p *al aceluiași student*, rezumat diff + extras
   unified diff + scorul CodeBERT,
   apoi apelează Ollama (gpt-oss) pentru 1–4 hinturi rubrică, cu
   ``HintValidator`` + schemă ca la ``llm_bootstrap``.

3. **Mode rapid** ``--no-llm``: hinturi doar din template (fără Ollama).

Perechile sunt **numai în interiorul aceluiași anon_id** (aceeași limbă).
Per student: **un singur** fișier 100p (cel cu numele cel mai mic, stabil)
și **până la N** submisii eșuate (implicit 3, eșantion reproducebil cu
`--seed` dacă sunt mai multe). Astfel evităm produsul cartezian
(failing×passing) care explodează numărul de apeluri LLM.
Studenții nu se încrucișează între ei.

Cheie de resume: ``(problem_id, failing_basename, passing_basename)`` —
vezi ``submission_name`` + ``passing_file`` în JSONL.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.io_utils import read_json, read_jsonl
from src.common.ollama_client import OllamaClient, OllamaConfig
from src.common.paths import (
    ANNOTATIONS_DIR,
    CANONICAL_FILTERED_JSONL,
    EXTRACTED_SOLUTIONS_DIR,
    HINTS_DIR,
    PROCESSED_DIR,
    ensure_dirs,
)
from src.common.schemas import validate as schema_validate
from src.stage3_hints.code_embeddings import CodeBERTEncoder, encode_files_with_cache
from src.stage3_hints.diff_utils import code_diff
from src.stage3_hints.prompt_builder import (
    build_silver_pair_user_prompt,
    build_system_prompt_silver,
)
from src.stage3_hints.validator import HintValidator, cap_hints_to_rubric

SILVER_OUT = HINTS_DIR / "silver_diff.jsonl"
SILVER_INVALID_OUT = HINTS_DIR / "silver_diff_invalid.jsonl"
PACKETS_DIR = PROCESSED_DIR / "packets"

_FILE_RE = re.compile(
    r"^(?P<anon>anon_\d+)_(?P<score>\d+(?:\.\d+)?)\.(?P<ext>cpp|java)$",
    re.IGNORECASE,
)

_EXTRA_ROW_KEYS = frozenset(
    {
        "embedding_similarity",
        "passing_file",
        "validator_metrics",
        "_schema_errors",
        "_error",
    }
)


def _list_solutions(year: str, pid: str) -> list[tuple[Path, str, float, str]]:
    """Return list of (path, anon, score, language)."""
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


def _scores_to_buckets(score: float) -> str:
    if score >= 99.999:
        return "passing"
    return "failing"


def _pairs_one_pass_per_student(
    failing: list[tuple[Path, str, float, str]],
    passing: list[tuple[Path, str, float, str]],
    *,
    max_failing_per_student: int,
    rng: random.Random,
) -> list[
    tuple[tuple[Path, str, float, str], tuple[Path, str, float, str]]
]:
    """Perechi (fail, pass) cu același anon_id: 1×100p și până la N eșuate.

    - Codul reușit: unul singur per anon — primul după sortare după numele
      fișierului (reproducibil între rulări).
    - Codurile greșite: toate dacă sunt ≤N; altfel ``rng.sample(..., N)``,
      apoi sortate după nume pentru ordine stabilă în output.
    """
    k = max(1, max_failing_per_student)
    fail_by: dict[str, list[tuple[Path, str, float, str]]] = {}
    for s in failing:
        fail_by.setdefault(s[1], []).append(s)
    pass_by: dict[str, list[tuple[Path, str, float, str]]] = {}
    for s in passing:
        pass_by.setdefault(s[1], []).append(s)
    pairs: list[
        tuple[tuple[Path, str, float, str], tuple[Path, str, float, str]]
    ] = []
    for anon in sorted(set(fail_by) & set(pass_by)):
        flist = fail_by[anon]
        plist = pass_by[anon]
        p = sorted(plist, key=lambda x: x[0].name)[0]
        if len(flist) <= k:
            chosen_f = sorted(flist, key=lambda x: x[0].name)
        else:
            chosen_f = rng.sample(flist, k)
            chosen_f.sort(key=lambda x: x[0].name)
        for f in chosen_f:
            pairs.append((f, p))
    pairs.sort(key=lambda fp: (fp[0][0].name, fp[1][0].name))
    return pairs


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _hint_macro(diff: Any, primary_concept: str, difficulty: str) -> str:
    total = diff.n_lines_added + diff.n_lines_removed
    return (
        f"Soluția ta diferă de o variantă corectă pe aproximativ {total} linii "
        f"semnificative; concentrează-te pe ideea cheie a problemei "
        f"({primary_concept}, dificultate {difficulty})."
    )


def _hint_structural(diff: Any) -> str | None:
    add = diff.structural_added
    rem = diff.structural_removed
    if not add and not rem:
        return None
    parts = []
    if add:
        parts.append("introduce structuri: " + ", ".join(add[:4]))
    if rem:
        parts.append("renunță la structuri: " + ", ".join(rem[:4]))
    return "Ca să te apropii de soluția corectă, " + "; ".join(parts) + "."


def _hint_specific(diff: Any) -> str | None:
    if not diff.added_blocks:
        return None
    longest = max(diff.added_blocks, key=len)
    if not longest:
        return None
    keywords = []
    for ln in longest:
        for tok in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]+\b", ln):
            if tok.lower() in {
                "for", "while", "if", "else", "swap", "sort",
                "vector", "queue", "stack", "set", "map",
                "memset", "fill", "push", "pop", "insert",
                "ArrayList", "HashMap", "HashSet", "PriorityQueue",
            }:
                keywords.append(tok)
    keywords = list(dict.fromkeys(keywords))[:3]
    if not keywords:
        return (
            "Verifică partea finală a logicii principale: există un bloc "
            "consistent care lipsește din varianta ta."
        )
    return (
        "Verifică zona în care apar operațiile cu " + ", ".join(keywords)
        + "; acolo este localizată cea mai consistentă diferență."
    )


def _build_template_row(
    failing: tuple[Path, str, float, str],
    passing: tuple[Path, str, float, str],
    *,
    problem_meta: dict[str, Any],
    similarity: float,
) -> dict[str, Any]:
    f_path, f_anon, _, f_lang = failing
    p_path, _, _, _p_lang = passing
    f_code = f_path.read_text(encoding="utf-8", errors="replace")
    p_code = p_path.read_text(encoding="utf-8", errors="replace")
    diff = code_diff(f_code, p_code)
    hints: list[dict[str, Any]] = []
    hints.append(
        {
            "level": "macro",
            "text": _hint_macro(
                diff,
                primary_concept=problem_meta.get("primary_concept", "concept central"),
                difficulty=problem_meta.get("difficulty", "medium"),
            ),
        }
    )
    h_struct = _hint_structural(diff)
    if h_struct:
        hints.append({"level": "structural", "text": h_struct})
    h_spec = _hint_specific(diff)
    if h_spec:
        hints.append({"level": "specific", "text": h_spec})

    pc = problem_meta.get("primary_concept", "")
    concepts = [pc] if pc else []

    return {
        "problem_id": problem_meta["problem_id"],
        "anon_id": f_anon,
        "submission_name": f_path.name,
        "language": f_lang,
        "verdict": "WA",
        "issues": [],
        "concepts_targeted": concepts,
        "hints": hints,
        "source": "silver_diff",
        "validator_passed": True,
        "validator_violations": [],
        "embedding_similarity": round(float(similarity), 4),
        "passing_file": p_path.name,
    }


def _statement_excerpt_for(problem_id: str) -> str:
    p = PACKETS_DIR / f"{problem_id}.json"
    if not p.exists():
        return ""
    try:
        return read_json(p).get("statement_text", "") or ""
    except OSError:
        return ""


def _representative_solution_for(problem_id: str) -> str:
    p = PACKETS_DIR / f"{problem_id}.json"
    if not p.exists():
        return ""
    try:
        reps = read_json(p).get("representative_solutions", []) or []
        return reps[0]["code"] if reps else ""
    except (OSError, KeyError, IndexError):
        return ""


def _verdict(pts: float, issues: list[str]) -> str:
    if pts >= 99.999:
        return "OK"
    s = " ".join(issues or []).lower()
    if any(k in s for k in ("compile", "compilation", "ce")):
        return "CE"
    if any(k in s for k in ("runtime", "segmentation")):
        return "RE"
    if any(k in s for k in ("tle", "time limit")):
        return "TLE"
    if any(k in s for k in ("mle", "memory limit")):
        return "MLE"
    if "wa" in s or pts < 100:
        return "WA"
    return "OTHER"


def _issues_lookup() -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not CANONICAL_FILTERED_JSONL.is_file():
        return out
    for r in read_jsonl(CANONICAL_FILTERED_JSONL):
        key = (r["year"], r["pid"], r.get("anon_id", ""))
        if key not in out:
            out[key] = r
    return out


def _existing_pair_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for row in read_jsonl(path):
        pf = row.get("passing_file")
        if not pf:
            continue
        keys.add(
            (
                row.get("problem_id", ""),
                row.get("submission_name") or "",
                str(pf),
            )
        )
    return keys


def _existing_pair_counts_by_problem(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for row in read_jsonl(path):
        problem_id = row.get("problem_id")
        if not problem_id:
            continue
        counts[problem_id] = counts.get(problem_id, 0) + 1
    return counts


def _existing_pair_counts_by_problem_and_lang(path: Path) -> dict[str, dict[str, int]]:
    """Return counts by problem_id then by language (e.g. {'2021_x': {'cpp': 3, 'java': 2}})."""
    if not path.exists():
        return {}
    counts: dict[str, dict[str, int]] = {}
    for row in read_jsonl(path):
        problem_id = row.get("problem_id")
        lang = row.get("language")
        if not problem_id or not lang:
            continue
        counts.setdefault(problem_id, {})
        counts[problem_id][lang] = counts[problem_id].get(lang, 0) + 1
    return counts


def _shareable_hints_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in _EXTRA_ROW_KEYS}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-pairs-per-problem",
        type=int,
        default=0,
        help=(
            "0 = fără tăiere suplimentară; "
            "altfel după construirea perechilor (1 pass × N fail/student), "
            "amestecă și păstrează primele K perechi"
        ),
    )
    parser.add_argument(
        "--max-total-pairs-per-problem",
        type=int,
        default=0,
        help=(
            "0 = fără limită totală; altfel, numărul maxim de perechi pentru "
            "problemă este calculat ca totalul istoric din silver_diff.jsonl plus "
            "perechile noi din rularea curentă, peste cpp + java"
        ),
    )
    parser.add_argument(
        "--max-failing-per-student",
        type=int,
        default=3,
        help=(
            "per student și limbă: până la N submisii eșunate față de același 100p; "
            "dacă sunt mai multe, se aleg N la întâmplare cu --seed"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="random seed (eșantion eșuate dacă > max-failing; max-pairs-per-problem)",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="comma-separated problem_ids to limit to (debug)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="CodeBERT batch size"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="doar template mecanic (fără Ollama); implicit se folosește LLM",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.35,
        help="temperatură Ollama (doar --llm)",
    )
    parser.add_argument(
        "--prefer-min-hints",
        type=int,
        default=3,
        help="în prompt LLM: țintește cel puțin N trepte când diff-ul e bogat",
    )
    args = parser.parse_args()

    use_llm = not args.no_llm

    ensure_dirs()
    rng = random.Random(args.seed)

    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    pid_filter = (
        set(s.strip() for s in args.problems.split(",")) if args.problems else None
    )
    problems = [
        p for p in problems if (pid_filter is None or p["problem_id"] in pid_filter)
    ]
    print(f"running on {len(problems)} problems (llm={use_llm})")

    dag = read_json(ANNOTATIONS_DIR / "concepts_dag.json")
    valid_concept_ids = [c["id"] for c in dag["concepts"]]

    issues_lookup = _issues_lookup()
    existing = _existing_pair_keys(SILVER_OUT)
    existing_counts_by_problem_and_lang = _existing_pair_counts_by_problem_and_lang(
        SILVER_OUT
    )
    print(f"existing silver pairs to skip: {len(existing)}")

    client: OllamaClient | None = None
    if use_llm:
        cfg = OllamaConfig()
        cfg.temperature = args.temperature
        client = OllamaClient(cfg)
        if not client.health():
            print("ERROR: Ollama not reachable (sau rulează fără LLM: --no-llm).")
            return 2
        print(f"Ollama model={cfg.model} temp={cfg.temperature}")

    validator = HintValidator()
    sys_prompt = build_system_prompt_silver()

    encoder = CodeBERTEncoder()
    silver_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    t0 = time.time()
    # split total cap evenly between cpp and java (floor/ceil)
    total_cap = args.max_total_pairs_per_problem
    cap_cpp = cap_java = 0
    if total_cap:
        cap_cpp = total_cap // 2
        cap_java = total_cap - cap_cpp

    for prob in tqdm(problems, desc="silver-pairs"):
        pid_short = prob["pid"]
        year = prob["year"]
        # per-language historical counts for this problem
        existing_lang_counts = existing_counts_by_problem_and_lang.get(
            prob["problem_id"], {}
        )
        existing_cpp = existing_lang_counts.get("cpp", 0)
        existing_java = existing_lang_counts.get("java", 0)
        # if both languages already hit their share, skip the problem
        if total_cap and existing_cpp >= cap_cpp and existing_java >= cap_java:
            continue
        all_sols = _list_solutions(year, pid_short)
        passing = [s for s in all_sols if _scores_to_buckets(s[2]) == "passing"]
        failing = [s for s in all_sols if _scores_to_buckets(s[2]) == "failing"]
        if not passing or not failing:
            continue

        statement = _statement_excerpt_for(prob["problem_id"])
        gold_solution = _representative_solution_for(prob["problem_id"])

        # iterate languages; for a single-language run use a one-element tuple: ("cpp",)
        for lang in ("cpp", "java"):
            # per-language cap and existing count
            problem_pairs_added = existing_lang_counts.get(lang, 0)
            cap_lang = cap_cpp if lang == "cpp" else cap_java
            if total_cap and problem_pairs_added >= cap_lang:
                break
            f_lang = [s for s in failing if s[3] == lang]
            p_lang = [s for s in passing if s[3] == lang]
            if not f_lang or not p_lang:
                continue

            pairs = _pairs_one_pass_per_student(
                f_lang,
                p_lang,
                max_failing_per_student=args.max_failing_per_student,
                rng=rng,
            )
            if not pairs:
                continue
            if args.max_pairs_per_problem and len(pairs) > args.max_pairs_per_problem:
                rng.shuffle(pairs)
                pairs = pairs[: args.max_pairs_per_problem]

            unique_paths = sorted(
                {f[0] for f, p in pairs} | {p[0] for f, p in pairs},
                key=lambda x: x.name,
            )
            try:
                emb_arr, path_strs = encode_files_with_cache(
                    unique_paths,
                    encoder=encoder,
                    batch_size=args.batch_size,
                    cache_name=f"sols_{lang}.npz",
                )
            except Exception as e:  # noqa: BLE001
                invalid_rows.append(
                    {
                        "problem_id": prob["problem_id"],
                        "language": lang,
                        "_error": f"embedding error: {e!r}",
                    }
                )
                continue

            path_to_i = {
                Path(ps).resolve(): idx for idx, ps in enumerate(path_strs)
            }

            for fail_sol, pass_sol in pairs:
                f_path, f_anon, f_score, _ = fail_sol
                p_path, p_anon, _p_score, _ = pass_sol
                if f_anon != p_anon:
                    continue
                pair_key = (prob["problem_id"], f_path.name, p_path.name)
                if pair_key in existing:
                    continue

                fi = path_to_i.get(f_path.resolve())
                pi = path_to_i.get(p_path.resolve())
                if fi is None or pi is None:
                    continue
                sim = _cosine_rows(emb_arr[fi], emb_arr[pi])

                if use_llm:
                    try:
                        f_code = f_path.read_text(encoding="utf-8", errors="replace")
                        p_code = p_path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    diff = code_diff(f_code, p_code)
                    rec = issues_lookup.get((year, pid_short, f_anon), {})
                    issues = rec.get("issues") or []
                    verdict = _verdict(f_score, issues)

                    user_prompt = build_silver_pair_user_prompt(
                        problem_meta=prob,
                        statement_excerpt=statement,
                        failing_code=f_code,
                        reference_passing_code=p_code,
                        verdict=verdict,
                        issues=issues,
                        valid_concept_ids=valid_concept_ids,
                        codebert_similarity=sim,
                        diff_summary=diff.to_summary(),
                        diff_unified_excerpt=diff.unified_excerpt or "",
                        passing_file_hint=p_path.name,
                        prefer_min_hints=args.prefer_min_hints,
                    )
                    assert client is not None
                    try:
                        print(
                            f"[LLM] calling Ollama model for {prob['problem_id']} {f_path.name} -> {p_path.name} anon={f_anon}",
                            flush=True,
                        )
                        t_call = time.time()
                        result = client.chat_json(system=sys_prompt, user=user_prompt)
                        print(
                            f"[LLM] Ollama returned in {time.time() - t_call:.1f}s for {prob['problem_id']} {f_path.name}",
                            flush=True,
                        )
                    except Exception as e:  # noqa: BLE001
                        inv = {
                            "problem_id": prob["problem_id"],
                            "anon_id": f_anon,
                            "submission_name": f_path.name,
                            "passing_file": p_path.name,
                            "_error": f"LLM: {e!r}",
                        }
                        invalid_rows.append(inv)
                        with open(SILVER_INVALID_OUT, "ab") as fb:
                            fb.write(orjson.dumps(inv, option=orjson.OPT_APPEND_NEWLINE))
                        continue

                    hints = cap_hints_to_rubric(result.get("hints") or [])
                    concepts_targeted = [
                        c
                        for c in (result.get("concepts_targeted") or [])
                        if c in valid_concept_ids
                    ]
                    row = {
                        "problem_id": prob["problem_id"],
                        "anon_id": f_anon,
                        "submission_name": f_path.name,
                        "language": lang,
                        "verdict": verdict,
                        "issues": issues,
                        "concepts_targeted": concepts_targeted,
                        "hints": hints,
                        "source": "silver_diff",
                        "embedding_similarity": round(sim, 4),
                        "passing_file": p_path.name,
                    }
                    if not hints:
                        inv = {**row, "_error": "llm returned no hints"}
                        invalid_rows.append(inv)
                        with open(SILVER_INVALID_OUT, "ab") as fb:
                            fb.write(orjson.dumps(inv, option=orjson.OPT_APPEND_NEWLINE))
                        continue

                    try:
                        hints, rep = validator.validate_with_order_retry(
                            hints,
                            statement=statement,
                            solution_code=gold_solution,
                        )
                    except Exception as e:  # noqa: BLE001
                        inv = {**row, "_error": f"validator: {e!r}"}
                        invalid_rows.append(inv)
                        with open(SILVER_INVALID_OUT, "ab") as fb:
                            fb.write(orjson.dumps(inv, option=orjson.OPT_APPEND_NEWLINE))
                        continue

                    row["hints"] = hints
                    row["validator_passed"] = rep.passed
                    row["validator_violations"] = rep.violations + sum(
                        rep.per_hint_violations, []
                    )
                    row["validator_metrics"] = rep.metrics

                    shareable = _shareable_hints_row(row)
                    schema_errs = schema_validate("hints", shareable)
                    if schema_errs:
                        inv = {**row, "_schema_errors": schema_errs}
                        inv.pop("validator_metrics", None)
                        invalid_rows.append(inv)
                        with open(SILVER_INVALID_OUT, "ab") as fb:
                            fb.write(orjson.dumps(inv, option=orjson.OPT_APPEND_NEWLINE))
                        continue

                    if not rep.passed:
                        invalid_rows.append(row)
                        row_invalid = dict(row)
                        row_invalid.pop("validator_metrics", None)
                        with open(SILVER_INVALID_OUT, "ab") as fb:
                            fb.write(
                                orjson.dumps(
                                    row_invalid,
                                    option=orjson.OPT_APPEND_NEWLINE,
                                )
                            )
                        continue

                    row_out = dict(row)
                    row_out.pop("validator_metrics", None)
                    with open(SILVER_OUT, "ab") as fb:
                        fb.write(orjson.dumps(row_out, option=orjson.OPT_APPEND_NEWLINE))
                    silver_rows.append(row_out)
                    existing.add(pair_key)
                    problem_pairs_added += 1
                    # update our in-memory existing count so the other branch sees it
                    existing_lang_counts[lang] = problem_pairs_added
                    if total_cap and problem_pairs_added >= cap_lang:
                        break
                else:
                    row = _build_template_row(
                        fail_sol, pass_sol, problem_meta=prob, similarity=sim
                    )
                    try:
                        hints_tpl, rep = validator.validate_with_order_retry(
                            row["hints"],
                            statement=statement,
                            solution_code=gold_solution,
                        )
                    except Exception as e:  # noqa: BLE001
                        row["_error"] = f"validator: {e!r}"
                        invalid_rows.append(row)
                        with open(SILVER_INVALID_OUT, "ab") as fb:
                            fb.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
                        continue
                    row["hints"] = hints_tpl
                    row["validator_passed"] = rep.passed
                    row["validator_violations"] = rep.violations + sum(
                        rep.per_hint_violations, []
                    )
                    shareable = _shareable_hints_row(row)
                    schema_errs = schema_validate("hints", shareable)
                    if schema_errs:
                        row["_schema_errors"] = schema_errs
                        invalid_rows.append(row)
                        with open(SILVER_INVALID_OUT, "ab") as fb:
                            fb.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
                        continue
                    if not rep.passed:
                        invalid_rows.append(row)
                        with open(SILVER_INVALID_OUT, "ab") as fb:
                            fb.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
                        continue
                    row.pop("validator_metrics", None)
                    with open(SILVER_OUT, "ab") as fb:
                        fb.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
                    silver_rows.append(row)
                    existing.add(pair_key)
                    problem_pairs_added += 1
                    existing_lang_counts[lang] = problem_pairs_added
                    if total_cap and problem_pairs_added >= cap_lang:
                        break

    encoder.close()
    if client is not None:
        client.close()

    n_new = len(silver_rows)
    print(
        f"finished: {n_new} new silver rows this run in {time.time() - t0:.1f}s "
        f"(invalid events logged: {len(invalid_rows)})"
    )
    print(f"  -> append/write {SILVER_OUT}")
    if invalid_rows:
        print(f"  -> {SILVER_INVALID_OUT}")
    if use_llm:
        print(
            "Note: se face append la silver_diff.jsonl; șterge fișierul pentru o regenerare curată."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 3.C - LLM bootstrap of hint sets.

For each (failing_submission, problem) pair we ask `gpt-oss:20b` (via
Ollama) to produce 1-4 graded hints respecting the strict rubric encoded
in `prompt_builder.py`. We then run `validator.HintValidator` on every
generated set; valid sets go to `data/hints/llm_bootstrap.jsonl`,
invalid (with reasons) go to `data/hints/llm_bootstrap_invalid.jsonl`.

Notes:
- We need actual *failing code*, not just a row from canonical_filtered.
  We index failing solutions on disk by (problem_id, anon_id) and pair
  them with the canonical row that has the matching anon. If multiple
  files exist for the same anon (e.g. cpp + java), we pick the one
  whose score in filename matches `pts` from canonical (best effort).
- We cap submissions per problem at `--per-problem` (default 60) to keep
  runtime reasonable: ~30s per LLM call × 60 × 32 ≈ 16h. The user can
  reduce or run overnight.
- Resumable: skips (problem_id, anon_id) pairs already produced in a prior
  run.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.common.io_utils import read_json, read_jsonl, write_jsonl
from src.common.ollama_client import OllamaClient, OllamaConfig
from src.common.paths import (
    ANNOTATIONS_DIR,
    CANONICAL_FILTERED_JSONL,
    EXTRACTED_SOLUTIONS_DIR,
    HINTS_DIR,
    ensure_dirs,
)
from src.common.schemas import validate as schema_validate
from src.stage3_hints.prompt_builder import build_system_prompt, build_user_prompt
from src.stage3_hints.validator import HintValidator, cap_hints_to_rubric

BOOT_OUT = HINTS_DIR / "llm_bootstrap.jsonl"
BOOT_INVALID = HINTS_DIR / "llm_bootstrap_invalid.jsonl"

_FILE_RE = re.compile(
    r"^(?P<anon>anon_\d+)_(?P<score>\d+(?:\.\d+)?)\.(?P<ext>cpp|java)$",
    re.IGNORECASE,
)


def _index_solutions_by_problem() -> dict[str, list[tuple[Path, str, float, str]]]:
    """Walk solutions/ once and index files by problem_id."""
    index: dict[str, list[tuple[Path, str, float, str]]] = defaultdict(list)
    base = EXTRACTED_SOLUTIONS_DIR / "solutions"
    if not base.exists():
        return index
    for d in base.iterdir():
        if not d.is_dir():
            continue
        # folder name: <year>_<pid>
        parts = d.name.split("_", 1)
        if len(parts) != 2:
            continue
        year, pid = parts
        # we don't know tema from solutions; we'll match later via problems.json
        for f in d.iterdir():
            if not f.is_file():
                continue
            m = _FILE_RE.match(f.name)
            if not m:
                continue
            index[f"{year}|{pid}"].append(
                (f, m.group("anon"), float(m.group("score")), m.group("ext").lower())
            )
    return index


def _pick_failing_for_anon(
    pool: list[tuple[Path, str, float, str]],
    anon: str,
    target_pts: float,
    target_lang: str | None = None,
) -> tuple[Path, str, float, str] | None:
    """Pick the file for `anon` whose score is closest to `target_pts`."""
    cand = [t for t in pool if t[1] == anon]
    if target_lang:
        cand = [t for t in cand if t[3] == target_lang] or cand
    if not cand:
        return None
    cand.sort(key=lambda t: abs(t[2] - target_pts))
    return cand[0]


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


def _statement_excerpt_for(problem_id: str, packets_dir: Path) -> str:
    p = packets_dir / f"{problem_id}.json"
    if not p.exists():
        return ""
    try:
        return read_json(p).get("statement_text", "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _representative_solution_for(problem_id: str, packets_dir: Path) -> str:
    p = packets_dir / f"{problem_id}.json"
    if not p.exists():
        return ""
    try:
        reps = read_json(p).get("representative_solutions", []) or []
        return reps[0]["code"] if reps else ""
    except Exception:  # noqa: BLE001
        return ""


def _existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        keys.add(
            (
                row.get("problem_id", ""),
                row.get("anon_id", ""),
                row.get("submission_name", "") or "",
            )
        )
    return keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-problem",
        type=int,
        default=15,
        help="max number of failing submissions to bootstrap per problem (default 15 -> ~4-5h total)",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = no global cap")
    parser.add_argument(
        "--problems", type=str, default=None, help="comma-list of problem_ids"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4, help="LLM temperature"
    )
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    ensure_dirs()

    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    pid_filter = (
        set(s.strip() for s in args.problems.split(",")) if args.problems else None
    )
    problems = {p["problem_id"]: p for p in problems if (pid_filter is None or p["problem_id"] in pid_filter)}
    print(f"running on {len(problems)} annotated problems")

    # iterate over failing FILES on disk (the most reliable input)
    sol_index = _index_solutions_by_problem()

    # also build a lookup (year, pid, anon) -> issues from canonical for context
    canonical = list(read_jsonl(CANONICAL_FILTERED_JSONL))
    issues_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in canonical:
        key = (r["year"], r["pid"], r.get("anon_id", ""))
        if key not in issues_lookup:
            issues_lookup[key] = r
    print(f"canonical rows for context: {len(issues_lookup)}")

    import random
    rng = random.Random(args.seed)

    failing_by_pid: dict[str, list[tuple[Path, str, float, str]]] = defaultdict(list)
    for problem_id, prob in problems.items():
        year, pid = prob["year"], prob["pid"]
        pool = sol_index.get(f"{year}|{pid}", [])
        failing = [t for t in pool if t[2] < 99.999]
        random.Random(args.seed).shuffle(failing)
        failing_by_pid[problem_id] = failing[: args.per_problem]
        print(
            f"  {problem_id}: {len(failing)} failing on disk, "
            f"sampling {len(failing_by_pid[problem_id])}"
        )

    packets_dir = ANNOTATIONS_DIR.parent / "processed" / "packets"
    dag = read_json(ANNOTATIONS_DIR / "concepts_dag.json")
    valid_concept_ids = [c["id"] for c in dag["concepts"]]

    cfg = OllamaConfig()
    cfg.temperature = args.temperature
    client = OllamaClient(cfg)
    if not client.health():
        print("ERROR: Ollama not reachable.")
        return 2
    print(f"using model={cfg.model} temp={cfg.temperature}")

    validator = HintValidator()

    existing = _existing_keys(BOOT_OUT)
    written = 0
    invalid = 0

    work: list[tuple[str, tuple[Path, str, float, str]]] = []
    for problem_id, files in failing_by_pid.items():
        for t in files:
            work.append((problem_id, t))
    if args.limit:
        work = work[: args.limit]

    sys_prompt = build_system_prompt()

    t0 = time.time()
    for problem_id, file_tuple in tqdm(work, desc="llm bootstrap"):
        f_path, anon, f_score, f_lang = file_tuple
        sub_name = ""  # we don't have submission_name when iterating files
        if (problem_id, anon, sub_name) in existing:
            continue

        prob = problems[problem_id]

        try:
            failing_code = f_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            invalid += 1
            continue

        statement = _statement_excerpt_for(problem_id, packets_dir)
        gold_solution = _representative_solution_for(problem_id, packets_dir)

        # pull issues from canonical lookup, if the user has a record
        issues_record = issues_lookup.get(
            (prob["year"], prob["pid"], anon), {}
        )
        issues = issues_record.get("issues") or []
        verdict = _verdict(f_score, issues)

        user_prompt = build_user_prompt(
            problem_meta=prob,
            statement_excerpt=statement,
            failing_code=failing_code,
            verdict=verdict,
            issues=issues,
            valid_concept_ids=valid_concept_ids,
        )

        try:
            result = client.chat_json(
                system=sys_prompt, user=user_prompt
            )
        except Exception as e:  # noqa: BLE001
            invalid += 1
            with open(BOOT_INVALID, "ab") as f:
                import orjson
                f.write(
                    orjson.dumps(
                        {
                            "problem_id": problem_id,
                            "anon_id": anon,
                            "submission_name": sub_name,
                            "_error": f"LLM error: {e!r}",
                        },
                        option=orjson.OPT_APPEND_NEWLINE,
                    )
                )
            continue

        hints = cap_hints_to_rubric(result.get("hints") or [])
        concepts_targeted = [
            c
            for c in (result.get("concepts_targeted") or [])
            if c in valid_concept_ids
        ]

        candidate = {
            "problem_id": problem_id,
            "anon_id": anon,
            "submission_name": sub_name,
            "language": f_lang,
            "verdict": verdict,
            "issues": issues,
            "concepts_targeted": concepts_targeted,
            "hints": hints,
            "source": "llm_bootstrap",
        }

        # short-circuit: empty hints are an obvious LLM failure
        if not hints:
            candidate["_error"] = "llm returned no hints"
            invalid += 1
            with open(BOOT_INVALID, "ab") as f:
                import orjson
                f.write(orjson.dumps(candidate, option=orjson.OPT_APPEND_NEWLINE))
            continue

        # rubric validation (defensive: never let a validator crash kill the loop)
        try:
            hints, rep = validator.validate_with_order_retry(
                hints,
                statement=statement,
                solution_code=gold_solution,
            )
        except Exception as e:  # noqa: BLE001
            candidate["_error"] = f"validator crashed: {e!r}"
            invalid += 1
            with open(BOOT_INVALID, "ab") as f:
                import orjson
                f.write(orjson.dumps(candidate, option=orjson.OPT_APPEND_NEWLINE))
            continue
        candidate["hints"] = hints
        candidate["validator_passed"] = rep.passed
        candidate["validator_violations"] = rep.violations + sum(
            rep.per_hint_violations, []
        )
        candidate["validator_metrics"] = rep.metrics

        # schema validation
        shareable = {k: v for k, v in candidate.items() if k != "validator_metrics"}
        schema_errs = schema_validate("hints", shareable)
        if schema_errs:
            candidate["_schema_errors"] = schema_errs
            invalid += 1
            with open(BOOT_INVALID, "ab") as f:
                import orjson
                f.write(orjson.dumps(candidate, option=orjson.OPT_APPEND_NEWLINE))
            continue

        if not rep.passed:
            invalid += 1
            with open(BOOT_INVALID, "ab") as f:
                import orjson
                f.write(orjson.dumps(candidate, option=orjson.OPT_APPEND_NEWLINE))
            continue

        with open(BOOT_OUT, "ab") as f:
            import orjson
            f.write(orjson.dumps(candidate, option=orjson.OPT_APPEND_NEWLINE))
        written += 1

    elapsed = time.time() - t0
    print(
        f"\nfinished: {written} valid, {invalid} invalid, "
        f"in {elapsed:.0f}s ({elapsed / 60:.1f} min)"
    )
    print(f"  -> {BOOT_OUT}")
    if invalid:
        print(f"  -> {BOOT_INVALID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

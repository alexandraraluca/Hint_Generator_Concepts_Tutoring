"""Run prompt ablation: 6 variants × 3 temperatures = 18 LLM calls.

Default target:
  data/raw/solutions/solutions/2021_adrese/anon_1474_40.cpp
  (problem_id: 2021_tema2_adrese, passing pair: anon_1474_100.cpp)

Usage (from repo root, Ollama running):
  python discussion/prompt_ablation_run.py
  python discussion/prompt_ablation_run.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import orjson

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discussion.ollama_helpers import AblationOllamaClient, AblationOllamaConfig
from discussion.prompt_variants import PROMPT_VARIANTS, build_prompts, is_bootstrap_variant
from src.common.io_utils import read_json, read_jsonl
from src.common.paths import ANNOTATIONS_DIR, CANONICAL_FILTERED_JSONL, PROCESSED_DIR
from src.common.schemas import validate as schema_validate
from src.stage3_hints.code_embeddings import CodeBERTEncoder
from src.stage3_hints.diff_utils import code_diff
from src.stage3_hints.validator import HintValidator, cap_hints_to_rubric

OUT_DIR = Path(__file__).resolve().parent / "output"
PACKETS_DIR = PROCESSED_DIR / "packets"

DEFAULT_FAILING = (
    # ROOT / "data" / "raw" / "solutions" / "solutions" / "2021_adrese" / "anon_1474_40.cpp"
    ROOT / "data" / "raw" / "solutions" / "solutions" / "2022_curatare" / "anon_1195_65.java"
)
DEFAULT_PASSING = (
    # ROOT / "data" / "raw" / "solutions" / "solutions" / "2021_adrese" / "anon_1474_100.cpp"
    ROOT / "data" / "raw" / "solutions" / "solutions" / "2022_curatare" / "anon_1195_100.java"
)
# DEFAULT_PROBLEM_ID = "2021_tema2_adrese"
DEFAULT_PROBLEM_ID = "2022_tema2_curatare"

VARIANTS = list(PROMPT_VARIANTS.keys())
TEMPERATURES = [0.2, 0.6, 0.9]


def _verdict(score: float, issues: list[str]) -> str:
    if score >= 99.999:
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
    if "wa" in s or score < 100:
        return "WA"
    return "OTHER"


def _statement(problem_id: str) -> str:
    p = PACKETS_DIR / f"{problem_id}.json"
    if not p.exists():
        return ""
    return read_json(p).get("statement_text", "") or ""


def _gold_solution(problem_id: str) -> str:
    p = PACKETS_DIR / f"{problem_id}.json"
    if not p.exists():
        return ""
    reps = read_json(p).get("representative_solutions", []) or []
    return reps[0]["code"] if reps else ""


def _issues_for(year: str, pid: str, anon: str) -> list[str]:
    for r in read_jsonl(CANONICAL_FILTERED_JSONL):
        if r.get("year") == year and r.get("pid") == pid and r.get("anon_id") == anon:
            return r.get("issues") or []
    return []


def _cosine(a, b) -> float:
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt ablation runner (discussion/)")
    parser.add_argument("--failing-file", type=Path, default=DEFAULT_FAILING)
    parser.add_argument("--passing-file", type=Path, default=DEFAULT_PASSING)
    parser.add_argument("--problem-id", type=str, default=DEFAULT_PROBLEM_ID)
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.2,0.6,0.9",
        help="comma-separated temperatures",
    )
    parser.add_argument("--dry-run", action="store_true", help="build prompts only, no Ollama")
    parser.add_argument("--model", type=str, default=None, help="override OLLAMA_MODEL")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated variant ids (e.g. silver,silver_full_diff)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="skip runs already marked status=ok in output jsonl",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="truncate output jsonl before run",
    )
    args = parser.parse_args()

    failing_path: Path = args.failing_file
    passing_path: Path = args.passing_file
    if not failing_path.exists():
        print(f"ERROR: missing failing file: {failing_path}")
        return 1
    if not passing_path.exists():
        print(f"ERROR: missing passing file: {passing_path}")
        return 1

    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    prob = next((p for p in problems if p["problem_id"] == args.problem_id), None)
    if prob is None:
        print(f"ERROR: problem_id not in annotations: {args.problem_id}")
        return 1

    dag = read_json(ANNOTATIONS_DIR / "concepts_dag.json")
    valid_concept_ids = [c["id"] for c in dag["concepts"]]

    failing_code = failing_path.read_text(encoding="utf-8", errors="replace")
    passing_code = passing_path.read_text(encoding="utf-8", errors="replace")
    statement = _statement(args.problem_id)
    gold = _gold_solution(args.problem_id)

    # anon = "anon_1474"
    anon = "anon_1195"
    if "anon_" in failing_path.stem:
        anon = failing_path.stem.rsplit("_", 1)[0]
    score = 40.0
    try:
        score = float(failing_path.stem.rsplit("_", 1)[-1])
    except ValueError:
        pass
    issues = _issues_for(prob["year"], prob["pid"], anon)
    verdict = _verdict(score, issues)

    diff = code_diff(failing_code, passing_code)
    with CodeBERTEncoder() as encoder:
        emb_f = encoder.encode([failing_code])[0]
        emb_p = encoder.encode([passing_code])[0]
        codebert_sim = _cosine(emb_f, emb_p)

    temps = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]
    variant_list = VARIANTS
    if args.only:
        wanted = {v.strip() for v in args.only.split(",") if v.strip()}
        variant_list = [v for v in VARIANTS if v in wanted]
        unknown = wanted - set(variant_list)
        if unknown:
            print(f"ERROR: unknown variants: {sorted(unknown)}")
            return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # out_jsonl = OUT_DIR / "ablation_anon_1474_adrese.jsonl"
    # summary_md = OUT_DIR / "ablation_anon_1474_adrese.md"
    out_jsonl = OUT_DIR / "ablation_anon_1195_curatare.jsonl"
    summary_md = OUT_DIR / "ablation_anon_1195_curatare.md"

    ok_run_ids: set[str] = set()
    if out_jsonl.exists() and not args.fresh:
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = orjson.loads(line)
            if row.get("status") == "ok":
                ok_run_ids.add(row.get("run_id", ""))
    if args.fresh and out_jsonl.exists():
        out_jsonl.unlink()

    validator = HintValidator()
    client: AblationOllamaClient | None = None
    if not args.dry_run:
        cfg = AblationOllamaConfig()
        if args.model:
            cfg.model = args.model
        client = AblationOllamaClient(cfg)
        if not client.health():
            print("ERROR: Ollama not reachable. Start ollama serve or use --dry-run.")
            return 2
        print(
            f"Ollama model={cfg.model} num_ctx={cfg.num_ctx} "
            f"think={cfg.think_level if 'gpt-oss' in cfg.model else 'n/a'}"
        )

    runs: list[dict[str, Any]] = []
    work: list[tuple[str, float]] = [
        (v, t) for v in variant_list for t in temps
    ]
    total = len(work)
    idx = 0

    for variant in variant_list:
        sys_p, user_p = build_prompts(
            variant,
            problem_meta=prob,
            statement_excerpt=statement,
            failing_code=failing_code,
            verdict=verdict,
            issues=issues,
            valid_concept_ids=valid_concept_ids,
            reference_passing_code=passing_code,
            codebert_similarity=codebert_sim,
            diff_summary=diff.to_summary(),
            diff_unified_excerpt=diff.unified_excerpt or "",
            passing_file_hint=passing_path.name,
        )
        for temp in temps:
            idx += 1
            run_id = f"{variant}@T{temp}"
            if args.retry_failed and run_id in ok_run_ids:
                print(f"[{idx}/{total}] {run_id} SKIP (already ok)", flush=True)
                continue
            print(f"[{idx}/{total}] {run_id}", flush=True)

            record: dict[str, Any] = {
                "run_id": run_id,
                "variant": variant,
                "variant_label": PROMPT_VARIANTS[variant],
                "temperature": temp,
                "problem_id": args.problem_id,
                "anon_id": anon,
                "failing_file": failing_path.name,
                "passing_file": passing_path.name,
                "codebert_similarity": round(codebert_sim, 4),
                "is_bootstrap": is_bootstrap_variant(variant),
            }

            if args.dry_run:
                record["system_prompt_chars"] = len(sys_p)
                record["user_prompt_chars"] = len(user_p)
                record["total_prompt_chars"] = len(sys_p) + len(user_p)
                record["status"] = "dry_run"
                runs.append(record)
                continue

            assert client is not None
            t0 = time.time()
            try:
                result = client.chat_json(system=sys_p, user=user_p, temperature=temp)
                hints = cap_hints_to_rubric(result.get("hints") or [])
                schema_errs = schema_validate("hints", {"hints": hints})
                val_code = failing_code if is_bootstrap_variant(variant) else gold
                report = validator.validate(
                    hints, statement=statement, solution_code=val_code
                )
                record.update(
                    {
                        "status": "ok",
                        "latency_s": round(time.time() - t0, 1),
                        "hints": hints,
                        "concepts_targeted": result.get("concepts_targeted") or [],
                        "rationale_short": result.get("rationale_short", ""),
                        "validator_passed": report.passed,
                        "validator_violations": report.violations,
                        "validator_metrics": report.metrics,
                        "schema_errors": schema_errs,
                        "ollama_json_fallback": bool(
                            result.get("_ablation_fallback_no_json_format")
                        ),
                    }
                )
                result.pop("_ablation_fallback_no_json_format", None)
            except Exception as e:  # noqa: BLE001
                record.update(
                    {
                        "status": "error",
                        "latency_s": round(time.time() - t0, 1),
                        "_error": repr(e),
                    }
                )
            runs.append(record)
            with open(out_jsonl, "ab") as f:
                f.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    # Summary markdown table (merge with existing jsonl on retry)
    all_rows: dict[str, dict[str, Any]] = {}
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = orjson.loads(line)
                all_rows[row.get("run_id", "")] = row
    for r in runs:
        all_rows[r.get("run_id", "")] = r

    lines = [
        # "# Ablation — anon_1474 @ 2021_tema2_adrese",
        "# Ablation — anon_1195 @ 2022_tema2_curatare",
        "",
        f"Failing: `{failing_path.name}` · Passing: `{passing_path.name}` · "
        f"CodeBERT sim: {codebert_sim:.4f}",
        "",
        "| Variantă | T | Valid | #hints | sim→enunț | sim→cod | Latență (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id in sorted(all_rows.keys()):
        r = all_rows[run_id]
        m = r.get("validator_metrics") or {}
        lines.append(
            f"| {r['variant']} | {r['temperature']} | "
            f"{'✓' if r.get('validator_passed') else '✗' if r.get('status')=='ok' else '—'} | "
            f"{m.get('n_hints', '—')} | "
            f"{m.get('max_sim_to_statement', '—')} | "
            f"{m.get('max_sim_to_solution', '—')} | "
            f"{r.get('latency_s', '—')} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not args.dry_run and out_jsonl.exists():
        # Deduplicate jsonl by run_id (latest row wins).
        with open(out_jsonl, "wb") as f:
            for run_id in sorted(all_rows.keys()):
                f.write(orjson.dumps(all_rows[run_id], option=orjson.OPT_APPEND_NEWLINE))

    if args.dry_run:
        print(f"Dry-run: {len(runs)} prompt configs, no LLM calls.")
    else:
        if client:
            client.close()
        print(f"Wrote {out_jsonl} ({len(all_rows)} runs)")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

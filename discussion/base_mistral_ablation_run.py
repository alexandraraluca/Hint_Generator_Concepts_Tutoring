"""Base Mistral-7B (no LoRA) on the same cases as ``adapter_ablation_run.py``.

Isolates the fine-tuning variable: same backbone + prompt path as Stage 4 infer,
but without any adapter. Compare outputs to ``adapter_ablation_*.jsonl``.

Implicit: both cases (adrese, curatare), 3 temperatures → 6 runs total.

Usage (from repo root):
  $env:PA_INFER_DEVICE_MAP = "cpu"
  $env:PYTHONPATH = "$PWD"
  python discussion/base_mistral_ablation_run.py

  python discussion/base_mistral_ablation_run.py --dry-run
  python discussion/base_mistral_ablation_run.py --only-case adrese
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import orjson

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discussion.adapter_ablation_run import (
    DEFAULT_CASES,
    AblationCase,
    _anon_and_score,
    _issues_for,
    _verdict,
)
from discussion.base_mistral_generator import (
    DEFAULT_BASE_MODEL,
    BaseMistralHintGenerator,
)
from src.common.io_utils import read_json
from src.common.paths import ANNOTATIONS_DIR
from src.common.schemas import validate as schema_validate
from src.stage3_hints.validator import cap_hints_to_rubric

OUT_DIR = Path(__file__).resolve().parent / "output"
MODEL_ID = "mistral7b_base"

# Optional: read dtype/attn from an existing adapter manifest (same as LoRA runs).
DEFAULT_MANIFEST_HINT = (
    ROOT / "app" / "adapter_bootstrap_hints" / "mistral7b_instruct_pa_hints" / "manifest.json"
)


def _output_stem(anon: str, problem_id: str) -> str:
    pid_short = problem_id.split("_")[-1] if "_" in problem_id else problem_id
    return f"base_mistral_ablation_{anon}_{pid_short}"


def _unload_generator(gen: BaseMistralHintGenerator) -> None:
    gen._model = None
    gen._tokenizer = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _write_summary(
    *,
    out_jsonl: Path,
    summary_md: Path,
    case: AblationCase,
    anon: str,
    base_model: str,
    runs: list[dict[str, Any]],
    dedupe: bool,
) -> int:
    all_rows: dict[str, dict[str, Any]] = {}
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = orjson.loads(line)
                all_rows[row.get("run_id", "")] = row
    for r in runs:
        all_rows[r.get("run_id", "")] = r

    lines = [
        f"# Base Mistral ablation (no LoRA) — {anon} @ {case.problem_id}",
        "",
        f"Caz: `{case.case_id}` · Failing: `{case.failing_file.name}` · "
        f"Model: `{base_model}` · Device: `{os.environ.get('PA_INFER_DEVICE_MAP', 'auto')}`",
        "",
        "| Model | T | Valid | #hints | sim→enunț | sim→cod | Latență (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_id in sorted(all_rows.keys()):
        r = all_rows[run_id]
        m = r.get("validator_metrics") or {}
        valid_cell = (
            "✓"
            if r.get("validator_passed")
            else ("✗" if r.get("status") == "ok" else "—")
        )
        lines.append(
            f"| {r.get('model', '?')} | {r.get('temperature', '—')} | "
            f"{valid_cell} | "
            f"{m.get('n_hints', '—')} | "
            f"{m.get('max_sim_to_statement', '—')} | "
            f"{m.get('max_sim_to_solution', '—')} | "
            f"{r.get('latency_s', '—')} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if dedupe and out_jsonl.exists():
        with open(out_jsonl, "wb") as f:
            for run_id in sorted(all_rows.keys()):
                f.write(orjson.dumps(all_rows[run_id], option=orjson.OPT_APPEND_NEWLINE))

    return len(all_rows)


def _run_case(
    case: AblationCase,
    *,
    prob: dict[str, Any],
    base_model: str,
    manifest_path: Path | None,
    temps: list[float],
    dry_run: bool,
    retry_failed: bool,
    fresh: bool,
) -> int:
    failing_path = case.failing_file
    if not failing_path.exists():
        print(f"ERROR [{case.case_id}]: missing file {failing_path}")
        return 1

    failing_code = failing_path.read_text(encoding="utf-8", errors="replace")
    anon, score = _anon_and_score(failing_path)
    issues = _issues_for(prob["year"], prob["pid"], anon)
    verdict = _verdict(score, issues)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(anon, case.problem_id)
    out_jsonl = OUT_DIR / f"{stem}.jsonl"
    summary_md = OUT_DIR / f"{stem}.md"

    ok_run_ids: set[str] = set()
    if out_jsonl.exists() and not fresh:
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = orjson.loads(line)
                if row.get("status") == "ok":
                    ok_run_ids.add(row.get("run_id", ""))
    if fresh and out_jsonl.exists():
        out_jsonl.unlink()

    total = len(temps)
    runs: list[dict[str, Any]] = []

    print(
        f"\n=== Case '{case.case_id}': {case.problem_id} / {failing_path.name} "
        f"(base {base_model}, no adapter) ===",
        flush=True,
    )

    if dry_run:
        for temp in temps:
            run_id = f"{MODEL_ID}@T{temp}"
            print(f"[{case.case_id}] {run_id} DRY-RUN", flush=True)
            runs.append(
                {
                    "run_id": run_id,
                    "case_id": case.case_id,
                    "model": MODEL_ID,
                    "model_label": "Mistral-7B base (no LoRA)",
                    "base_model": base_model,
                    "fine_tuned": False,
                    "temperature": temp,
                    "problem_id": case.problem_id,
                    "anon_id": anon,
                    "failing_file": failing_path.name,
                    "status": "dry_run",
                }
            )
    else:
        print(f"Loading base model {base_model} ...", flush=True)
        gen = BaseMistralHintGenerator(
            base_model=base_model,
            manifest_path=manifest_path,
            temperature=temps[0],
        )
        try:
            for idx, temp in enumerate(temps, start=1):
                run_id = f"{MODEL_ID}@T{temp}"
                if retry_failed and run_id in ok_run_ids:
                    print(
                        f"[{case.case_id} {idx}/{total}] {run_id} SKIP (already ok)",
                        flush=True,
                    )
                    continue
                print(f"[{case.case_id} {idx}/{total}] {run_id}", flush=True)
                gen.temperature = temp
                record: dict[str, Any] = {
                    "run_id": run_id,
                    "case_id": case.case_id,
                    "model": MODEL_ID,
                    "model_label": "Mistral-7B base (no LoRA)",
                    "base_model": base_model,
                    "fine_tuned": False,
                    "manifest_hint": str(manifest_path) if manifest_path else None,
                    "temperature": temp,
                    "problem_id": case.problem_id,
                    "anon_id": anon,
                    "failing_file": failing_path.name,
                    "verdict": verdict,
                    "device_map": os.environ.get("PA_INFER_DEVICE_MAP", "auto"),
                }
                t0 = time.time()
                try:
                    result = gen.generate(
                        problem_id=case.problem_id,
                        failing_code=failing_code,
                        verdict=verdict,
                        issues=issues,
                        validate=True,
                    )
                    hints = cap_hints_to_rubric(result.get("hints") or [])
                    schema_errs = schema_validate("hints", {"hints": hints})
                    record.update(
                        {
                            "status": "ok",
                            "latency_s": round(time.time() - t0, 1),
                            "hints": hints,
                            "concepts_targeted": result.get("concepts_targeted") or [],
                            "validator_passed": result.get("validator_passed"),
                            "validator_violations": result.get("validator_violations", []),
                            "validator_metrics": result.get("validator_metrics", {}),
                            "schema_errors": schema_errs,
                        }
                    )
                    if result.get("raw_text") and not hints:
                        record["raw_text_preview"] = str(result["raw_text"])[:500]
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
        finally:
            _unload_generator(gen)
            print(f"Unloaded base model (case '{case.case_id}')", flush=True)

    n_rows = _write_summary(
        out_jsonl=out_jsonl,
        summary_md=summary_md,
        case=case,
        anon=anon,
        base_model=base_model,
        runs=runs,
        dedupe=not dry_run,
    )
    print(f"Wrote {out_jsonl} ({n_rows} runs)")
    print(f"Wrote {summary_md}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mistral-7B base ablation — same cases as adapter_ablation_run.py"
    )
    parser.add_argument(
        "--only-case",
        type=str,
        default=None,
        help="comma-separated case ids: adrese,curatare (default: both)",
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.2,0.6,0.9",
        help="comma-separated sampling temperatures",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"HuggingFace model id (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_HINT,
        help="optional adapter manifest.json for load dtype/attn (ignored if missing)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate paths only")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="skip runs already status=ok in output jsonl",
    )
    parser.add_argument("--fresh", action="store_true", help="truncate output jsonl per case")
    parser.add_argument(
        "--no-cpu",
        action="store_true",
        help="do not force CPU (default: PA_INFER_DEVICE_MAP=cpu)",
    )
    args = parser.parse_args()

    if not args.no_cpu:
        os.environ["PA_INFER_DEVICE_MAP"] = "cpu"
        print("PA_INFER_DEVICE_MAP=cpu", flush=True)

    case_ids = [c.case_id for c in DEFAULT_CASES]
    if args.only_case:
        wanted = {x.strip() for x in args.only_case.split(",") if x.strip()}
        cases = [c for c in DEFAULT_CASES if c.case_id in wanted]
        unknown = wanted - {c.case_id for c in cases}
        if unknown:
            print(f"ERROR: unknown cases: {sorted(unknown)}; choose from {case_ids}")
            return 1
        if not cases:
            print("ERROR: no cases selected")
            return 1
    else:
        cases = list(DEFAULT_CASES)

    manifest_path = args.manifest.resolve() if args.manifest else None
    if manifest_path is not None and not manifest_path.is_file():
        print(f"Note: manifest not found ({manifest_path}); using default bf16 load.", flush=True)
        manifest_path = None

    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    prob_by_id = {p["problem_id"]: p for p in problems}
    temps = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]

    exit_code = 0
    for case in cases:
        prob = prob_by_id.get(case.problem_id)
        if prob is None:
            print(f"ERROR [{case.case_id}]: unknown problem_id {case.problem_id}")
            exit_code = 1
            continue
        rc = _run_case(
            case,
            prob=prob,
            base_model=args.base_model,
            manifest_path=manifest_path,
            temps=temps,
            dry_run=args.dry_run,
            retry_failed=args.retry_failed,
            fresh=args.fresh,
        )
        if rc != 0:
            exit_code = rc

    if args.dry_run:
        print(f"\nDry-run OK: {len(cases)} case(s), {len(temps)} temperature(s) each.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

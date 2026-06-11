"""Compare Stage-4 LoRA adapters: bootstrap vs specific vs segments.

Uses ``HintGenerator`` from ``src.stage4_finetune.infer`` (same path as
``infer.py`` / Streamlit). Default device: CPU via ``PA_INFER_DEVICE_MAP=cpu``.

Implicit rulează pe **ambele** cazurile din ablația de prompturi:
  - anon_1474 @ 2021_tema2_adrese (cpp)
  - anon_1195 @ 2022_tema2_curatare (java)

3 adapters × 3 temperatures = 9 runs per case (18 total by default).

Usage (from repo root):
  $env:PA_INFER_DEVICE_MAP = "cpu"
  $env:PYTHONPATH = "$PWD"
  python discussion/adapter_ablation_run.py

  python discussion/adapter_ablation_run.py --only-case adrese
  python discussion/adapter_ablation_run.py --only bootstrap --dry-run
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import read_json, read_jsonl
from src.common.paths import ANNOTATIONS_DIR, CANONICAL_FILTERED_JSONL
from src.common.schemas import validate as schema_validate
from src.stage3_hints.validator import cap_hints_to_rubric
from src.stage4_finetune.infer import HintGenerator

OUT_DIR = Path(__file__).resolve().parent / "output"
SOLUTIONS = ROOT / "data" / "raw" / "solutions" / "solutions"

ADAPTERS: dict[str, str] = {
    "bootstrap": "app/adapter_bootstrap_hints/mistral7b_instruct_pa_hints",
    "specific": "app/adapter_specific_hints/mistral7b_instruct_pa_hints",
    "segments": "app/adapter_segments_hints/mistral7b_instruct_pa_hints",
}

ADAPTER_LABELS: dict[str, str] = {
    "bootstrap": "Bootstrap (finetune_train.jsonl)",
    "specific": "Specific (specific2_finetune_train.jsonl)",
    "segments": "Segments (segments_finetune_train.jsonl)",
}


@dataclass(frozen=True)
class AblationCase:
    case_id: str
    failing_file: Path
    problem_id: str


DEFAULT_CASES: tuple[AblationCase, ...] = (
    AblationCase(
        "adrese",
        SOLUTIONS / "2021_adrese" / "anon_1474_40.cpp",
        "2021_tema2_adrese",
    ),
    AblationCase(
        "curatare",
        SOLUTIONS / "2022_curatare" / "anon_1195_65.java",
        "2022_tema2_curatare",
    ),
)


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


def _issues_for(year: str, pid: str, anon: str) -> list[str]:
    for r in read_jsonl(CANONICAL_FILTERED_JSONL):
        if r.get("year") == year and r.get("pid") == pid and r.get("anon_id") == anon:
            return r.get("issues") or []
    return []


def _resolve_adapter_dir(rel: str) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _output_stem(anon: str, problem_id: str) -> str:
    pid_short = problem_id.split("_")[-1] if "_" in problem_id else problem_id
    return f"adapter_ablation_{anon}_{pid_short}"


def _anon_and_score(failing_path: Path) -> tuple[str, float]:
    anon = failing_path.stem.rsplit("_", 1)[0] if "anon_" in failing_path.stem else "anon"
    try:
        score = float(failing_path.stem.rsplit("_", 1)[-1])
    except ValueError:
        score = 0.0
    return anon, score


def _unload_generator(gen: HintGenerator) -> None:
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
        f"# Adapter ablation — {anon} @ {case.problem_id}",
        "",
        f"Caz: `{case.case_id}` · Failing: `{case.failing_file.name}` · "
        f"Device: `{os.environ.get('PA_INFER_DEVICE_MAP', 'auto')}`",
        "",
        "| Adapter | T | Valid | #hints | sim→enunț | sim→cod | Latență (s) |",
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
            f"| {r.get('adapter', '?')} | {r.get('temperature', '—')} | "
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
    adapter_ids: list[str],
    adapter_dirs: dict[str, Path],
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

    total = len(adapter_ids) * len(temps)
    idx = 0
    runs: list[dict[str, Any]] = []

    print(f"\n=== Case '{case.case_id}': {case.problem_id} / {failing_path.name} ===", flush=True)

    if dry_run:
        for aid in adapter_ids:
            for temp in temps:
                idx += 1
                run_id = f"{aid}@T{temp}"
                print(f"[{case.case_id} {idx}/{total}] {run_id} DRY-RUN", flush=True)
                runs.append(
                    {
                        "run_id": run_id,
                        "case_id": case.case_id,
                        "adapter": aid,
                        "adapter_label": ADAPTER_LABELS[aid],
                        "adapter_dir": str(adapter_dirs[aid]),
                        "temperature": temp,
                        "problem_id": case.problem_id,
                        "anon_id": anon,
                        "failing_file": failing_path.name,
                        "status": "dry_run",
                    }
                )
    else:
        for aid in adapter_ids:
            adapter_path = adapter_dirs[aid]
            print(f"Loading adapter '{aid}' for case '{case.case_id}' ...", flush=True)
            gen = HintGenerator(adapter_path, temperature=temps[0])
            try:
                for temp in temps:
                    idx += 1
                    run_id = f"{aid}@T{temp}"
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
                        "adapter": aid,
                        "adapter_label": ADAPTER_LABELS[aid],
                        "adapter_dir": str(adapter_path),
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
                                "validator_violations": result.get(
                                    "validator_violations", []
                                ),
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
                print(f"Unloaded adapter '{aid}' (case '{case.case_id}')", flush=True)

    n_rows = _write_summary(
        out_jsonl=out_jsonl,
        summary_md=summary_md,
        case=case,
        anon=anon,
        runs=runs,
        dedupe=not dry_run,
    )
    print(f"Wrote {out_jsonl} ({n_rows} runs)")
    print(f"Wrote {summary_md}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapter ablation (discussion/)")
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
        "--only",
        type=str,
        default=None,
        help="comma-separated adapter ids: bootstrap,specific,segments",
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

    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    prob_by_id = {p["problem_id"]: p for p in problems}

    adapter_ids = list(ADAPTERS.keys())
    if args.only:
        wanted = {a.strip() for a in args.only.split(",") if a.strip()}
        adapter_ids = [a for a in adapter_ids if a in wanted]
        unknown = wanted - set(adapter_ids)
        if unknown:
            print(f"ERROR: unknown adapters: {sorted(unknown)}")
            return 1

    adapter_dirs: dict[str, Path] = {}
    for aid in adapter_ids:
        p = _resolve_adapter_dir(ADAPTERS[aid])
        if not (p / "manifest.json").is_file():
            print(f"ERROR: missing manifest for {aid}: {p / 'manifest.json'}")
            return 1
        adapter_dirs[aid] = p

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
            adapter_ids=adapter_ids,
            adapter_dirs=adapter_dirs,
            temps=temps,
            dry_run=args.dry_run,
            retry_failed=args.retry_failed,
            fresh=args.fresh,
        )
        if rc != 0:
            exit_code = rc

    if args.dry_run:
        print(f"\nDry-run OK: {len(cases)} case(s), {len(adapter_ids)} adapters each.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

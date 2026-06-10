"""Stage 3.E - assemble the final fine-tuning dataset.

Inputs:
- `data/hints/silver_diff.jsonl`     (mechanical, lower quality)
- `data/hints/llm_bootstrap.jsonl`   (LLM-generated, higher quality)

Output:
- `data/hints/finetune_train.jsonl`
- `data/hints/finetune_val.jsonl`
- `data/hints/finetune_test.jsonl`
- `data/hints/finetune_stats.json`

Pipeline:
1. Filter both sources to those that passed `validator_passed=True`.
2. Mix according to a target ratio (default: 30% silver, 70% bootstrap).
3. Stratify the split by `problem_id` so train/val/test all see all
   problems but never the *same* (anon, problem) pair across splits.
4. Format every example as a chat-style instruction-following sample
   that can be consumed by `unsloth` / `transformers` for QLoRA.

The chat format used in `prompt`/`completion` mirrors the prompt structure
of the inference-time generator (see `prompt_builder.build_*_prompt`),
so the fine-tuned model can be dropped in directly.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.io_utils import read_json, read_jsonl, write_json, write_jsonl
from src.common.paths import (
    ANNOTATIONS_DIR,
    HINTS_DIR,
    PROCESSED_DIR,
    ensure_dirs,
)
from src.stage3_hints.prompt_builder import build_system_prompt, build_user_prompt

# SILVER_IN = HINTS_DIR / "silver_diff.jsonl"
# # BOOT_IN = HINTS_DIR / "llm_bootstrap.jsonl"

# TRAIN_OUT = HINTS_DIR / "specific2_finetune_train.jsonl"
# VAL_OUT = HINTS_DIR / "specific2_finetune_val.jsonl"
# TEST_OUT = HINTS_DIR / "specific2_finetune_test.jsonl"
# STATS_OUT = HINTS_DIR / "specific2_finetune_stats.json"

SILVER_IN = HINTS_DIR / "segments.jsonl"
TRAIN_OUT = HINTS_DIR / "segments_finetune_train.jsonl"
VAL_OUT = HINTS_DIR / "segments_finetune_val.jsonl"
TEST_OUT = HINTS_DIR / "segments_finetune_test.jsonl"
STATS_OUT = HINTS_DIR / "segments_finetune_stats.json"


def _filter_valid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r.get("validator_passed", False) and r.get("hints")
    ]


def _format_example(
    *,
    row: dict[str, Any],
    problem_meta: dict[str, Any],
    statement_excerpt: str,
    failing_code: str,
    valid_concept_ids: list[str],
) -> dict[str, Any]:
    """Produce a chat-format training sample.

    Format:
        {
          "system": str (rubric + JSON schema),
          "user":   str (problem + code + verdict + valid concepts),
          "assistant": str (compact JSON with "hints" + "concepts_targeted"),
          "meta":   {...} (problem_id, anon_id, source, language, verdict)
        }
    """
    system = build_system_prompt()
    user = build_user_prompt(
        problem_meta=problem_meta,
        statement_excerpt=statement_excerpt,
        failing_code=failing_code,
        verdict=row.get("verdict", "WA"),
        issues=row.get("issues") or [],
        valid_concept_ids=valid_concept_ids,
    )
    target_obj = {
        "hints": row["hints"],
        "concepts_targeted": row.get("concepts_targeted", []),
        "rationale_short": row.get("rationale_short", ""),
    }
    assistant = json.dumps(target_obj, ensure_ascii=False)
    return {
        "system": system,
        "user": user,
        "assistant": assistant,
        "meta": {
            "problem_id": row["problem_id"],
            "anon_id": row.get("anon_id"),
            "language": row.get("language"),
            "verdict": row.get("verdict"),
            "source": row["source"],
        },
    }


def _load_packets() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    pdir = PROCESSED_DIR / "packets"
    for p in pdir.glob("*.json"):
        try:
            out[p.stem] = read_json(p)
        except Exception:  # noqa: BLE001
            continue
    return out


def _read_failing_code_for(meta: dict[str, Any], packets: dict[str, dict[str, Any]]) -> str:
    """Best-effort: try to find the actual code on disk from anon_id."""
    from src.common.paths import EXTRACTED_SOLUTIONS_DIR

    pid_full = meta.get("problem_id", "")
    parts = pid_full.split("_", 2)
    if len(parts) < 3:
        return ""
    year, _tema, pid = parts[0], parts[1], parts[2]
    base = EXTRACTED_SOLUTIONS_DIR / "solutions" / f"{year}_{pid}"
    if not base.exists():
        return ""
    anon = meta.get("anon_id") or ""
    cands = sorted(base.glob(f"{anon}_*.cpp")) + sorted(base.glob(f"{anon}_*.java"))
    if not cands:
        return ""
    try:
        return cands[0].read_text(encoding="utf-8", errors="replace")[:5000]
    except OSError:
        return ""


def _stratified_split(
    items: list[dict[str, Any]],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list, list, list]:
    """Split by (problem_id, anon_id) keys: same student NEVER in 2 splits."""
    rng = random.Random(seed)
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_problem[it["meta"]["problem_id"]].append(it)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for pid, lst in by_problem.items():
        # group by anon to ensure same anon goes to one split
        by_anon: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for it in lst:
            by_anon[it["meta"]["anon_id"] or "_unknown"].append(it)
        anon_keys = list(by_anon.keys())
        rng.shuffle(anon_keys)
        n_train = max(1, int(len(anon_keys) * train_frac))
        n_val = max(0, int(len(anon_keys) * val_frac))
        train_keys = anon_keys[:n_train]
        val_keys = anon_keys[n_train : n_train + n_val]
        test_keys = anon_keys[n_train + n_val :]
        for k in train_keys:
            train.extend(by_anon[k])
        for k in val_keys:
            val.extend(by_anon[k])
        for k in test_keys:
            test.extend(by_anon[k])
    return train, val, test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--silver-ratio", type=float, default=0.30,
        help="target fraction of silver hints in the final mix (only used when --use-all is OFF)",
    )
    parser.add_argument(
        "--use-all", action="store_true", default=True,
        help="keep every valid hint regardless of ratio (default: True). "
             "Pass --no-use-all to enforce --silver-ratio.",
    )
    parser.add_argument(
        "--no-use-all", dest="use_all", action="store_false",
        help="opposite of --use-all: trim buckets to match --silver-ratio",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.80,
        help="fraction of unique anons routed to train (per problem)",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.10,
        help="fraction routed to val (test gets the rest)",
    )
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    ensure_dirs()

    silver = list(read_jsonl(SILVER_IN)) if SILVER_IN.exists() else []
    # boot = list(read_jsonl(BOOT_IN)) if BOOT_IN.exists() else []
    silver_valid = _filter_valid(silver)
    # boot_valid = _filter_valid(boot)
    print(f"silver: {len(silver)} total, {len(silver_valid)} valid")
    # print(f"bootstrap: {len(boot)} total, {len(boot_valid)} valid")

    # if not silver_valid and not boot_valid:
    #     print("no valid hints found; nothing to assemble")
    #     return 1

    # Mixing policy:
    # - With `--use-all` we keep every valid hint; the resulting silver/bootstrap
    #   ratio is whatever the data dictates. Recommended for the final dataset.
    # - Without `--use-all` we trim the over-represented bucket so the final
    #   ratio matches `--silver-ratio` (useful for ablations).
    # target_silver = args.silver_ratio
    # if args.use_all or not silver_valid or not boot_valid:
    #     n_silver = len(silver_valid)
    #     # n_boot = len(boot_valid)
    # else:
    #     target_total = min(
    #         len(silver_valid) / max(target_silver, 1e-6),
    #         # len(boot_valid) / max(1 - target_silver, 1e-6),
    #     )
    #     n_silver = int(target_total * target_silver)
    #     n_boot = int(target_total * (1 - target_silver))

    # rng = random.Random(args.seed)
    # rng.shuffle(silver_valid)
    # rng.shuffle(boot_valid)
    # mix = silver_valid[:n_silver] + boot_valid[:n_boot]
    mix = silver_valid
    # rng.shuffle(mix)
    # actual_silver_ratio = n_silver / max(1, len(mix))
    # print(
    #     f"mix: {len(mix)} (silver={n_silver}, bootstrap={n_boot}, "
    #     f"silver_ratio={actual_silver_ratio:.2%})"
    # )

    # build packets cache once
    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    pid_to_meta = {p["problem_id"]: p for p in problems}
    packets = _load_packets()
    dag = read_json(ANNOTATIONS_DIR / "concepts_dag.json")
    valid_concept_ids = [c["id"] for c in dag["concepts"]]

    formatted: list[dict[str, Any]] = []
    skipped = 0
    for r in mix:
        prob = pid_to_meta.get(r["problem_id"])
        if prob is None:
            skipped += 1
            continue
        statement = packets.get(r["problem_id"], {}).get("statement_text", "") or ""
        failing_code = _read_failing_code_for(r, packets)
        if not failing_code:
            failing_code = "(cod indisponibil)"
        formatted.append(
            _format_example(
                row=r,
                problem_meta=prob,
                statement_excerpt=statement,
                failing_code=failing_code,
                valid_concept_ids=valid_concept_ids,
            )
        )
    print(f"formatted: {len(formatted)} ({skipped} skipped)")

    train, val, test = _stratified_split(
        formatted,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VAL_OUT, val)
    write_jsonl(TEST_OUT, test)

    stats = {
        "n_silver_total": len(silver),
        "n_silver_valid": len(silver_valid),
        # "n_bootstrap_total": len(boot),
        # "n_bootstrap_valid": len(boot_valid),
        "target_silver_ratio": args.silver_ratio,
        "n_in_mix": len(mix),
        "n_formatted": len(formatted),
        "split": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "split_per_problem": {
            "train": dict(Counter(it["meta"]["problem_id"] for it in train)),
            "val": dict(Counter(it["meta"]["problem_id"] for it in val)),
            "test": dict(Counter(it["meta"]["problem_id"] for it in test)),
        },
    }
    write_json(STATS_OUT, stats)
    print(f"wrote {len(train)} train, {len(val)} val, {len(test)} test")
    print(f"  -> {TRAIN_OUT}")
    print(f"  -> {VAL_OUT}")
    print(f"  -> {TEST_OUT}")
    print(f"  -> {STATS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

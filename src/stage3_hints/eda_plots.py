"""Stage 3 - exploratory data analysis figures for the hint dataset.

Produces a set of PNGs in `data/figures/stage3/` summarising:
  fig11 - validator outcome (valid vs invalid + reasons)
  fig12 - hints per problem (coverage + balance)
  fig13 - hint length distribution (words & sentences) per level
  fig14 - similarity histograms (hint vs statement, hint vs solution)
  fig15 - language split & average score of failing solutions
  fig16 - concepts targeted (top-N) and DAG coverage
  fig17 - dataset split sizes (train/val/test) per problem
  fig18 - example fan-out: most frequent (verdict, language) combinations

These mirror the Stage 1 EDA conventions (matplotlib only, single-figure
PNGs, sane defaults so they're paste-ready in the thesis). Each figure
also writes its underlying table to `data/figures/stage3/<name>.csv` for
reproducibility.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.common.io_utils import read_json, read_jsonl
from src.common.paths import ANNOTATIONS_DIR, FIGURES_DIR, HINTS_DIR

OUT = FIGURES_DIR / "stage3"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    p = OUT / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")


def _write_csv(rows: list[tuple], header: tuple[str, ...], name: str) -> None:
    p = OUT / f"{name}.csv"
    with open(p, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(c) for c in r) + "\n")
    print(f"  -> {p}")


def _load_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_p = HINTS_DIR / "llm_bootstrap.jsonl"
    invalid_p = HINTS_DIR / "llm_bootstrap_invalid.jsonl"
    silver_p = HINTS_DIR / "silver_diff.jsonl"
    valid = list(read_jsonl(valid_p)) if valid_p.exists() else []
    invalid = list(read_jsonl(invalid_p)) if invalid_p.exists() else []
    silver = list(read_jsonl(silver_p)) if silver_p.exists() else []
    return valid, invalid, silver


# -------- figures --------


def fig11_validator_outcome(valid: list, invalid: list) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))

    total = len(valid) + len(invalid)
    rate = len(valid) / max(1, total)
    axs[0].bar(
        ["valid", "invalid"],
        [len(valid), len(invalid)],
        color=["#4caf50", "#e57373"],
    )
    axs[0].set_title(
        f"LLM bootstrap outcome (n={total}, {rate:.1%} valid)"
    )
    axs[0].set_ylabel("count")
    for x, v in enumerate([len(valid), len(invalid)]):
        axs[0].text(x, v + max(1, total * 0.01), str(v), ha="center", va="bottom")

    reasons = Counter()
    for r in invalid:
        if r.get("_error"):
            tag = r["_error"].split(":", 1)[0].strip()
            reasons[f"err:{tag}"] += 1
        elif r.get("_schema_errors"):
            reasons["schema"] += 1
        for v in r.get("validator_violations") or []:
            tag = v.split(":", 1)[0]
            reasons[f"rubric:{tag}"] += 1
    if reasons:
        labels, counts = zip(*reasons.most_common(10))
        axs[1].barh(list(labels)[::-1], list(counts)[::-1], color="#e57373")
        axs[1].set_title("Top reasons for invalid hint sets")
        axs[1].set_xlabel("count")
    else:
        axs[1].set_title("no invalid cases")
    _save(fig, "fig11_validator_outcome")
    _write_csv(
        sorted(reasons.items(), key=lambda kv: -kv[1]),
        ("reason", "count"),
        "fig11_validator_outcome",
    )


def fig12_hints_per_problem(valid: list) -> None:
    c = Counter(r["problem_id"] for r in valid)
    pids = sorted(c.keys())
    counts = [c[p] for p in pids]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(pids, counts, color="#5b8def")
    ax.set_xlabel("number of valid hint sets")
    ax.set_title(f"Hint coverage per problem (total={len(valid)}, {len(c)} problems)")
    ax.axvline(np.median(counts), color="#1f3e89", linestyle="--", label=f"median={np.median(counts):.0f}")
    ax.legend(loc="lower right")
    _save(fig, "fig12_hints_per_problem")
    _write_csv([(p, c[p]) for p in pids], ("problem_id", "count"), "fig12_hints_per_problem")


def _hint_lengths(valid: list) -> dict[str, list[tuple[int, int]]]:
    """Returns level -> [(words, sentences), ...]."""
    out: dict[str, list[tuple[int, int]]] = {}
    for r in valid:
        for h in r.get("hints", []):
            text = h.get("text", "")
            level = h.get("level", "unknown")
            words = len([t for t in text.split() if t])
            sents = max(1, sum(text.count(c) for c in ".!?"))
            out.setdefault(level, []).append((words, sents))
    return out


def fig13_hint_length(valid: list) -> None:
    by_level = _hint_lengths(valid)
    levels_order = ["macro", "structural", "specific", "very_specific"]
    levels = [l for l in levels_order if l in by_level] + [
        l for l in by_level if l not in levels_order
    ]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("viridis", max(2, len(levels)))
    for i, lvl in enumerate(levels):
        words = [w for w, _ in by_level[lvl]]
        sents = [s for _, s in by_level[lvl]]
        axs[0].hist(words, bins=20, alpha=0.6, label=lvl, color=cmap(i))
        axs[1].hist(sents, bins=range(1, 7), alpha=0.6, label=lvl, color=cmap(i))
    axs[0].set_xlabel("words per hint")
    axs[0].set_title("Distribution of hint length (words)")
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("sentences per hint")
    axs[1].set_title("Distribution of hint length (sentences)")
    axs[1].legend(fontsize=8)
    _save(fig, "fig13_hint_length")

    rows = []
    for lvl in levels:
        words = [w for w, _ in by_level[lvl]]
        sents = [s for _, s in by_level[lvl]]
        rows.append(
            (lvl, len(words), float(np.mean(words)), float(np.median(words)),
             float(np.mean(sents)), float(np.median(sents)))
        )
    _write_csv(
        rows,
        ("level", "n", "avg_words", "med_words", "avg_sents", "med_sents"),
        "fig13_hint_length",
    )


def fig14_similarity(valid: list) -> None:
    sim_stmt: list[float] = []
    sim_sol: list[float] = []
    for r in valid:
        m = r.get("validator_metrics", {})
        if "max_sim_to_statement" in m:
            sim_stmt.append(float(m["max_sim_to_statement"]))
        if "max_sim_to_solution" in m:
            sim_sol.append(float(m["max_sim_to_solution"]))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].hist(sim_stmt, bins=30, color="#5b8def", edgecolor="white")
    axs[0].axvline(0.55, color="#c62828", linestyle="--", label="rubric threshold (0.55)")
    axs[0].set_xlabel("max cosine sim (hint, statement)")
    axs[0].set_title(f"Anti-reformulation check\n(median={np.median(sim_stmt):.3f})")
    axs[0].legend()

    axs[1].hist(sim_sol, bins=30, color="#a5d6a7", edgecolor="white")
    axs[1].axvline(0.55, color="#c62828", linestyle="--", label="rubric threshold (0.55)")
    axs[1].set_xlabel("max cosine sim (hint, solution)")
    axs[1].set_title(f"Anti-leakage check\n(median={np.median(sim_sol):.3f})")
    axs[1].legend()
    _save(fig, "fig14_similarity")
    _write_csv(
        [
            ("sim_statement_median", float(np.median(sim_stmt))),
            ("sim_statement_p95", float(np.percentile(sim_stmt, 95))),
            ("sim_solution_median", float(np.median(sim_sol))),
            ("sim_solution_p95", float(np.percentile(sim_sol, 95))),
        ],
        ("metric", "value"),
        "fig14_similarity",
    )


def fig15_language_split(valid: list) -> None:
    by_lang = Counter(r.get("language", "?") for r in valid)
    by_verdict = Counter(r.get("verdict", "?") for r in valid)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    if by_lang:
        axs[0].pie(
            by_lang.values(),
            labels=[f"{k} ({v})" for k, v in by_lang.items()],
            autopct="%1.1f%%",
            colors=["#5b8def", "#ff8a65", "#a5d6a7"],
            startangle=90,
        )
        axs[0].set_title("Language split of failing solutions used")

    if by_verdict:
        labels, counts = zip(*by_verdict.most_common())
        axs[1].bar(labels, counts, color="#9575cd")
        axs[1].set_title("Verdict distribution (failing only)")
        axs[1].set_ylabel("count")
    _save(fig, "fig15_language_split")


def fig16_concepts(valid: list) -> None:
    dag = read_json(ANNOTATIONS_DIR / "concepts_dag.json")
    all_concepts = [c["id"] for c in dag["concepts"]]

    used = Counter()
    for r in valid:
        for c in r.get("concepts_targeted", []):
            used[c] += 1

    in_dag = sum(1 for c in used if c in all_concepts)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    if used:
        labels, counts = zip(*used.most_common(20))
        axs[0].barh(list(labels)[::-1], list(counts)[::-1], color="#5b8def")
        axs[0].set_title(f"Top concepts targeted by hints (n_unique={len(used)})")
        axs[0].set_xlabel("count")

    coverage = [used.get(c, 0) for c in all_concepts]
    axs[1].bar(all_concepts, coverage, color="#a5d6a7")
    axs[1].tick_params(axis="x", rotation=70, labelsize=7)
    axs[1].set_title(f"DAG coverage ({in_dag}/{len(all_concepts)} used)")
    axs[1].set_ylabel("times targeted")
    _save(fig, "fig16_concepts")


def fig17_split_per_problem() -> None:
    p = HINTS_DIR / "finetune_stats.json"
    if not p.exists():
        print("  skip fig17: finetune_stats.json missing")
        return
    stats = read_json(p)
    splits = stats.get("split_per_problem", {})
    if not splits:
        return
    pids = sorted({k for v in splits.values() for k in v.keys()})
    train = [splits.get("train", {}).get(p, 0) for p in pids]
    val = [splits.get("val", {}).get(p, 0) for p in pids]
    test = [splits.get("test", {}).get(p, 0) for p in pids]

    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(pids))
    ax.barh(y, train, color="#5b8def", label="train")
    ax.barh(y, val, left=train, color="#ffb74d", label="val")
    ax.barh(y, test, left=np.array(train) + np.array(val), color="#a5d6a7", label="test")
    ax.set_yticks(y, pids)
    ax.set_xlabel("examples")
    ax.set_title(
        f"Stratified split per problem (train={stats['split']['train']}, "
        f"val={stats['split']['val']}, test={stats['split']['test']})"
    )
    ax.legend()
    _save(fig, "fig17_split_per_problem")


def fig18_verdict_language(valid: list) -> None:
    keys = Counter((r.get("verdict", "?"), r.get("language", "?")) for r in valid)
    if not keys:
        return
    verdicts = sorted({k[0] for k in keys})
    langs = sorted({k[1] for k in keys})
    M = np.zeros((len(verdicts), len(langs)), dtype=int)
    for (v, l), c in keys.items():
        M[verdicts.index(v), langs.index(l)] = c

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(langs)), langs)
    ax.set_yticks(range(len(verdicts)), verdicts)
    for i in range(len(verdicts)):
        for j in range(len(langs)):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() * 0.55 else "black")
    ax.set_title("(verdict, language) matrix")
    fig.colorbar(im)
    _save(fig, "fig18_verdict_language")


def main() -> int:
    valid, invalid, silver = _load_data()
    print(f"loaded valid={len(valid)} invalid={len(invalid)} silver={len(silver)}")
    fig11_validator_outcome(valid, invalid)
    fig12_hints_per_problem(valid)
    fig13_hint_length(valid)
    fig14_similarity(valid)
    fig15_language_split(valid)
    fig16_concepts(valid)
    fig17_split_per_problem()
    fig18_verdict_language(valid)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 1.4 - exploratory data analysis figures for the report.

Produces the figures we agreed on for the data chapter:

- 01_submissions_per_year_tema.png       - bar chart submisii pe (an, tema)
- 02_solutions_per_problem.png           - bar chart soluții/problema (cu prag)
- 03_score_distribution_per_year.png     - histogramă scoruri per an
- 04_loc_distribution_per_language.png   - LOC C++ vs Java
- 05_verdict_distribution.png            - distribuție OK/WA/CE/RE/TLE/MLE
- 06_submissions_per_user_pa.png         - histogramă nr submisii / user / problema
- 07_user_assignment_coverage.png        - userii pe nr de assignment-uri PA
- 08_pts_progression_examples.png        - ~12 traiectorii reprezentative
- 09_avg_score_per_problem.png           - dificultate per problema (mean pts)
- 10_issue_categories.png                - top issue messages din checker

Toate figurile au titluri și axe în română (pentru raportul de licență).
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from src.common.io_utils import read_json, read_jsonl
from src.common.paths import (
    CANONICAL_FILTERED_JSONL,
    EXTRACTED_SOLUTIONS_DIR,
    FIGURES_DIR,
    PROBLEMS_INDEX_JSON,
    PROCESSED_DIR,
    USER_TRAJECTORIES_JSONL,
    ensure_dirs,
)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    out = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(FIGURES_DIR.parent)}")


def fig_submissions_per_year_tema(rows: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    seen_subs: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["anon_id"], r["submission_name"])
        if key in seen_subs:
            continue
        seen_subs.add(key)
        counts[(r["year"], r["tema"])] += 1
    years = sorted({y for y, _ in counts})
    temas = ["tema1", "tema2"]
    x = np.arange(len(years))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, t in enumerate(temas):
        vals = [counts.get((y, t), 0) for y in years]
        ax.bar(x + (i - 0.5) * w, vals, width=w, label=t)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + max(vals) * 0.01, str(v), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("an")
    ax.set_ylabel("nr. submisii unice")
    ax.set_title("Submisii unice pe (an, temă) - PA 2021-2024")
    ax.legend()
    _save(fig, "01_submissions_per_year_tema.png")


def fig_solutions_per_problem(problems_index: list[dict[str, Any]], threshold: int) -> None:
    sorted_idx = sorted(problems_index, key=lambda p: p["n_solutions"])
    labels = [p["problem_id"].replace("_tema", "_t") for p in sorted_idx]
    vals = [p["n_solutions"] for p in sorted_idx]
    colors = ["#d62728" if v < threshold else "#2ca02c" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(labels, vals, color=colors)
    ax.axvline(threshold, color="orange", ls="--", label=f"prag = {threshold}")
    ax.set_xlabel("nr. soluții pe disc (în solutions/)")
    ax.set_title("Soluții per problemă (PA 2021-2024) - cu prag de filtrare")
    ax.legend(loc="lower right")
    _save(fig, "02_solutions_per_problem.png")


def fig_score_distribution_per_year(rows: list[dict[str, Any]]) -> None:
    by_year = defaultdict(list)
    for r in rows:
        by_year[r["year"]].append(r["pts"])
    years = sorted(by_year)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for y in years:
        ax.hist(
            by_year[y],
            bins=np.linspace(0, 100, 21),
            alpha=0.5,
            label=f"{y} (n={len(by_year[y])})",
        )
    ax.set_xlabel("scor checker (pts)")
    ax.set_ylabel("nr. evaluări")
    ax.set_title("Distribuția scorurilor checker per an")
    ax.legend()
    _save(fig, "03_score_distribution_per_year.png")


def _count_loc(text: str) -> int:
    n = 0
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        n += 1
    return n


def fig_loc_distribution(problems_index: list[dict[str, Any]]) -> None:
    locs = {"cpp": [], "java": []}
    for p in tqdm(problems_index, desc="counting LOC"):
        base = (
            EXTRACTED_SOLUTIONS_DIR
            / "solutions"
            / f"{p['year']}_{p['pid']}"
        )
        if not base.exists():
            continue
        for f in base.iterdir():
            if not f.is_file():
                continue
            ext = f.suffix.lower().lstrip(".")
            if ext not in locs:
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            locs[ext].append(_count_loc(txt))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 600, 41)
    for k, vals in locs.items():
        if not vals:
            continue
        ax.hist(vals, bins=bins, alpha=0.55, label=f"{k} (n={len(vals)})")
    ax.set_xlim(0, 600)
    ax.set_xlabel("linii de cod (excl. linii goale și comentarii)")
    ax.set_ylabel("nr. fișiere")
    ax.set_title("Distribuție LOC per limbaj (toate problemele PA 2021-2024)")
    ax.legend()
    _save(fig, "04_loc_distribution_per_language.png")


def _verdict_for(pts: float, issues: list[str]) -> str:
    s = " ".join(issues or []).lower()
    if pts == 100:
        return "OK"
    if any(k in s for k in ("compile", "compilation", "ce")) and "wa" not in s:
        return "CE"
    if any(k in s for k in ("runtime", "segmentation", "re ")):
        return "RE"
    if any(k in s for k in ("tle", "time limit", "timeout")):
        return "TLE"
    if any(k in s for k in ("mle", "memory limit")):
        return "MLE"
    if "wa" in s or pts < 100:
        return "WA"
    return "OTHER"


def fig_verdict_distribution(rows: list[dict[str, Any]]) -> None:
    cnt = Counter(_verdict_for(r["pts"], r.get("issues", [])) for r in rows)
    order = ["OK", "WA", "TLE", "RE", "CE", "MLE", "OTHER"]
    vals = [cnt.get(k, 0) for k in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(order, vals, color=sns.color_palette("Set2", len(order)))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, str(v), ha="center", fontsize=9)
    ax.set_ylabel("nr. evaluări per (submisie, problemă)")
    ax.set_title("Distribuție verdict checker (L1)")
    _save(fig, "05_verdict_distribution.png")


def fig_submissions_per_user_pa(trajectories: list[dict[str, Any]]) -> None:
    n_subs = [t["n_submissions"] for t in trajectories]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n_subs, bins=np.arange(1, max(n_subs) + 2) - 0.5, color="#1f77b4")
    ax.set_xlabel("nr. submisii pe (user, problemă)")
    ax.set_ylabel("nr. perechi (user, problemă)")
    ax.set_title("Cât de des reîncearcă studenții aceeași problemă?")
    _save(fig, "06_submissions_per_user_pa.png")


def fig_user_assignment_coverage(rows: list[dict[str, Any]]) -> None:
    user_to_assigns: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        user_to_assigns[r["anon_id"]].add(f"{r['year']}_{r['tema']}")
    cnt = Counter(len(v) for v in user_to_assigns.values())
    keys = sorted(cnt)
    vals = [cnt[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(k) for k in keys], vals, color="#9467bd")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, str(v), ha="center", fontsize=9)
    ax.set_xlabel("nr. assignment-uri (an_tema) la care participă userul")
    ax.set_ylabel("nr. useri")
    ax.set_title("Acoperire user x assignment - PA 2021-2024")
    _save(fig, "07_user_assignment_coverage.png")


def _has_progress(t: dict[str, Any]) -> bool:
    p = t.get("pts_progression") or []
    return len(p) >= 2 and len(set(p)) >= 2


def fig_pts_progression_examples(trajectories: list[dict[str, Any]], k: int = 12) -> None:
    rng = random.Random(7)
    progress = [
        t
        for t in trajectories
        if t["n_submissions"] >= 2 and _has_progress(t)
    ]
    if len(progress) >= 4:
        candidates = progress
    else:
        candidates = [t for t in trajectories if t["n_submissions"] >= 2]
    if not candidates:
        print("  (no trajectories with >=2 submissions, skipping)")
        return
    candidates.sort(key=lambda t: t["n_submissions"], reverse=True)
    pool = candidates[: max(50, k * 4)]
    sample = rng.sample(pool, min(k, len(pool)))
    n = len(sample)
    cols = min(4, n)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows), sharey=True, squeeze=False)
    flat = axes.flatten()
    for ax, t in zip(flat, sample):
        ax.plot(t["pts_progression"], "-o", color="#2ca02c")
        ax.set_ylim(-5, 105)
        ax.set_title(
            f"{t['problem_id']}\n{t['anon_id']} (n={t['n_submissions']})", fontsize=8
        )
    for ax in flat[len(sample):]:
        ax.set_visible(False)
    fig.suptitle(
        "Trayectorii (pts) ale userilor pe aceeași problemă - exemple",
        fontsize=12,
        y=1.0,
    )
    _save(fig, "08_pts_progression_examples.png")


def fig_avg_score_per_problem(rows: list[dict[str, Any]]) -> None:
    by_prob: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_prob[r["problem_id"]].append(r["pts"])
    pids = sorted(by_prob, key=lambda p: np.mean(by_prob[p]))
    means = [np.mean(by_prob[p]) for p in pids]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh([p.replace("_tema", "_t") for p in pids], means, color="#ff7f0e")
    ax.set_xlim(0, 100)
    ax.set_xlabel("scor mediu (pts) per (user, problemă)")
    ax.set_title("Dificultate aparentă a problemelor (scor mediu pe checker)")
    _save(fig, "09_avg_score_per_problem.png")


_NUM_RE = re.compile(r"-?\d+")


def _normalize_issue(s: str) -> str:
    """Replace numbers so that 'WA 38 vs 29' and 'WA 11 vs 27' collapse together."""
    return _NUM_RE.sub("<N>", s).strip().lower()


def fig_issue_categories(rows: list[dict[str, Any]]) -> None:
    cnt: Counter[str] = Counter()
    for r in rows:
        for it in r.get("issues", []) or []:
            cnt[_normalize_issue(str(it))] += 1
    top = cnt.most_common(15)
    if not top:
        print("  (no issues found, skipping issue plot)")
        return
    labels, vals = zip(*top)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], vals[::-1], color="#17becf")
    ax.set_xlabel("nr. apariții (numere normalizate)")
    ax.set_title("Top mesaje de eroare din checker (după normalizare)")
    _save(fig, "10_issue_categories.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-solutions",
        type=int,
        default=30,
        help="Threshold to mark in the per-problem plot (default %(default)s).",
    )
    parser.add_argument(
        "--skip-loc",
        action="store_true",
        help="Skip the LOC plot (slow due to disk reads).",
    )
    args = parser.parse_args()

    ensure_dirs()
    rows = list(read_jsonl(CANONICAL_FILTERED_JSONL))
    problems_index = read_json(PROBLEMS_INDEX_JSON)
    trajectories = list(read_jsonl(USER_TRAJECTORIES_JSONL))

    print(f"loaded {len(rows)} canonical rows, {len(trajectories)} trajectories")
    print("rendering figures ...")

    fig_submissions_per_year_tema(rows)
    fig_solutions_per_problem(problems_index, args.min_solutions)
    fig_score_distribution_per_year(rows)
    if not args.skip_loc:
        fig_loc_distribution(problems_index)
    fig_verdict_distribution(rows)
    fig_submissions_per_user_pa(trajectories)
    fig_user_assignment_coverage(rows)
    fig_pts_progression_examples(trajectories)
    fig_avg_score_per_problem(rows)
    fig_issue_categories(rows)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

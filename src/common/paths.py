"""Centralized path configuration for the project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
HINTS_DIR = DATA_DIR / "hints"
FIGURES_DIR = DATA_DIR / "figures"

RAW_SOLUTIONS_ZIP = ROOT / "solutions.zip"
RAW_STATEMENTS_ZIP = ROOT / "statements.zip"
RAW_TESTS_ZIP = ROOT / "tests.zip"
RAW_FEEDBACK_JSONL = ROOT / "submission_feedback.jsonl"

EXTRACTED_SOLUTIONS_DIR = RAW_DIR / "solutions"
EXTRACTED_STATEMENTS_DIR = RAW_DIR / "statements"
EXTRACTED_TESTS_DIR = RAW_DIR / "tests"

CANONICAL_JSONL = PROCESSED_DIR / "canonical.jsonl"
CANONICAL_FILTERED_JSONL = PROCESSED_DIR / "canonical_filtered.jsonl"
PROBLEMS_INDEX_JSON = PROCESSED_DIR / "problems_index.json"
USER_TRAJECTORIES_JSONL = PROCESSED_DIR / "user_trajectories.jsonl"

PA_YEARS = ("2021", "2022", "2023", "2024")
PA_TEMAS = ("tema1", "tema2")

MIN_SOLUTIONS_PER_PROBLEM = 30


def ensure_dirs() -> None:
    for d in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        ANNOTATIONS_DIR,
        HINTS_DIR,
        FIGURES_DIR,
        EXTRACTED_SOLUTIONS_DIR,
        EXTRACTED_STATEMENTS_DIR,
        EXTRACTED_TESTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)

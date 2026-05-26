"""Stage 2.3 - test labelling.

For every (problem, test) pair we record:
- size_class (rule-based on input bytes / first-line N if extractable),
- input_bytes,
- edge_case (heuristic on filename: 'edge', 'corner', 'big', 'small'),
- discriminates: best-effort default = "correctness".

The LLM step (concepts tested by each test) is *optional* and gated by the
`--with-llm` flag, because (a) it is expensive (one call per test), and (b)
several tests per problem usually probe the same concept; we run LLM
concept-tagging only on tests that look "different" (size_class change).
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.common.io_utils import read_json, write_json
from src.common.paths import (
    ANNOTATIONS_DIR,
    EXTRACTED_TESTS_DIR,
    PA_TEMAS,
    PA_YEARS,
    ensure_dirs,
)
from src.common.schemas import validate

TESTS_LABELS_OUT = ANNOTATIONS_DIR / "tests_labels.json"


def _size_class(n_bytes: int) -> str:
    if n_bytes <= 200:
        return "tiny"
    if n_bytes <= 5_000:
        return "small"
    if n_bytes <= 200_000:
        return "medium"
    if n_bytes <= 5_000_000:
        return "large"
    return "stress"


def _extract_first_n(blob: bytes) -> int | None:
    """Many PA tests put N as the first integer of the input."""
    head = blob[:128].decode("utf-8", errors="replace").strip()
    m = re.match(r"\s*(-?\d+)", head)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _is_edge_case_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("edge", "corner", "stress", "small", "big"))


def _enumerate_tests_in_inner_zip(
    inner_zip_path: Path,
) -> list[tuple[str, str, int, bytes]]:
    """Yield (problem_pid, test_id, size, payload) for each test input file.

    We only include `*.in` files (input). The matching `*.ref` is implied.
    """
    out: list[tuple[str, str, int, bytes]] = []
    if not inner_zip_path.exists():
        return out
    with zipfile.ZipFile(inner_zip_path) as iz:
        for j in iz.infolist():
            name = j.filename.replace("\\", "/")
            parts = name.split("/")
            if len(parts) < 3 or parts[0] != "private_tests":
                continue
            if not name.endswith(".in"):
                continue
            pid = parts[1]
            test_id = parts[-1][:-3]
            try:
                payload = iz.read(j)
            except Exception:  # noqa: BLE001
                continue
            out.append((pid, test_id, j.file_size, payload))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="(stub) call Ollama for concept tagging on a sample of tests",
    )
    args = parser.parse_args()

    ensure_dirs()

    rows: list[dict[str, Any]] = []
    for year in PA_YEARS:
        for tema in PA_TEMAS:
            inner_dir = EXTRACTED_TESTS_DIR / f"pa_{year}_{tema}"
            inner_zip = inner_dir / f"pa_{year}_{tema}.zip"
            for pid, test_id, n_bytes, payload in tqdm(
                _enumerate_tests_in_inner_zip(inner_zip),
                desc=f"{year} {tema}",
            ):
                rows.append(
                    {
                        "problem_id": f"{year}_{tema}_{pid}",
                        "test_id": test_id,
                        "size_class": _size_class(n_bytes),
                        "input_bytes": int(n_bytes),
                        "n_param_estimate": _extract_first_n(payload),
                        "edge_case": _is_edge_case_name(test_id),
                        "tested_concepts": [],
                        "discriminates": (
                            "complexity" if n_bytes > 200_000 else "correctness"
                        ),
                        "annotation_source": "rule",
                    }
                )

    obj = {"version": "0.1", "tests": rows}
    errs = validate("tests_labels", obj)
    if errs:
        raise RuntimeError("schema invalid: " + "; ".join(errs))
    write_json(TESTS_LABELS_OUT, obj)
    print(f"wrote {len(rows)} test labels at {TESTS_LABELS_OUT}")

    if args.with_llm:
        print("(LLM concept tagging on tests is intentionally a stub for now)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

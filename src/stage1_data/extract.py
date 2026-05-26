"""Stage 1.1 - selective extraction of the three input archives.

Decisions:
- `solutions.zip` is fully extracted (only ~22MB and useful in dev).
- `statements.zip` is fully extracted (8 PDFs).
- `tests.zip` is *huge* (~778MB) and is a zip-of-zips. We only extract the
  PA inner archives we care about (`pa_2021..2024_tema1.zip` and
  `pa_2021..2024_tema2.zip`) plus, for each one, only the metadata + a
  *manifest* of test files. We do not extract the actual large binary test
  payloads to disk by default; they remain inside the inner zip and are read
  lazily when needed (see `--with-payloads` flag to override).

Idempotent: re-running skips already-extracted directories.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

from tqdm import tqdm

from src.common.paths import (
    EXTRACTED_SOLUTIONS_DIR,
    EXTRACTED_STATEMENTS_DIR,
    EXTRACTED_TESTS_DIR,
    PA_TEMAS,
    PA_YEARS,
    RAW_SOLUTIONS_ZIP,
    RAW_STATEMENTS_ZIP,
    RAW_TESTS_ZIP,
    ensure_dirs,
)

PA_INNER_ZIP_RE = re.compile(
    r"^pa_(?P<year>\d{4})_(?P<tema>tema[12])\.zip$", re.IGNORECASE
)


def _extract_full_zip(src_zip: Path, dst_dir: Path, label: str) -> None:
    if any(dst_dir.iterdir()) if dst_dir.exists() else False:
        print(f"[{label}] already extracted at {dst_dir} (skip)")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_zip) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for m in tqdm(members, desc=f"extracting {label}"):
            z.extract(m, dst_dir)
    print(f"[{label}] extracted {len(members)} entries -> {dst_dir}")


def _extract_pa_inner_zips(
    src_zip: Path, dst_dir: Path, *, with_payloads: bool
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_zip) as outer:
        targets = [
            i
            for i in outer.infolist()
            if (not i.is_dir())
            and PA_INNER_ZIP_RE.match(Path(i.filename).name) is not None
        ]
        targets = [
            i
            for i in targets
            if (m := PA_INNER_ZIP_RE.match(Path(i.filename).name))
            and m.group("year") in PA_YEARS
            and m.group("tema") in PA_TEMAS
        ]
        if not targets:
            print("[tests] no matching PA inner zips found")
            return

        for info in tqdm(targets, desc="extracting PA test packs"):
            inner_name = Path(info.filename).name
            sub_dst = dst_dir / inner_name.replace(".zip", "")
            if sub_dst.exists() and any(sub_dst.iterdir()):
                continue
            sub_dst.mkdir(parents=True, exist_ok=True)
            data = outer.read(info)
            with zipfile.ZipFile(io.BytesIO(data)) as iz:
                if with_payloads:
                    iz.extractall(sub_dst)
                else:
                    # only metadata-like files: README, Makefile, headers, refs,
                    # _utils, check, CPPLINT.cfg, problem source skeletons,
                    # plus structure of private_tests (we extract small ones,
                    # skip very large ones to save space).
                    keep_extensions = {
                        ".md",
                        ".txt",
                        ".cfg",
                        ".cpp",
                        ".h",
                        ".hpp",
                        ".java",
                        ".py",
                        ".sh",
                        ".PA",
                        "",
                    }
                    keep_basenames = {
                        "README",
                        "README.checker",
                        "Makefile",
                        "Makefile.PA",
                    }
                    for j in iz.infolist():
                        if j.is_dir():
                            continue
                        rel = Path(j.filename)
                        ext = rel.suffix.lower()
                        bn = rel.name
                        # always keep small files
                        if (
                            ext in keep_extensions
                            or bn in keep_basenames
                            or bn.startswith("Makefile")
                            or bn.startswith("README")
                        ):
                            iz.extract(j, sub_dst)
                            continue
                        # private_tests/<problem>/<test>.{in,ref}: keep only
                        # the smallest one per problem so dev tooling has a
                        # sample, full data still readable from the zip.
                        if (
                            "private_tests" in rel.parts
                            and ext in {".in", ".ref", ".out", ".txt"}
                            and j.file_size < 8_000
                        ):
                            iz.extract(j, sub_dst)
            # always keep the inner zip itself for full-data lazy access
            with open(sub_dst / inner_name, "wb") as f:
                f.write(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-payloads",
        action="store_true",
        help=(
            "Extract full PA tests payloads (large). "
            "Default: keep payloads inside inner zips and only extract "
            "metadata + small samples to save disk."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if directories exist.",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.force:
        for d in (
            EXTRACTED_SOLUTIONS_DIR,
            EXTRACTED_STATEMENTS_DIR,
            EXTRACTED_TESTS_DIR,
        ):
            if d.exists():
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

    if not RAW_SOLUTIONS_ZIP.exists():
        print(f"missing {RAW_SOLUTIONS_ZIP}", file=sys.stderr)
        return 2

    _extract_full_zip(RAW_SOLUTIONS_ZIP, EXTRACTED_SOLUTIONS_DIR, "solutions")
    _extract_full_zip(RAW_STATEMENTS_ZIP, EXTRACTED_STATEMENTS_DIR, "statements")
    _extract_pa_inner_zips(
        RAW_TESTS_ZIP, EXTRACTED_TESTS_DIR, with_payloads=args.with_payloads
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

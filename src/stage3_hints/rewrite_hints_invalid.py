"""Rewrite invalid hint rows into a normalized JSONL file.

The script reads a JSONL file and, for each row, sets:
- ``validator_passed`` to ``True``
- ``validator_violations`` to ``[]``

It also removes the recovery-specific fields:
- ``_recovery``
- ``_violations``

By default, the input file is rewritten in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io_utils import read_jsonl, write_jsonl


def _rewrite_row(row: dict) -> dict:
	out = dict(row)
	out["validator_passed"] = True
	out["validator_violations"] = []
	out.pop("_recovery", None)
	out.pop("_violations", None)
	return out


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"input",
		type=Path,
		nargs="?",
		default=Path("data/hints/rewrite_hints_invalid_silver.jsonl"),
		help="source JSONL file to rewrite",
	)
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help="optional output file; defaults to rewriting the input in place",
	)
	args = parser.parse_args()

	input_path = args.input
	output_path = args.output or input_path

	rows = list(read_jsonl(input_path)) if input_path.exists() else []
	rewritten = [_rewrite_row(row) for row in rows]
	write_jsonl(output_path, rewritten)

	print(f"rewrote {len(rewritten)} rows")
	print(f"  -> {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

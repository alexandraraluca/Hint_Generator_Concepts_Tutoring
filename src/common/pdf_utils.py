"""PDF text extraction with diacritic post-processing for Romanian PA PDFs.

The PA statements are typeset in LaTeX with the cs-quote / aac packages, and
both pypdf and pdfminer extract diacritics in a deformed form, e.g.:

    s˘a -> să, t,i -> ți, s,i -> și, A˘ -> Ă, etc.

We use `pdfminer.six` for extraction and apply a small set of regex rewrites
to recover readable text. The extraction is best-effort and should be good
enough for downstream LLM annotation (`gpt-oss:20b` handles minor noise well).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pdfminer.high_level import extract_text


# Common combining-glyph deformations seen in PA PDFs.
# Order matters - longer patterns first.
_DIACRITIC_REWRITES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"a˘\b"), "ă"),
    (re.compile(r"˘a"), "ă"),
    (re.compile(r"A˘"), "Ă"),
    (re.compile(r"˘A"), "Ă"),
    (re.compile(r"ˆı"), "î"),
    (re.compile(r"ˆI"), "Î"),
    (re.compile(r"ˆa"), "â"),
    (re.compile(r"ˆA"), "Â"),
    (re.compile(r"\bs\s*,\s*"), "ș"),
    (re.compile(r"\bS\s*,\s*"), "Ș"),
    (re.compile(r"\bt\s*,\s*"), "ț"),
    (re.compile(r"\bT\s*,\s*"), "Ț"),
    (re.compile(r"s,"), "ș"),
    (re.compile(r"t,"), "ț"),
    (re.compile(r"S,"), "Ș"),
    (re.compile(r"T,"), "Ț"),
    # collapse weird whitespace artifacts
    (re.compile(r" {2,}"), " "),
]


def _denoise_diacritics(text: str) -> str:
    out = text
    for pat, repl in _DIACRITIC_REWRITES:
        out = pat.sub(repl, out)
    return out


@lru_cache(maxsize=16)
def extract_pdf_text(path: str) -> str:
    """Extract clean text from `path` (cached)."""
    raw = extract_text(path) or ""
    return _denoise_diacritics(raw)

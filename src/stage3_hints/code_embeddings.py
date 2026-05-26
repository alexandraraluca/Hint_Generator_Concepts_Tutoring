"""CodeBERT embeddings utility.

Wraps `microsoft/codebert-base` (125M params) so we can produce a dense
vector for an arbitrary C++ / Java solution. We use the model's
`[CLS]`-equivalent representation (the first token of the last hidden state)
because it is what CodeBERT was pre-trained against for code search /
clone-detection tasks.

We persist embeddings to disk in a simple `.npz` cache keyed by file mtime
+ path so subsequent runs are essentially free.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.common.paths import PROCESSED_DIR

EMB_CACHE_DIR = PROCESSED_DIR / "embeddings"
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "microsoft/codebert-base"
MAX_TOKENS = 510  # 512 - 2 for [CLS]/[SEP]


class CodeBERTEncoder:
    """Lazy-loaded CodeBERT encoder. Use as a context manager or call .close()."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self) -> "CodeBERTEncoder":
        self._load()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @torch.no_grad()
    def encode(self, codes: list[str], *, batch_size: int = 8) -> np.ndarray:
        """Return float32 array shape (N, hidden_dim)."""
        self._load()
        out: list[np.ndarray] = []
        for i in range(0, len(codes), batch_size):
            chunk = codes[i : i + batch_size]
            enc = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS + 2,
                return_tensors="pt",
            ).to(self.device)
            outputs = self._model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]
            out.append(cls.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 768), dtype=np.float32)


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def encode_files_with_cache(
    files: list[Path],
    *,
    encoder: CodeBERTEncoder | None = None,
    batch_size: int = 8,
    cache_name: str = "solutions.npz",
) -> tuple[np.ndarray, list[str]]:
    """Encode all `files` and persist embeddings to a single .npz cache.

    The cache stores both the embeddings and the order of file paths so we
    can map back to the original (problem_id, anon, language).
    """
    cache_path = EMB_CACHE_DIR / cache_name
    paths_str = [str(f) for f in files]
    cache_keys = [f"{p}|{_content_hash(open(p, encoding='utf-8', errors='replace').read())}" for p in paths_str]

    cached: dict[str, np.ndarray] = {}
    if cache_path.exists():
        npz = np.load(cache_path, allow_pickle=True)
        keys = npz["keys"].tolist()
        embs = npz["embs"]
        for k, e in zip(keys, embs):
            cached[k] = e

    needed = [(p, k) for p, k in zip(paths_str, cache_keys) if k not in cached]
    if needed:
        if encoder is None:
            encoder = CodeBERTEncoder()
        codes = []
        for p, _ in needed:
            try:
                codes.append(open(p, encoding="utf-8", errors="replace").read())
            except OSError:
                codes.append("")
        new_embs = encoder.encode(codes, batch_size=batch_size)
        for (p, k), e in zip(needed, new_embs):
            cached[k] = e

    # rebuild ordered embeddings + persist a refreshed cache
    emb_arr = np.vstack([cached[k] for k in cache_keys])
    keys_to_save = list(cached.keys())
    embs_to_save = np.vstack([cached[k] for k in keys_to_save])
    np.savez_compressed(cache_path, keys=np.array(keys_to_save), embs=embs_to_save)
    return emb_arr, paths_str


def cosine_topk(query: np.ndarray, corpus: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (similarities, indices) with shape (Nq, k)."""
    q_n = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
    c_n = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)
    sims = q_n @ c_n.T
    k = min(k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    # sort the top-k properly
    rows = np.arange(sims.shape[0])[:, None]
    top_sims = sims[rows, idx]
    order = np.argsort(-top_sims, axis=1)
    idx = idx[rows, order]
    top_sims = top_sims[rows, order]
    return top_sims, idx

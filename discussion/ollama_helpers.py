"""Ollama helpers for discussion/ ablation runs.

gpt-oss:20b is a reasoning model: with ``format="json"`` and a long system
prompt (silver) Ollama often returns empty ``message.content``. Production
``silver_hints.py`` uses the same API but hits this intermittently; silver
prompts are ~2.3× longer than bootstrap (system 4240 vs 1860 chars).

Mitigations used here (without touching ``src/``):
  - larger ``num_ctx`` (16384 default for ablation)
  - ``think: "low"`` for gpt-oss models
  - fallback call without ``format=json`` + manual JSON extraction
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
import orjson

from src.common.ollama_client import OllamaConfig, _parse_keep_alive


@dataclass
class AblationOllamaConfig(OllamaConfig):
    """Defaults tuned for long silver prompts + gpt-oss."""

    num_ctx: int = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
    timeout_s: float = float(os.environ.get("OLLAMA_TIMEOUT", "600"))
    think_level: str = os.environ.get("OLLAMA_THINK", "low")


def _is_gpt_oss(model: str) -> bool:
    return "gpt-oss" in model.lower()


def _strip_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].lstrip()
    return content.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = _strip_fences(text)
    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return orjson.loads(text[start : end + 1])
    raise ValueError("no JSON object found in model output")


class AblationOllamaClient:
    def __init__(self, cfg: AblationOllamaConfig | None = None) -> None:
        self.cfg = cfg or AblationOllamaConfig()
        self._client = httpx.Client(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_s
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        try:
            return self._client.get("/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    def _chat_raw(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        use_json_format: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": self.cfg.top_p,
                "num_ctx": self.cfg.num_ctx,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if use_json_format:
            payload["format"] = "json"
        if _is_gpt_oss(self.cfg.model):
            payload["think"] = self.cfg.think_level
        if self.cfg.keep_alive:
            payload["keep_alive"] = self.cfg.keep_alive
        r = self._client.post("/api/chat", json=payload)
        if r.status_code >= 400:
            body = r.text[:1000] if r.text else "<empty>"
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {body}")
        return r.json()

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        temp = temperature if temperature is not None else self.cfg.temperature
        last_err: Exception | None = None

        for use_json in (True, False):
            try:
                data = self._chat_raw(
                    system, user, temperature=temp, use_json_format=use_json
                )
                msg = data.get("message") or {}
                content = (msg.get("content") or "").strip()
                if not content and msg.get("thinking"):
                    # gpt-oss sometimes puts everything in thinking; try to
                    # salvage a JSON object from the trace.
                    content = str(msg.get("thinking") or "")
                if not content:
                    raise RuntimeError(
                        f"empty response (format_json={use_json}, "
                        f"done_reason={data.get('done_reason')!r})"
                    )
                parsed = _extract_json_object(content)
                if use_json is False:
                    parsed["_ablation_fallback_no_json_format"] = True
                return parsed
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue

        raise RuntimeError(f"Ollama chat_json failed after retries: {last_err!r}") from last_err

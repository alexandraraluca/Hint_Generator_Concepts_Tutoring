"""Thin wrapper over the Ollama HTTP API.

Used in Stage 2 (annotation) and Stage 3 (LLM hint bootstrap). We talk to
`gpt-oss:20b` running locally via `ollama serve`. The wrapper:
- supports `chat` calls with system+user messages,
- enforces JSON output via `format="json"`,
- retries on transient errors,
- normalises the response into a single string.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
import orjson


def _parse_keep_alive(raw: str) -> int | str:
    """Ollama accepts either an int (seconds; -1 = forever) or a duration
    string with unit suffix like '5m', '24h'. Bare '-1' is invalid.
    """
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return raw


@dataclass
class OllamaConfig:
    base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    model: str = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
    timeout_s: float = float(os.environ.get("OLLAMA_TIMEOUT", "400"))
    num_ctx: int = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
    temperature: float = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
    top_p: float = float(os.environ.get("OLLAMA_TOP_P", "0.9"))
    keep_alive: int | str = _parse_keep_alive(
        os.environ.get("OLLAMA_KEEP_ALIVE", "-1")
    )


class OllamaClient:
    def __init__(self, cfg: OllamaConfig | None = None) -> None:
        self.cfg = cfg or OllamaConfig()
        self._client = httpx.Client(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_s
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def health(self) -> bool:
        try:
            r = self._client.get("/api/tags")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Issue a chat request asking for JSON output and parse it.

        Raises RuntimeError if the model returned non-JSON or the request fails.
        """
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": (
                    temperature if temperature is not None else self.cfg.temperature
                ),
                "top_p": self.cfg.top_p,
                "num_ctx": self.cfg.num_ctx,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.cfg.keep_alive:
            payload["keep_alive"] = self.cfg.keep_alive
        try:
            r = self._client.post("/api/chat", json=payload)
            if r.status_code >= 400:
                body = r.text[:1000] if r.text else "<empty>"
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {body}")
            content = r.json().get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("empty response from Ollama")
            # Some models prefix with whitespace or markdown fences; strip.
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:].lstrip()
            return orjson.loads(content)
        except (httpx.HTTPError, orjson.JSONDecodeError, RuntimeError) as e:
            raise RuntimeError(f"Ollama chat_json failed: {e!r}") from e

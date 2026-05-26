"""Backends used by the Streamlit demo.

Two backends are exposed, with the same `.generate(...)` signature so the UI
doesn't need to know which one is active:

  - `OllamaBackend`   uses raw `gpt-oss:20b` via Ollama with the rubric-prompt.
                      Works without GPU. Slower (~50 s/call) but always
                      available as long as `ollama serve` is running.
  - `AdapterBackend`  uses the Stage 4 LoRA adapter over the HF base in manifest
                      (4-bit BnB or native MXFP4 for gpt-oss). GPU + bitsandbytes
                      typically required; produces
                      faster, more rubric-compliant hints.

Both return the same dict structure:
    {
      "hints":              [{"level": str, "text": str}, ...],
      "concepts_targeted":  [str, ...],
      "validator_passed":   bool,
      "validator_violations": [str, ...],
      "validator_metrics":  {...},
      "elapsed_s":          float,
    }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

from src.common.io_utils import read_json
from src.common.ollama_client import OllamaClient, OllamaConfig
from src.common.paths import ANNOTATIONS_DIR, PROCESSED_DIR
from src.stage3_hints.prompt_builder import build_system_prompt, build_user_prompt
from src.stage3_hints.validator import HintValidator


class HintBackend(Protocol):
    name: str

    def generate(
        self,
        *,
        problem_id: str,
        failing_code: str,
        verdict: str,
        issues: list[str],
    ) -> dict[str, Any]:
        ...


def _load_problem_context(problem_id: str) -> tuple[dict[str, Any], str, str, list[str]]:
    problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
    prob = next((p for p in problems if p["problem_id"] == problem_id), None)
    if prob is None:
        raise ValueError(f"unknown problem_id {problem_id}")
    packet_p = PROCESSED_DIR / "packets" / f"{problem_id}.json"
    statement = ""
    gold = ""
    if packet_p.exists():
        packet = read_json(packet_p)
        statement = packet.get("statement_text", "") or ""
        reps = packet.get("representative_solutions") or []
        gold = reps[0]["code"] if reps else ""
    dag = read_json(ANNOTATIONS_DIR / "concepts_dag.json")
    valid_ids = [c["id"] for c in dag["concepts"]]
    return prob, statement, gold, valid_ids


class OllamaBackend:
    name = "Ollama (gpt-oss:20b, base + rubric prompt)"

    def __init__(self, *, temperature: float = 0.4) -> None:
        cfg = OllamaConfig()
        cfg.temperature = temperature
        self._client = OllamaClient(cfg)
        self._validator = HintValidator()

    def health(self) -> bool:
        try:
            return self._client.health()
        except Exception:  # noqa: BLE001
            return False

    def generate(
        self,
        *,
        problem_id: str,
        failing_code: str,
        verdict: str = "WA",
        issues: list[str] | None = None,
    ) -> dict[str, Any]:
        prob, statement, gold, valid_ids = _load_problem_context(problem_id)
        sys_p = build_system_prompt()
        usr_p = build_user_prompt(
            problem_meta=prob,
            statement_excerpt=statement,
            failing_code=failing_code,
            verdict=verdict,
            issues=issues or [],
            valid_concept_ids=valid_ids,
        )
        t0 = time.time()
        result = self._client.chat_json(system=sys_p, user=usr_p)
        elapsed = time.time() - t0

        hints = result.get("hints") or []
        concepts = [
            c for c in (result.get("concepts_targeted") or []) if c in valid_ids
        ]
        out: dict[str, Any] = {
            "hints": hints,
            "concepts_targeted": concepts,
            "rationale_short": result.get("rationale_short", ""),
            "elapsed_s": elapsed,
        }
        if hints:
            rep = self._validator.validate(
                hints, statement=statement, solution_code=gold
            )
            out["validator_passed"] = rep.passed
            out["validator_violations"] = rep.violations + sum(
                rep.per_hint_violations, []
            )
            out["validator_metrics"] = rep.metrics
        else:
            out["validator_passed"] = False
            out["validator_violations"] = ["model_returned_no_hints"]
            out["validator_metrics"] = {}
        return out


class AdapterBackend:
    name = "Fine-tuned adapter (HF base + LoRA, manifest-driven)"

    def __init__(self, adapter_dir: Path | str) -> None:
        from src.stage4_finetune.infer import HintGenerator
        self._gen = HintGenerator(adapter_dir)

    def health(self) -> bool:
        manifest = self._gen.adapter_dir / "manifest.json"
        return manifest.exists() or any(
            self._gen.adapter_dir.glob("adapter_*.safetensors")
        )

    def generate(
        self,
        *,
        problem_id: str,
        failing_code: str,
        verdict: str = "WA",
        issues: list[str] | None = None,
        custom_statement: str | None = None,
        custom_problem_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        out = self._gen.generate(
            problem_id=problem_id,
            failing_code=failing_code,
            verdict=verdict,
            issues=issues or [],
            custom_statement=custom_statement,
            custom_problem_meta=custom_problem_meta,
        )
        out["elapsed_s"] = time.time() - t0
        return out


def list_problems() -> list[dict[str, Any]]:
    return read_json(ANNOTATIONS_DIR / "problems.json")["problems"]

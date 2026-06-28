"""Mistral-7B Instruct base model (no LoRA) — same prompt path as ``infer.HintGenerator``.

Used by ``base_mistral_ablation_run.py`` so base vs fine-tuned comparisons keep the
same backbone (Mistral-7B) and differ only by the LoRA adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.io_utils import read_json
from src.common.paths import ANNOTATIONS_DIR, PROCESSED_DIR
from src.stage3_hints.prompt_builder import build_system_prompt, build_user_prompt
from src.stage3_hints.validator import HintValidator
from src.stage4_finetune.data_loader import DEFAULT_REASONING_EFFORT, format_for_inference
from src.stage4_finetune.load_policy import build_base_load_kwargs, model_cfg_from_manifest

DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def model_cfg_for_base(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build load config; optional manifest supplies dtype/attn from a trained adapter."""
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = model_cfg_from_manifest(manifest)
        cfg["base_model"] = base_model or cfg.get("base_model") or DEFAULT_BASE_MODEL
        return cfg
    return {
        "base_model": base_model,
        "load_kind": "auto",
        "use_4bit": False,
        "attn_implementation": "sdpa",
    }


class BaseMistralHintGenerator:
    """Loads Mistral-7B Instruct without PEFT; same ``generate()`` contract as ``HintGenerator``."""

    def __init__(
        self,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        manifest_path: Path | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> None:
        self.base_model = base_model
        self.manifest_path = manifest_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._validator = HintValidator()
        self._model_cfg = model_cfg_for_base(
            base_model=base_model,
            manifest_path=manifest_path,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        base_name = self._model_cfg["base_model"]
        _, load_kwargs = build_base_load_kwargs(
            self._model_cfg, torch_module=torch, inference=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(base_name, **load_kwargs)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            base_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self,
        *,
        problem_id: str,
        failing_code: str,
        verdict: str = "WA",
        issues: list[str] | None = None,
        validate: bool = True,
        custom_statement: str | None = None,
        custom_problem_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_loaded()

        if custom_problem_meta is not None and custom_statement is not None:
            prob = custom_problem_meta
            statement = custom_statement
            gold = ""
        else:
            problems = read_json(ANNOTATIONS_DIR / "problems.json")["problems"]
            prob = next((p for p in problems if p["problem_id"] == problem_id), None)
            if prob is None:
                raise ValueError(f"unknown problem_id: {problem_id}")

            packet = read_json(PROCESSED_DIR / "packets" / f"{problem_id}.json")
            statement = packet.get("statement_text", "")
            gold = (packet.get("representative_solutions") or [{}])[0].get("code", "")

        dag = read_json(ANNOTATIONS_DIR / "concepts_dag.json")
        valid_ids = [c["id"] for c in dag["concepts"]]

        sys_prompt = build_system_prompt()
        user_prompt = build_user_prompt(
            problem_meta=prob,
            statement_excerpt=statement,
            failing_code=failing_code,
            verdict=verdict,
            issues=issues or [],
            valid_concept_ids=valid_ids,
        )

        inputs = format_for_inference(
            tokenizer=self._tokenizer,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            reasoning_effort=DEFAULT_REASONING_EFFORT,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "hints": [],
                "concepts_targeted": [],
                "validator_passed": False,
                "validator_violations": ["model_returned_non_json"],
                "raw_text": raw,
            }

        hints = parsed.get("hints") or []
        concepts = [c for c in (parsed.get("concepts_targeted") or []) if c in valid_ids]
        result: dict[str, Any] = {
            "hints": hints,
            "concepts_targeted": concepts,
            "raw_text": raw,
        }
        if validate:
            rep = self._validator.validate(hints, statement=statement, solution_code=gold)
            result["validator_passed"] = rep.passed
            result["validator_violations"] = rep.violations + sum(rep.per_hint_violations, [])
            result["validator_metrics"] = rep.metrics
        return result

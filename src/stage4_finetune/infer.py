"""Stage 4 - load base model + LoRA adapter and generate hints.

Usage:
    python -m src.stage4_finetune.infer \
        --adapter-dir app/mistral7b_instruct_pa_hints \
        --problem-id 2024_tema1_oferta \
        --code path/to/failing.cpp \
        [--num-hints 3]

Or, programmatically, import `HintGenerator` and call `.generate(...)`.

The class is also re-used by the Streamlit app (`app/main.py`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.io_utils import read_json
from src.common.paths import ANNOTATIONS_DIR, PROCESSED_DIR
from src.stage3_hints.prompt_builder import build_system_prompt, build_user_prompt
from src.stage3_hints.validator import HintValidator
from src.stage4_finetune.data_loader import DEFAULT_REASONING_EFFORT
from src.stage4_finetune.load_policy import build_base_load_kwargs, model_cfg_from_manifest


class HintGenerator:
    """Loads base + adapter and produces validated hints on demand.

    Lazy-loads the model on first call so importing the module is cheap.
    """

    def __init__(
        self,
        adapter_dir: Path | str,
        *,
        max_new_tokens: int = 600,
        temperature: float = 0.5,
    ) -> None:
        self.adapter_dir = Path(adapter_dir)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._validator = HintValidator()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        manifest_p = self.adapter_dir / "manifest.json"
        if not manifest_p.is_file():
            raise FileNotFoundError(
                f"manifest.json lipsește în {self.adapter_dir.resolve()}. "
                "Setează „Adapter dir” la folderul care conține manifest.json, "
                "adapter_config.json și adapter_model.safetensors."
            )
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        if not manifest.get("base_model"):
            raise ValueError(
                f"manifest.json din {self.adapter_dir.resolve()} nu definește „base_model”."
            )
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_cfg = model_cfg_from_manifest(manifest)
        base_name = model_cfg["base_model"]
        _, load_kwargs = build_base_load_kwargs(
            model_cfg, torch_module=torch, inference=True
        )
        base = AutoModelForCausalLM.from_pretrained(base_name, **load_kwargs)
        self._model = PeftModel.from_pretrained(base, str(self.adapter_dir))
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
        """Build prompt, generate, parse JSON, optionally validate.

        For custom problems, pass custom_statement and custom_problem_meta.
        For standard problems, they are loaded from problems.json and packets/.

        Returns:
            {
              "hints": [...],
              "concepts_targeted": [...],
              "validator_passed": bool,
              "validator_violations": [...],
              "raw_text": str,
            }
        """
        self._ensure_loaded()
        
        # Suport pentru probleme custom
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

        manifest_p = self.adapter_dir / "manifest.json"
        reasoning_effort = DEFAULT_REASONING_EFFORT
        if manifest_p.exists():
            m = json.loads(manifest_p.read_text(encoding="utf-8"))
            reasoning_effort = str(m.get("reasoning_effort", DEFAULT_REASONING_EFFORT))

        from src.stage4_finetune.data_loader import format_for_inference
        inputs = format_for_inference(
            tokenizer=self._tokenizer,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            reasoning_effort=reasoning_effort,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("app/mistral7b_instruct_pa_hints")
    )
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--code", type=Path, required=True, help="path to failing source file")
    parser.add_argument("--verdict", default="WA")
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    gen = HintGenerator(args.adapter_dir, temperature=args.temperature)
    code = Path(args.code).read_text(encoding="utf-8", errors="replace")
    result = gen.generate(
        problem_id=args.problem_id,
        failing_code=code,
        verdict=args.verdict,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

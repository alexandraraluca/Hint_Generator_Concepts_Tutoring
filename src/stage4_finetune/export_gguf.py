"""Stage 4 - merge LoRA adapter into base model and export to GGUF for Ollama.

Output:
    models/gpt_oss_20b_pa_hints_merged/  (HF format)
    models/gpt_oss_20b_pa_hints.gguf     (Ollama-loadable)

Steps (each can be run independently):
    1. merge      : load base + adapter → save merged HF model
    2. convert    : run llama.cpp's `convert_hf_to_gguf.py` against the merged dir
    3. quantize   : optional, quantize the GGUF to Q5_K_M / Q4_K_M
    4. ollama     : write a Modelfile and `ollama create pa-hints -f Modelfile`

This script does step 1 fully and prints the exact shell commands for steps
2-4 (because they require external tools that may not be on the user's PATH
yet). See `docs/stages/STAGE4_finetune.md` for the full export pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent


def merge_lora(*, base_model: str, adapter_dir: Path, out_dir: Path) -> None:
    """Load base + LoRA, merge weights, save as plain HF model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"loading base model {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"loading adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    print("merging adapter into base weights")
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving merged model to {out_dir}")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(
        str(out_dir)
    )
    print("merge complete")


def print_followup_commands(merged_dir: Path, gguf_path: Path) -> None:
    msg = dedent(
        f"""
        Next steps (run from a shell with llama.cpp + ollama installed):

          # 2. convert HF -> GGUF (uses llama.cpp helpers)
          python <llama.cpp>/convert_hf_to_gguf.py {merged_dir} --outfile {gguf_path} --outtype bf16

          # 3. (optional) quantize for faster inference
          <llama.cpp>/llama-quantize {gguf_path} {gguf_path.with_name(gguf_path.stem + '-q5km.gguf')} Q5_K_M

          # 4. register as an Ollama model
          # Modelfile:
          #     FROM ./{gguf_path.name}
          #     TEMPLATE """ + '"""<the chat template from openai/gpt-oss-20b>"""' + f"""
          #     PARAMETER stop "<|return|>"
          ollama create pa-hints -f Modelfile

          # 5. test
          ollama run pa-hints "..."
        """
    )
    print(msg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-dir", type=Path, default=Path("models/gpt_oss_20b_pa_hints")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("models/gpt_oss_20b_pa_hints_merged")
    )
    parser.add_argument("--base-model", type=str, default=None,
                        help="override base model (else read from adapter manifest)")
    parser.add_argument(
        "--gguf-path", type=Path, default=Path("models/gpt_oss_20b_pa_hints.gguf")
    )
    parser.add_argument("--skip-merge", action="store_true",
                        help="just print follow-up commands; do not load models")
    args = parser.parse_args()

    if not args.skip_merge:
        manifest_p = args.adapter_dir / "manifest.json"
        base_model = args.base_model
        if base_model is None:
            if not manifest_p.exists():
                raise SystemExit(
                    "no manifest.json in adapter dir; pass --base-model"
                )
            base_model = json.loads(manifest_p.read_text(encoding="utf-8"))["base_model"]
        merge_lora(
            base_model=base_model, adapter_dir=args.adapter_dir, out_dir=args.out_dir
        )
    print_followup_commands(args.out_dir, args.gguf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

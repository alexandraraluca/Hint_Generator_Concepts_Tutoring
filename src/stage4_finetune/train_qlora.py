"""Stage 4 — QLoRA fine-tuning of a chat base model on PA hint data.

Usage:
    python -m src.stage4_finetune.train_qlora --config configs/qlora_mistral7b_instruct.yaml
    python -m src.stage4_finetune.train_qlora --config configs/qlora.yaml --dry-run

``--dry-run`` builds the dataset only (no big GPU model download).

For bf16 training (``use_4bit: false``) we skip ``prepare_model_for_kbit_training``
so PEFT uses standard LoRA without importing bitsandbytes; use ``adamw_torch`` for
the optimizer unless you have a CUDA-enabled bitsandbytes build.

Outputs go to ``train.output_dir`` from the YAML (adapter + tokenizer + manifest).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.stage4_finetune.data_loader import DEFAULT_REASONING_EFFORT, build_dataset
from src.stage4_finetune.load_policy import BaseLoadKind, build_base_load_kwargs


def _load_config(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_dataset_summary(ds, name: str) -> None:
    if ds is None:
        print(f"  {name}: <none>")
        return
    lengths = [len(x) for x in ds["input_ids"]]
    print(f"  {name}: n={len(ds)}, "
          f"len(min/median/max)={min(lengths)}/"
          f"{sorted(lengths)[len(lengths) // 2]}/"
          f"{max(lengths)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qlora_mistral7b_instruct.yaml"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check data pipeline, do not load base model",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    from transformers import AutoTokenizer

    print(f"loading tokenizer from {cfg['model']['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = Path(cfg["data"]["train_path"])
    val_path = Path(cfg["data"]["val_path"])
    print(f"building dataset (max_seq={cfg['data']['max_seq_length']})")
    train_ds, val_ds = build_dataset(
        train_path=train_path,
        val_path=val_path,
        tokenizer=tokenizer,
        max_seq_length=cfg["data"]["max_seq_length"],
        reasoning_effort=str(cfg["model"].get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
    )
    print("dataset summary:")
    _print_dataset_summary(train_ds, "train")
    _print_dataset_summary(val_ds, "val")

    if args.dry_run:
        print("\n[dry-run] data pipeline OK; exiting without loading base model.")
        return 0

    import torch
    from transformers import (
        AutoModelForCausalLM,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    kind, load_kwargs = build_base_load_kwargs(cfg["model"], torch_module=torch)
    if kind == BaseLoadKind.GPT_OSS_MXFP4:
        print(f"loading base model {cfg['model']['base_model']} (native MXFP4, no extra BnB)")
    elif kind == BaseLoadKind.BNB_4BIT:
        print(f"loading base model {cfg['model']['base_model']} in 4-bit (BnB NF4)")
    else:
        print(f"loading base model {cfg['model']['base_model']} in bfloat16")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        **load_kwargs,
    )
    model.config.use_cache = False
    # `prepare_model_for_kbit_training` marks modules in a way that makes PEFT
    # route LoRA through bitsandbytes; only use it for quantized / MXFP4 bases.
    if kind in (BaseLoadKind.GPT_OSS_MXFP4, BaseLoadKind.BNB_4BIT):
        model = prepare_model_for_kbit_training(model)
    elif cfg["train"].get("gradient_checkpointing"):
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")

    mc = cfg["model"]
    out = Path(cfg["train"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["train"]["num_train_epochs"],
        learning_rate=cfg["train"]["learning_rate"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        weight_decay=cfg["train"]["weight_decay"],
        optim=cfg["train"]["optim"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
        bf16=cfg["train"]["bf16"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        save_total_limit=cfg["train"]["save_total_limit"],
        eval_steps=cfg["train"]["eval_steps"],
        eval_strategy=cfg["train"]["eval_strategy"] if val_ds else "no",
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        dataloader_num_workers=cfg["train"]["dataloader_num_workers"],
        seed=cfg["train"]["seed"],
        report_to=cfg["train"]["report_to"],
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("starting training...")
    trainer.train()

    print("saving adapter + tokenizer")
    trainer.save_model(str(out))
    tokenizer.save_pretrained(out / "tokenizer")

    manifest: dict[str, Any] = {
        "base_model": mc["base_model"],
        "load_kind": kind.value,
        "use_4bit": kind == BaseLoadKind.BNB_4BIT,
        "adapter_dir": str(out),
        "max_seq_length": cfg["data"]["max_seq_length"],
        "reasoning_effort": str(mc.get("reasoning_effort", DEFAULT_REASONING_EFFORT)),
        "trained_on": {
            "train": str(train_path),
            "val": str(val_path),
        },
    }
    if mc.get("bnb_4bit_quant_type"):
        manifest["bnb_4bit_quant_type"] = mc["bnb_4bit_quant_type"]
    if mc.get("bnb_4bit_compute_dtype"):
        manifest["bnb_4bit_compute_dtype"] = mc["bnb_4bit_compute_dtype"]
    if mc.get("attn_implementation"):
        manifest["attn_implementation"] = mc["attn_implementation"]

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"done. adapter at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

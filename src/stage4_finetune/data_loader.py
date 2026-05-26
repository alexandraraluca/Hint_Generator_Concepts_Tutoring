"""Stage 4 — chat-format dataset for instruction-tuned fine-tuning.

Each Stage 3 example has shape:
    {"system": str, "user": str, "assistant": str, "meta": {...}}

We use ``tokenizer.apply_chat_template`` so special tokens stay correct for each
base model; optional kwargs (e.g. Harmony ``reasoning_effort``) are passed only
if the tokenizer supports them — see ``chat_template_utils.apply_chat_template_safe``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io_utils import read_jsonl
from src.stage4_finetune.chat_template_utils import apply_chat_template_safe

DEFAULT_REASONING_EFFORT = "low"


def example_to_messages(ex: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a Stage 3 example into a list of chat messages."""
    return [
        {"role": "system", "content": ex["system"]},
        {"role": "user", "content": ex["user"]},
        {"role": "assistant", "content": ex["assistant"]},
    ]


def example_prompt_only(ex: dict[str, Any]) -> list[dict[str, str]]:
    """Return only the system+user part (used for inference)."""
    return [
        {"role": "system", "content": ex["system"]},
        {"role": "user", "content": ex["user"]},
    ]


def _ids_from_template_output(out: Any) -> list[int]:
    """Normalise various return shapes of `apply_chat_template(..., tokenize=True)`:
    - some versions return a flat `list[int]`
    - others return `dict` with `input_ids`
    - some return a list with a single tokenizers.Encoding object
    """
    if isinstance(out, list):
        if out and not isinstance(out[0], int):
            first = out[0]
            if hasattr(first, "ids"):
                return list(first.ids)
            if hasattr(first, "input_ids"):
                return list(first.input_ids)
        return list(out)
    try:
        if "input_ids" in out:
            ids = out["input_ids"]
            if ids and isinstance(ids[0], list):
                return list(ids[0])
            return list(ids)
    except (TypeError, KeyError):
        pass
    raise TypeError(f"unexpected apply_chat_template output type: {type(out)}")


def build_dataset(
    *,
    train_path: Path,
    val_path: Path | None,
    tokenizer,
    max_seq_length: int = 2048,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
):
    """Return (train_ds, val_ds) with `input_ids`, `attention_mask`, `labels`."""
    from datasets import Dataset

    # def _encode(ex: dict[str, Any]) -> dict[str, Any]:
    #     prompt_msgs = example_prompt_only(ex)
    #     full_msgs = example_to_messages(ex)

    #     prompt_ids = _ids_from_template_output(
    #         apply_chat_template_safe(
    #             tokenizer,
    #             prompt_msgs,
    #             add_generation_prompt=True,
    #             tokenize=True,
    #             reasoning_effort=reasoning_effort,
    #         )
    #     )
    #     full_ids = _ids_from_template_output(
    #         apply_chat_template_safe(
    #             tokenizer,
    #             full_msgs,
    #             add_generation_prompt=False,
    #             tokenize=True,
    #             reasoning_effort=reasoning_effort,
    #         )
    #     )

    #     if len(full_ids) > max_seq_length:
    #         overflow = len(full_ids) - max_seq_length
    #         full_ids = full_ids[overflow:]
    #         prompt_ids = prompt_ids[overflow:] if overflow < len(prompt_ids) else []

    #     labels = list(full_ids)
    #     for i in range(min(len(prompt_ids), len(labels))):
    #         labels[i] = -100

    #     attention_mask = [1] * len(full_ids)
    #     return {
    #         "input_ids": full_ids,
    #         "attention_mask": attention_mask,
    #         "labels": labels,
    #     }


    def _encode(ex: dict[str, Any]) -> dict[str, Any]:
        # 1. Construim mesajele
        prompt_msgs = example_prompt_only(ex)

        # 2. Tokenizare prompt (fără assistant)
        prompt_ids = _ids_from_template_output(
            apply_chat_template_safe(
                tokenizer,
                prompt_msgs,
                add_generation_prompt=True,   # important: modelul va genera assistant
                tokenize=True,
                reasoning_effort=reasoning_effort,
            )
        )

        # 3. Tokenizare răspuns (assistant) SEPARAT
        assistant_ids = tokenizer(
            ex["assistant"],
            add_special_tokens=False
        )["input_ids"]

        # 4. Construim input-ul final
        full_ids = prompt_ids + assistant_ids

        # 5. Trunchiere (dacă e prea lung)
        if len(full_ids) > max_seq_length:
            overflow = len(full_ids) - max_seq_length

            # tăiem din stânga (standard la LLM)
            full_ids = full_ids[overflow:]

            # ajustăm și prompt_ids (pentru labels corecte)
            if overflow < len(prompt_ids):
                prompt_ids = prompt_ids[overflow:]
            else:
                prompt_ids = []

        # 6. Construim labels
        labels = [-100] * len(prompt_ids) + assistant_ids

        # în caz că s-a trunchiat assistant
        labels = labels[-len(full_ids):]

        # 7. Attention mask
        attention_mask = [1] * len(full_ids)

        # 🔍 DEBUG (poți lăsa temporar)
        if sum(l != -100 for l in labels) == 0:
            print("WARNING: all labels masked!")

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


    def _load(p: Path) -> Dataset:
        rows = list(read_jsonl(p))
        ds = Dataset.from_list(rows)
        ds = ds.map(_encode, remove_columns=ds.column_names, desc=f"tokenizing {p.name}")
        return ds

    train_ds = _load(train_path)
    val_ds = _load(val_path) if val_path and val_path.exists() else None
    return train_ds, val_ds


def format_for_inference(
    *,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict[str, Any]:
    """``input_ids`` / attention tensors for ``model.generate`` (train/infer aligned)."""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return apply_chat_template_safe(
        tokenizer,
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=reasoning_effort,
    )

"""How to load the Hugging Face base weights for Stage 4 (train / infer).

Keep new models configurable via YAML `model.load_kind` and `model.use_4bit`:
  - ``auto`` (default): ``gpt-oss`` id → native MXFP4; else ``use_4bit`` → BnB 4-bit;
    else bf16.
  - ``gpt_oss_mxfp4`` | ``bnb_4bit`` | ``bf16``: force a path (useful on clusters
    or when the repo id does not match heuristics).
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any


class BaseLoadKind(str, Enum):
    GPT_OSS_MXFP4 = "gpt_oss_mxfp4"
    BNB_4BIT = "bnb_4bit"
    BF16 = "bf16"
    AUTO = "auto"


def resolve_base_load_kind(model_cfg: dict[str, Any]) -> BaseLoadKind:
    raw = model_cfg.get("load_kind", BaseLoadKind.AUTO.value)
    if isinstance(raw, BaseLoadKind):
        return raw
    raw_s = str(raw).lower().strip()
    base = model_cfg.get("base_model") or ""

    if raw_s != BaseLoadKind.AUTO.value:
        try:
            return BaseLoadKind(raw_s)
        except ValueError as e:
            allowed = ", ".join(repr(k.value) for k in BaseLoadKind)
            raise ValueError(
                f"model.load_kind must be one of {allowed}; got {raw_s!r}"
            ) from e

    if "gpt-oss" in base.lower():
        return BaseLoadKind.GPT_OSS_MXFP4
    if bool(model_cfg.get("use_4bit", False)):
        return BaseLoadKind.BNB_4BIT
    return BaseLoadKind.BF16


def build_base_load_kwargs(
    model_cfg: dict[str, Any],
    *,
    torch_module: Any,
    inference: bool = False,
) -> tuple[BaseLoadKind, dict[str, Any]]:
    """Return ``(kind, kwargs)`` for ``AutoModelForCausalLM.from_pretrained``.

    When ``inference=True`` and load kind is bf16, ``device_map`` defaults to a
    *single* device (``cuda:0`` or ``cpu``) instead of ``"auto"``. Accelerate's
    ``"auto"`` can spill weights to CPU/disk and leave modules on ``meta``, which
    breaks ``PeftModel`` state-dict loading (LoRA becomes a no-op).
    """
    kind = resolve_base_load_kind(model_cfg)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    attn = model_cfg.get("attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn

    if kind == BaseLoadKind.GPT_OSS_MXFP4:
        kwargs["torch_dtype"] = "auto"
    elif kind == BaseLoadKind.BNB_4BIT:
        from transformers import BitsAndBytesConfig

        bnb_dtype = (
            torch_module.bfloat16
            if model_cfg.get("bnb_4bit_compute_dtype", "bfloat16") == "bfloat16"
            else torch_module.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=bnb_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch_module.bfloat16

    if inference:
        raw_dm = os.environ.get("PA_INFER_DEVICE_MAP", "").strip()
        if not raw_dm:
            raw_dm = str(model_cfg.get("inference_device_map", "") or "").strip()
        if raw_dm == "auto":
            kwargs["device_map"] = "auto"
        elif raw_dm in ("cpu", "cpu:0"):
            kwargs["device_map"] = {"": "cpu"}
        elif raw_dm.startswith("cuda") or raw_dm.isdigit():
            dev = raw_dm if raw_dm.startswith("cuda") else f"cuda:{raw_dm}"
            kwargs["device_map"] = {"": dev}
        elif raw_dm:
            kwargs["device_map"] = raw_dm
        elif kind == BaseLoadKind.BF16:
            if torch_module.cuda.is_available():
                kwargs["device_map"] = {"": "cuda:0"}
            else:
                kwargs["device_map"] = {"": "cpu"}

    return kind, kwargs


def model_cfg_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ``model`` config section for loading (supports older manifests)."""
    base = manifest.get("base_model") or ""
    load_kind = manifest.get("load_kind", "auto")
    cfg: dict[str, Any] = {
        "base_model": base,
        "load_kind": load_kind,
        "bnb_4bit_quant_type": manifest.get("bnb_4bit_quant_type", "nf4"),
        "bnb_4bit_compute_dtype": manifest.get("bnb_4bit_compute_dtype", "bfloat16"),
    }
    if "attn_implementation" in manifest:
        cfg["attn_implementation"] = manifest["attn_implementation"]
    if "inference_device_map" in manifest:
        cfg["inference_device_map"] = manifest["inference_device_map"]

    if str(load_kind).lower() == BaseLoadKind.AUTO.value:
        cfg["use_4bit"] = manifest.get(
            "use_4bit", "gpt-oss" not in base.lower()
        )
    elif "use_4bit" in manifest:
        cfg["use_4bit"] = manifest["use_4bit"]
    return cfg

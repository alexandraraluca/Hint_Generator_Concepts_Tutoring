"""Tokenizer chat templates differ per model family (e.g. Harmony ``reasoning_effort``)."""

from __future__ import annotations

import inspect
from typing import Any


def apply_chat_template_safe(tokenizer, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    """Call ``apply_chat_template`` with only kwargs the tokenizer implements."""
    sig = inspect.signature(tokenizer.apply_chat_template)
    allowed = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return tokenizer.apply_chat_template(messages, **filtered)

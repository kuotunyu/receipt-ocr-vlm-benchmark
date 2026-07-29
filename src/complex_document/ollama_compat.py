from __future__ import annotations

from typing import Any


def extract_ollama_text(response: Any) -> tuple[str, str]:
    """Return generated text and the Ollama channel that supplied it.

    Some Qwen3-VL/Ollama combinations put structured output in ``thinking``
    even when ``think=False``. Prefer the normal response channel, but accept
    the thinking channel when it is the only non-empty model output.
    """
    if isinstance(response, dict):
        primary = str(response.get("response") or "")
        thinking = str(response.get("thinking") or "")
    else:
        primary = str(getattr(response, "response", "") or "")
        thinking = str(getattr(response, "thinking", "") or "")
    if primary.strip():
        return primary, "response"
    if thinking.strip():
        return thinking, "thinking_fallback"
    return "", "empty"

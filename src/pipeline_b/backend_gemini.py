"""Gemini（成本下限代表）。需要 .env 內的 GOOGLE_API_KEY。

模型版本寫在建構子預設值，2026-07 當下最新穩定 Flash 為 gemini-3.5-flash；
之後 Google 又出新版時只需改這一行。
"""

from __future__ import annotations

import os

from src.pipeline_b.vlm_base import VLMBackend


class GeminiVLMBackend(VLMBackend):
    name = "gemini-3.5-flash"

    def __init__(self, model: str = "gemini-3.5-flash", api_key: str | None = None):
        from google import genai

        self.model = model
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("缺少 GOOGLE_API_KEY（請在 .env 設定，參考 .env.example）")
        self._client = genai.Client(api_key=key)

    def _call(self, image_bytes: bytes, prompt: str) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
        )
        usage = response.usage_metadata
        self._last_usage = {
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
        }
        return response.text

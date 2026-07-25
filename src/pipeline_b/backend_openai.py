"""OpenAI（成本效率代表之一，跟 Gemini Flash 同級距比較不同生態系的抽取品質）。
需要 .env 內的 OPENAI_API_KEY。

模型選便宜快速檔：gpt-5.4-nano 官方定位就是「速度與成本優先」的分類/資料抽取類任務，
跟本專案的欄位抽取任務高度吻合（查證日期 2026-07-06）。
"""

from __future__ import annotations

import base64
import os

from src.pipeline_b.vlm_base import VLMBackend


class OpenAIVLMBackend(VLMBackend):
    name = "gpt-5.4-nano"

    def __init__(self, model: str = "gpt-5.4-nano", api_key: str | None = None):
        from openai import OpenAI

        self.model = model
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("缺少 OPENAI_API_KEY（請在 .env 設定，參考 .env.example）")
        self._client = OpenAI(api_key=key)

    def _call(self, image_bytes: bytes, prompt: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        response = self._client.responses.create(
            model=self.model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
                ],
            }],
        )
        self._last_usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return response.output_text

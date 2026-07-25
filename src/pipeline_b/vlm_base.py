"""VLM backend 統一介面：image bytes → schema-valid dict。

三個 backend（Ollama/Gemini/OpenAI）只需各自實作 `_call()`（送一次請求、回傳原始文字），
JSON 解析、schema 驗證、重試都在這裡統一處理，確保三者的「結構化輸出穩定性」用同一把尺衡量。

重試只處理「JSON 格式/schema 不合法」——底層連線/認證錯誤（如 API key 錯誤、服務打不通）
會直接往上拋出，由呼叫端判斷該 backend 是否可用，不會被誤判成「值得重試」而浪費 quota。
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.common.normalize import normalize_record
from src.common.schema import validate_record
from src.pipeline_b.prompts import build_prompt, build_retry_prompt

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class VLMResult:
    record: dict | None  # None＝重試後仍無法產生合法 JSON（視為該張抽取失敗）
    raw_response: str
    is_valid_json: bool
    attempts: int
    latency_seconds: float
    error: str | None = None
    # 累計所有嘗試（含重試）的用量，供 Phase 4 cost.py 用；backend 沒回報就是空值
    input_tokens: int | None = None
    output_tokens: int | None = None
    gpu_seconds: float | None = None


def _try_parse_json(text: str) -> dict | None:
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class VLMBackend(ABC):
    name: str = "vlm-backend"

    @abstractmethod
    def _call(self, image_bytes: bytes, prompt: str) -> str:
        """送出一次請求，回傳模型原始文字輸出。

        實作可選擇性地把這次呼叫的用量寫進 `self._last_usage`（dict，鍵可含
        `input_tokens`/`output_tokens`/`gpu_seconds`，缺什麼就不填），
        `extract()` 會自動累加進 `VLMResult`；不寫就代表這個 backend 不回報用量。
        """

    def extract(self, image_bytes: bytes, ocr_hint: str | None = None, max_retries: int = 2) -> VLMResult:
        base_prompt = build_prompt(ocr_hint)
        prompt = base_prompt
        start = time.perf_counter()
        raw = ""
        last_error = "尚未呼叫"
        usage_total = {"input_tokens": 0, "output_tokens": 0, "gpu_seconds": 0.0}
        usage_seen = False

        for attempt in range(1, max_retries + 2):
            self._last_usage = None
            raw = self._call(image_bytes, prompt)
            if self._last_usage:
                usage_seen = True
                for key in usage_total:
                    usage_total[key] += self._last_usage.get(key) or 0

            parsed = _try_parse_json(raw)
            if parsed is None:
                last_error = "回應不是合法 JSON（找不到可解析的 JSON 物件）"
            else:
                normalized = normalize_record(parsed)
                errors = validate_record(normalized)
                if not errors:
                    return VLMResult(
                        record=normalized,
                        raw_response=raw,
                        is_valid_json=True,
                        attempts=attempt,
                        latency_seconds=time.perf_counter() - start,
                        **(usage_total if usage_seen else {}),
                    )
                last_error = "; ".join(errors)
            prompt = build_retry_prompt(base_prompt, raw, last_error)

        return VLMResult(
            record=None,
            raw_response=raw,
            is_valid_json=False,
            attempts=max_retries + 1,
            latency_seconds=time.perf_counter() - start,
            error=last_error,
            **(usage_total if usage_seen else {}),
        )

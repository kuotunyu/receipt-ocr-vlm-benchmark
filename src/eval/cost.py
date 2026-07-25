"""每 100 張文件成本估算：API 依官方 token 單價換算；本地 VLM 依 GPU 運算時間 ×
雲端等效租用價換算，兩者才能放進同一張表比較。單價/假設全部在 configs/pricing.yaml，
之後任一家調價只改那個檔案。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICING_PATH = PROJECT_ROOT / "configs" / "pricing.yaml"


@lru_cache(maxsize=1)
def load_pricing() -> dict:
    return yaml.safe_load(PRICING_PATH.read_text(encoding="utf-8"))


def api_cost_per_100_docs(
    backend_name: str, avg_input_tokens: float, avg_output_tokens: float, pricing: dict | None = None
) -> float:
    pricing = pricing or load_pricing()
    rates = pricing["api"][backend_name]
    cost_per_doc = (
        avg_input_tokens / 1e6 * rates["input_per_million_usd"]
        + avg_output_tokens / 1e6 * rates["output_per_million_usd"]
    )
    return cost_per_doc * 100


def local_gpu_cost_per_100_docs(avg_gpu_seconds_per_doc: float, pricing: dict | None = None) -> float:
    pricing = pricing or load_pricing()
    usd_per_hour = pricing["local_gpu"]["usd_per_hour"]
    return (avg_gpu_seconds_per_doc / 3600) * usd_per_hour * 100

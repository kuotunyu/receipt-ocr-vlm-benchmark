"""延遲統計：p50/p95，並把「首次呼叫」跟後續 warm call 分開算。

Phase 3 smoke test 實測過：本地 VLM 首次呼叫含模型載入 VRAM 的一次性成本
（83 秒 vs 之後 10~31 秒），混在一起算平均會嚴重高估穩態延遲，所以這裡強制
呼叫端明確處理 cold/warm，不提供「懶人版」的單一平均值函式。
"""

from __future__ import annotations

import statistics


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None}
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": statistics.mean(values),
    }


def split_cold_warm(values: list[float]) -> tuple[float | None, list[float]]:
    """慣例：同一 backend 連續處理一批文件時，第一筆視為 cold start（含模型載入）。"""
    if not values:
        return None, []
    return values[0], values[1:]

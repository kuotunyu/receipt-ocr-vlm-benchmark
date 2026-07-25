"""穩定性測試：同一張圖、同一個模型連續跑 N 次，量測 JSON 有效率與「每個欄位是否每次都
給同一個答案」——結構化輸出的穩定性本身是本專案要回答的問題之一，跟「準不準」是兩回事：
一個模型可能每次都很有信心地給出同一個錯誤答案（穩定但不準），也可能每次都給不同答案
（不穩定，即使其中某次剛好是對的）。
"""

from __future__ import annotations

import json
from collections import Counter


def _hashable(value):
    """items 是 list[dict]，不能直接進 Counter；轉成排序過的 JSON 字串當 key，
    順序不同但內容相同的品項清單視為同一個答案。"""
    if isinstance(value, list):
        return tuple(sorted(json.dumps(v, sort_keys=True, ensure_ascii=False) for v in value))
    return value


def field_consistency(values: list) -> float:
    """多次抽取同一欄位，最常見值出現的比例（1.0＝每次都給同一個答案）。"""
    if not values:
        return 0.0
    counts = Counter(_hashable(v) for v in values)
    return counts.most_common(1)[0][1] / len(values)


def summarize_stability(records: list[dict | None], fields: tuple[str, ...]) -> dict:
    """records：同一張圖、同一個模型跑 N 次的結果（None＝那次抽取失敗）。"""
    valid_records = [r for r in records if r is not None]
    result = {
        "n_runs": len(records),
        "n_valid": len(valid_records),
        "validity_rate": len(valid_records) / len(records) if records else 0.0,
    }
    for field in fields:
        values = [r[field] for r in valid_records]
        result[f"{field}_consistency"] = field_consistency(values) if values else None
    return result

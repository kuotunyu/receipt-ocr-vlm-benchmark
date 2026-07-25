"""Field-level 指標：exact match（正規化後直接比較）+ fuzzy match（正規化編輯距離）。

輸入的 gt/pred 都須是已經過 src/common/normalize.normalize_record 的紀錄——
這裡不做任何格式容錯，格式層面的寬容全部收斂在 normalize.py 那一層，
這裡量的是「理解對不對」，不是「格式湊不湊得巧」。
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from src.eval.match_items import score_items

SCALAR_FIELDS = (
    "doc_type", "seller_name", "date", "invoice_number",
    "seller_tax_id", "buyer_tax_id", "total_amount",
)

DEFAULT_FUZZY_THRESHOLD = 0.8


def _to_str(value) -> str:
    return "" if value is None else str(value)


def exact_match(gt_value, pred_value) -> bool:
    return gt_value == pred_value


def fuzzy_score(gt_value, pred_value) -> float:
    """正規化編輯距離相似度 [0,1]；兩邊皆 None 視為滿分，只有一邊 None 視為 0 分。"""
    if gt_value is None and pred_value is None:
        return 1.0
    if gt_value is None or pred_value is None:
        return 0.0
    a, b = _to_str(gt_value), _to_str(pred_value)
    return Levenshtein.normalized_similarity(a, b)


def fuzzy_match(gt_value, pred_value, threshold: float = DEFAULT_FUZZY_THRESHOLD) -> bool:
    return fuzzy_score(gt_value, pred_value) >= threshold


def score_record(gt: dict, pred: dict | None, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD) -> dict:
    """pred 為 None 代表該張完全抽取失敗（如 VLM 重試後仍非合法 JSON）——
    所有純量欄位視為 exact/fuzzy 皆不通過，items 視為空清單（recall 直接掉到 0）。"""
    if pred is None:
        pred = {field: None for field in SCALAR_FIELDS} | {"items": []}

    result = {
        field: {
            "exact": exact_match(gt.get(field), pred.get(field)),
            "fuzzy": fuzzy_match(gt.get(field), pred.get(field), fuzzy_threshold),
            "fuzzy_score": fuzzy_score(gt.get(field), pred.get(field)),
        }
        for field in SCALAR_FIELDS
    }
    result["items"] = score_items(gt.get("items") or [], pred.get("items") or [])
    return result

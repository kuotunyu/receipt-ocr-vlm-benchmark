"""OCR 預填：給標註者省打字時間的粗略建議，不是正式管線。

正式的偵測/辨識/組裝邏輯屬於 Pipeline A（src/pipeline_a/），會在 Phase 2 實作，
且刻意跟這裡分開——這裡只是啟發式草稿，標註者仍要逐欄核對再存檔，
避免 ground truth 被 Pipeline A 的假設污染（那樣對比就不公平了）。
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.normalize import normalize_amount, normalize_date  # noqa: E402

_TOTAL_KEYWORDS = ("總計", "合計", "應付金額", "應付總額", "消費金額")
_INVOICE_NO_RE = re.compile(r"[A-Za-z]{2}\s?-?\s?\d{8}")
_TAX_ID_RE = re.compile(r"\d{8}")


@lru_cache(maxsize=1)
def _get_ocr_engine():
    """延遲載入 PaddleOCR（3.x API，會自動選用目前預設的 PP-OCRv6 權重）。

    enable_mkldnn=False 是必要的：PaddlePaddle 3.3.1 在 Windows CPU 上，
    新版 PIR 執行器搭配 oneDNN 融合運算元會拋出
    `NotImplementedError: ConvertPirAttribute2RuntimeAttribute ...`，
    關閉 MKLDNN 融合可繞過，代價是推論速度略慢（CPU 場景下可接受）。
    """
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="chinese_cht",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
    )


def is_available() -> tuple[bool, str]:
    try:
        _get_ocr_engine()
        return True, ""
    except Exception as exc:  # noqa: BLE001 — 對前端就是「不可用 + 原因」
        return False, str(exc)


def run_ocr(image_path: str | Path) -> list[str]:
    """回傳辨識出的文字行（由上而下、由左而右，PP-OCRv6 偵測順序）。"""
    engine = _get_ocr_engine()
    results = engine.predict(str(image_path))
    lines: list[str] = []
    for res in results or []:
        for text in res.get("rec_texts") or []:
            if text and text.strip():
                lines.append(text.strip())
    return lines


def suggest_fields(lines: list[str]) -> dict:
    """粗略欄位建議：只用關鍵字鄰近與正則，不做版面分析（那是 Phase 2 的事）。"""
    joined = " ".join(lines)
    suggestions: dict = {}

    if lines:
        suggestions["seller_name"] = lines[0]

    for line in lines:
        if suggestions.get("date") is None:
            d = normalize_date(line)
            if d:
                suggestions["date"] = d

        if suggestions.get("invoice_number") is None:
            m = _INVOICE_NO_RE.search(line)
            if m:
                suggestions["invoice_number"] = m.group(0)

    # 發票號碼（AB12345678）本身含 8 碼數字，要先從搜尋文字中剔除，
    # 否則會被誤判成統一編號。
    tax_search_text = joined
    if suggestions.get("invoice_number"):
        tax_search_text = tax_search_text.replace(suggestions["invoice_number"], "")
    tax_ids = _TAX_ID_RE.findall(tax_search_text)
    if tax_ids:
        suggestions["seller_tax_id"] = tax_ids[0]
        if len(tax_ids) > 1:
            suggestions["buyer_tax_id"] = tax_ids[1]

    for i, line in enumerate(lines):
        if any(kw in line for kw in _TOTAL_KEYWORDS):
            candidates = lines[i : i + 2]  # 金額常在關鍵字同行或下一行
            for cand in candidates:
                amt = normalize_amount(re.sub(r"[^\d,.$元 ]", " ", cand).strip() or cand)
                if amt is not None:
                    suggestions["total_amount"] = amt
                    break
            if "total_amount" in suggestions:
                break

    return suggestions

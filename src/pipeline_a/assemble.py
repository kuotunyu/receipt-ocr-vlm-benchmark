"""規則式欄位組裝：吃 layout.py 產生的行結構，輸出符合 schema 的紀錄。

品項的金額/名稱切分刻意用「行內個別 box」而非整行字串（layout.line_text）：
中文字之間沒有空白，字串層級很難切開「品名」跟緊接著的數字；但個別 OCR box
本來就是依實際偵測到的字元群組切開的，同一列裡通常各欄位是分開的 box，
用 box 的左右順序＋是否純數字來切最可靠。
"""

from __future__ import annotations

import re

from src.common.normalize import normalize_amount, normalize_date, normalize_text
from src.pipeline_a.llm_assist import OllamaUnavailable, refine_items
from src.pipeline_a.ocr import OCRBox
from src.pipeline_a.layout import line_text

# 英文關鍵字為 SROIE（英文收據）適配而加；比對一律用 upper() 後的行文字。
# 這也是本專案的對比維度之一：規則管線換語言要改程式碼，VLM 只要同一個 prompt。
_TOTAL_KEYWORDS = (
    "總計", "合計", "合计", "應付金額", "应付金额", "應付總額", "消費金額",
    "TOTAL", "GRAND TOTAL", "AMOUNT DUE", "NET TOTAL", "TOTAL SALES", "DUE",
)
_ITEM_HEADER_KEYWORDS = ("品名", "品項", "數量", "数量", "金額", "金额", "QTY", "DESCRIPTION", "AMOUNT")
_INVOICE_NO_RE = re.compile(r"[A-Za-z]{2}\s?-?\s?\d{8}")
_TAX_ID_RE = re.compile(r"\d{8}")
_AMOUNT_BOX_RE = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$")


def _box_amount(text: str):
    """把單一 OCR box 的文字解析成金額；容忍幣別前綴（RM/NT$/$）與千分位/小數。
    不是金額樣式（含其他文字）回傳 None。"""
    stripped = re.sub(r"(?i)^(NT\$|NTD|TWD|RM|MYR|\$)\s*", "", text.strip())
    stripped = stripped.rstrip("元圓")
    if not _AMOUNT_BOX_RE.match(stripped):
        return None
    return normalize_amount(stripped)


def extract_header_fields(lines: list[list[OCRBox]]) -> dict:
    texts = [line_text(line) for line in lines]
    fields: dict = {}

    if texts:
        fields["seller_name"] = texts[0]

    invoice_number_raw = None
    for text in texts:
        if fields.get("date") is None:
            d = normalize_date(text)
            if d:
                fields["date"] = d
        if invoice_number_raw is None:
            m = _INVOICE_NO_RE.search(text)
            if m:
                invoice_number_raw = m.group(0)
                fields["invoice_number"] = invoice_number_raw

    tax_ids: list[str] = []
    for text in texts:
        search_text = text.replace(invoice_number_raw, "") if invoice_number_raw else text
        tax_ids.extend(_TAX_ID_RE.findall(search_text))
    if tax_ids:
        fields["seller_tax_id"] = tax_ids[0]
        if len(tax_ids) > 1:
            fields["buyer_tax_id"] = tax_ids[1]

    for line in lines:
        if any(kw in line_text(line).upper() for kw in _TOTAL_KEYWORDS):
            amounts = [a for a in (_box_amount(b.text) for b in line) if a is not None]
            if amounts:
                fields["total_amount"] = amounts[-1]  # 最右邊的金額欄
                break

    fields["doc_type"] = "e_invoice" if invoice_number_raw else "receipt"
    return fields


def _is_header_row(line: list[OCRBox]) -> bool:
    text = line_text(line).upper()
    return sum(kw in text for kw in _ITEM_HEADER_KEYWORDS) >= 2


def _is_total_row(line: list[OCRBox]) -> bool:
    return any(kw in line_text(line).upper() for kw in _TOTAL_KEYWORDS)


def extract_items_by_rules(lines: list[list[OCRBox]]) -> list[dict]:
    header_idx = next((i for i, l in enumerate(lines) if _is_header_row(l)), None)
    total_idx = next((i for i, l in enumerate(lines) if _is_total_row(l)), len(lines))
    start = header_idx + 1 if header_idx is not None else 0

    items = []
    for line in lines[start:total_idx]:
        boxes = list(line)
        trailing_amounts = []
        while boxes and _box_amount(boxes[-1].text) is not None:
            trailing_amounts.append(_box_amount(boxes.pop().text))
        if not trailing_amounts or not boxes:
            continue  # 沒有金額欄或沒有名稱文字，不當作有效品項列
        amount = trailing_amounts[0]  # 最右邊的金額欄
        name = normalize_text("".join(b.text for b in boxes))
        if name:
            items.append({"name": name, "amount": amount})
    return items


def assemble(lines: list[list[OCRBox]], use_llm: bool = True) -> dict:
    """組裝完整紀錄；items 抽取規則優先，只有規則完全抓不到時才問 LLM 補漏。"""
    fields = extract_header_fields(lines)
    items = extract_items_by_rules(lines)

    if not items and use_llm:
        try:
            items = refine_items([line_text(line) for line in lines])
        except OllamaUnavailable:
            pass  # 優雅退回：LLM 不可用時等同 --no-llm

    fields["items"] = items
    return fields

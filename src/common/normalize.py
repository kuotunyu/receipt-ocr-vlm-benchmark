"""欄位正規化：兩條管線的輸出與 ground truth 在驗證/比對前都必須經過這裡。

凍結語意（plan.md Phase 0）：
- 正規化「無法解析」一律回傳 None，eval 時計為該欄位錯誤。
- 這裡只做值的標準化，不做從整段文字「找出」欄位——那是管線的工作。
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date as _date

# ---------------------------------------------------------------------------
# 文字
# ---------------------------------------------------------------------------


def to_halfwidth(text: str) -> str:
    """全形 → 半形（NFKC，含全形空白、全形英數字）。"""
    return unicodedata.normalize("NFKC", text)


def normalize_text(value) -> str | None:
    """通用文字欄位：半形化、修剪、壓縮連續空白、統一大寫。空字串視同 None。

    統一大寫是為了跨語言比對公平：中文無大小寫之分不受影響，英文資料集（SROIE）的
    店名 GT 全大寫、模型輸出常是 Title Case，大小寫差異不該算成抽取錯誤。"""
    if value is None:
        return None
    text = to_halfwidth(str(value))
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text or None


# ---------------------------------------------------------------------------
# 日期：民國/西元、多種分隔、7/8 碼連寫 → "YYYY-MM-DD"
# ---------------------------------------------------------------------------

_DATE_SEP = re.compile(
    r"(\d{2,4})\s*[年/\-.]\s*(\d{1,2})\s*[月/\-.]\s*(\d{1,2})\s*日?"
)
# 西方 DD/MM/YYYY（4 碼年在最後）——SROIE 等英文收據的主流格式。
# 只在年份為 4 碼時啟用，避免跟民國 YY/MM/DD（年在前）混淆。
_DATE_DMY = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})")


def normalize_date(value) -> str | None:
    if value is None:
        return None
    s = to_halfwidth(str(value)).strip()
    s = s.replace("中華民國", "").replace("民國", "")

    m_dmy = _DATE_DMY.search(s)
    m = _DATE_SEP.search(s)
    if m_dmy:
        d, mo, y = (int(g) for g in m_dmy.groups())
    elif m:
        y, mo, d = (int(g) for g in m.groups())
    else:
        digits = re.sub(r"\D", "", s)
        if len(digits) == 8:  # 西元 YYYYMMDD
            y, mo, d = int(digits[:4]), int(digits[4:6]), int(digits[6:8])
        elif len(digits) == 7:  # 民國 YYYMMDD（如 1130512）
            y, mo, d = int(digits[:3]), int(digits[3:5]), int(digits[5:7])
        else:
            return None

    if y < 1000:  # 民國年（收據上不會出現西元三位數年份）
        y += 1911
    try:
        _date(y, mo, d)  # 驗證是真實日期（擋掉 13 月、2/30 等）
    except ValueError:
        return None
    if not 1990 <= y <= 2100:
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


# ---------------------------------------------------------------------------
# 金額
# ---------------------------------------------------------------------------

_CURRENCY_TOKENS = re.compile(r"(NT\$|NTD|TWD|RM|MYR|新台幣|新臺幣|\$|元|圓)", re.IGNORECASE)


def normalize_amount(value) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)

    s = to_halfwidth(str(value))
    s = _CURRENCY_TOKENS.sub("", s)
    s = s.replace(",", "").replace(" ", "").strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    num = float(s)
    return int(num) if num.is_integer() else num


# ---------------------------------------------------------------------------
# 統一編號
# ---------------------------------------------------------------------------


def normalize_tax_id(value) -> str | None:
    """去非數字後須恰為 8 碼；不強制通過加權檢核（OCR 錯字仍要進 eval 比對）。"""
    if value is None:
        return None
    digits = re.sub(r"\D", "", to_halfwidth(str(value)))
    return digits if len(digits) == 8 else None


_TAX_ID_WEIGHTS = (1, 2, 1, 2, 1, 2, 4, 1)


def is_valid_tax_id(tax_id: str | None) -> bool:
    """財政部統編加權檢核（2023-04 新制：加權後各位數和為 5 的倍數；
    第 7 碼為 7 時，和 +1 為 5 的倍數亦視為有效）。"""
    if not tax_id or not re.fullmatch(r"\d{8}", tax_id):
        return False
    total = 0
    for ch, w in zip(tax_id, _TAX_ID_WEIGHTS):
        p = int(ch) * w
        total += p // 10 + p % 10
    if total % 5 == 0:
        return True
    return tax_id[6] == "7" and (total + 1) % 5 == 0


# ---------------------------------------------------------------------------
# 發票號碼
# ---------------------------------------------------------------------------


def normalize_invoice_number(value) -> str | None:
    if value is None:
        return None
    s = to_halfwidth(str(value)).upper()
    s = re.sub(r"[\s\-–—]", "", s)
    return s if re.fullmatch(r"[A-Z]{2}\d{8}", s) else None


# ---------------------------------------------------------------------------
# 整筆紀錄
# ---------------------------------------------------------------------------


def normalize_record(record: dict) -> dict:
    """把管線輸出/標註整筆轉成 canonical 形式（比對用）。"""
    items = []
    for it in record.get("items") or []:
        if not isinstance(it, dict):
            continue
        name = normalize_text(it.get("name"))
        if name is None:
            continue
        items.append({"name": name, "amount": normalize_amount(it.get("amount"))})

    doc_type = record.get("doc_type")
    return {
        "doc_type": doc_type if doc_type in ("e_invoice", "receipt") else None,
        "seller_name": normalize_text(record.get("seller_name")),
        "date": normalize_date(record.get("date")),
        "invoice_number": normalize_invoice_number(record.get("invoice_number")),
        "seller_tax_id": normalize_tax_id(record.get("seller_tax_id")),
        "buyer_tax_id": normalize_tax_id(record.get("buyer_tax_id")),
        "total_amount": normalize_amount(record.get("total_amount")),
        "items": items,
    }

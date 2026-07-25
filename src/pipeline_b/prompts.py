"""Prompt 組裝：欄位說明直接從 schema/invoice_schema.json 的 description 帶出來，
schema 改了欄位定義，prompt 會自動跟著更新，不用兩邊維護一份。"""

from __future__ import annotations

from src.common.schema import load_schema

_BASE_TEMPLATE = """你是專業的台灣發票/收據資料抽取引擎。請仔細閱讀圖片中的文件內容，抽取以下欄位，
只輸出一個 JSON 物件，不要輸出任何其他文字、不要用 markdown code block、不要加註解。
看不到或無法判斷的欄位（品項除外）一律填 null；完全沒有品項就輸出空陣列 []。

欄位定義：
{field_descriptions}

輸出範例（僅示意格式，不代表本次答案）：
{{"doc_type": "e_invoice", "seller_name": "全家便利商店", "date": "2024-05-12", "invoice_number": "AB12345678", "seller_tax_id": "22555003", "buyer_tax_id": null, "total_amount": 95, "items": [{{"name": "拿鐵咖啡", "amount": 60}}]}}"""

_OCR_HINT_TEMPLATE = """

以下是 OCR 初步辨識的文字，僅供參考、可能有辨識錯誤，請以圖片本身內容為準：
---
{ocr_text}
---"""

_RETRY_TEMPLATE = """

你上一次的回應有問題：{error}

上一次的原始回應：
{raw_response}

請重新輸出，只要一個合法 JSON 物件，不要任何其他文字。"""


def _field_descriptions() -> str:
    schema = load_schema()
    lines = [f"- {name}: {prop.get('description', '')}" for name, prop in schema["properties"].items()]
    return "\n".join(lines)


def build_prompt(ocr_hint: str | None = None) -> str:
    prompt = _BASE_TEMPLATE.format(field_descriptions=_field_descriptions())
    if ocr_hint:
        prompt += _OCR_HINT_TEMPLATE.format(ocr_text=ocr_hint)
    return prompt


def build_retry_prompt(base_prompt: str, raw_response: str, error: str) -> str:
    return base_prompt + _RETRY_TEMPLATE.format(error=error, raw_response=raw_response)

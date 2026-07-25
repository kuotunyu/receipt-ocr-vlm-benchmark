import pytest

from src.pipeline_a.assemble import (
    assemble,
    extract_header_fields,
    extract_items_by_rules,
)
from src.pipeline_a.llm_assist import OllamaUnavailable
from src.pipeline_a.ocr import OCRBox


def box(text, x1=0, y1=0, x2=10, y2=10):
    return OCRBox(text=text, score=1.0, x1=x1, y1=y1, x2=x2, y2=y2)


def line(*texts_with_x):
    """texts_with_x: 依 x 由左到右排序的 (text, x1) tuples，模擬 layout.group_into_lines 的輸出。"""
    return [box(t, x1=x1, x2=x1 + 10) for t, x1 in texts_with_x]


INVOICE_LINES = [
    [box("全家便利商店")],
    [box("電子發票證明聯")],
    [box("113年05月12日")],
    [box("AB-12345678")],
    [box("賣方統一編號：22555003")],
    line(("品名", 0), ("數量", 200), ("金額", 300)),
    line(("拿鐵咖啡", 0), ("1", 200), ("60", 300)),
    line(("御飯糰-鮪魚", 0), ("1", 200), ("35", 300)),
    line(("總計", 0), ("95", 300)),
]


class TestExtractHeaderFields:
    def test_full_invoice(self):
        fields = extract_header_fields(INVOICE_LINES)
        assert fields["seller_name"] == "全家便利商店"
        assert fields["date"] == "2024-05-12"
        assert fields["invoice_number"] == "AB-12345678"
        assert fields["seller_tax_id"] == "22555003"
        assert fields.get("buyer_tax_id") is None
        assert fields["total_amount"] == 95
        assert fields["doc_type"] == "e_invoice"

    def test_receipt_without_invoice_number_is_doc_type_receipt(self):
        lines = [
            [box("小美冰淇淋")],
            [box("114年01月20日")],
            [box("統編：87654321")],
            line(("合計", 0), ("200", 300)),
        ]
        fields = extract_header_fields(lines)
        assert fields["doc_type"] == "receipt"
        assert fields.get("invoice_number") is None
        assert fields["seller_tax_id"] == "87654321"
        assert fields["total_amount"] == 200

    def test_invoice_number_digits_excluded_from_tax_id_search(self):
        # 發票號碼裡的 8 碼數字不該被誤判成統編
        lines = [[box("AB-12345678")], [box("賣方統一編號：22555003")]]
        fields = extract_header_fields(lines)
        assert fields["seller_tax_id"] == "22555003"

    def test_buyer_tax_id_when_two_present(self):
        lines = [[box("賣方統編：22555003")], [box("買方統編：87654321")]]
        fields = extract_header_fields(lines)
        assert fields["seller_tax_id"] == "22555003"
        assert fields["buyer_tax_id"] == "87654321"


class TestExtractItemsByRules:
    def test_header_and_total_rows_excluded(self):
        items = extract_items_by_rules(INVOICE_LINES)
        assert items == [
            {"name": "拿鐵咖啡", "amount": 60},
            {"name": "御飯糰-鮪魚", "amount": 35},
        ]

    def test_no_header_row_still_extracts_up_to_total(self):
        lines = [
            line(("珍珠奶茶", 0), ("50", 300)),
            line(("總計", 0), ("50", 300)),
        ]
        items = extract_items_by_rules(lines)
        assert items == [{"name": "珍珠奶茶", "amount": 50}]

    def test_row_without_trailing_number_is_skipped(self):
        lines = [
            line(("品名", 0), ("金額", 300)),
            [box("備註：內用")],  # 沒有金額欄，不應被當成品項
            line(("總計", 0), ("0", 300)),
        ]
        assert extract_items_by_rules(lines) == []

    def test_quantity_column_ignored_amount_is_rightmost(self):
        lines = [
            line(("咖啡", 0), ("2", 200), ("120", 300)),
            line(("總計", 0), ("120", 300)),
        ]
        items = extract_items_by_rules(lines)
        assert items == [{"name": "咖啡", "amount": 120}]


class TestAssemble:
    def test_rule_based_items_skip_llm(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise AssertionError("規則抽得到品項時不該呼叫 LLM")

        monkeypatch.setattr("src.pipeline_a.assemble.refine_items", boom)
        record = assemble(INVOICE_LINES, use_llm=True)
        assert record["items"] == [
            {"name": "拿鐵咖啡", "amount": 60},
            {"name": "御飯糰-鮪魚", "amount": 35},
        ]

    def test_llm_unavailable_falls_back_to_empty_items(self, monkeypatch):
        lines = [[box("備註：內用")], line(("總計", 0), ("0", 300))]

        def unavailable(*_args, **_kwargs):
            raise OllamaUnavailable("Ollama 未啟動")

        monkeypatch.setattr("src.pipeline_a.assemble.refine_items", unavailable)
        record = assemble(lines, use_llm=True)
        assert record["items"] == []

    def test_no_llm_mode_never_calls_llm(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise AssertionError("--no-llm 模式不該呼叫 LLM")

        monkeypatch.setattr("src.pipeline_a.assemble.refine_items", boom)
        lines = [[box("備註：內用")], line(("總計", 0), ("0", 300))]
        record = assemble(lines, use_llm=False)
        assert record["items"] == []

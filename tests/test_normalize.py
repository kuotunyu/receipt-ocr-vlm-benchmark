"""normalize.py 單元測試——這些案例即「凍結的正規化規則」的可執行規格。"""

import pytest

from src.common.normalize import (
    is_valid_tax_id,
    normalize_amount,
    normalize_date,
    normalize_invoice_number,
    normalize_record,
    normalize_tax_id,
    normalize_text,
    to_halfwidth,
)


class TestText:
    def test_fullwidth_to_halfwidth(self):
        assert to_halfwidth("１２３ＡＢｃ") == "123ABc"  # to_halfwidth 本身不改大小寫

    def test_fullwidth_space_and_collapse(self):
        assert normalize_text("　統一　　超商　") == "統一 超商"

    def test_uppercase_for_cross_language_fairness(self):
        # SROIE 店名 GT 全大寫、模型輸出常 Title Case——大小寫不該算抽取錯誤
        assert normalize_text("Ojc Marketing Sdn Bhd") == "OJC MARKETING SDN BHD"

    def test_empty_becomes_none(self):
        assert normalize_text("   ") is None
        assert normalize_text(None) is None


class TestDate:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("113/05/12", "2024-05-12"),          # 民國、斜線
            ("民國113年5月12日", "2024-05-12"),    # 民國、年月日
            ("中華民國113年05月12日", "2024-05-12"),
            ("99年1月5日", "2010-01-05"),          # 民國二位數年
            ("2024-05-12", "2024-05-12"),          # 西元 ISO
            ("2024/5/3", "2024-05-03"),            # 西元、單位數月日
            ("2024.05.12", "2024-05-12"),          # 點分隔
            ("20240512", "2024-05-12"),            # 西元 8 碼連寫
            ("1130512", "2024-05-12"),             # 民國 7 碼連寫
            ("２０２４年５月１２日", "2024-05-12"),  # 全形
            ("15/01/2019", "2019-01-15"),          # DD/MM/YYYY（SROIE 英文收據）
            ("05.02.2018", "2018-02-05"),          # DD.MM.YYYY
            ("DATE: 23/11/2017 10:23", "2017-11-23"),  # 內嵌在其他文字中
        ],
    )
    def test_valid_dates(self, raw, expected):
        assert normalize_date(raw) == expected

    def test_roc_slash_still_year_first(self):
        # DD/MM/YYYY 規則只在年份 4 碼時啟用，不影響民國 YYY/MM/DD
        assert normalize_date("113/05/12") == "2024-05-12"

    @pytest.mark.parametrize(
        "raw",
        [
            "2024-13-01",   # 13 月
            "113/02/30",    # 2/30 不存在
            "無日期",
            "12345",        # 位數不對
            "",
            None,
        ],
    )
    def test_invalid_dates(self, raw):
        assert normalize_date(raw) is None


class TestAmount:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("NT$1,234", 1234),
            ("１，２３４元", 1234),
            ("$ 100", 100),
            ("1234.00", 1234),      # 整數化
            ("12.5", 12.5),
            ("新臺幣360元", 360),
            ("RM193.00", 193),      # SROIE 馬來西亞令吉
            ("RM 7.30", 7.3),
            ("-6.42", -6.42),       # 退貨收據的負數總計
            (250, 250),             # 已是數字直接通過
            (99.0, 99),
        ],
    )
    def test_valid_amounts(self, raw, expected):
        assert normalize_amount(raw) == expected

    @pytest.mark.parametrize("raw", ["免費", "一千二", "", None, True])
    def test_invalid_amounts(self, raw):
        assert normalize_amount(raw) is None


class TestTaxId:
    def test_strip_and_keep_8_digits(self):
        assert normalize_tax_id(" 22555003 ") == "22555003"
        assert normalize_tax_id("統編:22555003") == "22555003"
        assert normalize_tax_id("２２５５５００３") == "22555003"

    def test_wrong_length_is_none(self):
        assert normalize_tax_id("1234567") is None
        assert normalize_tax_id("123456789") is None

    def test_checksum_valid(self):
        assert is_valid_tax_id("22555003")  # 統一超商

    def test_checksum_special_case_digit7(self):
        # 第 7 碼為 7：加權和 +1 為 5 的倍數亦有效
        assert is_valid_tax_id("40000070")

    def test_checksum_invalid(self):
        assert not is_valid_tax_id("12345678")
        assert not is_valid_tax_id(None)


class TestInvoiceNumber:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("AB12345678", "AB12345678"),
            ("AB-12345678", "AB12345678"),
            ("ab 12345678", "AB12345678"),
            ("ＡＢ１２３４５６７８", "AB12345678"),
        ],
    )
    def test_valid(self, raw, expected):
        assert normalize_invoice_number(raw) == expected

    @pytest.mark.parametrize("raw", ["A12345678", "ABC1234567", "AB1234567", "", None])
    def test_invalid(self, raw):
        assert normalize_invoice_number(raw) is None


class TestRecord:
    def test_full_record(self):
        raw = {
            "doc_type": "e_invoice",
            "seller_name": "　統一　超商　",
            "date": "113年5月12日",
            "invoice_number": "ab-12345678",
            "seller_tax_id": "統編 22555003",
            "buyer_tax_id": None,
            "total_amount": "NT$1,234",
            "items": [
                {"name": "　拿鐵　咖啡", "amount": "６０元"},
                {"name": None, "amount": 5},          # 無名品項會被剔除
                "not-a-dict",                          # 髒資料防禦
            ],
        }
        assert normalize_record(raw) == {
            "doc_type": "e_invoice",
            "seller_name": "統一 超商",
            "date": "2024-05-12",
            "invoice_number": "AB12345678",
            "seller_tax_id": "22555003",
            "buyer_tax_id": None,
            "total_amount": 1234,
            "items": [{"name": "拿鐵 咖啡", "amount": 60}],
        }

    def test_unknown_doc_type_becomes_none(self):
        assert normalize_record({"doc_type": "invoice?"})["doc_type"] is None

    def test_missing_keys_all_none(self):
        out = normalize_record({})
        assert out["items"] == []
        assert all(
            out[k] is None
            for k in (
                "doc_type", "seller_name", "date", "invoice_number",
                "seller_tax_id", "buyer_tax_id", "total_amount",
            )
        )

from src.eval.metrics import exact_match, fuzzy_match, fuzzy_score, score_record

GT = {
    "doc_type": "e_invoice",
    "seller_name": "全家便利商店",
    "date": "2024-05-12",
    "invoice_number": "AB12345678",
    "seller_tax_id": "22555003",
    "buyer_tax_id": None,
    "total_amount": 95,
    "items": [{"name": "拿鐵咖啡", "amount": 60}, {"name": "御飯糰-鮪魚", "amount": 35}],
}


class TestExactMatch:
    def test_equal_values(self):
        assert exact_match("22555003", "22555003")
    def test_different_values(self):
        assert not exact_match("22555003", "22333001")
    def test_both_none(self):
        assert exact_match(None, None)


class TestFuzzyScore:
    def test_identical_strings_score_1(self):
        assert fuzzy_score("拿鐵咖啡", "拿鐵咖啡") == 1.0

    def test_both_none_scores_1(self):
        assert fuzzy_score(None, None) == 1.0

    def test_one_none_scores_0(self):
        assert fuzzy_score("拿鐵咖啡", None) == 0.0
        assert fuzzy_score(None, "拿鐵咖啡") == 0.0

    def test_single_char_typo_scores_high(self):
        # 御飯糰-鮪魚 vs 御飯糰-鱻魚.（真實 VLM 誤讀案例）：多數字元相同，分數應該偏高
        score = fuzzy_score("御飯糰-鮪魚", "御飯糰-鱻魚.")
        assert 0.6 < score < 1.0

    def test_numeric_field_treated_as_string(self):
        assert fuzzy_score(95, 95) == 1.0
        assert fuzzy_score(95, 96) < 1.0


class TestFuzzyMatch:
    def test_above_threshold_passes(self):
        assert fuzzy_match("全家便利商店", "全家便利商店", threshold=0.8)

    def test_below_threshold_fails(self):
        assert not fuzzy_match("全家便利商店", "完全不一樣的名稱", threshold=0.8)


class TestScoreRecord:
    def test_perfect_prediction(self):
        result = score_record(GT, GT)
        for field in ("doc_type", "seller_name", "date", "invoice_number", "seller_tax_id", "buyer_tax_id", "total_amount"):
            assert result[field]["exact"] is True
            assert result[field]["fuzzy"] is True
        assert result["items"]["f1"] == 1.0

    def test_none_prediction_treated_as_total_failure(self):
        result = score_record(GT, None)
        assert result["doc_type"]["exact"] is False
        assert result["total_amount"]["exact"] is False
        assert result["items"]["recall"] == 0.0

    def test_missing_field_in_pred_dict_treated_as_none(self):
        partial_pred = {"doc_type": "e_invoice"}  # 其餘欄位缺席
        result = score_record(GT, partial_pred)
        assert result["seller_name"]["exact"] is False
        assert result["items"]["n_pred"] == 0

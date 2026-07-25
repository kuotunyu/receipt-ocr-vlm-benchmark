from src.eval.stability import field_consistency, summarize_stability

FIELDS = ("doc_type", "total_amount", "items")


class TestFieldConsistency:
    def test_all_same_scalar_is_fully_consistent(self):
        assert field_consistency([95, 95, 95]) == 1.0

    def test_majority_value_ratio(self):
        assert field_consistency([95, 95, 96]) == 2 / 3

    def test_all_different_is_least_consistent(self):
        assert field_consistency([1, 2, 3]) == 1 / 3

    def test_empty_list(self):
        assert field_consistency([]) == 0.0

    def test_items_list_order_independent(self):
        a = [{"name": "咖啡", "amount": 60}, {"name": "糰子", "amount": 35}]
        b = [{"name": "糰子", "amount": 35}, {"name": "咖啡", "amount": 60}]  # 順序不同
        assert field_consistency([a, b]) == 1.0  # 視為同一個答案

    def test_items_list_content_difference_detected(self):
        a = [{"name": "咖啡", "amount": 60}]
        b = [{"name": "咖啡", "amount": 99}]  # 金額不同，視為不同答案
        assert field_consistency([a, b]) == 0.5

    def test_none_values_are_valid_consistent_answer(self):
        assert field_consistency([None, None, None]) == 1.0


class TestSummarizeStability:
    def test_all_valid_and_consistent(self):
        record = {"doc_type": "e_invoice", "total_amount": 95, "items": []}
        result = summarize_stability([record, dict(record), dict(record)], FIELDS)
        assert result["n_runs"] == 3
        assert result["n_valid"] == 3
        assert result["validity_rate"] == 1.0
        assert result["doc_type_consistency"] == 1.0
        assert result["total_amount_consistency"] == 1.0

    def test_some_runs_invalid_json(self):
        record = {"doc_type": "e_invoice", "total_amount": 95, "items": []}
        result = summarize_stability([record, None, None], FIELDS)
        assert result["n_valid"] == 1
        assert result["validity_rate"] == 1 / 3
        # 只有一次有效結果，一致率視為滿分（沒有分歧可言）
        assert result["doc_type_consistency"] == 1.0

    def test_all_invalid(self):
        result = summarize_stability([None, None, None], FIELDS)
        assert result["n_valid"] == 0
        assert result["validity_rate"] == 0.0
        assert result["doc_type_consistency"] is None

    def test_inconsistent_field_across_runs(self):
        r1 = {"doc_type": "e_invoice", "total_amount": 95, "items": []}
        r2 = {"doc_type": "e_invoice", "total_amount": 96, "items": []}
        r3 = {"doc_type": "receipt", "total_amount": 95, "items": []}
        result = summarize_stability([r1, r2, r3], FIELDS)
        assert result["doc_type_consistency"] == 2 / 3
        assert result["total_amount_consistency"] == 2 / 3

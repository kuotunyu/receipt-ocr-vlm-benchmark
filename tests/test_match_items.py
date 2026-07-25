from src.eval.match_items import align_items, score_items


class TestAlignItems:
    def test_perfect_match(self):
        gt = [{"name": "拿鐵咖啡", "amount": 60}, {"name": "御飯糰-鮪魚", "amount": 35}]
        pred = [{"name": "拿鐵咖啡", "amount": 60}, {"name": "御飯糰-鮪魚", "amount": 35}]
        matches = align_items(gt, pred)
        assert all(m.gt_index is not None and m.pred_index is not None for m in matches)
        assert all(m.amount_exact for m in matches)

    def test_order_independent(self):
        gt = [{"name": "A", "amount": 1}, {"name": "B", "amount": 2}]
        pred = [{"name": "B", "amount": 2}, {"name": "A", "amount": 1}]
        matches = align_items(gt, pred)
        assert len(matches) == 2
        assert all(m.gt_index is not None and m.pred_index is not None for m in matches)

    def test_extra_predicted_item_is_false_positive_only(self):
        gt = [{"name": "拿鐵咖啡", "amount": 60}]
        pred = [{"name": "拿鐵咖啡", "amount": 60}, {"name": "多預測的品項", "amount": 99}]
        matches = align_items(gt, pred)
        fp = [m for m in matches if m.gt_index is None]
        assert len(fp) == 1
        # 原本正確的那筆不該被多出來的那筆牽連
        tp = [m for m in matches if m.gt_index is not None and m.pred_index is not None]
        assert len(tp) == 1 and tp[0].amount_exact

    def test_missing_item_is_false_negative(self):
        gt = [{"name": "拿鐵咖啡", "amount": 60}, {"name": "御飯糰", "amount": 35}]
        pred = [{"name": "拿鐵咖啡", "amount": 60}]
        matches = align_items(gt, pred)
        fn = [m for m in matches if m.pred_index is None]
        assert len(fn) == 1

    def test_low_similarity_not_forced_matched(self):
        gt = [{"name": "拿鐵咖啡", "amount": 60}]
        pred = [{"name": "完全不同的品項名稱", "amount": 60}]
        matches = align_items(gt, pred, name_threshold=0.6)
        assert not any(m.gt_index is not None and m.pred_index is not None for m in matches)

    def test_ocr_typo_still_matches_above_threshold(self):
        gt = [{"name": "御飯糰-鮪魚", "amount": 35}]
        pred = [{"name": "御飯糰-鱻魚.", "amount": 35}]  # 真實 VLM 誤讀案例
        matches = align_items(gt, pred, name_threshold=0.6)
        tp = [m for m in matches if m.gt_index is not None and m.pred_index is not None]
        assert len(tp) == 1


class TestScoreItems:
    def test_both_empty_is_perfect(self):
        result = score_items([], [])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["amount_accuracy"] is None

    def test_amount_mismatch_lowers_amount_accuracy_not_recall(self):
        gt = [{"name": "拿鐵咖啡", "amount": 60}]
        pred = [{"name": "拿鐵咖啡", "amount": 99}]  # 名稱對、金額錯
        result = score_items(gt, pred)
        assert result["recall"] == 1.0  # 名稱有配對到
        assert result["amount_accuracy"] == 0.0

    def test_name_typo_gives_perfect_f1_but_not_perfect_name_accuracy(self):
        # 實測案例：qwen3-vl 把「御飯糰-鮪魚」讀成「御飯糰-鯖魚」，金額對、名稱有配對到
        # （f1=1.0），但名稱本身其實錯了——這正是 name_exact_rate 存在的理由。
        gt = [{"name": "御飯糰-鮪魚", "amount": 35}]
        pred = [{"name": "御飯糰-鯖魚", "amount": 35}]
        result = score_items(gt, pred)
        assert result["f1"] == 1.0
        assert result["amount_accuracy"] == 1.0
        assert result["name_exact_rate"] == 0.0
        assert 0.6 < result["name_avg_similarity"] < 1.0

    def test_name_exact_rate_perfect_when_all_names_exact(self):
        gt = [{"name": "拿鐵咖啡", "amount": 60}, {"name": "御飯糰-鮪魚", "amount": 35}]
        pred = [{"name": "拿鐵咖啡", "amount": 60}, {"name": "御飯糰-鮪魚", "amount": 35}]
        result = score_items(gt, pred)
        assert result["name_exact_rate"] == 1.0
        assert result["name_avg_similarity"] == 1.0

    def test_name_metrics_none_when_no_matches(self):
        result = score_items([{"name": "A", "amount": 1}], [{"name": "完全不同", "amount": 1}])
        assert result["name_exact_rate"] is None
        assert result["name_avg_similarity"] is None

    def test_precision_recall_f1_with_mixed_errors(self):
        gt = [{"name": "A", "amount": 1}, {"name": "B", "amount": 2}, {"name": "C", "amount": 3}]
        pred = [{"name": "A", "amount": 1}, {"name": "D", "amount": 4}]  # 漏 B、C，多預測 D
        result = score_items(gt, pred)
        assert result["n_matched"] == 1
        assert result["precision"] == 0.5  # 1 tp / (1 tp + 1 fp)
        assert round(result["recall"], 4) == round(1 / 3, 4)

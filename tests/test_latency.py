from src.eval.latency import percentile, split_cold_warm, summarize


class TestPercentile:
    def test_p50_of_odd_length(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3

    def test_p0_is_min_p100_is_max(self):
        values = [5, 1, 3, 2, 4]
        assert percentile(values, 0) == 1
        assert percentile(values, 100) == 5

    def test_empty_returns_nan(self):
        import math
        assert math.isnan(percentile([], 50))


class TestSummarize:
    def test_basic_stats(self):
        stats = summarize([10, 20, 30])
        assert stats["n"] == 3
        assert stats["mean"] == 20

    def test_empty(self):
        stats = summarize([])
        assert stats["n"] == 0
        assert stats["p50"] is None


class TestSplitColdWarm:
    def test_first_value_is_cold(self):
        cold, warm = split_cold_warm([83.0, 30.9, 11.3, 10.6, 25.6])
        assert cold == 83.0
        assert warm == [30.9, 11.3, 10.6, 25.6]

    def test_empty_list(self):
        cold, warm = split_cold_warm([])
        assert cold is None
        assert warm == []

    def test_single_value_has_no_warm(self):
        cold, warm = split_cold_warm([42.0])
        assert cold == 42.0
        assert warm == []

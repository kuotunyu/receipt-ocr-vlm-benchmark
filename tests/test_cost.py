from src.eval.cost import api_cost_per_100_docs, local_gpu_cost_per_100_docs

TEST_PRICING = {
    "api": {
        "fake-model": {"input_per_million_usd": 2.0, "output_per_million_usd": 10.0},
    },
    "local_gpu": {"usd_per_hour": 0.36},
}


class TestApiCost:
    def test_known_token_counts(self):
        # 1M input token + 1M output token = $2 + $10 = $12/doc；100 份 = $1200
        cost = api_cost_per_100_docs("fake-model", 1_000_000, 1_000_000, pricing=TEST_PRICING)
        assert cost == 1200.0

    def test_zero_tokens_is_free(self):
        assert api_cost_per_100_docs("fake-model", 0, 0, pricing=TEST_PRICING) == 0.0

    def test_scales_linearly_with_token_count(self):
        base = api_cost_per_100_docs("fake-model", 1000, 500, pricing=TEST_PRICING)
        doubled = api_cost_per_100_docs("fake-model", 2000, 1000, pricing=TEST_PRICING)
        assert doubled == base * 2


class TestLocalGpuCost:
    def test_one_hour_per_doc_matches_hourly_rate_times_100(self):
        cost = local_gpu_cost_per_100_docs(3600, pricing=TEST_PRICING)
        assert cost == 36.0

    def test_zero_seconds_is_free(self):
        assert local_gpu_cost_per_100_docs(0, pricing=TEST_PRICING) == 0.0

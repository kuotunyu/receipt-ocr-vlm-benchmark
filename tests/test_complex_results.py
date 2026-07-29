import json
from pathlib import Path


def test_compact_complex_result_has_all_factors_and_honest_skips():
    result = json.loads(
        Path("results/complex_document/benchmark_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["receipt_benchmark_untouched"] is True
    assert result["dataset"] == {
        "documents": 5,
        "selected_pages": 26,
        "human_hard_cases": 37,
        "routing_gold_pages": 26,
        "questions": 14,
    }
    factors = result["factor_at_a_time"]
    assert [factor["factor"] for factor in factors] == [
        "1_current_parser_fixed",
        "1b_paddleocr_layout_fixed",
        "2_liteparse_fixed",
        "3_liteparse_structure",
        "3b_liteparse_table_reconstruction",
        "3c_hybrid_table_page_router",
        "4_vlm_parser_structure",
        "4a_targeted_vlm_fixed_diagnostic",
        "4b_targeted_vlm_structure",
        "5_caption_and_index",
    ]
    assert result["decision"]["recommendation"] in {"GO", "NO-GO"}
    assert result["decision"]["targeted_table_routing"]["recommendation"] in {
        "GO",
        "NO-GO",
    }
    assert "mrr_delta" in result["decision"]
    if result["parser_results"]["qwen3-vl"]["status"] == "completed":
        assert result["decision"]["vlm_parser"]["recommendation"] in {
            "GO",
            "NO-GO",
        }
    if result["parser_results"]["targeted-vlm"]["status"] == "completed":
        assert result["decision"]["targeted_vlm"]["recommendation"] in {
            "GO",
            "NO-GO",
        }
    if result["parser_results"]["paddleocr-layout"]["status"] == "completed":
        assert result["decision"]["paddleocr_layout"]["recommendation"] in {
            "GO",
            "NO-GO",
        }
    caption = next(
        item
        for item in result["factor_at_a_time"]
        if item["factor"] == "5_caption_and_index"
    )
    if caption["status"] == "completed":
        assert result["decision"]["caption_and_index"]["recommendation"] in {
            "GO",
            "NO-GO",
        }
    assert result["table_routing_audit"]["page_count"] == 26
    assert result["table_bbox_audit"]["hybrid-table-router"]["case_count"] >= 1
    assert result["parser_results"]["hybrid-table-router"]["status"] == "completed"
    assert result["parser_results"]["paddleocr-layout"]["status"] in {
        "completed",
        "skipped",
    }
    assert result["parser_results"]["qwen3-vl"]["status"] in {
        "completed",
        "skipped",
    }
    if result["parser_results"]["qwen3-vl"]["status"] == "skipped":
        assert "not installed" in result["parser_results"]["qwen3-vl"]["reason"]

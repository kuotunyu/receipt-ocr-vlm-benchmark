import json
import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = PROJECT_ROOT / "results" / "official"
SCALAR_FIELDS = (
    "doc_type",
    "seller_name",
    "date",
    "invoice_number",
    "seller_tax_id",
    "buyer_tax_id",
    "total_amount",
)
CONFIGS = {
    "pipeline_a_with_preprocess",
    "pipeline_a_no_preprocess",
    "pipeline_b_qwen3-vl-8b-local",
    "pipeline_b_qwen3-vl-8b-local_with_ocr_hint",
    "pipeline_b_gpt-5.4-nano",
    "pipeline_b_gpt-5.4-nano_with_ocr_hint",
}
PUBLIC_RECEIPT_CONFIGS = {
    "pipeline_a_with_preprocess",
    "pipeline_a_no_preprocess",
    "pipeline_b_qwen3-vl-8b-local",
    "pipeline_b_qwen3-vl-8b-local_with_ocr_hint",
}
ARTIFACTS = {
    "synthetic": OFFICIAL_DIR / "synthetic_45_summary.json",
    "sroie": OFFICIAL_DIR / "sroie_45_summary.json",
    "public_receipts": OFFICIAL_DIR / "public_zh_receipts_5_summary.json",
}
EXPECTED_AVG_EXACT = {
    "synthetic": {
        "pipeline_a_with_preprocess": 0.917460,
        "pipeline_a_no_preprocess": 0.965079,
        "pipeline_b_qwen3-vl-8b-local": 0.971429,
        "pipeline_b_qwen3-vl-8b-local_with_ocr_hint": 0.980952,
        "pipeline_b_gpt-5.4-nano": 0.660317,
        "pipeline_b_gpt-5.4-nano_with_ocr_hint": 0.907937,
    },
    "sroie": {
        "pipeline_a_with_preprocess": 0.390476,
        "pipeline_a_no_preprocess": 0.415873,
        "pipeline_b_qwen3-vl-8b-local": 0.828571,
        "pipeline_b_qwen3-vl-8b-local_with_ocr_hint": 0.803175,
        "pipeline_b_gpt-5.4-nano": 0.857143,
        "pipeline_b_gpt-5.4-nano_with_ocr_hint": 0.857143,
    },
    "public_receipts": {
        "pipeline_a_with_preprocess": 0.485714,
        "pipeline_a_no_preprocess": 0.400000,
        "pipeline_b_qwen3-vl-8b-local": 0.714286,
        "pipeline_b_qwen3-vl-8b-local_with_ocr_hint": 0.914286,
    },
}


def load_artifact(key: str) -> dict:
    return json.loads(ARTIFACTS[key].read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("key", "dataset_id", "n_documents", "items_scored", "configs"),
    (
        ("synthetic", "synthetic_zh_tw_seed42_45", 45, True, CONFIGS),
        ("sroie", "sroie_test_seed42_45", 45, False, CONFIGS),
        (
            "public_receipts",
            "public_zh_tw_wikimedia_5",
            5,
            True,
            PUBLIC_RECEIPT_CONFIGS,
        ),
    ),
)
def test_official_dataset_and_configuration_manifest(
    key,
    dataset_id,
    n_documents,
    items_scored,
    configs,
):
    artifact = load_artifact(key)

    assert artifact["schema_version"] == 1
    assert artifact["artifact"] == "ocr_vlm_aggregate_evaluation"
    assert artifact["dataset"]["id"] == dataset_id
    assert artifact["dataset"]["n_documents"] == n_documents
    assert artifact["dataset"]["items_scored"] is items_scored
    assert set(artifact["evaluation"]["configurations"]) == configs
    assert set(artifact["results"]) == configs
    assert re.fullmatch(r"[0-9a-f]{64}", artifact["evaluation"]["source_summary_sha256"])


@pytest.mark.parametrize("key", ("synthetic", "sroie", "public_receipts"))
def test_headline_metrics_are_derived_from_full_results(key):
    artifact = load_artifact(key)

    for config, result in artifact["results"].items():
        derived_avg = sum(result[f"{field}_exact"] for field in SCALAR_FIELDS) / len(SCALAR_FIELDS)
        headline = artifact["headline_metrics"][config]
        assert headline["avg_exact"] == pytest.approx(derived_avg, abs=0.5e-6)
        assert headline["latency_warm_p50_s"] == pytest.approx(
            result["latency_warm"]["p50"], abs=0.5e-3
        )


@pytest.mark.parametrize("key", ("synthetic", "sroie", "public_receipts"))
def test_headline_avg_exact_matches_published_report(key):
    artifact = load_artifact(key)
    actual = {name: values["avg_exact"] for name, values in artifact["headline_metrics"].items()}
    assert actual == EXPECTED_AVG_EXACT[key]


@pytest.mark.parametrize("key", ("synthetic", "sroie", "public_receipts"))
def test_official_artifacts_exclude_local_paths_credentials_and_per_document_data(key):
    text = ARTIFACTS[key].read_text(encoding="utf-8")
    forbidden_patterns = (
        r"(?i)[a-z]:[\\/]",
        r"(?i)/(?:users|home)/",
        r"(?i)\b(?:api[_-]?key|password|authorization|bearer)\b\s*[:=]",
        r"\bsk-[A-Za-z0-9_-]{12,}\b",
        r"\bAIza[0-9A-Za-z_-]{20,}\b",
    )

    assert all(re.search(pattern, text) is None for pattern in forbidden_patterns)
    assert '"per_image"' not in text
    assert '"record"' not in text
    assert text.endswith("\n")

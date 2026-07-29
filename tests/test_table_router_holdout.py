import hashlib
import json
from pathlib import Path

import pytest

from src.complex_document.router_holdout import (
    HoldoutDefinitionError,
    evaluate_router_holdout,
    load_holdout_ir,
    read_json,
    validate_holdout_definition,
)
from src.complex_document.routing import (
    DEFAULT_TABLE_ROUTE_THRESHOLD,
    TABLE_ROUTER_VERSION,
)

MANIFEST_PATH = Path("data/complex_document/holdout/manifest.json")
GOLD_PATH = Path(
    "data/complex_document/holdout/gold/table_routing_pages.json"
)
RESULT_PATH = Path("results/complex_document/table_router_holdout.json")
RAW_DIR = Path("data/complex_document/holdout/raw")
ARTIFACT_ROOT = Path("artifacts/complex_document/holdout")


def test_external_holdout_is_frozen_balanced_and_disjoint():
    manifest = read_json(MANIFEST_PATH)
    gold = read_json(GOLD_PATH)
    development = read_json("data/complex_document/manifest.json")
    validate_holdout_definition(manifest, gold)

    assert manifest["benchmark_version"] == "0.4.0"
    assert manifest["router_frozen"] == {
        "version": TABLE_ROUTER_VERSION,
        "threshold": DEFAULT_TABLE_ROUTE_THRESHOLD,
        "frozen_from_benchmark_version": "0.3.0",
        "retuning_allowed": False,
    }
    assert len(manifest["documents"]) == 2
    assert len(gold["pages"]) == 12
    assert sum(item["should_route"] for item in gold["pages"]) == 6
    assert {
        item["document_id"] for item in manifest["documents"]
    }.isdisjoint(
        item["document_id"] for item in development["documents"]
    )
    assert all(item["license_url"] for item in manifest["documents"])
    assert "no router predictions" in gold["annotation_method"]


def test_primary_holdout_threshold_cannot_be_retuned():
    with pytest.raises(HoldoutDefinitionError, match="frozen threshold"):
        evaluate_router_holdout({}, [], threshold=0.61)


def test_holdout_downloads_match_manifest_when_present():
    manifest = read_json(MANIFEST_PATH)
    for item in manifest["documents"]:
        path = RAW_DIR / item["filename"]
        if path.exists():
            assert path.stat().st_size == item["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_checked_in_holdout_result_is_complete_and_reproducible_when_local():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["benchmark_version"] == "0.4.0"
    assert result["dataset"] == {
        "documents": 2,
        "selected_pages": 12,
        "positive_pages": 6,
        "negative_pages": 6,
        "development_document_overlap": 0,
    }
    assert result["environment"]["device_policy"].startswith("CPU only")
    assert result["receipt_benchmark_untouched"] is True
    assert (
        result["evaluation"]["protocol"]["primary_threshold"]
        == DEFAULT_TABLE_ROUTE_THRESHOLD
    )
    assert result["evaluation"]["protocol"]["retuning_allowed"] is False
    assert len(result["evaluation"]["sensitivity"]) == 5

    ir_paths = [
        ARTIFACT_ROOT
        / "ir"
        / item["document_id"]
        / "pymupdf"
        / "document.ir.json"
        for item in read_json(MANIFEST_PATH)["documents"]
    ]
    if all(path.exists() for path in ir_paths):
        documents = load_holdout_ir(read_json(MANIFEST_PATH), ARTIFACT_ROOT)
        recomputed = evaluate_router_holdout(
            documents, read_json(GOLD_PATH)["pages"]
        )
        assert recomputed == result["evaluation"]

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.verify_real_receipt_dataset import (
    RealReceiptValidationError,
    validate_dataset,
)


def _write_dataset(root: Path, *, privacy_reviewed=True):
    raw = root / "raw"
    labels = root / "labels"
    raw.mkdir()
    labels.mkdir()
    items = []
    for index in range(5):
        name = f"receipt_{index + 1:03d}"
        Image.new("RGB", (1600, 900), "white").save(
            raw / f"{name}.jpg"
        )
        label = {
            "doc_type": "receipt",
            "seller_name": "測試商店",
            "date": "2026-07-30",
            "invoice_number": None,
            "seller_tax_id": None,
            "buyer_tax_id": None,
            "total_amount": 100,
            "items": [{"name": "測試品項", "amount": 100}],
        }
        (labels / f"{name}.json").write_text(
            json.dumps(label, ensure_ascii=False), encoding="utf-8"
        )
        if index < 2:
            tags = ["handwriting"]
        elif index < 4:
            tags = ["stamp_occlusion"]
        else:
            tags = ["fold"]
        items.append(
            {
                "id": name,
                "filename": f"{name}.jpg",
                "challenge_tags": tags,
                "contains_personal_data": False,
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_version": "1.0.0",
                "privacy_reviewed": privacy_reviewed,
                "local_processing_only": True,
                "items": items,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return raw, labels, manifest


def test_real_receipt_dataset_requires_human_gold_and_hard_cases(tmp_path):
    raw, labels, manifest = _write_dataset(tmp_path)
    result = validate_dataset(raw, labels, manifest)
    assert result["status"] == "verified-local-only"
    assert result["document_count"] == 5
    assert result["challenge_counts"]["handwriting"] == 2
    assert result["challenge_counts"]["stamp_occlusion"] == 2
    assert all(item["label_schema_valid"] for item in result["items"])


def test_real_receipt_dataset_requires_privacy_attestation(tmp_path):
    raw, labels, manifest = _write_dataset(
        tmp_path, privacy_reviewed=False
    )
    with pytest.raises(RealReceiptValidationError, match="privacy_reviewed"):
        validate_dataset(raw, labels, manifest)


def test_real_receipt_dataset_rejects_missing_label(tmp_path):
    raw, labels, manifest = _write_dataset(tmp_path)
    (labels / "receipt_001.json").unlink()
    with pytest.raises(RealReceiptValidationError, match="missing label"):
        validate_dataset(raw, labels, manifest)

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.download_public_receipts import (
    PublicReceiptDownloadError,
    download_dataset,
    validate_public_manifest,
)
from src.common.normalize import normalize_record
from src.common.schema import validate_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1600, 900), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _manifest(image_bytes: bytes) -> dict:
    checksum = hashlib.sha256(image_bytes).hexdigest()
    return {
        "dataset_version": "1.0.0",
        "privacy_reviewed": True,
        "local_processing_only": True,
        "redistribution_policy": "metadata-only",
        "items": [
            {
                "id": f"public_receipt_{index:03d}",
                "filename": f"public_receipt_{index:03d}.jpg",
                "label_file": f"public_receipt_{index:03d}.json",
                "challenge_tags": ["handwriting", "stamp_occlusion"],
                "contains_personal_data": False,
                "privacy_review_status": "human-reviewed",
                "source": {
                    "page_url": (
                        "https://commons.wikimedia.org/wiki/"
                        f"File:receipt_{index}.jpg"
                    ),
                    "download_url": (
                        "https://upload.wikimedia.org/wikipedia/commons/"
                        f"a/ab/receipt_{index}.jpg"
                    ),
                    "author": "Example Author",
                    "license": "CC-BY-SA-4.0",
                    "sha256": checksum,
                    "width": 1600,
                    "height": 900,
                },
            }
            for index in range(1, 6)
        ],
    }


def test_download_public_receipts_verifies_and_reuses(tmp_path: Path):
    image_bytes = _image_bytes()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest(image_bytes)), encoding="utf-8"
    )
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(image_bytes)

    raw_dir = tmp_path / "raw"
    first = download_dataset(
        manifest_path, raw_dir, opener=opener, sleep_fn=lambda _: None
    )
    assert first["document_count"] == 5
    assert {item["status"] for item in first["items"]} == {"downloaded"}
    assert len(calls) == 5

    second = download_dataset(
        manifest_path, raw_dir, opener=opener, sleep_fn=lambda _: None
    )
    assert {item["status"] for item in second["items"]} == {"reused"}
    assert len(calls) == 5


def test_download_public_receipts_rejects_bad_checksum(tmp_path: Path):
    image_bytes = _image_bytes()
    manifest = _manifest(image_bytes)
    manifest["items"][0]["source"]["sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def opener(request, timeout):
        return io.BytesIO(image_bytes)

    with pytest.raises(
        PublicReceiptDownloadError, match="downloaded checksum mismatch"
    ):
        download_dataset(
            manifest_path,
            tmp_path / "raw",
            opener=opener,
            sleep_fn=lambda _: None,
        )
    assert not (tmp_path / "raw" / "public_receipt_001.jpg").exists()


def test_public_receipt_manifest_rejects_non_wikimedia_source():
    manifest = _manifest(_image_bytes())
    manifest["items"][0]["source"]["download_url"] = (
        "https://example.com/receipt.jpg"
    )
    with pytest.raises(
        PublicReceiptDownloadError, match="source.download_url"
    ):
        validate_public_manifest(manifest)


def test_repository_public_receipt_manifest_and_gold_are_valid():
    manifest = json.loads(
        (PROJECT_ROOT / "data/public_receipts_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    items = validate_public_manifest(manifest)
    challenge_counts = {}
    for item in items:
        for tag in set(item["challenge_tags"]):
            challenge_counts[tag] = challenge_counts.get(tag, 0) + 1
        label = json.loads(
            (
                PROJECT_ROOT
                / "data/public_receipt_labels"
                / item["label_file"]
            ).read_text(encoding="utf-8")
        )
        assert validate_record(label) == []
        assert normalize_record(label) == label

    assert challenge_counts["handwriting"] >= 2
    assert challenge_counts["stamp_occlusion"] >= 2

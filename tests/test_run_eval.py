import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_eval


def _label(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "doc_type": "receipt",
                "seller_name": "測試商店",
                "date": None,
                "invoice_number": None,
                "seller_tax_id": None,
                "buyer_tax_id": None,
                "total_amount": 10,
                "items": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


class _Backend:
    name = "fake-local"

    def __init__(self, expected_hint: str | None = "OCR hint"):
        self.expected_hint = expected_hint

    def extract(self, image_bytes: bytes, *, ocr_hint: str | None = None):
        assert image_bytes == b"image"
        assert ocr_hint == self.expected_hint
        return SimpleNamespace(
            record={
                "doc_type": "receipt",
                "seller_name": "測試商店",
                "date": None,
                "invoice_number": None,
                "seller_tax_id": None,
                "buyer_tax_id": None,
                "total_amount": 10,
                "items": [],
            },
            latency_seconds=3.0,
            is_valid_json=True,
            attempts=1,
            input_tokens=10,
            output_tokens=5,
            gpu_seconds=2.0,
        )


def test_vlm_with_ocr_hint_reports_end_to_end_latency(tmp_path, monkeypatch):
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"not decoded because loader is patched")
    label = _label(tmp_path / "receipt.json")

    monkeypatch.setattr(run_eval, "load_image_bytes_for_vlm", lambda _: b"image")
    monkeypatch.setattr(run_eval, "ocr_hint_for", lambda *_: "OCR hint")
    ticks = iter((10.0, 12.5))
    monkeypatch.setattr(run_eval.time, "perf_counter", lambda: next(ticks))

    result = run_eval.run_pipeline_b_config(
        [(image, label)],
        "pipeline_b_fake_with_ocr_hint",
        _Backend(),
        with_ocr_hint=True,
        ocr_lang="chinese_cht",
        score_items_metrics=True,
    )

    row = result["per_image"][0]
    assert row["backend_latency"] == 3.0
    assert row["ocr_hint_latency"] == 2.5
    assert row["latency"] == 5.5
    assert result["summary"]["latency_cold_s"] == 5.5
    assert result["summary"]["latency_warm"] == {
        "n": 0,
        "p50": None,
        "p95": None,
        "mean": None,
    }


def test_vlm_without_hint_has_zero_hint_latency(tmp_path, monkeypatch):
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"not decoded because loader is patched")
    label = _label(tmp_path / "receipt.json")

    monkeypatch.setattr(run_eval, "load_image_bytes_for_vlm", lambda _: b"image")

    result = run_eval.run_pipeline_b_config(
        [(image, label)],
        "pipeline_b_fake",
        _Backend(expected_hint=None),
        with_ocr_hint=False,
        ocr_lang="chinese_cht",
        score_items_metrics=True,
    )

    row = result["per_image"][0]
    assert row["backend_latency"] == 3.0
    assert row["ocr_hint_latency"] == 0.0
    assert row["latency"] == 3.0

from pathlib import Path

import pytest
from fastapi import HTTPException

from annotator import main as annotator


def _use_temp_data_dirs(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    labels = tmp_path / "labels"
    raw.mkdir()
    labels.mkdir()
    monkeypatch.setattr(annotator, "RAW_DIR", raw)
    monkeypatch.setattr(annotator, "LABELS_DIR", labels)
    return raw, labels


@pytest.mark.parametrize(
    "filename",
    ("../outside.jpg", "/tmp/outside.jpg", r"..\outside.jpg", r"C:\outside.jpg"),
)
def test_image_path_rejects_path_traversal(monkeypatch, tmp_path, filename):
    _use_temp_data_dirs(monkeypatch, tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        annotator._image_path(filename)

    assert exc_info.value.status_code == 404


def test_label_requires_an_existing_supported_image(monkeypatch, tmp_path):
    raw, _ = _use_temp_data_dirs(monkeypatch, tmp_path)
    (raw / "receipt.jpg").write_bytes(b"test fixture")

    assert annotator.get_label("receipt.jpg") == annotator.EMPTY_RECORD
    with pytest.raises(HTTPException) as exc_info:
        annotator.get_label("missing.jpg")

    assert exc_info.value.status_code == 404

"""標註工具後端。啟動：

    .venv\\Scripts\\python -m uvicorn annotator.main:app --reload --port 8010

開啟 http://localhost:8010
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import read_json, write_json  # noqa: E402
from src.common.normalize import normalize_record  # noqa: E402
from src.common.schema import validate_record  # noqa: E402

from annotator import ocr_prefill  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
RAW_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

EMPTY_RECORD = {
    "doc_type": "e_invoice",
    "seller_name": None,
    "date": None,
    "invoice_number": None,
    "seller_tax_id": None,
    "buyer_tax_id": None,
    "total_amount": None,
    "items": [],
}

app = FastAPI(title="繁中發票/收據標註工具")


def _list_images() -> list[str]:
    return sorted(
        p.name for p in RAW_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )


def _label_path(filename: str) -> Path:
    return LABELS_DIR / f"{Path(filename).stem}.json"


def _image_path(filename: str) -> Path:
    """只允許 data/raw 目錄內的單一檔名，避免路徑穿越讀取或寫入其他檔案。"""
    if not filename or Path(filename).name != filename:
        raise HTTPException(404, "image not found")
    path = RAW_DIR / filename
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(404, "image not found")
    return path


@app.get("/api/images")
def api_list_images():
    images = _list_images()
    return [
        {"filename": name, "labeled": _label_path(name).exists()} for name in images
    ]


@app.get("/api/progress")
def api_progress():
    images = _list_images()
    labeled = sum(1 for name in images if _label_path(name).exists())
    return {"labeled": labeled, "total": len(images)}


@app.get("/images/{filename}")
def get_image(filename: str):
    return FileResponse(_image_path(filename))


@app.get("/api/label/{filename}")
def get_label(filename: str):
    _image_path(filename)
    path = _label_path(filename)
    if path.exists():
        return read_json(path)
    return EMPTY_RECORD


@app.post("/api/label/{filename}")
def save_label(filename: str, record: dict):
    _image_path(filename)
    normalized = normalize_record(record)
    errors = validate_record(normalized)
    if errors:
        return JSONResponse(status_code=422, content={"ok": False, "errors": errors})
    write_json(_label_path(filename), normalized)
    return {"ok": True, "errors": []}


@app.get("/api/ocr_status")
def ocr_status():
    available, reason = ocr_prefill.is_available()
    return {"available": available, "reason": reason}


@app.post("/api/ocr_prefill/{filename}")
def api_ocr_prefill(filename: str):
    path = _image_path(filename)
    available, reason = ocr_prefill.is_available()
    if not available:
        raise HTTPException(503, f"OCR 引擎尚不可用：{reason}")
    lines = ocr_prefill.run_ocr(path)
    suggestions = ocr_prefill.suggest_fields(lines)
    return {"raw_lines": lines, "suggestions": suggestions}


app.mount("/", StaticFiles(directory=PROJECT_ROOT / "annotator" / "static", html=True), name="static")

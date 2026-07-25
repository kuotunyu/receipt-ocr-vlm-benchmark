"""載入與驗證 schema/invoice_schema.json。"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schema" / "invoice_schema.json"


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_record(record: dict) -> list[str]:
    """回傳錯誤訊息列表；空列表代表通過。"""
    validator = jsonschema.Draft7Validator(load_schema())
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '(root)'}: {err.message}"
        for err in validator.iter_errors(record)
    ]


def is_valid_record(record: dict) -> bool:
    return not validate_record(record)

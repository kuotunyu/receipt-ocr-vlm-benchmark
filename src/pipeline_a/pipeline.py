"""Pipeline A 完整流程：前處理 → OCR → 版面分析 → 欄位組裝 → 正規化。"""

from __future__ import annotations

import numpy as np

from src.common.normalize import normalize_record
from src.pipeline_a.assemble import assemble
from src.pipeline_a.layout import group_into_lines
from src.pipeline_a.ocr import run_ocr
from src.pipeline_a.preprocess import PreprocessConfig, preprocess


def run_pipeline_a(
    image: np.ndarray,
    config: PreprocessConfig | None = None,
    use_llm: bool = True,
    ocr_lang: str = "chinese_cht",
) -> dict:
    processed = preprocess(image, config)
    boxes = run_ocr(processed, lang=ocr_lang)
    lines = group_into_lines(boxes)
    record = assemble(lines, use_llm=use_llm)
    return normalize_record(record)

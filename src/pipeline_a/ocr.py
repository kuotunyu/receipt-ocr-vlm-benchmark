"""PaddleOCR PP-OCRv6 文字偵測與辨識（CPU）。

刻意關掉 PaddleOCR 自帶的 use_doc_orientation_classify / use_textline_orientation：
Pipeline A 的旋轉校正完全交給 preprocess.py 的 deskew。若讓 PaddleOCR 自己偷偷校正
方向，Phase 4「有無前處理」消融實驗會失真——關掉前處理時 OCR 引擎仍會自動修正，
兩組結果就測不出前處理的價值了。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


@dataclass
class OCRBox:
    text: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def x_center(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@lru_cache(maxsize=4)
def _get_engine(lang: str = "chinese_cht"):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )


def run_ocr(image: np.ndarray, lang: str = "chinese_cht") -> list[OCRBox]:
    """image：灰階或 BGR 皆可；PaddleOCR 的前處理管線要求 3 通道輸入，
    灰階（preprocess.py 的輸出）在這裡統一轉回 BGR。
    lang：繁中資料集用預設 chinese_cht，SROIE 英文收據用 "en"（各語言引擎分開快取）。"""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    engine = _get_engine(lang)
    results = engine.predict(image)
    boxes: list[OCRBox] = []
    for res in results or []:
        texts = res.get("rec_texts") or []
        scores = res.get("rec_scores") or []
        rec_boxes = res.get("rec_boxes")
        if rec_boxes is None:
            continue
        for text, score, box in zip(texts, scores, rec_boxes):
            if not text or not text.strip():
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            boxes.append(OCRBox(text=text.strip(), score=float(score), x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes

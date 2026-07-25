"""版面分析：把 OCR 偵測到的離散文字框，依幾何位置還原成「行」的閱讀順序。

PaddleOCR 偵測順序不保證符合文件的實際列結構（尤其偏轉校正後殘留誤差、
或同一列的中英數字與中文字被切成多個框），這裡用 y 座標群聚 + x 座標排序
重建行結構，是 assemble.py 做關鍵字/欄位比對的基礎。
"""

from __future__ import annotations

from src.pipeline_a.ocr import OCRBox


def group_into_lines(boxes: list[OCRBox], y_tolerance_ratio: float = 0.6) -> list[list[OCRBox]]:
    """依 y_center 群聚成行；容忍度用該行目前平均字高的比例動態計算，
    避免固定像素門檻在不同解析度照片上失準。"""
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b.y_center)
    lines: list[list[OCRBox]] = []
    current = [sorted_boxes[0]]

    def ref_y(line: list[OCRBox]) -> float:
        return sum(b.y_center for b in line) / len(line)

    def avg_height(line: list[OCRBox]) -> float:
        return sum(b.height for b in line) / len(line)

    for box in sorted_boxes[1:]:
        tol = max(avg_height(current) * y_tolerance_ratio, 5.0)
        if abs(box.y_center - ref_y(current)) <= tol:
            current.append(box)
        else:
            lines.append(sorted(current, key=lambda b: b.x_center))
            current = [box]
    lines.append(sorted(current, key=lambda b: b.x_center))
    return lines


def line_text(line: list[OCRBox]) -> str:
    """整行文字直接串接（中文無需分隔），供關鍵字/正則比對用。"""
    return "".join(b.text for b in line)

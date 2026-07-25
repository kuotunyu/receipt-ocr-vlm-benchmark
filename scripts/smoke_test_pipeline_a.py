"""Phase 2 smoke test：在 5 張合成測試圖上跑 Pipeline A，確認每個 stage 都通，
並產生前處理前後對比圖 + OCR box overlay，方便肉眼檢查（正式的準確率評估留給 Phase 4）。

用法：
    .venv\\Scripts\\python scripts\\smoke_test_pipeline_a.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — backend 必須先設定

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8")  # Windows 預設主控台編碼常是 cp950，印中文會炸

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.imageio import imread_unicode  # noqa: E402
from src.common.io import read_json  # noqa: E402
from src.common.normalize import normalize_record  # noqa: E402
from src.pipeline_a.assemble import assemble  # noqa: E402
from src.pipeline_a.layout import group_into_lines  # noqa: E402
from src.pipeline_a.ocr import run_ocr  # noqa: E402
from src.pipeline_a.preprocess import PreprocessConfig, preprocess  # noqa: E402

FIXTURES_DIR = PROJECT_ROOT / "data" / "dev_fixtures" / "raw"
LABELS_DIR = PROJECT_ROOT / "data" / "dev_fixtures" / "labels"
OUT_DIR = PROJECT_ROOT / "results" / "phase2_smoke"

FIELDS = (
    "doc_type", "seller_name", "date", "invoice_number",
    "seller_tax_id", "buyer_tax_id", "total_amount", "items",
)


def draw_boxes(image, boxes):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for b in boxes:
        cv2.rectangle(vis, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 0), 2)
    return vis


def save_visualization(name, raw_gray, processed, boxes, out_path):
    overlay = draw_boxes(processed, boxes)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(raw_gray, cmap="gray")
    axes[0].set_title("原始（灰階）")
    axes[1].imshow(processed, cmap="gray")
    axes[1].set_title("前處理後（deskew+denoise+binarize）")
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"OCR box overlay（{len(boxes)} 框）")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_one(image, config: PreprocessConfig) -> dict:
    processed = preprocess(image, config)
    boxes = run_ocr(processed)
    lines = group_into_lines(boxes)
    record = assemble(lines, use_llm=False)  # Ollama 未安裝時走 --no-llm 等效路徑
    return normalize_record(record), processed, boxes


def diff_line(field, expected, with_pre, without_pre):
    mark_with = "OK" if with_pre == expected else "X "
    mark_without = "OK" if without_pre == expected else "X "
    return f"  {field:15s} gt={expected!r:35s} with_pre[{mark_with}]={with_pre!r:35s} no_pre[{mark_without}]={without_pre!r}"


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(FIXTURES_DIR.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"{FIXTURES_DIR} 內沒有測試圖，先跑 scripts/make_sample_image.py")

    for path in image_paths:
        name = path.stem
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        image = imread_unicode(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        record_with, processed_with, boxes_with = run_one(image, PreprocessConfig())
        record_without, processed_without, boxes_without = run_one(
            image, PreprocessConfig(deskew=False, denoise=False, binarize=False)
        )

        save_visualization(
            f"{name} (with preprocessing)", gray, processed_with, boxes_with,
            OUT_DIR / f"{name}_with_preprocess.png",
        )
        save_visualization(
            f"{name} (no preprocessing)", gray, processed_without, boxes_without,
            OUT_DIR / f"{name}_no_preprocess.png",
        )

        label_path = LABELS_DIR / f"{name}.json"
        if label_path.exists():
            gt = normalize_record(read_json(label_path))
            print("欄位比對（gt / 有前處理 / 無前處理）：")
            for field in FIELDS:
                print(diff_line(field, gt[field], record_with[field], record_without[field]))
        else:
            print("（無 ground truth，僅顯示抽取結果）")
            print("有前處理：", record_with)
            print("無前處理：", record_without)

    print(f"\n視覺化結果已存到 {OUT_DIR}")


if __name__ == "__main__":
    main()

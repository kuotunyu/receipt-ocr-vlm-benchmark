"""產生合成測試用發票/收據圖片（非真實個資），供 Pipeline A/B 在真實照片備齊前
先跑 smoke test。輸出到 data/dev_fixtures/{raw,labels}/——刻意跟 data/raw/（真實
45 張測試集）分開，避免合成圖混進正式資料集或被 make_splits.py 誤收。

5 張圖各對應一種 Phase 2 消融實驗關心的挑戰：
    sample_001  清晰基準
    sample_002  旋轉（測試 deskew 的價值）
    sample_003  雜訊+模糊（測試去噪的價值）
    sample_004  褪色（測試二值化的價值）
    sample_005  另一份不同內容的收據（doc_type=receipt，測試品項/欄位多樣性）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import write_json  # noqa: E402

OUT_RAW = PROJECT_ROOT / "data" / "dev_fixtures" / "raw"
OUT_LABELS = PROJECT_ROOT / "data" / "dev_fixtures" / "labels"
FONT_PATH = Path(r"C:\Windows\Fonts\kaiu.ttf")  # 標楷體，繁中


def render_document(lines: list[str], size: tuple[int, int] = (500, 720)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(str(FONT_PATH), 28)
    body_font = ImageFont.truetype(str(FONT_PATH), 20)

    y = 30
    for i, text in enumerate(lines):
        font = title_font if i == 0 else body_font
        draw.text((40, y), text, fill="black", font=font)
        y += 45
    return img


def rotate_on_white_canvas(img: Image.Image, degrees: float) -> Image.Image:
    rotated = img.rotate(degrees, expand=True, fillcolor="white", resample=Image.BICUBIC)
    return rotated


def add_noise_and_blur(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.int16)
    noise = np.random.default_rng(0).normal(0, 18, arr.shape).astype(np.int16)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(noisy)
    return out.filter(ImageFilter.GaussianBlur(radius=1.6))


def fade(img: Image.Image) -> Image.Image:
    """模擬熱感紙褪色：整體提亮、降低對比、偏灰黃色調。"""
    arr = np.array(img).astype(np.float32)
    arr = arr * 0.55 + 255 * 0.45  # 拉近白色，降低對比
    arr[:, :, 2] *= 0.92  # 稍微降藍，偏黃
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


INVOICE_LINES = [
    "全家便利商店",
    "電子發票證明聯",
    "113年05月12日",
    "隨機碼：1234",
    "AB-12345678",
    "賣方統一編號：22555003",
    "—————————————",
    "品名          數量  金額",
    "拿鐵咖啡        1    60",
    "御飯糰-鮪魚      1    35",
    "—————————————",
    "總計            95",
]

INVOICE_LABEL = {
    "doc_type": "e_invoice",
    "seller_name": "全家便利商店",
    "date": "2024-05-12",
    "invoice_number": "AB12345678",
    "seller_tax_id": "22555003",
    "buyer_tax_id": None,
    "total_amount": 95,
    "items": [
        {"name": "拿鐵咖啡", "amount": 60},
        {"name": "御飯糰-鮪魚", "amount": 35},
    ],
}

RECEIPT_LINES = [
    "小美冰淇淋",
    "手寫收據",
    "114年01月20日",
    "統編：87654321",
    "—————————————",
    "品名          數量  金額",
    "芒果冰          1   120",
    "紅豆湯          2    80",
    "—————————————",
    "合計           200",
]

RECEIPT_LABEL = {
    "doc_type": "receipt",
    "seller_name": "小美冰淇淋",
    "date": "2025-01-20",
    "invoice_number": None,
    "seller_tax_id": "87654321",
    "buyer_tax_id": None,
    "total_amount": 200,
    "items": [
        {"name": "芒果冰", "amount": 120},
        {"name": "紅豆湯", "amount": 80},
    ],
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    base = render_document(INVOICE_LINES)
    variants = {
        "sample_001": base,
        "sample_002": rotate_on_white_canvas(base, degrees=12),
        "sample_003": add_noise_and_blur(base),
        "sample_004": fade(base),
        "sample_005": render_document(RECEIPT_LINES),
    }
    labels = {
        "sample_001": INVOICE_LABEL,
        "sample_002": INVOICE_LABEL,
        "sample_003": INVOICE_LABEL,
        "sample_004": INVOICE_LABEL,
        "sample_005": RECEIPT_LABEL,
    }

    for name, img in variants.items():
        out_path = OUT_RAW / f"{name}.jpg"
        img.convert("RGB").save(out_path, quality=95)
        write_json(OUT_LABELS / f"{name}.json", labels[name])
        print(f"已產生：{out_path}")


if __name__ == "__main__":
    main()

"""產生 45 張合成繁中發票/收據資料集（`data/synthetic/`），取代人工拍照+標註。

設計原則：
- **零人工標註**：內容（店名/品項/金額/日期/發票號碼/統編）由固定種子隨機生成，
  ground truth 在生成當下就已知，直接寫出 labels JSON。
- **每張圖只施加一種劣化**（清晰/褪色/皺摺/旋轉/模糊/印章），刻意不做複合劣化——
  這樣 Phase 4 的錯誤分析才能把「哪種劣化造成哪種失敗」歸因乾淨。
- **manifest.json 記錄每張圖的 doc_type 與 challenge 標籤**，評估時可分組報告。
- 字型多樣性：發票在正黑/明體/Noto 之間輪替，手寫收據用標楷體（最接近手寫的內建字型）。
- 統編用財政部檢核規則生成「真的合法」的號碼（is_valid_tax_id），日期混用民國/西元格式。

用法：
    .venv\\Scripts\\python scripts\\make_synthetic_dataset.py [--seed 42]

重跑同一個 seed 會得到完全相同的資料集（影像 + 標註），因此 data/synthetic/ 不進 git，
repo 只保留本腳本。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.common.io import write_json  # noqa: E402
from src.common.normalize import is_valid_tax_id  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "synthetic"

FONTS = {
    "jhenghei": r"C:\Windows\Fonts\msjh.ttc",
    "mingliu": r"C:\Windows\Fonts\mingliu.ttc",
    "noto": r"C:\Windows\Fonts\NotoSansTC-VF.ttf",
    "kaiu": r"C:\Windows\Fonts\kaiu.ttf",  # 標楷體，最接近手寫
}
INVOICE_FONTS = ["jhenghei", "mingliu", "noto"]

STORE_NAMES = [
    "全家便利商店", "統一超商", "萊爾富便利店", "美廉社", "家樂福超市",
    "大潤發量販", "全聯福利中心", "頂好超市", "小北百貨", "寶雅生活館",
    "老王牛肉麵館", "阿珠海產店", "幸福早餐坊", "金好味自助餐", "山田日式食堂",
    "轉角咖啡館", "春水堂茶飲", "五十嵐飲料店", "阿宏雞排", "廟口蚵仔煎",
]

ITEM_POOL = [
    ("拿鐵咖啡", 45, 80), ("美式咖啡", 35, 60), ("珍珠奶茶", 40, 70), ("紅茶拿鐵", 45, 65),
    ("御飯糰-鮪魚", 30, 40), ("御飯糰-肉鬆", 28, 38), ("茶葉蛋", 10, 15), ("關東煮", 15, 45),
    ("波蘿麵包", 25, 40), ("起司蛋糕", 60, 95), ("巧克力餅乾", 30, 55), ("洋芋片", 25, 45),
    ("礦泉水", 12, 25), ("運動飲料", 25, 39), ("鮮乳", 42, 88), ("優酪乳", 35, 60),
    ("牛肉麵", 120, 180), ("滷肉飯", 35, 60), ("燙青菜", 30, 50), ("貢丸湯", 25, 45),
    ("雞排", 70, 95), ("鹽酥雞", 60, 90), ("蚵仔煎", 65, 80), ("臭豆腐", 50, 70),
    ("生魚片拼盤", 180, 320), ("味噌湯", 20, 35), ("炸蝦天婦羅", 90, 150), ("茶碗蒸", 40, 60),
    ("衛生紙", 89, 139), ("洗髮精", 119, 249), ("牙膏", 59, 99), ("電池4入", 99, 159),
    ("原子筆", 15, 35), ("筆記本", 39, 79), ("膠帶", 25, 45), ("迴紋針", 20, 30),
]


# ---------------------------------------------------------------------------
# 內容生成
# ---------------------------------------------------------------------------


def gen_tax_id(rng: random.Random) -> str:
    """生成通過財政部加權檢核的合法統編。"""
    while True:
        candidate = "".join(rng.choices("0123456789", k=8))
        if is_valid_tax_id(candidate):
            return candidate


def gen_invoice_number(rng: random.Random) -> str:
    letters = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
    digits = "".join(rng.choices("0123456789", k=8))
    return letters + digits


def format_date(rng: random.Random, iso: str, doc_type: str) -> str:
    """把 ISO 日期渲染成發票/收據上會出現的格式（民國/西元混用）。"""
    y, m, d = int(iso[:4]), int(iso[5:7]), int(iso[8:10])
    roc = y - 1911
    styles = [
        f"{roc}年{m:02d}月{d:02d}日",
        f"{roc}/{m:02d}/{d:02d}",
        f"{y}-{m:02d}-{d:02d}",
        f"{y}/{m:02d}/{d:02d}",
    ]
    return rng.choice(styles)


def gen_record(rng: random.Random, doc_type: str) -> dict:
    y = rng.randint(2023, 2026)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    n_items = rng.randint(1, 5)
    items = []
    for name, lo, hi in rng.sample(ITEM_POOL, n_items):
        qty = rng.randint(1, 3)
        unit = rng.randint(lo, hi)
        items.append({"name": name, "qty": qty, "amount": unit * qty})

    record = {
        "doc_type": doc_type,
        "seller_name": rng.choice(STORE_NAMES),
        "date": f"{y:04d}-{m:02d}-{d:02d}",
        "invoice_number": gen_invoice_number(rng) if doc_type == "e_invoice" else None,
        "seller_tax_id": gen_tax_id(rng) if (doc_type == "e_invoice" or rng.random() < 0.7) else None,
        "buyer_tax_id": gen_tax_id(rng) if (doc_type == "e_invoice" and rng.random() < 0.2) else None,
        "total_amount": sum(it["amount"] for it in items),
        "items": [{"name": it["name"], "amount": it["amount"]} for it in items],
    }
    return record, items


# ---------------------------------------------------------------------------
# 版面渲染
# ---------------------------------------------------------------------------


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTS[name], size)


def render_document(rng: random.Random, record: dict, items_with_qty: list[dict], font_name: str) -> Image.Image:
    w, h = 520, 760
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(font_name, rng.randint(26, 30))
    body_font = _font(font_name, rng.randint(19, 22))

    is_invoice = record["doc_type"] == "e_invoice"
    date_text = format_date(rng, record["date"], record["doc_type"])
    inv_no = record["invoice_number"]
    inv_no_text = rng.choice([inv_no, f"{inv_no[:2]}-{inv_no[2:]}"]) if inv_no else None

    x0 = rng.randint(35, 55)
    y = rng.randint(24, 40)

    def line(text, font=None, dy=44):
        nonlocal y
        jitter = rng.randint(-2, 2)
        draw.text((x0 + jitter, y), text, fill="black", font=font or body_font)
        y += dy

    line(record["seller_name"], title_font, 50)
    line("電子發票證明聯" if is_invoice else rng.choice(["收據", "免用統一發票收據"]))
    line(date_text)
    if is_invoice:
        line(f"隨機碼：{rng.randint(1000, 9999)}")
        line(inv_no_text)
        line(f"賣方統一編號：{record['seller_tax_id']}")
        if record["buyer_tax_id"]:
            line(f"買方統一編號：{record['buyer_tax_id']}")
    elif record["seller_tax_id"]:
        line(f"統編：{record['seller_tax_id']}")
    line("—" * 13)
    line(f"品名{'':10s}數量  金額")
    for it in items_with_qty:
        name = it["name"]
        pad = max(1, 14 - len(name) * 2)
        line(f"{name}{' ' * pad}{it['qty']}    {it['amount']}")
    line("—" * 13)
    total_label = "總計" if is_invoice else rng.choice(["合計", "總計"])
    total_text = rng.choice([str(record["total_amount"]), f"NT${record['total_amount']}", f"{record['total_amount']}元"])
    line(f"{total_label}{'':10s}{total_text}")

    return img


# ---------------------------------------------------------------------------
# 劣化（每張只施加一種，方便錯誤歸因）
# ---------------------------------------------------------------------------


def deg_clean(rng, img):
    return img


def deg_fade(rng, img):
    arr = np.array(img).astype(np.float32)
    strength = rng.uniform(0.45, 0.6)  # 越低越褪
    arr = arr * strength + 255 * (1 - strength)
    arr[:, :, 2] *= rng.uniform(0.9, 0.95)  # 偏黃
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def deg_blur(rng, img):
    arr = np.array(img).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(12, 20), arr.shape)
    noisy = np.clip(arr + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy).filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.2, 1.9)))


def deg_rotate(rng, img):
    angle = rng.uniform(8, 18) * rng.choice([-1, 1])
    return img.rotate(angle, expand=True, fillcolor="white", resample=Image.BICUBIC)


def deg_wrinkle(rng, img):
    """正弦波位移模擬紙張皺摺的幾何變形 + 沿摺線的陰影。"""
    arr = np.array(img)
    h, w = arr.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    amp = rng.uniform(3, 6)
    wavelength = rng.uniform(80, 160)
    phase = rng.uniform(0, 2 * np.pi)
    map_x = xx + amp * np.sin(2 * np.pi * yy / wavelength + phase)
    map_y = yy + amp * 0.5 * np.cos(2 * np.pi * xx / (wavelength * 1.3) + phase)
    warped = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # 摺線陰影：沿著一條水平帶稍微壓暗
    fold_y = rng.randint(h // 4, 3 * h // 4)
    band = np.exp(-((np.arange(h) - fold_y) ** 2) / (2 * 18.0**2)) * 45
    warped = np.clip(warped.astype(np.float32) - band[:, None, None], 0, 255).astype(np.uint8)
    return Image.fromarray(warped)


def deg_stamp(rng, img):
    """半透明紅色橢圓章覆蓋在品項/金額區域上（模擬店章/發票章）。"""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cx = rng.randint(w // 3, 2 * w // 3)
    cy = rng.randint(h // 3, 2 * h // 3)
    rx, ry = rng.randint(70, 100), rng.randint(50, 70)
    red = (200, 30, 30, 130)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=red, width=6)
    stamp_font = _font("kaiu", 24)
    draw.text((cx - rx + 18, cy - 14), "統一發票專用章", fill=red, font=stamp_font)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


DEGRADATIONS = {
    "clean": deg_clean,
    "fade": deg_fade,
    "blur": deg_blur,
    "rotate": deg_rotate,
    "wrinkle": deg_wrinkle,
    "stamp": deg_stamp,
    "handwriting": deg_clean,  # 「手寫感」由標楷體 + 收據版面承擔，影像本身不再劣化
}

# (challenge, doc_type) × 張數，總計 45；配比對應 docs/DATA_COLLECTION_GUIDE.md 的 diversity matrix
SPEC = (
    [("clean", "e_invoice")] * 9 + [("clean", "receipt")] * 3        # 清晰基準 12
    + [("fade", "e_invoice")] * 6 + [("fade", "receipt")] * 2        # 褪色 8
    + [("wrinkle", "e_invoice")] * 4 + [("wrinkle", "receipt")] * 2  # 皺摺 6
    + [("rotate", "e_invoice")] * 5 + [("rotate", "receipt")] * 1    # 旋轉 6
    + [("blur", "e_invoice")] * 3 + [("blur", "receipt")] * 1        # 模糊 4
    + [("stamp", "e_invoice")] * 3 + [("stamp", "receipt")] * 1      # 印章 4
    + [("handwriting", "receipt")] * 5                               # 手寫（標楷體收據）5
)
assert len(SPEC) == 45, len(SPEC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    raw_dir = OUT_DIR / "raw"
    labels_dir = OUT_DIR / "labels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, (challenge, doc_type) in enumerate(SPEC, start=1):
        name = f"syn_{i:03d}"
        record, items_with_qty = gen_record(rng, doc_type)
        # 手寫感：收據一律標楷體；發票在印刷字型間輪替
        font_name = "kaiu" if doc_type == "receipt" else rng.choice(INVOICE_FONTS)
        img = render_document(rng, record, items_with_qty, font_name)
        img = DEGRADATIONS[challenge](rng, img)

        img.convert("RGB").save(raw_dir / f"{name}.jpg", quality=92)
        write_json(labels_dir / f"{name}.json", record)
        manifest.append({"name": name, "doc_type": doc_type, "challenge": challenge, "font": font_name})

    write_json(OUT_DIR / "manifest.json", manifest)
    print(f"已產生 {len(SPEC)} 張 → {OUT_DIR}")
    counts = {}
    for entry in manifest:
        counts[entry["challenge"]] = counts.get(entry["challenge"], 0) + 1
    print("challenge 配比：", counts)


if __name__ == "__main__":
    main()

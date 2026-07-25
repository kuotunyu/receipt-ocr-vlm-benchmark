"""從 HuggingFace 下載 SROIE 2019（ICDAR 真實英文收據基準）並轉成本專案 schema。

- 來源：rth/sroie-2019-v2 的 test split（347 張，已去重）
- 來源、授權標示與論文引用：見根目錄 THIRD_PARTY_NOTICES.md
- 固定種子抽 45 張 → data/sroie/raw/*.jpg + data/sroie/labels/*.json
- 欄位對應：company→seller_name、date（DD/MM/YYYY→ISO）→date、total→total_amount；
  doc_type 一律 "receipt"；invoice_number/seller_tax_id/buyer_tax_id 為 null
  （馬來西亞收據沒有台灣的發票號碼/統編概念）
- **SROIE 沒有標註品項清單**，labels 的 items 一律空陣列——評估時要用
  `run_eval.py --no-items` 跳過品項指標，否則模型「有抽出品項」反而會被扣分。

用法：
    .venv\\Scripts\\python scripts\\download_sroie.py [--n 45] [--seed 42]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime  # noqa: E402

from src.common.io import write_json  # noqa: E402
from src.common.normalize import normalize_amount, normalize_text  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "sroie"


def parse_sroie_date(raw: str) -> str | None:
    """SROIE GT 日期以 DD/MM/YYYY 為主，但有少數變體；轉不出來就回 None
    （該張會被抽樣流程跳過，確保 45 張 GT 全部乾淨）。"""
    raw = raw.strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d %b %Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def convert(entities: dict) -> dict | None:
    date = parse_sroie_date(entities.get("date") or "")
    total = normalize_amount(entities.get("total"))
    seller = normalize_text(entities.get("company"))
    if not (date and seller and total is not None):
        return None  # GT 本身不完整/轉不乾淨的樣本不進資料集
    return {
        "doc_type": "receipt",
        "seller_name": seller,
        "date": date,
        "invoice_number": None,
        "seller_tax_id": None,
        "buyer_tax_id": None,
        "total_amount": total,
        "items": [],  # SROIE 未標註品項；評估時用 --no-items 跳過
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("rth/sroie-2019-v2", split="test")
    print(f"SROIE test split：{len(ds)} 張")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    raw_dir = OUT_DIR / "raw"
    labels_dir = OUT_DIR / "labels"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0
    manifest = []
    for idx in indices:
        if kept >= args.n:
            break
        row = ds[idx]
        record = convert(row["objects"]["entities"])
        if record is None:
            skipped += 1
            continue
        kept += 1
        name = f"sroie_{kept:03d}"
        row["image"].convert("RGB").save(raw_dir / f"{name}.jpg", quality=92)
        write_json(labels_dir / f"{name}.json", record)
        manifest.append({"name": name, "source_index": idx})

    write_json(OUT_DIR / "manifest.json", manifest)
    print(f"已轉出 {kept} 張（跳過 GT 不完整 {skipped} 張）→ {OUT_DIR}")


if __name__ == "__main__":
    main()

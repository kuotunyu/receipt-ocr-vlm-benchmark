"""從 data/labels/ 已標註的檔案產生 dev/test split（固定種子，可重現）。

用法：
    .venv\\Scripts\\python scripts\\make_splits.py [--dev-size 10] [--seed 42]

輸出 data/splits.json：{"dev": [...], "test": [...]}（存檔名不含副檔名，
對應 data/raw/<name>.jpg 與 data/labels/<name>.json）。

prompt 調參與 fuzzy 閾值選擇只准用 dev；test 定案後不再用於任何決策，
只在 Phase 4 最終跑一次得出報告數字。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import write_json  # noqa: E402

LABELS_DIR = PROJECT_ROOT / "data" / "labels"
SPLITS_PATH = PROJECT_ROOT / "data" / "splits.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    names = sorted(p.stem for p in LABELS_DIR.glob("*.json"))
    if not names:
        raise SystemExit(f"{LABELS_DIR} 內沒有任何標註檔，請先用標註工具標完再執行本腳本。")

    rng = random.Random(args.seed)
    rng.shuffle(names)

    dev_size = min(args.dev_size, len(names))
    dev, test = names[:dev_size], names[dev_size:]

    write_json(SPLITS_PATH, {"dev": sorted(dev), "test": sorted(test)})
    print(f"共 {len(names)} 筆 → dev {len(dev)} / test {len(test)}")
    print(f"已寫入 {SPLITS_PATH}")
    if len(names) < 45:
        print(f"提醒：目前僅 {len(names)} 筆標註，計畫目標為 45 筆；之後補標完可重跑本腳本更新 split。")


if __name__ == "__main__":
    main()

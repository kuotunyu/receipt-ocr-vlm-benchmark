"""把 run_eval 的逐張結果按 manifest.json 的劣化類型（challenge）分組，輸出
「每個配置 × 每種劣化」的準確率細目表——合成資料集「每張只施加一種劣化」的
設計就是為了讓這張表的歸因是乾淨的。

用法：
    .venv\\Scripts\\python scripts\\challenge_breakdown.py results/eval_synthetic_45/raw.json data/synthetic/manifest.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import read_json  # noqa: E402
from src.eval.metrics import SCALAR_FIELDS  # noqa: E402

CHALLENGE_ORDER = ["clean", "fade", "wrinkle", "rotate", "blur", "stamp", "handwriting"]


def avg_exact(image_results: list[dict]) -> float:
    return statistics.mean(
        statistics.mean(r["scores"][f]["exact"] for f in SCALAR_FIELDS) for r in image_results
    )


def avg_items_f1(image_results: list[dict]) -> float:
    return statistics.mean(r["scores"]["items"]["f1"] for r in image_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="run_eval.py 產生的 raw.json")
    parser.add_argument("manifest", type=Path, help="合成資料集的 manifest.json")
    args = parser.parse_args()

    raw = read_json(args.raw)
    manifest = {m["name"]: m["challenge"] for m in read_json(args.manifest)}
    challenges = [c for c in CHALLENGE_ORDER if c in set(manifest.values())]

    for metric_name, metric in (("表頭欄位 exact", avg_exact), ("items F1", avg_items_f1)):
        print(f"\n## {metric_name}（配置 × 劣化類型）\n")
        print("| config | " + " | ".join(challenges) + " |")
        print("|---|" + "---|" * len(challenges))
        for config, per_image in raw.items():
            groups: dict[str, list] = {c: [] for c in challenges}
            for r in per_image:
                groups[manifest[r["name"]]].append(r)
            cells = " | ".join(
                f"{metric(g):.2f}" if g else "—" for g in (groups[c] for c in challenges)
            )
            print(f"| {config.replace('pipeline_', '')} | {cells} |")


if __name__ == "__main__":
    main()

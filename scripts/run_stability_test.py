"""穩定性測試：同一張圖、同一個 VLM backend 連續呼叫 N 次（預設 3 次），
量測 JSON 有效率與「每個欄位是否每次都給同一個答案」。

跟 run_eval.py 的差別：run_eval.py 比的是「準不準」（vs ground truth），
這裡比的是「穩不穩」（同一輸入，模型自己前後一不一致）——兩者都是本專案
「結構化輸出穩定性」要回答的問題，但角度不同。

用法：
    .venv\\Scripts\\python scripts\\run_stability_test.py
    .venv\\Scripts\\python scripts\\run_stability_test.py --repeats 3 --images-dir data/raw
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.common.io import write_json  # noqa: E402
from src.eval.metrics import SCALAR_FIELDS  # noqa: E402
from src.eval.stability import summarize_stability  # noqa: E402
from src.pipeline_b.backend_gemini import GeminiVLMBackend  # noqa: E402
from src.pipeline_b.backend_ollama import OllamaVLMBackend  # noqa: E402
from src.pipeline_b.backend_openai import OpenAIVLMBackend  # noqa: E402

DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "dev_fixtures" / "raw"
ALL_FIELDS = SCALAR_FIELDS + ("items",)


def build_available_backends() -> list:
    backends = []
    try:
        import ollama
        ollama.list()
        backends.append(OllamaVLMBackend())
    except Exception as exc:  # noqa: BLE001
        print(f"[跳過] {OllamaVLMBackend.name}：{exc}")
    try:
        backends.append(GeminiVLMBackend())
    except Exception as exc:  # noqa: BLE001
        print(f"[跳過] {GeminiVLMBackend.name}：{exc}")
    try:
        backends.append(OpenAIVLMBackend())
    except Exception as exc:  # noqa: BLE001
        print(f"[跳過] {OpenAIVLMBackend.name}：{exc}")
    return backends


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    image_paths = sorted(args.images_dir.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"{args.images_dir} 內沒有圖片")

    backends = build_available_backends()
    if not backends:
        raise SystemExit("沒有任何可用的 VLM backend")
    print(f"可用 backend：{[b.name for b in backends]}，每張圖跑 {args.repeats} 次\n")

    all_results = {}
    for backend in backends:
        print(f"== {backend.name} ==")
        per_image = {}
        for img_path in image_paths:
            image_bytes = img_path.read_bytes()
            records = []
            for i in range(args.repeats):
                try:
                    result = backend.extract(image_bytes)
                    records.append(result.record)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {img_path.stem} 第 {i + 1} 次呼叫失敗：{exc}")
                    records.append(None)
            summary = summarize_stability(records, ALL_FIELDS)
            per_image[img_path.stem] = summary
            print(f"  {img_path.stem}: validity_rate={summary['validity_rate']:.2f} "
                  f"({summary['n_valid']}/{summary['n_runs']})")
            inconsistent = [f for f in ALL_FIELDS if (summary.get(f"{f}_consistency") or 1.0) < 1.0]
            if inconsistent:
                print(f"    不一致的欄位：{inconsistent}")

        validity_rates = [s["validity_rate"] for s in per_image.values()]
        print(f"  --> {backend.name} 平均 validity_rate：{statistics.mean(validity_rates):.2f}\n")
        all_results[backend.name] = per_image

    out_path = args.out or (PROJECT_ROOT / "results" / "stability_test.json")
    write_json(out_path, all_results)
    print(f"結果已存到 {out_path}")


if __name__ == "__main__":
    main()

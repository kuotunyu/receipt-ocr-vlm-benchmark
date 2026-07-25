"""Phase 3 smoke test：在 5 張合成測試圖上跑 Pipeline B 的三個 backend，
確認 JSON 穩定性處理（parse → schema 驗證 → 重試）在真實模型上真的有用，
並印出跟 ground truth 的欄位比對（正式準確率/延遲/成本評估留給 Phase 4）。

用法：
    .venv\\Scripts\\python scripts\\smoke_test_pipeline_b.py

各 backend 缺 API key／本機沒裝 Ollama 時會被優雅跳過並印出原因，不會讓整個腳本掛掉。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")  # Windows 預設主控台編碼常是 cp950，印中文會炸

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.common.io import read_json  # noqa: E402
from src.common.normalize import normalize_record  # noqa: E402
from src.pipeline_b.backend_openai import OpenAIVLMBackend  # noqa: E402
from src.pipeline_b.backend_gemini import GeminiVLMBackend  # noqa: E402
from src.pipeline_b.backend_ollama import OllamaVLMBackend  # noqa: E402
from src.pipeline_b.vlm_base import VLMBackend  # noqa: E402

FIXTURES_DIR = PROJECT_ROOT / "data" / "dev_fixtures" / "raw"
LABELS_DIR = PROJECT_ROOT / "data" / "dev_fixtures" / "labels"

FIELDS = (
    "doc_type", "seller_name", "date", "invoice_number",
    "seller_tax_id", "buyer_tax_id", "total_amount", "items",
)


def build_available_backends() -> list[VLMBackend]:
    candidates = []

    try:
        backend = OllamaVLMBackend()
        import ollama

        ollama.list()  # 服務沒啟動/沒裝的話這裡就會丟例外
        candidates.append(backend)
    except Exception as exc:  # noqa: BLE001
        print(f"[跳過] {OllamaVLMBackend.name}：{exc}")

    try:
        candidates.append(GeminiVLMBackend())
    except Exception as exc:  # noqa: BLE001
        print(f"[跳過] {GeminiVLMBackend.name}：{exc}")

    try:
        candidates.append(OpenAIVLMBackend())
    except Exception as exc:  # noqa: BLE001
        print(f"[跳過] {OpenAIVLMBackend.name}：{exc}")

    return candidates


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    backends = build_available_backends()
    if not backends:
        print("\n沒有任何可用的 VLM backend——請至少設定 .env 的 GOOGLE_API_KEY 或 OPENAI_API_KEY，")
        print("或安裝 Ollama（https://ollama.com）並執行 `ollama pull qwen3-vl:8b`。")
        return

    print(f"可用 backend：{[b.name for b in backends]}\n")

    image_paths = sorted(FIXTURES_DIR.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"{FIXTURES_DIR} 內沒有測試圖，先跑 scripts/make_sample_image.py")

    for path in image_paths:
        name = path.stem
        image_bytes = path.read_bytes()
        gt_path = LABELS_DIR / f"{name}.json"
        gt = normalize_record(read_json(gt_path)) if gt_path.exists() else None

        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        for backend in backends:
            t0 = time.perf_counter()
            try:
                result = backend.extract(image_bytes)
            except Exception as exc:  # noqa: BLE001 — 呼叫本身失敗（額度、網路等），跳過這張
                print(f"[{backend.name}] 呼叫失敗：{exc}")
                continue
            elapsed = time.perf_counter() - t0

            print(f"[{backend.name}] valid_json={result.is_valid_json} "
                  f"attempts={result.attempts} latency={elapsed:.2f}s")
            if not result.is_valid_json:
                print(f"    錯誤：{result.error}")
                print(f"    原始回應（前 200 字）：{result.raw_response[:200]!r}")
                continue
            if gt is not None:
                for field in FIELDS:
                    mark = "OK" if result.record[field] == gt[field] else "X "
                    print(f"    {field:15s} [{mark}] gt={gt[field]!r:35} got={result.record[field]!r}")
            else:
                print(f"    {result.record}")


if __name__ == "__main__":
    main()

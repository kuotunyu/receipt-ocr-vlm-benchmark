"""單張圖片快速跑一次 Pipeline A 或 B，印出 JSON——示範/除錯用的最小工具，
不是正式評估（正式評估跑整個實驗矩陣、算指標，見 scripts/run_eval.py）。

用法：
    .venv\\Scripts\\python scripts\\run_pipeline.py --pipeline a --image data/dev_fixtures/raw/sample_001.jpg
    .venv\\Scripts\\python scripts\\run_pipeline.py --pipeline a --image <path> --no-preprocess
    .venv\\Scripts\\python scripts\\run_pipeline.py --pipeline b --image <path> --backend qwen3-vl
    .venv\\Scripts\\python scripts\\run_pipeline.py --pipeline b --image <path> --backend gemini --ocr-hint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 預設主控台編碼常是 cp950，印中文會炸

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.common.imageio import imread_unicode  # noqa: E402
from src.pipeline_a.pipeline import run_pipeline_a  # noqa: E402
from src.pipeline_a.preprocess import PreprocessConfig  # noqa: E402


def get_backend(name: str):
    if name == "qwen3-vl":
        from src.pipeline_b.backend_ollama import OllamaVLMBackend
        return OllamaVLMBackend()
    if name == "gemini":
        from src.pipeline_b.backend_gemini import GeminiVLMBackend
        return GeminiVLMBackend()
    if name == "openai":
        from src.pipeline_b.backend_openai import OpenAIVLMBackend
        return OpenAIVLMBackend()
    raise ValueError(f"未知 backend：{name}")


def build_ocr_hint(image_path: Path) -> str:
    from src.pipeline_a.layout import group_into_lines, line_text
    from src.pipeline_a.ocr import run_ocr

    image = imread_unicode(image_path)
    boxes = run_ocr(preprocess_for_hint(image))
    return "\n".join(line_text(l) for l in group_into_lines(boxes))


def preprocess_for_hint(image):
    from src.pipeline_a.preprocess import preprocess

    return preprocess(image, PreprocessConfig())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["a", "b"], required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--backend", choices=["qwen3-vl", "gemini", "openai"], default="qwen3-vl",
                         help="僅 --pipeline b 需要")
    parser.add_argument("--no-preprocess", action="store_true", help="僅 --pipeline a：關閉前處理")
    parser.add_argument("--no-llm", action="store_true", help="僅 --pipeline a：關閉品項補漏 LLM")
    parser.add_argument("--ocr-hint", action="store_true", help="僅 --pipeline b：附上 Pipeline A 的 OCR 文字當輔助")
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"找不到圖片：{args.image}")

    if args.pipeline == "a":
        image = imread_unicode(args.image)
        config = PreprocessConfig(
            deskew=not args.no_preprocess,
            denoise=not args.no_preprocess,
            binarize=not args.no_preprocess,
        )
        record = run_pipeline_a(image, config=config, use_llm=not args.no_llm)
    else:
        backend = get_backend(args.backend)
        hint = build_ocr_hint(args.image) if args.ocr_hint else None
        result = backend.extract(args.image.read_bytes(), ocr_hint=hint)
        if not result.is_valid_json:
            print(f"抽取失敗（重試 {result.attempts} 次後）：{result.error}", file=sys.stderr)
            sys.exit(1)
        record = result.record

    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

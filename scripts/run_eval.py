"""評估框架主入口：跑完整實驗矩陣（Pipeline A 有/無前處理；Pipeline B 各 backend，
本地 backend 額外跑有/無 OCR 輔助），存 raw.json + summary.json 到 results/eval_<timestamp>/。

預設指向 data/dev_fixtures/（合成圖，用來驗證框架本身能跑通）；45 張真實測試集標註完成後，
改用 --images-dir data/raw --labels-dir data/labels 重跑一次即為正式評估數字。

用法：
    .venv\\Scripts\\python scripts\\run_eval.py
    .venv\\Scripts\\python scripts\\run_eval.py --images-dir data/raw --labels-dir data/labels
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 預設主控台編碼常是 cp950，印中文會炸

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.common.imageio import imread_unicode  # noqa: E402
from src.common.io import read_json, write_json  # noqa: E402
from src.common.normalize import normalize_record  # noqa: E402
from src.eval import cost, latency  # noqa: E402
from src.eval.metrics import SCALAR_FIELDS, score_record  # noqa: E402
from src.pipeline_a.layout import group_into_lines, line_text  # noqa: E402
from src.pipeline_a.ocr import run_ocr  # noqa: E402
from src.pipeline_a.pipeline import run_pipeline_a  # noqa: E402
from src.pipeline_a.preprocess import PreprocessConfig, preprocess  # noqa: E402
from src.pipeline_b.backend_openai import OpenAIVLMBackend  # noqa: E402
from src.pipeline_b.backend_gemini import GeminiVLMBackend  # noqa: E402
from src.pipeline_b.backend_ollama import OllamaVLMBackend  # noqa: E402

DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "dev_fixtures" / "raw"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "data" / "dev_fixtures" / "labels"


def load_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{img_path.stem}.json"
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def ocr_hint_for(img_path: Path, ocr_lang: str = "chinese_cht") -> str:
    image = imread_unicode(img_path)
    boxes = run_ocr(preprocess(image, PreprocessConfig()), lang=ocr_lang)
    lines = group_into_lines(boxes)
    return "\n".join(line_text(l) for l in lines)


def aggregate(config_name: str, per_image: list[dict], score_items_metrics: bool = True) -> dict:
    n = len(per_image)
    agg: dict = {"config": config_name, "n": n}

    for field in SCALAR_FIELDS:
        agg[f"{field}_exact"] = statistics.mean(r["scores"][field]["exact"] for r in per_image)
        agg[f"{field}_fuzzy"] = statistics.mean(r["scores"][field]["fuzzy"] for r in per_image)

    if score_items_metrics:
        agg["items_precision"] = statistics.mean(r["scores"]["items"]["precision"] for r in per_image)
        agg["items_recall"] = statistics.mean(r["scores"]["items"]["recall"] for r in per_image)
        agg["items_f1"] = statistics.mean(r["scores"]["items"]["f1"] for r in per_image)
        name_exact_rates = [r["scores"]["items"]["name_exact_rate"] for r in per_image
                             if r["scores"]["items"]["name_exact_rate"] is not None]
        agg["items_name_exact_rate"] = statistics.mean(name_exact_rates) if name_exact_rates else None
    else:
        # 資料集沒有品項標註（如 SROIE）——不計品項指標，避免「有抽出品項」反被扣分
        agg["items_precision"] = agg["items_recall"] = agg["items_f1"] = None
        agg["items_name_exact_rate"] = None

    cold, warm = latency.split_cold_warm([r["latency"] for r in per_image])
    agg["latency_cold_s"] = cold
    agg["latency_warm"] = latency.summarize(warm)

    agg["est_cost_per_100_docs_usd"] = None  # 預設值；Pipeline A 跑在本機 CPU，暫不建模成本
    if "is_valid_json" in per_image[0]:
        agg["json_validity_rate"] = statistics.mean(r["is_valid_json"] for r in per_image)
        input_tokens = [r["input_tokens"] for r in per_image if r["input_tokens"] is not None]
        output_tokens = [r["output_tokens"] for r in per_image if r["output_tokens"] is not None]
        gpu_seconds = [r["gpu_seconds"] for r in per_image if r["gpu_seconds"] is not None]
        try:
            if input_tokens and output_tokens:
                agg["est_cost_per_100_docs_usd"] = cost.api_cost_per_100_docs(
                    per_image[0]["backend_name"], statistics.mean(input_tokens), statistics.mean(output_tokens)
                )
            elif gpu_seconds:
                agg["est_cost_per_100_docs_usd"] = cost.local_gpu_cost_per_100_docs(statistics.mean(gpu_seconds))
        except KeyError:
            pass  # pricing.yaml 沒有這個 backend 的價格，維持預設 None

    return agg


def run_pipeline_a_config(pairs, config_name: str, preprocess_config: PreprocessConfig,
                          ocr_lang: str, score_items_metrics: bool) -> dict:
    per_image = []
    for img_path, label_path in pairs:
        gt = normalize_record(read_json(label_path))
        image = imread_unicode(img_path)
        t0 = time.perf_counter()
        record = run_pipeline_a(image, config=preprocess_config, use_llm=True, ocr_lang=ocr_lang)
        elapsed = time.perf_counter() - t0
        per_image.append({"name": img_path.stem, "record": record, "latency": elapsed,
                           "scores": score_record(gt, record)})
    return {"per_image": per_image,
            "summary": aggregate(config_name, per_image, score_items_metrics)}


MAX_IMAGE_SIDE = 1600  # Pipeline B 統一的影像長邊上限（所有 backend 一視同仁，保持組內可比較性）


def load_image_bytes_for_vlm(img_path: Path, max_side: int = MAX_IMAGE_SIDE) -> bytes:
    """VLM 輸入前把過大的影像等比縮到長邊 ≤ max_side。

    起因：SROIE 真實掃描圖有些非常大，實測把本地 Ollama 服務直接撐爆
    （模型執行錯誤 → 服務崩潰 → 之後 34 張全部連不上）。縮圖也是生產環境
    上傳前的標準做法；合成資料集（520×760）低於上限，完全不受影響。"""
    from io import BytesIO

    from PIL import Image

    with Image.open(img_path) as img:
        if max(img.size) <= max_side:
            return img_path.read_bytes()
        ratio = max_side / max(img.size)
        new_size = (round(img.width * ratio), round(img.height * ratio))
        buf = BytesIO()
        img.convert("RGB").resize(new_size, Image.LANCZOS).save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def run_pipeline_b_config(pairs, config_name: str, backend, with_ocr_hint: bool,
                          ocr_lang: str, score_items_metrics: bool) -> dict:
    per_image = []
    for img_path, label_path in pairs:
        gt = normalize_record(read_json(label_path))
        image_bytes = load_image_bytes_for_vlm(img_path)
        hint = ocr_hint_for(img_path, ocr_lang) if with_ocr_hint else None
        try:
            result = backend.extract(image_bytes, ocr_hint=hint)
        except Exception as exc:  # noqa: BLE001 — 配額/連線層級的失敗記為該張抽取失敗，整批繼續
            print(f"  [{backend.name}] {img_path.stem} 呼叫失敗：{str(exc)[:120]}")
            per_image.append({
                "name": img_path.stem, "record": None, "latency": 0.0,
                "is_valid_json": False, "attempts": 0,
                "input_tokens": None, "output_tokens": None,
                "gpu_seconds": None, "backend_name": backend.name,
                "call_error": str(exc)[:300],
                "scores": score_record(gt, None),
            })
            continue
        per_image.append({
            "name": img_path.stem, "record": result.record, "latency": result.latency_seconds,
            "is_valid_json": result.is_valid_json, "attempts": result.attempts,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "gpu_seconds": result.gpu_seconds, "backend_name": backend.name,
            "scores": score_record(gt, result.record),
        })
    return {"per_image": per_image,
            "summary": aggregate(config_name, per_image, score_items_metrics)}


def build_available_b_backends(wanted: set[str]) -> list:
    backends = []
    if "ollama" in wanted:
        try:
            import ollama
            ollama.list()
            backends.append(OllamaVLMBackend())
        except Exception as exc:  # noqa: BLE001
            print(f"[跳過] {OllamaVLMBackend.name}：{exc}")
    if "gemini" in wanted:
        try:
            backends.append(GeminiVLMBackend())
        except Exception as exc:  # noqa: BLE001
            print(f"[跳過] {GeminiVLMBackend.name}：{exc}")
    if "openai" in wanted:
        try:
            backends.append(OpenAIVLMBackend())
        except Exception as exc:  # noqa: BLE001
            print(f"[跳過] {OpenAIVLMBackend.name}：{exc}")
    return backends


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ocr-lang", default="chinese_cht",
                        help="Pipeline A 與 OCR 輔助的 PaddleOCR 語言（SROIE 英文收據用 en）")
    parser.add_argument("--no-items", action="store_true",
                        help="資料集沒有品項標註時（如 SROIE）跳過品項指標")
    parser.add_argument("--backends", default="ollama,gemini,openai",
                        help="要跑的 Pipeline B backend（逗號分隔；例如省 Gemini 配額用 ollama,openai）")
    parser.add_argument("--only", default=None,
                        choices=["a_pre", "a_nopre", "ollama", "ollama_hint",
                                 "gemini", "gemini_hint", "openai", "openai_hint"],
                        help="只跑單一配置並把結果合併進 out 目錄的既有檔案——供逐配置獨立行程"
                             "執行（記憶體隔離 + 斷點續跑），驅動腳本見 scripts/run_both_evals.cmd")
    args = parser.parse_args()

    pairs = load_pairs(args.images_dir, args.labels_dir)
    if not pairs:
        raise SystemExit(f"{args.images_dir} / {args.labels_dir} 沒有可配對的圖片+標註")
    print(f"共 {len(pairs)} 筆（{args.images_dir}）", flush=True)
    score_items_metrics = not args.no_items

    out_dir = args.out or (PROJECT_ROOT / "results" / f"eval_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_b_config(backend_key: str, with_hint: bool):
        def run():
            backends = build_available_b_backends({backend_key})
            if not backends:
                return None
            backend = backends[0]
            suffix = "_with_ocr_hint" if with_hint else ""
            return (f"pipeline_b_{backend.name}{suffix}",
                    lambda: run_pipeline_b_config(
                        pairs, f"pipeline_b_{backend.name}{suffix}", backend,
                        with_ocr_hint=with_hint, ocr_lang=args.ocr_lang,
                        score_items_metrics=score_items_metrics))
        return run

    config_registry = {
        "a_pre": lambda: ("pipeline_a_with_preprocess",
                          lambda: run_pipeline_a_config(pairs, "pipeline_a_with_preprocess",
                                                        PreprocessConfig(), args.ocr_lang, score_items_metrics)),
        "a_nopre": lambda: ("pipeline_a_no_preprocess",
                            lambda: run_pipeline_a_config(
                                pairs, "pipeline_a_no_preprocess",
                                PreprocessConfig(deskew=False, denoise=False, binarize=False),
                                args.ocr_lang, score_items_metrics)),
        "ollama": make_b_config("ollama", False),
        "ollama_hint": make_b_config("ollama", True),
        "gemini": make_b_config("gemini", False),
        "gemini_hint": make_b_config("gemini", True),
        "openai": make_b_config("openai", False),
        "openai_hint": make_b_config("openai", True),
    }

    wanted_backends = {b.strip() for b in args.backends.split(",") if b.strip()}
    if args.only:
        selected = [args.only]
    else:
        selected = ["a_pre", "a_nopre"]
        for key in ("ollama", "gemini", "openai"):
            if key in wanted_backends:
                selected += [key, f"{key}_hint"]

    raw_path, summary_path = out_dir / "raw.json", out_dir / "summary.json"
    all_raw = read_json(raw_path) if raw_path.exists() else {}
    all_summary = read_json(summary_path) if summary_path.exists() else {}

    for key in selected:
        prepared = config_registry[key]()
        if prepared is None:
            continue  # backend 不可用，build_available_b_backends 已印原因
        config_name, runner = prepared
        print(f"\n== {config_name} ==", flush=True)
        result = runner()
        # 每個配置跑完立刻合併落盤：行程中途被砍時已完成的配置不會丟
        all_raw[config_name] = result["per_image"]
        all_summary[config_name] = result["summary"]
        write_json(raw_path, all_raw)
        write_json(summary_path, all_summary)
        print(f"  已寫入 {config_name}（累計 {len(all_summary)} 個配置）", flush=True)

    print(f"\n結果已存到 {out_dir}", flush=True)
    print("執行 scripts/make_report.py 產生對比表：")
    print(f'  .venv\\Scripts\\python scripts\\make_report.py "{out_dir / "summary.json"}"')


if __name__ == "__main__":
    main()

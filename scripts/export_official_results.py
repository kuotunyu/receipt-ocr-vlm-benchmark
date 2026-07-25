"""將本機正式評估 summary 匯出成可公開、去識別化的固定格式。

這支腳本只讀 JSON 並做欄位白名單、結構驗證與 deterministic serialization；
不載入模型、不讀取 .env，也不發出網路請求。

用法：
    .venv\\Scripts\\python scripts\\export_official_results.py
    .venv\\Scripts\\python scripts\\export_official_results.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCALAR_FIELDS = (
    "doc_type",
    "seller_name",
    "date",
    "invoice_number",
    "seller_tax_id",
    "buyer_tax_id",
    "total_amount",
)

CONFIG_ORDER = (
    "pipeline_a_with_preprocess",
    "pipeline_a_no_preprocess",
    "pipeline_b_qwen3-vl-8b-local",
    "pipeline_b_qwen3-vl-8b-local_with_ocr_hint",
    "pipeline_b_gpt-5.4-nano",
    "pipeline_b_gpt-5.4-nano_with_ocr_hint",
)

ITEM_METRICS = (
    "items_precision",
    "items_recall",
    "items_f1",
    "items_name_exact_rate",
)

DATASETS = {
    "synthetic": {
        "input": PROJECT_ROOT / "results" / "eval_synthetic_45" / "summary.json",
        "output": "synthetic_45_summary.json",
        "metadata": {
            "id": "synthetic_zh_tw_seed42_45",
            "display_name": "合成繁體中文發票／收據",
            "n_documents": 45,
            "items_scored": True,
            "provenance": {
                "kind": "generated",
                "generator": "scripts/make_synthetic_dataset.py",
                "seed": 42,
            },
        },
    },
    "sroie": {
        "input": PROJECT_ROOT / "results" / "eval_sroie_45" / "summary.json",
        "output": "sroie_45_summary.json",
        "metadata": {
            "id": "sroie_test_seed42_45",
            "display_name": "SROIE 真實英文收據",
            "n_documents": 45,
            "items_scored": False,
            "provenance": {
                "kind": "sampled_public_benchmark",
                "dataset": "rth/sroie-2019-v2",
                "split": "test",
                "seed": 42,
            },
        },
    },
}


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"找不到來源 summary：{_display_path(path)}") from exc
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"來源不是有效 UTF-8 JSON：{_display_path(path)}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"來源 summary 頂層必須是 object：{_display_path(path)}")
    return value, payload


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _number(value: Any, label: str, *, nullable: bool = False) -> int | float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必須是有限數字{'或 null' if nullable else ''}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} 不可為 NaN 或 infinity")
    return value


def _rate(value: Any, label: str, *, nullable: bool = False) -> int | float | None:
    number = _number(value, label, nullable=nullable)
    if number is not None and not 0 <= number <= 1:
        raise ValueError(f"{label} 必須介於 0 與 1")
    return number


def _sanitize_config(name: str, raw: Any, *, items_scored: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} 必須是 object")

    rate_keys = {f"{field}_{kind}" for field in SCALAR_FIELDS for kind in ("exact", "fuzzy")}
    required = {"config", "n", *rate_keys, *ITEM_METRICS,
                "latency_cold_s", "latency_warm", "est_cost_per_100_docs_usd"}
    optional = {"json_validity_rate"}
    unknown = set(raw) - required - optional
    missing = required - set(raw)
    if unknown:
        raise ValueError(f"{name} 含未允許欄位：{', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{name} 缺少欄位：{', '.join(sorted(missing))}")
    if raw["config"] != name:
        raise ValueError(f"{name}.config 與外層 key 不一致")
    if raw["n"] != 45:
        raise ValueError(f"{name}.n 應為 45")

    clean: dict[str, Any] = {"config": name, "n": 45}
    for field in SCALAR_FIELDS:
        for kind in ("exact", "fuzzy"):
            key = f"{field}_{kind}"
            clean[key] = _rate(raw[key], f"{name}.{key}")

    for key in ITEM_METRICS:
        clean[key] = _rate(raw[key], f"{name}.{key}", nullable=True)
        if items_scored and clean[key] is None:
            raise ValueError(f"{name}.{key} 在此資料集不可為 null")
        if not items_scored and clean[key] is not None:
            raise ValueError(f"{name}.{key} 在此資料集應為 null")

    cold = _number(raw["latency_cold_s"], f"{name}.latency_cold_s")
    if cold is not None and cold < 0:
        raise ValueError(f"{name}.latency_cold_s 不可小於 0")
    clean["latency_cold_s"] = cold

    warm = raw["latency_warm"]
    if not isinstance(warm, dict) or set(warm) != {"n", "p50", "p95", "mean"}:
        raise ValueError(f"{name}.latency_warm 格式不符")
    if warm["n"] != 44:
        raise ValueError(f"{name}.latency_warm.n 應為 44")
    clean_warm: dict[str, Any] = {"n": 44}
    for key in ("p50", "p95", "mean"):
        clean_warm[key] = _number(warm[key], f"{name}.latency_warm.{key}")
        if clean_warm[key] is not None and clean_warm[key] < 0:
            raise ValueError(f"{name}.latency_warm.{key} 不可小於 0")
    clean["latency_warm"] = clean_warm

    cost = _number(raw["est_cost_per_100_docs_usd"],
                   f"{name}.est_cost_per_100_docs_usd", nullable=True)
    if cost is not None and cost < 0:
        raise ValueError(f"{name}.est_cost_per_100_docs_usd 不可小於 0")
    clean["est_cost_per_100_docs_usd"] = cost

    is_pipeline_b = name.startswith("pipeline_b_")
    if is_pipeline_b != ("json_validity_rate" in raw):
        raise ValueError(f"{name}.json_validity_rate 的有無與管線類型不符")
    if is_pipeline_b:
        clean["json_validity_rate"] = _rate(
            raw["json_validity_rate"], f"{name}.json_validity_rate"
        )
    return clean


def build_artifact(source: Path, metadata: dict[str, Any]) -> bytes:
    raw, source_payload = _read_json(source)
    if set(raw) != set(CONFIG_ORDER):
        missing = set(CONFIG_ORDER) - set(raw)
        extra = set(raw) - set(CONFIG_ORDER)
        details = []
        if missing:
            details.append(f"缺少 {', '.join(sorted(missing))}")
        if extra:
            details.append(f"多出 {', '.join(sorted(extra))}")
        raise ValueError("正式配置集合不符：" + "；".join(details))

    results = {
        name: _sanitize_config(name, raw[name], items_scored=metadata["items_scored"])
        for name in CONFIG_ORDER
    }
    headline = {}
    for name, metrics in results.items():
        avg_exact = sum(metrics[f"{field}_exact"] for field in SCALAR_FIELDS) / len(SCALAR_FIELDS)
        cost = metrics["est_cost_per_100_docs_usd"]
        headline[name] = {
            "avg_exact": round(avg_exact, 6),
            "latency_warm_p50_s": round(metrics["latency_warm"]["p50"], 3),
            "est_cost_per_100_docs_usd": None if cost is None else round(cost, 6),
        }

    artifact = {
        "schema_version": 1,
        "artifact": "ocr_vlm_aggregate_evaluation",
        "generated_by": "scripts/export_official_results.py",
        "dataset": metadata,
        "evaluation": {
            "scalar_fields": list(SCALAR_FIELDS),
            "configurations": list(CONFIG_ORDER),
            "source_summary_sha256": hashlib.sha256(source_payload).hexdigest(),
        },
        "headline_metrics": headline,
        "results": results,
    }
    return (json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-summary", type=Path, default=DATASETS["synthetic"]["input"])
    parser.add_argument("--sroie-summary", type=Path, default=DATASETS["sroie"]["input"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "official")
    parser.add_argument("--check", action="store_true",
                        help="只比對既有 official artifacts，不寫入檔案")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {"synthetic": args.synthetic_summary, "sroie": args.sroie_summary}
    mismatches: list[str] = []

    for key, spec in DATASETS.items():
        payload = build_artifact(sources[key], spec["metadata"])
        output = args.output_dir / spec["output"]
        if args.check:
            if not output.exists() or output.read_bytes() != payload:
                mismatches.append(_display_path(output))
            else:
                print(f"[OK] {_display_path(output)}")
        else:
            _write_atomic(output, payload)
            print(f"[written] {_display_path(output)}")

    if mismatches:
        joined = ", ".join(mismatches)
        raise SystemExit(f"official artifact 與來源 summary 不一致：{joined}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

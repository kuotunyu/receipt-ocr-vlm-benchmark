"""讀 run_eval.py 產生的 summary.json，轉成 Markdown 對比表（Phase 5 寫 EVAL_REPORT.md 時
直接貼這張表當底稿）。

用法：
    .venv\\Scripts\\python scripts\\make_report.py results/eval_<timestamp>/summary.json
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


def fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return str(value)


def overview_table(summary: dict) -> str:
    header = (
        "| config | n | avg exact | avg fuzzy | items F1 | items name exact | "
        "latency p50 (warm) | latency p95 (warm) | validity | est. cost/100 docs (USD) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for name, agg in summary.items():
        avg_exact = statistics.mean(agg[f"{f}_exact"] for f in SCALAR_FIELDS)
        avg_fuzzy = statistics.mean(agg[f"{f}_fuzzy"] for f in SCALAR_FIELDS)
        warm = agg.get("latency_warm") or {}
        rows.append(
            f"| {name} | {agg['n']} | {fmt(avg_exact)} | {fmt(avg_fuzzy)} | {fmt(agg['items_f1'])} | "
            f"{fmt(agg.get('items_name_exact_rate'))} | "
            f"{fmt(warm.get('p50'), 2)}s | {fmt(warm.get('p95'), 2)}s | "
            f"{fmt(agg.get('json_validity_rate'))} | {fmt(agg.get('est_cost_per_100_docs_usd'), 2)} |"
        )
    return "\n".join(rows)


def per_field_table(summary: dict) -> str:
    header = "| config | " + " | ".join(SCALAR_FIELDS) + " |"
    sep = "|---|" + "---|" * len(SCALAR_FIELDS)
    rows = [header, sep]
    for name, agg in summary.items():
        cells = " | ".join(fmt(agg[f"{f}_exact"]) for f in SCALAR_FIELDS)
        rows.append(f"| {name} | {cells} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="run_eval.py 產生的 summary.json")
    args = parser.parse_args()

    summary_path = args.summary
    summary = read_json(summary_path)

    report = (
        "## 整體對比\n\n" + overview_table(summary) + "\n\n"
        "## 各欄位 exact match 準確率\n\n" + per_field_table(summary) + "\n"
    )
    print(report)

    out_path = summary_path.parent / "report_table.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n已存到 {out_path}")


if __name__ == "__main__":
    main()

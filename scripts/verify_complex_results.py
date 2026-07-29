"""Verify checked summary metrics by recomputing from normalized IR artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.chunking import (
    context_chunks,
    fixed_size_chunks,
    hybrid_routed_chunks,
    structure_aware_chunks,
)
from src.complex_document.downstream_eval import evaluate_downstream
from src.complex_document.ir import SpatialDocument
from src.complex_document.normalization_audit import (
    aggregate_audits,
    audit_artifact,
)
from src.complex_document.parser_metrics import evaluate_parser
from src.complex_document.routing import (
    evaluate_reconstructed_table_bboxes,
    evaluate_table_router,
)
from scripts.run_complex_benchmark import _caption_ablation


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    summary_path = Path("results/complex_document/benchmark_summary.json")
    if not summary_path.is_file():
        raise SystemExit("benchmark summary missing; run run_complex_benchmark.py")
    summary = load(summary_path)
    manifest = load(Path("data/complex_document/manifest.json"))
    cases = load(Path("data/complex_document/gold/hard_cases.json"))["cases"]
    questions = load(Path("data/complex_document/questions.json"))["questions"]
    routing_gold = load(
        Path("data/complex_document/gold/table_routing_pages.json")
    )["pages"]

    checked = 0
    completed_documents = {}
    for parser_key, parser_result in summary["parser_results"].items():
        if parser_result["status"] != "completed":
            continue
        parser_name = parser_result["timing_and_cost"]["parser_name"]
        documents = {
            item["document_id"]: SpatialDocument.from_json(
                (
                    Path("artifacts/complex_document/ir")
                    / item["document_id"]
                    / parser_name
                    / "document.ir.json"
                ).read_text(encoding="utf-8")
            )
            for item in manifest["documents"]
        }
        actual = evaluate_parser(documents, cases)
        completed_documents[parser_key] = documents
        assert actual["mean_score"] == parser_result["parser_metrics"]["mean_score"]
        assert actual["dimensions"] == parser_result["parser_metrics"]["dimensions"]
        normalization_audits = []
        for item in manifest["documents"]:
            raw_path = (
                Path("artifacts/complex_document/parser_raw")
                / item["document_id"]
                / parser_name
                / "raw.json"
            )
            normalization_audits.append(
                {
                    "document_id": item["document_id"],
                    **audit_artifact(raw_path, documents[item["document_id"]]),
                }
            )
        normalization = aggregate_audits(normalization_audits)
        for metric in (
            "mean_text_character_recall",
            "mean_text_character_precision",
            "native_items",
            "ir_elements",
            "shadowed_native_elements",
        ):
            assert normalization[metric] == parser_result[
                "normalization_audit"
            ][metric]
        checked += 1

    if "pymupdf" in completed_documents:
        routing = evaluate_table_router(
            completed_documents["pymupdf"], routing_gold
        )
        assert routing == summary["table_routing_audit"]
    for parser_key, documents in completed_documents.items():
        bbox_audit = evaluate_reconstructed_table_bboxes(documents, cases)
        assert bbox_audit == summary["table_bbox_audit"][parser_key]

    for factor in summary["factor_at_a_time"]:
        if factor.get("status") != "completed" or "downstream" not in factor:
            continue
        parser_name = summary["parser_results"][factor["parser"]][
            "timing_and_cost"
        ]["parser_name"]
        chunks = []
        for item in manifest["documents"]:
            document = SpatialDocument.from_json(
                (
                    Path("artifacts/complex_document/ir")
                    / item["document_id"]
                    / parser_name
                    / "document.ir.json"
                ).read_text(encoding="utf-8")
            )
            if factor["chunking"] == "fixed":
                chunks.extend(fixed_size_chunks(document))
            elif factor["chunking"] == "hybrid-routed":
                chunks.extend(hybrid_routed_chunks(document))
            else:
                chunks.extend(context_chunks(structure_aware_chunks(document)))
        actual = evaluate_downstream(questions, chunks, k=5)
        for metric in (
            "retrieval_recall_at_k",
            "mrr",
            "answer_correctness",
            "citation_validity",
            "error_attribution",
        ):
            assert actual[metric] == factor["downstream"][metric], (
                factor["factor"],
                metric,
            )
        checked += 1
    caption_factor = next(
        (
            factor
            for factor in summary["factor_at_a_time"]
            if factor["factor"] == "5_caption_and_index"
        ),
        None,
    )
    if (
        caption_factor
        and caption_factor.get("status", "").startswith("completed")
        and "liteparse" in completed_documents
    ):
        caption_base = []
        for document in completed_documents["liteparse"].values():
            caption_base.extend(
                context_chunks(structure_aware_chunks(document))
            )
        actual = _caption_ablation(
            caption_base,
            questions,
            Path("artifacts/complex_document/chart_captions/qwen3-vl.json"),
        )
        for key in (
            "status",
            "caption_model",
            "caption_generation",
            "modes",
        ):
            assert actual[key] == caption_factor[key], (
                "5_caption_and_index",
                key,
            )
        checked += 1
    print(f"OK: verified {checked} parser/factor result blocks from IR artifacts")


if __name__ == "__main__":
    main()

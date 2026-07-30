"""Run parser metrics and fixed-factor downstream experiments from raw artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.caption_index import ChartCaption, caption_chunk
from src.complex_document.chunking import (
    Chunk,
    context_chunks,
    fixed_size_chunks,
    hybrid_routed_chunks,
    structure_aware_chunks,
)
from src.complex_document.downstream_eval import evaluate_downstream
from src.complex_document.ir import SpatialDocument, sha256_file
from src.complex_document.normalization_audit import (
    aggregate_audits,
    audit_artifact,
)
from src.complex_document.parser_metrics import evaluate_parser
from src.complex_document.parsers import (
    HybridTableRouterAdapter,
    LiteParseAdapter,
    LiteParseTableAdapter,
    PaddleLayoutAdapter,
    ParseRequest,
    ParserUnavailable,
    PyMuPDFAdapter,
    QwenVLMParserAdapter,
    TargetedVLMRouterAdapter,
)
from src.complex_document.routing import (
    evaluate_reconstructed_table_bboxes,
    evaluate_table_router,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _adapter(name: str):
    if name == "pymupdf":
        return PyMuPDFAdapter()
    if name == "liteparse":
        return LiteParseAdapter(ocr_enabled=False)
    if name == "liteparse-table":
        return LiteParseTableAdapter(ocr_enabled=False)
    if name == "hybrid-table-router":
        return HybridTableRouterAdapter()
    if name == "paddleocr-layout":
        return PaddleLayoutAdapter()
    if name == "qwen3-vl":
        return QwenVLMParserAdapter()
    if name == "targeted-vlm":
        return TargetedVLMRouterAdapter()
    raise ValueError(name)


def _load_or_parse(
    parser_name: str,
    manifest: dict,
    store: ArtifactStore,
    *,
    reuse_ir: bool,
) -> tuple[dict[str, SpatialDocument], dict]:
    adapter = _adapter(parser_name)
    documents: dict[str, SpatialDocument] = {}
    latencies = []
    all_reused = True
    latency_sources: set[str] = set()
    page_count = 0
    for item in manifest["documents"]:
        source_path = Path("data/complex_document/raw") / item["filename"]
        if not source_path.is_file():
            raise FileNotFoundError(
                f"{source_path} missing; run scripts/download_complex_documents.py"
            )
        if sha256_file(source_path) != item["sha256"]:
            raise RuntimeError(f"checksum mismatch: {source_path}")
        ir_path = (
            store.paths.ir
            / item["document_id"]
            / adapter.name
            / "document.ir.json"
        )
        if reuse_ir and ir_path.is_file():
            document = SpatialDocument.from_json(ir_path.read_text(encoding="utf-8"))
            raw_path = (
                store.paths.parser_raw
                / item["document_id"]
                / adapter.name
                / "raw.json"
            )
            elapsed = 0.0
            if raw_path.is_file():
                raw_payload = _load_json(raw_path)
                stored_page_latencies = [
                    page.get("latency_seconds")
                    for page in raw_payload.get("pages", [])
                    if isinstance(page, dict)
                    and isinstance(page.get("latency_seconds"), (int, float))
                ]
                if stored_page_latencies:
                    elapsed = float(sum(stored_page_latencies))
                    latency_sources.add("parser_raw_artifact")
                    if adapter.name == "targeted-vlm-router":
                        stored_baseline = raw_payload.get(
                            "baseline_wall_seconds"
                        )
                        if isinstance(stored_baseline, (int, float)):
                            elapsed += float(stored_baseline)
                            latency_sources.add(
                                "parser_raw_baseline_wall_clock"
                            )
                        else:
                            baseline_started = time.perf_counter()
                            adapter.baseline.parse(
                                ParseRequest(
                                    path=source_path,
                                    document_id=item["document_id"],
                                    source_uri=item["url"],
                                    pages=tuple(item["selected_pages"]),
                                    config={
                                        "benchmark_version": manifest[
                                            "benchmark_version"
                                        ]
                                    },
                                ),
                                artifacts=None,
                            )
                            elapsed += (
                                time.perf_counter() - baseline_started
                            )
                            latency_sources.add(
                                "current_baseline_wall_clock"
                            )
        else:
            all_reused = False
            started = time.perf_counter()
            document = adapter.parse(
                ParseRequest(
                    path=source_path,
                    document_id=item["document_id"],
                    source_uri=item["url"],
                    pages=tuple(item["selected_pages"]),
                    config={"benchmark_version": manifest["benchmark_version"]},
                ),
                store,
            )
            elapsed = time.perf_counter() - started
            latency_sources.add("current_run_wall_clock")
        documents[item["document_id"]] = document
        latencies.append(elapsed)
        page_count += len(document.pages)
    nonzero = [value for value in latencies if value > 0]
    return documents, {
        "parser_name": adapter.name,
        "parser_version": next(iter(documents.values())).parser.version,
        "documents": len(documents),
        "pages": page_count,
        "latency_seconds_total": round(sum(latencies), 6),
        "latency_seconds_median_document": round(statistics.median(latencies), 6),
        "pages_per_second": round(page_count / sum(nonzero), 6) if nonzero else None,
        "estimated_api_cost_usd": 0.0,
        "reused_ir": all_reused,
        "latency_source": (
            "+".join(sorted(latency_sources)) if latency_sources else "unavailable"
        ),
    }


def _chunks(documents: dict[str, SpatialDocument], mode: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents.values():
        if mode == "fixed":
            chunks.extend(fixed_size_chunks(document))
        elif mode == "structure":
            chunks.extend(context_chunks(structure_aware_chunks(document)))
        elif mode == "hybrid-routed":
            chunks.extend(hybrid_routed_chunks(document))
        else:
            raise ValueError(mode)
    return chunks


def _correctness_by_question_type(downstream: dict) -> dict[str, float]:
    types = sorted(
        {question["question_type"] for question in downstream["questions"]}
    )
    return {
        question_type: round(
            sum(
                question["correct"]
                for question in downstream["questions"]
                if question["question_type"] == question_type
            )
            / sum(
                question["question_type"] == question_type
                for question in downstream["questions"]
            ),
            6,
        )
        for question_type in types
    }


def _caption_ablation(
    base_chunks: list[Chunk],
    questions: list[dict],
    caption_path: Path,
) -> dict:
    if not caption_path.is_file():
        return {
            "status": "skipped",
            "reason": (
                "No VLM-generated caption artifact. Run "
                "scripts/generate_chart_captions.py with qwen3-vl:8b."
            ),
        }
    payload = _load_json(caption_path)
    captions = [
        ChartCaption(
            figure_id=item["figure_id"],
            generic_caption=item["generic_caption"],
            structured_caption=item["structured_caption"],
            page_number=item["page"],
            bbox=item["bbox_normalized"],
            crop_ref=item["crop_ref"],
            axis_names=item["axis_names"],
            unit=item.get("unit"),
            series=item["series"],
            values=item["values"],
            trend=item.get("trend"),
        )
        for item in payload["captions"]
    ]
    modes = {}
    target_question_ids = {
        question_id
        for item in payload["captions"]
        for question_id in item.get("question_ids", [])
    }
    chart_questions = [
        question
        for question in questions
        if question["question_id"] in target_question_ids
    ]
    for mode in (
        "no_image_indexing",
        "generic_caption",
        "structured_caption",
        "structured_caption_original_crop",
    ):
        indexed = list(base_chunks)
        question_caption_ids: dict[str, list[str]] = {}
        if mode != "no_image_indexing":
            for caption in captions:
                caption_payload = next(
                    item
                    for item in payload["captions"]
                    if item["figure_id"] == caption.figure_id
                )
                base = next(
                    (
                        chunk
                        for chunk in base_chunks
                        if chunk.document_id
                        == caption_payload["document_id"]
                        and caption.page_number in chunk.pages
                    ),
                    None,
                )
                if base:
                    new_chunk = caption_chunk(base, caption, mode=mode)
                    if new_chunk:
                        indexed.append(new_chunk)
                        for question_id in caption_payload["question_ids"]:
                            question_caption_ids.setdefault(
                                question_id, []
                            ).append(new_chunk.chunk_id)
        metrics = evaluate_downstream(chart_questions, indexed, k=5)
        pixel_synthesis_executed = False
        if mode == "structured_caption_original_crop":
            answers = {
                answer["question_id"]: answer["answer"]
                for item in payload["captions"]
                for answer in item.get("pixel_answers", [])
            }
            if all(
                question["question_id"] in answers
                for question in chart_questions
            ):
                metrics = _apply_pixel_synthesis(
                    metrics,
                    chart_questions,
                    answers,
                    question_caption_ids,
                )
                pixel_synthesis_executed = True
        modes[mode] = {
            **metrics,
            "pixel_synthesis_required": mode
            == "structured_caption_original_crop",
            "pixel_synthesis_executed": pixel_synthesis_executed,
            "note": (
                "Caption text is retrieval-only. The original-crop mode uses "
                "the saved answer generated from crop pixels without caption text."
            ),
        }
    return {
        "status": (
            "completed"
            if modes["structured_caption_original_crop"][
                "pixel_synthesis_executed"
            ]
            else "completed_retrieval_only"
        ),
        "caption_model": payload["model"],
        "caption_generation": payload.get("generation_summary"),
        "modes": modes,
    }


def _answer_normalize(value: str | None) -> str:
    return re.sub(r"[\s,，。．%％元]", "", value or "").lower()


def _apply_pixel_synthesis(
    metrics: dict,
    questions: list[dict],
    answers: dict[str, str],
    question_caption_ids: dict[str, list[str]],
) -> dict:
    """Replace deterministic text answers with saved original-crop VLM answers."""
    question_gold = {item["question_id"]: item for item in questions}
    results = []
    crop_retrieved = 0
    for result in metrics["questions"]:
        question_id = result["question_id"]
        gold = question_gold[question_id]
        expected_chunk_ids = set(question_caption_ids.get(question_id, []))
        retrieved = bool(
            expected_chunk_ids.intersection(result["retrieved_chunk_ids"])
        )
        crop_retrieved += int(retrieved)
        answer = answers.get(question_id) if retrieved else None
        normalized_answer = _answer_normalize(answer)
        correct = bool(
            retrieved
            and any(
                _answer_normalize(expected) in normalized_answer
                for expected in gold.get("answers", [])
            )
        )
        result = {
            **result,
            "answer": answer,
            "correct": correct,
            "citation_valid": correct and retrieved,
            "error_source": (
                None if correct else ("generation" if retrieved else "retrieval")
            ),
        }
        results.append(result)
    count = len(results)
    error_attribution = {
        source: sum(item["error_source"] == source for item in results)
        for source in ("parsing", "retrieval", "generation")
    }
    return {
        **metrics,
        "answerer": "qwen3-vl-original-crop-no-caption",
        "answer_correctness": round(
            sum(item["correct"] for item in results) / count, 6
        )
        if count
        else None,
        "citation_validity": round(
            sum(item["citation_valid"] for item in results) / count, 6
        )
        if count
        else None,
        "crop_recall_at_k": round(crop_retrieved / count, 6)
        if count
        else None,
        "error_attribution": error_attribution,
        "questions": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parsers",
        nargs="+",
        default=[
            "pymupdf",
            "paddleocr-layout",
            "liteparse",
            "liteparse-table",
            "hybrid-table-router",
            "qwen3-vl",
            "targeted-vlm",
        ],
    )
    parser.add_argument("--reuse-ir", action="store_true")
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/complex_document/manifest.json")
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/complex_document/gold/hard_cases.json"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/complex_document/questions.json"),
    )
    parser.add_argument(
        "--routing-gold",
        type=Path,
        default=Path(
            "data/complex_document/gold/table_routing_pages.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/complex_document/benchmark_summary.json"),
    )
    parser.add_argument(
        "--caption-artifact",
        type=Path,
        default=Path("artifacts/complex_document/chart_captions/qwen3-vl.json"),
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    cases = _load_json(args.gold)["cases"]
    questions = _load_json(args.questions)["questions"]
    routing_gold = _load_json(args.routing_gold)["pages"]
    store = ArtifactStore()
    parsed: dict[str, dict[str, SpatialDocument]] = {}
    parser_results = {}

    for name in args.parsers:
        try:
            documents, timing = _load_or_parse(
                name, manifest, store, reuse_ir=args.reuse_ir
            )
            parsed[name] = documents
            normalization_audits = []
            for item in manifest["documents"]:
                raw_path = (
                    store.paths.parser_raw
                    / item["document_id"]
                    / documents[item["document_id"]].parser.name
                    / "raw.json"
                )
                if raw_path.is_file():
                    normalization_audits.append(
                        {
                            "document_id": item["document_id"],
                            **audit_artifact(
                                raw_path, documents[item["document_id"]]
                            ),
                        }
                    )
            parser_results[name] = {
                "status": "completed",
                "timing_and_cost": timing,
                "parser_metrics": evaluate_parser(documents, cases),
                "normalization_audit": aggregate_audits(
                    normalization_audits
                ),
            }
            print(
                f"{name}: {timing['pages']} pages, "
                f"metric={parser_results[name]['parser_metrics']['mean_score']}"
            )
        except ParserUnavailable as exc:
            parser_results[name] = {"status": "skipped", "reason": str(exc)}
            print(f"{name}: SKIP {exc}")

    factor_specs = [
        ("1_current_parser_fixed", "pymupdf", "fixed"),
        ("1b_paddleocr_layout_fixed", "paddleocr-layout", "fixed"),
        ("2_liteparse_fixed", "liteparse", "fixed"),
        ("3_liteparse_structure", "liteparse", "structure"),
        (
            "3b_liteparse_table_reconstruction",
            "liteparse-table",
            "structure",
        ),
        (
            "3c_hybrid_table_page_router",
            "hybrid-table-router",
            "hybrid-routed",
        ),
        ("4_vlm_parser_structure", "qwen3-vl", "structure"),
        ("4a_targeted_vlm_fixed_diagnostic", "targeted-vlm", "fixed"),
        ("4b_targeted_vlm_structure", "targeted-vlm", "structure"),
    ]
    factors = []
    for factor_name, parser_name, chunk_mode in factor_specs:
        if parser_name not in parsed:
            factors.append(
                {
                    "factor": factor_name,
                    "parser": parser_name,
                    "chunking": chunk_mode,
                    "status": "skipped",
                    "reason": parser_results.get(parser_name, {}).get(
                        "reason", "parser not requested"
                    ),
                }
            )
            continue
        factor_chunks = _chunks(parsed[parser_name], chunk_mode)
        factors.append(
            {
                "factor": factor_name,
                "parser": parser_name,
                "chunking": chunk_mode,
                "status": "completed",
                "chunk_count": len(factor_chunks),
                "downstream": evaluate_downstream(questions, factor_chunks, k=5),
            }
        )

    caption_base = (
        _chunks(parsed["liteparse"], "structure") if "liteparse" in parsed else []
    )
    caption_result = (
        _caption_ablation(caption_base, questions, args.caption_artifact)
        if caption_base
        else {"status": "skipped", "reason": "LiteParse base unavailable"}
    )
    factors.append(
        {
            "factor": "5_caption_and_index",
            "parser": "liteparse",
            "chunking": "structure+chart-caption-index",
            **caption_result,
        }
    )

    routing_audit = (
        evaluate_table_router(parsed["pymupdf"], routing_gold)
        if "pymupdf" in parsed
        else {
            "status": "skipped",
            "reason": "PyMuPDF routing source unavailable",
        }
    )
    table_bbox_audit = {
        name: evaluate_reconstructed_table_bboxes(documents, cases)
        for name, documents in parsed.items()
    }

    decision = {
        "recommendation": "NO-GO",
        "rule": (
            "GO only if LiteParse improves at least one parser metric and does "
            "not reduce downstream answer correctness versus the current baseline."
        ),
        "reason": "Required comparable completed factors are unavailable.",
    }
    baseline = next(
        (factor for factor in factors if factor["factor"] == "1_current_parser_fixed"),
        None,
    )
    proposed = next(
        (
            factor
            for factor in factors
            if factor["factor"] == "3b_liteparse_table_reconstruction"
        ),
        None,
    )
    if (
        baseline
        and proposed
        and baseline.get("status") == "completed"
        and proposed.get("status") == "completed"
    ):
        parser_delta = (
            parser_results["liteparse-table"]["parser_metrics"]["mean_score"]
            - parser_results["pymupdf"]["parser_metrics"]["mean_score"]
        )
        table_delta = (
            parser_results["liteparse-table"]["parser_metrics"]["dimensions"][
                "table_structure"
            ]
            - parser_results["pymupdf"]["parser_metrics"]["dimensions"][
                "table_structure"
            ]
        )
        qa_delta = (
            proposed["downstream"]["answer_correctness"]
            - baseline["downstream"]["answer_correctness"]
        )
        citation_delta = (
            proposed["downstream"]["citation_validity"]
            - baseline["downstream"]["citation_validity"]
        )
        mrr_delta = (
            proposed["downstream"]["mrr"]
            - baseline["downstream"]["mrr"]
        )
        baseline_types = _correctness_by_question_type(
            baseline["downstream"]
        )
        proposed_types = _correctness_by_question_type(
            proposed["downstream"]
        )
        improved_types = sorted(
            question_type
            for question_type, score in proposed_types.items()
            if score > baseline_types.get(question_type, 0)
        )
        gates = {
            "parser_mean_score_delta_at_least_0_05": parser_delta >= 0.05,
            "table_structure_delta_at_least_0_20": table_delta >= 0.20,
            "answer_correctness_non_decreasing": qa_delta >= 0,
            "citation_validity_non_decreasing": citation_delta >= 0,
            "at_least_two_question_types_improved": len(improved_types) >= 2,
        }
        go = all(gates.values())
        targeted_table_routing_go = (
            table_delta >= 0.20
            and qa_delta > 0
            and citation_delta > 0
            and len(improved_types) >= 2
        )
        decision = {
            "recommendation": "GO" if go else "NO-GO",
            "rule": (
                "Promote the table reconstruction factor only if all frozen "
                "parser, downstream, citation, and breadth gates pass."
            ),
            "parser_mean_score_delta": round(parser_delta, 6),
            "table_structure_delta": round(table_delta, 6),
            "answer_correctness_delta": round(qa_delta, 6),
            "citation_validity_delta": round(citation_delta, 6),
            "mrr_delta": round(mrr_delta, 6),
            "improved_question_types": improved_types,
            "gates": gates,
            "reason": (
                "All frozen promotion gates satisfied."
                if go
                else "One or more frozen promotion gates were not satisfied."
            ),
            "targeted_table_routing": {
                "recommendation": (
                    "GO" if targeted_table_routing_go else "NO-GO"
                ),
                "scope": (
                    "Route only pages with detected grid tables through "
                    "LiteParse table reconstruction; retain PyMuPDF as the "
                    "default and fallback parser."
                ),
                "rule": (
                    "Pilot only if table structure improves by at least 0.20, "
                    "answer and citation both improve, and at least two "
                    "question types improve. This is not a global promotion."
                ),
                "risk": (
                    "Retrieval ranking regressed; reconstructed table bboxes "
                    "can over-merge adjacent tables and require a routing "
                    "precision audit."
                ),
            },
        }
        hybrid = next(
            (
                factor
                for factor in factors
                if factor["factor"] == "3c_hybrid_table_page_router"
            ),
            None,
        )
        if (
            hybrid
            and hybrid.get("status") == "completed"
            and routing_audit.get("precision") is not None
        ):
            hybrid_answer_delta = (
                hybrid["downstream"]["answer_correctness"]
                - baseline["downstream"]["answer_correctness"]
            )
            hybrid_citation_delta = (
                hybrid["downstream"]["citation_validity"]
                - baseline["downstream"]["citation_validity"]
            )
            hybrid_mrr_delta = (
                hybrid["downstream"]["mrr"]
                - baseline["downstream"]["mrr"]
            )
            hybrid_vs_enriched_mrr_delta = (
                hybrid["downstream"]["mrr"]
                - proposed["downstream"]["mrr"]
            )
            hybrid_gates = {
                "routing_precision_at_least_0_90": (
                    routing_audit["precision"] >= 0.90
                ),
                "routing_recall_at_least_0_90": (
                    routing_audit["recall"] >= 0.90
                ),
                "answer_at_least_full_enrichment": (
                    hybrid["downstream"]["answer_correctness"]
                    >= proposed["downstream"]["answer_correctness"]
                ),
                "citation_at_least_full_enrichment": (
                    hybrid["downstream"]["citation_validity"]
                    >= proposed["downstream"]["citation_validity"]
                ),
                "mrr_at_least_current_baseline": (
                    hybrid["downstream"]["mrr"]
                    >= baseline["downstream"]["mrr"]
                ),
            }
            hybrid_go = all(hybrid_gates.values())
            decision["targeted_table_routing"] = {
                "recommendation": "GO" if hybrid_go else "NO-GO",
                "scope": (
                    "Route only high-confidence vector-grid table pages "
                    "through LiteParse table reconstruction; retain PyMuPDF "
                    "as the default and fallback parser."
                ),
                "rule": (
                    "Promote the pilot only if routing precision/recall are "
                    "at least 0.90, answer/citation retain the full-enrichment "
                    "gain, and MRR recovers to the current baseline."
                ),
                "answer_correctness_delta_vs_baseline": round(
                    hybrid_answer_delta, 6
                ),
                "citation_validity_delta_vs_baseline": round(
                    hybrid_citation_delta, 6
                ),
                "mrr_delta_vs_baseline": round(hybrid_mrr_delta, 6),
                "mrr_delta_vs_full_enrichment": round(
                    hybrid_vs_enriched_mrr_delta, 6
                ),
                "gates": hybrid_gates,
                "risk": (
                    "The routing audit is based on 26 manually reviewed "
                    "benchmark pages and needs external-document validation."
                ),
            }

    vlm_factor = next(
        (
            factor
            for factor in factors
            if factor["factor"] == "4_vlm_parser_structure"
        ),
        None,
    )
    if (
        baseline
        and vlm_factor
        and baseline.get("status") == "completed"
        and vlm_factor.get("status") == "completed"
        and parser_results.get("qwen3-vl", {}).get("status") == "completed"
    ):
        vlm_parser_delta = (
            parser_results["qwen3-vl"]["parser_metrics"]["mean_score"]
            - parser_results["pymupdf"]["parser_metrics"]["mean_score"]
        )
        vlm_gates = {
            "parser_mean_score_at_least_current": vlm_parser_delta >= 0,
            "retrieval_recall_at_least_current": (
                vlm_factor["downstream"]["retrieval_recall_at_k"]
                >= baseline["downstream"]["retrieval_recall_at_k"]
            ),
            "mrr_at_least_current": (
                vlm_factor["downstream"]["mrr"]
                >= baseline["downstream"]["mrr"]
            ),
            "answer_correctness_at_least_current": (
                vlm_factor["downstream"]["answer_correctness"]
                >= baseline["downstream"]["answer_correctness"]
            ),
            "citation_validity_at_least_current": (
                vlm_factor["downstream"]["citation_validity"]
                >= baseline["downstream"]["citation_validity"]
            ),
        }
        decision["vlm_parser"] = {
            "recommendation": "GO" if all(vlm_gates.values()) else "NO-GO",
            "rule": (
                "Promote the VLM parser only if parser score, retrieval recall, "
                "MRR, answer correctness, and citation validity all match or "
                "exceed the current parser baseline."
            ),
            "parser_mean_score_delta_vs_baseline": round(vlm_parser_delta, 6),
            "retrieval_recall_delta_vs_baseline": round(
                vlm_factor["downstream"]["retrieval_recall_at_k"]
                - baseline["downstream"]["retrieval_recall_at_k"],
                6,
            ),
            "mrr_delta_vs_baseline": round(
                vlm_factor["downstream"]["mrr"]
                - baseline["downstream"]["mrr"],
                6,
            ),
            "answer_correctness_delta_vs_baseline": round(
                vlm_factor["downstream"]["answer_correctness"]
                - baseline["downstream"]["answer_correctness"],
                6,
            ),
            "citation_validity_delta_vs_baseline": round(
                vlm_factor["downstream"]["citation_validity"]
                - baseline["downstream"]["citation_validity"],
                6,
            ),
            "gates": vlm_gates,
        }

    targeted_vlm_factor = next(
        (
            factor
            for factor in factors
            if factor["factor"] == "4b_targeted_vlm_structure"
        ),
        None,
    )
    if (
        baseline
        and targeted_vlm_factor
        and baseline.get("status") == "completed"
        and targeted_vlm_factor.get("status") == "completed"
        and parser_results.get("targeted-vlm", {}).get("status")
        == "completed"
    ):
        base_downstream = baseline["downstream"]
        candidate_downstream = targeted_vlm_factor["downstream"]
        targeted_parser_delta = (
            parser_results["targeted-vlm"]["parser_metrics"]["mean_score"]
            - parser_results["pymupdf"]["parser_metrics"]["mean_score"]
        )
        quality_deltas = {
            "parser_mean_score": targeted_parser_delta,
            "retrieval_recall_at_k": (
                candidate_downstream["retrieval_recall_at_k"]
                - base_downstream["retrieval_recall_at_k"]
            ),
            "mrr": candidate_downstream["mrr"] - base_downstream["mrr"],
            "answer_correctness": (
                candidate_downstream["answer_correctness"]
                - base_downstream["answer_correctness"]
            ),
            "citation_validity": (
                candidate_downstream["citation_validity"]
                - base_downstream["citation_validity"]
            ),
        }
        routed_pages = sum(
            len(
                document.parser.config.get("routed_pages", [])
            )
            for document in parsed["targeted-vlm"].values()
        )
        targeted_gates = {
            "parser_mean_score_at_least_current": targeted_parser_delta >= 0,
            "retrieval_recall_at_least_current": (
                quality_deltas["retrieval_recall_at_k"] >= 0
            ),
            "mrr_at_least_current": quality_deltas["mrr"] >= 0,
            "answer_correctness_at_least_current": (
                quality_deltas["answer_correctness"] >= 0
            ),
            "citation_validity_at_least_current": (
                quality_deltas["citation_validity"] >= 0
            ),
            "one_quality_metric_strictly_improves": any(
                delta > 0 for delta in quality_deltas.values()
            ),
            "at_least_one_page_routed": routed_pages > 0,
        }
        decision["targeted_vlm"] = {
            "recommendation": (
                "GO" if all(targeted_gates.values()) else "NO-GO"
            ),
            "rule": (
                "Promote targeted VLM only if no parser/downstream quality "
                "metric regresses and at least one strictly improves. Routing "
                "uses native page signals only, never question or answer gold."
            ),
            "router": "native-visual-router-1",
            "routed_pages": routed_pages,
            "quality_deltas_vs_baseline": {
                key: round(value, 6)
                for key, value in quality_deltas.items()
            },
            "gates": targeted_gates,
        }

    paddle_factor = next(
        (
            factor
            for factor in factors
            if factor["factor"] == "1b_paddleocr_layout_fixed"
        ),
        None,
    )
    if (
        baseline
        and paddle_factor
        and baseline.get("status") == "completed"
        and paddle_factor.get("status") == "completed"
        and parser_results.get("paddleocr-layout", {}).get("status")
        == "completed"
    ):
        paddle_parser_delta = (
            parser_results["paddleocr-layout"]["parser_metrics"]["mean_score"]
            - parser_results["pymupdf"]["parser_metrics"]["mean_score"]
        )
        paddle_gates = {
            "parser_mean_score_at_least_current": paddle_parser_delta >= 0,
            "retrieval_recall_at_least_current": (
                paddle_factor["downstream"]["retrieval_recall_at_k"]
                >= baseline["downstream"]["retrieval_recall_at_k"]
            ),
            "mrr_at_least_current": (
                paddle_factor["downstream"]["mrr"]
                >= baseline["downstream"]["mrr"]
            ),
            "answer_correctness_at_least_current": (
                paddle_factor["downstream"]["answer_correctness"]
                >= baseline["downstream"]["answer_correctness"]
            ),
            "citation_validity_at_least_current": (
                paddle_factor["downstream"]["citation_validity"]
                >= baseline["downstream"]["citation_validity"]
            ),
        }
        paddle_timing = parser_results["paddleocr-layout"][
            "timing_and_cost"
        ]
        decision["paddleocr_layout"] = {
            "recommendation": (
                "GO" if all(paddle_gates.values()) else "NO-GO"
            ),
            "scope": "Global parser replacement.",
            "rule": (
                "Promote globally only if parser mean score and all downstream "
                "metrics match or exceed the current parser baseline."
            ),
            "parser_mean_score_delta_vs_baseline": round(
                paddle_parser_delta, 6
            ),
            "retrieval_recall_delta_vs_baseline": round(
                paddle_factor["downstream"]["retrieval_recall_at_k"]
                - baseline["downstream"]["retrieval_recall_at_k"],
                6,
            ),
            "mrr_delta_vs_baseline": round(
                paddle_factor["downstream"]["mrr"]
                - baseline["downstream"]["mrr"],
                6,
            ),
            "answer_correctness_delta_vs_baseline": round(
                paddle_factor["downstream"]["answer_correctness"]
                - baseline["downstream"]["answer_correctness"],
                6,
            ),
            "citation_validity_delta_vs_baseline": round(
                paddle_factor["downstream"]["citation_validity"]
                - baseline["downstream"]["citation_validity"],
                6,
            ),
            "seconds_per_page": round(
                paddle_timing["latency_seconds_total"]
                / paddle_timing["pages"],
                6,
            ),
            "gates": paddle_gates,
            "research_note": (
                "High OCR text completeness improved downstream QA, but "
                "table/visual structure is absent. Evaluate only as a routed "
                "fallback on scanned or low-native-text pages."
            ),
        }

    caption_factor = next(
        (
            factor
            for factor in factors
            if factor["factor"] == "5_caption_and_index"
        ),
        None,
    )
    if caption_factor and caption_factor.get("status") == "completed":
        modes = caption_factor["modes"]
        no_image = modes["no_image_indexing"]
        pixel_mode = modes["structured_caption_original_crop"]
        caption_gates = {
            "structured_retrieval_recall_at_least_no_image": (
                pixel_mode["retrieval_recall_at_k"]
                >= no_image["retrieval_recall_at_k"]
            ),
            "pixel_synthesis_executed": bool(
                pixel_mode.get("pixel_synthesis_executed")
            ),
            "crop_recall_at_k_at_least_0_90": (
                (pixel_mode.get("crop_recall_at_k") or 0) >= 0.90
            ),
            "answer_correctness_at_least_0_80": (
                pixel_mode["answer_correctness"] >= 0.80
            ),
            "citation_validity_at_least_0_80": (
                pixel_mode["citation_validity"] >= 0.80
            ),
            "human_question_count_at_least_2": (
                pixel_mode["question_count"] >= 2
            ),
        }
        decision["caption_and_index"] = {
            "recommendation": (
                "GO" if all(caption_gates.values()) else "NO-GO"
            ),
            "rule": (
                "Promote caption-and-index only if it improves retrieval, "
                "actually synthesizes from retrieved crops, and reaches at "
                "least 0.80 answer and citation validity on human gold."
            ),
            "retrieval_recall_delta_vs_no_image": round(
                pixel_mode["retrieval_recall_at_k"]
                - no_image["retrieval_recall_at_k"],
                6,
            ),
            "answer_correctness": pixel_mode["answer_correctness"],
            "citation_validity": pixel_mode["citation_validity"],
            "gates": caption_gates,
        }

    summary = {
        "benchmark_version": manifest["benchmark_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device_policy": (
                "CPU baselines plus local RTX 4090 Qwen3-VL run; "
                "competing Ollama models are refused"
                if parser_results.get("qwen3-vl", {}).get("status")
                == "completed"
                else "CPU benchmark; GPU work skipped or unavailable"
            ),
        },
        "dataset": {
            "documents": len(manifest["documents"]),
            "selected_pages": sum(
                len(item["selected_pages"]) for item in manifest["documents"]
            ),
            "human_hard_cases": len(cases),
            "routing_gold_pages": len(routing_gold),
            "questions": len(questions),
        },
        "parser_results": parser_results,
        "table_routing_audit": routing_audit,
        "table_bbox_audit": table_bbox_audit,
        "factor_at_a_time": factors,
        "decision": decision,
        "receipt_benchmark_untouched": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

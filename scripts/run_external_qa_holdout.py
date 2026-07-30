"""Run the frozen external end-to-end parsing/chunking/QA holdout."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.downstream_eval import evaluate_downstream
from src.complex_document.ir import SpatialDocument
from src.complex_document.parsers import ParseRequest, ParserUnavailable
from src.complex_document.qa_holdout import (
    definition_sha256,
    read_json,
    validate_qa_holdout,
)
from src.complex_document.router_holdout import verify_holdout_sources
from scripts.run_complex_benchmark import _adapter, _chunks

FACTOR_SPECS = [
    ("current_parser_fixed", "pymupdf", "fixed"),
    ("liteparse_fixed", "liteparse", "fixed"),
    ("liteparse_structure", "liteparse", "structure"),
    ("liteparse_table_structure", "liteparse-table", "structure"),
    ("hybrid_table_router", "hybrid-table-router", "hybrid-routed"),
    ("qwen3_vl_structure", "qwen3-vl", "structure"),
    ("targeted_vlm_fixed_posthoc", "targeted-vlm", "fixed"),
    ("targeted_vlm_structure", "targeted-vlm", "structure"),
]


def _parse_or_reuse(
    parser_key: str,
    manifest: dict,
    raw_dir: Path,
    store: ArtifactStore,
    *,
    reuse_ir: bool,
    parser_config: dict | None = None,
) -> tuple[dict[str, SpatialDocument], dict]:
    adapter = _adapter(parser_key)
    documents = {}
    latencies = []
    reused_documents = 0
    latency_sources: set[str] = set()
    for item in manifest["documents"]:
        ir_path = (
            store.paths.ir
            / item["document_id"]
            / adapter.name
            / "document.ir.json"
        )
        if reuse_ir and ir_path.is_file():
            document = SpatialDocument.from_json(ir_path.read_text(encoding="utf-8"))
            elapsed = 0.0
            reused_documents += 1
            raw_path = (
                store.paths.parser_raw
                / item["document_id"]
                / adapter.name
                / "raw.json"
            )
            if raw_path.is_file():
                raw_payload = read_json(raw_path)
                stored_page_latencies = [
                    page.get("latency_seconds")
                    for page in raw_payload.get("pages", [])
                    if isinstance(page, dict)
                    and isinstance(
                        page.get("latency_seconds"), (int, float)
                    )
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
                                    path=raw_dir / item["filename"],
                                    document_id=item["document_id"],
                                    source_uri=item["url"],
                                    pages=tuple(item["selected_pages"]),
                                    config={
                                        "benchmark_role": manifest["role"],
                                        "benchmark_version": manifest[
                                            "benchmark_version"
                                        ],
                                        **(parser_config or {}),
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
            started = time.perf_counter()
            document = adapter.parse(
                ParseRequest(
                    path=raw_dir / item["filename"],
                    document_id=item["document_id"],
                    source_uri=item["url"],
                    pages=tuple(item["selected_pages"]),
                    config={
                        "benchmark_role": manifest["role"],
                        "benchmark_version": manifest["benchmark_version"],
                        **(parser_config or {}),
                    },
                ),
                store,
            )
            elapsed = time.perf_counter() - started
            latency_sources.add("current_run_wall_clock")
        documents[item["document_id"]] = document
        latencies.append(elapsed)
    nonzero = [value for value in latencies if value > 0]
    page_count = sum(len(document.pages) for document in documents.values())
    return documents, {
        "parser_name": adapter.name,
        "parser_version": next(iter(documents.values())).parser.version,
        "documents": len(documents),
        "pages": page_count,
        "latency_seconds_total": round(sum(latencies), 6),
        "latency_seconds_median_document": round(
            statistics.median(latencies), 6
        ),
        "pages_per_second": (
            round(page_count / sum(nonzero), 6) if nonzero else None
        ),
        "estimated_api_cost_usd": (
            None if adapter.name == "llamaparse-cloud" else 0.0
        ),
        "billing_note": (
            "Cloud usage is billable; inspect the LlamaCloud project billing "
            "dashboard for the authoritative charge."
            if adapter.name == "llamaparse-cloud"
            else "Local parser; no parser API charge."
        ),
        "reused_documents": reused_documents,
        "latency_source": (
            "+".join(sorted(latency_sources))
            if latency_sources
            else "unavailable"
        ),
    }


def _decision(factors: list[dict]) -> dict:
    by_name = {factor["factor"]: factor for factor in factors}
    baseline = by_name.get("current_parser_fixed")
    hybrid = by_name.get("hybrid_table_router")
    if not baseline or not hybrid:
        return {
            "recommendation": "NO-GO",
            "reason": "Comparable baseline and hybrid factors are unavailable.",
        }
    if baseline["status"] != "completed" or hybrid["status"] != "completed":
        return {
            "recommendation": "NO-GO",
            "reason": "Comparable baseline and hybrid factors did not complete.",
        }
    base = baseline["downstream"]
    candidate = hybrid["downstream"]
    gates = {
        "recall_at_least_baseline": (
            candidate["retrieval_recall_at_k"]
            >= base["retrieval_recall_at_k"]
        ),
        "mrr_at_least_baseline": candidate["mrr"] >= base["mrr"],
        "answer_at_least_baseline": (
            candidate["answer_correctness"] >= base["answer_correctness"]
        ),
        "citation_at_least_baseline": (
            candidate["citation_validity"] >= base["citation_validity"]
        ),
        "all_15_human_questions_evaluated": (
            candidate["question_count"] == 15
        ),
    }
    result = {
        "recommendation": "GO" if all(gates.values()) else "NO-GO",
        "rule": (
            "Promote hybrid routing on external QA only if Recall@5, MRR, "
            "answer correctness, and citation validity all match or exceed "
            "the external current-parser baseline."
        ),
        "deltas_vs_baseline": {
            "retrieval_recall_at_k": round(
                candidate["retrieval_recall_at_k"]
                - base["retrieval_recall_at_k"],
                6,
            ),
            "mrr": round(candidate["mrr"] - base["mrr"], 6),
            "answer_correctness": round(
                candidate["answer_correctness"]
                - base["answer_correctness"],
                6,
            ),
            "citation_validity": round(
                candidate["citation_validity"]
                - base["citation_validity"],
                6,
            ),
        },
        "gates": gates,
    }
    targeted = by_name.get("targeted_vlm_structure")
    if targeted and targeted.get("status") == "completed":
        candidate = targeted["downstream"]
        deltas = {
            "retrieval_recall_at_k": (
                candidate["retrieval_recall_at_k"]
                - base["retrieval_recall_at_k"]
            ),
            "mrr": candidate["mrr"] - base["mrr"],
            "answer_correctness": (
                candidate["answer_correctness"]
                - base["answer_correctness"]
            ),
            "citation_validity": (
                candidate["citation_validity"]
                - base["citation_validity"]
            ),
        }
        targeted_gates = {
            "retrieval_recall_at_least_baseline": (
                deltas["retrieval_recall_at_k"] >= 0
            ),
            "mrr_at_least_baseline": deltas["mrr"] >= 0,
            "answer_at_least_baseline": deltas["answer_correctness"] >= 0,
            "citation_at_least_baseline": deltas["citation_validity"] >= 0,
            "one_metric_strictly_improves": any(
                value > 0 for value in deltas.values()
            ),
            "all_15_human_questions_evaluated": (
                candidate["question_count"] == 15
            ),
        }
        result["targeted_vlm"] = {
            "recommendation": (
                "GO" if all(targeted_gates.values()) else "NO-GO"
            ),
            "rule": (
                "Promote targeted VLM on the frozen external holdout only if "
                "no downstream metric regresses and at least one strictly "
                "improves."
            ),
            "deltas_vs_baseline": {
                key: round(value, 6) for key, value in deltas.items()
            },
            "gates": targeted_gates,
        }
    targeted_fixed = by_name.get("targeted_vlm_fixed_posthoc")
    if targeted_fixed and targeted_fixed.get("status") == "completed":
        candidate = targeted_fixed["downstream"]
        deltas = {
            "retrieval_recall_at_k": (
                candidate["retrieval_recall_at_k"]
                - base["retrieval_recall_at_k"]
            ),
            "mrr": candidate["mrr"] - base["mrr"],
            "answer_correctness": (
                candidate["answer_correctness"]
                - base["answer_correctness"]
            ),
            "citation_validity": (
                candidate["citation_validity"]
                - base["citation_validity"]
            ),
        }
        promising = all(value >= 0 for value in deltas.values()) and any(
            value > 0 for value in deltas.values()
        )
        result["targeted_vlm_fixed_posthoc"] = {
            "classification": (
                "PROMISING-NOT-VALIDATED" if promising else "NO-SIGNAL"
            ),
            "promotion_eligible": False,
            "reason": (
                "This fixed-chunk factor was added after observing the "
                "targeted-structure MRR drop. It diagnoses parser/chunker "
                "confounding but cannot convert the exposed holdout into GO."
            ),
            "required_next_step": (
                "Freeze the adapter, fixed chunker, and gates, then evaluate "
                "a new untouched document/question holdout."
            ),
            "deltas_vs_baseline": {
                key: round(value, 6) for key, value in deltas.items()
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/complex_document/qa_holdout/manifest.json"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/complex_document/qa_holdout/questions.json"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/complex_document/holdout/raw"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/complex_document/qa_holdout"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/complex_document/qa_holdout_summary.json"),
    )
    parser.add_argument(
        "--parsers",
        nargs="+",
        default=[
            "pymupdf",
            "liteparse",
            "liteparse-table",
            "hybrid-table-router",
        ],
    )
    parser.add_argument("--include-qwen", action="store_true")
    parser.add_argument("--reuse-ir", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    questions_payload = read_json(args.questions)
    validate_qa_holdout(manifest, questions_payload)
    source_verification = verify_holdout_sources(manifest, args.raw_dir)
    parser_keys = list(dict.fromkeys(args.parsers))
    if args.include_qwen:
        for parser_key in ("qwen3-vl", "targeted-vlm"):
            if parser_key not in parser_keys:
                parser_keys.append(parser_key)

    store = ArtifactStore(args.artifact_root)
    parsed = {}
    parser_results = {}
    for parser_key in parser_keys:
        try:
            documents, timing = _parse_or_reuse(
                parser_key,
                manifest,
                args.raw_dir,
                store,
                reuse_ir=args.reuse_ir,
            )
            parsed[parser_key] = documents
            parser_results[parser_key] = {
                "status": "completed",
                "timing_and_cost": timing,
            }
            print(
                f"{parser_key}: {timing['pages']} pages "
                f"in {timing['latency_seconds_total']:.3f}s"
            )
        except ParserUnavailable as exc:
            parser_results[parser_key] = {
                "status": "skipped",
                "reason": str(exc),
            }
            print(f"{parser_key}: SKIP {exc}")

    factors = []
    questions = questions_payload["questions"]
    for factor_name, parser_key, chunk_mode in FACTOR_SPECS:
        if parser_key not in parsed:
            factors.append(
                {
                    "factor": factor_name,
                    "parser": parser_key,
                    "chunking": chunk_mode,
                    "status": "skipped",
                    "reason": parser_results.get(parser_key, {}).get(
                        "reason", "parser not requested"
                    ),
                }
            )
            continue
        chunks = _chunks(parsed[parser_key], chunk_mode)
        factors.append(
            {
                "factor": factor_name,
                "parser": parser_key,
                "chunking": chunk_mode,
                "status": "completed",
                "chunk_count": len(chunks),
                **(
                    {
                        "analysis_role": (
                            "post-hoc diagnostic added after observing the "
                            "targeted-structure MRR drop; not promotion evidence"
                        )
                    }
                    if factor_name == "targeted_vlm_fixed_posthoc"
                    else {}
                ),
                "downstream": evaluate_downstream(questions, chunks, k=5),
            }
        )

    report = {
        "benchmark_version": manifest["benchmark_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition_sha256": definition_sha256(manifest, questions_payload),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device_policy": (
                "CPU factors plus guarded local Qwen3-VL GPU factor"
                if "qwen3-vl" in parsed
                else "CPU only; Qwen3-VL factor intentionally deferred"
            ),
        },
        "dataset": {
            "documents": len(manifest["documents"]),
            "selected_pages": sum(
                len(item["selected_pages"]) for item in manifest["documents"]
            ),
            "human_questions": len(questions),
            "question_types": sorted(
                {question["type"] for question in questions}
            ),
            "development_document_overlap": 0,
        },
        "source_verification": source_verification,
        "parser_results": parser_results,
        "factor_at_a_time": factors,
        "decision": _decision(factors),
        "receipt_benchmark_untouched": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

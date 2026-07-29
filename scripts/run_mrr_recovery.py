"""Tune late-max table retrieval on development QA, then test frozen external QA."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.chunking import hybrid_routed_chunks
from src.complex_document.downstream_eval import evaluate_downstream
from src.complex_document.ir import SpatialDocument
from src.complex_document.qa_holdout import read_json
from src.complex_document.retrieval import (
    CharNgramRetriever,
    WindowedCharNgramRetriever,
)

VARIANTS = [
    CharNgramRetriever(),
    WindowedCharNgramRetriever(
        window_size=240, overlap=80, global_weight=0.0
    ),
    WindowedCharNgramRetriever(
        window_size=240, overlap=80, global_weight=0.25
    ),
    WindowedCharNgramRetriever(
        window_size=320, overlap=80, global_weight=0.25
    ),
    WindowedCharNgramRetriever(
        window_size=480, overlap=120, global_weight=0.50
    ),
]


def _load_chunks(
    manifest: dict,
    artifact_root: Path,
    parser_name: str = "hybrid-table-router",
) -> list:
    chunks = []
    for item in manifest["documents"]:
        path = (
            artifact_root
            / "ir"
            / item["document_id"]
            / parser_name
            / "document.ir.json"
        )
        document = SpatialDocument.from_json(path.read_text(encoding="utf-8"))
        chunks.extend(hybrid_routed_chunks(document))
    return chunks


def _compact(evaluation: dict) -> dict:
    return {
        key: evaluation[key]
        for key in (
            "question_count",
            "retriever",
            "answerer",
            "k",
            "retrieval_recall_at_k",
            "mrr",
            "answer_correctness",
            "citation_validity",
            "error_attribution",
        )
    }


def main() -> None:
    development_manifest = read_json(
        Path("data/complex_document/manifest.json")
    )
    development_questions = read_json(
        Path("data/complex_document/questions.json")
    )["questions"]
    external_manifest = read_json(
        Path("data/complex_document/qa_holdout/manifest.json")
    )
    external_questions = read_json(
        Path("data/complex_document/qa_holdout/questions.json")
    )["questions"]
    main_summary = read_json(
        Path("results/complex_document/benchmark_summary.json")
    )
    external_summary = read_json(
        Path("results/complex_document/qa_holdout_summary.json")
    )

    development_chunks = _load_chunks(
        development_manifest, Path("artifacts/complex_document")
    )
    development_results = []
    for retriever in VARIANTS:
        evaluation = evaluate_downstream(
            development_questions,
            development_chunks,
            k=5,
            retriever=retriever,
        )
        development_results.append(
            {
                "retriever": retriever.name,
                "config": (
                    {
                        "window_size": retriever.window_size,
                        "overlap": retriever.overlap,
                        "global_weight": retriever.global_weight,
                        "coverage_weight": retriever.coverage_weight,
                    }
                    if isinstance(retriever, WindowedCharNgramRetriever)
                    else {"mode": "global-baseline"}
                ),
                "metrics": _compact(evaluation),
            }
        )

    current = next(
        item
        for item in main_summary["factor_at_a_time"]
        if item["factor"] == "3c_hybrid_table_page_router"
    )["downstream"]
    eligible = [
        result
        for result in development_results
        if result["metrics"]["answer_correctness"]
        >= current["answer_correctness"]
        and result["metrics"]["citation_validity"]
        >= current["citation_validity"]
    ]
    selected = max(
        eligible,
        key=lambda result: (
            result["metrics"]["mrr"],
            result["metrics"]["retrieval_recall_at_k"],
            result["retriever"],
        ),
    )
    selected_config = selected["config"]
    if selected_config.get("mode") == "global-baseline":
        selected_retriever = CharNgramRetriever()
    else:
        selected_retriever = WindowedCharNgramRetriever(**selected_config)

    external_chunks = _load_chunks(
        external_manifest,
        Path("artifacts/complex_document/qa_holdout"),
    )
    external_evaluation = evaluate_downstream(
        external_questions,
        external_chunks,
        k=5,
        retriever=selected_retriever,
    )
    external_baseline = next(
        item
        for item in external_summary["factor_at_a_time"]
        if item["factor"] == "current_parser_fixed"
    )["downstream"]
    external_hybrid = next(
        item
        for item in external_summary["factor_at_a_time"]
        if item["factor"] == "hybrid_table_router"
    )["downstream"]
    ranker_gates = {
        "mrr_strictly_improves_hybrid": (
            external_evaluation["mrr"] > external_hybrid["mrr"]
        ),
        "recall_not_lower_than_hybrid": (
            external_evaluation["retrieval_recall_at_k"]
            >= external_hybrid["retrieval_recall_at_k"]
        ),
        "answer_not_lower_than_hybrid": (
            external_evaluation["answer_correctness"]
            >= external_hybrid["answer_correctness"]
        ),
        "citation_not_lower_than_hybrid": (
            external_evaluation["citation_validity"]
            >= external_hybrid["citation_validity"]
        ),
    }
    promotion_gates = {
        "recall_at_least_current_parser": (
            external_evaluation["retrieval_recall_at_k"]
            >= external_baseline["retrieval_recall_at_k"]
        ),
        "mrr_at_least_current_parser": (
            external_evaluation["mrr"] >= external_baseline["mrr"]
        ),
        "answer_at_least_current_parser": (
            external_evaluation["answer_correctness"]
            >= external_baseline["answer_correctness"]
        ),
        "citation_at_least_current_parser": (
            external_evaluation["citation_validity"]
            >= external_baseline["citation_validity"]
        ),
    }
    report = {
        "experiment_version": "late-max-table-retrieval-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "selection_dataset": "five-document development benchmark",
            "selection_rule": (
                "Among predeclared variants that preserve development answer "
                "and citation validity, choose the highest development MRR."
            ),
            "external_dataset": "frozen 15-question QA holdout",
            "external_retuning_allowed": False,
            "feature_family": "same character-bigram cosine",
            "atomic_table_chunks_preserved": True,
        },
        "development_variants": development_results,
        "selected": selected,
        "external_evaluation": _compact(external_evaluation),
        "external_current_hybrid": _compact(external_hybrid),
        "external_questions": external_evaluation["questions"],
        "decision": {
            "ranker_for_hybrid_branch": {
                "recommendation": (
                    "GO" if all(ranker_gates.values()) else "NO-GO"
                ),
                "rule": (
                    "Adopt the ranker in the Hybrid research branch only if "
                    "external MRR strictly improves without reducing Hybrid "
                    "Recall@5, answer correctness, or citation validity."
                ),
                "gates": ranker_gates,
            },
            "full_hybrid_promotion": {
                "recommendation": (
                    "GO" if all(promotion_gates.values()) else "NO-GO"
                ),
                "rule": (
                    "Promote the full Hybrid pipeline only if it matches or "
                    "exceeds the external current-parser baseline on Recall@5, "
                    "MRR, answer correctness, and citation validity."
                ),
                "gates": promotion_gates,
            },
        },
        "receipt_benchmark_untouched": True,
    }
    output = Path("results/complex_document/mrr_recovery.json")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"selected {selected_retriever.name}")
    print(
        "development "
        f"MRR={selected['metrics']['mrr']:.3f}; "
        "external "
        f"MRR={external_evaluation['mrr']:.3f} "
        "ranker="
        f"{report['decision']['ranker_for_hybrid_branch']['recommendation']} "
        "promotion="
        f"{report['decision']['full_hybrid_promotion']['recommendation']}"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

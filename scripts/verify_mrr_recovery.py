"""Verify the selected MRR recovery variant from frozen normalized IR."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.downstream_eval import evaluate_downstream
from src.complex_document.qa_holdout import read_json
from src.complex_document.retrieval import (
    CharNgramRetriever,
    WindowedCharNgramRetriever,
)
from scripts.run_mrr_recovery import _compact, _load_chunks


def main() -> None:
    result = read_json(Path("results/complex_document/mrr_recovery.json"))
    config = result["selected"]["config"]
    retriever = (
        CharNgramRetriever()
        if config.get("mode") == "global-baseline"
        else WindowedCharNgramRetriever(**config)
    )
    manifest = read_json(
        Path("data/complex_document/qa_holdout/manifest.json")
    )
    questions = read_json(
        Path("data/complex_document/qa_holdout/questions.json")
    )["questions"]
    chunks = _load_chunks(
        manifest, Path("artifacts/complex_document/qa_holdout")
    )
    actual = evaluate_downstream(
        questions, chunks, k=5, retriever=retriever
    )
    if _compact(actual) != result["external_evaluation"]:
        raise SystemExit("MRR recovery external metrics do not match IR")
    if actual["questions"] != result["external_questions"]:
        raise SystemExit("MRR recovery question details do not match IR")
    if result.get("receipt_benchmark_untouched") is not True:
        raise SystemExit("receipt preservation marker is missing")
    print(
        f"verified MRR recovery: {retriever.name}, "
        f"external MRR={actual['mrr']:.3f}, "
        "ranker="
        f"{result['decision']['ranker_for_hybrid_branch']['recommendation']}, "
        "promotion="
        f"{result['decision']['full_hybrid_promotion']['recommendation']}"
    )


if __name__ == "__main__":
    main()

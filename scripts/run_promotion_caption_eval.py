"""Evaluate frozen caption-and-index crops on the v0.8 promotion sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_complex_benchmark import _caption_ablation, _chunks
from src.complex_document.qa_holdout import (
    load_qa_ir,
    read_json,
    validate_qa_holdout,
)


def _definition_sha256(
    manifest: dict, questions: dict, targets: dict
) -> str:
    payload = json.dumps(
        {
            "manifest": manifest,
            "questions": questions,
            "targets": targets,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decision(result: dict, protocol: dict) -> dict:
    if result.get("status") != "completed":
        return {
            "recommendation": "PENDING",
            "promotion_eligible": False,
            "reason": "Original-crop pixel synthesis has not completed.",
        }
    modes = result["modes"]
    baseline = modes["no_image_indexing"]
    candidate = modes["structured_caption_original_crop"]
    gates = {
        "structured_retrieval_at_least_no_image": (
            not protocol["structured_retrieval_must_not_regress"]
            or candidate["retrieval_recall_at_k"]
            >= baseline["retrieval_recall_at_k"]
        ),
        "pixel_synthesis_executed": (
            not protocol["pixel_synthesis_must_execute"]
            or bool(candidate.get("pixel_synthesis_executed"))
        ),
        "crop_recall_at_k": (
            (candidate.get("crop_recall_at_k") or 0)
            >= protocol["minimum_crop_recall_at_k"]
        ),
        "answer_correctness": (
            candidate["answer_correctness"]
            >= protocol["minimum_answer_correctness"]
        ),
        "citation_validity": (
            candidate["citation_validity"]
            >= protocol["minimum_citation_validity"]
        ),
        "frozen_question_count": (
            candidate["question_count"]
            >= protocol["minimum_question_count"]
        ),
    }
    return {
        "recommendation": "GO" if all(gates.values()) else "NO-GO",
        "promotion_eligible": True,
        "retrieval_recall_delta_vs_no_image": round(
            candidate["retrieval_recall_at_k"]
            - baseline["retrieval_recall_at_k"],
            6,
        ),
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/complex_document/promotion_holdout/manifest.json"
        ),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(
            "data/complex_document/promotion_holdout/questions.json"
        ),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "data/complex_document/promotion_holdout/chart_targets.json"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/complex_document/promotion_holdout"),
    )
    parser.add_argument(
        "--caption-artifact",
        type=Path,
        default=Path(
            "artifacts/complex_document/promotion_holdout/"
            "chart_captions/qwen3-vl.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/complex_document/promotion_caption_summary.json"
        ),
    )
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    questions = read_json(args.questions)
    targets = read_json(args.targets)
    validate_qa_holdout(manifest, questions)
    if targets.get("benchmark_version") != manifest["benchmark_version"]:
        raise SystemExit("caption target benchmark version mismatch")
    if targets.get("freeze_status") != (
        "frozen-before-caption-generation"
    ):
        raise SystemExit("caption targets are not frozen")

    documents = load_qa_ir(
        manifest, args.artifact_root, parser_name="pymupdf"
    )
    result = _caption_ablation(
        _chunks(documents, "fixed"),
        questions["questions"],
        args.caption_artifact,
    )
    report = {
        "benchmark_version": manifest["benchmark_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition_sha256": _definition_sha256(
            manifest, questions, targets
        ),
        "dataset": {
            "documents": len(manifest["documents"]),
            "visual_targets": len(targets["targets"]),
            "question_links": sum(
                len(target["question_ids"])
                for target in targets["targets"]
            ),
        },
        "factor": result,
        "decision": _decision(
            result, targets["evaluation_protocol"]
        ),
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

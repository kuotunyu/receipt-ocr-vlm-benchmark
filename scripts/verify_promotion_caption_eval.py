"""Verify the frozen v0.8 caption-and-index result from saved artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_complex_benchmark import _caption_ablation, _chunks
from scripts.run_promotion_caption_eval import (
    _decision,
    _definition_sha256,
)
from src.complex_document.qa_holdout import (
    load_qa_ir,
    read_json,
    validate_qa_holdout,
)


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
        "--result",
        type=Path,
        default=Path(
            "results/complex_document/promotion_caption_summary.json"
        ),
    )
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    questions = read_json(args.questions)
    targets = read_json(args.targets)
    report = read_json(args.result)
    validate_qa_holdout(manifest, questions)
    expected_hash = _definition_sha256(manifest, questions, targets)
    if report.get("definition_sha256") != expected_hash:
        raise SystemExit("promotion caption definition hash mismatch")
    documents = load_qa_ir(
        manifest, args.artifact_root, parser_name="pymupdf"
    )
    actual = _caption_ablation(
        _chunks(documents, "fixed"),
        questions["questions"],
        args.caption_artifact,
    )
    if actual != report["factor"]:
        raise SystemExit("promotion caption factor mismatch")
    if _decision(
        actual,
        targets["evaluation_protocol"],
        promotion_eligible=(
            manifest.get("role") != "external-scale-validation"
        ),
    ) != report["decision"]:
        raise SystemExit("promotion caption decision mismatch")
    if report.get("receipt_benchmark_untouched") is not True:
        raise SystemExit("receipt preservation marker is missing")
    print(
        "verified promotion caption holdout: "
        f"{report['dataset']['visual_targets']} targets, "
        f"{report['dataset']['question_links']} question links"
    )


if __name__ == "__main__":
    main()

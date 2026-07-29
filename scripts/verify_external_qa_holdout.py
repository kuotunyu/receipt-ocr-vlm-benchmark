"""Recompute the external downstream QA holdout from normalized IR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.downstream_eval import evaluate_downstream
from src.complex_document.qa_holdout import (
    definition_sha256,
    load_qa_ir,
    read_json,
    validate_qa_holdout,
)
from src.complex_document.router_holdout import verify_holdout_sources
from scripts.run_complex_benchmark import _chunks


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
        "--result",
        type=Path,
        default=Path("results/complex_document/qa_holdout_summary.json"),
    )
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    questions_payload = read_json(args.questions)
    result = read_json(args.result)
    validate_qa_holdout(manifest, questions_payload)
    if result["definition_sha256"] != definition_sha256(
        manifest, questions_payload
    ):
        raise SystemExit("external QA definition hash mismatch")
    if result["source_verification"] != verify_holdout_sources(
        manifest, args.raw_dir
    ):
        raise SystemExit("external QA source verification mismatch")

    checked = 0
    for factor in result["factor_at_a_time"]:
        if factor.get("status") != "completed":
            continue
        parser_name = result["parser_results"][factor["parser"]][
            "timing_and_cost"
        ]["parser_name"]
        documents = load_qa_ir(
            manifest, args.artifact_root, parser_name
        )
        actual = evaluate_downstream(
            questions_payload["questions"],
            _chunks(documents, factor["chunking"]),
            k=5,
        )
        if actual != factor["downstream"]:
            raise SystemExit(
                f"external QA factor mismatch: {factor['factor']}"
            )
        checked += 1
    if result.get("receipt_benchmark_untouched") is not True:
        raise SystemExit("receipt preservation marker is missing")
    print(
        f"verified external QA holdout: {checked} factors, "
        f"{len(questions_payload['questions'])} human questions"
    )


if __name__ == "__main__":
    main()

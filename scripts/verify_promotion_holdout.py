"""Recompute the fresh promotion holdout from pinned sources and IR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_complex_benchmark import _chunks
from src.complex_document.downstream_eval import evaluate_downstream
from src.complex_document.promotion_holdout import (
    promotion_decision,
    promotion_definition_sha256,
    validate_promotion_protocol,
)
from src.complex_document.qa_holdout import (
    load_qa_ir,
    read_json,
    validate_qa_holdout,
)
from src.complex_document.router_holdout import verify_holdout_sources


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
        "--protocol",
        type=Path,
        default=Path(
            "data/complex_document/promotion_holdout/protocol.json"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/complex_document/promotion_holdout/raw"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/complex_document/promotion_holdout"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "results/complex_document/promotion_holdout_summary.json"
        ),
    )
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    questions_payload = read_json(args.questions)
    protocol = read_json(args.protocol)
    result = read_json(args.result)
    validate_qa_holdout(manifest, questions_payload)
    validate_promotion_protocol(protocol, manifest, questions_payload)
    if result["definition_sha256"] != promotion_definition_sha256(
        manifest, questions_payload, protocol
    ):
        raise SystemExit("promotion holdout definition hash mismatch")
    if result["source_verification"] != verify_holdout_sources(
        manifest, args.raw_dir
    ):
        raise SystemExit("promotion holdout source verification mismatch")

    completed = {}
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
        if factor["parser"] == "targeted-vlm":
            expected_router = protocol["candidate"]["router_version"]
            if {
                document.parser.config.get("router")
                for document in documents.values()
            } != {expected_router}:
                raise SystemExit("frozen targeted-VLM router mismatch")
        actual = evaluate_downstream(
            questions_payload["questions"],
            _chunks(documents, factor["chunking"]),
            k=int(protocol["retrieval_k"]),
        )
        if actual != factor["downstream"]:
            raise SystemExit(
                f"promotion factor mismatch: {factor['factor']}"
            )
        completed[factor["factor"]] = actual
        checked += 1

    expected_decision = promotion_decision(
        completed.get("current_parser_fixed"),
        completed.get("targeted_vlm_fixed_frozen"),
        protocol,
    )
    if result["decision"] != expected_decision:
        raise SystemExit("promotion decision mismatch")
    if result.get("receipt_benchmark_untouched") is not True:
        raise SystemExit("receipt preservation marker is missing")
    print(
        f"verified promotion holdout: {checked} factors, "
        f"{len(questions_payload['questions'])} frozen questions"
    )


if __name__ == "__main__":
    main()

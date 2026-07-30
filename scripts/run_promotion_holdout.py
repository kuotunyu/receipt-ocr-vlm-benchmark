"""Run the fresh, frozen targeted-VLM + fixed-chunk promotion holdout."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_complex_benchmark import _chunks
from scripts.run_external_qa_holdout import _parse_or_reuse
from src.complex_document.artifacts import ArtifactStore
from src.complex_document.downstream_eval import evaluate_downstream
from src.complex_document.parsers import ParserUnavailable
from src.complex_document.promotion_holdout import (
    promotion_decision,
    promotion_definition_sha256,
    validate_promotion_protocol,
)
from src.complex_document.qa_holdout import read_json, validate_qa_holdout
from src.complex_document.router_holdout import verify_holdout_sources

FACTOR_SPECS = (
    ("current_parser_fixed", "pymupdf", "fixed"),
    ("targeted_vlm_fixed_frozen", "targeted-vlm", "fixed"),
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
        "--output",
        type=Path,
        default=Path(
            "results/complex_document/promotion_holdout_summary.json"
        ),
    )
    parser.add_argument(
        "--include-candidate",
        action="store_true",
        help="Run the frozen targeted-VLM factor; this normally requires GPU.",
    )
    parser.add_argument("--reuse-ir", action="store_true")
    parser.add_argument(
        "--reparse-baseline",
        action="store_true",
        help=(
            "Measure a fresh CPU baseline while reusing candidate IR; useful "
            "because PyMuPDF raw artifacts do not store page latencies."
        ),
    )
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    questions_payload = read_json(args.questions)
    protocol = read_json(args.protocol)
    validate_qa_holdout(manifest, questions_payload)
    validate_promotion_protocol(protocol, manifest, questions_payload)
    source_verification = verify_holdout_sources(manifest, args.raw_dir)

    parser_keys = ["pymupdf"]
    if args.include_candidate:
        parser_keys.append("targeted-vlm")
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
                reuse_ir=(
                    args.reuse_ir
                    and not (
                        args.reparse_baseline and parser_key == "pymupdf"
                    )
                ),
            )
            if parser_key == "targeted-vlm":
                expected_router = protocol["candidate"]["router_version"]
                actual_routers = {
                    document.parser.config.get("router")
                    for document in documents.values()
                }
                if actual_routers != {expected_router}:
                    raise RuntimeError(
                        "candidate router does not match frozen protocol: "
                        f"{sorted(str(value) for value in actual_routers)}"
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
    completed = {}
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
                        "reason", "GPU candidate intentionally deferred"
                    ),
                }
            )
            continue
        chunks = _chunks(parsed[parser_key], chunk_mode)
        downstream = evaluate_downstream(
            questions,
            chunks,
            k=int(protocol["retrieval_k"]),
        )
        completed[factor_name] = downstream
        factors.append(
            {
                "factor": factor_name,
                "parser": parser_key,
                "chunking": chunk_mode,
                "status": "completed",
                "chunk_count": len(chunks),
                "downstream": downstream,
            }
        )

    report = {
        "benchmark_version": manifest["benchmark_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition_sha256": promotion_definition_sha256(
            manifest, questions_payload, protocol
        ),
        "protocol": protocol,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device_policy": (
                "CPU baseline plus frozen local targeted-VLM candidate"
                if args.include_candidate
                else "CPU baseline only; candidate intentionally deferred"
            ),
        },
        "dataset": {
            "documents": len(manifest["documents"]),
            "selected_pages": sum(
                len(item["selected_pages"])
                for item in manifest["documents"]
            ),
            "manual_questions": len(questions),
            "question_types": sorted(
                {question["type"] for question in questions}
            ),
            "development_document_overlap": 0,
        },
        "source_verification": source_verification,
        "parser_results": parser_results,
        "factor_at_a_time": factors,
        "decision": promotion_decision(
            completed.get("current_parser_fixed"),
            completed.get("targeted_vlm_fixed_frozen"),
            protocol,
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

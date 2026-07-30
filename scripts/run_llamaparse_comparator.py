"""Run the optional LlamaParse cloud comparator on the scale-validation set.

The command is safe by default: cloud upload requires both --allow-cloud and
LLAMA_CLOUD_API_KEY.  Missing credentials or the optional SDK produce a
recorded skip with exit code zero.
"""

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
    promotion_definition_sha256,
    validate_promotion_protocol,
)
from src.complex_document.qa_holdout import read_json, validate_qa_holdout
from src.complex_document.router_holdout import verify_holdout_sources


PRIMARY_METRICS = (
    "retrieval_recall_at_k",
    "mrr",
    "answer_correctness",
    "citation_validity",
)


def _decision(baseline: dict | None, candidate: dict | None) -> dict:
    if baseline is None or candidate is None:
        return {
            "recommendation": "PENDING",
            "promotion_eligible": False,
            "reason": "The optional cloud comparator did not complete.",
        }
    deltas = {
        metric: round(candidate[metric] - baseline[metric], 6)
        for metric in PRIMARY_METRICS
    }
    return {
        "recommendation": "DESCRIPTIVE-ONLY",
        "promotion_eligible": False,
        "scale_finding": (
            "MATCHES-OR-IMPROVES-LOCAL-BASELINE"
            if all(value >= 0 for value in deltas.values())
            else "REGRESSES-ON-AT-LEAST-ONE-METRIC"
        ),
        "deltas_vs_current_parser_fixed": deltas,
        "reason": (
            "LlamaParse is an optional commercial comparator and cannot replace "
            "the locally reproducible promotion path."
        ),
    }


def _cached_cloud_ir_complete(
    manifest: dict, artifact_root: Path
) -> bool:
    return all(
        (
            artifact_root
            / "ir"
            / item["document_id"]
            / "llamaparse-cloud"
            / "document.ir.json"
        ).is_file()
        for item in manifest["documents"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/complex_document/scale_validation/manifest.json"
        ),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(
            "data/complex_document/scale_validation/questions.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "data/complex_document/scale_validation/protocol.json"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/complex_document/scale_validation/raw"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "artifacts/complex_document/scale_validation_llamaparse"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/complex_document/"
            "scale_validation_llamaparse_summary.json"
        ),
    )
    parser.add_argument(
        "--allow-cloud",
        action="store_true",
        help=(
            "Acknowledge that selected public PDFs will be uploaded to a "
            "billable third-party cloud parser."
        ),
    )
    parser.add_argument("--reuse-ir", action="store_true")
    parser.add_argument("--tier", default="agentic")
    parser.add_argument(
        "--parser-version",
        default="2026-07-15",
        help="Pin the LlamaParse backend version; avoid 'latest' for results.",
    )
    parser.add_argument("--disable-cache", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    questions_payload = read_json(args.questions)
    protocol = read_json(args.protocol)
    validate_qa_holdout(manifest, questions_payload)
    validate_promotion_protocol(protocol, manifest, questions_payload)
    source_verification = verify_holdout_sources(manifest, args.raw_dir)

    store = ArtifactStore(args.artifact_root)
    parsed = {}
    parser_results = {}

    baseline_documents, baseline_timing = _parse_or_reuse(
        "pymupdf",
        manifest,
        args.raw_dir,
        store,
        reuse_ir=args.reuse_ir,
    )
    parsed["pymupdf"] = baseline_documents
    parser_results["pymupdf"] = {
        "status": "completed",
        "timing_and_cost": baseline_timing,
    }

    cached_cloud_ir = (
        args.reuse_ir
        and _cached_cloud_ir_complete(manifest, args.artifact_root)
    )
    if not args.allow_cloud and not cached_cloud_ir:
        parser_results["llamaparse"] = {
            "status": "skipped",
            "reason": (
                "cloud upload disabled and complete cached IR unavailable; "
                "rerun with --allow-cloud after reviewing data handling and billing"
            ),
        }
    else:
        try:
            cloud_documents, cloud_timing = _parse_or_reuse(
                "llamaparse",
                manifest,
                args.raw_dir,
                store,
                reuse_ir=args.reuse_ir,
                parser_config={
                    "tier": args.tier,
                    "version": args.parser_version,
                    "disable_cache": args.disable_cache,
                },
            )
            parsed["llamaparse"] = cloud_documents
            parser_results["llamaparse"] = {
                "status": "completed",
                "timing_and_cost": cloud_timing,
            }
        except ParserUnavailable as exc:
            parser_results["llamaparse"] = {
                "status": "skipped",
                "reason": str(exc),
            }

    factor_specs = (
        ("current_parser_fixed", "pymupdf", "fixed"),
        ("llamaparse_fixed", "llamaparse", "fixed"),
        ("llamaparse_structure", "llamaparse", "structure"),
    )
    factors = []
    completed = {}
    for factor_name, parser_key, chunk_mode in factor_specs:
        if parser_key not in parsed:
            factors.append(
                {
                    "factor": factor_name,
                    "parser": parser_key,
                    "chunking": chunk_mode,
                    "status": "skipped",
                    "reason": parser_results[parser_key]["reason"],
                }
            )
            continue
        chunks = _chunks(parsed[parser_key], chunk_mode)
        downstream = evaluate_downstream(
            questions_payload["questions"],
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
        "role": "optional-commercial-parser-comparator",
        "definition_sha256": promotion_definition_sha256(
            manifest, questions_payload, protocol
        ),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cloud_upload_authorized": args.allow_cloud,
        },
        "dataset": {
            "documents": len(manifest["documents"]),
            "selected_pages": sum(
                len(item["selected_pages"])
                for item in manifest["documents"]
            ),
            "manual_questions": len(questions_payload["questions"]),
        },
        "source_verification": source_verification,
        "parser_results": parser_results,
        "factor_at_a_time": factors,
        "decision": _decision(
            completed.get("current_parser_fixed"),
            completed.get("llamaparse_fixed"),
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

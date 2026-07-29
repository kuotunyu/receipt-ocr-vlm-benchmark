"""Run the frozen CPU-only table-router evaluation on external documents."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.parsers import ParseRequest, PyMuPDFAdapter
from src.complex_document.router_holdout import (
    evaluate_router_holdout,
    read_json,
    validate_holdout_definition,
    verify_holdout_sources,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/complex_document/holdout/manifest.json"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path(
            "data/complex_document/holdout/gold/table_routing_pages.json"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/complex_document/holdout/raw"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/complex_document/holdout"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/complex_document/table_router_holdout.json"),
    )
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    gold = read_json(args.gold)
    validate_holdout_definition(manifest, gold)
    source_verification = verify_holdout_sources(manifest, args.raw_dir)

    adapter = PyMuPDFAdapter()
    store = ArtifactStore(args.artifact_root)
    documents = {}
    per_document_seconds = {}
    started = perf_counter()
    for item in manifest["documents"]:
        document_id = item["document_id"]
        document_started = perf_counter()
        documents[document_id] = adapter.parse(
            ParseRequest(
                path=args.raw_dir / item["filename"],
                document_id=document_id,
                source_uri=item["url"],
                pages=tuple(item["selected_pages"]),
                config={
                    "benchmark_role": manifest["role"],
                    "router_threshold_frozen": manifest["router_frozen"][
                        "threshold"
                    ],
                },
            ),
            store,
        )
        per_document_seconds[document_id] = round(
            perf_counter() - document_started, 6
        )
    total_seconds = round(perf_counter() - started, 6)

    evaluation = evaluate_router_holdout(documents, gold["pages"])
    parser_version = next(iter(documents.values())).parser.version
    report = {
        "benchmark_version": manifest["benchmark_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device_policy": "CPU only; no GPU model or external API used",
        },
        "dataset": {
            "documents": len(manifest["documents"]),
            "selected_pages": len(gold["pages"]),
            "positive_pages": sum(
                bool(item["should_route"]) for item in gold["pages"]
            ),
            "negative_pages": sum(
                not bool(item["should_route"]) for item in gold["pages"]
            ),
            "development_document_overlap": 0,
        },
        "source_verification": source_verification,
        "parser": {
            "name": adapter.name,
            "version": parser_version,
            "render_dpi": adapter.render_dpi,
            "detect_tables": adapter.detect_tables,
            "per_document_seconds": per_document_seconds,
            "total_seconds": total_seconds,
        },
        "evaluation": evaluation,
        "receipt_benchmark_untouched": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        "primary: "
        f"precision={evaluation['primary']['precision']:.3f} "
        f"recall={evaluation['primary']['recall']:.3f} "
        f"f1={evaluation['primary']['f1']:.3f} "
        f"accuracy={evaluation['primary']['accuracy']:.3f} "
        f"{evaluation['validation']['recommendation']}"
    )


if __name__ == "__main__":
    main()

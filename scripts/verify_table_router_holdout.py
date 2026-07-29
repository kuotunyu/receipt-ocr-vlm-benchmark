"""Verify the checked-in holdout result from checksummed PDFs and normalized IR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.router_holdout import (
    evaluate_router_holdout,
    load_holdout_ir,
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
        "--result",
        type=Path,
        default=Path("results/complex_document/table_router_holdout.json"),
    )
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    gold = read_json(args.gold)
    result = read_json(args.result)
    validate_holdout_definition(manifest, gold)
    verified_sources = verify_holdout_sources(manifest, args.raw_dir)
    documents = load_holdout_ir(manifest, args.artifact_root)
    recomputed = evaluate_router_holdout(documents, gold["pages"])

    if result["benchmark_version"] != manifest["benchmark_version"]:
        raise SystemExit("result benchmark version does not match manifest")
    if result["source_verification"] != verified_sources:
        raise SystemExit("result source verification does not match local PDFs")
    if result["evaluation"] != recomputed:
        raise SystemExit("result routing metrics do not match normalized IR")
    if result.get("receipt_benchmark_untouched") is not True:
        raise SystemExit("receipt preservation marker is missing")
    print(
        "verified table-router holdout: "
        f"{recomputed['primary']['page_count']} pages, "
        f"threshold={recomputed['protocol']['primary_threshold']}, "
        f"{recomputed['validation']['recommendation']}"
    )


if __name__ == "__main__":
    main()

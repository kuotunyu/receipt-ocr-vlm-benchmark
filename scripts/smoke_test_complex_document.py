"""One-page integration smoke for every locally available parser adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.parsers import (
    HybridTableRouterAdapter,
    LiteParseAdapter,
    LiteParseTableAdapter,
    PaddleLayoutAdapter,
    ParseRequest,
    ParserUnavailable,
    PyMuPDFAdapter,
    QwenVLMParserAdapter,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paddle", action="store_true", help="include the slower CPU Paddle smoke"
    )
    args = parser.parse_args()
    manifest = json.loads(
        Path("data/complex_document/manifest.json").read_text(encoding="utf-8")
    )
    item = next(
        document
        for document in manifest["documents"]
        if document["document_id"] == "ndc_sdg_policy_2022"
    )
    request = ParseRequest(
        path=Path("data/complex_document/raw") / item["filename"],
        document_id="complex-smoke",
        source_uri=item["url"],
        pages=(2,),
    )
    adapters = [
        PyMuPDFAdapter(),
        LiteParseAdapter(),
        LiteParseTableAdapter(),
        HybridTableRouterAdapter(),
        QwenVLMParserAdapter(),
    ]
    if args.paddle:
        adapters.insert(2, PaddleLayoutAdapter(dpi=100))
    store = ArtifactStore()
    for adapter in adapters:
        try:
            result = adapter.parse(request, store)
            print(
                f"PASS {adapter.name}: pages={len(result.pages)} "
                f"elements={len(result.all_elements())} version={adapter.version()}"
            )
        except ParserUnavailable as exc:
            print(f"SKIP {adapter.name}: {exc}")


if __name__ == "__main__":
    main()

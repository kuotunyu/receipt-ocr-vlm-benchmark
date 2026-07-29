"""Overlay human gold and parser bboxes for a reproducible failure case."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.complex_document.ir import SpatialDocument


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="arc-05")
    parser.add_argument("--parser", default="liteparse")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/complex_document/failures/arc-05-liteparse.png"),
    )
    args = parser.parse_args()

    import fitz

    manifest = json.loads(
        Path("data/complex_document/manifest.json").read_text(encoding="utf-8")
    )
    cases = json.loads(
        Path("data/complex_document/gold/hard_cases.json").read_text(encoding="utf-8")
    )["cases"]
    case = next(item for item in cases if item["case_id"] == args.case_id)
    if "bbox_normalized" not in case:
        raise SystemExit("visualization requires a case with bbox_normalized")
    document_info = next(
        item
        for item in manifest["documents"]
        if item["document_id"] == case["document_id"]
    )
    ir_path = (
        Path("artifacts/complex_document/ir")
        / case["document_id"]
        / args.parser
        / "document.ir.json"
    )
    if not ir_path.is_file():
        raise FileNotFoundError(
            f"{ir_path} missing; run scripts/run_complex_benchmark.py first"
        )
    document = SpatialDocument.from_json(ir_path.read_text(encoding="utf-8"))
    page_number = case["pages"][0]
    ir_page = next(page for page in document.pages if page.page_number == page_number)
    target_text = case.get("target_text") or case.get("anchor")
    target = compact(target_text)
    candidates = [
        element
        for element in ir_page.elements
        if element.bbox
        and target in compact(element.text)
        and (
            case["dimension"] != "table_structure"
            or element.element_type == "table"
        )
    ]

    source_path = Path("data/complex_document/raw") / document_info["filename"]
    with fitz.open(source_path) as pdf:
        page = pdf[page_number - 1]
        gx0, gy0, gx1, gy1 = case["bbox_normalized"]
        gold_rect = fitz.Rect(
            gx0 * page.rect.width,
            gy0 * page.rect.height,
            gx1 * page.rect.width,
            gy1 * page.rect.height,
        )
        page.draw_rect(gold_rect, color=(0, 0.8, 0), width=3, overlay=True)
        page.insert_text(
            (gold_rect.x0, max(10, gold_rect.y0 - 4)),
            "GOLD",
            fontsize=9,
            color=(0, 0.6, 0),
            overlay=True,
        )
        predicted_rects = []
        for element in candidates:
            bbox = element.bbox
            if bbox.coordinate_space == "normalized":
                rect = fitz.Rect(
                    bbox.x0 * page.rect.width,
                    bbox.y0 * page.rect.height,
                    bbox.x1 * page.rect.width,
                    bbox.y1 * page.rect.height,
                )
            else:
                rect = fitz.Rect(
                    bbox.x0 / ir_page.width * page.rect.width,
                    bbox.y0 / ir_page.height * page.rect.height,
                    bbox.x1 / ir_page.width * page.rect.width,
                    bbox.y1 / ir_page.height * page.rect.height,
                )
            predicted_rects.append(list(rect))
            page.draw_rect(rect, color=(1, 0, 0), width=2, overlay=True)
            page.insert_text(
                (rect.x0, min(page.rect.height - 5, rect.y1 + 10)),
                "PRED",
                fontsize=9,
                color=(0.8, 0, 0),
                overlay=True,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(dpi=160, alpha=False).save(args.output)

    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "case_id": args.case_id,
                "parser": args.parser,
                "document_id": case["document_id"],
                "page": page_number,
                "target_text": target_text,
                "gold_bbox_normalized": case["bbox_normalized"],
                "prediction_count": len(candidates),
                "predicted_rects_pdf_points": predicted_rects,
                "interpretation": (
                    "Green is human gold; red is parser prediction. "
                    "No red box means the target was omitted or text-corrupted."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} with {len(candidates)} matching parser bbox(es)"
    )


if __name__ == "__main__":
    main()

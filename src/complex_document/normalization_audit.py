"""Measure parser-native to Spatial IR normalization loss separately."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from src.complex_document.ir import SpatialDocument


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _native_payload_text(payload: dict) -> tuple[str, int]:
    if payload.get("text"):
        pages = payload.get("pages", [])
        item_count = sum(len(page.get("text_items", [])) for page in pages)
        return str(payload["text"]), item_count
    if payload.get("base_ir_before_reconstruction"):
        base = SpatialDocument.from_dict(payload["base_ir_before_reconstruction"])
        return base.plain_text(), len(base.all_elements())
    if payload.get("base_ir_before_routing"):
        base = SpatialDocument.from_dict(payload["base_ir_before_routing"])
        return base.plain_text(), len(base.all_elements())
    pages = payload.get("pages", [])
    page_texts = []
    item_count = 0
    for page in pages:
        if page.get("text"):
            page_texts.append(str(page["text"]))
            item_count += len(page.get("text_items", []))
            continue
        if page.get("response"):
            try:
                response = json.loads(str(page["response"]))
            except json.JSONDecodeError:
                response = {}
            elements = response.get("elements", [])
            if isinstance(elements, list):
                page_texts.append(
                    "\n".join(
                        str(element.get("text") or element.get("markdown") or "")
                        for element in elements
                        if isinstance(element, dict)
                    )
                )
                item_count += sum(
                    isinstance(element, dict) for element in elements
                )
            continue
        if "ocr_boxes" in page:
            boxes = [
                box
                for box in page.get("ocr_boxes", [])
                if isinstance(box, dict) and box.get("text")
            ]
            page_texts.append(
                "\n".join(str(box["text"]) for box in boxes)
            )
            item_count += len(boxes)
            continue
        page_texts.append(
            "\n".join(
                str(block.get("text", ""))
                for block in page.get("blocks", [])
                if block.get("text")
            )
        )
        item_count += len(page.get("text_items", page.get("blocks", [])))
    text = "\n".join(page_texts)
    return text, item_count


def _counter_overlap(source: str, target: str) -> int:
    source_counter = Counter(source)
    target_counter = Counter(target)
    return sum(
        min(count, target_counter.get(character, 0))
        for character, count in source_counter.items()
    )


def audit_normalization(
    native_payload: dict, document: SpatialDocument
) -> dict:
    native_text, native_item_count = _native_payload_text(native_payload)
    ir_text = document.plain_text()
    native_compact = _compact(native_text)
    ir_compact = _compact(ir_text)
    overlap = _counter_overlap(native_compact, ir_compact)
    type_counts = Counter(
        element.element_type for element in document.all_elements()
    )
    shadowed = sum(
        bool(element.metadata.get("shadowed_by_reconstructed_tables"))
        for element in document.all_elements()
    )
    return {
        "native_text_characters": len(native_compact),
        "ir_text_characters": len(ir_compact),
        "native_item_count": native_item_count,
        "ir_element_count": len(document.all_elements()),
        "text_character_recall": round(
            overlap / len(native_compact), 6
        )
        if native_compact
        else 1.0,
        "text_character_precision": round(
            overlap / len(ir_compact), 6
        )
        if ir_compact
        else 1.0,
        "ir_element_types": dict(sorted(type_counts.items())),
        "shadowed_native_elements": shadowed,
    }


def audit_artifact(
    raw_path: str | Path, document: SpatialDocument
) -> dict:
    payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    return audit_normalization(payload, document)


def aggregate_audits(audits: list[dict]) -> dict:
    if not audits:
        return {"documents": 0}
    return {
        "documents": len(audits),
        "mean_text_character_recall": round(
            sum(audit["text_character_recall"] for audit in audits)
            / len(audits),
            6,
        ),
        "mean_text_character_precision": round(
            sum(audit["text_character_precision"] for audit in audits)
            / len(audits),
            6,
        ),
        "native_items": sum(audit["native_item_count"] for audit in audits),
        "ir_elements": sum(audit["ir_element_count"] for audit in audits),
        "shadowed_native_elements": sum(
            audit["shadowed_native_elements"] for audit in audits
        ),
        "documents_detail": audits,
    }

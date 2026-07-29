"""Deterministic page routing and human-gold audit for local table enrichment."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.complex_document.ir import BBox, Element, Page, SpatialDocument
from src.complex_document.table_reconstruction import find_table_caption

DEFAULT_TABLE_ROUTE_THRESHOLD = 0.62
TABLE_ROUTER_VERSION = "vector-grid-router-1"


@dataclass(frozen=True)
class TableCandidateScore:
    element_id: str
    confidence: float
    row_count: int
    column_count: int
    populated_row_ratio: float
    filled_cell_ratio: float
    width_ratio: float
    height_ratio: float
    explicit_caption: bool


@dataclass(frozen=True)
class PageRouteDecision:
    document_id: str
    page_number: int
    should_route: bool
    threshold: float
    max_confidence: float
    candidates: list[TableCandidateScore]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["candidates"] = [asdict(item) for item in self.candidates]
        return value


def _candidate_score(element: Element, page: Page) -> TableCandidateScore:
    row_count = int(element.metadata.get("row_count", 0))
    column_count = int(element.metadata.get("column_count", 0))
    rows = [
        [cell.strip() for cell in line.split("\t")]
        for line in element.text.splitlines()
        if line.strip()
    ]
    populated_rows = sum(
        sum(bool(cell) for cell in row) >= 2 for row in rows
    )
    nonempty_cells = sum(bool(cell) for row in rows for cell in row)
    populated_row_ratio = populated_rows / max(row_count, 1)
    filled_cell_ratio = nonempty_cells / max(row_count * column_count, 1)
    bbox = element.bbox
    width_ratio = (
        (bbox.x1 - bbox.x0) / page.width if bbox is not None else 0.0
    )
    height_ratio = (
        (bbox.y1 - bbox.y0) / page.height if bbox is not None else 0.0
    )
    caption_elements = (
        find_table_caption(page.elements, bbox) if bbox is not None else []
    )
    explicit_caption = any(
        candidate.text.strip().startswith("表")
        or candidate.element_type == "caption"
        for candidate in caption_elements
    )
    confidence = (
        0.25 * min(row_count / 6, 1)
        + 0.15 * min(column_count / 4, 1)
        + 0.25 * min(populated_row_ratio, 1)
        + 0.15 * min(filled_cell_ratio / 0.5, 1)
        + 0.10 * float(width_ratio >= 0.25 and height_ratio >= 0.025)
        + 0.10 * float(explicit_caption)
    )
    return TableCandidateScore(
        element_id=element.element_id,
        confidence=round(confidence, 6),
        row_count=row_count,
        column_count=column_count,
        populated_row_ratio=round(populated_row_ratio, 6),
        filled_cell_ratio=round(filled_cell_ratio, 6),
        width_ratio=round(width_ratio, 6),
        height_ratio=round(height_ratio, 6),
        explicit_caption=explicit_caption,
    )


def score_page_for_table_routing(
    document_id: str,
    page: Page,
    *,
    threshold: float = DEFAULT_TABLE_ROUTE_THRESHOLD,
) -> PageRouteDecision:
    candidates = [
        _candidate_score(element, page)
        for element in page.elements
        if element.element_type == "table"
        and int(element.metadata.get("row_count", 0)) >= 2
        and int(element.metadata.get("column_count", 0)) >= 2
    ]
    max_confidence = max(
        (candidate.confidence for candidate in candidates), default=0.0
    )
    return PageRouteDecision(
        document_id=document_id,
        page_number=page.page_number,
        should_route=max_confidence >= threshold,
        threshold=threshold,
        max_confidence=round(max_confidence, 6),
        candidates=candidates,
    )


def route_document_pages(
    document: SpatialDocument,
    *,
    threshold: float = DEFAULT_TABLE_ROUTE_THRESHOLD,
) -> list[PageRouteDecision]:
    return [
        score_page_for_table_routing(
            document.document.document_id,
            page,
            threshold=threshold,
        )
        for page in document.pages
    ]


def evaluate_table_router(
    documents: dict[str, SpatialDocument],
    gold_pages: list[dict],
    *,
    threshold: float = DEFAULT_TABLE_ROUTE_THRESHOLD,
) -> dict:
    decisions = {
        (decision.document_id, decision.page_number): decision
        for document in documents.values()
        for decision in route_document_pages(document, threshold=threshold)
    }
    cases = []
    for item in gold_pages:
        key = (item["document_id"], int(item["page"]))
        decision = decisions[key]
        expected = bool(item["should_route"])
        predicted = decision.should_route
        cases.append(
            {
                "document_id": key[0],
                "page": key[1],
                "expected_route": expected,
                "predicted_route": predicted,
                "correct": expected == predicted,
                "max_confidence": decision.max_confidence,
                "reason": item["reason"],
            }
        )
    true_positive = sum(
        item["expected_route"] and item["predicted_route"] for item in cases
    )
    false_positive = sum(
        not item["expected_route"] and item["predicted_route"] for item in cases
    )
    false_negative = sum(
        item["expected_route"] and not item["predicted_route"] for item in cases
    )
    true_negative = sum(
        not item["expected_route"] and not item["predicted_route"] for item in cases
    )
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "router": TABLE_ROUTER_VERSION,
        "threshold": threshold,
        "page_count": len(cases),
        "positive_pages": sum(item["expected_route"] for item in cases),
        "predicted_positive_pages": sum(
            item["predicted_route"] for item in cases
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            6,
        ),
        "accuracy": round(
            (true_positive + true_negative) / len(cases), 6
        )
        if cases
        else None,
        "cases": cases,
    }


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _normalized_bbox(bbox: BBox, page: Page) -> tuple[float, float, float, float]:
    if bbox.coordinate_space == "normalized":
        return bbox.x0, bbox.y0, bbox.x1, bbox.y1
    return (
        bbox.x0 / page.width,
        bbox.y0 / page.height,
        bbox.x1 / page.width,
        bbox.y1 / page.height,
    )


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(
        0.0, left[3] - left[1]
    )
    right_area = max(0.0, right[2] - right[0]) * max(
        0.0, right[3] - right[1]
    )
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def evaluate_reconstructed_table_bboxes(
    documents: dict[str, SpatialDocument], cases: list[dict]
) -> dict:
    """Audit table-region IoU on human cases that include a table bbox."""
    results = []
    for case in cases:
        if (
            case.get("dimension") != "table_structure"
            or "bbox_normalized" not in case
            or case["document_id"] not in documents
        ):
            continue
        document = documents[case["document_id"]]
        pages = set(case.get("pages", []))
        anchor = _compact(case["anchor"])
        expected_cells = [
            _compact(value) for value in case.get("expected_cells", [])
        ]
        scores = []
        for page in document.pages:
            if pages and page.page_number not in pages:
                continue
            for element in page.elements:
                element_text = _compact(
                    f"{element.text}\n{element.markdown}"
                )
                nearby_caption = (
                    _compact(
                        " ".join(
                            item.text
                            for item in find_table_caption(
                                page.elements, element.bbox
                            )
                        )
                    )
                    if element.bbox is not None
                    else ""
                )
                cell_matches = sum(
                    value in element_text for value in expected_cells
                )
                matches_case = (
                    anchor in element_text
                    or anchor in nearby_caption
                    or (
                        expected_cells
                        and cell_matches >= min(2, len(expected_cells))
                    )
                )
                if (
                    element.element_type != "table"
                    or element.bbox is None
                    or element.metadata.get(
                        "shadowed_by_reconstructed_tables"
                    )
                    or not matches_case
                ):
                    continue
                scores.append(
                    _iou(
                        _normalized_bbox(element.bbox, page),
                        tuple(float(value) for value in case["bbox_normalized"]),
                    )
                )
        results.append(
            {
                "case_id": case["case_id"],
                "score": round(max(scores, default=0.0), 6),
                "candidate_count": len(scores),
            }
        )
    return {
        "case_count": len(results),
        "mean_iou": round(
            sum(item["score"] for item in results) / len(results), 6
        )
        if results
        else None,
        "cases": results,
    }

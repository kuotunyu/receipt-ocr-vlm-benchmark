from __future__ import annotations

import re
from dataclasses import dataclass

from src.complex_document.ir import BBox, Element, Page, SpatialDocument


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _contains(text: str, term: str) -> bool:
    return _compact(term) in _compact(text)


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
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


@dataclass(frozen=True)
class ParserCaseResult:
    case_id: str
    dimension: str
    score: float
    passed: bool
    detail: str


def _page_elements(document: SpatialDocument, page_numbers: list[int] | None) -> list[Element]:
    if not page_numbers:
        return document.all_elements()
    selected = set(page_numbers)
    return [
        element
        for element in document.all_elements()
        if element.page_number in selected
    ]


def evaluate_parser_case(
    document: SpatialDocument, case: dict, pass_threshold: float = 0.999
) -> ParserCaseResult:
    dimension = case["dimension"]
    elements = _page_elements(document, case.get("pages"))
    full_text = "\n".join(element.text for element in elements)
    score = 0.0
    detail = ""

    if dimension in {"text_completeness", "chart_data"}:
        terms = case["expected_terms"]
        found = [term for term in terms if _contains(full_text, term)]
        score = len(found) / len(terms) if terms else 1.0
        detail = f"found {len(found)}/{len(terms)} terms"

    elif dimension == "reading_order":
        ordered_terms = case["ordered_terms"]
        compact = _compact(full_text)
        positions = [compact.find(_compact(term)) for term in ordered_terms]
        comparisons = [
            positions[left] >= 0
            and positions[right] >= 0
            and positions[left] < positions[right]
            for left in range(len(positions))
            for right in range(left + 1, len(positions))
        ]
        score = sum(comparisons) / len(comparisons) if comparisons else 1.0
        detail = f"pairwise order {sum(comparisons)}/{len(comparisons)}"

    elif dimension == "heading_list_style":
        expected = case["expected_elements"]
        matches = [
            any(
                element.element_type == item["type"]
                and _contains(element.text, item["text"])
                for element in elements
            )
            for item in expected
        ]
        score = sum(matches) / len(matches) if matches else 1.0
        detail = f"typed elements {sum(matches)}/{len(matches)}"

    elif dimension == "footnote_anchor":
        anchor = case["anchor"]
        footnote = case["footnote"]
        compact = _compact(full_text)
        anchor_position = compact.find(_compact(anchor))
        footnote_position = compact.find(_compact(footnote))
        typed = any(
            element.element_type == "footnote" and _contains(element.text, footnote)
            for element in elements
        )
        checks = [
            anchor_position >= 0,
            footnote_position >= 0,
            anchor_position >= 0
            and footnote_position >= 0
            and anchor_position < footnote_position,
            typed,
        ]
        score = sum(checks) / len(checks)
        detail = f"anchor/footnote checks {sum(checks)}/{len(checks)}"

    elif dimension == "table_structure":
        anchor = case["anchor"]
        candidates = [
            element
            for element in elements
            if element.element_type == "table"
            and (_contains(element.text, anchor) or _contains(element.markdown, anchor))
        ]
        if candidates:
            table = candidates[0]
            expected_cells = case.get("expected_cells", [])
            cells_found = sum(
                _contains(table.markdown, cell) or _contains(table.text, cell)
                for cell in expected_cells
            )
            row_ok = int(table.metadata.get("row_count", 0)) >= case.get(
                "minimum_rows", 1
            )
            column_ok = int(table.metadata.get("column_count", 0)) >= case.get(
                "minimum_columns", 1
            )
            checks = [bool(row_ok), bool(column_ok)] + [
                _contains(table.markdown, cell) or _contains(table.text, cell)
                for cell in expected_cells
            ]
            score = sum(checks) / len(checks)
            detail = (
                f"rows={table.metadata.get('row_count')} "
                f"cols={table.metadata.get('column_count')} "
                f"cells={cells_found}/{len(expected_cells)}"
            )
        else:
            detail = "no matching table element"

    elif dimension == "bbox_grounding":
        target = case["target_text"]
        expected_bbox = tuple(float(value) for value in case["bbox_normalized"])
        candidates = [
            element for element in elements if element.bbox and _contains(element.text, target)
        ]
        scores = []
        for element in candidates:
            page = next(
                page
                for page in document.pages
                if page.page_number == element.page_number
            )
            scores.append(_iou(_normalized_bbox(element.bbox, page), expected_bbox))
        score = max(scores, default=0.0)
        detail = f"best IoU={score:.3f} across {len(candidates)} candidates"
        pass_threshold = float(case.get("iou_threshold", 0.5))

    elif dimension == "header_footer_contamination":
        term = case["repeated_text"]
        count = _compact(full_text).count(_compact(term))
        allowed = int(case.get("max_occurrences", 0))
        score = 1.0 if count <= allowed else max(0.0, allowed / count)
        detail = f"occurrences={count}, allowed={allowed}"

    elif dimension == "cross_page_continuity":
        left = case["left_term"]
        right = case["right_term"]
        left_pages = [
            element.page_number for element in elements if _contains(element.text, left)
        ]
        right_pages = [
            element.page_number for element in elements if _contains(element.text, right)
        ]
        consecutive = any(
            right_page == left_page + 1
            for left_page in left_pages
            for right_page in right_pages
        )
        checks = [bool(left_pages), bool(right_pages), consecutive]
        if case.get("requires_link"):
            continuity_groups: dict[str, set[int]] = {}
            for element in elements:
                continuity_id = element.metadata.get("continuity_id")
                if continuity_id:
                    continuity_groups.setdefault(continuity_id, set()).add(
                        element.page_number
                    )
            linked = any(
                any(right_page == left_page + 1 for left_page in page_set for right_page in page_set)
                for page_set in continuity_groups.values()
            )
            checks.append(linked)
        score = sum(checks) / len(checks)
        detail = (
            f"left_pages={left_pages}, right_pages={right_pages}, "
            f"linked={checks[-1] if case.get('requires_link') else 'not-required'}"
        )

    elif dimension == "visual_element":
        expected_type = case.get("expected_type", "figure")
        candidates = [
            element for element in elements if element.element_type == expected_type
        ]
        count_ok = len(candidates) >= int(case.get("minimum_count", 1))
        bbox_ok = (
            any(element.bbox is not None for element in candidates)
            if case.get("require_bbox")
            else True
        )
        checks = [count_ok, bbox_ok]
        score = sum(checks) / len(checks)
        detail = (
            f"{expected_type} count={len(candidates)}, "
            f"bbox_present={any(element.bbox is not None for element in candidates)}"
        )

    else:
        raise ValueError(f"unsupported parser metric dimension: {dimension}")

    return ParserCaseResult(
        case_id=case["case_id"],
        dimension=dimension,
        score=round(float(score), 6),
        passed=score >= pass_threshold,
        detail=detail,
    )


def evaluate_parser(
    documents: dict[str, SpatialDocument], cases: list[dict]
) -> dict:
    results = [
        evaluate_parser_case(documents[case["document_id"]], case)
        for case in cases
        if case["document_id"] in documents
    ]
    dimensions = sorted({result.dimension for result in results})
    dimension_scores = {
        dimension: round(
            sum(result.score for result in results if result.dimension == dimension)
            / sum(result.dimension == dimension for result in results),
            6,
        )
        for dimension in dimensions
    }
    return {
        "case_count": len(results),
        "mean_score": round(
            sum(result.score for result in results) / len(results), 6
        )
        if results
        else None,
        "pass_rate": round(
            sum(result.passed for result in results) / len(results), 6
        )
        if results
        else None,
        "dimensions": dimension_scores,
        "cases": [result.__dict__ for result in results],
    }

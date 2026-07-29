"""Local grid-table reconstruction and cross-page continuity linking."""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable

from src.complex_document.ir import BBox, Element, Page

_TABLE_CAPTION_RE = re.compile(r"^\s*表")


def table_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return ""
    padded = [row + [""] * (width - len(row)) for row in rows]
    return "\n".join(
        [
            "| " + " | ".join(padded[0]) + " |",
            "| " + " | ".join(["---"] * width) + " |",
            *("| " + " | ".join(row) + " |" for row in padded[1:]),
        ]
    )


def clean_rows(rows: Iterable[Iterable[object]]) -> list[list[str]]:
    return [
        [
            "" if cell is None else " ".join(str(cell).replace("|", "｜").split())
            for cell in row
        ]
        for row in rows
    ]


def bbox_contains_center(container: BBox, candidate: BBox) -> bool:
    center_x = (candidate.x0 + candidate.x1) / 2
    center_y = (candidate.y0 + candidate.y1) / 2
    return (
        container.x0 <= center_x <= container.x1
        and container.y0 <= center_y <= container.y1
    )


def _intersection_area(left: BBox, right: BBox) -> float:
    return max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0)) * max(
        0.0, min(left.y1, right.y1) - max(left.y0, right.y0)
    )


def trim_nested_spanning_tail(
    *,
    table_index: int,
    bbox: BBox,
    rows: list[list[str]],
    row_bboxes: list[tuple[float, float, float, float]],
    row_nonempty_cell_counts: list[int],
    all_table_bboxes: list[BBox],
) -> tuple[BBox, list[list[str]], dict]:
    """Trim a giant tail row when it geometrically swallows another table.

    PyMuPDF can return both the real second table and a first-table candidate
    whose final one-cell row extends across the prose and the second table.
    We only trim when an independently detected table is almost fully inside
    that oversized spanning row, so ordinary merged cells remain untouched.
    """
    if len(rows) != len(row_bboxes) or len(rows) != len(
        row_nonempty_cell_counts
    ):
        return bbox, rows, {"applied": False, "reason": "row metadata mismatch"}

    children = []
    for other_index, other in enumerate(all_table_bboxes):
        if other_index == table_index:
            continue
        other_area = max(0.0, other.x1 - other.x0) * max(
            0.0, other.y1 - other.y0
        )
        contained = (
            _intersection_area(bbox, other) / other_area
            if other_area
            else 0.0
        )
        if contained >= 0.95 and other.y0 > bbox.y0:
            children.append((other.y0, other_index))
    if not children or len(rows) < 3:
        return bbox, rows, {"applied": False}

    child_y0, child_index = min(children)
    prior_heights = [
        max(0.0, row[3] - row[1])
        for row in row_bboxes[:-1]
        if row[3] > row[1]
    ]
    typical_height = statistics.median(prior_heights) if prior_heights else 0
    for row_index, row_bbox in enumerate(row_bboxes):
        row_height = max(0.0, row_bbox[3] - row_bbox[1])
        contains_child_start = row_bbox[1] <= child_y0 <= row_bbox[3]
        oversized = typical_height > 0 and row_height >= typical_height * 3
        spanning = row_nonempty_cell_counts[row_index] <= 1
        if (
            contains_child_start
            and oversized
            and spanning
            and row_index >= 2
        ):
            trimmed_bbox = BBox(
                bbox.x0,
                bbox.y0,
                bbox.x1,
                row_bboxes[row_index - 1][3],
                bbox.coordinate_space,
            )
            return (
                trimmed_bbox,
                rows[:row_index],
                {
                    "applied": True,
                    "reason": "nested-table-inside-oversized-spanning-tail",
                    "removed_rows": len(rows) - row_index,
                    "nested_table_index": child_index,
                    "original_bbox": [
                        bbox.x0,
                        bbox.y0,
                        bbox.x1,
                        bbox.y1,
                    ],
                },
            )
    return bbox, rows, {"applied": False}


def find_table_caption(
    elements: list[Element], table_bbox: BBox, max_vertical_gap: float = 48
) -> list[Element]:
    candidates = []
    for element in elements:
        if not element.bbox or not element.text.strip():
            continue
        horizontal_overlap = max(
            0.0,
            min(table_bbox.x1, element.bbox.x1)
            - max(table_bbox.x0, element.bbox.x0),
        )
        gap = table_bbox.y0 - element.bbox.y1
        table_center = (table_bbox.x0 + table_bbox.x1) / 2
        element_center = (element.bbox.x0 + element.bbox.x1) / 2
        centered = abs(table_center - element_center) <= (
            table_bbox.x1 - table_bbox.x0
        ) * 0.25
        caption_like = (
            element.element_type == "caption"
            or bool(_TABLE_CAPTION_RE.search(element.text))
            or (centered and len(element.text.strip()) <= 100)
        )
        if (
            caption_like
            and horizontal_overlap > 0
            and -5 <= gap <= max_vertical_gap
        ):
            candidates.append((gap, element))
    if not candidates:
        return []
    seed = min(candidates, key=lambda value: value[0])[1]
    seed_y = (seed.bbox.y0 + seed.bbox.y1) / 2
    same_line = [
        element
        for element in elements
        if element.bbox
        and abs((element.bbox.y0 + element.bbox.y1) / 2 - seed_y) <= 8
        and element.bbox.x1 >= table_bbox.x0
        and element.bbox.x0 <= table_bbox.x1
        and len(element.text.strip()) <= 140
    ]
    same_line.sort(key=lambda element: element.bbox.x0)
    return same_line


def _normalized_geometry(element: Element, page: Page) -> tuple[float, float, float, float]:
    bbox = element.bbox
    if bbox is None:
        return 0, 0, 0, 0
    if bbox.coordinate_space == "normalized":
        return bbox.x0, bbox.y0, bbox.x1, bbox.y1
    return (
        bbox.x0 / page.width,
        bbox.y0 / page.height,
        bbox.x1 / page.width,
        bbox.y1 / page.height,
    )


def link_cross_page_tables(document_id: str, pages: list[Page]) -> int:
    """Attach continuity metadata without collapsing page-level provenance."""
    pages_by_number = {page.page_number: page for page in pages}
    link_count = 0
    for page_number in sorted(pages_by_number):
        next_page = pages_by_number.get(page_number + 1)
        if next_page is None:
            continue
        page = pages_by_number[page_number]
        current_tables = [
            element for element in page.elements if element.element_type == "table"
        ]
        next_tables = [
            element for element in next_page.elements if element.element_type == "table"
        ]
        if not current_tables or not next_tables:
            continue
        current = max(
            current_tables,
            key=lambda element: _normalized_geometry(element, page)[3],
        )
        following = min(
            next_tables,
            key=lambda element: _normalized_geometry(element, next_page)[1],
        )
        current_geometry = _normalized_geometry(current, page)
        next_geometry = _normalized_geometry(following, next_page)
        current_columns = int(current.metadata.get("column_count", 0))
        next_columns = int(following.metadata.get("column_count", 0))
        column_delta = abs(current_columns - next_columns) / max(
            current_columns, next_columns, 1
        )
        x_aligned = (
            abs(current_geometry[0] - next_geometry[0]) <= 0.08
            and abs(current_geometry[2] - next_geometry[2]) <= 0.08
        )
        edge_aligned = current_geometry[3] >= 0.75 and next_geometry[1] <= 0.2
        following_has_new_caption = bool(following.metadata.get("caption"))
        if (
            not x_aligned
            or not edge_aligned
            or column_delta > 0.3
            or following_has_new_caption
        ):
            continue
        continuity_id = (
            current.metadata.get("continuity_id")
            or f"{document_id}:table-continuity:p{page_number:04d}"
        )
        current.metadata.update(
            {
                "continuity_id": continuity_id,
                "continues_to_page": next_page.page_number,
            }
        )
        following.metadata.update(
            {
                "continuity_id": continuity_id,
                "continued_from_page": page.page_number,
            }
        )
        link_count += 1
    return link_count

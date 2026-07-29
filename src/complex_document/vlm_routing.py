"""Deterministic routing for conservative page-level VLM enrichment."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.complex_document.ir import BBox, Page, SpatialDocument

VLM_ROUTER_VERSION = "native-visual-router-1"
MAX_NATIVE_TEXT_CHARS = 700
MIN_VISUAL_AREA_RATIO = 0.15
MAX_SCANNED_TEXT_CHARS = 80
MIN_SCANNED_AREA_RATIO = 0.50


@dataclass(frozen=True)
class VLMPageRouteDecision:
    document_id: str
    page_number: int
    should_route: bool
    mode: str
    reason: str
    native_text_characters: int
    figure_count: int
    table_count: int
    max_figure_area_ratio: float

    def to_dict(self) -> dict:
        return asdict(self)


def _area_ratio(bbox: BBox | None, page: Page) -> float:
    if bbox is None:
        return 0.0
    if bbox.coordinate_space == "normalized":
        width = max(0.0, min(1.0, bbox.x1) - max(0.0, bbox.x0))
        height = max(0.0, min(1.0, bbox.y1) - max(0.0, bbox.y0))
        return width * height
    width = max(0.0, min(page.width, bbox.x1) - max(0.0, bbox.x0))
    height = max(0.0, min(page.height, bbox.y1) - max(0.0, bbox.y0))
    return width * height / (page.width * page.height)


def route_page_for_vlm(document_id: str, page: Page) -> VLMPageRouteDecision:
    """Route only scan-like or clearly raster-visual pages.

    Thresholds are intentionally frozen from native PyMuPDF signals. They do
    not inspect questions, gold answers, VLM output, or holdout performance.
    """
    native_text = re.sub(
        r"\s+",
        "",
        "\n".join(
            element.text
            for element in page.elements
            if element.element_type != "figure"
        ),
    )
    figures = [
        element for element in page.elements if element.element_type == "figure"
    ]
    table_count = sum(
        element.element_type == "table" for element in page.elements
    )
    max_figure_area = max(
        (_area_ratio(element.bbox, page) for element in figures),
        default=0.0,
    )

    mode = "none"
    reason = "native text/structure is sufficient"
    if (
        len(native_text) < MAX_SCANNED_TEXT_CHARS
        and max_figure_area >= MIN_SCANNED_AREA_RATIO
    ):
        mode = "replace"
        reason = "scan-like page with near-full-page raster and little native text"
    elif (
        len(native_text) <= MAX_NATIVE_TEXT_CHARS
        and max_figure_area >= MIN_VISUAL_AREA_RATIO
        and table_count == 0
    ):
        mode = "enrich"
        reason = "raster-visual page with limited native text and no native table"

    return VLMPageRouteDecision(
        document_id=document_id,
        page_number=page.page_number,
        should_route=mode != "none",
        mode=mode,
        reason=reason,
        native_text_characters=len(native_text),
        figure_count=len(figures),
        table_count=table_count,
        max_figure_area_ratio=round(max_figure_area, 6),
    )


def route_document_for_vlm(
    document: SpatialDocument,
) -> list[VLMPageRouteDecision]:
    return [
        route_page_for_vlm(document.document.document_id, page)
        for page in document.pages
    ]

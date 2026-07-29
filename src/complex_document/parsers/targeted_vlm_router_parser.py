from __future__ import annotations

import copy
import time

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.ir import BBox, Page, ParserMetadata, SpatialDocument
from src.complex_document.parsers.base import DocumentParserAdapter, ParseRequest
from src.complex_document.parsers.pymupdf_parser import PyMuPDFAdapter
from src.complex_document.parsers.vlm_parser import QwenVLMParserAdapter
from src.complex_document.vlm_routing import (
    VLM_ROUTER_VERSION,
    VLMPageRouteDecision,
    route_document_for_vlm,
)


class TargetedVLMRouterAdapter(DocumentParserAdapter):
    """PyMuPDF default with conservative Qwen enrichment on visual pages."""

    name = "targeted-vlm-router"

    def __init__(
        self,
        *,
        baseline: DocumentParserAdapter | None = None,
        vlm: DocumentParserAdapter | None = None,
        render_dpi: int = 144,
    ):
        self.baseline = baseline or PyMuPDFAdapter(render_dpi=render_dpi)
        self.vlm = vlm or QwenVLMParserAdapter()
        self.render_dpi = render_dpi

    def version(self) -> str:
        return (
            f"{VLM_ROUTER_VERSION}+{self.baseline.name}-{self.baseline.version()}"
            f"+{self.vlm.name}-{self.vlm.version()}"
        )

    @staticmethod
    def _to_page_coordinates(bbox: BBox | None, page: Page) -> BBox | None:
        if bbox is None or bbox.coordinate_space != "normalized":
            return copy.deepcopy(bbox)
        return BBox(
            bbox.x0 * page.width,
            bbox.y0 * page.height,
            bbox.x1 * page.width,
            bbox.y1 * page.height,
            coordinate_space=page.coordinate_space,
        )

    @staticmethod
    def _nearby_visual_title(page: Page, bbox: BBox | None) -> str:
        if bbox is None:
            return ""
        candidates = [
            element
            for element in page.elements
            if element.bbox is not None
            and element.element_type in {"heading", "caption", "paragraph"}
            and element.text.strip()
            and len(element.text.strip()) <= 200
            and element.bbox.y1 <= bbox.y0 + page.height * 0.03
        ]
        if not candidates:
            return ""
        return max(candidates, key=lambda item: item.bbox.y1).text.strip()

    def _enrich_page(
        self,
        baseline_page: Page,
        vlm_page: Page,
        decision: VLMPageRouteDecision,
    ) -> Page:
        page = copy.deepcopy(baseline_page)
        for element in page.elements:
            element.metadata["vlm_router_selected"] = False
            element.metadata["routed_source_parser"] = self.baseline.name

        visual_elements = [
            copy.deepcopy(element)
            for element in vlm_page.elements
            if element.element_type in {"figure", "caption", "table"}
            and (element.text.strip() or element.markdown.strip())
        ]
        for element in visual_elements:
            element.bbox = self._to_page_coordinates(element.bbox, page)
            title = self._nearby_visual_title(page, element.bbox)
            if title and element.element_type in {"figure", "table"}:
                element.text = f"{title}\n{element.text}".strip()
                element.markdown = f"{title}\n\n{element.markdown}".strip()
                element.metadata["nearby_native_title"] = title
            element.metadata["vlm_router_selected"] = True
            element.metadata["routed_source_parser"] = self.vlm.name
            element.metadata["routing_mode"] = decision.mode
            element.metadata["routing_reason"] = decision.reason
            page.elements.append(element)

        page.elements.sort(
            key=lambda element: (
                element.bbox.y0 if element.bbox is not None else float("inf"),
                0 if element.metadata.get("vlm_router_selected") else 1,
                element.bbox.x0 if element.bbox is not None else float("inf"),
            )
        )
        for order, element in enumerate(page.elements):
            element.reading_order = order
            element.element_id = f"p{page.page_number}-e{order:04d}"
        return page

    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        baseline_started = time.perf_counter()
        baseline = self.baseline.parse(request, artifacts=None)
        baseline_wall_seconds = time.perf_counter() - baseline_started
        decisions = route_document_for_vlm(baseline)
        routed_pages = tuple(
            decision.page_number
            for decision in decisions
            if decision.should_route
        )
        decision_by_page = {
            decision.page_number: decision for decision in decisions
        }

        vlm_document = None
        vlm_wall_seconds = 0.0
        if routed_pages:
            started = time.perf_counter()
            vlm_document = self.vlm.parse(
                ParseRequest(
                    path=request.path,
                    document_id=request.document_id,
                    source_uri=request.source_uri,
                    pages=routed_pages,
                    config=request.config,
                ),
                artifacts=None,
            )
            vlm_wall_seconds = time.perf_counter() - started
        vlm_pages = (
            {page.page_number: page for page in vlm_document.pages}
            if vlm_document is not None
            else {}
        )

        pages: list[Page] = []
        for baseline_page in baseline.pages:
            decision = decision_by_page[baseline_page.page_number]
            if decision.mode == "replace":
                page = copy.deepcopy(vlm_pages[baseline_page.page_number])
                for element in page.elements:
                    element.metadata["vlm_router_selected"] = True
                    element.metadata["routed_source_parser"] = self.vlm.name
                    element.metadata["routing_mode"] = decision.mode
                    element.metadata["routing_reason"] = decision.reason
            elif decision.mode == "enrich":
                page = self._enrich_page(
                    baseline_page,
                    vlm_pages[baseline_page.page_number],
                    decision,
                )
            else:
                page = copy.deepcopy(baseline_page)
                for element in page.elements:
                    element.metadata["vlm_router_selected"] = False
                    element.metadata["routed_source_parser"] = self.baseline.name
            pages.append(page)

        if artifacts:
            try:
                import fitz

                with fitz.open(request.path) as pdf:
                    for page in pages:
                        screenshot_path = (
                            artifacts.screenshot_dir(
                                request.document_id, self.name
                            )
                            / f"page-{page.page_number:04d}.png"
                        )
                        pdf[page.page_number - 1].get_pixmap(
                            dpi=self.render_dpi, alpha=False
                        ).save(screenshot_path)
                        screenshot_ref = str(screenshot_path.as_posix())
                        page.screenshot_ref = screenshot_ref
                        for element in page.elements:
                            element.source_image_ref = screenshot_ref
            except ImportError:
                pass

        result = SpatialDocument(
            schema_version=baseline.schema_version,
            document=baseline.document,
            parser=ParserMetadata(
                name=self.name,
                version=self.version(),
                config={
                    "router": VLM_ROUTER_VERSION,
                    "routed_pages": list(routed_pages),
                    "route_modes": {
                        str(decision.page_number): decision.mode
                        for decision in decisions
                        if decision.should_route
                    },
                    "default_parser": self.baseline.name,
                    "routed_parser": self.vlm.name,
                    "routing_granularity": "page-replace-or-visual-enrich",
                    "render_dpi": self.render_dpi,
                    **request.config,
                },
            ),
            pages=pages,
        )
        if artifacts:
            allocated_latency = (
                vlm_wall_seconds / len(routed_pages) if routed_pages else 0.0
            )
            artifacts.write_parser_raw(
                request.document_id,
                self.name,
                {
                    "base_ir_before_routing": baseline.to_dict(),
                    "routed_vlm_ir": (
                        vlm_document.to_dict()
                        if vlm_document is not None
                        else None
                    ),
                    "route_decisions": [
                        decision.to_dict() for decision in decisions
                    ],
                    "routed_pages": list(routed_pages),
                    "baseline_wall_seconds": baseline_wall_seconds,
                    "vlm_wall_seconds": vlm_wall_seconds,
                    "pages": [
                        {
                            "page_number": page_number,
                            "latency_seconds": allocated_latency,
                            "latency_source": "batch_wall_clock_evenly_allocated",
                        }
                        for page_number in routed_pages
                    ],
                },
            )
            artifacts.write_ir(result)
        return result

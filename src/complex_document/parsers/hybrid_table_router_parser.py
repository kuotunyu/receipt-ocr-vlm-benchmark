from __future__ import annotations

import copy

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.ir import Page, ParserMetadata, SpatialDocument
from src.complex_document.parsers.base import DocumentParserAdapter, ParseRequest
from src.complex_document.parsers.liteparse_table_parser import (
    LiteParseTableAdapter,
)
from src.complex_document.parsers.pymupdf_parser import PyMuPDFAdapter
from src.complex_document.routing import (
    DEFAULT_TABLE_ROUTE_THRESHOLD,
    TABLE_ROUTER_VERSION,
    route_document_pages,
)
from src.complex_document.table_reconstruction import bbox_contains_center


class HybridTableRouterAdapter(DocumentParserAdapter):
    """PyMuPDF default with selective LiteParse table-region enrichment."""

    name = "hybrid-table-router"

    def __init__(
        self,
        *,
        route_threshold: float = DEFAULT_TABLE_ROUTE_THRESHOLD,
        render_dpi: int = 144,
    ):
        self.route_threshold = route_threshold
        self.render_dpi = render_dpi
        self.baseline = PyMuPDFAdapter(render_dpi=render_dpi)
        self.enriched = LiteParseTableAdapter(render_dpi=render_dpi)

    def version(self) -> str:
        return (
            f"{TABLE_ROUTER_VERSION}+pymupdf-{self.baseline.version()}"
            f"+{self.enriched.version()}"
        )

    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        baseline = self.baseline.parse(request, artifacts=None)
        decisions = route_document_pages(
            baseline, threshold=self.route_threshold
        )
        routed_pages = tuple(
            decision.page_number
            for decision in decisions
            if decision.should_route
        )
        enriched_pages: dict[int, Page] = {}
        if routed_pages:
            enriched = self.enriched.parse(
                ParseRequest(
                    path=request.path,
                    document_id=request.document_id,
                    source_uri=request.source_uri,
                    pages=routed_pages,
                    config=request.config,
                ),
                artifacts=None,
            )
            enriched_pages = {
                page.page_number: page for page in enriched.pages
            }

        pages = []
        for baseline_page in baseline.pages:
            routed = baseline_page.page_number in enriched_pages
            page = copy.deepcopy(baseline_page)
            for element in page.elements:
                element.metadata["table_router_selected"] = False
                element.metadata["routed_source_parser"] = self.baseline.name
            if routed:
                baseline_elements = list(page.elements)
                reconstructed_tables = [
                    copy.deepcopy(element)
                    for element in enriched_pages[
                        baseline_page.page_number
                    ].elements
                    if element.element_type == "table"
                ]
                for table_index, table in enumerate(reconstructed_tables):
                    table.metadata["table_router_selected"] = True
                    table.metadata["routed_source_parser"] = self.enriched.name
                    table.metadata["hybrid_region_enrichment"] = True
                    for element in baseline_elements:
                        if not element.bbox or not table.bbox:
                            continue
                        if (
                            element.element_type == "table"
                            or bbox_contains_center(table.bbox, element.bbox)
                        ):
                            element.metadata.setdefault(
                                "shadowed_by_reconstructed_tables", []
                            ).append(table_index)
                    page.elements.append(table)
                page.elements.sort(
                    key=lambda element: (
                        element.bbox.y0
                        if element.bbox is not None
                        else float("inf"),
                        0 if element.element_type == "table" else 1,
                        element.bbox.x0
                        if element.bbox is not None
                        else float("inf"),
                    )
                )
                for order, element in enumerate(page.elements):
                    element.reading_order = order
                    element.element_id = (
                        f"p{page.page_number}-e{order:04d}"
                    )
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
                    "router": TABLE_ROUTER_VERSION,
                    "route_threshold": self.route_threshold,
                    "routed_pages": list(routed_pages),
                    "default_parser": self.baseline.name,
                    "routed_parser": self.enriched.name,
                    "routing_granularity": "page-detect/table-region-replace",
                    "render_dpi": self.render_dpi,
                    **request.config,
                },
            ),
            pages=pages,
        )
        if artifacts:
            artifacts.write_parser_raw(
                request.document_id,
                self.name,
                {
                    "base_ir_before_routing": baseline.to_dict(),
                    "route_decisions": [
                        decision.to_dict() for decision in decisions
                    ],
                    "routed_pages": list(routed_pages),
                },
            )
            artifacts.write_ir(result)
        return result

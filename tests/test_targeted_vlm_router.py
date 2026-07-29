from __future__ import annotations

from pathlib import Path

from src.complex_document.ir import (
    BBox,
    DocumentMetadata,
    Element,
    Page,
    ParserMetadata,
    SpatialDocument,
)
from src.complex_document.parsers.base import DocumentParserAdapter, ParseRequest
from src.complex_document.parsers.targeted_vlm_router_parser import (
    TargetedVLMRouterAdapter,
)
from src.complex_document.vlm_routing import route_page_for_vlm


def _element(
    page: int,
    kind: str,
    text: str,
    bbox: BBox,
    order: int,
) -> Element:
    return Element(
        element_id=f"p{page}-e{order}",
        page_number=page,
        element_type=kind,
        text=text,
        markdown=text,
        bbox=bbox,
        reading_order=order,
    )


def test_vlm_router_uses_only_native_page_signals():
    scanned = Page(
        1,
        100,
        100,
        "pdf_points",
        [_element(1, "figure", "", BBox(0, 0, 100, 100, "pdf_points"), 0)],
    )
    visual = Page(
        2,
        100,
        100,
        "pdf_points",
        [
            _element(2, "heading", "道路事故", BBox(10, 5, 80, 15, "pdf_points"), 0),
            _element(2, "figure", "", BBox(10, 20, 90, 50, "pdf_points"), 1),
        ],
    )
    ordinary = Page(
        3,
        100,
        100,
        "pdf_points",
        [
            _element(
                3,
                "paragraph",
                "足夠的原生文字" * 100,
                BBox(10, 10, 90, 90, "pdf_points"),
                0,
            )
        ],
    )
    assert route_page_for_vlm("doc", scanned).mode == "replace"
    assert route_page_for_vlm("doc", visual).mode == "enrich"
    assert route_page_for_vlm("doc", ordinary).should_route is False


class _StaticAdapter(DocumentParserAdapter):
    def __init__(self, name: str, document: SpatialDocument):
        self.name = name
        self.document = document
        self.requested_pages: tuple[int, ...] | None = None

    def version(self) -> str:
        return "test-1"

    def parse(self, request: ParseRequest, artifacts=None) -> SpatialDocument:
        self.requested_pages = request.pages
        selected = (
            self.document.pages
            if request.pages is None
            else [
                page
                for page in self.document.pages
                if page.page_number in request.pages
            ]
        )
        return SpatialDocument(
            "1.0",
            self.document.document,
            ParserMetadata(self.name, self.version()),
            selected,
        )


def test_targeted_adapter_calls_vlm_only_for_routed_page(tmp_path: Path):
    metadata = DocumentMetadata("doc", "0" * 64)
    baseline = SpatialDocument(
        "1.0",
        metadata,
        ParserMetadata("base", "test-1"),
        [
            Page(
                1,
                100,
                100,
                "pdf_points",
                [
                    _element(
                        1,
                        "paragraph",
                        "原生文字" * 200,
                        BBox(5, 5, 95, 95, "pdf_points"),
                        0,
                    )
                ],
            ),
            Page(
                2,
                100,
                100,
                "pdf_points",
                [
                    _element(
                        2,
                        "heading",
                        "道路事故",
                        BBox(10, 5, 80, 15, "pdf_points"),
                        0,
                    ),
                    _element(
                        2,
                        "figure",
                        "",
                        BBox(10, 20, 90, 50, "pdf_points"),
                        1,
                    ),
                ],
            ),
        ],
    )
    vlm = SpatialDocument(
        "1.0",
        metadata,
        ParserMetadata("vlm", "test-1"),
        [
            Page(
                2,
                1000,
                1000,
                "normalized",
                [
                    _element(
                        2,
                        "figure",
                        "113\n45,219",
                        BBox(0.1, 0.2, 0.9, 0.5, "normalized"),
                        0,
                    )
                ],
            )
        ],
    )
    base_adapter = _StaticAdapter("base", baseline)
    vlm_adapter = _StaticAdapter("vlm", vlm)
    source = tmp_path / "placeholder.pdf"
    source.write_bytes(b"placeholder")
    result = TargetedVLMRouterAdapter(
        baseline=base_adapter,
        vlm=vlm_adapter,
    ).parse(ParseRequest(source, "doc"))

    assert vlm_adapter.requested_pages == (2,)
    assert result.parser.name == "targeted-vlm-router"
    assert result.parser.config["routed_pages"] == [2]
    assert all(
        not element.metadata["vlm_router_selected"]
        for element in result.pages[0].elements
    )
    enriched = [
        element
        for element in result.pages[1].elements
        if element.metadata["vlm_router_selected"]
    ]
    assert len(enriched) == 1
    assert "道路事故" in enriched[0].text
    assert "45,219" in enriched[0].text

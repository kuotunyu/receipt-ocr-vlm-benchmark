from __future__ import annotations

import importlib.metadata
from typing import Any

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.ir import (
    BBox,
    DocumentMetadata,
    Element,
    Page,
    ParserMetadata,
    SpatialDocument,
)
from src.complex_document.parsers.base import (
    DocumentParserAdapter,
    ParseRequest,
    ParserUnavailable,
)
from src.complex_document.parsers.heuristics import classify_text, infer_section_paths


class LiteParseAdapter(DocumentParserAdapter):
    name = "liteparse"

    def __init__(self, *, ocr_enabled: bool = False, ocr_language: str = "chi_tra"):
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language

    def version(self) -> str:
        try:
            return importlib.metadata.version("liteparse")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ParserUnavailable("liteparse is not installed") from exc

    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        try:
            from liteparse import LiteParse
        except ImportError as exc:
            raise ParserUnavailable(
                "liteparse is not installed; install the complex-document extra"
            ) from exc

        target_pages = (
            ",".join(str(page) for page in request.pages) if request.pages else None
        )
        options: dict[str, Any] = {
            "ocr_enabled": self.ocr_enabled,
            "ocr_language": self.ocr_language,
        }
        if target_pages:
            options["target_pages"] = target_pages
        native_result = LiteParse(**options).parse(str(request.path))

        pages: list[Page] = []
        raw_pages: list[dict[str, Any]] = []
        for native_page in native_result.pages:
            page_number = int(native_page.page_num)
            elements: list[Element] = []
            raw_items: list[dict[str, Any]] = []
            for native_item in native_page.text_items:
                text = str(native_item.text or "").strip()
                if not text:
                    continue
                bbox = BBox(
                    x0=float(native_item.x),
                    y0=float(native_item.y),
                    x1=float(native_item.x + native_item.width),
                    y1=float(native_item.y + native_item.height),
                    coordinate_space="pdf_points",
                )
                element_type = classify_text(
                    text, font_size=float(native_item.font_size or 0) or None
                )
                elements.append(
                    Element(
                        element_id=f"p{page_number}-e{len(elements):04d}",
                        page_number=page_number,
                        element_type=element_type,
                        text=text,
                        markdown=text,
                        bbox=bbox,
                        reading_order=len(elements),
                        confidence=(
                            float(native_item.confidence)
                            if native_item.confidence is not None
                            else None
                        ),
                        metadata={
                            "font_name": native_item.font_name,
                            "font_size": native_item.font_size,
                            "rotation": native_item.rotation,
                        },
                    )
                )
                raw_items.append(
                    {
                        "text": text,
                        "x": native_item.x,
                        "y": native_item.y,
                        "width": native_item.width,
                        "height": native_item.height,
                        "confidence": native_item.confidence,
                        "font_name": native_item.font_name,
                        "font_size": native_item.font_size,
                    }
                )
            infer_section_paths(elements)

            screenshot_ref = None
            if artifacts:
                try:
                    import fitz

                    screenshot_path = (
                        artifacts.screenshot_dir(request.document_id, self.name)
                        / f"page-{page_number:04d}.png"
                    )
                    with fitz.open(request.path) as pdf:
                        pdf[page_number - 1].get_pixmap(dpi=144, alpha=False).save(
                            screenshot_path
                        )
                    screenshot_ref = str(screenshot_path.as_posix())
                    for element in elements:
                        element.source_image_ref = screenshot_ref
                except ImportError:
                    screenshot_ref = None

            pages.append(
                Page(
                    page_number=page_number,
                    width=float(native_page.width),
                    height=float(native_page.height),
                    coordinate_space="pdf_points",
                    elements=elements,
                    screenshot_ref=screenshot_ref,
                )
            )
            raw_pages.append(
                {
                    "page_number": page_number,
                    "width": native_page.width,
                    "height": native_page.height,
                    "text": native_page.text,
                    "markdown": native_page.markdown,
                    "complexity": native_page.complexity,
                    "text_items": raw_items,
                }
            )

        result = SpatialDocument(
            schema_version="1.0",
            document=DocumentMetadata(
                document_id=request.document_id,
                checksum_sha256=request.checksum(),
                source_uri=request.source_uri,
            ),
            parser=ParserMetadata(
                name=self.name,
                version=self.version(),
                config={**options, **request.config},
            ),
            pages=pages,
        )
        if artifacts:
            artifacts.write_parser_raw(
                request.document_id,
                self.name,
                {
                    "text": native_result.text,
                    "form_type": native_result.form_type,
                    "creator": native_result.creator,
                    "producer": native_result.producer,
                    "pages": raw_pages,
                },
            )
            artifacts.write_ir(result)
        return result

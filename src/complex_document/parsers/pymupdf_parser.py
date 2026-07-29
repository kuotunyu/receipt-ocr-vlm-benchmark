from __future__ import annotations

import importlib.metadata
import statistics
from pathlib import Path
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


class PyMuPDFAdapter(DocumentParserAdapter):
    """Fast spatial-text/image baseline using PyMuPDF only."""

    name = "pymupdf"

    def __init__(self, *, render_dpi: int = 144, detect_tables: bool = True):
        self.render_dpi = render_dpi
        self.detect_tables = detect_tables

    def version(self) -> str:
        try:
            return importlib.metadata.version("PyMuPDF")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ParserUnavailable("PyMuPDF is not installed") from exc

    @staticmethod
    def _table_markdown(rows: list[list[Any]]) -> str:
        clean_rows = [
            ["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row]
            for row in rows
        ]
        if not clean_rows:
            return ""
        width = max(len(row) for row in clean_rows)
        padded = [row + [""] * (width - len(row)) for row in clean_rows]
        header = padded[0]
        body = padded[1:]
        return "\n".join(
            [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(["---"] * width) + " |",
                *("| " + " | ".join(row) + " |" for row in body),
            ]
        )

    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        try:
            import fitz
        except ImportError as exc:
            raise ParserUnavailable("PyMuPDF is not installed") from exc

        native_pages: list[dict[str, Any]] = []
        pages: list[Page] = []
        with fitz.open(request.path) as pdf:
            selected = self.selected_pages(pdf.page_count, request.pages)
            for page_number in selected:
                source_page = pdf[page_number - 1]
                blocks = source_page.get_text("dict", sort=True).get("blocks", [])
                font_sizes = [
                    float(span.get("size", 0))
                    for block in blocks
                    if block.get("type") == 0
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                    if span.get("text", "").strip()
                ]
                median_size = statistics.median(font_sizes) if font_sizes else None
                elements: list[Element] = []
                native_blocks: list[dict[str, Any]] = []

                for block in blocks:
                    block_type = block.get("type")
                    bbox_value = block.get("bbox")
                    if not bbox_value or len(bbox_value) != 4:
                        continue
                    bbox = BBox(*map(float, bbox_value), coordinate_space="pdf_points")
                    if block_type == 0:
                        spans = [
                            span
                            for line in block.get("lines", [])
                            for span in line.get("spans", [])
                            if span.get("text", "").strip()
                        ]
                        text = "\n".join(
                            "".join(
                                span.get("text", "")
                                for span in line.get("spans", [])
                            ).strip()
                            for line in block.get("lines", [])
                            if "".join(
                                span.get("text", "")
                                for span in line.get("spans", [])
                            ).strip()
                        )
                        if not text:
                            continue
                        max_size = max(
                            (float(span.get("size", 0)) for span in spans), default=0
                        )
                        element_type = classify_text(
                            text,
                            font_size=max_size or None,
                            median_font_size=median_size,
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
                                confidence=1.0,
                                metadata={
                                    "max_font_size": max_size,
                                    "font_names": sorted(
                                        {
                                            str(span.get("font"))
                                            for span in spans
                                            if span.get("font")
                                        }
                                    ),
                                },
                            )
                        )
                        native_blocks.append(
                            {
                                "type": "text",
                                "bbox": list(map(float, bbox_value)),
                                "text": text,
                                "spans": [
                                    {
                                        "text": span.get("text", ""),
                                        "size": span.get("size"),
                                        "font": span.get("font"),
                                        "bbox": span.get("bbox"),
                                    }
                                    for span in spans
                                ],
                            }
                        )
                    elif block_type == 1:
                        elements.append(
                            Element(
                                element_id=f"p{page_number}-e{len(elements):04d}",
                                page_number=page_number,
                                element_type="figure",
                                text="",
                                markdown="",
                                bbox=bbox,
                                reading_order=len(elements),
                                confidence=1.0,
                                metadata={
                                    "width": block.get("width"),
                                    "height": block.get("height"),
                                    "extension": block.get("ext"),
                                },
                            )
                        )
                        native_blocks.append(
                            {
                                "type": "image",
                                "bbox": list(map(float, bbox_value)),
                                "width": block.get("width"),
                                "height": block.get("height"),
                                "extension": block.get("ext"),
                            }
                        )

                known_figure_boxes = {
                    (
                        round(element.bbox.x0, 2),
                        round(element.bbox.y0, 2),
                        round(element.bbox.x1, 2),
                        round(element.bbox.y1, 2),
                    )
                    for element in elements
                    if element.element_type == "figure" and element.bbox
                }
                for image_info in source_page.get_images(full=True):
                    xref = int(image_info[0])
                    for image_rect in source_page.get_image_rects(xref):
                        key = tuple(round(float(value), 2) for value in image_rect)
                        if key in known_figure_boxes or image_rect.is_empty:
                            continue
                        bbox = BBox(
                            *map(float, image_rect), coordinate_space="pdf_points"
                        )
                        elements.append(
                            Element(
                                element_id=f"p{page_number}-e{len(elements):04d}",
                                page_number=page_number,
                                element_type="figure",
                                text="",
                                markdown="",
                                bbox=bbox,
                                reading_order=len(elements),
                                confidence=1.0,
                                metadata={
                                    "xref": xref,
                                    "native_detector": "pymupdf.get_images",
                                },
                            )
                        )
                        native_blocks.append(
                            {
                                "type": "image",
                                "bbox": list(map(float, image_rect)),
                                "xref": xref,
                                "native_detector": "pymupdf.get_images",
                            }
                        )
                        known_figure_boxes.add(key)

                if self.detect_tables:
                    try:
                        found_tables = source_page.find_tables().tables
                    except Exception:
                        found_tables = []
                    for table in found_tables:
                        rows = table.extract()
                        markdown = self._table_markdown(rows)
                        if not markdown:
                            continue
                        bbox = BBox(
                            *map(float, table.bbox), coordinate_space="pdf_points"
                        )
                        elements.append(
                            Element(
                                element_id=f"p{page_number}-e{len(elements):04d}",
                                page_number=page_number,
                                element_type="table",
                                text="\n".join(
                                    "\t".join("" if cell is None else str(cell) for cell in row)
                                    for row in rows
                                ),
                                markdown=markdown,
                                bbox=bbox,
                                reading_order=len(elements),
                                confidence=None,
                                metadata={
                                    "row_count": len(rows),
                                    "column_count": max(
                                        (len(row) for row in rows), default=0
                                    ),
                                    "native_detector": "pymupdf.find_tables",
                                },
                            )
                        )

                elements.sort(
                    key=lambda element: (
                        element.bbox.y0 if element.bbox else float("inf"),
                        element.bbox.x0 if element.bbox else float("inf"),
                    )
                )
                for order, element in enumerate(elements):
                    element.reading_order = order
                    element.element_id = f"p{page_number}-e{order:04d}"
                infer_section_paths(elements)

                screenshot_ref = None
                if artifacts:
                    screenshot_path = (
                        artifacts.screenshot_dir(request.document_id, self.name)
                        / f"page-{page_number:04d}.png"
                    )
                    source_page.get_pixmap(dpi=self.render_dpi, alpha=False).save(
                        screenshot_path
                    )
                    screenshot_ref = str(screenshot_path.as_posix())
                    for element in elements:
                        element.source_image_ref = screenshot_ref

                pages.append(
                    Page(
                        page_number=page_number,
                        width=float(source_page.rect.width),
                        height=float(source_page.rect.height),
                        coordinate_space="pdf_points",
                        elements=elements,
                        screenshot_ref=screenshot_ref,
                    )
                )
                native_pages.append(
                    {
                        "page_number": page_number,
                        "width": float(source_page.rect.width),
                        "height": float(source_page.rect.height),
                        "blocks": native_blocks,
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
                config={
                    "sort": True,
                    "detect_tables": self.detect_tables,
                    "render_dpi": self.render_dpi,
                    **request.config,
                },
            ),
            pages=pages,
        )
        if artifacts:
            artifacts.write_parser_raw(
                request.document_id, self.name, {"pages": native_pages}
            )
            artifacts.write_ir(result)
        return result

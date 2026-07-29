from __future__ import annotations

import copy
from typing import Any

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.ir import (
    BBox,
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
from src.complex_document.parsers.heuristics import infer_section_paths
from src.complex_document.parsers.liteparse_parser import LiteParseAdapter
from src.complex_document.table_reconstruction import (
    bbox_contains_center,
    clean_rows,
    find_table_caption,
    link_cross_page_tables,
    table_markdown,
    trim_nested_spanning_tail,
)


class LiteParseTableAdapter(DocumentParserAdapter):
    """LiteParse spatial text enriched with local grid-table reconstruction."""

    name = "liteparse-table"
    reconstruction_version = "1"

    def __init__(self, *, ocr_enabled: bool = False, render_dpi: int = 144):
        self.base = LiteParseAdapter(ocr_enabled=ocr_enabled)
        self.render_dpi = render_dpi

    def version(self) -> str:
        return (
            f"liteparse-{self.base.version()}"
            f"+table-reconstruction-{self.reconstruction_version}"
        )

    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        try:
            import fitz
        except ImportError as exc:
            raise ParserUnavailable("PyMuPDF is required for table geometry") from exc

        base_document = self.base.parse(request, artifacts=None)
        reconstructed_pages: list[Page] = []
        native_tables: list[dict[str, Any]] = []
        with fitz.open(request.path) as pdf:
            for base_page in base_document.pages:
                source_page = pdf[base_page.page_number - 1]
                original_elements = copy.deepcopy(base_page.elements)
                table_elements: list[Element] = []
                try:
                    detected_tables = source_page.find_tables().tables
                except Exception:
                    detected_tables = []
                detected_bboxes = [
                    BBox(
                        *map(float, detected.bbox),
                        coordinate_space="pdf_points",
                    )
                    for detected in detected_tables
                ]
                for table_index, detected in enumerate(detected_tables):
                    if detected.row_count < 2 or detected.col_count < 2:
                        continue
                    rows = clean_rows(detected.extract())
                    if not rows or not any(any(cell for cell in row) for row in rows):
                        continue
                    row_bboxes = [
                        tuple(map(float, row.bbox)) for row in detected.rows
                    ]
                    row_nonempty_cell_counts = [
                        sum(cell is not None for cell in row.cells)
                        for row in detected.rows
                    ]
                    bbox, rows, bbox_adjustment = trim_nested_spanning_tail(
                        table_index=table_index,
                        bbox=detected_bboxes[table_index],
                        rows=rows,
                        row_bboxes=row_bboxes,
                        row_nonempty_cell_counts=row_nonempty_cell_counts,
                        all_table_bboxes=detected_bboxes,
                    )
                    if len(rows) < 2:
                        continue
                    column_count = max(
                        (len(row) for row in rows), default=0
                    )
                    caption_elements = find_table_caption(
                        original_elements, bbox
                    )
                    caption_text = " ".join(
                        element.text.strip()
                        for element in caption_elements
                        if element.text.strip()
                    )
                    table_text = "\n".join("\t".join(row) for row in rows)
                    if caption_text:
                        table_text = f"{caption_text}\n{table_text}"
                    markdown = table_markdown(rows)
                    if caption_text:
                        markdown = f"**{caption_text}**\n\n{markdown}"
                    enclosed = [
                        element
                        for element in original_elements
                        if element.bbox
                        and bbox_contains_center(bbox, element.bbox)
                    ]
                    confidences = [
                        element.confidence
                        for element in enclosed
                        if element.confidence is not None
                    ]
                    table_element = Element(
                        element_id=(
                            f"p{base_page.page_number}-table-{table_index:03d}"
                        ),
                        page_number=base_page.page_number,
                        element_type="table",
                        text=table_text,
                        markdown=markdown,
                        bbox=bbox,
                        reading_order=0,
                        parent_section_path=(
                            list(caption_elements[0].parent_section_path)
                            if caption_elements
                            else []
                        ),
                        confidence=(
                            sum(confidences) / len(confidences)
                            if confidences
                            else None
                        ),
                        metadata={
                            "row_count": len(rows),
                            "column_count": column_count,
                            "caption": caption_text or None,
                            "caption_element_id": (
                                caption_elements[0].element_id
                                if caption_elements
                                else None
                            ),
                            "caption_element_ids": [
                                element.element_id for element in caption_elements
                            ],
                            "geometry_source": "pymupdf.find_tables",
                            "text_source": "grid-cell-extraction",
                            "bbox_adjustment": bbox_adjustment,
                            "enclosed_liteparse_elements": len(enclosed),
                            "_insertion_order": min(
                                (
                                    element.reading_order
                                    for element in enclosed
                                ),
                                default=len(original_elements),
                            ),
                        },
                    )
                    table_elements.append(table_element)
                    for element in enclosed:
                        if element not in caption_elements:
                            element.metadata.setdefault(
                                "shadowed_by_reconstructed_tables", []
                            ).append(table_index)
                    native_tables.append(
                        {
                            "page_number": base_page.page_number,
                            "table_index": table_index,
                            "bbox": [
                                bbox.x0,
                                bbox.y0,
                                bbox.x1,
                                bbox.y1,
                            ],
                            "row_count": len(rows),
                            "column_count": column_count,
                            "caption": caption_text or None,
                            "rows": rows,
                            "bbox_adjustment": bbox_adjustment,
                        }
                    )

                combined = original_elements + table_elements
                combined.sort(
                    key=lambda element: (
                        element.metadata.get(
                            "_insertion_order", element.reading_order
                        ),
                        0 if element.element_type == "table" else 1,
                    )
                )
                for order, element in enumerate(combined):
                    element.metadata.pop("_insertion_order", None)
                    element.reading_order = order
                    element.element_id = (
                        f"p{base_page.page_number}-e{order:04d}"
                    )
                infer_section_paths(combined)

                screenshot_ref = None
                if artifacts:
                    screenshot_path = (
                        artifacts.screenshot_dir(request.document_id, self.name)
                        / f"page-{base_page.page_number:04d}.png"
                    )
                    source_page.get_pixmap(
                        dpi=self.render_dpi, alpha=False
                    ).save(screenshot_path)
                    screenshot_ref = str(screenshot_path.as_posix())
                    for element in combined:
                        element.source_image_ref = screenshot_ref
                reconstructed_pages.append(
                    Page(
                        page_number=base_page.page_number,
                        width=base_page.width,
                        height=base_page.height,
                        coordinate_space=base_page.coordinate_space,
                        elements=combined,
                        screenshot_ref=screenshot_ref,
                    )
                )

        link_count = link_cross_page_tables(
            request.document_id, reconstructed_pages
        )
        result = SpatialDocument(
            schema_version=base_document.schema_version,
            document=base_document.document,
            parser=ParserMetadata(
                name=self.name,
                version=self.version(),
                config={
                    **base_document.parser.config,
                    "table_reconstruction": "grid-geometry-v1",
                    "cross_page_linking": "edge-alignment-v1",
                    "render_dpi": self.render_dpi,
                    **request.config,
                },
            ),
            pages=reconstructed_pages,
        )
        if artifacts:
            artifacts.write_parser_raw(
                request.document_id,
                self.name,
                {
                    "base_parser": base_document.parser.__dict__,
                    "base_ir_before_reconstruction": base_document.to_dict(),
                    "detected_tables": native_tables,
                    "cross_page_links": link_count,
                },
            )
            artifacts.write_ir(result)
        return result

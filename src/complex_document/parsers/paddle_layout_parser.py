from __future__ import annotations

import importlib.metadata
import time
from functools import lru_cache

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


@lru_cache(maxsize=4)
def _cpu_engine(language: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=language,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        device="cpu",
    )


def _run_ocr_cpu(image, language: str):
    from src.pipeline_a.ocr import OCRBox

    boxes = []
    for result in _cpu_engine(language).predict(image) or []:
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []
        native_boxes = result.get("rec_boxes")
        if native_boxes is None:
            continue
        for text, score, box in zip(texts, scores, native_boxes):
            if not text or not text.strip():
                continue
            x0, y0, x1, y1 = (float(value) for value in box)
            boxes.append(
                OCRBox(
                    text=text.strip(),
                    score=float(score),
                    x1=x0,
                    y1=y0,
                    x2=x1,
                    y2=y1,
                )
            )
    return boxes


class PaddleLayoutAdapter(DocumentParserAdapter):
    """Current PaddleOCR plus deterministic spatial line grouping baseline."""

    name = "paddleocr-layout"

    def __init__(self, *, dpi: int = 150, language: str = "chinese_cht"):
        self.dpi = dpi
        self.language = language

    def version(self) -> str:
        try:
            return importlib.metadata.version("paddleocr")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ParserUnavailable("PaddleOCR is not installed") from exc

    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        try:
            import cv2
            import fitz
            import numpy as np

            from src.pipeline_a.layout import group_into_lines, line_text
        except ImportError as exc:
            raise ParserUnavailable(
                "PaddleOCR/PyMuPDF dependencies are not installed"
            ) from exc

        pages: list[Page] = []
        raw_pages = []
        with fitz.open(request.path) as pdf:
            selected = self.selected_pages(pdf.page_count, request.pages)
            for page_number in selected:
                source_page = pdf[page_number - 1]
                pixmap = source_page.get_pixmap(dpi=self.dpi, alpha=False)
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                if pixmap.n == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                started = time.perf_counter()
                boxes = _run_ocr_cpu(image, self.language)
                latency_seconds = time.perf_counter() - started
                lines = group_into_lines(boxes)
                elements: list[Element] = []
                for line in lines:
                    text = line_text(line).strip()
                    if not text:
                        continue
                    bbox = BBox(
                        x0=min(box.x1 for box in line),
                        y0=min(box.y1 for box in line),
                        x1=max(box.x2 for box in line),
                        y1=max(box.y2 for box in line),
                        coordinate_space="pixels",
                    )
                    elements.append(
                        Element(
                            element_id=f"p{page_number}-e{len(elements):04d}",
                            page_number=page_number,
                            element_type=classify_text(text),
                            text=text,
                            markdown=text,
                            bbox=bbox,
                            reading_order=len(elements),
                            confidence=sum(box.score for box in line) / len(line),
                            metadata={"ocr_token_count": len(line)},
                        )
                    )
                infer_section_paths(elements)

                screenshot_ref = None
                if artifacts:
                    screenshot_path = (
                        artifacts.screenshot_dir(request.document_id, self.name)
                        / f"page-{page_number:04d}.png"
                    )
                    pixmap.save(screenshot_path)
                    screenshot_ref = str(screenshot_path.as_posix())
                    for element in elements:
                        element.source_image_ref = screenshot_ref

                pages.append(
                    Page(
                        page_number=page_number,
                        width=float(pixmap.width),
                        height=float(pixmap.height),
                        coordinate_space="pixels",
                        elements=elements,
                        screenshot_ref=screenshot_ref,
                    )
                )
                raw_pages.append(
                    {
                        "page_number": page_number,
                        "width": pixmap.width,
                        "height": pixmap.height,
                        "latency_seconds": latency_seconds,
                        "ocr_boxes": [
                            {
                                "text": box.text,
                                "confidence": box.score,
                                "bbox": [box.x1, box.y1, box.x2, box.y2],
                            }
                            for box in boxes
                        ],
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
                    "dpi": self.dpi,
                    "language": self.language,
                    "device": "cpu",
                    "layout": "existing-y-line-grouping",
                    **request.config,
                },
            ),
            pages=pages,
        )
        if artifacts:
            artifacts.write_parser_raw(
                request.document_id, self.name, {"pages": raw_pages}
            )
            artifacts.write_ir(result)
        return result

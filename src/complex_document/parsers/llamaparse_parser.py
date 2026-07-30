"""Optional commercial LlamaParse comparator.

The adapter is intentionally lazy: neither the SDK nor a cloud credential is
required by the default, locally reproducible benchmark.  Native page items
are normalized to the same Spatial Document IR used by every local parser.
"""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Iterable
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


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _model_dump(value: Any) -> Any:
    """Convert Pydantic SDK responses to stable JSON-compatible data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        return legacy_dict()
    return str(value)


def _bbox(item: Any) -> tuple[BBox | None, float | None]:
    boxes = _field(item, "bbox") or []
    if not isinstance(boxes, Iterable) or isinstance(boxes, (str, bytes)):
        return None, None
    valid = []
    confidences = []
    for box in boxes:
        x = _field(box, "x")
        y = _field(box, "y")
        width = _field(box, "w")
        height = _field(box, "h")
        if not all(
            isinstance(value, (int, float))
            for value in (x, y, width, height)
        ):
            continue
        valid.append((float(x), float(y), float(x + width), float(y + height)))
        confidence = _field(box, "confidence")
        if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
            confidences.append(float(confidence))
    if not valid:
        return None, None
    bbox = BBox(
        x0=min(value[0] for value in valid),
        y0=min(value[1] for value in valid),
        x1=max(value[2] for value in valid),
        y1=max(value[3] for value in valid),
        coordinate_space="pdf_points",
    )
    confidence = (
        sum(confidences) / len(confidences) if confidences else None
    )
    return bbox, confidence


def _item_text(item: Any, source_type: str) -> tuple[str, str]:
    markdown = str(_field(item, "md", "") or "")
    if source_type == "table":
        text = "\n".join(
            "\t".join("" if cell is None else str(cell) for cell in row)
            for row in (_field(item, "rows", []) or [])
        )
    elif source_type == "image":
        text = str(_field(item, "caption", "") or "")
    elif source_type == "link":
        text = str(_field(item, "text", "") or "")
    else:
        text = str(_field(item, "value", "") or "")
    return text or markdown, markdown or text


def _normalized_elements(
    native_items: list[Any],
    page_number: int,
    page_confidence: float | None,
) -> list[Element]:
    elements: list[Element] = []
    section_levels: list[tuple[int, str]] = []

    for item in native_items:
        source_type = str(_field(item, "type", "text") or "text")
        element_type = {
            "heading": "heading",
            "table": "table",
            "image": "figure",
            "list": "list",
        }.get(source_type, "paragraph")
        text, markdown = _item_text(item, source_type)
        bbox, bbox_confidence = _bbox(item)
        confidence = bbox_confidence
        if confidence is None and isinstance(page_confidence, (int, float)):
            confidence = float(page_confidence)

        if element_type == "heading":
            level = int(_field(item, "level", 1) or 1)
            section_levels = [
                value for value in section_levels if value[0] < level
            ]
            parent_section_path = [value[1] for value in section_levels]
        else:
            parent_section_path = [value[1] for value in section_levels]

        metadata: dict[str, Any] = {"llamaparse_item_type": source_type}
        if source_type in {"header", "footer"}:
            metadata["page_marginalia"] = source_type
        if source_type == "table":
            metadata.update(
                {
                    "rows": _model_dump(_field(item, "rows", [])),
                    "csv": _field(item, "csv"),
                    "html": _field(item, "html"),
                    "merged_from_pages": _model_dump(
                        _field(item, "merged_from_pages")
                    ),
                    "merged_into_page": _field(item, "merged_into_page"),
                    "parse_concerns": _model_dump(
                        _field(item, "parse_concerns")
                    ),
                }
            )
        if source_type == "list":
            metadata["ordered"] = bool(_field(item, "ordered", False))

        elements.append(
            Element(
                element_id=f"p{page_number}-e{len(elements):04d}",
                page_number=page_number,
                element_type=element_type,
                text=text,
                markdown=markdown,
                bbox=bbox,
                reading_order=len(elements),
                parent_section_path=parent_section_path,
                confidence=confidence,
                metadata=metadata,
            )
        )

        if element_type == "heading":
            section_levels.append((level, text.strip()))

        caption = (
            str(_field(item, "caption", "") or "").strip()
            if source_type == "image"
            else ""
        )
        if caption:
            elements.append(
                Element(
                    element_id=f"p{page_number}-e{len(elements):04d}",
                    page_number=page_number,
                    element_type="caption",
                    text=caption,
                    markdown=caption,
                    bbox=bbox,
                    reading_order=len(elements),
                    parent_section_path=[
                        value[1] for value in section_levels
                    ],
                    confidence=confidence,
                    metadata={"llamaparse_item_type": "image-caption"},
                )
            )
    return elements


def _normalize_result(
    native_result: Any,
    request: ParseRequest,
    *,
    parser_version: str,
    parser_config: dict[str, Any],
) -> SpatialDocument:
    item_container = _field(native_result, "items")
    item_pages = _field(item_container, "pages", []) or []
    markdown_container = _field(native_result, "markdown")
    markdown_pages = {
        int(_field(page, "page_number")): str(
            _field(page, "markdown", "") or ""
        )
        for page in (_field(markdown_container, "pages", []) or [])
        if _field(page, "success", True)
        and isinstance(_field(page, "page_number"), int)
    }
    metadata_container = _field(native_result, "metadata")
    page_confidences = {
        int(_field(page, "page_number")): _field(page, "confidence")
        for page in (_field(metadata_container, "pages", []) or [])
        if isinstance(_field(page, "page_number"), int)
    }

    pages: list[Page] = []
    seen_pages: set[int] = set()
    for item_page in item_pages:
        if not _field(item_page, "success", True):
            continue
        page_number = int(_field(item_page, "page_number"))
        width = float(_field(item_page, "page_width", 1.0) or 1.0)
        height = float(_field(item_page, "page_height", 1.0) or 1.0)
        elements = _normalized_elements(
            list(_field(item_page, "items", []) or []),
            page_number,
            page_confidences.get(page_number),
        )
        if not elements and markdown_pages.get(page_number):
            markdown = markdown_pages[page_number]
            elements = [
                Element(
                    element_id=f"p{page_number}-e0000",
                    page_number=page_number,
                    element_type="paragraph",
                    text=markdown,
                    markdown=markdown,
                    bbox=None,
                    reading_order=0,
                    confidence=page_confidences.get(page_number),
                    metadata={"fallback": "page-markdown"},
                )
            ]
        pages.append(
            Page(
                page_number=page_number,
                width=width,
                height=height,
                coordinate_space="pdf_points",
                elements=elements,
            )
        )
        seen_pages.add(page_number)

    for page_number, markdown in sorted(markdown_pages.items()):
        if page_number in seen_pages:
            continue
        pages.append(
            Page(
                page_number=page_number,
                width=1.0,
                height=1.0,
                coordinate_space="normalized",
                elements=[
                    Element(
                        element_id=f"p{page_number}-e0000",
                        page_number=page_number,
                        element_type="paragraph",
                        text=markdown,
                        markdown=markdown,
                        bbox=None,
                        reading_order=0,
                        confidence=page_confidences.get(page_number),
                        metadata={"fallback": "page-markdown-no-layout"},
                    )
                ],
            )
        )

    if not pages:
        raise RuntimeError("LlamaParse returned no successful pages")

    return SpatialDocument(
        schema_version="1.0",
        document=DocumentMetadata(
            document_id=request.document_id,
            checksum_sha256=request.checksum(),
            source_uri=request.source_uri,
        ),
        parser=ParserMetadata(
            name=LlamaParseAdapter.name,
            version=parser_version,
            config=parser_config,
        ),
        pages=sorted(pages, key=lambda page: page.page_number),
    )


class LlamaParseAdapter(DocumentParserAdapter):
    name = "llamaparse-cloud"

    def version(self) -> str:
        try:
            return importlib.metadata.version("llama-cloud")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ParserUnavailable(
                "install the optional 'llamaparse' project extra"
            ) from exc

    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ParserUnavailable("LLAMA_CLOUD_API_KEY is not set")
        try:
            from llama_cloud import LlamaCloud
        except ImportError as exc:
            raise ParserUnavailable(
                "install the optional 'llamaparse' project extra"
            ) from exc

        tier = str(request.config.get("tier", "agentic"))
        version = str(request.config.get("version", "latest"))
        expand = ["text", "markdown", "items", "metadata", "job_metadata"]
        page_ranges = (
            {"target_pages": ",".join(str(page) for page in request.pages)}
            if request.pages
            else None
        )
        parse_kwargs: dict[str, Any] = {
            "file_id": None,
            "tier": tier,
            "version": version,
            "expand": expand,
            "client_name": "receipt-ocr-vlm-benchmark",
        }
        if page_ranges:
            parse_kwargs["page_ranges"] = page_ranges
        if "disable_cache" in request.config:
            parse_kwargs["disable_cache"] = bool(
                request.config["disable_cache"]
            )
        if request.config.get("project_id"):
            parse_kwargs["project_id"] = str(request.config["project_id"])

        client = LlamaCloud(api_key=api_key)
        try:
            uploaded = client.files.create(file=request.path, purpose="parse")
            parse_kwargs["file_id"] = uploaded.id
            native_result = client.parsing.parse(**parse_kwargs)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        safe_request_config = {
            key: value
            for key, value in request.config.items()
            if not any(
                marker in key.lower()
                for marker in ("key", "token", "secret", "password")
            )
            and key != "project_id"
        }
        parser_config = {
            "tier": tier,
            "version": version,
            "expand": expand,
            "page_ranges": page_ranges,
            "project_id_configured": bool(request.config.get("project_id")),
            **safe_request_config,
        }
        result = _normalize_result(
            native_result,
            request,
            parser_version=self.version(),
            parser_config=parser_config,
        )
        if artifacts:
            artifacts.write_parser_raw(
                request.document_id,
                self.name,
                {
                    "file_id": uploaded.id,
                    "parse_result": _model_dump(native_result),
                },
            )
            artifacts.write_ir(result)
        return result

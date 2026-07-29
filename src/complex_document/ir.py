"""Shared Spatial Document Intermediate Representation (IR).

The IR intentionally keeps provenance alongside content.  Every adapter may
retain its native output separately, while downstream code consumes only this
stable, JSON-serializable representation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ElementType = Literal[
    "heading",
    "paragraph",
    "table",
    "figure",
    "caption",
    "footnote",
    "list",
]
CoordinateSpace = Literal["pdf_points", "pixels", "normalized"]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float
    coordinate_space: CoordinateSpace

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox maximum must not be smaller than minimum")
        if self.coordinate_space == "normalized":
            values = (self.x0, self.y0, self.x1, self.y1)
            if any(value < 0 or value > 1 for value in values):
                raise ValueError("normalized bbox values must be within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParserMetadata:
    name: str
    version: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: str
    checksum_sha256: str
    source_uri: str | None = None


@dataclass
class Element:
    element_id: str
    page_number: int
    element_type: ElementType
    text: str
    markdown: str
    bbox: BBox | None
    reading_order: int
    parent_section_path: list[str] = field(default_factory=list)
    confidence: float | None = None
    source_image_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number is 1-based")
        if self.reading_order < 0:
            raise ValueError("reading_order must be non-negative")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")


@dataclass
class Page:
    page_number: int
    width: float
    height: float
    coordinate_space: CoordinateSpace
    elements: list[Element] = field(default_factory=list)
    screenshot_ref: str | None = None

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number is 1-based")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("page dimensions must be positive")
        for element in self.elements:
            if element.page_number != self.page_number:
                raise ValueError("element page_number does not match page")


@dataclass
class SpatialDocument:
    schema_version: str
    document: DocumentMetadata
    parser: ParserMetadata
    pages: list[Page]
    parsing_timestamp: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        page_numbers = [page.page_number for page in self.pages]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("page numbers must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpatialDocument":
        pages: list[Page] = []
        for page_value in value["pages"]:
            elements = []
            for element_value in page_value.get("elements", []):
                bbox_value = element_value.get("bbox")
                bbox = BBox(**bbox_value) if bbox_value else None
                elements.append(Element(**{**element_value, "bbox": bbox}))
            pages.append(Page(**{**page_value, "elements": elements}))
        return cls(
            schema_version=value["schema_version"],
            document=DocumentMetadata(**value["document"]),
            parser=ParserMetadata(**value["parser"]),
            pages=pages,
            parsing_timestamp=value["parsing_timestamp"],
        )

    @classmethod
    def from_json(cls, payload: str) -> "SpatialDocument":
        return cls.from_dict(json.loads(payload))

    def all_elements(self) -> list[Element]:
        return [
            element
            for page in sorted(self.pages, key=lambda item: item.page_number)
            for element in sorted(page.elements, key=lambda item: item.reading_order)
        ]

    def plain_text(self) -> str:
        return "\n".join(element.text for element in self.all_elements() if element.text)

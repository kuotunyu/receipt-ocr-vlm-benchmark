from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.complex_document.ir import (
    BBox,
    Element,
    Page,
    ParserMetadata,
    SpatialDocument,
)

ChunkKind = Literal["context", "citation"]


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    markdown: str
    pages: list[int]
    bboxes: list[BBox]
    section_path: list[str]
    parser_name: str
    parser_version: str
    element_ids: list[str]
    source_image_refs: list[str] = field(default_factory=list)
    kind: ChunkKind = "context"
    atomic_type: str | None = None
    parent_chunk_id: str | None = None
    metadata: dict = field(default_factory=dict)


def _make_chunk(
    document: SpatialDocument,
    index: int,
    elements: list[Element],
    *,
    atomic_type: str | None = None,
    kind: ChunkKind = "context",
    parent_chunk_id: str | None = None,
    text_override: str | None = None,
) -> Chunk:
    pages = sorted({element.page_number for element in elements})
    section_path = next(
        (
            list(element.parent_section_path)
            for element in reversed(elements)
            if element.parent_section_path
        ),
        [],
    )
    return Chunk(
        chunk_id=f"{document.document.document_id}:{document.parser.name}:c{index:05d}",
        document_id=document.document.document_id,
        text=(
            text_override
            if text_override is not None
            else "\n".join(element.text for element in elements if element.text)
        ),
        markdown="\n\n".join(
            element.markdown or element.text
            for element in elements
            if element.markdown or element.text
        ),
        pages=pages,
        bboxes=[element.bbox for element in elements if element.bbox is not None],
        section_path=section_path,
        parser_name=document.parser.name,
        parser_version=document.parser.version,
        element_ids=[element.element_id for element in elements],
        source_image_refs=sorted(
            {
                element.source_image_ref
                for element in elements
                if element.source_image_ref
            }
        ),
        kind=kind,
        atomic_type=atomic_type,
        parent_chunk_id=parent_chunk_id,
    )


def fixed_size_chunks(
    document: SpatialDocument,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Recursive/fixed baseline while retaining source element provenance."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("require chunk_size > overlap >= 0")
    elements = document.all_elements()
    chunks: list[Chunk] = []
    current: list[Element] = []
    current_length = 0

    def flush() -> None:
        nonlocal current, current_length
        if not current:
            return
        chunks.append(_make_chunk(document, len(chunks), current))
        if overlap:
            retained: list[Element] = []
            retained_length = 0
            for element in reversed(current):
                retained.insert(0, element)
                retained_length += len(element.text) + 1
                if retained_length >= overlap:
                    break
            current = retained
            current_length = retained_length
        else:
            current = []
            current_length = 0

    for element in elements:
        length = len(element.text) + 1
        if current and current_length + length > chunk_size:
            flush()
        current.append(element)
        current_length += length
    flush()
    return chunks


def structure_aware_chunks(
    document: SpatialDocument,
    *,
    max_section_chars: int = 1800,
    include_sentence_nodes: bool = True,
) -> list[Chunk]:
    """Chunk sections, atomic tables, figure+caption, and attached footnotes."""
    context_chunks: list[Chunk] = []
    pending: list[Element] = []
    pending_length = 0
    elements = document.all_elements()
    known_pages = {page.page_number for page in document.pages}
    processed_element_ids: set[str] = set()

    def flush_pending() -> None:
        nonlocal pending, pending_length
        if pending:
            context_chunks.append(
                _make_chunk(document, len(context_chunks), pending, atomic_type="section")
            )
        pending = []
        pending_length = 0

    index = 0
    while index < len(elements):
        element = elements[index]
        if element.element_id in processed_element_ids:
            index += 1
            continue
        if element.metadata.get("shadowed_by_reconstructed_tables"):
            index += 1
            continue
        if element.element_type == "table":
            flush_pending()
            continuity_id = element.metadata.get("continuity_id")
            table_unit = (
                [
                    candidate
                    for candidate in elements
                    if candidate.element_type == "table"
                    and candidate.metadata.get("continuity_id") == continuity_id
                ]
                if continuity_id
                else [element]
            )
            processed_element_ids.update(
                candidate.element_id for candidate in table_unit
            )
            chunk = _make_chunk(
                document,
                len(context_chunks),
                table_unit,
                atomic_type=(
                    "cross-page-table" if len(table_unit) > 1 else "table"
                ),
            )
            if continuity_id:
                chunk.metadata["continuity_id"] = continuity_id
            context_chunks.append(chunk)
            index += 1
            continue
        if element.element_type == "figure":
            flush_pending()
            unit = [element]
            if (
                index + 1 < len(elements)
                and elements[index + 1].element_type == "caption"
                and elements[index + 1].page_number == element.page_number
            ):
                unit.append(elements[index + 1])
                index += 1
            context_chunks.append(
                _make_chunk(
                    document,
                    len(context_chunks),
                    unit,
                    atomic_type="figure-caption",
                )
            )
            index += 1
            continue
        if element.element_type == "footnote":
            if pending:
                pending.append(element)
                pending_length += len(element.text)
            elif context_chunks:
                previous = context_chunks[-1]
                previous.text = f"{previous.text}\n{element.text}".strip()
                previous.markdown = f"{previous.markdown}\n\n{element.markdown}".strip()
                previous.pages = sorted(set(previous.pages + [element.page_number]))
                if element.bbox:
                    previous.bboxes.append(element.bbox)
                previous.element_ids.append(element.element_id)
            else:
                pending.append(element)
                pending_length += len(element.text)
            index += 1
            continue

        starts_new_section = element.element_type == "heading" and pending
        would_overflow = pending and pending_length + len(element.text) > max_section_chars
        if starts_new_section or would_overflow:
            flush_pending()
        pending.append(element)
        pending_length += len(element.text) + 1
        index += 1
    flush_pending()

    represented_pages = {page for chunk in context_chunks for page in chunk.pages}
    for page_number in sorted(known_pages - represented_pages):
        page_elements = next(
            page.elements for page in document.pages if page.page_number == page_number
        )
        context_chunks.append(
            _make_chunk(
                document,
                len(context_chunks),
                page_elements,
                atomic_type="page-fallback",
                text_override="[此頁只有視覺內容，請取回原始頁面像素]",
            )
        )

    if not include_sentence_nodes:
        return context_chunks

    citation_nodes: list[Chunk] = []
    sentence_split = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
    for parent in context_chunks:
        sentences = [
            sentence.strip()
            for sentence in sentence_split.split(parent.text)
            if sentence.strip()
        ]
        for sentence in sentences:
            citation_nodes.append(
                Chunk(
                    chunk_id=f"{parent.chunk_id}:s{len(citation_nodes):05d}",
                    document_id=parent.document_id,
                    text=sentence,
                    markdown=sentence,
                    pages=list(parent.pages),
                    bboxes=list(parent.bboxes),
                    section_path=list(parent.section_path),
                    parser_name=parent.parser_name,
                    parser_version=parent.parser_version,
                    element_ids=list(parent.element_ids),
                    source_image_refs=list(parent.source_image_refs),
                    kind="citation",
                    atomic_type="sentence",
                    parent_chunk_id=parent.chunk_id,
                )
            )
    return context_chunks + citation_nodes


def context_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Sentence nodes are citation-only and never primary synthesis context."""
    return [chunk for chunk in chunks if chunk.kind == "context"]


def hybrid_routed_chunks(document: SpatialDocument) -> list[Chunk]:
    """Keep fixed chunks on fallback pages and structure chunks on routed pages."""

    def filtered_document(routed: bool) -> SpatialDocument:
        pages = [
            Page(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                coordinate_space=page.coordinate_space,
                elements=[
                    element
                    for element in page.elements
                    if bool(
                        element.metadata.get("table_router_selected", False)
                    )
                    is routed
                    and not (
                        not routed
                        and element.metadata.get(
                            "shadowed_by_reconstructed_tables"
                        )
                    )
                ],
                screenshot_ref=page.screenshot_ref,
            )
            for page in document.pages
            if any(
                bool(element.metadata.get("table_router_selected", False))
                is routed
                and not (
                    not routed
                    and element.metadata.get(
                        "shadowed_by_reconstructed_tables"
                    )
                )
                for element in page.elements
            )
        ]
        return SpatialDocument(
            schema_version=document.schema_version,
            document=document.document,
            parser=ParserMetadata(
                name=document.parser.name,
                version=document.parser.version,
                config={
                    **document.parser.config,
                    "chunk_route": "structure" if routed else "fixed",
                },
            ),
            pages=pages,
            parsing_timestamp=document.parsing_timestamp,
        )

    fallback_document = filtered_document(False)
    routed_document = filtered_document(True)
    chunks = []
    if fallback_document.pages:
        chunks.extend(fixed_size_chunks(fallback_document))
    if routed_document.pages:
        chunks.extend(
            context_chunks(structure_aware_chunks(routed_document))
        )
    for index, chunk in enumerate(chunks):
        chunk.chunk_id = (
            f"{document.document.document_id}:"
            f"{document.parser.name}:c{index:05d}"
        )
        chunk.parser_name = document.parser.name
        chunk.metadata["chunk_route"] = (
            "structure"
            if any(
                element.metadata.get("table_router_selected", False)
                for page in document.pages
                for element in page.elements
                if element.element_id in chunk.element_ids
            )
            else "fixed"
        )
    return chunks

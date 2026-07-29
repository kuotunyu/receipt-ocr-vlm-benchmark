from src.complex_document.chunking import (
    context_chunks,
    fixed_size_chunks,
    hybrid_routed_chunks,
    structure_aware_chunks,
)
from src.complex_document.ir import (
    BBox,
    DocumentMetadata,
    Element,
    Page,
    ParserMetadata,
    SpatialDocument,
)


def make_document() -> SpatialDocument:
    specs = [
        ("heading", "第一節", "# 第一節"),
        ("paragraph", "第一句。第二句。", "第一句。第二句。"),
        ("table", "欄A\t欄B\n1\t2", "|欄A|欄B|\n|-|-|\n|1|2|"),
        ("figure", "", ""),
        ("caption", "圖 1 趨勢圖", "圖 1 趨勢圖"),
        ("footnote", "註：資料來源", "註：資料來源"),
    ]
    elements = [
        Element(
            element_id=f"p1-e{index:04d}",
            page_number=1,
            element_type=element_type,
            text=text,
            markdown=markdown,
            bbox=BBox(0, index * 10, 100, index * 10 + 8, "pdf_points"),
            reading_order=index,
            parent_section_path=["第一節"],
            source_image_ref="page.png",
        )
        for index, (element_type, text, markdown) in enumerate(specs)
    ]
    return SpatialDocument(
        "1.0",
        DocumentMetadata("doc", "a" * 64),
        ParserMetadata("parser", "2.0"),
        [Page(1, 100, 200, "pdf_points", elements)],
    )


def test_structure_chunking_preserves_atomic_units_and_metadata():
    chunks = structure_aware_chunks(make_document())
    primary = context_chunks(chunks)
    table = next(chunk for chunk in primary if chunk.atomic_type == "table")
    figure = next(chunk for chunk in primary if chunk.atomic_type == "figure-caption")
    assert table.text.startswith("欄A")
    assert "圖 1 趨勢圖" in figure.text
    assert "資料來源" in figure.text
    assert all(chunk.pages == [1] for chunk in primary)
    assert all(chunk.bboxes for chunk in primary)
    assert all(chunk.section_path == ["第一節"] for chunk in primary)
    assert all(chunk.parser_version == "2.0" for chunk in primary)
    assert any(chunk.kind == "citation" for chunk in chunks)


def test_sentence_nodes_never_become_primary_context():
    chunks = structure_aware_chunks(make_document())
    assert all(chunk.kind == "context" for chunk in context_chunks(chunks))
    assert len(context_chunks(chunks)) < len(chunks)


def test_fixed_chunking_retains_provenance():
    chunks = fixed_size_chunks(make_document(), chunk_size=30, overlap=5)
    assert len(chunks) >= 2
    assert all(chunk.element_ids for chunk in chunks)
    assert all(chunk.parser_name == "parser" for chunk in chunks)


def test_hybrid_chunking_changes_only_routed_page():
    fallback = Element(
        "p1-e0000",
        1,
        "paragraph",
        "一般頁面內容",
        "一般頁面內容",
        BBox(0, 0, 100, 20, "pdf_points"),
        0,
        metadata={"table_router_selected": False},
    )
    table = Element(
        "p2-e0000",
        2,
        "table",
        "欄一\t欄二\n甲\t1",
        "| 欄一 | 欄二 |\n| --- | --- |\n| 甲 | 1 |",
        BBox(0, 0, 100, 80, "pdf_points"),
        0,
        metadata={
            "table_router_selected": True,
            "row_count": 2,
            "column_count": 2,
        },
    )
    document = SpatialDocument(
        "1.0",
        DocumentMetadata("hybrid", "a" * 64),
        ParserMetadata("hybrid-table-router", "1"),
        [
            Page(1, 100, 100, "pdf_points", [fallback]),
            Page(2, 100, 100, "pdf_points", [table]),
        ],
    )
    chunks = hybrid_routed_chunks(document)
    assert {chunk.metadata["chunk_route"] for chunk in chunks} == {
        "fixed",
        "structure",
    }
    routed = next(
        chunk for chunk in chunks if chunk.metadata["chunk_route"] == "structure"
    )
    assert routed.atomic_type == "table"
    assert routed.pages == [2]

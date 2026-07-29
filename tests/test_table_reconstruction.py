from src.complex_document.chunking import context_chunks, structure_aware_chunks
from src.complex_document.ir import (
    BBox,
    DocumentMetadata,
    Element,
    Page,
    ParserMetadata,
    SpatialDocument,
)
from src.complex_document.normalization_audit import audit_normalization
from src.complex_document.routing import (
    evaluate_reconstructed_table_bboxes,
    evaluate_table_router,
    score_page_for_table_routing,
)
from src.complex_document.table_reconstruction import (
    link_cross_page_tables,
    trim_nested_spanning_tail,
)


def table_element(page: int, y0: float, y1: float, caption=None) -> Element:
    return Element(
        element_id=f"p{page}-table",
        page_number=page,
        element_type="table",
        text=f"table page {page}",
        markdown=f"| table page {page} |",
        bbox=BBox(10, y0, 90, y1, "pdf_points"),
        reading_order=0,
        metadata={
            "row_count": 5,
            "column_count": 3,
            "caption": caption,
        },
    )


def test_cross_page_link_and_chunk_are_single_atomic_context():
    first = Page(
        1,
        100,
        100,
        "pdf_points",
        [table_element(1, 20, 90, caption="Table 1")],
    )
    second = Page(
        2,
        100,
        100,
        "pdf_points",
        [table_element(2, 5, 90)],
    )
    assert link_cross_page_tables("doc", [first, second]) == 1
    continuity_id = first.elements[0].metadata["continuity_id"]
    assert second.elements[0].metadata["continuity_id"] == continuity_id

    document = SpatialDocument(
        "1.0",
        DocumentMetadata("doc", "a" * 64),
        ParserMetadata("liteparse-table", "1"),
        [first, second],
    )
    chunks = context_chunks(structure_aware_chunks(document))
    linked = next(
        chunk for chunk in chunks if chunk.atomic_type == "cross-page-table"
    )
    assert linked.pages == [1, 2]
    assert linked.metadata["continuity_id"] == continuity_id


def test_new_caption_prevents_false_cross_page_link():
    first = Page(
        1,
        100,
        100,
        "pdf_points",
        [table_element(1, 20, 90, caption="Table 1")],
    )
    second = Page(
        2,
        100,
        100,
        "pdf_points",
        [table_element(2, 5, 90, caption="Table 2")],
    )
    assert link_cross_page_tables("doc", [first, second]) == 0


def test_normalization_audit_exposes_enrichment_without_losing_native_text():
    native_element = Element(
        element_id="p1-e1",
        page_number=1,
        element_type="paragraph",
        text="原始文字",
        markdown="原始文字",
        bbox=BBox(0, 0, 20, 10, "pdf_points"),
        reading_order=0,
        metadata={"shadowed_by_reconstructed_tables": [0]},
    )
    table = Element(
        element_id="p1-table",
        page_number=1,
        element_type="table",
        text="原始文字\t100",
        markdown="| 原始文字 | 100 |",
        bbox=BBox(0, 0, 100, 50, "pdf_points"),
        reading_order=1,
    )
    document = SpatialDocument(
        "1.0",
        DocumentMetadata("doc", "a" * 64),
        ParserMetadata("liteparse-table", "1"),
        [Page(1, 100, 100, "pdf_points", [native_element, table])],
    )
    audit = audit_normalization(
        {
            "base_ir_before_reconstruction": SpatialDocument(
                "1.0",
                document.document,
                ParserMetadata("liteparse", "1"),
                [
                    Page(
                        1,
                        100,
                        100,
                        "pdf_points",
                        [native_element],
                    )
                ],
            ).to_dict()
        },
        document,
    )
    assert audit["text_character_recall"] == 1.0
    assert audit["text_character_precision"] < 1.0
    assert audit["shadowed_native_elements"] == 1
    assert audit["ir_element_types"]["table"] == 1


def test_nested_table_trims_only_oversized_spanning_tail_row():
    bbox, rows, adjustment = trim_nested_spanning_tail(
        table_index=0,
        bbox=BBox(0, 0, 100, 100, "pdf_points"),
        rows=[
            ["h1", "h2"],
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
            ["g", "h"],
            ["swallowed prose and nested table", ""],
        ],
        row_bboxes=[
            (0, 0, 100, 10),
            (0, 10, 100, 20),
            (0, 20, 100, 30),
            (0, 30, 100, 40),
            (0, 40, 100, 50),
            (0, 50, 100, 100),
        ],
        row_nonempty_cell_counts=[2, 2, 2, 2, 2, 1],
        all_table_bboxes=[
            BBox(0, 0, 100, 100, "pdf_points"),
            BBox(10, 60, 90, 100, "pdf_points"),
        ],
    )
    assert adjustment["applied"] is True
    assert adjustment["removed_rows"] == 1
    assert len(rows) == 5
    assert bbox.y1 == 50


def test_table_router_scores_pages_and_audits_human_gold():
    caption = Element(
        "caption",
        1,
        "caption",
        "表 1 測試",
        "表 1 測試",
        BBox(10, 5, 90, 15, "pdf_points"),
        0,
    )
    table = Element(
        "table",
        1,
        "table",
        "\n".join(
            [
                "欄一\t欄二\t欄三",
                "甲\t1\t2",
                "乙\t3\t4",
                "丙\t5\t6",
                "丁\t7\t8",
                "戊\t9\t10",
            ]
        ),
        "| 欄一 | 欄二 | 欄三 |",
        BBox(10, 20, 90, 90, "pdf_points"),
        1,
        metadata={"row_count": 6, "column_count": 3},
    )
    positive_page = Page(
        1, 100, 100, "pdf_points", [caption, table]
    )
    negative_page = Page(
        2,
        100,
        100,
        "pdf_points",
        [
            Element(
                "paragraph",
                2,
                "paragraph",
                "只有段落",
                "只有段落",
                BBox(10, 10, 90, 20, "pdf_points"),
                0,
            )
        ],
    )
    document = SpatialDocument(
        "1.0",
        DocumentMetadata("doc", "a" * 64),
        ParserMetadata("pymupdf", "1"),
        [positive_page, negative_page],
    )
    assert score_page_for_table_routing(
        "doc", positive_page
    ).should_route
    assert not score_page_for_table_routing(
        "doc", negative_page
    ).should_route
    audit = evaluate_table_router(
        {"doc": document},
        [
            {
                "document_id": "doc",
                "page": 1,
                "should_route": True,
                "reason": "vector grid",
            },
            {
                "document_id": "doc",
                "page": 2,
                "should_route": False,
                "reason": "prose",
            },
        ],
    )
    assert audit["precision"] == 1.0
    assert audit["recall"] == 1.0
    bbox_audit = evaluate_reconstructed_table_bboxes(
        {"doc": document},
        [
            {
                "case_id": "table-bbox",
                "document_id": "doc",
                "dimension": "table_structure",
                "pages": [1],
                "anchor": "測試",
                "bbox_normalized": [0.1, 0.2, 0.9, 0.9],
            }
        ],
    )
    assert bbox_audit["mean_iou"] == 1.0

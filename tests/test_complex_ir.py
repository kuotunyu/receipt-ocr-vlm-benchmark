import json
from pathlib import Path

import jsonschema
import pytest

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.ir import (
    BBox,
    DocumentMetadata,
    Element,
    Page,
    ParserMetadata,
    SpatialDocument,
)


def sample_document() -> SpatialDocument:
    return SpatialDocument(
        schema_version="1.0",
        document=DocumentMetadata("doc", "a" * 64, "https://example.invalid/doc.pdf"),
        parser=ParserMetadata("fake", "1.2.3", {"mode": "test"}),
        pages=[
            Page(
                page_number=1,
                width=100,
                height=200,
                coordinate_space="pdf_points",
                elements=[
                    Element(
                        element_id="p1-e0000",
                        page_number=1,
                        element_type="heading",
                        text="第一節",
                        markdown="# 第一節",
                        bbox=BBox(10, 10, 90, 30, "pdf_points"),
                        reading_order=0,
                        parent_section_path=["第一節"],
                        confidence=0.9,
                        source_image_ref="screens/page-1.png",
                    )
                ],
            )
        ],
        parsing_timestamp="2026-07-29T00:00:00+00:00",
    )


def test_ir_round_trip_and_json_schema():
    document = sample_document()
    restored = SpatialDocument.from_json(document.to_json())
    assert restored == document
    schema = json.loads(
        Path("schema/spatial_document_ir.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(document.to_dict())


def test_bbox_validation():
    with pytest.raises(ValueError):
        BBox(0.8, 0.2, 0.1, 0.9, "normalized")
    with pytest.raises(ValueError):
        BBox(0, 0, 1.1, 1, "normalized")


def test_artifact_layers_are_separate(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    document = sample_document()
    raw_path = store.write_parser_raw("doc", "fake", {"native": True})
    ir_path = store.write_ir(document)
    assert "parser_raw" in raw_path.parts
    assert "ir" in ir_path.parts
    assert raw_path != ir_path
    assert json.loads(raw_path.read_text(encoding="utf-8"))["native"] is True

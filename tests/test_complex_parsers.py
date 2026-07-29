import json
from pathlib import Path

import pytest

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.parsers import (
    HybridTableRouterAdapter,
    LiteParseAdapter,
    LiteParseTableAdapter,
    ParseRequest,
    ParserUnavailable,
    PyMuPDFAdapter,
    QwenVLMParserAdapter,
)
from src.complex_document.parsers.llamaparse_parser import LlamaParseAdapter


@pytest.fixture()
def sample_pdf(tmp_path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "sample.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=400)
    page.insert_text((30, 40), "Section 1", fontsize=18)
    page.insert_text((30, 80), "Traditional Chinese structure benchmark")
    pdf.save(path)
    pdf.close()
    return path


def test_pymupdf_adapter_writes_native_and_ir(sample_pdf, tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    result = PyMuPDFAdapter(detect_tables=False).parse(
        ParseRequest(sample_pdf, "sample", pages=(1,)), store
    )
    assert result.parser.name == "pymupdf"
    assert result.pages[0].elements
    assert result.pages[0].screenshot_ref
    assert (
        tmp_path
        / "artifacts"
        / "parser_raw"
        / "sample"
        / "pymupdf"
        / "raw.json"
    ).is_file()
    assert (
        tmp_path
        / "artifacts"
        / "ir"
        / "sample"
        / "pymupdf"
        / "document.ir.json"
    ).is_file()


def test_liteparse_local_smoke(sample_pdf):
    pytest.importorskip("liteparse")
    result = LiteParseAdapter().parse(ParseRequest(sample_pdf, "sample", pages=(1,)))
    assert result.parser.name == "liteparse"
    assert result.pages[0].coordinate_space == "pdf_points"
    assert "structure benchmark" in result.plain_text()


def test_vlm_adapter_uses_rendered_pixels_and_normalizes(sample_pdf):
    calls = []

    def fake_call(image_bytes, prompt):
        calls.append((image_bytes, prompt))
        return json.dumps(
            {
                "elements": [
                    {
                        "type": "heading",
                        "text": "第一節",
                        "markdown": "# 第一節",
                        "bbox": [0.1, 0.1, 0.8, 0.2],
                        "confidence": 0.95,
                    }
                ]
            },
            ensure_ascii=False,
        )

    result = QwenVLMParserAdapter(call=fake_call).parse(
        ParseRequest(sample_pdf, "sample", pages=(1,))
    )
    assert calls and calls[0][0].startswith(b"\x89PNG")
    assert "原始頁面像素" in calls[0][1]
    assert result.pages[0].elements[0].bbox.coordinate_space == "normalized"
    assert result.pages[0].elements[0].element_type == "heading"


def test_vlm_table_text_falls_back_to_markdown(sample_pdf):
    def fake_call(_image_bytes, _prompt):
        return json.dumps(
            {
                "elements": [
                    {
                        "type": "table",
                        "text": "",
                        "markdown": "| 年 | 值 |\n|---|---|\n| 2024 | 42 |",
                        "bbox": [0.1, 0.2, 0.9, 0.8],
                        "confidence": 0.9,
                    }
                ]
            },
            ensure_ascii=False,
        )

    result = QwenVLMParserAdapter(call=fake_call).parse(
        ParseRequest(sample_pdf, "sample", pages=(1,))
    )
    table = result.pages[0].elements[0]
    assert "2024" in table.text
    assert table.metadata["text_filled_from_markdown"] is True


def test_vlm_ollama_call_disables_thinking_and_uses_schema(monkeypatch):
    ollama = pytest.importorskip("ollama")
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return {
            "response": '{"elements":[]}',
            "prompt_eval_count": 10,
            "eval_count": 4,
            "thinking": "",
            "done_reason": "stop",
        }

    monkeypatch.setattr(ollama, "generate", fake_generate)
    response, usage = QwenVLMParserAdapter(check_model=False)._call(b"png")
    assert response == '{"elements":[]}'
    assert captured["think"] is False
    assert captured["format"]["required"] == ["elements"]
    assert captured["options"]["num_predict"] == 6144
    assert usage["thinking_chars"] == 0
    assert usage["output_channel"] == "response"
    assert usage["gpu_seconds"] == 0


def test_vlm_ollama_call_accepts_thinking_channel_fallback(monkeypatch):
    ollama = pytest.importorskip("ollama")

    def fake_generate(**kwargs):
        return {
            "response": "",
            "thinking": '{"elements":[]}',
            "prompt_eval_count": 10,
            "eval_count": 4,
            "done_reason": "stop",
        }

    monkeypatch.setattr(ollama, "generate", fake_generate)
    response, usage = QwenVLMParserAdapter(check_model=False)._call(b"png")
    assert response == '{"elements":[]}'
    assert usage["output_channel"] == "thinking_fallback"


def test_vlm_refuses_competing_ollama_model(monkeypatch):
    ollama = pytest.importorskip("ollama")

    class LoadedModel:
        model = "other-project:latest"

    class ProcessResponse:
        models = [LoadedModel()]

    monkeypatch.setattr(ollama, "ps", lambda: ProcessResponse())
    with pytest.raises(ParserUnavailable, match="other-project"):
        QwenVLMParserAdapter(check_model=False).ensure_gpu_available()


def test_vlm_allows_its_own_loaded_model(monkeypatch):
    ollama = pytest.importorskip("ollama")

    class LoadedModel:
        model = "qwen3-vl:8b"

    class ProcessResponse:
        models = [LoadedModel()]

    monkeypatch.setattr(ollama, "ps", lambda: ProcessResponse())
    QwenVLMParserAdapter(check_model=False).ensure_gpu_available()


def test_llamaparse_is_optional_and_missing_key_is_skip(sample_pdf, monkeypatch):
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    with pytest.raises(ParserUnavailable, match="API_KEY"):
        LlamaParseAdapter().parse(ParseRequest(sample_pdf, "sample"))


def test_liteparse_table_reconstruction_adds_atomic_table(tmp_path):
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("liteparse")
    path = tmp_path / "table.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=240)
    page.insert_text((85, 40), "Table 1 Revenue")
    for x in (30, 140, 270):
        page.draw_line((x, 60), (x, 180), color=(0, 0, 0))
    for y in (60, 90, 120, 150, 180):
        page.draw_line((30, y), (270, y), color=(0, 0, 0))
    cells = [
        (45, 80, "Year"),
        (160, 80, "Revenue"),
        (45, 110, "2024"),
        (160, 110, "100"),
        (45, 140, "2025"),
        (160, 140, "120"),
    ]
    for x, y, text in cells:
        page.insert_text((x, y), text)
    pdf.save(path)
    pdf.close()

    result = LiteParseTableAdapter().parse(
        ParseRequest(path, "table", pages=(1,))
    )
    tables = [
        element
        for element in result.all_elements()
        if element.element_type == "table"
    ]
    assert tables
    assert tables[0].metadata["row_count"] >= 3
    assert tables[0].metadata["column_count"] == 2
    assert "Revenue" in tables[0].markdown
    assert any(
        element.metadata.get("shadowed_by_reconstructed_tables")
        for element in result.all_elements()
    )


def test_hybrid_table_router_selects_only_grid_page(tmp_path):
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("liteparse")
    path = tmp_path / "hybrid.pdf"
    pdf = fitz.open()
    table_page = pdf.new_page(width=300, height=240)
    table_page.insert_text((90, 30), "Table 1")
    for x in (30, 140, 270):
        table_page.draw_line((x, 50), (x, 200), color=(0, 0, 0))
    for y in (50, 80, 110, 140, 170, 200):
        table_page.draw_line((30, y), (270, y), color=(0, 0, 0))
    for row, values in enumerate(
        [("Year", "Value"), ("2022", "10"), ("2023", "20"), ("2024", "30")]
    ):
        table_page.insert_text((45, 72 + row * 30), values[0])
        table_page.insert_text((160, 72 + row * 30), values[1])
    prose_page = pdf.new_page(width=300, height=240)
    prose_page.insert_text((30, 50), "Plain prose page")
    pdf.save(path)
    pdf.close()

    result = HybridTableRouterAdapter().parse(
        ParseRequest(path, "hybrid", pages=(1, 2))
    )
    assert result.parser.name == "hybrid-table-router"
    assert result.parser.config["routed_pages"] == [1]
    routed_elements = [
        element
        for element in result.pages[0].elements
        if element.metadata["table_router_selected"]
    ]
    assert routed_elements
    assert all(
        element.element_type == "table" for element in routed_elements
    )
    assert any(
        not element.metadata["table_router_selected"]
        for element in result.pages[0].elements
    )
    assert all(
        not element.metadata["table_router_selected"]
        for element in result.pages[1].elements
    )

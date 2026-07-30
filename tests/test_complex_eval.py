import json
from pathlib import Path

import pytest

from src.complex_document.caption_index import (
    ChartCaption,
    answer_chart_from_original_crop,
    caption_chunk,
)
from src.complex_document.chunking import Chunk
from src.complex_document.downstream_eval import evaluate_downstream
from src.complex_document.ir import BBox
from src.complex_document.parser_metrics import evaluate_parser_case
from src.complex_document.normalization_audit import audit_normalization
from scripts.run_complex_benchmark import _apply_pixel_synthesis
from tests.test_complex_chunking import make_document


def base_chunk(text: str, *, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        text=text,
        markdown=text,
        pages=[1],
        bboxes=[BBox(0, 0, 1, 1, "normalized")],
        section_path=["第一節"],
        parser_name="fake",
        parser_version="1",
        element_ids=["e1"],
    )


def test_parser_metrics_are_human_rule_based():
    document = make_document()
    result = evaluate_parser_case(
        document,
        {
            "case_id": "case",
            "dimension": "table_structure",
            "anchor": "欄A",
            "minimum_rows": 1,
            "minimum_columns": 1,
            "expected_cells": ["欄B", "2"],
        },
    )
    # The fixture table lacks detector metadata, which is intentionally scored.
    assert result.score == 0.5
    assert not result.passed


def test_downstream_error_attribution_distinguishes_stages():
    questions = [
        {
            "question_id": "parse",
            "type": "table_cell",
            "question": "不存在證據",
            "answers": ["43"],
            "answer_regex": "(?P<answer>43)",
            "evidence": [{"document_id": "doc", "page": 1, "text_contains": ["缺字"]}],
        },
        {
            "question_id": "generation",
            "type": "table_cell",
            "question": "答案",
            "answers": ["99"],
            "answer_regex": "(?P<answer>42)",
            "evidence": [{"document_id": "doc", "page": 1, "text_contains": ["答案", "42"]}],
        },
    ]
    result = evaluate_downstream(questions, [base_chunk("答案是 42")], k=1)
    assert result["error_attribution"]["parsing"] == 1
    assert result["error_attribution"]["generation"] == 1


def test_cross_page_evidence_requires_one_chunk_covering_all_pages():
    chunk = base_chunk("2024 第二級毒品 3,599")
    chunk.pages = [54, 55]
    question = {
        "question_id": "cross-page",
        "type": "cross_page",
        "question": "2024 年第二級毒品破獲數？",
        "answers": ["3599"],
        "answer_regex": "第二級毒品\\s*(?P<answer>3[,]?599)",
        "evidence": [
            {
                "document_id": "doc",
                "pages": [54, 55],
                "text_contains": ["2024", "第二級毒品", "3,599"],
            }
        ],
    }
    result = evaluate_downstream([question], [chunk], k=1)
    assert result["retrieval_recall_at_k"] == 1.0
    assert result["citation_validity"] == 1.0


def test_cross_page_evidence_mode_all_covers_separate_source_chunks():
    revenue = base_chunk("113 年歲入淨額 4,159,577", chunk_id="revenue")
    revenue.pages = [45]
    financing = base_chunk("113 年政府融資淨收入 -79,682", chunk_id="financing")
    financing.pages = [47]
    question = {
        "question_id": "multi-source",
        "type": "cross_page",
        "question": "113 年歲入淨額加上政府融資淨收入是多少？",
        "answers": ["4079895"],
        "answer_regex": "(?P<answer>4[,]?159[,]?577|-79[,]?682)",
        "operation": "sum",
        "evidence_mode": "all",
        "evidence": [
            {
                "document_id": "doc",
                "page": 45,
                "text_contains": ["歲入淨額", "4,159,577"],
            },
            {
                "document_id": "doc",
                "page": 47,
                "text_contains": ["政府融資淨收入", "-79,682"],
            },
        ],
    }
    result = evaluate_downstream(
        [question], [revenue, financing], k=2
    )
    assert result["retrieval_recall_at_k"] == 1.0
    assert result["mrr"] == 0.5
    assert result["answer_correctness"] == 1.0
    assert result["citation_validity"] == 1.0


def test_cross_page_evidence_mode_all_detects_partial_retrieval():
    revenue = base_chunk("113 年歲入淨額 4,159,577", chunk_id="revenue")
    revenue.pages = [45]
    financing = base_chunk("113 年政府融資淨收入 -79,682", chunk_id="financing")
    financing.pages = [47]
    distractor = base_chunk("其他年度政府收支", chunk_id="distractor")
    question = {
        "question_id": "multi-source-partial",
        "type": "cross_page",
        "question": "113 年歲入淨額是多少？",
        "answers": ["4079895"],
        "answer_regex": "(?P<answer>4[,]?159[,]?577|-79[,]?682)",
        "operation": "sum",
        "evidence_mode": "all",
        "evidence": [
            {
                "document_id": "doc",
                "page": 45,
                "text_contains": ["歲入淨額", "4,159,577"],
            },
            {
                "document_id": "doc",
                "page": 47,
                "text_contains": ["政府融資淨收入", "-79,682"],
            },
        ],
    }
    result = evaluate_downstream(
        [question], [revenue, financing, distractor], k=1
    )
    assert result["retrieval_recall_at_k"] == 0.0
    assert result["citation_validity"] == 0.0
    assert result["error_attribution"]["retrieval"] == 1


def test_sum_answerer_deduplicates_overlapping_chunks_without_dropping_zeroes():
    from src.complex_document.answering import DeterministicAnswerer

    question = {
        "answer_regex": "(?P<answer>146,350|227,800)",
        "operation": "sum",
    }
    chunks = [base_chunk("146,350 227,800 146,350")]
    assert DeterministicAnswerer().answer(question, chunks) == "374150"


def test_caption_is_retrieval_only_and_synthesis_reads_crop(tmp_path):
    crop = tmp_path / "chart.png"
    crop.write_bytes(b"original-pixels")
    caption = ChartCaption(
        figure_id="fig1",
        generic_caption="一張圖表",
        structured_caption="X 軸年份；Y 軸億美元；經常帳 112 年為 1053.29",
        page_number=1,
        bbox=[0, 0, 1, 1],
        crop_ref=str(crop),
        axis_names=["年份", "經常帳"],
        unit="億美元",
        series=["經常帳"],
        values=["112:1053.29"],
        trend="上升",
    )
    indexed = caption_chunk(
        base_chunk("", chunk_id="figure"), caption, mode="structured_caption_original_crop"
    )
    assert indexed.metadata["caption_is_retrieval_only"] is True
    assert indexed.metadata["figure_id"] == "fig1"

    class FakeVision:
        def __init__(self):
            self.received = None

        def answer(self, *, question, image_bytes):
            self.received = image_bytes
            return "1053.29"

    vision = FakeVision()
    assert answer_chart_from_original_crop("112 年？", indexed, vision) == "1053.29"
    assert vision.received == b"original-pixels"


def test_caption_mode_requires_original_crop():
    hit = base_chunk("caption")
    with pytest.raises(ValueError):
        answer_chart_from_original_crop("question", hit, object())


def test_ollama_pixel_answerer_disables_thinking(monkeypatch):
    import ollama

    from src.complex_document.caption_index import OllamaPixelVisionAnswerer

    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return {"response": '{"answer":"1053.29"}'}

    monkeypatch.setattr(ollama, "generate", fake_generate)
    answer = OllamaPixelVisionAnswerer().answer(
        question="圖中數值？", image_bytes=b"pixels"
    )
    assert answer == "1053.29"
    assert captured["think"] is False
    assert captured["format"]["required"] == ["answer"]
    assert captured["options"]["num_predict"] == 512


def test_ollama_pixel_answerer_accepts_thinking_channel_fallback(monkeypatch):
    import ollama

    from src.complex_document.caption_index import OllamaPixelVisionAnswerer

    def fake_generate(**kwargs):
        return {"response": "", "thinking": '{"answer":"像素答案"}'}

    monkeypatch.setattr(ollama, "generate", fake_generate)
    answer = OllamaPixelVisionAnswerer().answer(
        question="問題", image_bytes=b"png"
    )
    assert answer == "像素答案"


def test_normalization_audit_reads_vlm_response():
    document = make_document()
    document.pages[0].elements = document.pages[0].elements[:1]
    document.pages[0].elements[0].text = "表格內容 42"
    native = {
        "pages": [
            {
                "page_number": 1,
                "response": json.dumps(
                    {
                        "elements": [
                            {
                                "type": "table",
                                "text": "",
                                "markdown": "表格內容 42",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
        ]
    }
    audit = audit_normalization(native, document)
    assert audit["native_item_count"] == 1


def test_normalization_audit_reads_paddle_ocr_boxes():
    document = make_document()
    document.pages[0].elements = document.pages[0].elements[:1]
    document.pages[0].elements[0].text = "第一行 第二行"
    native = {
        "pages": [
            {
                "page_number": 1,
                "ocr_boxes": [
                    {"text": "第一行", "bbox": [0, 0, 10, 10]},
                    {"text": "第二行", "bbox": [0, 12, 10, 22]},
                ],
            }
        ]
    }
    audit = audit_normalization(native, document)
    assert audit["native_item_count"] == 2
    assert audit["text_character_recall"] == 1.0
    assert audit["text_character_recall"] == 1.0


def test_saved_pixel_synthesis_requires_retrieved_original_crop():
    questions = [
        {
            "question_id": "q1",
            "answers": ["42"],
        },
        {
            "question_id": "q2",
            "answers": ["國家發展計畫"],
        },
    ]
    metrics = {
        "answerer": "deterministic",
        "questions": [
            {
                "question_id": "q1",
                "retrieved_chunk_ids": ["fig:q1"],
                "answer": None,
                "correct": False,
                "citation_valid": False,
                "error_source": "generation",
            },
            {
                "question_id": "q2",
                "retrieved_chunk_ids": ["unrelated"],
                "answer": None,
                "correct": False,
                "citation_valid": False,
                "error_source": "retrieval",
            },
        ],
    }
    result = _apply_pixel_synthesis(
        metrics,
        questions,
        {
            "q1": "圖中數值為 42。",
            "q2": "國家發展計畫",
        },
        {
            "q1": ["fig:q1"],
            "q2": ["fig:q2"],
        },
    )
    assert result["answerer"] == "qwen3-vl-original-crop-no-caption"
    assert result["answer_correctness"] == 0.5
    assert result["citation_validity"] == 0.5
    assert result["crop_recall_at_k"] == 0.5
    assert result["error_attribution"] == {
        "parsing": 0,
        "retrieval": 1,
        "generation": 0,
    }

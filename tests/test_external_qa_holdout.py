import re
import json
from pathlib import Path

from src.complex_document.answering import DeterministicAnswerer
from src.complex_document.chunking import Chunk
from src.complex_document.qa_holdout import (
    REQUIRED_QUESTION_TYPES,
    definition_sha256,
    read_json,
    validate_qa_holdout,
)


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="test",
        document_id="nia_2024_annual",
        text=text,
        markdown=text,
        pages=[158, 160, 172],
        bboxes=[],
        section_path=[],
        parser_name="test",
        parser_version="1",
        element_ids=[],
    )


def _definitions():
    return (
        read_json(Path("data/complex_document/qa_holdout/manifest.json")),
        read_json(Path("data/complex_document/qa_holdout/questions.json")),
    )


def test_external_qa_definition_is_frozen_and_complete():
    manifest, questions = _definitions()
    validate_qa_holdout(manifest, questions)
    assert questions["question_count"] == 15
    assert REQUIRED_QUESTION_TYPES.issubset(
        {question["type"] for question in questions["questions"]}
    )
    assert len(definition_sha256(manifest, questions)) == 64
    for question in questions["questions"]:
        if question.get("answer_regex"):
            re.compile(question["answer_regex"])


def test_external_qa_sum_gold_is_executable():
    _, payload = _definitions()
    answerer = DeterministicAnswerer()
    questions = {item["question_id"]: item for item in payload["questions"]}
    assert answerer.answer(
        questions["qh09"], [_chunk("花蓮機場 8,032 馬公機場 1,628")]
    ) == "9660"
    assert answerer.answer(
        questions["qh11"],
        [_chunk("金門 56,303 高雄 288,700 松山 67,613")],
    ) == "412616"
    assert answerer.answer(
        questions["qh13"], [_chunk("勞力剝削 41 性剝削 83 器官摘除 1")]
    ) == "125"


def test_targeted_fixed_result_is_labeled_posthoc_when_present():
    result_path = Path("results/complex_document/qa_holdout_summary.json")
    if not result_path.is_file():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    factor = next(
        (
            item
            for item in result["factor_at_a_time"]
            if item["factor"] == "targeted_vlm_fixed_posthoc"
        ),
        None,
    )
    if factor and factor["status"] == "completed":
        assert "post-hoc" in factor["analysis_role"]
        assert "not promotion evidence" in factor["analysis_role"]

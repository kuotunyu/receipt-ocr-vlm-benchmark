import copy
from pathlib import Path

import pytest

from src.complex_document.promotion_holdout import (
    PromotionProtocolError,
    promotion_decision,
    promotion_definition_sha256,
    validate_promotion_protocol,
)
from src.complex_document.qa_holdout import read_json, validate_qa_holdout
from src.complex_document.router_holdout import verify_holdout_sources
from scripts.run_promotion_caption_eval import _decision as caption_decision

ROOT = Path("data/complex_document/promotion_holdout")


def _definitions():
    return (
        read_json(ROOT / "manifest.json"),
        read_json(ROOT / "questions.json"),
        read_json(ROOT / "protocol.json"),
    )


def test_promotion_holdout_is_fresh_frozen_and_complete():
    manifest, questions, protocol = _definitions()
    validate_qa_holdout(manifest, questions)
    validate_promotion_protocol(protocol, manifest, questions)
    assert len(manifest["documents"]) == 2
    assert sum(
        len(document["selected_pages"])
        for document in manifest["documents"]
    ) == 14
    assert questions["question_count"] == 26
    assert len(promotion_definition_sha256(
        manifest, questions, protocol
    )) == 64

    development = read_json(Path("data/complex_document/manifest.json"))
    old_holdout = read_json(
        Path("data/complex_document/qa_holdout/manifest.json")
    )
    new_ids = {
        document["document_id"] for document in manifest["documents"]
    }
    prior_ids = {
        document["document_id"]
        for payload in (development, old_holdout)
        for document in payload["documents"]
    }
    assert new_ids.isdisjoint(prior_ids)


def test_promotion_protocol_rejects_post_prediction_retuning():
    manifest, questions, protocol = _definitions()
    changed = copy.deepcopy(protocol)
    changed["candidate"]["chunking"] = "structure"
    with pytest.raises(PromotionProtocolError):
        validate_promotion_protocol(changed, manifest, questions)


def test_promotion_holdout_has_true_multi_source_cross_page_gold():
    _, questions, _ = _definitions()
    cross_page = next(
        question
        for question in questions["questions"]
        if question["question_id"] == "ph24"
    )
    assert cross_page["evidence_mode"] == "all"
    assert {item["page"] for item in cross_page["evidence"]} == {45, 47}


def test_promotion_chart_targets_reference_frozen_sources_and_questions():
    manifest, questions, _ = _definitions()
    payload = read_json(ROOT / "chart_targets.json")
    targets = payload["targets"]
    selected = {
        document["document_id"]: set(document["selected_pages"])
        for document in manifest["documents"]
    }
    assert payload["freeze_status"] == "frozen-before-caption-generation"
    assert payload["evaluation_protocol"] == {
        "retrieval_k": 5,
        "minimum_question_count": 7,
        "minimum_crop_recall_at_k": 0.9,
        "minimum_answer_correctness": 0.8,
        "minimum_citation_validity": 0.8,
        "structured_retrieval_must_not_regress": True,
        "pixel_synthesis_must_execute": True,
    }
    question_ids = {
        question["question_id"] for question in questions["questions"]
    }
    assert len(targets) == 5
    assert sum(len(target["question_ids"]) for target in targets) == 7
    assert len({target["figure_id"] for target in targets}) == len(targets)
    for target in targets:
        assert target["document_id"] in selected
        assert target["page"] in selected[target["document_id"]]
        assert set(target["question_ids"]).issubset(question_ids)
        x0, y0, x1, y1 = target["bbox_normalized"]
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


def test_promotion_decision_uses_frozen_no_regression_gate():
    _, _, protocol = _definitions()
    baseline = {
        "question_count": 26,
        "retrieval_recall_at_k": 0.8,
        "mrr": 0.7,
        "answer_correctness": 0.6,
        "citation_validity": 0.6,
    }
    improved = {
        **baseline,
        "answer_correctness": 0.65,
        "citation_validity": 0.65,
    }
    assert promotion_decision(
        baseline, improved, protocol
    )["recommendation"] == "GO"

    regressed = {**improved, "mrr": 0.69}
    assert promotion_decision(
        baseline, regressed, protocol
    )["recommendation"] == "NO-GO"
    assert promotion_decision(
        baseline, None, protocol
    )["recommendation"] == "PENDING"


def test_promotion_caption_decision_uses_frozen_pixel_gates():
    _, _, _ = _definitions()
    protocol = read_json(
        ROOT / "chart_targets.json"
    )["evaluation_protocol"]
    result = {
        "status": "completed",
        "modes": {
            "no_image_indexing": {
                "retrieval_recall_at_k": 0.7,
            },
            "structured_caption_original_crop": {
                "retrieval_recall_at_k": 0.9,
                "pixel_synthesis_executed": True,
                "crop_recall_at_k": 1.0,
                "answer_correctness": 0.86,
                "citation_validity": 0.86,
                "question_count": 7,
            },
        },
    }
    assert caption_decision(
        result, protocol
    )["recommendation"] == "GO"
    result["modes"]["structured_caption_original_crop"][
        "citation_validity"
    ] = 0.71
    assert caption_decision(
        result, protocol
    )["recommendation"] == "NO-GO"


def test_promotion_downloads_match_manifest_when_present():
    manifest, _, _ = _definitions()
    raw_dir = ROOT / "raw"
    if not all(
        (raw_dir / document["filename"]).is_file()
        for document in manifest["documents"]
    ):
        return
    verified = verify_holdout_sources(manifest, raw_dir)
    assert set(verified) == {
        document["document_id"] for document in manifest["documents"]
    }

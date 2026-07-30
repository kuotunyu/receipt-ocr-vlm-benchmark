import copy
from pathlib import Path

import pytest

from scripts.run_llamaparse_comparator import (
    _cached_cloud_ir_complete,
    _decision as llamaparse_decision,
)
from scripts.run_promotion_caption_eval import (
    _decision as caption_decision,
)
from src.complex_document.promotion_holdout import (
    PromotionProtocolError,
    promotion_decision,
    promotion_definition_sha256,
    validate_promotion_protocol,
)
from src.complex_document.qa_holdout import read_json, validate_qa_holdout
from src.complex_document.router_holdout import verify_holdout_sources


ROOT = Path("data/complex_document/scale_validation")


def _definitions():
    return (
        read_json(ROOT / "manifest.json"),
        read_json(ROOT / "questions.json"),
        read_json(ROOT / "protocol.json"),
    )


def test_scale_validation_is_disclosed_disjoint_and_complete():
    manifest, questions, protocol = _definitions()
    validate_qa_holdout(manifest, questions)
    validate_promotion_protocol(protocol, manifest, questions)
    assert len(manifest["documents"]) == 3
    assert sum(
        len(document["selected_pages"])
        for document in manifest["documents"]
    ) == 24
    assert questions["question_count"] == 39
    assert len(
        promotion_definition_sha256(manifest, questions, protocol)
    ) == 64

    prior_manifests = [
        Path("data/complex_document/manifest.json"),
        Path("data/complex_document/holdout/manifest.json"),
        Path("data/complex_document/qa_holdout/manifest.json"),
        Path("data/complex_document/promotion_holdout/manifest.json"),
    ]
    new_ids = {
        document["document_id"] for document in manifest["documents"]
    }
    prior_ids = {
        document["document_id"]
        for path in prior_manifests
        for document in read_json(path)["documents"]
    }
    assert new_ids.isdisjoint(prior_ids)


def test_scale_validation_rejects_missing_selection_disclosure():
    manifest, questions, protocol = _definitions()
    changed = copy.deepcopy(protocol)
    changed["selection_disclosure"] = ""
    with pytest.raises(PromotionProtocolError):
        validate_promotion_protocol(changed, manifest, questions)


def test_scale_validation_is_never_promotion_evidence():
    _, _, protocol = _definitions()
    baseline = {
        "question_count": 39,
        "retrieval_recall_at_k": 0.7,
        "mrr": 0.6,
        "answer_correctness": 0.6,
        "citation_validity": 0.6,
    }
    candidate = {
        **baseline,
        "retrieval_recall_at_k": 0.8,
        "mrr": 0.7,
        "answer_correctness": 0.7,
        "citation_validity": 0.7,
    }
    decision = promotion_decision(baseline, candidate, protocol)
    assert decision["recommendation"] == "NOT-PROMOTION-EVIDENCE"
    assert decision["promotion_eligible"] is False
    assert decision["scale_finding"] == "SUPPORTS-CANDIDATE"


def test_scale_chart_targets_reference_registered_sources_and_questions():
    manifest, questions, _ = _definitions()
    payload = read_json(ROOT / "chart_targets.json")
    selected = {
        document["document_id"]: set(document["selected_pages"])
        for document in manifest["documents"]
    }
    question_ids = {
        question["question_id"] for question in questions["questions"]
    }
    assert payload["freeze_status"] == "frozen-before-caption-generation"
    assert len(payload["targets"]) == 8
    assert sum(
        len(target["question_ids"]) for target in payload["targets"]
    ) == 9
    for target in payload["targets"]:
        assert target["document_id"] in selected
        assert target["page"] in selected[target["document_id"]]
        assert set(target["question_ids"]).issubset(question_ids)
        x0, y0, x1, y1 = target["bbox_normalized"]
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1


def test_scale_downloads_match_manifest_when_present():
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


def test_llamaparse_comparator_is_descriptive_only():
    baseline = {
        "retrieval_recall_at_k": 0.7,
        "mrr": 0.6,
        "answer_correctness": 0.5,
        "citation_validity": 0.5,
    }
    decision = llamaparse_decision(
        baseline,
        {
            **baseline,
            "retrieval_recall_at_k": 0.8,
        },
    )
    assert decision["recommendation"] == "DESCRIPTIVE-ONLY"
    assert decision["promotion_eligible"] is False
    assert decision["scale_finding"] == (
        "MATCHES-OR-IMPROVES-LOCAL-BASELINE"
    )


def test_llamaparse_cached_reverification_requires_every_document(tmp_path):
    manifest, _, _ = _definitions()
    assert not _cached_cloud_ir_complete(manifest, tmp_path)
    for item in manifest["documents"]:
        path = (
            tmp_path
            / "ir"
            / item["document_id"]
            / "llamaparse-cloud"
            / "document.ir.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
    assert _cached_cloud_ir_complete(manifest, tmp_path)


def test_scale_caption_result_is_never_promotion_evidence():
    protocol = read_json(ROOT / "chart_targets.json")[
        "evaluation_protocol"
    ]
    result = {
        "status": "completed",
        "modes": {
            "no_image_indexing": {"retrieval_recall_at_k": 0.2},
            "structured_caption_original_crop": {
                "retrieval_recall_at_k": 1.0,
                "pixel_synthesis_executed": True,
                "crop_recall_at_k": 1.0,
                "answer_correctness": 1.0,
                "citation_validity": 1.0,
                "question_count": 9,
            },
        },
    }
    decision = caption_decision(
        result, protocol, promotion_eligible=False
    )
    assert decision["recommendation"] == "NOT-PROMOTION-EVIDENCE"
    assert decision["promotion_eligible"] is False
    assert decision["scale_finding"] == "SUPPORTS-CAPTION-AND-INDEX"

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_and_gold_are_small_real_benchmark():
    manifest = json.loads(
        Path("data/complex_document/manifest.json").read_text(encoding="utf-8")
    )
    gold = json.loads(
        Path("data/complex_document/gold/hard_cases.json").read_text(encoding="utf-8")
    )
    assert 5 <= len(manifest["documents"]) <= 10
    assert len(gold["cases"]) >= 30
    assert gold["case_count"] == len(gold["cases"])
    assert "Human" in gold["annotation_method"]
    assert all(document["license_note"] for document in manifest["documents"])
    assert all(len(document["sha256"]) == 64 for document in manifest["documents"])


def test_local_raw_documents_match_manifest_when_present():
    manifest = json.loads(
        Path("data/complex_document/manifest.json").read_text(encoding="utf-8")
    )
    for document in manifest["documents"]:
        path = Path("data/complex_document/raw") / document["filename"]
        if path.exists():
            assert file_sha256(path) == document["sha256"]


def test_questions_cover_required_types():
    payload = json.loads(
        Path("data/complex_document/questions.json").read_text(encoding="utf-8")
    )
    questions = payload["questions"]
    types = {question["type"] for question in questions}
    assert {
        "single_text_fact",
        "table_cell",
        "table_aggregation",
        "chart_value",
        "cross_page",
        "unanswerable",
    } <= types
    assert len(questions) == 14


def test_chart_targets_are_human_verified_and_reference_questions():
    questions = {
        question["question_id"]
        for question in json.loads(
            Path("data/complex_document/questions.json").read_text(
                encoding="utf-8"
            )
        )["questions"]
    }
    payload = json.loads(
        Path("data/complex_document/gold/chart_targets.json").read_text(
            encoding="utf-8"
        )
    )
    assert "Human-verified" in payload["annotation_method"]
    assert len(payload["targets"]) == 3
    assert sum(
        len(target["question_ids"]) for target in payload["targets"]
    ) == 4
    assert all(
        question_id in questions
        for target in payload["targets"]
        for question_id in target["question_ids"]
    )


def test_table_routing_gold_covers_every_selected_page():
    manifest = json.loads(
        Path("data/complex_document/manifest.json").read_text(encoding="utf-8")
    )
    routing = json.loads(
        Path(
            "data/complex_document/gold/table_routing_pages.json"
        ).read_text(encoding="utf-8")
    )
    expected = {
        (document["document_id"], page)
        for document in manifest["documents"]
        for page in document["selected_pages"]
    }
    actual = {
        (item["document_id"], item["page"]) for item in routing["pages"]
    }
    assert actual == expected
    assert len(routing["pages"]) == 26
    assert sum(item["should_route"] for item in routing["pages"]) == 13
    assert "Manual" in routing["annotation_method"]
    hard_cases = json.loads(
        Path(
            "data/complex_document/gold/hard_cases.json"
        ).read_text(encoding="utf-8")
    )
    questions = json.loads(
        Path("data/complex_document/questions.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        manifest["benchmark_version"],
        routing["benchmark_version"],
        hard_cases["benchmark_version"],
        questions["benchmark_version"],
    } == {"0.3.0"}

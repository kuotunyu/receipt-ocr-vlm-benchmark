"""Validation helpers for the frozen external downstream QA holdout."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.complex_document.ir import SpatialDocument

REQUIRED_QUESTION_TYPES = {
    "single_text_fact",
    "table_cell",
    "table_aggregation",
    "chart_value",
    "cross_page",
    "unanswerable",
}


class QAHoldoutDefinitionError(ValueError):
    """The external QA manifest or manual annotations are inconsistent."""


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def definition_sha256(manifest: dict, questions: dict) -> str:
    payload = json.dumps(
        {"manifest": manifest, "questions": questions},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_qa_holdout(manifest: dict, questions_payload: dict) -> None:
    if manifest.get("role") != "external-end-to-end-qa-holdout":
        raise QAHoldoutDefinitionError("manifest role is not an external QA holdout")
    if manifest.get("annotation_status") != "frozen-before-downstream-evaluation":
        raise QAHoldoutDefinitionError("QA annotations must be frozen before evaluation")
    if manifest.get("benchmark_version") != questions_payload.get(
        "benchmark_version"
    ):
        raise QAHoldoutDefinitionError("manifest and question versions differ")
    if manifest.get("development_document_overlap") != 0:
        raise QAHoldoutDefinitionError("development document overlap must be zero")

    selected_pages = {
        document["document_id"]: {
            int(page) for page in document.get("selected_pages", [])
        }
        for document in manifest.get("documents", [])
    }
    if not selected_pages:
        raise QAHoldoutDefinitionError("manifest has no documents")
    if len(selected_pages) != len(manifest["documents"]):
        raise QAHoldoutDefinitionError("duplicate document_id in manifest")

    questions = questions_payload.get("questions", [])
    if questions_payload.get("question_count") != len(questions):
        raise QAHoldoutDefinitionError("question_count does not match annotations")
    question_ids = [question.get("question_id") for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise QAHoldoutDefinitionError("duplicate question_id")
    present_types = {question.get("type") for question in questions}
    if not REQUIRED_QUESTION_TYPES.issubset(present_types):
        missing = sorted(REQUIRED_QUESTION_TYPES - present_types)
        raise QAHoldoutDefinitionError(
            f"required question types are missing: {missing}"
        )

    for question in questions:
        unanswerable = bool(question.get("unanswerable"))
        if unanswerable:
            if question.get("answers") or question.get("evidence"):
                raise QAHoldoutDefinitionError(
                    f"{question['question_id']} unanswerable gold is not empty"
                )
            continue
        if not question.get("answers"):
            raise QAHoldoutDefinitionError(
                f"{question['question_id']} has no accepted answer"
            )
        if not question.get("answer_regex"):
            raise QAHoldoutDefinitionError(
                f"{question['question_id']} has no answer_regex"
            )
        re.compile(question["answer_regex"])
        evidence_items = question.get("evidence", [])
        if not evidence_items:
            raise QAHoldoutDefinitionError(
                f"{question['question_id']} has no evidence"
            )
        for evidence in evidence_items:
            document_id = evidence.get("document_id")
            if document_id not in selected_pages:
                raise QAHoldoutDefinitionError(
                    f"{question['question_id']} references unknown document"
                )
            pages = (
                {int(evidence["page"])}
                if evidence.get("page") is not None
                else {int(page) for page in evidence.get("pages", [])}
            )
            if not pages or not pages.issubset(selected_pages[document_id]):
                raise QAHoldoutDefinitionError(
                    f"{question['question_id']} evidence page is not selected"
                )
            if not evidence.get("text_contains"):
                raise QAHoldoutDefinitionError(
                    f"{question['question_id']} evidence has no text anchors"
                )


def load_qa_ir(
    manifest: dict,
    artifact_root: str | Path,
    parser_name: str,
) -> dict[str, SpatialDocument]:
    root = Path(artifact_root)
    documents: dict[str, SpatialDocument] = {}
    for item in manifest["documents"]:
        path = (
            root
            / "ir"
            / item["document_id"]
            / parser_name
            / "document.ir.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing QA holdout IR: {path}")
        documents[item["document_id"]] = SpatialDocument.from_json(
            path.read_text(encoding="utf-8")
        )
    return documents

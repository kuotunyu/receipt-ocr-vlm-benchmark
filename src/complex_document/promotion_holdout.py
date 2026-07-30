"""Frozen promotion protocol for a fresh targeted-VLM QA holdout."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.complex_document.qa_holdout import QAHoldoutDefinitionError

PRIMARY_METRICS = (
    "retrieval_recall_at_k",
    "mrr",
    "answer_correctness",
    "citation_validity",
)
PROMOTION_ROLE = "targeted-vlm-fixed-promotion"
SCALE_VALIDATION_ROLE = "targeted-vlm-fixed-scale-validation"


class PromotionProtocolError(QAHoldoutDefinitionError):
    """The promotion protocol is incomplete or no longer comparable."""


def promotion_definition_sha256(
    manifest: dict[str, Any],
    questions: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "manifest": manifest,
            "questions": questions,
            "protocol": protocol,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_promotion_protocol(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    questions_payload: dict[str, Any],
) -> None:
    role = protocol.get("role")
    if role not in {PROMOTION_ROLE, SCALE_VALIDATION_ROLE}:
        raise PromotionProtocolError("unexpected promotion protocol role")
    expected_freeze_status = (
        "frozen-before-predictions"
        if role == PROMOTION_ROLE
        else "frozen-before-scoring"
    )
    if protocol.get("freeze_status") != expected_freeze_status:
        raise PromotionProtocolError(
            f"{role} requires freeze_status={expected_freeze_status}"
        )
    if role == SCALE_VALIDATION_ROLE and not protocol.get(
        "selection_disclosure"
    ):
        raise PromotionProtocolError(
            "scale validation must disclose how pages and gold were selected"
        )
    if protocol.get("benchmark_version") != manifest.get(
        "benchmark_version"
    ) or protocol.get("benchmark_version") != questions_payload.get(
        "benchmark_version"
    ):
        raise PromotionProtocolError(
            "protocol, manifest, and questions must use one benchmark version"
        )
    if manifest.get("development_document_overlap") != 0:
        raise PromotionProtocolError("promotion documents must be disjoint")

    baseline = protocol.get("baseline", {})
    candidate = protocol.get("candidate", {})
    if baseline != {"parser": "pymupdf", "chunking": "fixed"}:
        raise PromotionProtocolError("baseline must remain PyMuPDF + fixed")
    if candidate.get("parser") != "targeted-vlm":
        raise PromotionProtocolError("candidate parser must remain targeted-vlm")
    if candidate.get("chunking") != "fixed":
        raise PromotionProtocolError("candidate chunking must remain fixed")
    if candidate.get("router_version") != "native-visual-router-1":
        raise PromotionProtocolError("targeted VLM router version changed")

    metrics = tuple(protocol.get("primary_metrics", []))
    if metrics != PRIMARY_METRICS:
        raise PromotionProtocolError(
            "primary metrics or their frozen order changed"
        )
    if int(protocol.get("retrieval_k", 0)) != 5:
        raise PromotionProtocolError("retrieval K must remain 5")

    minimum_questions = int(protocol.get("minimum_question_count", 0))
    questions = questions_payload.get("questions", [])
    if minimum_questions < 20 or len(questions) < minimum_questions:
        raise PromotionProtocolError(
            "fresh promotion holdout requires at least 20 questions"
        )
    required_types = set(protocol.get("required_question_types", []))
    present_types = {question.get("type") for question in questions}
    if not required_types.issubset(present_types):
        raise PromotionProtocolError(
            f"promotion question types missing: "
            f"{sorted(required_types - present_types)}"
        )

    rule = protocol.get("promotion_rule", {})
    if rule != {
        "all_primary_metrics_at_least_baseline": True,
        "at_least_one_primary_metric_strictly_improves": True,
    }:
        raise PromotionProtocolError("promotion rule changed")


def promotion_decision(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply only the gates frozen before the candidate predictions."""
    promotion_eligible = protocol.get("role") == PROMOTION_ROLE
    if baseline is None or candidate is None:
        return {
            "recommendation": "PENDING",
            "promotion_eligible": False,
            "reason": "Both frozen baseline and candidate must complete.",
        }

    metrics = tuple(protocol["primary_metrics"])
    deltas = {
        metric: round(candidate[metric] - baseline[metric], 6)
        for metric in metrics
    }
    minimum_questions = int(protocol["minimum_question_count"])
    gates = {
        **{
            f"{metric}_at_least_baseline": deltas[metric] >= 0
            for metric in metrics
        },
        "one_primary_metric_strictly_improves": any(
            delta > 0 for delta in deltas.values()
        ),
        "minimum_question_count_evaluated": (
            baseline.get("question_count", 0) >= minimum_questions
            and candidate.get("question_count", 0) >= minimum_questions
            and baseline.get("question_count")
            == candidate.get("question_count")
        ),
    }
    go = all(gates.values())
    if not promotion_eligible:
        return {
            "recommendation": "NOT-PROMOTION-EVIDENCE",
            "promotion_eligible": False,
            "scale_finding": (
                "SUPPORTS-CANDIDATE" if go else "DOES-NOT-SUPPORT-CANDIDATE"
            ),
            "rule": (
                "Apply the frozen no-regression gates descriptively, but do "
                "not promote from this source-assisted scale-validation set."
            ),
            "deltas_vs_baseline": deltas,
            "gates": gates,
        }
    return {
        "recommendation": "GO" if go else "NO-GO",
        "promotion_eligible": promotion_eligible,
        "rule": (
            "Promote targeted VLM + fixed chunks only if every primary metric "
            "matches or exceeds PyMuPDF + fixed chunks, at least one primary "
            "metric strictly improves, and the full frozen holdout is scored."
        ),
        "deltas_vs_baseline": deltas,
        "gates": gates,
    }

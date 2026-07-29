"""Frozen external holdout utilities for the vector-grid table page router."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.complex_document.ir import SpatialDocument
from src.complex_document.routing import (
    DEFAULT_TABLE_ROUTE_THRESHOLD,
    TABLE_ROUTER_VERSION,
    evaluate_table_router,
)

SENSITIVITY_THRESHOLDS = (0.50, 0.56, 0.62, 0.68, 0.74)
ROUTING_VALIDATION_MINIMUM = 0.90


class HoldoutDefinitionError(ValueError):
    """The manifest and manual gold do not describe the same frozen holdout."""


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selected_page_keys(manifest: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (document["document_id"], int(page))
        for document in manifest["documents"]
        for page in document["selected_pages"]
    }


def validate_holdout_definition(
    manifest: dict[str, Any], gold: dict[str, Any]
) -> None:
    frozen = manifest.get("router_frozen", {})
    gold_frozen = gold.get("frozen_router", {})
    if frozen.get("version") != TABLE_ROUTER_VERSION:
        raise HoldoutDefinitionError("manifest router version is not frozen to code")
    if gold_frozen.get("version") != TABLE_ROUTER_VERSION:
        raise HoldoutDefinitionError("gold router version is not frozen to code")
    if frozen.get("threshold") != DEFAULT_TABLE_ROUTE_THRESHOLD:
        raise HoldoutDefinitionError("manifest threshold is not the frozen threshold")
    if gold_frozen.get("threshold") != DEFAULT_TABLE_ROUTE_THRESHOLD:
        raise HoldoutDefinitionError("gold threshold is not the frozen threshold")
    if frozen.get("retuning_allowed") is not False:
        raise HoldoutDefinitionError("manifest must prohibit holdout retuning")
    if gold_frozen.get("retuning_allowed") is not False:
        raise HoldoutDefinitionError("gold must prohibit holdout retuning")
    if manifest.get("benchmark_version") != gold.get("benchmark_version"):
        raise HoldoutDefinitionError("manifest and gold versions differ")

    document_ids = [
        document["document_id"] for document in manifest.get("documents", [])
    ]
    if len(document_ids) != len(set(document_ids)):
        raise HoldoutDefinitionError("duplicate document_id in manifest")

    expected = _selected_page_keys(manifest)
    page_keys = [
        (item["document_id"], int(item["page"]))
        for item in gold.get("pages", [])
    ]
    if len(page_keys) != len(set(page_keys)):
        raise HoldoutDefinitionError("duplicate page label in gold")
    if set(page_keys) != expected:
        raise HoldoutDefinitionError("gold must cover every selected page exactly once")
    if not any(item["should_route"] for item in gold["pages"]):
        raise HoldoutDefinitionError("holdout has no positive routing pages")
    if not any(not item["should_route"] for item in gold["pages"]):
        raise HoldoutDefinitionError("holdout has no negative routing pages")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_holdout_sources(
    manifest: dict[str, Any], raw_dir: str | Path
) -> dict[str, dict[str, Any]]:
    raw_path = Path(raw_dir)
    verified: dict[str, dict[str, Any]] = {}
    for document in manifest["documents"]:
        source = raw_path / document["filename"]
        if not source.is_file():
            raise FileNotFoundError(
                f"missing holdout PDF for {document['document_id']}: {source}"
            )
        actual_bytes = source.stat().st_size
        actual_sha256 = _sha256(source)
        if actual_bytes != int(document["bytes"]):
            raise RuntimeError(
                f"byte-size mismatch for {document['document_id']}: {actual_bytes}"
            )
        if actual_sha256 != document["sha256"]:
            raise RuntimeError(
                f"checksum mismatch for {document['document_id']}: {actual_sha256}"
            )
        verified[document["document_id"]] = {
            "status": "verified",
            "filename": document["filename"],
            "bytes": actual_bytes,
            "sha256": actual_sha256,
        }
    return verified


def load_holdout_ir(
    manifest: dict[str, Any], artifact_root: str | Path
) -> dict[str, SpatialDocument]:
    root = Path(artifact_root)
    documents: dict[str, SpatialDocument] = {}
    for item in manifest["documents"]:
        document_id = item["document_id"]
        path = root / "ir" / document_id / "pymupdf" / "document.ir.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing holdout IR: {path}")
        documents[document_id] = SpatialDocument.from_json(
            path.read_text(encoding="utf-8")
        )
    return documents


def _compact_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if key != "cases"
    }


def evaluate_router_holdout(
    documents: dict[str, SpatialDocument],
    gold_pages: list[dict[str, Any]],
    *,
    threshold: float = DEFAULT_TABLE_ROUTE_THRESHOLD,
    sensitivity_thresholds: tuple[float, ...] = SENSITIVITY_THRESHOLDS,
) -> dict[str, Any]:
    if threshold != DEFAULT_TABLE_ROUTE_THRESHOLD:
        raise HoldoutDefinitionError(
            "primary holdout evaluation must use the frozen threshold"
        )

    primary = evaluate_table_router(
        documents, gold_pages, threshold=threshold
    )
    document_ids = sorted({item["document_id"] for item in gold_pages})
    per_document = {}
    for document_id in document_ids:
        evaluation = evaluate_table_router(
            {document_id: documents[document_id]},
            [
                item
                for item in gold_pages
                if item["document_id"] == document_id
            ],
            threshold=threshold,
        )
        per_document[document_id] = _compact_metrics(evaluation)

    sensitivity = []
    for candidate_threshold in sensitivity_thresholds:
        evaluation = evaluate_table_router(
            documents, gold_pages, threshold=candidate_threshold
        )
        sensitivity.append(_compact_metrics(evaluation))

    gates = {
        "precision_at_least_0_90": (
            primary["precision"] >= ROUTING_VALIDATION_MINIMUM
        ),
        "recall_at_least_0_90": (
            primary["recall"] >= ROUTING_VALIDATION_MINIMUM
        ),
        "accuracy_at_least_0_90": (
            primary["accuracy"] >= ROUTING_VALIDATION_MINIMUM
        ),
    }
    return {
        "protocol": {
            "primary_threshold": threshold,
            "router": TABLE_ROUTER_VERSION,
            "retuning_allowed": False,
            "sensitivity_is_descriptive_only": True,
        },
        "primary": primary,
        "per_document": per_document,
        "sensitivity": sensitivity,
        "validation": {
            "recommendation": "PASS" if all(gates.values()) else "FAIL",
            "minimum": ROUTING_VALIDATION_MINIMUM,
            "gates": gates,
            "scope": (
                "Router generalization only. This result does not override "
                "the downstream MRR promotion gate."
            ),
        },
    }

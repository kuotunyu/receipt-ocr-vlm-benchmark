"""Validate the local-only Traditional Chinese real-receipt add-on.

No image, label, or manifest content is uploaded or copied into tracked
artifacts.  The detailed verification report defaults to the ignored tmp/
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import read_json  # noqa: E402
from src.common.normalize import normalize_record  # noqa: E402
from src.common.schema import validate_record  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REQUIRED_CHALLENGES = {"handwriting": 2, "stamp_occlusion": 2}


class RealReceiptValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(
    raw_dir: Path,
    labels_dir: Path,
    manifest_path: Path,
    *,
    minimum_documents: int = 5,
    maximum_documents: int = 10,
    minimum_long_side: int = 1600,
) -> dict:
    manifest = read_json(manifest_path)
    if manifest.get("privacy_reviewed") is not True:
        raise RealReceiptValidationError(
            "manifest privacy_reviewed must be true"
        )
    if manifest.get("local_processing_only") is not True:
        raise RealReceiptValidationError(
            "manifest local_processing_only must be true"
        )
    items = manifest.get("items")
    if not isinstance(items, list):
        raise RealReceiptValidationError("manifest items must be a list")
    if not minimum_documents <= len(items) <= maximum_documents:
        raise RealReceiptValidationError(
            f"expected {minimum_documents}-{maximum_documents} receipts, "
            f"found {len(items)}"
        )

    filenames = [item.get("filename") for item in items]
    if any(
        not isinstance(filename, str)
        or Path(filename).name != filename
        or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS
        for filename in filenames
    ):
        raise RealReceiptValidationError(
            "each filename must be one supported image basename"
        )
    if len(filenames) != len(set(filenames)):
        raise RealReceiptValidationError("manifest filenames must be unique")

    challenge_counts: dict[str, int] = {}
    verified_items = []
    for item, filename in zip(items, filenames):
        if item.get("contains_personal_data") is not False:
            raise RealReceiptValidationError(
                f"{filename}: contains_personal_data must be false after review"
            )
        tags = item.get("challenge_tags")
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) and tag for tag in tags
        ):
            raise RealReceiptValidationError(
                f"{filename}: challenge_tags must be non-empty strings"
            )
        for tag in set(tags):
            challenge_counts[tag] = challenge_counts.get(tag, 0) + 1

        image_path = raw_dir / filename
        label_path = labels_dir / f"{Path(filename).stem}.json"
        if not image_path.is_file():
            raise RealReceiptValidationError(f"missing image: {image_path}")
        if not label_path.is_file():
            raise RealReceiptValidationError(f"missing label: {label_path}")

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            raise RealReceiptValidationError(
                f"{filename}: unreadable image: {exc}"
            ) from exc
        if max(width, height) < minimum_long_side:
            raise RealReceiptValidationError(
                f"{filename}: long side {max(width, height)} is below "
                f"{minimum_long_side}px"
            )

        label = read_json(label_path)
        errors = validate_record(label)
        if errors:
            raise RealReceiptValidationError(
                f"{label_path}: schema errors: {'; '.join(errors)}"
            )
        if normalize_record(label) != label:
            raise RealReceiptValidationError(
                f"{label_path}: label is not in canonical normalized form"
            )
        verified_items.append(
            {
                "id": item.get("id"),
                "filename": filename,
                "sha256": _sha256(image_path),
                "width": width,
                "height": height,
                "challenge_tags": sorted(set(tags)),
                "label_schema_valid": True,
            }
        )

    missing_challenges = {
        challenge: minimum
        for challenge, minimum in REQUIRED_CHALLENGES.items()
        if challenge_counts.get(challenge, 0) < minimum
    }
    if missing_challenges:
        raise RealReceiptValidationError(
            "insufficient required challenge coverage: "
            + ", ".join(
                f"{challenge}>={minimum}"
                for challenge, minimum in missing_challenges.items()
            )
        )

    return {
        "dataset_version": manifest.get("dataset_version"),
        "status": "verified-local-only",
        "document_count": len(items),
        "challenge_counts": dict(sorted(challenge_counts.items())),
        "items": verified_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw")
    )
    parser.add_argument(
        "--labels-dir", type=Path, default=Path("data/labels")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/real_receipts_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/real_receipt_verification.json"),
    )
    args = parser.parse_args()
    report = validate_dataset(
        args.raw_dir, args.labels_dir, args.manifest
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"verified {report['document_count']} local receipts; "
        f"report: {args.output}"
    )


if __name__ == "__main__":
    main()

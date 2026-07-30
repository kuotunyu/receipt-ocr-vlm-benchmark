"""Download the public Traditional Chinese receipt add-on.

The repository keeps only source metadata, checksums, and human labels.  The
CC-licensed source images are materialized into the ignored ``data/raw``
directory and are never required to be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DOWNLOAD_HOSTS = {"upload.wikimedia.org"}
ALLOWED_SOURCE_HOSTS = {"commons.wikimedia.org"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
USER_AGENT = (
    "TraditionalChineseReceiptBenchmark/1.0 "
    "(research dataset materializer; checksum verified)"
)


class PublicReceiptDownloadError(ValueError):
    """Raised when provenance or downloaded bytes fail validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_https_host(url: str, allowed_hosts: set[str], field: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise PublicReceiptDownloadError(
            f"{field} must use HTTPS on one of: {sorted(allowed_hosts)}"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise PublicReceiptDownloadError(
            f"{field} must not contain credentials, query, or fragment"
        )


def validate_public_manifest(manifest: dict) -> list[dict]:
    if manifest.get("privacy_reviewed") is not True:
        raise PublicReceiptDownloadError("privacy_reviewed must be true")
    if manifest.get("local_processing_only") is not True:
        raise PublicReceiptDownloadError("local_processing_only must be true")
    if manifest.get("redistribution_policy") != "metadata-only":
        raise PublicReceiptDownloadError(
            "redistribution_policy must be metadata-only"
        )

    items = manifest.get("items")
    if not isinstance(items, list) or not 5 <= len(items) <= 10:
        raise PublicReceiptDownloadError("manifest must contain 5-10 items")

    seen_filenames: set[str] = set()
    seen_label_files: set[str] = set()
    for item in items:
        filename = item.get("filename")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS
        ):
            raise PublicReceiptDownloadError(
                "each filename must be one supported image basename"
            )
        if filename in seen_filenames:
            raise PublicReceiptDownloadError(
                f"duplicate filename in manifest: {filename}"
            )
        seen_filenames.add(filename)

        label_file = item.get("label_file")
        if (
            not isinstance(label_file, str)
            or Path(label_file).name != label_file
            or Path(label_file).suffix.lower() != ".json"
        ):
            raise PublicReceiptDownloadError(
                f"{filename}: label_file must be one JSON basename"
            )
        if label_file in seen_label_files:
            raise PublicReceiptDownloadError(
                f"duplicate label_file in manifest: {label_file}"
            )
        seen_label_files.add(label_file)

        if item.get("contains_personal_data") is not False:
            raise PublicReceiptDownloadError(
                f"{filename}: contains_personal_data must be false"
            )
        if item.get("privacy_review_status") != "human-reviewed":
            raise PublicReceiptDownloadError(
                f"{filename}: privacy review must be human-reviewed"
            )
        tags = item.get("challenge_tags")
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) and tag for tag in tags
        ):
            raise PublicReceiptDownloadError(
                f"{filename}: challenge_tags must be non-empty strings"
            )

        source = item.get("source")
        if not isinstance(source, dict):
            raise PublicReceiptDownloadError(
                f"{filename}: source metadata is required"
            )
        _validate_https_host(
            source.get("page_url", ""),
            ALLOWED_SOURCE_HOSTS,
            f"{filename}: source.page_url",
        )
        _validate_https_host(
            source.get("download_url", ""),
            ALLOWED_DOWNLOAD_HOSTS,
            f"{filename}: source.download_url",
        )
        if not source.get("author") or not source.get("license"):
            raise PublicReceiptDownloadError(
                f"{filename}: source author and license are required"
            )
        checksum = source.get("sha256")
        if not isinstance(checksum, str) or not SHA256_PATTERN.fullmatch(
            checksum
        ):
            raise PublicReceiptDownloadError(
                f"{filename}: source.sha256 must be lowercase SHA-256"
            )
        if not all(
            isinstance(source.get(key), int) and source[key] > 0
            for key in ("width", "height")
        ):
            raise PublicReceiptDownloadError(
                f"{filename}: positive source width and height are required"
            )

    return items


def _open_with_retry(
    request: Request,
    *,
    opener: Callable = urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
    attempts: int = 4,
    timeout: float = 60.0,
):
    for attempt in range(attempts):
        try:
            return opener(request, timeout=timeout)
        except HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            wait_seconds = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else float(2 ** attempt)
            )
            sleep_fn(min(wait_seconds, 30.0))
    raise AssertionError("unreachable")


def download_dataset(
    manifest_path: Path,
    raw_dir: Path,
    *,
    opener: Callable = urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = validate_public_manifest(manifest)
    raw_dir.mkdir(parents=True, exist_ok=True)

    report_items = []
    for item in items:
        filename = item["filename"]
        source = item["source"]
        destination = raw_dir / filename
        expected_sha256 = source["sha256"]

        if destination.exists():
            actual_sha256 = _sha256(destination)
            if actual_sha256 != expected_sha256:
                raise PublicReceiptDownloadError(
                    f"{filename}: existing file checksum mismatch; "
                    "move it aside explicitly before retrying"
                )
            try:
                with Image.open(destination) as image:
                    image.verify()
                with Image.open(destination) as image:
                    actual_size = image.size
            except Exception as exc:
                raise PublicReceiptDownloadError(
                    f"{filename}: existing file is not a readable image"
                ) from exc
            expected_size = (source["width"], source["height"])
            if actual_size != expected_size:
                raise PublicReceiptDownloadError(
                    f"{filename}: expected {expected_size}, got "
                    f"{actual_size}"
                )
            status = "reused"
        else:
            request = Request(
                source["download_url"],
                headers={"User-Agent": USER_AGENT},
            )
            temp_path: Path | None = None
            try:
                with _open_with_retry(
                    request, opener=opener, sleep_fn=sleep_fn
                ) as response:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        dir=raw_dir,
                        prefix=f".{filename}.",
                        suffix=".part",
                        delete=False,
                    ) as handle:
                        temp_path = Path(handle.name)
                        while block := response.read(1024 * 1024):
                            handle.write(block)
                actual_sha256 = _sha256(temp_path)
                if actual_sha256 != expected_sha256:
                    raise PublicReceiptDownloadError(
                        f"{filename}: downloaded checksum mismatch"
                    )
                with Image.open(temp_path) as image:
                    image.verify()
                with Image.open(temp_path) as image:
                    actual_size = image.size
                expected_size = (source["width"], source["height"])
                if actual_size != expected_size:
                    raise PublicReceiptDownloadError(
                        f"{filename}: expected {expected_size}, got "
                        f"{actual_size}"
                    )
                os.replace(temp_path, destination)
                temp_path = None
                status = "downloaded"
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()

        with Image.open(destination) as image:
            width, height = image.size
        report_items.append(
            {
                "id": item["id"],
                "filename": filename,
                "status": status,
                "sha256": expected_sha256,
                "width": width,
                "height": height,
                "source_page_url": source["page_url"],
                "license": source["license"],
                "author": source["author"],
            }
        )

    return {
        "dataset_version": manifest.get("dataset_version"),
        "status": "materialized-public-source",
        "document_count": len(report_items),
        "items": report_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/public_receipts_manifest.json"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/public_receipt_download.json"),
    )
    args = parser.parse_args()
    report = download_dataset(args.manifest, args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"materialized {report['document_count']} public receipts; "
        f"report: {args.output}"
    )


if __name__ == "__main__":
    main()

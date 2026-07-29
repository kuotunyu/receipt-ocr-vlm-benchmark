"""Download pinned public documents without redistributing PDFs in the repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(document: dict, output_dir: Path, force: bool) -> str:
    target = output_dir / document["filename"]
    if target.exists() and not force:
        actual = sha256(target)
        if actual == document["sha256"]:
            return "verified"
        raise RuntimeError(
            f"{target} exists with checksum {actual}; use --force to replace"
        )
    request = urllib.request.Request(
        document["url"], headers={"User-Agent": "structure-benchmark/0.1"}
    )
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            temporary.write_bytes(response.read())
        actual = sha256(temporary)
        if actual != document["sha256"]:
            raise RuntimeError(
                f"checksum mismatch for {document['document_id']}: {actual}"
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/complex_document/manifest.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/complex_document/raw")
    )
    parser.add_argument("--document-id", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = set(args.document_id or [])
    documents = [
        document
        for document in manifest["documents"]
        if not requested or document["document_id"] in requested
    ]
    missing = requested - {document["document_id"] for document in documents}
    if missing:
        raise SystemExit(f"unknown document IDs: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for document in documents:
        status = download_one(document, args.output_dir, args.force)
        print(f"{document['document_id']}: {status}")


if __name__ == "__main__":
    main()

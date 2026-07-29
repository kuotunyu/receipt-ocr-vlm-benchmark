"""Layered artifact storage for parser-native output and normalized IR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.complex_document.ir import SpatialDocument


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    parser_raw: Path
    ir: Path
    screenshots: Path
    crops: Path


class ArtifactStore:
    def __init__(self, root: str | Path = "artifacts/complex_document"):
        root_path = Path(root)
        self.paths = ArtifactPaths(
            root=root_path,
            parser_raw=root_path / "parser_raw",
            ir=root_path / "ir",
            screenshots=root_path / "screenshots",
            crops=root_path / "crops",
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in value
        )

    def _artifact_dir(self, layer: Path, document_id: str, parser_name: str) -> Path:
        target = layer / self._safe_name(document_id) / self._safe_name(parser_name)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def write_parser_raw(
        self,
        document_id: str,
        parser_name: str,
        payload: dict[str, Any] | list[Any] | str,
    ) -> Path:
        target = self._artifact_dir(
            self.paths.parser_raw, document_id, parser_name
        ) / "raw.json"
        serializable = payload if not isinstance(payload, str) else {"raw_text": payload}
        target.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return target

    def write_ir(self, document: SpatialDocument) -> Path:
        target = self._artifact_dir(
            self.paths.ir, document.document.document_id, document.parser.name
        ) / "document.ir.json"
        target.write_text(document.to_json(), encoding="utf-8")
        return target

    def screenshot_dir(self, document_id: str, parser_name: str) -> Path:
        return self._artifact_dir(self.paths.screenshots, document_id, parser_name)

    def crop_dir(self, document_id: str, parser_name: str) -> Path:
        return self._artifact_dir(self.paths.crops, document_id, parser_name)

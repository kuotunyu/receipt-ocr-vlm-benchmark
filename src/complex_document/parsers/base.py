from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.ir import SpatialDocument, sha256_file


class ParserUnavailable(RuntimeError):
    """Expected skip condition such as a missing optional dependency/model/key."""


@dataclass(frozen=True)
class ParseRequest:
    path: Path
    document_id: str
    source_uri: str | None = None
    pages: tuple[int, ...] | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def checksum(self) -> str:
        return sha256_file(self.path)


class DocumentParserAdapter(ABC):
    name: str

    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        raise NotImplementedError

    @staticmethod
    def selected_pages(total: int, requested: tuple[int, ...] | None) -> list[int]:
        if requested is None:
            return list(range(1, total + 1))
        selected = sorted(set(requested))
        invalid = [page for page in selected if page < 1 or page > total]
        if invalid:
            raise ValueError(f"page selection outside document: {invalid}")
        return selected

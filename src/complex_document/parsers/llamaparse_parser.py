"""Optional commercial comparator.

This module is deliberately not imported by the parser package.  Both the SDK
and LLAMA_CLOUD_API_KEY are optional, and absence is a normal skip condition.
"""

from __future__ import annotations

import importlib.metadata
import os

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.ir import (
    DocumentMetadata,
    Element,
    Page,
    ParserMetadata,
    SpatialDocument,
)
from src.complex_document.parsers.base import (
    DocumentParserAdapter,
    ParseRequest,
    ParserUnavailable,
)


class LlamaParseAdapter(DocumentParserAdapter):
    name = "llamaparse-cloud"

    def version(self) -> str:
        try:
            return importlib.metadata.version("llama-cloud")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ParserUnavailable(
                "install the optional 'llamaparse' project extra"
            ) from exc

    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise ParserUnavailable("LLAMA_CLOUD_API_KEY is not set")
        try:
            from llama_cloud import LlamaCloud
        except ImportError as exc:
            raise ParserUnavailable(
                "install the optional 'llamaparse' project extra"
            ) from exc

        client = LlamaCloud(api_key=api_key)
        uploaded = client.files.create(file=request.path, purpose="parse")
        native_result = client.parsing.parse(
            file_id=uploaded.id,
            tier=request.config.get("tier", "agentic"),
            version=request.config.get("version", "latest"),
            expand=["text", "markdown", "items"],
        )
        markdown = str(native_result.markdown or native_result.text or "")
        page = Page(
            page_number=1,
            width=1.0,
            height=1.0,
            coordinate_space="normalized",
            elements=[
                Element(
                    element_id="p1-e0000",
                    page_number=1,
                    element_type="paragraph",
                    text=markdown,
                    markdown=markdown,
                    bbox=None,
                    reading_order=0,
                    confidence=None,
                    metadata={
                        "limitation": "legacy SDK markdown response has no stable bbox"
                    },
                )
            ],
        )
        result = SpatialDocument(
            schema_version="1.0",
            document=DocumentMetadata(
                document_id=request.document_id,
                checksum_sha256=request.checksum(),
                source_uri=request.source_uri,
            ),
            parser=ParserMetadata(
                name=self.name,
                version=self.version(),
                config={
                    "tier": request.config.get("tier", "agentic"),
                    "version": request.config.get("version", "latest"),
                    "expand": ["text", "markdown", "items"],
                    **request.config,
                },
            ),
            pages=[page],
        )
        if artifacts:
            artifacts.write_parser_raw(
                request.document_id,
                self.name,
                {
                    "file_id": uploaded.id,
                    "text": native_result.text,
                    "markdown": native_result.markdown,
                    "items": native_result.items,
                },
            )
            artifacts.write_ir(result)
        return result

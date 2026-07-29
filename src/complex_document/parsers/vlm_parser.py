from __future__ import annotations

import importlib.metadata
import json
import re
import time
from typing import Any

from src.complex_document.artifacts import ArtifactStore
from src.complex_document.ir import (
    BBox,
    DocumentMetadata,
    Element,
    Page,
    ParserMetadata,
    SpatialDocument,
)
from src.complex_document.ollama_compat import extract_ollama_text
from src.complex_document.parsers.base import (
    DocumentParserAdapter,
    ParseRequest,
    ParserUnavailable,
)
from src.complex_document.parsers.heuristics import infer_section_paths

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_ALLOWED_TYPES = {
    "heading",
    "paragraph",
    "table",
    "figure",
    "caption",
    "footnote",
    "list",
}
_PAGE_PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": sorted(_ALLOWED_TYPES),
                    },
                    "text": {"type": "string"},
                    "markdown": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": [
                    "type",
                    "text",
                    "markdown",
                    "bbox",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["elements"],
    "additionalProperties": False,
}

_PAGE_PARSE_PROMPT = """你是繁體中文文件版面解析器。請直接分析原始頁面像素。
只輸出一個 JSON object，不要 code fence：
{"elements":[{"type":"heading|paragraph|table|figure|caption|footnote|list",
"text":"逐字內容","markdown":"保留表格/清單結構的 Markdown",
"bbox":[x0,y0,x1,y1],"confidence":0.0}]}

bbox 使用 0 到 1 的頁面正規化座標。依人類閱讀順序排列 elements。
表格須保留列欄，且 table 的 text 與 markdown 都必須包含儲存格文字。
圖表本體 type=figure，圖說另列 caption。
看不清楚的文字不要臆測；confidence 反映不確定性。"""


def _parse_json_object(text: str) -> dict[str, Any]:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("VLM response does not contain a JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("VLM response root must be an object")
    return value


class QwenVLMParserAdapter(DocumentParserAdapter):
    """Page-level Qwen3-VL parser, separate from the receipt schema pipeline."""

    name = "qwen3-vl-parser"

    def __init__(
        self,
        *,
        model: str = "qwen3-vl:8b",
        dpi: int = 160,
        call=None,
        check_model: bool = True,
        allow_shared_gpu: bool = False,
    ):
        self.model = model
        self.dpi = dpi
        self._injected_call = call
        self.check_model = check_model
        self.allow_shared_gpu = allow_shared_gpu

    def version(self) -> str:
        if self._injected_call is not None:
            return "injected-test-double"
        try:
            return f"ollama-{importlib.metadata.version('ollama')}:{self.model}"
        except importlib.metadata.PackageNotFoundError as exc:
            raise ParserUnavailable("ollama Python client is not installed") from exc

    def _ensure_model(self) -> None:
        if self._injected_call is not None or not self.check_model:
            return
        try:
            import ollama

            response = ollama.list()
            models = getattr(response, "models", None)
            if models is None and isinstance(response, dict):
                models = response.get("models", [])
            names = set()
            for model in models or []:
                if isinstance(model, dict):
                    names.add(str(model.get("model") or model.get("name") or ""))
                else:
                    names.add(str(getattr(model, "model", getattr(model, "name", ""))))
            if self.model not in names:
                raise ParserUnavailable(
                    f"required local model {self.model!r} is not installed"
                )
        except ParserUnavailable:
            raise
        except Exception as exc:
            raise ParserUnavailable(f"Ollama is unavailable: {exc}") from exc

    def ensure_gpu_available(self) -> None:
        """Refuse to load alongside another Ollama model unless explicitly allowed."""
        if self._injected_call is not None or self.allow_shared_gpu:
            return
        try:
            import ollama

            response = ollama.ps()
            models = getattr(response, "models", None)
            if models is None and isinstance(response, dict):
                models = response.get("models", [])
            loaded_names = set()
            for model in models or []:
                if isinstance(model, dict):
                    loaded_names.add(
                        str(model.get("model") or model.get("name") or "")
                    )
                else:
                    loaded_names.add(
                        str(
                            getattr(
                                model,
                                "model",
                                getattr(model, "name", ""),
                            )
                        )
                    )
            competing = sorted(
                name
                for name in loaded_names
                if name and name != self.model
            )
            if competing:
                raise ParserUnavailable(
                    "GPU/Ollama busy with other model(s): "
                    + ", ".join(competing)
                )
        except ParserUnavailable:
            raise
        except Exception as exc:
            raise ParserUnavailable(
                f"cannot verify Ollama GPU availability: {exc}"
            ) from exc

    def _call(self, image_bytes: bytes) -> tuple[str, dict[str, Any]]:
        if self._injected_call is not None:
            return str(self._injected_call(image_bytes, _PAGE_PARSE_PROMPT)), {}
        import ollama

        response = ollama.generate(
            model=self.model,
            prompt=_PAGE_PARSE_PROMPT,
            images=[image_bytes],
            think=False,
            format=_PAGE_PARSE_SCHEMA,
            options={"temperature": 0, "num_predict": 6144},
        )
        text, output_channel = extract_ollama_text(response)
        if isinstance(response, dict):
            return text, {
                "prompt_tokens": response.get("prompt_eval_count"),
                "output_tokens": response.get("eval_count"),
                "done_reason": response.get("done_reason"),
                "thinking_chars": len(str(response.get("thinking") or "")),
                "output_channel": output_channel,
                "gpu_seconds": (
                    (response.get("prompt_eval_duration") or 0)
                    + (response.get("eval_duration") or 0)
                )
                / 1e9,
            }
        return text, {
            "prompt_tokens": getattr(response, "prompt_eval_count", None),
            "output_tokens": getattr(response, "eval_count", None),
            "done_reason": getattr(response, "done_reason", None),
            "thinking_chars": len(str(getattr(response, "thinking", "") or "")),
            "output_channel": output_channel,
            "gpu_seconds": (
                (getattr(response, "prompt_eval_duration", 0) or 0)
                + (getattr(response, "eval_duration", 0) or 0)
            )
            / 1e9,
        }

    def parse(
        self, request: ParseRequest, artifacts: ArtifactStore | None = None
    ) -> SpatialDocument:
        try:
            import fitz
        except ImportError as exc:
            raise ParserUnavailable("PyMuPDF is required for VLM page rendering") from exc
        self._ensure_model()
        self.ensure_gpu_available()

        pages: list[Page] = []
        raw_pages: list[dict[str, Any]] = []
        with fitz.open(request.path) as pdf:
            selected = self.selected_pages(pdf.page_count, request.pages)
            for page_number in selected:
                self.ensure_gpu_available()
                source_page = pdf[page_number - 1]
                pixmap = source_page.get_pixmap(dpi=self.dpi, alpha=False)
                image_bytes = pixmap.tobytes("png")
                started = time.perf_counter()
                raw_response, usage = self._call(image_bytes)
                latency = time.perf_counter() - started
                parsed = _parse_json_object(raw_response)
                elements: list[Element] = []
                for raw_element in parsed.get("elements", []):
                    if not isinstance(raw_element, dict):
                        continue
                    element_type = str(raw_element.get("type", "paragraph")).lower()
                    if element_type not in _ALLOWED_TYPES:
                        element_type = "paragraph"
                    bbox_values = raw_element.get("bbox")
                    bbox = None
                    if isinstance(bbox_values, list) and len(bbox_values) == 4:
                        clipped = [min(1.0, max(0.0, float(v))) for v in bbox_values]
                        bbox = BBox(*clipped, coordinate_space="normalized")
                    confidence = raw_element.get("confidence")
                    if confidence is not None:
                        confidence = min(1.0, max(0.0, float(confidence)))
                    text = str(raw_element.get("text") or "").strip()
                    markdown = str(raw_element.get("markdown") or text).strip()
                    text_filled_from_markdown = False
                    if element_type == "table" and not text and markdown:
                        text = markdown
                        text_filled_from_markdown = True
                    if not text and element_type not in {"figure", "table"}:
                        continue
                    elements.append(
                        Element(
                            element_id=f"p{page_number}-e{len(elements):04d}",
                            page_number=page_number,
                            element_type=element_type,
                            text=text,
                            markdown=markdown,
                            bbox=bbox,
                            reading_order=len(elements),
                            confidence=confidence,
                            metadata={
                                "text_filled_from_markdown": (
                                    text_filled_from_markdown
                                )
                            },
                        )
                    )
                infer_section_paths(elements)

                screenshot_ref = None
                if artifacts:
                    screenshot_path = (
                        artifacts.screenshot_dir(request.document_id, self.name)
                        / f"page-{page_number:04d}.png"
                    )
                    screenshot_path.write_bytes(image_bytes)
                    screenshot_ref = str(screenshot_path.as_posix())
                    for element in elements:
                        element.source_image_ref = screenshot_ref

                pages.append(
                    Page(
                        page_number=page_number,
                        width=float(pixmap.width),
                        height=float(pixmap.height),
                        coordinate_space="normalized",
                        elements=elements,
                        screenshot_ref=screenshot_ref,
                    )
                )
                raw_pages.append(
                    {
                        "page_number": page_number,
                        "latency_seconds": latency,
                        "usage": usage,
                        "response": raw_response,
                    }
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
                    "model": self.model,
                    "dpi": self.dpi,
                    "temperature": 0,
                    "think": False,
                    "structured_output": "json-schema",
                    "num_predict": 6144,
                    "allow_shared_gpu": self.allow_shared_gpu,
                    **request.config,
                },
            ),
            pages=pages,
        )
        if artifacts:
            artifacts.write_parser_raw(
                request.document_id, self.name, {"pages": raw_pages}
            )
            artifacts.write_ir(result)
        return result

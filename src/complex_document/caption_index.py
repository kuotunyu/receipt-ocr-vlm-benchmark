from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.complex_document.chunking import Chunk
from src.complex_document.ollama_compat import extract_ollama_text

GENERIC_CAPTION_SCHEMA = {
    "type": "object",
    "properties": {"caption": {"type": "string"}},
    "required": ["caption"],
    "additionalProperties": False,
}
PIXEL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _json_field(text: str, field: str) -> str:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"VLM response does not contain JSON field {field!r}")
    value = json.loads(match.group(0))
    result = value.get(field)
    if not isinstance(result, str):
        raise ValueError(f"VLM response field {field!r} is not a string")
    return result.strip()


@dataclass(frozen=True)
class ChartCaption:
    figure_id: str
    generic_caption: str
    structured_caption: str
    page_number: int
    bbox: list[float]
    crop_ref: str
    axis_names: list[str]
    unit: str | None
    series: list[str]
    values: list[str]
    trend: str | None


class PixelVisionAnswerer(Protocol):
    def answer(self, *, question: str, image_bytes: bytes) -> str:
        ...


class OllamaPixelVisionAnswerer:
    """Answer from the crop only; indexed caption is never included in the prompt."""

    def __init__(self, model: str = "qwen3-vl:8b"):
        self.model = model

    def answer(self, *, question: str, image_bytes: bytes) -> str:
        import ollama

        response = ollama.generate(
            model=self.model,
            prompt=(
                "請只根據這張原始圖表 crop 的像素回答問題。"
                "若像素不足以回答，請回答「無法判讀」。\n問題："
                + question
            ),
            images=[image_bytes],
            think=False,
            format=PIXEL_ANSWER_SCHEMA,
            options={"temperature": 0, "num_predict": 512},
        )
        text, _ = extract_ollama_text(response)
        return _json_field(text, "answer")


def caption_chunk(
    base: Chunk,
    caption: ChartCaption,
    *,
    mode: str,
) -> Chunk | None:
    if mode == "no_image_indexing":
        return None
    if mode == "generic_caption":
        text = caption.generic_caption
    elif mode in {"structured_caption", "structured_caption_original_crop"}:
        text = caption.structured_caption
    else:
        raise ValueError(f"unsupported caption indexing mode: {mode}")
    return Chunk(
        chunk_id=f"{base.chunk_id}:caption:{mode}",
        document_id=base.document_id,
        text=text,
        markdown=text,
        pages=[caption.page_number],
        bboxes=list(base.bboxes),
        section_path=list(base.section_path),
        parser_name=base.parser_name,
        parser_version=base.parser_version,
        element_ids=list(base.element_ids),
        source_image_refs=[caption.crop_ref],
        kind="context",
        atomic_type="chart-caption-index",
        metadata={
            "figure_id": caption.figure_id,
            "caption_mode": mode,
            "caption_is_retrieval_only": True,
            "original_crop_ref": caption.crop_ref,
        },
    )


def answer_chart_from_original_crop(
    question: str,
    caption_hit: Chunk,
    answerer: PixelVisionAnswerer,
) -> str:
    """Use caption for retrieval, but prohibit caption-only answer synthesis."""
    crop_ref = caption_hit.metadata.get("original_crop_ref")
    if not crop_ref:
        raise ValueError("retrieved chart caption has no original crop reference")
    crop_path = Path(crop_ref)
    if not crop_path.is_file():
        raise FileNotFoundError(crop_path)
    return answerer.answer(question=question, image_bytes=crop_path.read_bytes())

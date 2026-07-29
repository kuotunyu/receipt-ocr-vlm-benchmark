from __future__ import annotations

import re
from collections.abc import Iterable

_CAPTION_RE = re.compile(r"^\s*(?:圖|表)\s*[0-9一二三四五六七八九十IVXivx\-–—.]+")
_FOOTNOTE_RE = re.compile(r"^\s*(?:註|注|備註|資料來源|來源)[:：\s]|\(\s*註\s*\d+\s*\)")
_LIST_RE = re.compile(
    r"^\s*(?:[\-•●▪]|[0-9一二三四五六七八九十]+[.)、．]|[（(][0-9一二三四五六七八九十]+[）)])"
)
_HEADING_RE = re.compile(
    r"^\s*(?:第[一二三四五六七八九十百0-9]+[章節篇]|[0-9]+(?:\.[0-9]+)*\s+\S+)"
)


def classify_text(
    text: str,
    *,
    font_size: float | None = None,
    median_font_size: float | None = None,
) -> str:
    clean = text.strip()
    if not clean:
        return "paragraph"
    if _CAPTION_RE.search(clean) and len(clean) <= 120:
        return "caption"
    if _FOOTNOTE_RE.search(clean) and len(clean) <= 240:
        return "footnote"
    if _LIST_RE.search(clean):
        return "list"
    if _HEADING_RE.search(clean) and len(clean) <= 100:
        return "heading"
    if (
        font_size is not None
        and median_font_size is not None
        and font_size >= median_font_size * 1.28
        and len(clean) <= 100
    ):
        return "heading"
    return "paragraph"


def infer_section_paths(elements: Iterable) -> None:
    section_stack: list[str] = []
    for element in elements:
        if element.element_type == "heading":
            heading = element.text.strip()
            if heading:
                section_stack = [heading]
        element.parent_section_path = list(section_stack)

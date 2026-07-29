from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from src.complex_document.chunking import Chunk


class DeterministicAnswerer:
    """Small extractive answerer used to isolate parser/retriever changes."""

    name = "gold-regex-extractive-v1"

    def answer(self, question: dict, chunks: list[Chunk]) -> str | None:
        if question.get("unanswerable"):
            return None
        text = "\n".join(chunk.text for chunk in chunks)
        pattern = question.get("answer_regex")
        if not pattern:
            return None
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if not matches:
            return None
        operation = question.get("operation", "first")
        values = [
            match.group("answer") if "answer" in match.groupdict() else match.group(0)
            for match in matches
        ]
        if operation == "first":
            return values[0].strip()
        if operation == "unique_join":
            return "、".join(dict.fromkeys(value.strip() for value in values))
        if operation == "sum":
            try:
                unique_values = list(
                    dict.fromkeys(re.sub(r"[,，%\s]", "", value) for value in values)
                )
                total = sum(
                    Decimal(value) for value in unique_values
                )
            except InvalidOperation:
                return None
            if total == total.to_integral_value():
                return str(int(total))
            return format(total.normalize(), "f")
        raise ValueError(f"unsupported answer operation: {operation}")

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.complex_document.chunking import Chunk


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _features(text: str, n: int = 2) -> Counter[str]:
    normalized = _normalize(text)
    if len(normalized) < n:
        return Counter([normalized]) if normalized else Counter()
    return Counter(normalized[index : index + n] for index in range(len(normalized) - n + 1))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _query_coverage(
    query: Counter[str], candidate: Counter[str]
) -> float:
    total = sum(query.values())
    if not total:
        return 0.0
    matched = sum(
        min(count, candidate.get(feature, 0))
        for feature, count in query.items()
    )
    return matched / total


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int


class CharNgramRetriever:
    """Dependency-free CPU retriever held constant across factor experiments."""

    name = "char-bigram-cosine-v1"

    def retrieve(self, query: str, chunks: list[Chunk], k: int = 5) -> list[RetrievalHit]:
        query_features = _features(query)
        scored = [
            (_cosine(query_features, _features(chunk.text)), index, chunk)
            for index, chunk in enumerate(chunks)
        ]
        scored.sort(key=lambda value: (-value[0], value[1]))
        return [
            RetrievalHit(chunk=chunk, score=score, rank=rank)
            for rank, (score, _, chunk) in enumerate(scored[:k], start=1)
        ]


@dataclass(frozen=True)
class WindowedCharNgramRetriever:
    """Late-max score for long atomic chunks using the same char-bigram features."""

    window_size: int = 320
    overlap: int = 80
    global_weight: float = 0.25
    coverage_weight: float = 0.50

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.overlap < 0 or self.overlap >= self.window_size:
            raise ValueError("require window_size > overlap >= 0")
        if not 0 <= self.global_weight <= 1:
            raise ValueError("global_weight must be between zero and one")
        if not 0 <= self.coverage_weight <= 1:
            raise ValueError("coverage_weight must be between zero and one")

    @property
    def name(self) -> str:
        return (
            "char-bigram-windowed-v1:"
            f"w{self.window_size}:o{self.overlap}:"
            f"g{self.global_weight:.2f}:c{self.coverage_weight:.2f}"
        )

    def _window_texts(self, chunk: Chunk) -> list[str]:
        prefix = " ".join(chunk.section_path)
        text = f"{prefix}\n{chunk.text}".strip()
        if len(text) <= self.window_size:
            return [text]
        step = self.window_size - self.overlap
        return [
            text[start : start + self.window_size]
            for start in range(0, len(text), step)
            if text[start : start + self.window_size]
        ]

    def retrieve(
        self, query: str, chunks: list[Chunk], k: int = 5
    ) -> list[RetrievalHit]:
        query_features = _features(query)
        scored = []
        for index, chunk in enumerate(chunks):
            global_score = _cosine(query_features, _features(chunk.text))
            local_scores = []
            for window in self._window_texts(chunk):
                window_features = _features(window)
                local_scores.append(
                    (1 - self.coverage_weight)
                    * _cosine(query_features, window_features)
                    + self.coverage_weight
                    * _query_coverage(query_features, window_features)
                )
            local_score = max(local_scores, default=0.0)
            score = (
                self.global_weight * global_score
                + (1 - self.global_weight) * local_score
            )
            scored.append((score, index, chunk))
        scored.sort(key=lambda value: (-value[0], value[1]))
        return [
            RetrievalHit(chunk=chunk, score=score, rank=rank)
            for rank, (score, _, chunk) in enumerate(scored[:k], start=1)
        ]

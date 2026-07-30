from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.complex_document.answering import DeterministicAnswerer
from src.complex_document.chunking import Chunk
from src.complex_document.retrieval import CharNgramRetriever, RetrievalHit


def _normalize(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"[\s,，。．%％元]", "", text).lower()


def _evidence_match(chunk: Chunk, evidence: dict) -> bool:
    if evidence.get("document_id") and chunk.document_id != evidence["document_id"]:
        return False
    if evidence.get("page") and evidence["page"] not in chunk.pages:
        return False
    if evidence.get("pages") and not set(evidence["pages"]).issubset(
        set(chunk.pages)
    ):
        return False
    terms = evidence.get("text_contains", [])
    compact = _normalize(chunk.text)
    return all(_normalize(term) in compact for term in terms)


def _has_gold_evidence(chunk: Chunk, evidence_sets: list[dict]) -> bool:
    return any(_evidence_match(chunk, evidence) for evidence in evidence_sets)


def _evidence_coverage(
    chunks: list[Chunk],
    evidence_sets: list[dict],
    *,
    mode: str,
) -> bool:
    if not evidence_sets:
        return False
    if mode == "all":
        return all(
            any(_evidence_match(chunk, evidence) for chunk in chunks)
            for evidence in evidence_sets
        )
    if mode != "any":
        raise ValueError(f"unsupported evidence_mode: {mode}")
    return any(
        _evidence_match(chunk, evidence)
        for chunk in chunks
        for evidence in evidence_sets
    )


def _evidence_reciprocal_rank(
    hits: list[RetrievalHit],
    evidence_sets: list[dict],
    *,
    mode: str,
) -> float:
    if mode == "all":
        first_ranks = [
            min(
                (
                    hit.rank
                    for hit in hits
                    if _evidence_match(hit.chunk, evidence)
                ),
                default=None,
            )
            for evidence in evidence_sets
        ]
        if not first_ranks or any(rank is None for rank in first_ranks):
            return 0.0
        # Rank at which the retriever has accumulated every required source.
        return 1.0 / max(int(rank) for rank in first_ranks if rank is not None)
    if mode != "any":
        raise ValueError(f"unsupported evidence_mode: {mode}")
    ranks = [
        hit.rank for hit in hits if _has_gold_evidence(hit.chunk, evidence_sets)
    ]
    return 1.0 / min(ranks) if ranks else 0.0


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    question_type: str
    answer: str | None
    correct: bool
    retrieval_recall_at_k: float
    reciprocal_rank: float
    citation_valid: bool
    error_source: str | None
    retrieved_chunk_ids: list[str]
    retrieved_scores: list[float]


def evaluate_question(
    question: dict,
    all_chunks: list[Chunk],
    *,
    k: int,
    retriever: CharNgramRetriever,
    answerer: DeterministicAnswerer,
) -> QuestionResult:
    hits: list[RetrievalHit] = retriever.retrieve(question["question"], all_chunks, k=k)
    retrieved = [hit.chunk for hit in hits]
    evidence_sets = question.get("evidence", [])
    evidence_mode = question.get("evidence_mode", "any")
    unanswerable = bool(question.get("unanswerable"))

    evidence_in_corpus = _evidence_coverage(
        all_chunks, evidence_sets, mode=evidence_mode
    )
    evidence_retrieved = _evidence_coverage(
        retrieved, evidence_sets, mode=evidence_mode
    )
    recall = 1.0 if (unanswerable or evidence_retrieved) else 0.0
    reciprocal_rank = (
        1.0
        if unanswerable
        else _evidence_reciprocal_rank(
            hits, evidence_sets, mode=evidence_mode
        )
    )

    answer = answerer.answer(question, retrieved)
    accepted = [_normalize(value) for value in question.get("answers", [])]
    correct = answer is None if unanswerable else _normalize(answer) in accepted

    if unanswerable:
        citation_valid = answer is None
    else:
        citation_valid = bool(
            correct
            and evidence_retrieved
            and (
                question.get("operation") == "sum"
                or any(
                    _has_gold_evidence(chunk, evidence_sets)
                    and _normalize(answer) in _normalize(chunk.text)
                    for chunk in retrieved
                )
            )
        )

    error_source = None
    if not correct:
        if not evidence_in_corpus:
            error_source = "parsing"
        elif not evidence_retrieved:
            error_source = "retrieval"
        else:
            error_source = "generation"

    return QuestionResult(
        question_id=question["question_id"],
        question_type=question["type"],
        answer=answer,
        correct=correct,
        retrieval_recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        citation_valid=citation_valid,
        error_source=error_source,
        retrieved_chunk_ids=[hit.chunk.chunk_id for hit in hits],
        retrieved_scores=[round(hit.score, 6) for hit in hits],
    )


def evaluate_downstream(
    questions: list[dict],
    chunks: list[Chunk],
    *,
    k: int = 5,
    retriever: CharNgramRetriever | None = None,
    answerer: DeterministicAnswerer | None = None,
) -> dict:
    retriever = retriever or CharNgramRetriever()
    answerer = answerer or DeterministicAnswerer()
    results = [
        evaluate_question(
            question, chunks, k=k, retriever=retriever, answerer=answerer
        )
        for question in questions
    ]
    error_counts = {
        source: sum(result.error_source == source for result in results)
        for source in ("parsing", "retrieval", "generation")
    }
    return {
        "question_count": len(results),
        "retriever": retriever.name,
        "answerer": answerer.name,
        "k": k,
        "retrieval_recall_at_k": round(
            sum(result.retrieval_recall_at_k for result in results) / len(results),
            6,
        )
        if results
        else None,
        "mrr": round(
            sum(result.reciprocal_rank for result in results) / len(results), 6
        )
        if results
        else None,
        "answer_correctness": round(
            sum(result.correct for result in results) / len(results), 6
        )
        if results
        else None,
        "citation_validity": round(
            sum(result.citation_valid for result in results) / len(results), 6
        )
        if results
        else None,
        "error_attribution": error_counts,
        "questions": [asdict(result) for result in results],
    }

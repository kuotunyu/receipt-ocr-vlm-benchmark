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
    unanswerable = bool(question.get("unanswerable"))

    evidence_in_corpus = any(
        _has_gold_evidence(chunk, evidence_sets) for chunk in all_chunks
    )
    evidence_ranks = [
        hit.rank for hit in hits if _has_gold_evidence(hit.chunk, evidence_sets)
    ]
    recall = 1.0 if (unanswerable or evidence_ranks) else 0.0
    reciprocal_rank = (
        1.0 if unanswerable else (1.0 / min(evidence_ranks) if evidence_ranks else 0.0)
    )

    answer = answerer.answer(question, retrieved)
    accepted = [_normalize(value) for value in question.get("answers", [])]
    correct = answer is None if unanswerable else _normalize(answer) in accepted

    if unanswerable:
        citation_valid = answer is None
    else:
        citation_valid = bool(
            correct
            and any(
                _has_gold_evidence(chunk, evidence_sets)
                and (
                    _normalize(answer) in _normalize(chunk.text)
                    or question.get("operation") == "sum"
                )
                for chunk in retrieved
            )
        )

    error_source = None
    if not correct:
        if not evidence_in_corpus:
            error_source = "parsing"
        elif not evidence_ranks:
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

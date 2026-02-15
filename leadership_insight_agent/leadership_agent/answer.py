from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .retriever import retrieve, confidence_from_scores, RetrievedChunk
from .prompts import build_context, SYSTEM_PROMPT, user_prompt
from .llm import LLMClient


@dataclass
class AnswerResult:
    question: str
    answer: str
    confidence: float
    retrieved: List[RetrievedChunk]


def answer_question(
    index_dir: str,
    question: str,
    llm: LLMClient,
    top_k: int,
    embed_model: Optional[str] = None,
) -> AnswerResult:
    retrieved = retrieve(index_dir=index_dir, question=question, top_k=top_k, embed_model_name=embed_model)
    context = build_context(retrieved)
    resp = llm.generate(SYSTEM_PROMPT, user_prompt(question, context))
    conf = confidence_from_scores([c.score for c in retrieved])
    return AnswerResult(question=question, answer=resp, confidence=conf, retrieved=retrieved)

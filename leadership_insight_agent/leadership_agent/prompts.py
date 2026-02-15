from __future__ import annotations

from typing import List

from .retriever import RetrievedChunk


def build_context(chunks: List[RetrievedChunk], max_chars: int = 1200) -> str:
    """
    Build compact context so the model stays grounded and avoids drift.
    """
    parts = []
    for c in chunks:
        text = (c.text or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        parts.append(
            f"[CITATION:{c.chunk_id}]\nSOURCE: {c.doc_id}\nTEXT:\n{text}\n"
        )
    return "\n\n".join(parts)

SYSTEM_PROMPT = """
You are an AI Leadership Insight Agent reading company documents.

Strict rules:
1) Use ONLY the provided Context. Do NOT use outside knowledge.
2) Every key claim MUST include at least one citation tag exactly as provided: [CITATION:<chunk_id>].
3) If the Context does not contain the answer, respond ONLY with:
   - "Not found in provided documents."
   - 2–4 follow-up questions to locate the missing info.
4) Do not use marketing fluff or vague phrases. Prefer exact wording from the documents.
5) When asked for definitions (e.g., mission, segments, strategy), include 1–2 short direct quotes from the Context.
6) Be precise. Do not say “not explicitly stated” unless you also show what IS stated with citations.
""".strip()


def user_prompt(question: str, context: str) -> str:
    return f"""
TASK
Answer the Question using ONLY the Context.

QUESTION
{question}

CONTEXT
{context}

OUTPUT FORMAT (must follow exactly)
Executive Summary:
- 2 to 4 sentences.
- Each sentence must end with at least one citation tag.

Key Findings:
- 4 to 8 bullet points.
- Each bullet must end with at least one citation tag.

Direct Quotes:
- 1 to 3 short quotes (max 25 words each) copied verbatim from Context.
- Each quote must end with a citation tag.

If not answerable from Context:
- Output exactly:
  Not found in provided documents.
  Follow-up questions:
  - ...
  - ...
""".strip()

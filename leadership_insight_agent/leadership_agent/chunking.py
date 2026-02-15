from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def _normalize_whitespace(s: str) -> str:
    return " ".join(s.split())


def chunk_text(doc_id: str, text: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """Simple character-based chunking with overlap.

    Keeps chunk boundaries stable and avoids external dependencies.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    clean = clean.strip()
    if not clean:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        piece = clean[start:end].strip()
        piece = _normalize_whitespace(piece)
        if piece:
            chunks.append(Chunk(chunk_id=f"{doc_id}::chunk{idx}", doc_id=doc_id, text=piece))
            idx += 1
        if end == len(clean):
            break
        start = max(0, end - chunk_overlap)

    return chunks

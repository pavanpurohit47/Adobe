from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    score: float


def _load_chunks(path: Path) -> List[dict]:
    chunks: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def retrieve(index_dir: str, question: str, top_k: int, embed_model_name: str | None = None) -> List[RetrievedChunk]:
    idx_dir = Path(index_dir)
    meta = json.loads((idx_dir / "meta.json").read_text(encoding="utf-8"))
    embed_model = embed_model_name or meta["embed_model"]

    model = SentenceTransformer(embed_model)
    q_emb = model.encode([question], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    index = faiss.read_index(str(idx_dir / meta["faiss_index"]))
    chunks = _load_chunks(idx_dir / meta["chunks_file"])

    scores, ids = index.search(q_emb, top_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    out: List[RetrievedChunk] = []
    for rank, (i, s) in enumerate(zip(ids, scores)):
        if i < 0 or i >= len(chunks):
            continue
        ch = chunks[i]
        out.append(
            RetrievedChunk(
                chunk_id=ch["chunk_id"],
                doc_id=ch["doc_id"],
                source_path=ch.get("source_path", ""),
                text=ch["text"],
                score=float(s),
            )
        )
    return out


def confidence_from_scores(scores: List[float]) -> float:
    """Heuristic confidence in [0,1] from FAISS inner product scores.

    With normalized embeddings, inner product ~ cosine similarity in [-1,1].
    We map the top score and the separation between top-1 and top-2.
    """
    if not scores:
        return 0.0
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else (top1 - 0.05)
    sep = max(0.0, top1 - top2)
    # map cosine-ish to 0..1
    base = (top1 + 1.0) / 2.0
    boost = min(0.2, sep * 0.8)
    return float(max(0.0, min(1.0, base + boost)))

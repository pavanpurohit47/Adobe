from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .loaders import load_documents
from .chunking import chunk_text, Chunk


@dataclass
class IndexedChunk:
    chunk_id: str
    doc_id: str
    text: str
    source_path: str


def build_index(docs_dir: str, out_dir: str, embed_model_name: str, chunk_size: int, chunk_overlap: int) -> str:
    """Ingest docs -> chunk -> embed -> build FAISS index.

    Persists:
    - faiss.index
    - chunks.jsonl (metadata)
    - meta.json (settings)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs = load_documents(docs_dir)

    model = SentenceTransformer(embed_model_name)

    indexed: List[IndexedChunk] = []
    all_texts: List[str] = []

    for d in docs:
        chs = chunk_text(d.doc_id, d.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c in chs:
            indexed.append(
                IndexedChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    text=c.text,
                    source_path=d.source_path,
                )
            )
            all_texts.append(c.text)

    if not indexed:
        raise ValueError("No chunks produced from documents. Check input docs.")

    embs = model.encode(all_texts, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim because embeddings are normalized
    index.add(embs)

    faiss.write_index(index, str(out / "faiss.index"))

    with (out / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for ch in indexed:
            f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")

    meta: Dict[str, Any] = {
        "docs_dir": docs_dir,
        "embed_model": embed_model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(indexed),
        "faiss_index": "faiss.index",
        "chunks_file": "chunks.jsonl",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return str(out)

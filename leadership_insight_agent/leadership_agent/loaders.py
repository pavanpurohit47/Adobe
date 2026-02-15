from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader
from docx import Document


@dataclass
class LoadedDoc:
    doc_id: str
    source_path: str
    text: str


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(f"\n\n--- Page {i+1} ---\n\n{txt}")
    return "".join(pages)


def _load_docx(path: Path) -> str:
    doc = Document(str(path))
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras)


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(docs_dir: str) -> List[LoadedDoc]:
    root = Path(docs_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    docs: List[LoadedDoc] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt"}:
            continue

        if suffix == ".pdf":
            text = _load_pdf(path)
        elif suffix == ".docx":
            text = _load_docx(path)
        else:
            text = _load_txt(path)

        text = (text or "").strip()
        if not text:
            continue

        docs.append(
            LoadedDoc(
                doc_id=str(path.relative_to(root)).replace("\\\\", "/"),
                source_path=str(path),
                text=text,
            )
        )

    if not docs:
        raise ValueError(
            f"No supported documents found in {docs_dir}. Supported: .pdf, .docx, .txt"
        )

    return docs

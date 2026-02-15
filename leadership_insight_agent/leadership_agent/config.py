from __future__ import annotations

from dataclasses import dataclass
import os


def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"Env var {name} must be an int, got: {v!r}") from e


@dataclass(frozen=True)
class Settings:
    # LLM (OpenAI-compatible)
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Retrieval
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = _get_int("TOP_K", 6)
    chunk_size: int = _get_int("CHUNK_SIZE", 900)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 120)


def get_settings() -> Settings:
    return Settings()

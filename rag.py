"""
rag.py — lightweight RAG (Retrieval-Augmented Generation) for the Task Agent
=============================================================================
Uses a simple in-process vector store backed by sentence-transformers embeddings
and cosine similarity. No external vector DB required — just drop .txt or .md
files into the `docs/` folder and they'll be indexed automatically on startup.

To add your own knowledge:
  • Create a `docs/` directory next to this file
  • Drop any .txt or .md files in there
  • They'll be chunked, embedded, and searchable at agent runtime

Swap out SentenceTransformer for OpenAI / Cohere embeddings if you prefer;
just replace the `_embed()` function.
"""

from __future__ import annotations

import math
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Optional heavy deps — graceful fallback if not installed ───────────────────
try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    _MODEL = None

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str           # filename
    chunk_id: int
    embedding: list[float] = field(default_factory=list)


# ── Embedding ──────────────────────────────────────────────────────────────────

def _embed(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors, one per text."""
    if _HAS_ST and _MODEL is not None:
        vecs = _MODEL.encode(texts, normalize_embeddings=True)
        return vecs.tolist()
    # Fallback: trivial bag-of-char-bigrams embedding (no dependencies)
    result = []
    for text in texts:
        text_lower = text.lower()
        bigrams: dict[str, int] = {}
        for i in range(len(text_lower) - 1):
            bg = text_lower[i : i + 2]
            bigrams[bg] = bigrams.get(bg, 0) + 1
        total = math.sqrt(sum(v * v for v in bigrams.values())) or 1.0
        # Use a fixed vocab of 256 slots (hash mod)
        vec = [0.0] * 256
        for bg, cnt in bigrams.items():
            vec[hash(bg) % 256] += cnt / total
        result.append(vec)
    return result


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Document loading & chunking ────────────────────────────────────────────────

_CHUNK_SIZE = 400     # characters per chunk
_CHUNK_OVERLAP = 80  # overlap between consecutive chunks


def _chunk_text(text: str, source: str) -> list[Chunk]:
    """Split text into overlapping chunks."""
    chunks: list[Chunk] = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + _CHUNK_SIZE
        snippet = text[start:end].strip()
        if snippet:
            chunks.append(Chunk(text=snippet, source=source, chunk_id=chunk_id))
            chunk_id += 1
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


def _load_docs(docs_dir: str | Path = "docs") -> list[Chunk]:
    """Load all .txt and .md files from docs_dir into chunks."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return []

    chunks: list[Chunk] = []
    for fpath in sorted(docs_path.glob("**/*")):
        if fpath.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(_chunk_text(text, fpath.name))
        except OSError:
            pass
    return chunks


# ── Vector store ───────────────────────────────────────────────────────────────

class VectorStore:
    """Simple in-memory vector store with cosine-similarity retrieval."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._ready = False

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self._ready = True
            return
        texts = [c.text for c in chunks]
        embeddings = _embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        self._chunks = chunks
        self._ready = True
        print(f"   [RAG] indexed {len(chunks)} chunks from {len({c.source for c in chunks})} file(s)")

    def query(self, text: str, top_k: int = 3) -> list[tuple[float, Chunk]]:
        if not self._chunks:
            return []
        q_emb = _embed([text])[0]
        scored = [((_cosine(q_emb, c.embedding)), c) for c in self._chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    @property
    def has_docs(self) -> bool:
        return bool(self._chunks)


# ── Module-level store (singleton) ─────────────────────────────────────────────

_store = VectorStore()


def init(docs_dir: str | Path = "docs") -> None:
    """Call once at startup to load and index documents."""
    chunks = _load_docs(docs_dir)
    _store.index(chunks)


def retrieve(query: str, top_k: int = 3, min_score: float = 0.1) -> str:
    """
    Retrieve the most relevant document snippets for a query.
    Returns a formatted string ready to inject into a prompt.
    Returns empty string if no docs are loaded or nothing is relevant.
    """
    if not _store.has_docs:
        return ""

    results = _store.query(query, top_k=top_k)
    relevant = [(score, chunk) for score, chunk in results if score >= min_score]

    if not relevant:
        return ""

    parts = ["[Retrieved context from knowledge base]"]
    for score, chunk in relevant:
        header = f"— {chunk.source} (relevance: {score:.2f})"
        body = textwrap.fill(chunk.text, width=90)
        parts.append(f"{header}\n{body}")

    return "\n\n".join(parts)

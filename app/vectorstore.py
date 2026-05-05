"""
ChromaDB wrapper. Persists to disk; uses sentence-transformers
(all-MiniLM-L6-v2) for embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.chunker import Chunk
from app.config import COLLECTION_NAME, settings


@dataclass
class RetrievalHit:
    text: str
    source: str
    url: str
    title: str
    score: float       # cosine similarity in [0, 1] (1 - cosine distance)

    @property
    def similarity(self) -> float:
        """Alias for score, exposed for downstream metrics consumers."""
        return self.score


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embed_model
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        """Public accessor so api.py can call .get() for corpus-wide metadata."""
        return self._collection

    # ----- write -----
    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "url": c.url,
                    "title": c.title,
                    "position": c.position,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    # ----- read -----
    def query(self, text: str, k: int | None = None) -> list[RetrievalHit]:
        k = k or settings.top_k
        res = self._collection.query(query_texts=[text], n_results=k)
        hits: list[RetrievalHit] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB returns cosine distance. Similarity = 1 - distance,
            # clamped to [0, 1] in case of small floating-point drift.
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            hits.append(
                RetrievalHit(
                    text=doc,
                    source=meta.get("source", ""),
                    url=meta.get("url", ""),
                    title=meta.get("title", ""),
                    score=similarity,
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()
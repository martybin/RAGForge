"""Dense retrieval: LangChain `Chroma` vector store with BGE embeddings."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.models.schemas import ChunkMetadata, RetrievalMethod, RetrievedChunk
from app.storage import IndexNotFoundError


class DenseRetriever:
    """Semantic search over a Chroma vector store.

    The collection uses cosine distance, so ``similarity = 1 - distance``; with
    L2-normalized embeddings that is exactly the dot product of query and chunk.
    """

    def __init__(self, embeddings: Embeddings, store: Chroma) -> None:
        if store._collection.count() == 0:
            raise IndexNotFoundError("Vector index is empty. Run `python scripts/ingest.py`.")
        self._embeddings = embeddings
        self._store = store

    def __len__(self) -> int:
        return self._store._collection.count()

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(self._embeddings.embed_query(query))

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self.search_by_embedding(self.embed_query(query), top_k)

    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        # Chroma returns cosine *distances* here despite the method name.
        hits = self._store.similarity_search_by_vector_with_relevance_scores(
            query_embedding.tolist(), k=min(top_k, len(self))
        )
        return [
            RetrievedChunk(
                content=document.page_content,
                metadata=ChunkMetadata.model_validate(document.metadata),
                score=1.0 - float(distance),
                method=RetrievalMethod.DENSE,
                rank=rank,
            )
            for rank, (document, distance) in enumerate(hits, start=1)
        ]

    def score(self, query_embedding: np.ndarray, chunk_ids: Iterable[str]) -> dict[str, float]:
        """Exact cosine similarity of specific chunks (scores BM25-only hybrid candidates)."""
        ids = list(chunk_ids)
        if not ids:
            return {}
        stored = self._store.get(ids=ids, include=["embeddings"])
        similarities = np.asarray(stored["embeddings"], dtype=float) @ query_embedding
        return dict(zip(stored["ids"], similarities.tolist(), strict=True))

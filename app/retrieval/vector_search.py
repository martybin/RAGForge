"""Dense retrieval: BGE query embedding + nearest-neighbour search in Chroma."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from chromadb.api.models.Collection import Collection

from app.embeddings import Embedder
from app.models.schemas import ChunkMetadata, RetrievalMethod, RetrievedChunk
from app.storage import IndexNotFoundError


class DenseRetriever:
    """Semantic search over the Chroma collection.

    The collection uses cosine distance, so ``similarity = 1 - distance``; with
    L2-normalized embeddings that is exactly the dot product of query and chunk.
    """

    def __init__(self, embedder: Embedder, collection: Collection) -> None:
        if collection.count() == 0:
            raise IndexNotFoundError("Vector index is empty. Run `python scripts/ingest.py`.")
        self._embedder = embedder
        self._collection = collection

    def __len__(self) -> int:
        return self._collection.count()

    def embed_query(self, query: str) -> np.ndarray:
        return self._embedder.embed_query(query)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self.search_by_embedding(self.embed_query(query), top_k)

    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, len(self)),
            include=["documents", "metadatas", "distances"],
        )
        rows = zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
        return [
            RetrievedChunk(
                content=document,
                metadata=ChunkMetadata.model_validate(metadata),
                score=1.0 - float(distance),
                method=RetrievalMethod.DENSE,
                rank=rank,
            )
            for rank, (document, metadata, distance) in enumerate(rows, start=1)
        ]

    def score(self, query_embedding: np.ndarray, chunk_ids: Iterable[str]) -> dict[str, float]:
        """Exact cosine similarity of specific chunks (scores BM25-only hybrid candidates)."""
        ids = list(chunk_ids)
        if not ids:
            return {}
        stored = self._collection.get(ids=ids, include=["embeddings"])
        similarities = np.asarray(stored["embeddings"], dtype=float) @ query_embedding
        return dict(zip(stored["ids"], similarities.tolist(), strict=True))

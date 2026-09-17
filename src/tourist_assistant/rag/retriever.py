from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from tourist_assistant.config import settings
from tourist_assistant.models import Attraction
from tourist_assistant.rag.ingest import build_index, load_attractions

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    attraction_id: str
    attraction_name: str
    text: str
    source_url: str | None
    score: float  # cosine similarity, roughly in [-1, 1]


class Retriever:
    """FAISS-backed semantic retriever over the tourism knowledge base."""

    def __init__(self) -> None:
        index_path = Path(settings.faiss_index_path)
        meta_path = Path(settings.faiss_meta_path)

        if not index_path.exists() or not meta_path.exists():
            logger.info("FAISS index missing, building it now.")
            build_index()

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta: list[dict] = json.load(f)

        self.model = SentenceTransformer(settings.embedding_model)
        self._attractions_by_id: dict[str, Attraction] = {
            a.id: a for a in load_attractions()
        }

    def _embed_query(self, query: str) -> "faiss.swig_ptr":
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top-k chunks for the query."""
        if not query.strip():
            return []

        scores, indices = self.index.search(self._embed_query(query), top_k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.meta[idx]
            results.append(
                RetrievedChunk(
                    attraction_id=meta["attraction_id"],
                    attraction_name=meta["attraction_name"],
                    text=meta["text"],
                    source_url=meta.get("source_url"),
                    score=float(score),
                )
            )
        return results

    def search_attractions(
        self, query: str, top_k: int = 5, min_score: float = 0.2
    ) -> list[tuple[Attraction, float]]:
        """
        Aggregate chunk-level hits to attraction level.

        Keeps the best score per attraction, filters weak matches, and
        returns Attraction objects sorted by relevance.
        """
        chunks = self.search(query, top_k=top_k * 3)
        best: dict[str, float] = {}
        for c in chunks:
            best[c.attraction_id] = max(best.get(c.attraction_id, -1.0), c.score)

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (self._attractions_by_id[aid], score)
            for aid, score in ranked
            if aid in self._attractions_by_id and score >= min_score
        ][:top_k]

    def get_attraction(self, attraction_id: str) -> Attraction | None:
        return self._attractions_by_id.get(attraction_id)

    def all_attractions(self) -> list[Attraction]:
        return list(self._attractions_by_id.values())


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Singleton so the embedding model loads once per process."""
    return Retriever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = get_retriever()

    for q in [
        "I like archaeology and museums",
        "Where can I go with children?",
        "indoor activities for a rainy day",
        "beach near the city",
        "traditional food and old town",
    ]:
        print(f"\nQuery: {q}")
        for att, score in r.search_attractions(q, top_k=3):
            print(f"  {score:.3f}  {att.name}  ({', '.join(c.value for c in att.categories)})")
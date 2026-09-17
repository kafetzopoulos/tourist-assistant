from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from tourist_assistant.config import settings
from tourist_assistant.models import Attraction

logger = logging.getLogger(__name__)


def load_attractions(path: str | None = None) -> list[Attraction]:
    """Load and validate the tourism knowledge base."""
    path = path or settings.attractions_path
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Attraction.model_validate(item) for item in raw]


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    Simple word-based chunker with overlap.

    For our short attraction descriptions this usually returns one chunk,
    but the same logic scales to longer documents (guides, articles).
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_documents(attractions: list[Attraction]) -> list[dict]:
    """
    Convert attractions into retrievable chunks.

    Each chunk carries metadata so the retriever can reconstruct the source
    attraction and cite it in the answer.
    """
    docs: list[dict] = []
    for attraction in attractions:
        # The header improves retrieval for queries about categories,
        # indoor/outdoor, and named entities.
        header = (
            f"{attraction.name}. "
            f"Categories: {', '.join(c.value for c in attraction.categories)}. "
            f"Indoor: {'yes' if attraction.indoor else 'no'}."
        )
        full_text = f"{header} {attraction.description}"

        for i, chunk in enumerate(chunk_text(full_text)):
            docs.append(
                {
                    "attraction_id": attraction.id,
                    "attraction_name": attraction.name,
                    "chunk_index": i,
                    "text": chunk,
                    "source_url": attraction.source_url,
                }
            )
    return docs


def build_index(force: bool = False) -> None:
    """
    Build the FAISS index from attractions.json.

    Uses normalized embeddings + IndexFlatIP so inner product equals
    cosine similarity. Flat index is fine for a prototype; for thousands
    of attractions we would swap to IVF or HNSW.
    """
    index_path = Path(settings.faiss_index_path)
    meta_path = Path(settings.faiss_meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists() and meta_path.exists() and not force:
        logger.info("FAISS index already exists at %s. Use force=True to rebuild.", index_path)
        return

    attractions = load_attractions()
    docs = build_documents(attractions)
    if not docs:
        raise ValueError("No documents to index. Check attractions.json.")

    logger.info("Embedding %d chunks with %s", len(docs), settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode(
        [d["text"] for d in docs],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    logger.info(
        "Wrote FAISS index (%d vectors, dim=%d) to %s",
        index.ntotal,
        index.d,
        index_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_index(force=True)
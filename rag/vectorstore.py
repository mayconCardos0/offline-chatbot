"""
FAISS-backed vector store with disk persistence.
Persists index.faiss and metadata.json to the configured index directory.
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.json"


class VectorStore:
    """Stores and retrieves document chunk embeddings using a FAISS flat L2 index."""

    def __init__(self, index_dir: str, embedding_dim: int) -> None:
        """Initialise the store, loading an existing index from disk if present.

        Args:
            index_dir: Directory where index.faiss and metadata.json are stored.
            embedding_dim: Dimensionality of the embedding vectors.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from exc

        self._faiss = faiss
        self._index_dir = Path(index_dir)
        self._embedding_dim = embedding_dim
        self._metadata: list[dict] = []
        self._index = faiss.IndexFlatL2(embedding_dim)

        self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Append chunks and their embeddings to the index.

        Args:
            chunks: List of {text, source} dicts.
            embeddings: Parallel list of float vectors.
        """
        if not chunks:
            return

        vectors = np.array(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._metadata.extend(chunks)
        logger.debug("Added %d chunks to vector store (total: %d).", len(chunks), len(self._metadata))

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self._index_dir / _INDEX_FILE))
        (self._index_dir / _META_FILE).write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Vector store saved to '%s' (%d chunks).", self._index_dir, len(self._metadata))

    def load(self) -> bool:
        """Load an existing index from disk if both files are present.

        Returns:
            True if an existing index was loaded, False otherwise.
        """
        index_path = self._index_dir / _INDEX_FILE
        meta_path = self._index_dir / _META_FILE

        if not (index_path.exists() and meta_path.exists()):
            return False

        try:
            self._index = self._faiss.read_index(str(index_path))
            self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            logger.info(
                "Loaded existing vector store from '%s' (%d chunks).",
                self._index_dir,
                len(self._metadata),
            )
            return True
        except Exception as exc:
            logger.error("Failed to load vector store from '%s': %s", self._index_dir, exc)
            return False

    def search(self, query_vec: list[float], k: int) -> list[dict]:
        """Return the top-k most similar chunks for a query vector.

        Args:
            query_vec: Float vector of the same dimensionality as stored embeddings.
            k: Number of results to return.

        Returns:
            List of {text, source} dicts ordered by similarity (closest first).
        """
        total = len(self._metadata)
        if total == 0:
            return []

        k = min(k, total)
        query = np.array([query_vec], dtype=np.float32)
        _, indices = self._index.search(query, k)
        return [self._metadata[i] for i in indices[0] if i < total]

"""
Retriever: embeds a query and returns the top-K most relevant chunks from the VectorStore.
"""
import logging

from .embeddings import EmbeddingModel
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Combines EmbeddingModel and VectorStore to answer semantic queries."""

    def __init__(self, vectorstore: VectorStore, embed_model: EmbeddingModel, top_k: int = 4) -> None:
        """
        Args:
            vectorstore: Populated VectorStore instance.
            embed_model: EmbeddingModel used to embed queries.
            top_k: Number of chunks to return per query.
        """
        self._vectorstore = vectorstore
        self._embed_model = embed_model
        self._top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        """Embed the query and return the top-K most similar chunks.

        Args:
            query: User query string.

        Returns:
            List of {text: str, source: str} dicts, ordered by similarity.
            Returns an empty list if the index is empty.
        """
        if self._vectorstore._index.ntotal == 0:
            logger.warning("Vector store is empty — returning no context for query: '%s'", query)
            return []

        query_vec = self._embed_model.embed([query])[0]
        results = self._vectorstore.search(query_vec, self._top_k)
        logger.debug("Retrieved %d chunks for query: '%s'", len(results), query)
        return results

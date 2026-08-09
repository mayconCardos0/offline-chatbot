"""
Cross-encoder reranker for the RAG pipeline.

Loads a cross-encoder model once and provides a rerank() function that
scores (query, passage) pairs and fuses the CE score with the existing
hybrid score using configurable weights.

Phase 3E/3F: Designed for CPU-only inference on Raspberry Pi 5 (8GB).
The model is loaded once at startup and reused across all queries.

Usage:
    from rag.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        hybrid_weight=0.3,
        ce_top_k=20,
    )
    reranked = reranker.rerank(query, candidates, final_k=5)
"""

import logging

logger = logging.getLogger(__name__)


def _normalize_scores(values: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1].

    Returns all zeros if all values are equal (avoids division by zero).
    """
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


class CrossEncoderReranker:
    """Reranks candidates using a cross-encoder with hybrid score fusion.

    The reranker:
    1. Takes the top ce_top_k candidates from the hybrid-reranked pool.
    2. Scores each (query, passage) pair with the cross-encoder.
    3. Normalizes both hybrid and CE scores to [0, 1].
    4. Fuses: final = hybrid_weight * hybrid_norm + ce_weight * ce_norm
    5. Returns the top final_k by fused score.

    Attributes:
        hybrid_weight: Weight for the hybrid score in fusion (0.0–1.0).
                       ce_weight = 1.0 - hybrid_weight.
        ce_top_k:      How many hybrid candidates to pass to the CE.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        hybrid_weight: float = 0.3,
        ce_top_k: int = 20,
    ) -> None:
        self._model_name = model_name
        self._hybrid_weight = hybrid_weight
        self._ce_weight = 1.0 - hybrid_weight
        self._ce_top_k = ce_top_k
        self._model = None

        logger.info(
            "CrossEncoderReranker init: model=%s, hybrid_weight=%.2f, ce_top_k=%d",
            model_name,
            hybrid_weight,
            ce_top_k,
        )

    def _load_model(self) -> None:
        """Lazy-load the cross-encoder model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install: pip install sentence-transformers"
            ) from exc

        logger.info("Loading cross-encoder model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        logger.info("Cross-encoder model loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        final_k: int = 5,
    ) -> list[dict]:
        """Rerank candidates using hybrid + CE score fusion.

        Args:
            query:      The user query string.
            candidates: Hybrid-reranked candidate list (each has 'score' = hybrid score).
            final_k:    Number of results to return after reranking.

        Returns:
            Top final_k candidates ordered by fused score. Each candidate dict
            is augmented with 'hybrid_score', 'ce_score', and the final 'score'.
            Falls back to input ordering if CE inference fails.
        """
        if not candidates:
            return []

        # Take top ce_top_k for CE scoring
        ce_candidates = candidates[: self._ce_top_k]

        if not ce_candidates:
            return candidates[:final_k]

        # Load model on first use (not at startup if never called)
        try:
            self._load_model()
        except Exception as exc:
            logger.error(
                "Failed to load cross-encoder model: %s. Falling back to hybrid ranking.",
                exc,
            )
            return candidates[:final_k]

        # Preserve hybrid scores before overwriting
        hybrid_scores = [c.get("score", 0.0) for c in ce_candidates]

        # Score all (query, passage) pairs
        try:
            pairs = [(query, c.get("text", "")) for c in ce_candidates]
            ce_raw_scores = self._model.predict(pairs).tolist()
        except Exception as exc:
            logger.error(
                "Cross-encoder inference failed: %s. Falling back to hybrid ranking.",
                exc,
            )
            return candidates[:final_k]

        # Normalize both score arrays to [0, 1]
        hybrid_norm = _normalize_scores(hybrid_scores)
        ce_norm = _normalize_scores(ce_raw_scores)

        # Fuse scores
        reranked = []
        for chunk, h_norm, ce_n, ce_raw in zip(
            ce_candidates, hybrid_norm, ce_norm, ce_raw_scores
        ):
            fused = self._hybrid_weight * h_norm + self._ce_weight * ce_n
            new_chunk = dict(chunk)
            new_chunk["hybrid_score"] = chunk.get("score", 0.0)
            new_chunk["ce_score"] = float(ce_raw)
            new_chunk["score"] = fused
            reranked.append(new_chunk)

        reranked.sort(key=lambda c: c["score"], reverse=True)

        logger.debug(
            "CE rerank: %d candidates → top %d (hw=%.2f, cew=%.2f)",
            len(ce_candidates),
            final_k,
            self._hybrid_weight,
            self._ce_weight,
        )
        return reranked[:final_k]

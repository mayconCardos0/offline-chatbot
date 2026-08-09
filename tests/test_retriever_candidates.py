"""
Phase 3B tests — Candidate pool size and candidate_multiplier behavior.

Tests verify:
  - candidate_multiplier controls FAISS pool size correctly
  - Higher cm produces more FAISS candidates
  - Final top_k is independent of candidate_multiplier
  - Interaction between candidate pool and final top_k
  - Deterministic retrieval at different pool sizes
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from rag.retriever import Retriever

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_retriever(n_chunks=50, cm=4, top_k=5):
    """Create a retriever with n_chunks available and configurable cm."""
    chunks = [
        {
            "text": f"chunk {i} with content keyword text",
            "chunk_id": f"c{i:03d}",
            "score": 0.8 - i * 0.01,
        }
        for i in range(n_chunks)
    ]
    vs = MagicMock()
    vs.size = n_chunks
    vs.search.return_value = chunks
    em = MagicMock()
    em.embed.return_value = [np.random.randn(64).tolist()]
    return (
        Retriever(
            vectorstore=vs,
            embed_model=em,
            top_k=top_k,
            candidate_multiplier=cm,
            min_score=0.25,
            lexical_weight=0.0,
            adaptive_sigma=999.0,  # adaptive OFF
            gap_filter_enabled=False,
            keyword_filter_enabled=False,
        ),
        vs,
    )


# ---------------------------------------------------------------------------
# Candidate multiplier controls pool size
# ---------------------------------------------------------------------------


class TestCandidateMultiplierPoolSize:
    def test_cm4_requests_20_candidates(self):
        """cm=4, top_k=5 → FAISS should be asked for 20 candidates."""
        r, vs = _make_retriever(n_chunks=50, cm=4, top_k=5)
        r.retrieve("query keyword content", k=5)
        # vectorstore.search called with k = effective_k * cm = 5 * 4 = 20
        vs.search.assert_called_once()
        _, call_k = vs.search.call_args[0]
        assert call_k == 20

    def test_cm6_requests_30_candidates(self):
        """cm=6, top_k=5 → FAISS asked for 30."""
        r, vs = _make_retriever(n_chunks=50, cm=6, top_k=5)
        r.retrieve("query keyword content", k=5)
        _, call_k = vs.search.call_args[0]
        assert call_k == 30

    def test_cm8_requests_40_candidates(self):
        """cm=8, top_k=5 → FAISS asked for 40."""
        r, vs = _make_retriever(n_chunks=50, cm=8, top_k=5)
        r.retrieve("query keyword content", k=5)
        _, call_k = vs.search.call_args[0]
        assert call_k == 40

    def test_cm10_requests_50_candidates(self):
        """cm=10, top_k=5 → FAISS asked for 50."""
        r, vs = _make_retriever(n_chunks=50, cm=10, top_k=5)
        r.retrieve("query keyword content", k=5)
        _, call_k = vs.search.call_args[0]
        assert call_k == 50

    def test_explicit_k_uses_that_k_for_pool(self):
        """retrieve(query, k=3) with cm=4 → pool = 3*4 = 12."""
        r, vs = _make_retriever(n_chunks=50, cm=4, top_k=5)
        r.retrieve("query keyword content", k=3)
        _, call_k = vs.search.call_args[0]
        assert call_k == 12


# ---------------------------------------------------------------------------
# Final top_k is independent of candidate pool
# ---------------------------------------------------------------------------


class TestFinalTopKIndependence:
    def test_final_results_capped_at_top_k(self):
        """Regardless of pool size, final results <= top_k."""
        r, _ = _make_retriever(n_chunks=50, cm=10, top_k=5)
        results = r.retrieve("chunk content keyword text", k=5)
        assert len(results) <= 5

    def test_larger_pool_does_not_increase_final_count(self):
        """cm=10 should not return more than cm=4 when top_k=5."""
        r4, _ = _make_retriever(n_chunks=50, cm=4, top_k=5)
        r10, _ = _make_retriever(n_chunks=50, cm=10, top_k=5)
        res4 = r4.retrieve("chunk content keyword text", k=5)
        res10 = r10.retrieve("chunk content keyword text", k=5)
        assert len(res4) == len(res10)

    def test_explicit_k_overrides_top_k(self):
        """retrieve(query, k=3) returns at most 3 regardless of constructor top_k."""
        r, _ = _make_retriever(n_chunks=50, cm=6, top_k=10)
        results = r.retrieve("chunk content keyword text", k=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Higher cm can surface lower-ranked chunks
# ---------------------------------------------------------------------------


class TestHigherCmSurfacesMoreChunks:
    def test_chunk_at_rank_25_found_with_cm6(self):
        """A chunk at rank 25 should be in pool with cm=6 (pool=30) but not cm=4 (pool=20)."""
        # Create chunks where the target is at position 24 (0-indexed)
        n = 50
        chunks = [
            {
                "text": f"generic chunk {i} filler content text",
                "chunk_id": f"c{i:03d}",
                "score": 0.8 - i * 0.005,
            }
            for i in range(n)
        ]
        target_idx = 24
        chunks[target_idx]["text"] = "specific target chunk with unique keyword content"
        chunks[target_idx]["chunk_id"] = "target"

        vs = MagicMock()
        vs.size = n

        em = MagicMock()
        em.embed.return_value = [np.random.randn(64).tolist()]

        # cm=4: pool=20 — target at rank 25 NOT in pool
        vs.search.return_value = chunks[:20]  # only first 20
        r4 = Retriever(
            vectorstore=vs,
            embed_model=em,
            top_k=5,
            candidate_multiplier=4,
            min_score=0.25,
            lexical_weight=0.0,
            adaptive_sigma=999.0,
            gap_filter_enabled=False,
            keyword_filter_enabled=False,
        )
        res4 = r4.retrieve("target keyword content", k=5)
        ids4 = {c["chunk_id"] for c in res4}
        assert "target" not in ids4

        # cm=6: pool=30 — target at rank 25 IS in pool
        vs.search.return_value = chunks[:30]  # first 30 includes rank 25
        r6 = Retriever(
            vectorstore=vs,
            embed_model=em,
            top_k=5,
            candidate_multiplier=6,
            min_score=0.25,
            lexical_weight=0.0,
            adaptive_sigma=999.0,
            gap_filter_enabled=False,
            keyword_filter_enabled=False,
        )
        res6 = r6.retrieve("target keyword content", k=30)
        ids6 = {c["chunk_id"] for c in res6}
        assert "target" in ids6


# ---------------------------------------------------------------------------
# Deterministic behavior
# ---------------------------------------------------------------------------


class TestDeterministicBehavior:
    def test_same_cm_same_results(self):
        """Same configuration produces identical results across calls."""
        r, _ = _make_retriever(n_chunks=30, cm=6, top_k=5)
        res1 = r.retrieve("chunk content keyword text", k=5)
        res2 = r.retrieve("chunk content keyword text", k=5)
        ids1 = [c["chunk_id"] for c in res1]
        ids2 = [c["chunk_id"] for c in res2]
        assert ids1 == ids2

    def test_results_ordered_by_score(self):
        """Final results must be score-descending."""
        r, _ = _make_retriever(n_chunks=30, cm=6, top_k=10)
        results = r.retrieve("chunk content keyword text", k=10)
        scores = [c["score"] for c in results]
        assert scores == sorted(scores, reverse=True)

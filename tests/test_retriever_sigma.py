"""
Phase 3A tests — Adaptive filter candidate-selection strategies.

Tests verify:
  - Adaptive sigma=1.0 (baseline) produces fewer results than sigma=999
  - sigma=2.0 is intermediate between sigma=1.0 and sigma=999
  - sigma=999 effectively disables adaptive pruning
  - Fixed top-N with disabled adaptive passes all candidates through
  - Score ties are handled deterministically
  - Edge cases: empty candidates, single candidate, all same score
  - Adaptive filter always preserves at least the best-scoring chunk
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from rag.retriever import Retriever

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_retriever(chunks, sigma=1.0, gap=False, keyword=False, top_k=5, cm=2):
    vs = MagicMock()
    vs.size = 100
    vs.search.return_value = chunks
    em = MagicMock()
    em.embed.return_value = [np.random.randn(64).tolist()]
    return Retriever(
        vectorstore=vs,
        embed_model=em,
        top_k=top_k,
        candidate_multiplier=cm,
        min_score=0.25,
        lexical_weight=0.0,
        gap_filter_enabled=gap,
        keyword_filter_enabled=keyword,
        adaptive_sigma=sigma,
    )


def _chunks_with_scores(scores):
    """Generate chunk dicts with given scores and matching text for keyword filter."""
    return [
        {"text": f"content chunk number {i} keyword", "chunk_id": f"c{i}", "score": s}
        for i, s in enumerate(scores)
    ]


# ---------------------------------------------------------------------------
# Sigma behavior
# ---------------------------------------------------------------------------


class TestAdaptiveSigmaBehavior:
    def test_sigma_1_aggressive_pruning(self):
        """sigma=1.0 keeps only chunks within 1σ of best — typically few survivors."""
        # Tight distribution: best=0.70, std≈0.06, threshold≈0.64 → ~4/10 pass
        scores = [0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.54, 0.52]
        chunks = _chunks_with_scores(scores)
        r = _make_retriever(chunks, sigma=1.0, top_k=10)
        results = r.retrieve("content chunk keyword", k=10)
        # Should be fewer than all 10
        assert len(results) < 10

    def test_sigma_999_passes_all_above_min_score(self):
        """sigma=999 threshold ≈ 0 — all above min_score survive."""
        scores = [0.70, 0.60, 0.50, 0.40, 0.30, 0.26]
        chunks = _chunks_with_scores(scores)
        r = _make_retriever(chunks, sigma=999.0, top_k=10)
        results = r.retrieve("content chunk keyword", k=10)
        # All 6 are above 0.25
        assert len(results) == 6

    def test_sigma_2_is_intermediate(self):
        """sigma=2.0 should pass more than sigma=1.0 but potentially less than sigma=999."""
        scores = [0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.54, 0.52]
        chunks = _chunks_with_scores(scores)
        r1 = _make_retriever(chunks, sigma=1.0, top_k=10)
        r2 = _make_retriever(chunks, sigma=2.0, top_k=10)
        r999 = _make_retriever(chunks, sigma=999.0, top_k=10)

        n1 = len(r1.retrieve("content chunk keyword", k=10))
        n2 = len(r2.retrieve("content chunk keyword", k=10))
        n999 = len(r999.retrieve("content chunk keyword", k=10))

        assert n1 <= n2 <= n999

    def test_sigma_3_passes_more_than_sigma_2(self):
        """sigma=3.0 is less aggressive than sigma=2.0."""
        scores = [0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.28, 0.26]
        chunks = _chunks_with_scores(scores)
        r2 = _make_retriever(chunks, sigma=2.0, top_k=10)
        r3 = _make_retriever(chunks, sigma=3.0, top_k=10)

        n2 = len(r2.retrieve("content chunk keyword", k=10))
        n3 = len(r3.retrieve("content chunk keyword", k=10))
        assert n3 >= n2

    def test_adaptive_always_keeps_best(self):
        """The best-scoring chunk must always survive regardless of sigma."""
        scores = [0.90, 0.30, 0.29, 0.28, 0.27]
        chunks = _chunks_with_scores(scores)
        r = _make_retriever(chunks, sigma=1.0, top_k=5)
        results = r.retrieve("content chunk keyword", k=5)
        assert len(results) >= 1
        assert results[0]["chunk_id"] == "c0"  # best scorer


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAdaptiveEdgeCases:
    def test_single_candidate(self):
        """One candidate above min_score should survive any sigma."""
        chunks = _chunks_with_scores([0.50])
        r = _make_retriever(chunks, sigma=1.0, top_k=5)
        results = r.retrieve("content chunk keyword", k=5)
        assert len(results) == 1

    def test_all_same_score(self):
        """When all scores are identical (σ=0), threshold = best - 0 → all pass."""
        chunks = _chunks_with_scores([0.50, 0.50, 0.50, 0.50, 0.50])
        r = _make_retriever(chunks, sigma=1.0, top_k=5)
        results = r.retrieve("content chunk keyword", k=5)
        assert len(results) == 5

    def test_empty_after_absolute_filter(self):
        """All scores below min_score → empty results."""
        chunks = _chunks_with_scores([0.20, 0.15, 0.10])
        r = _make_retriever(chunks, sigma=1.0, top_k=5)
        results = r.retrieve("content chunk keyword", k=5)
        assert results == []

    def test_scores_at_exact_threshold(self):
        """Scores exactly at the threshold should be kept (>= not >)."""
        # best=0.60, σ≈0.10, threshold≈0.50
        scores = [0.60, 0.50, 0.50, 0.40, 0.30]
        chunks = _chunks_with_scores(scores)
        r = _make_retriever(chunks, sigma=1.0, top_k=5)
        results = r.retrieve("content chunk keyword", k=5)
        # 0.50 should survive (>= threshold)
        result_ids = {c["chunk_id"] for c in results}
        assert "c1" in result_ids or "c2" in result_ids

    def test_large_candidate_pool(self):
        """20 candidates with realistic score spread."""
        scores = [0.70 - i * 0.02 for i in range(20)]  # 0.70 → 0.32
        chunks = _chunks_with_scores(scores)
        r1 = _make_retriever(chunks, sigma=1.0, top_k=20, cm=1)
        r999 = _make_retriever(chunks, sigma=999.0, top_k=20, cm=1)

        n1 = len(r1.retrieve("content chunk keyword", k=20))
        n999 = len(r999.retrieve("content chunk keyword", k=20))

        # sigma=1 should be substantially fewer
        assert n1 < n999
        # sigma=999 should keep all above 0.25 (first 23 but only 20 available)
        above_min = sum(1 for s in scores if s >= 0.25)
        assert n999 == above_min


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    def test_results_ordered_by_score_descending(self):
        """Final results must be ordered by score, highest first."""
        scores = [0.50, 0.70, 0.60, 0.40, 0.55]
        chunks = _chunks_with_scores(scores)
        r = _make_retriever(chunks, sigma=999.0, top_k=5)
        results = r.retrieve("content chunk keyword", k=5)
        result_scores = [c["score"] for c in results]
        assert result_scores == sorted(result_scores, reverse=True)

    def test_stable_across_repeated_calls(self):
        """Same input must produce same output order."""
        scores = [0.60, 0.58, 0.56, 0.54, 0.52]
        chunks = _chunks_with_scores(scores)
        r = _make_retriever(chunks, sigma=1.0, top_k=5)
        r1 = r.retrieve("content chunk keyword", k=5)
        r2 = r.retrieve("content chunk keyword", k=5)
        assert [c["chunk_id"] for c in r1] == [c["chunk_id"] for c in r2]

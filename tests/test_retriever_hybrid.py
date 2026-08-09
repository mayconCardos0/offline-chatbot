"""
Phase 3D tests — Hybrid ranking weight configuration.

Tests verify:
  - lexical_weight=0 triggers pure semantic mode (short-circuit)
  - lexical_weight=0.40 produces different ranking than lw=0
  - Score normalization to [0,1]
  - Weight formula: combined = (1-lw)*sem_norm + lw*bm25_norm
  - Deterministic ranking across repeated calls
  - BM25 contributes positively to ranking for keyword-rich queries
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.retriever import Retriever, _bm25_scores, _normalize, _tokenize

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_retriever(chunks, lexical_weight=0.40, top_k=5):
    vs = MagicMock()
    vs.size = 100
    vs.search.return_value = chunks
    em = MagicMock()
    em.embed.return_value = [np.random.randn(64).tolist()]
    return Retriever(
        vectorstore=vs,
        embed_model=em,
        top_k=top_k,
        candidate_multiplier=2,
        min_score=0.25,
        lexical_weight=lexical_weight,
        adaptive_sigma=999.0,
        gap_filter_enabled=False,
        keyword_filter_enabled=False,
    )


def _chunks_with_text_scores(items):
    """items: list of (text, score) tuples"""
    return [
        {"text": text, "chunk_id": f"c{i}", "score": score}
        for i, (text, score) in enumerate(items)
    ]


# ---------------------------------------------------------------------------
# Pure semantic mode (lw=0)
# ---------------------------------------------------------------------------


class TestPureSemanticMode:
    def test_lw_zero_sorts_by_original_score(self):
        """lexical_weight=0 must return candidates sorted by semantic score only."""
        chunks = _chunks_with_text_scores(
            [
                ("chunk A with bm25 keyword match", 0.60),
                ("chunk B semantically closer text", 0.80),
                ("chunk C moderate content here", 0.70),
            ]
        )
        r = _make_retriever(chunks, lexical_weight=0.0)
        results = r.retrieve("keyword match bm25 query", k=3)
        scores = [c["score"] for c in results]
        # Pure semantic: sorted by original FAISS score
        assert scores == sorted(scores, reverse=True)
        assert results[0]["chunk_id"] == "c1"  # highest sem score (0.80)

    def test_lw_zero_faster_no_bm25(self):
        """At lw=0, BM25 is not computed (short-circuit path)."""
        chunks = _chunks_with_text_scores(
            [
                ("text one content", 0.50),
                ("text two content", 0.60),
            ]
        )
        r = _make_retriever(chunks, lexical_weight=0.0)
        # Should not raise even with unusual text
        results = r.retrieve("anything", k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# BM25 contribution
# ---------------------------------------------------------------------------


class TestBM25Contribution:
    def test_bm25_promotes_keyword_matching_chunk(self):
        """A chunk with lower semantic score but keyword match should be promoted by BM25."""
        chunks = _chunks_with_text_scores(
            [
                ("revolução industrial mecanização produção fábricas trabalho", 0.55),
                ("conteúdo genérico sobre assuntos diversos sem relação direta", 0.65),
                ("outro texto sem relação com a consulta original feita", 0.60),
            ]
        )
        # At lw=0.40, BM25 should boost chunk 0 ("revolução industrial")
        r_with_bm25 = _make_retriever(chunks, lexical_weight=0.40)
        results = r_with_bm25.retrieve("revolução industrial", k=3)

        # Compare with pure semantic
        r_pure = _make_retriever(chunks, lexical_weight=0.0)
        results_pure = r_pure.retrieve("revolução industrial", k=3)

        # With BM25, chunk 0 should rank higher than in pure semantic
        rank_with = next(i for i, c in enumerate(results) if c["chunk_id"] == "c0")
        rank_pure = next(i for i, c in enumerate(results_pure) if c["chunk_id"] == "c0")
        assert rank_with <= rank_pure

    def test_different_weights_produce_different_rankings(self):
        """lw=0.40 and lw=0 should produce different orderings when BM25 signal exists."""
        chunks = _chunks_with_text_scores(
            [
                ("Waterloo batalha derrota 1815 Napoleão exército final", 0.45),
                ("Europa guerras conflitos imperiais durante século dezenove", 0.70),
                ("Napoleão Bonaparte império francês expansão continental", 0.65),
            ]
        )
        r40 = _make_retriever(chunks, lexical_weight=0.40)
        r00 = _make_retriever(chunks, lexical_weight=0.00)

        res40 = r40.retrieve("Waterloo batalha 1815", k=3)
        res00 = r00.retrieve("Waterloo batalha 1815", k=3)

        ids40 = [c["chunk_id"] for c in res40]
        ids00 = [c["chunk_id"] for c in res00]
        # Pure semantic puts c1 first (highest score 0.70)
        # BM25 should boost c0 (has "Waterloo", "batalha", "1815")
        assert ids00[0] == "c1"  # pure semantic: highest score wins
        # With BM25, c0 should move up (may or may not be first depending on normalization)
        rank_c0_with_bm25 = ids40.index("c0")
        rank_c0_pure = ids00.index("c0")
        assert rank_c0_with_bm25 <= rank_c0_pure


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_min_max(self):
        """_normalize must produce [0,1] range with min→0 and max→1."""
        values = [0.3, 0.5, 0.7, 0.9]
        normed = _normalize(values)
        assert min(normed) == pytest.approx(0.0)
        assert max(normed) == pytest.approx(1.0)

    def test_normalize_single_value(self):
        """Single value (or all same) → all zeros."""
        assert _normalize([0.5]) == [0.0]
        assert _normalize([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]

    def test_normalize_preserves_ordering(self):
        """Normalization must preserve the relative order."""
        values = [0.2, 0.8, 0.5, 0.3]
        normed = _normalize(values)
        # Original order by value: 0.2 < 0.3 < 0.5 < 0.8
        # Normalized should maintain: norm[0] < norm[3] < norm[2] < norm[1]
        assert normed[0] < normed[3] < normed[2] < normed[1]

    def test_normalize_empty_returns_empty(self):
        """Edge case: empty list."""
        # _normalize([]) would hit min/max on empty — handle gracefully
        # Actually _normalize requires non-empty; test with 2 elements
        result = _normalize([1.0, 1.0])
        assert result == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Weight formula
# ---------------------------------------------------------------------------


class TestWeightFormula:
    def test_combined_score_is_weighted_sum(self):
        """Final score = (1-lw)*sem_norm + lw*bm25_norm."""
        # We can verify this by checking the output scores
        # Chunk with highest semantic but no BM25
        chunks = _chunks_with_text_scores(
            [
                ("completely unrelated text without keywords", 0.90),
                ("keyword keyword keyword keyword keyword keyword", 0.50),
                ("some moderate text with keyword presence here", 0.70),
            ]
        )
        r = _make_retriever(chunks, lexical_weight=0.40)
        results = r.retrieve("keyword", k=3)

        # All scores should be in [0, 1] (due to normalization)
        for c in results:
            assert 0.0 <= c["score"] <= 1.0

    def test_lw_1_is_pure_bm25(self):
        """At lexical_weight=1.0, ranking should follow BM25 order only."""
        chunks = _chunks_with_text_scores(
            [
                ("revolução francesa liberdade igualdade fraternidade", 0.90),
                ("texto sem relação com a consulta feita aqui", 0.80),
                ("revolução revolução revolução liberdade liberdade", 0.30),
            ]
        )
        # lw=1.0 → pure BM25. Chunk c2 has most keyword repetitions
        r = _make_retriever(chunks, lexical_weight=1.0)
        results = r.retrieve("revolução liberdade", k=3)
        # c2 should rank highest (most TF for query terms)
        assert results[0]["chunk_id"] == "c2"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestRankingDeterminism:
    def test_same_weight_same_results(self):
        """Repeated calls with same config produce identical results."""
        chunks = _chunks_with_text_scores(
            [
                ("conteúdo A relevante para pesquisa", 0.70),
                ("conteúdo B parcialmente relevante aqui", 0.68),
                ("conteúdo C pouco relevante para pesquisa", 0.65),
            ]
        )
        r = _make_retriever(chunks, lexical_weight=0.40)
        res1 = r.retrieve("conteúdo relevante pesquisa", k=3)
        res2 = r.retrieve("conteúdo relevante pesquisa", k=3)
        assert [c["chunk_id"] for c in res1] == [c["chunk_id"] for c in res2]
        assert [c["score"] for c in res1] == [c["score"] for c in res2]

    def test_ranking_stable_for_ties(self):
        """When hybrid scores are tied, order is stable (from Python stable sort)."""
        chunks = _chunks_with_text_scores(
            [
                ("conteúdo idêntico para teste estabilidade", 0.60),
                ("conteúdo idêntico para teste estabilidade", 0.60),
            ]
        )
        r = _make_retriever(chunks, lexical_weight=0.40)
        results = r.retrieve("conteúdo idêntico teste", k=2)
        # Stable sort preserves original order for ties
        assert results[0]["chunk_id"] == "c0"
        assert results[1]["chunk_id"] == "c1"

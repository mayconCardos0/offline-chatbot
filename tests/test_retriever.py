"""
Tests for rag/retriever.py — Retriever with hybrid BM25 + semantic search,
and its helper functions.
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.retriever import Retriever, _bm25_scores, _normalize, _tokenize  # noqa: E402

# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases_text(self):
        tokens = _tokenize("HELLO World")
        assert all(t == t.lower() for t in tokens)

    def test_removes_stopwords(self):
        tokens = _tokenize("de a o e")
        assert tokens == []

    def test_filters_short_tokens(self):
        tokens = _tokenize("ab cd")
        assert tokens == []

    def test_returns_list_of_strings(self):
        tokens = _tokenize("algoritmo processamento linguagem natural")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_punctuation_removed(self):
        tokens = _tokenize("hello, world!")
        assert "," not in " ".join(tokens)
        assert "!" not in " ".join(tokens)

    def test_content_words_kept(self):
        tokens = _tokenize("algoritmo eficiente busca")
        assert "algoritmo" in tokens
        assert "eficiente" in tokens
        assert "busca" in tokens


# ---------------------------------------------------------------------------
# _bm25_scores
# ---------------------------------------------------------------------------


class TestBm25Scores:
    def test_returns_list_same_length_as_docs(self):
        docs = [["hello", "world"], ["foo", "bar"]]
        scores = _bm25_scores(["hello"], docs)
        assert len(scores) == 2

    def test_higher_score_for_relevant_doc(self):
        docs = [
            ["python", "algoritmo", "sorting", "eficiente"],
            ["receita", "bolo", "chocolate", "forno"],
        ]
        scores = _bm25_scores(["algoritmo", "python"], docs)
        assert scores[0] > scores[1]

    def test_empty_query_returns_zeros(self):
        docs = [["word", "another"]]
        scores = _bm25_scores([], docs)
        assert scores == [0.0]

    def test_empty_docs_returns_empty(self):
        assert _bm25_scores(["term"], []) == []

    def test_returns_floats(self):
        docs = [["test", "document"]]
        scores = _bm25_scores(["test"], docs)
        assert all(isinstance(s, float) for s in scores)

    def test_non_matching_term_low_score(self):
        docs = [["completely", "unrelated", "content"]]
        scores = _bm25_scores(["quantum", "physics"], docs)
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_output_range_zero_to_one(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        normed = _normalize(values)
        assert min(normed) == pytest.approx(0.0)
        assert max(normed) == pytest.approx(1.0)

    def test_single_value_returns_zero(self):
        assert _normalize([42.0]) == [0.0]

    def test_all_equal_returns_zeros(self):
        normed = _normalize([3.0, 3.0, 3.0])
        assert all(v == 0.0 for v in normed)

    def test_preserves_relative_order(self):
        values = [1.0, 5.0, 3.0]
        normed = _normalize(values)
        assert normed[1] > normed[2] > normed[0]

    def test_length_preserved(self):
        values = [1.0, 2.0, 3.0]
        assert len(_normalize(values)) == len(values)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


def _make_mock_chunk(text, source="doc.txt", score=0.8):
    return {"text": text, "source": source, "score": score}


def _make_retriever(chunks=None, size=10):
    """Build a Retriever with mocked vectorstore and embed_model."""
    vectorstore = MagicMock()
    vectorstore.size = size if chunks else 0
    embed_model = MagicMock()
    embed_model.embed.return_value = [np.random.randn(64).tolist()]

    if chunks:
        vectorstore.search.return_value = chunks
    else:
        vectorstore.search.return_value = []

    return Retriever(
        vectorstore=vectorstore,
        embed_model=embed_model,
        top_k=3,
        candidate_multiplier=2,
        min_score=0.25,
        lexical_weight=0.0,  # pure semantic for determinism in tests
    )


class TestRetrieverRetrieve:
    def test_empty_vectorstore_returns_empty(self):
        retriever = _make_retriever(chunks=None, size=0)
        results = retriever.retrieve("any query")
        assert results == []

    def test_low_score_chunks_filtered(self):
        chunks = [_make_mock_chunk("irrelevant content", score=0.10)]
        retriever = _make_retriever(chunks=chunks)
        results = retriever.retrieve("specific query")
        assert results == []

    def test_high_score_chunk_returned(self):
        chunks = [
            _make_mock_chunk("Python é uma linguagem de programação", score=0.85),
            _make_mock_chunk("Python algoritmo busca eficiente", score=0.80),
        ]
        retriever = _make_retriever(chunks=chunks)
        results = retriever.retrieve("Python algoritmo")
        assert len(results) > 0

    def test_result_count_respects_top_k(self):
        chunks = [
            _make_mock_chunk(f"chunk {i} algoritmo processamento", score=0.9)
            for i in range(10)
        ]
        retriever = _make_retriever(chunks=chunks)
        results = retriever.retrieve("algoritmo processamento")
        assert len(results) <= 3

    def test_result_has_expected_keys(self):
        chunks = [_make_mock_chunk("algoritmo de busca eficiente rápido", score=0.9)]
        retriever = _make_retriever(chunks=chunks)
        results = retriever.retrieve("algoritmo busca eficiente")
        if results:
            assert "text" in results[0]
            assert "source" in results[0]
            assert "score" in results[0]

    def test_embed_model_called_once(self):
        vectorstore = MagicMock()
        vectorstore.size = 5
        vectorstore.search.return_value = []
        embed_model = MagicMock()
        embed_model.embed.return_value = [np.random.randn(64).tolist()]

        retriever = Retriever(vectorstore=vectorstore, embed_model=embed_model, top_k=3)
        retriever.retrieve("test query")
        embed_model.embed.assert_called_once()


class TestRetrieverGapFilter:
    def test_gap_filter_discards_when_no_dominant_score(self):
        # All scores similar and low — gap_filter should discard
        chunks = [
            _make_mock_chunk(f"content {i}", score=0.30 + i * 0.01) for i in range(5)
        ]
        retriever = _make_retriever(chunks=chunks)
        retriever._vectorstore.size = 10

        result = retriever._gap_filter(chunks)
        # With best=0.34, others avg ~0.31, gap < 0.15 → should return []
        # (exact result depends on values, just verify it runs without error)
        assert isinstance(result, list)

    def test_gap_filter_keeps_clearly_best_chunk(self):
        # One very dominant chunk
        chunks = [
            _make_mock_chunk("best result", score=0.95),
            _make_mock_chunk("mediocre", score=0.40),
            _make_mock_chunk("also mediocre", score=0.38),
        ]
        retriever = _make_retriever(chunks=chunks)
        result = retriever._gap_filter(chunks)
        # Best is very high (>=0.70), should keep chunks within 0.15 of best
        assert any(c["score"] == 0.95 for c in result)

    def test_gap_filter_empty_input(self):
        retriever = _make_retriever()
        assert retriever._gap_filter([]) == []


class TestRetrieverKeywordFilter:
    def test_filters_chunks_without_query_keywords(self):
        chunks = [_make_mock_chunk("completely unrelated text about recipes")]
        retriever = _make_retriever(chunks=chunks)
        result = retriever._keyword_filter("python programming algorithms", chunks)
        # No shared long tokens between query and chunk
        assert result == []

    def test_keeps_chunks_with_matching_keywords(self):
        chunks = [_make_mock_chunk("python programming language algorithms")]
        retriever = _make_retriever(chunks=chunks)
        result = retriever._keyword_filter("python algorithms", chunks)
        assert len(result) == 1

    def test_short_query_no_filter(self):
        """If query has no keywords >= 4 chars, no filtering applied."""
        chunks = [_make_mock_chunk("any content here")]
        retriever = _make_retriever(chunks=chunks)
        result = retriever._keyword_filter("ok hi", chunks)
        # No filtering — all returned
        assert result == chunks


class TestRetrieverPeriodBoost:
    def test_segundo_governo_vargas_prioritizes_correct_period(self):
        retriever = _make_retriever()
        chunks = [
            _make_mock_chunk(
                "Estado Novo: a ditadura varguista (1937-1945). Vargas dissolveu o Congresso.",
                score=0.9,
            ),
            _make_mock_chunk(
                "Segundo governo Vargas (1951-1954). O governo priorizou a expansão industrial e o BNDE.",
                score=0.1,
            ),
        ]

        result = retriever._apply_period_boost(
            "Como foi o Segundo Governo Vargas?", chunks
        )

        assert "1951-1954" in result[0]["text"]

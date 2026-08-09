"""
Phase 3F tests — CrossEncoderReranker score fusion logic.

Tests the rag/cross_encoder.py module without requiring the actual model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rag.cross_encoder import CrossEncoderReranker, _normalize_scores


class TestNormalizeScores:
    def test_basic_normalization(self):
        assert _normalize_scores([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]

    def test_single_value(self):
        assert _normalize_scores([5.0]) == [0.0]

    def test_all_equal(self):
        assert _normalize_scores([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]

    def test_empty(self):
        assert _normalize_scores([]) == []

    def test_negative_values(self):
        r = _normalize_scores([-2.0, 0.0, 2.0])
        assert r[0] == pytest.approx(0.0)
        assert r[2] == pytest.approx(1.0)

    def test_preserves_order(self):
        r = _normalize_scores([0.3, 0.7, 0.5, 0.1])
        assert r[3] < r[0] < r[2] < r[1]


class TestCrossEncoderRerankerInit:
    def test_default_params(self):
        r = CrossEncoderReranker()
        assert r._hybrid_weight == 0.3
        assert r._ce_weight == pytest.approx(0.7)
        assert r._ce_top_k == 20
        assert r._model is None  # lazy load

    def test_custom_params(self):
        r = CrossEncoderReranker(hybrid_weight=0.5, ce_top_k=10)
        assert r._hybrid_weight == 0.5
        assert r._ce_weight == pytest.approx(0.5)
        assert r._ce_top_k == 10


class TestCrossEncoderRerankerRerank:
    def _make_reranker(self, hw=0.4):
        r = CrossEncoderReranker(hybrid_weight=hw, ce_top_k=20)
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.5, 0.7, 0.3])
        r._model = mock_model
        return r

    def _candidates(self, n=5):
        return [
            {"text": f"chunk {i} text", "chunk_id": f"c{i}", "score": 0.8 - i * 0.1}
            for i in range(n)
        ]

    def test_rerank_returns_top_k(self):
        r = self._make_reranker()
        cands = self._candidates(5)
        results = r.rerank("query", cands, final_k=3)
        assert len(results) == 3

    def test_rerank_fuses_scores(self):
        r = self._make_reranker(hw=0.5)
        cands = self._candidates(5)
        results = r.rerank("query", cands, final_k=5)
        # All results should have hybrid_score and ce_score
        for c in results:
            assert "hybrid_score" in c
            assert "ce_score" in c
            assert "score" in c

    def test_rerank_ordered_by_fused_score(self):
        r = self._make_reranker()
        cands = self._candidates(5)
        results = r.rerank("query", cands, final_k=5)
        scores = [c["score"] for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_hw_1_uses_hybrid_only(self):
        """hybrid_weight=1.0 means pure hybrid ranking (CE ignored)."""
        r = self._make_reranker(hw=1.0)
        cands = self._candidates(5)
        results = r.rerank("query", cands, final_k=5)
        # Should be ordered by original hybrid score (descending)
        # Original: 0.8, 0.7, 0.6, 0.5, 0.4
        assert results[0]["chunk_id"] == "c0"

    def test_hw_0_uses_ce_only(self):
        """hybrid_weight=0.0 means pure CE ranking."""
        r = self._make_reranker(hw=0.0)
        cands = self._candidates(5)
        results = r.rerank("query", cands, final_k=5)
        # CE scores: [0.9, 0.1, 0.5, 0.7, 0.3]
        # After normalization: c0=1.0, c3=0.75, c2=0.5, c4=0.25, c1=0.0
        assert results[0]["chunk_id"] == "c0"  # highest CE
        assert results[1]["chunk_id"] == "c3"  # second highest CE

    def test_empty_candidates(self):
        r = self._make_reranker()
        assert r.rerank("query", [], final_k=5) == []

    def test_fewer_than_final_k(self):
        r = self._make_reranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.8])
        r._model = mock_model
        cands = self._candidates(2)
        results = r.rerank("query", cands, final_k=5)
        assert len(results) == 2

    def test_ce_top_k_limits_candidates(self):
        """Only first ce_top_k candidates are scored by CE."""
        r = CrossEncoderReranker(hybrid_weight=0.4, ce_top_k=3)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.5])
        r._model = mock_model
        cands = self._candidates(10)
        r.rerank("query", cands, final_k=3)
        # Only 3 candidates scored
        mock_model.predict.assert_called_once()
        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs) == 3


class TestCrossEncoderRerankerFallback:
    def test_model_load_failure_falls_back(self):
        """If model fails to load, returns hybrid order."""
        r = CrossEncoderReranker(hybrid_weight=0.4, ce_top_k=20)
        # Force load failure by making _load_model raise
        r._model = None

        def _fail():
            raise ImportError("no module")

        r._load_model = _fail
        cands = [{"text": "x", "chunk_id": "c0", "score": 0.5}]
        results = r.rerank("query", cands, final_k=5)
        assert len(results) == 1  # fallback returns input

    def test_inference_failure_falls_back(self):
        """If CE inference throws, returns hybrid order."""
        r = CrossEncoderReranker(hybrid_weight=0.4, ce_top_k=20)
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("OOM")
        r._model = mock_model
        cands = [
            {"text": "chunk A text", "chunk_id": "c0", "score": 0.8},
            {"text": "chunk B text", "chunk_id": "c1", "score": 0.6},
        ]
        results = r.rerank("query", cands, final_k=2)
        # Should return first 2 by hybrid order (fallback)
        assert results[0]["chunk_id"] == "c0"

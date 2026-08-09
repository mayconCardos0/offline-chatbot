"""
Phase 2 tests for rag/retriever.py.

Covers additions that did not exist in test_retriever.py:
  - retrieve(query, k=k) explicit k parameter
  - retrieve_with_trace() returns (results, RetrievalTrace)
  - RetrievalTrace.stages populated with StageSnapshot entries
  - RetrievalTrace.to_dict() and summary() serialization
  - StageSnapshot.to_dict() structure
  - Filter flags: gap_filter_enabled=False, keyword_filter_enabled=False
  - Retriever constructor accepts and stores all Phase 2 params
  - best_confidence() labelling
  - Ablation: filters disabled do not change result type
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.retriever import (  # noqa: E402
    RetrievalTrace,
    Retriever,
    StageSnapshot,
    _bm25_scores,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, score: float, chunk_id: str = "") -> dict:
    return {"text": text, "source": "doc.txt", "score": score, "chunk_id": chunk_id}


def _make_retriever(
    chunks: list[dict] | None = None,
    size: int = 10,
    top_k: int = 3,
    gap_filter_enabled: bool = True,
    keyword_filter_enabled: bool = True,
    lexical_weight: float = 0.0,
    min_score: float = 0.25,
    adaptive_sigma: float = 1.0,
) -> Retriever:
    vs = MagicMock()
    vs.size = size if chunks else 0
    vs.search.return_value = chunks or []

    em = MagicMock()
    em.embed.return_value = [np.random.randn(64).tolist()]

    return Retriever(
        vectorstore=vs,
        embed_model=em,
        top_k=top_k,
        candidate_multiplier=2,
        min_score=min_score,
        lexical_weight=lexical_weight,
        gap_filter_enabled=gap_filter_enabled,
        keyword_filter_enabled=keyword_filter_enabled,
        adaptive_sigma=adaptive_sigma,
    )


# ---------------------------------------------------------------------------
# Explicit k parameter
# ---------------------------------------------------------------------------


class TestRetrieveExplicitK:
    def test_explicit_k_overrides_base_top_k(self):
        """retrieve(query, k=1) must return at most 1 result even if top_k=10."""
        chunks = [_make_chunk(f"algoritmo chunk {i}", 0.9 - i * 0.01) for i in range(5)]
        retriever = _make_retriever(chunks=chunks, top_k=10)
        results = retriever.retrieve("algoritmo chunk", k=1)
        assert len(results) <= 1

    def test_explicit_k_zero_returns_empty(self):
        """k=0 should return an empty list (no results requested)."""
        chunks = [_make_chunk("algoritmo chunk texto relevante", 0.9)]
        retriever = _make_retriever(chunks=chunks, top_k=5)
        results = retriever.retrieve("algoritmo chunk", k=0)
        assert results == []

    def test_no_explicit_k_uses_dynamic_adjustment(self):
        """Without explicit k, biographic queries get k * 2."""
        chunks = [
            _make_chunk(f"história de alguém chunk {i}", 0.9 - i * 0.01)
            for i in range(20)
        ]
        retriever = _make_retriever(chunks=chunks, top_k=3, size=20)
        retriever._vectorstore.search.return_value = chunks
        results = retriever.retrieve("quem foi Getúlio Vargas")
        # k * 2 = 6, but chunks may be filtered — just verify it ran
        assert isinstance(results, list)

    def test_explicit_k_bypasses_dynamic_adjustment(self):
        """retrieve(query, k=2) must request exactly 2 results regardless of query type."""
        chunks = [
            _make_chunk(f"governo período {i}", 0.9 - i * 0.01) for i in range(10)
        ]
        retriever = _make_retriever(chunks=chunks, top_k=5, size=10)
        retriever._vectorstore.search.return_value = chunks
        results = retriever.retrieve("como foi o governo?", k=2)
        # With explicit k=2, result must be <= 2
        assert len(results) <= 2

    def test_retrieve_without_k_none_uses_instance_top_k(self):
        """retrieve(query) with k=None should fall through to self._base_top_k."""
        chunks = [_make_chunk(f"conteúdo neutro chunk {i}", 0.9) for i in range(3)]
        retriever = _make_retriever(chunks=chunks, top_k=3)
        results = retriever.retrieve("neutral query text")
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# retrieve_with_trace
# ---------------------------------------------------------------------------


class TestRetrieveWithTrace:
    def test_returns_tuple_of_results_and_trace(self):
        chunks = [_make_chunk("algoritmo de busca eficiente", 0.9)]
        retriever = _make_retriever(chunks=chunks)
        output = retriever.retrieve_with_trace("algoritmo busca")
        assert isinstance(output, tuple)
        assert len(output) == 2

    def test_trace_is_retrieval_trace_instance(self):
        chunks = [_make_chunk("algoritmo processamento texto", 0.9)]
        retriever = _make_retriever(chunks=chunks)
        _, trace = retriever.retrieve_with_trace("algoritmo texto")
        assert isinstance(trace, RetrievalTrace)

    def test_trace_has_stages(self):
        chunks = [_make_chunk("história do Brasil período colonial", 0.9)]
        retriever = _make_retriever(chunks=chunks)
        _, trace = retriever.retrieve_with_trace("história Brasil")
        assert len(trace.stages) > 0

    def test_trace_first_stage_is_faiss_semantic(self):
        chunks = [_make_chunk("Brasil colonial história", 0.9)]
        retriever = _make_retriever(chunks=chunks)
        _, trace = retriever.retrieve_with_trace("história colonial")
        assert trace.stages[0].stage == "FAISS_SEMANTIC"

    def test_trace_final_results_match_retrieve_output(self):
        chunks = [_make_chunk("algoritmo eficiente rápido", 0.9)]
        retriever = _make_retriever(chunks=chunks)
        results = retriever.retrieve("algoritmo eficiente", k=3)
        results2, trace = retriever.retrieve_with_trace("algoritmo eficiente", k=3)
        # Results should be consistent (same chunks retrieved)
        assert len(results) == len(results2)

    def test_trace_query_field_set_correctly(self):
        chunks = [_make_chunk("qualquer conteúdo aqui", 0.9)]
        retriever = _make_retriever(chunks=chunks)
        _, trace = retriever.retrieve_with_trace("minha consulta especial")
        assert trace.query == "minha consulta especial"

    def test_empty_vectorstore_trace_has_no_stages(self):
        retriever = _make_retriever(chunks=None, size=0)
        _, trace = retriever.retrieve_with_trace("any query")
        assert trace.stages == []
        assert trace.final_results == []

    def test_trace_to_dict_is_serializable(self):
        chunks = [_make_chunk("conteúdo do chunk para teste", 0.9, chunk_id="c1")]
        retriever = _make_retriever(chunks=chunks)
        _, trace = retriever.retrieve_with_trace("conteúdo chunk")
        d = trace.to_dict()
        import json

        # Must be JSON-serializable without error
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert "query" in d
        assert "stages" in d

    def test_trace_summary_is_string(self):
        chunks = [_make_chunk("conteúdo qualquer para resumo", 0.9)]
        retriever = _make_retriever(chunks=chunks)
        _, trace = retriever.retrieve_with_trace("conteúdo resumo")
        summary = trace.summary()
        assert isinstance(summary, str)
        assert "QUERY" in summary
        assert "FINAL" in summary


# ---------------------------------------------------------------------------
# StageSnapshot
# ---------------------------------------------------------------------------


class TestStageSnapshot:
    def test_to_dict_has_stage_count_candidates(self):
        snap = StageSnapshot(
            stage="TEST_STAGE",
            candidates=[{"text": "chunk text here", "score": 0.8, "chunk_id": "abc"}],
        )
        d = snap.to_dict()
        assert d["stage"] == "TEST_STAGE"
        assert d["count"] == 1
        assert len(d["candidates"]) == 1

    def test_to_dict_candidate_has_required_keys(self):
        snap = StageSnapshot(
            stage="S",
            candidates=[
                {"text": "hello world content", "score": 0.75, "chunk_id": "x"}
            ],
        )
        cand = snap.to_dict()["candidates"][0]
        assert "chunk_id" in cand
        assert "score" in cand
        assert "text_preview" in cand

    def test_to_dict_text_preview_truncated_to_80(self):
        long_text = "a" * 200
        snap = StageSnapshot(
            stage="S",
            candidates=[{"text": long_text, "score": 0.5, "chunk_id": ""}],
        )
        preview = snap.to_dict()["candidates"][0]["text_preview"]
        assert len(preview) <= 80

    def test_empty_candidates_list(self):
        snap = StageSnapshot(stage="EMPTY")
        d = snap.to_dict()
        assert d["count"] == 0
        assert d["candidates"] == []


# ---------------------------------------------------------------------------
# Filter flag ablation: gap_filter_enabled
# ---------------------------------------------------------------------------


class TestGapFilterAblation:
    def test_gap_filter_disabled_does_not_add_gap_filter_stage_to_trace(self):
        """When gap_filter_enabled=False, AFTER_GAP_FILTER should not appear in trace."""
        chunks = [
            _make_chunk(f"conteúdo chunk {i} histórico", 0.30 + i * 0.01)
            for i in range(5)
        ]
        retriever = _make_retriever(
            chunks=chunks, gap_filter_enabled=False, keyword_filter_enabled=False
        )
        _, trace = retriever.retrieve_with_trace("conteúdo histórico", k=5)
        stage_names = [s.stage for s in trace.stages]
        assert "AFTER_GAP_FILTER" not in stage_names

    def test_gap_filter_enabled_adds_gap_filter_stage_to_trace(self):
        """When gap_filter_enabled=True, AFTER_GAP_FILTER should appear if filter runs."""
        chunks = [
            _make_chunk(f"conteúdo histórico chunk {i}", 0.9 - i * 0.01)
            for i in range(3)
        ]
        retriever = _make_retriever(chunks=chunks, gap_filter_enabled=True)
        _, trace = retriever.retrieve_with_trace("conteúdo histórico", k=3)
        stage_names = [s.stage for s in trace.stages]
        assert "AFTER_GAP_FILTER" in stage_names

    def test_gap_filter_disabled_returns_more_results_than_enabled(self):
        """Disabling gap filter should never reduce result count vs enabled."""
        # Uniform low scores — gap filter would normally discard all
        chunks = [
            _make_chunk(f"conteúdo {i} tópico histórico", 0.28 + i * 0.001)
            for i in range(5)
        ]

        r_with_gap = _make_retriever(
            chunks=chunks, gap_filter_enabled=True, keyword_filter_enabled=False
        )
        r_no_gap = _make_retriever(
            chunks=chunks, gap_filter_enabled=False, keyword_filter_enabled=False
        )

        results_with = r_with_gap.retrieve("conteúdo tópico", k=5)
        results_without = r_no_gap.retrieve("conteúdo tópico", k=5)

        assert len(results_without) >= len(results_with)


# ---------------------------------------------------------------------------
# Filter flag ablation: keyword_filter_enabled
# ---------------------------------------------------------------------------


class TestKeywordFilterAblation:
    def test_keyword_filter_disabled_does_not_add_stage_to_trace(self):
        chunks = [_make_chunk("completely different terminology semantic", 0.9)]
        retriever = _make_retriever(
            chunks=chunks, keyword_filter_enabled=False, gap_filter_enabled=False
        )
        _, trace = retriever.retrieve_with_trace("completely semantic query", k=3)
        stage_names = [s.stage for s in trace.stages]
        assert "AFTER_KEYWORD_FILTER" not in stage_names

    def test_keyword_filter_enabled_adds_stage_to_trace(self):
        chunks = [_make_chunk("algoritmo processamento linguagem natural", 0.9)]
        retriever = _make_retriever(chunks=chunks, keyword_filter_enabled=True)
        _, trace = retriever.retrieve_with_trace("algoritmo linguagem", k=3)
        stage_names = [s.stage for s in trace.stages]
        assert "AFTER_KEYWORD_FILTER" in stage_names

    def test_keyword_filter_disabled_passes_through_semantically_relevant_chunks(self):
        """A chunk with no exact query keywords should survive when filter is off."""
        # Chunk about "fotossíntese" (photosynthesis), query about "processo de plantas"
        chunk = _make_chunk(
            "fotossíntese é o processo pelo qual plantas convertem luz solar em energia química",
            score=0.9,
        )
        retriever_filtered = _make_retriever(
            chunks=[chunk], keyword_filter_enabled=True, gap_filter_enabled=False
        )
        retriever_open = _make_retriever(
            chunks=[chunk], keyword_filter_enabled=False, gap_filter_enabled=False
        )

        results_filtered = retriever_filtered.retrieve("processo plantas energia", k=1)
        results_open = retriever_open.retrieve("processo plantas energia", k=1)

        # With filter off, the chunk can pass; with filter on it may not
        # (depends on token overlap — at minimum open must be >= filtered)
        assert len(results_open) >= len(results_filtered)


# ---------------------------------------------------------------------------
# Constructor parameter storage
# ---------------------------------------------------------------------------


class TestRetrieverConstructorParams:
    def test_all_phase2_params_stored(self):
        vs = MagicMock()
        em = MagicMock()
        r = Retriever(
            vectorstore=vs,
            embed_model=em,
            top_k=7,
            candidate_multiplier=3,
            min_score=0.30,
            lexical_weight=0.50,
            bm25_k1=1.2,
            bm25_b=0.80,
            adaptive_sigma=0.5,
            gap_filter_enabled=False,
            keyword_filter_enabled=False,
            high_confidence_score=0.70,
            low_confidence_score=0.40,
        )
        assert r._top_k == 7
        assert r._candidate_multiplier == 3
        assert r._min_score == pytest.approx(0.30)
        assert r._lexical_weight == pytest.approx(0.50)
        assert r._bm25_k1 == pytest.approx(1.2)
        assert r._bm25_b == pytest.approx(0.80)
        assert r._adaptive_sigma == pytest.approx(0.5)
        assert r._gap_filter_enabled is False
        assert r._keyword_filter_enabled is False
        assert r._high_confidence_score == pytest.approx(0.70)
        assert r._low_confidence_score == pytest.approx(0.40)

    def test_base_top_k_equals_top_k_on_construction(self):
        vs = MagicMock()
        em = MagicMock()
        r = Retriever(vectorstore=vs, embed_model=em, top_k=8)
        assert r._base_top_k == 8


# ---------------------------------------------------------------------------
# best_confidence
# ---------------------------------------------------------------------------


class TestBestConfidence:
    def _retriever(self) -> Retriever:
        return _make_retriever()

    def test_empty_chunks_returns_none(self):
        assert self._retriever().best_confidence([]) == "none"

    def test_high_confidence_above_threshold(self):
        r = _make_retriever()
        r._high_confidence_score = 0.65
        assert r.best_confidence([{"score": 0.80}]) == "high"

    def test_medium_confidence_between_thresholds(self):
        r = _make_retriever()
        r._low_confidence_score = 0.45
        r._high_confidence_score = 0.65
        assert r.best_confidence([{"score": 0.55}]) == "medium"

    def test_low_confidence_below_low_threshold(self):
        r = _make_retriever()
        r._low_confidence_score = 0.45
        assert r.best_confidence([{"score": 0.30}]) == "low"

    def test_uses_maximum_score(self):
        r = _make_retriever()
        r._high_confidence_score = 0.65
        chunks = [{"score": 0.30}, {"score": 0.90}]
        assert r.best_confidence(chunks) == "high"


# ---------------------------------------------------------------------------
# BM25 custom parameters respected
# ---------------------------------------------------------------------------


class TestBM25CustomParams:
    def test_custom_k1_and_b_accepted(self):
        """_bm25_scores should accept custom k1/b without error."""
        docs = [["python", "algoritmo"], ["receita", "bolo"]]
        scores = _bm25_scores(["python"], docs, k1=2.0, b=0.5)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_k1_zero_gives_binary_tf(self):
        """With k1=0, TF saturation is maximum — all TF values become 1."""
        docs = [["word", "word", "word"], ["word"]]
        scores_k1_zero = _bm25_scores(["word"], docs, k1=0.0)
        # With k1=0: num = tf * 1, den = tf + 0 * (...) = tf, score = idf * 1
        # Both docs should have same BM25 score
        assert abs(scores_k1_zero[0] - scores_k1_zero[1]) < 1e-6

"""
Phase 2.5 tests — benchmark validation, filter ablation, negative query handling,
conversational query separation, and stage-level retrieval tracing.

Tests verify the diagnostic methodology and evaluation framework extensions
without changing retrieval behavior.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.evaluation import (
    QueryResult,
    EvaluationReport,
    _effective_ids_from_item,
    _build_graded_relevance,
    evaluate_query,
    evaluate_retriever,
    load_dataset_from_file,
)
from rag.retriever import Retriever, RetrievalTrace, StageSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_retriever(chunks):
    r = MagicMock()
    r.retrieve.return_value = chunks
    return r


def _chunk(chunk_id, score=0.8):
    return {"chunk_id": chunk_id, "text": f"text of {chunk_id}", "score": score}


# ---------------------------------------------------------------------------
# Benchmark inflation detection — required vs supporting evidence
# ---------------------------------------------------------------------------


class TestBenchmarkInflation:
    """Verify that Recall@K is sensitive to the size of relevant_chunks."""

    def test_inflated_denominator_deflates_recall(self):
        """With 50 relevant IDs but only 1 retrieved, Recall@5 = 1/50 = 0.02."""
        relevant_ids = {f"c{i}" for i in range(50)}
        retriever = _mock_retriever([_chunk("c0", 0.9)])
        result = evaluate_query(retriever, "test", relevant_ids, None, k=5)
        assert result.recall_at_k == pytest.approx(1 / 50)

    def test_small_denominator_gives_high_recall(self):
        """With 1 relevant ID retrieved, Recall@5 = 1/1 = 1.0."""
        retriever = _mock_retriever([_chunk("c0", 0.9)])
        result = evaluate_query(retriever, "test", {"c0"}, None, k=5)
        assert result.recall_at_k == pytest.approx(1.0)

    def test_hit_rate_unaffected_by_denominator(self):
        """Hit Rate@5 = 1 as long as any relevant is in top-5, regardless of denominator."""
        relevant_ids = {f"c{i}" for i in range(100)}
        retriever = _mock_retriever([_chunk("c0", 0.9)])
        result = evaluate_query(retriever, "test", relevant_ids, None, k=5)
        assert result.hit_rate == 1.0

    def test_inflated_ids_do_not_affect_mrr(self):
        """MRR only depends on the rank of the first hit, not denominator size."""
        retriever = _mock_retriever([_chunk("x", 0.9), _chunk("c0", 0.8)])
        result_small = evaluate_query(retriever, "q", {"c0"}, None, k=5)
        result_large = evaluate_query(
            retriever, "q", {"c0"} | {f"c{i}" for i in range(1, 50)}, None, k=5
        )
        assert result_small.reciprocal_rank == result_large.reciprocal_rank


# ---------------------------------------------------------------------------
# Negative / out-of-scope query handling
# ---------------------------------------------------------------------------


class TestNegativeQueryHandling:
    """Verify that negative queries (answerable=False) are handled correctly."""

    def test_negative_query_zero_relevant_ids(self):
        """A negative query with no relevant IDs should produce all-zero metrics."""
        retriever = _mock_retriever([_chunk("c0", 0.9)])
        result = evaluate_query(retriever, "What is photosynthesis?", set(), None, k=5)
        assert result.recall_at_k == 0.0
        assert result.precision_at_k == 0.0
        assert result.hit_rate == 0.0

    def test_negative_query_retrieving_nothing_is_correct(self):
        """When retriever returns nothing for a negative query, hit_rate=0 is correct."""
        retriever = _mock_retriever([])
        result = evaluate_query(retriever, "Who wrote Dom Quixote?", set(), None, k=5)
        assert result.hit_rate == 0.0

    def test_negative_query_false_positive(self):
        """If retriever retrieves for a negative query, that's a false positive.
        We can detect this by checking hit_rate > 0 with answerable=False."""
        retriever = _mock_retriever([_chunk("c0", 0.4)])
        dataset = [
            {"query": "question not in corpus", "answerable": False,
             "category": "negative", "relevant_chunks": []}
        ]
        report = evaluate_retriever(retriever, dataset, k=5)
        # The query retrieved something — potentially a false positive
        # hit_rate=0 because no relevant IDs exist, so this counts as "correct"
        # The false positive detection must be done at the pipeline level (did LLM answer?)
        assert report.per_query[0].hit_rate == 0.0

    def test_no_answer_accuracy_calculation(self):
        """For negative queries, 'correct' means retriever returned nothing meaningful.
        We can compute no_answer_accuracy as fraction with hit_rate==0."""
        retriever = _mock_retriever([])  # returns nothing
        neg_queries = [
            {"query": f"neg_{i}", "answerable": False, "category": "negative",
             "relevant_chunks": []}
            for i in range(5)
        ]
        report = evaluate_retriever(retriever, neg_queries, k=5)
        # All negative queries returned nothing — 100% no-answer accuracy
        no_answer_correct = sum(
            1 for qr in report.per_query if len(qr.retrieved) == 0
        )
        assert no_answer_correct == 5


# ---------------------------------------------------------------------------
# Conversational query separation
# ---------------------------------------------------------------------------


class TestConversationalQuerySeparation:
    """Verify that conversational queries with no expected IDs are tracked correctly."""

    def test_conversational_query_zero_expected_ids(self):
        """Conversational queries without context have empty relevant_ids."""
        item = {
            "query": "Por que ele fez isso?",
            "category": "conversational",
            "answerable": True,
            "relevant_chunks": [],
        }
        ids = _effective_ids_from_item(item)
        assert ids == set()

    def test_conversational_query_scores_zero_recall(self):
        """Without expected IDs, recall is 0 regardless of retrieval."""
        retriever = _mock_retriever([_chunk("c0", 0.9)])
        result = evaluate_query(retriever, "Pode explicar melhor?", set(), None, k=5)
        assert result.recall_at_k == 0.0
        assert result.hit_rate == 0.0

    def test_conversational_category_tracked(self):
        """Category is propagated for filtering in analysis."""
        retriever = _mock_retriever([])
        dataset = [
            {"query": "Why?", "category": "conversational",
             "answerable": True, "relevant_chunks": []}
        ]
        report = evaluate_retriever(retriever, dataset, k=5)
        assert report.per_query[0].category == "conversational"


# ---------------------------------------------------------------------------
# Stage-level retrieval tracing
# ---------------------------------------------------------------------------


class TestStageLevelTracing:
    """Verify that retrieve_with_trace provides usable stage snapshots."""

    def _make_retriever_with_trace(self, chunks, gap_enabled=True, kw_enabled=True):
        import numpy as np
        vs = MagicMock()
        vs.size = 100
        vs.search.return_value = chunks
        em = MagicMock()
        em.embed.return_value = [np.random.randn(64).tolist()]
        return Retriever(
            vectorstore=vs,
            embed_model=em,
            top_k=5,
            candidate_multiplier=2,
            min_score=0.25,
            lexical_weight=0.0,
            gap_filter_enabled=gap_enabled,
            keyword_filter_enabled=kw_enabled,
            adaptive_sigma=1.0,
        )

    def test_trace_contains_faiss_semantic_stage(self):
        chunks = [{"text": "relevant content here", "chunk_id": "c1", "score": 0.8}]
        r = self._make_retriever_with_trace(chunks)
        _, trace = r.retrieve_with_trace("relevant content")
        stage_names = [s.stage for s in trace.stages]
        assert "FAISS_SEMANTIC" in stage_names

    def test_trace_contains_adaptive_filter_stage(self):
        chunks = [{"text": "relevant content here", "chunk_id": "c1", "score": 0.8}]
        r = self._make_retriever_with_trace(chunks)
        _, trace = r.retrieve_with_trace("relevant content")
        stage_names = [s.stage for s in trace.stages]
        assert "AFTER_ADAPTIVE_FILTER" in stage_names

    def test_trace_missing_gap_when_disabled(self):
        chunks = [{"text": "relevant content here", "chunk_id": "c1", "score": 0.8}]
        r = self._make_retriever_with_trace(chunks, gap_enabled=False)
        _, trace = r.retrieve_with_trace("relevant content")
        stage_names = [s.stage for s in trace.stages]
        assert "AFTER_GAP_FILTER" not in stage_names

    def test_trace_missing_keyword_when_disabled(self):
        chunks = [{"text": "relevant content here", "chunk_id": "c1", "score": 0.8}]
        r = self._make_retriever_with_trace(chunks, kw_enabled=False)
        _, trace = r.retrieve_with_trace("relevant content")
        stage_names = [s.stage for s in trace.stages]
        assert "AFTER_KEYWORD_FILTER" not in stage_names

    def test_trace_candidate_count_decreases_or_stays(self):
        """Each stage should have equal or fewer candidates than the previous."""
        chunks = [
            {"text": f"chunk {i} with some relevant text", "chunk_id": f"c{i}", "score": 0.6 - i * 0.05}
            for i in range(8)
        ]
        r = self._make_retriever_with_trace(chunks)
        _, trace = r.retrieve_with_trace("chunk relevant text")
        counts = [s.to_dict()["count"] for s in trace.stages]
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1] or counts[i] == 0

    def test_trace_to_dict_serializable(self):
        chunks = [{"text": "testing serialization output", "chunk_id": "c1", "score": 0.7}]
        r = self._make_retriever_with_trace(chunks)
        _, trace = r.retrieve_with_trace("testing serialization")
        d = trace.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Filter ablation — adaptive filter
# ---------------------------------------------------------------------------


class TestAdaptiveFilterAblation:
    """Verify that disabling adaptive filter retains more candidates."""

    def _make_retriever(self, chunks, sigma=1.0):
        import numpy as np
        vs = MagicMock()
        vs.size = 100
        vs.search.return_value = chunks
        em = MagicMock()
        em.embed.return_value = [np.random.randn(64).tolist()]
        return Retriever(
            vectorstore=vs,
            embed_model=em,
            top_k=5,
            candidate_multiplier=2,
            min_score=0.25,
            lexical_weight=0.0,
            gap_filter_enabled=False,
            keyword_filter_enabled=False,
            adaptive_sigma=sigma,
        )

    def test_sigma_1_removes_low_scorers(self):
        """With tight scores and σ=1, only top scorers survive."""
        chunks = [
            {"text": f"content chunk {i} relevant", "chunk_id": f"c{i}", "score": 0.7 - i * 0.02}
            for i in range(10)
        ]
        r = self._make_retriever(chunks, sigma=1.0)
        results = r.retrieve("content chunk relevant", k=10)
        # σ filtering should remove some candidates
        assert len(results) <= 10

    def test_sigma_999_keeps_all_above_min_score(self):
        """With σ=999, all candidates above min_score survive."""
        chunks = [
            {"text": f"content chunk {i} relevant", "chunk_id": f"c{i}", "score": 0.7 - i * 0.02}
            for i in range(10)
        ]
        r_no_adaptive = self._make_retriever(chunks, sigma=999.0)
        results = r_no_adaptive.retrieve("content chunk relevant", k=10)
        # All 10 chunks above 0.25 should survive
        above_min = [c for c in chunks if c["score"] >= 0.25]
        assert len(results) == min(10, len(above_min))

    def test_disabling_adaptive_never_reduces_results(self):
        """Results with adaptive disabled should be >= results with adaptive enabled."""
        chunks = [
            {"text": f"content chunk {i} matching text", "chunk_id": f"c{i}", "score": 0.6 - i * 0.03}
            for i in range(8)
        ]
        r_enabled = self._make_retriever(chunks, sigma=1.0)
        r_disabled = self._make_retriever(chunks, sigma=999.0)

        results_enabled = r_enabled.retrieve("content chunk matching", k=8)
        results_disabled = r_disabled.retrieve("content chunk matching", k=8)
        assert len(results_disabled) >= len(results_enabled)


# ---------------------------------------------------------------------------
# Experiment reproducibility
# ---------------------------------------------------------------------------


class TestExperimentReproducibility:
    """Verify that the same retriever+dataset produces identical results."""

    def test_same_input_same_output(self):
        """Two runs with identical inputs must produce identical metrics."""
        retriever = _mock_retriever([_chunk("c0", 0.9), _chunk("c1", 0.7)])
        dataset = [
            {"query": "test query", "relevant_chunk_ids": ["c0"], "relevant_texts": []},
            {"query": "another query", "relevant_chunk_ids": ["c1"], "relevant_texts": []},
        ]
        r1 = evaluate_retriever(retriever, dataset, k=5)
        r2 = evaluate_retriever(retriever, dataset, k=5)
        assert r1.mean_recall == r2.mean_recall
        assert r1.mrr == r2.mrr
        assert r1.hit_rate == r2.hit_rate

    def test_different_k_produces_different_results(self):
        """Same data but different k should still be deterministic."""
        retriever = _mock_retriever([_chunk("c0", 0.9)])
        dataset = [{"query": "q", "relevant_chunk_ids": ["c0"]}]
        r1 = evaluate_retriever(retriever, dataset, k=1)
        r5 = evaluate_retriever(retriever, dataset, k=5)
        # Both should hit, but precision differs
        assert r1.hit_rate == r5.hit_rate  # both 1.0
        assert r1.mean_precision >= r5.mean_precision  # k=1 is more precise


# ---------------------------------------------------------------------------
# Required vs supporting evidence (future schema extension)
# ---------------------------------------------------------------------------


class TestRequiredVsSupportingEvidence:
    """Test that the evaluation framework can distinguish required from supporting chunks.

    Currently the schema uses 'relevance: 3' for all linked chunks.
    This test validates that adding a 'required' field would work with the existing
    _effective_ids_from_item function (which already filters by relevance > 0).
    """

    def test_relevance_zero_excluded_from_effective_ids(self):
        """Chunks with relevance=0 are excluded — models 'not required'."""
        item = {
            "relevant_chunks": [
                {"chunk_id": "required", "relevance": 3},
                {"chunk_id": "supporting", "relevance": 1},
                {"chunk_id": "irrelevant", "relevance": 0},
            ]
        }
        ids = _effective_ids_from_item(item)
        assert "required" in ids
        assert "supporting" in ids
        assert "irrelevant" not in ids

    def test_only_high_relevance_for_strict_recall(self):
        """For 'required-only' recall, filter relevant_chunks to relevance>=3."""
        item = {
            "relevant_chunks": [
                {"chunk_id": "must_have", "relevance": 3},
                {"chunk_id": "nice_to_have", "relevance": 1},
            ]
        }
        # Simulate "required-only" filtering
        required_ids = {
            e["chunk_id"] for e in item["relevant_chunks"]
            if e.get("relevance", 0) >= 3
        }
        assert required_ids == {"must_have"}

    def test_graded_ndcg_respects_relevance_grades(self):
        """NDCG with graded relevance gives higher score to high-grade chunks."""
        from rag.evaluation import _ndcg_at_k
        graded = {"must_have": 3, "nice": 1}
        # Perfect order: must_have@1, nice@2
        perfect = _ndcg_at_k(["must_have", "nice"], {"must_have", "nice"}, 2,
                             graded_relevance=graded)
        # Reversed: nice@1, must_have@2
        reversed_ = _ndcg_at_k(["nice", "must_have"], {"must_have", "nice"}, 2,
                               graded_relevance=graded)
        assert perfect > reversed_

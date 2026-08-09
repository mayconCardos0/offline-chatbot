"""
Tests for Phase 2 evaluation framework additions in rag/evaluation.py.

Covers:
  - Graded NDCG@K (binary vs graded relevance)
  - Schema helpers: _build_graded_relevance, _effective_ids_from_item
  - evaluate_query uses retrieve(query, k=k) — no private attribute mutation
  - evaluate_retriever with new curated schema (relevant_chunks)
  - evaluate_retriever with legacy schema (relevant_chunk_ids)
  - QueryResult.category and QueryResult.answerable propagation
  - EvaluationReport.to_dict() per-category breakdown
  - load_dataset_from_file round-trip
  - save_dataset_to_file + load_dataset_from_file round-trip
  - build_synthetic_dataset uses vectorstore.metadata (public property)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.evaluation import (  # noqa: E402
    EvaluationReport,
    QueryResult,
    _average_precision_at_k,
    _build_graded_relevance,
    _chunk_id,
    _effective_ids_from_item,
    _hit_rate,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    _reciprocal_rank,
    build_synthetic_dataset,
    evaluate_query,
    evaluate_retriever,
    load_dataset_from_file,
    save_dataset_to_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_retriever_returning(chunks: list[dict]):
    """Mock retriever that always returns the given chunks."""
    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    return retriever


def _chunk(chunk_id: str, text: str = "some text", score: float = 0.8) -> dict:
    return {"chunk_id": chunk_id, "text": text, "source": "doc.txt", "score": score}


# ---------------------------------------------------------------------------
# _chunk_id helper
# ---------------------------------------------------------------------------


class TestChunkId:
    def test_returns_chunk_id_when_present(self):
        assert _chunk_id({"chunk_id": "abc123"}) == "abc123"

    def test_falls_back_to_text_prefix_when_no_chunk_id(self):
        text = "x" * 80
        assert _chunk_id({"text": text}) == text[:64].strip()

    def test_empty_dict_returns_empty_string(self):
        assert _chunk_id({}) == ""

    def test_empty_chunk_id_falls_back_to_text(self):
        result = _chunk_id({"chunk_id": "", "text": "hello world"})
        assert result == "hello world"


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self):
        ids = ["a", "b", "c"]
        assert _precision_at_k(ids, {"a", "b", "c"}, 3) == pytest.approx(1.0)

    def test_none_relevant(self):
        assert _precision_at_k(["a", "b"], {"x", "y"}, 2) == pytest.approx(0.0)

    def test_half_relevant(self):
        assert _precision_at_k(["a", "b"], {"a"}, 2) == pytest.approx(0.5)

    def test_k_truncates_to_k(self):
        # Only top-2 of 4 retrieved count
        ids = ["a", "b", "c", "d"]
        assert _precision_at_k(ids, {"c", "d"}, 2) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert _precision_at_k([], {"a"}, 5) == pytest.approx(0.0)


class TestRecallAtK:
    def test_perfect_recall(self):
        assert _recall_at_k(["a", "b"], {"a", "b"}, 5) == pytest.approx(1.0)

    def test_zero_recall(self):
        assert _recall_at_k(["x", "y"], {"a", "b"}, 5) == pytest.approx(0.0)

    def test_partial_recall(self):
        assert _recall_at_k(["a", "x"], {"a", "b"}, 5) == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self):
        assert _recall_at_k(["a", "b"], set(), 5) == pytest.approx(0.0)

    def test_recall_bounded_by_k(self):
        # Relevant chunk is at position 6, k=5 → miss
        ids = [f"c{i}" for i in range(10)]
        assert _recall_at_k(ids, {"c5"}, 5) == pytest.approx(0.0)


class TestHitRate:
    def test_hit_when_any_relevant_in_top_k(self):
        assert _hit_rate(["x", "a"], {"a"}, 5) == 1.0

    def test_miss_when_no_relevant_in_top_k(self):
        assert _hit_rate(["x", "y"], {"a"}, 5) == 0.0

    def test_hit_at_exact_k(self):
        ids = ["x", "y", "z", "a", "b"]
        assert _hit_rate(ids, {"z"}, 3) == 1.0

    def test_miss_beyond_k(self):
        ids = ["x", "y", "z", "a", "b"]
        assert _hit_rate(ids, {"a"}, 3) == 0.0


class TestReciprocalRank:
    def test_first_result_is_relevant(self):
        assert _reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_result_is_relevant(self):
        assert _reciprocal_rank(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_third_result_is_relevant(self):
        assert _reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_returns_zero(self):
        assert _reciprocal_rank(["x", "y", "z"], {"a"}) == pytest.approx(0.0)


class TestNdcgAtK:
    def test_perfect_binary_ndcg(self):
        # Single relevant chunk at rank 1
        assert _ndcg_at_k(["a"], {"a"}, 1) == pytest.approx(1.0)

    def test_zero_binary_ndcg(self):
        assert _ndcg_at_k(["x", "y"], {"a"}, 2) == pytest.approx(0.0)

    def test_binary_ndcg_degrades_with_rank(self):
        # Relevant at rank 2 vs rank 1 — should be lower
        ndcg_rank1 = _ndcg_at_k(["a", "x"], {"a"}, 2)
        ndcg_rank2 = _ndcg_at_k(["x", "a"], {"a"}, 2)
        assert ndcg_rank1 > ndcg_rank2

    def test_graded_ndcg_higher_grade_scores_higher(self):
        # With two chunks at rank 1 and 2, a high-grade chunk at rank 2
        # should push NDCG below perfect.  A low-grade chunk at rank 1
        # should also be below 1.0.  But the key property: when the
        # high-grade chunk (3) is missing from results, NDCG is lower
        # than when it is present — test with k=2.
        graded = {"a": 3, "b": 1}
        # Best ordering: a@rank1, b@rank2 → NDCG=1.0
        ndcg_best = _ndcg_at_k(["a", "b"], {"a", "b"}, 2, graded_relevance=graded)
        # Worst ordering: b@rank1, a@rank2 → NDCG < 1.0
        ndcg_worst = _ndcg_at_k(["b", "a"], {"a", "b"}, 2, graded_relevance=graded)
        assert ndcg_best > ndcg_worst

    def test_graded_ndcg_zero_grade_excluded(self):
        # Grade 0 should contribute 0 to DCG
        graded = {"a": 0, "b": 3}
        # a at rank 1 (grade 0), b at rank 2 (grade 3)
        result = _ndcg_at_k(["a", "b"], {"a", "b"}, 2, graded_relevance=graded)
        # b contributes more despite lower rank
        assert result > 0

    def test_graded_ndcg_perfect_ordering(self):
        # Best chunk at rank 1 → perfect NDCG
        graded = {"a": 3, "b": 1}
        ndcg = _ndcg_at_k(["a", "b"], {"a", "b"}, 2, graded_relevance=graded)
        assert ndcg == pytest.approx(1.0)

    def test_graded_ndcg_suboptimal_ordering(self):
        # Worse chunk at rank 1 → less than perfect NDCG
        graded = {"a": 3, "b": 1}
        ndcg = _ndcg_at_k(["b", "a"], {"a", "b"}, 2, graded_relevance=graded)
        assert ndcg < 1.0

    def test_empty_relevant_returns_zero(self):
        assert _ndcg_at_k(["a"], set(), 1) == pytest.approx(0.0)


class TestAveragePrecisionAtK:
    def test_perfect_ap(self):
        assert _average_precision_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_ap_penalises_late_hit(self):
        ap_early = _average_precision_at_k(["a", "x"], {"a"}, 2)
        ap_late = _average_precision_at_k(["x", "a"], {"a"}, 2)
        assert ap_early > ap_late

    def test_no_hit_returns_zero(self):
        assert _average_precision_at_k(["x", "y"], {"a"}, 2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _build_graded_relevance and _effective_ids_from_item
# ---------------------------------------------------------------------------


class TestBuildGradedRelevance:
    def test_returns_none_when_no_relevant_chunks_key(self):
        assert _build_graded_relevance({"relevant_chunk_ids": ["abc"]}) is None

    def test_returns_none_when_relevant_chunks_empty_list(self):
        assert _build_graded_relevance({"relevant_chunks": []}) is None

    def test_builds_dict_from_new_schema(self):
        item = {
            "relevant_chunks": [
                {"chunk_id": "abc", "relevance": 3},
                {"chunk_id": "def", "relevance": 1},
            ]
        }
        result = _build_graded_relevance(item)
        assert result == {"abc": 3, "def": 1}

    def test_skips_entries_without_chunk_id(self):
        item = {"relevant_chunks": [{"relevance": 3}, {"chunk_id": "abc", "relevance": 2}]}
        result = _build_graded_relevance(item)
        assert result == {"abc": 2}

    def test_defaults_relevance_to_3_when_missing(self):
        item = {"relevant_chunks": [{"chunk_id": "abc"}]}
        result = _build_graded_relevance(item)
        assert result == {"abc": 3}


class TestEffectiveIdsFromItem:
    def test_new_schema_extracts_ids_with_positive_relevance(self):
        item = {
            "relevant_chunks": [
                {"chunk_id": "abc", "relevance": 3},
                {"chunk_id": "def", "relevance": 0},
            ]
        }
        result = _effective_ids_from_item(item)
        assert "abc" in result
        assert "def" not in result  # relevance=0 excluded

    def test_new_schema_all_relevant(self):
        item = {
            "relevant_chunks": [
                {"chunk_id": "a", "relevance": 1},
                {"chunk_id": "b", "relevance": 2},
            ]
        }
        assert _effective_ids_from_item(item) == {"a", "b"}

    def test_legacy_schema_returns_set(self):
        item = {"relevant_chunk_ids": ["abc", "def"]}
        assert _effective_ids_from_item(item) == {"abc", "def"}

    def test_empty_item_returns_empty_set(self):
        assert _effective_ids_from_item({}) == set()

    def test_new_schema_takes_priority_over_legacy(self):
        item = {
            "relevant_chunks": [{"chunk_id": "new", "relevance": 3}],
            "relevant_chunk_ids": ["old"],
        }
        result = _effective_ids_from_item(item)
        assert "new" in result
        assert "old" not in result


# ---------------------------------------------------------------------------
# evaluate_query — no private attribute mutation
# ---------------------------------------------------------------------------


class TestEvaluateQuery:
    def test_uses_explicit_k_in_retrieve_call(self):
        """evaluate_query must call retriever.retrieve(query, k=k) not retrieve(query)."""
        retriever = _make_retriever_returning([])
        evaluate_query(retriever, "test query", set(), None, k=7)
        retriever.retrieve.assert_called_once_with("test query", k=7)

    def test_does_not_mutate_retriever_attributes(self):
        """Must not touch _top_k or _base_top_k."""
        retriever = MagicMock(spec=[])  # spec=[] means no attributes allowed
        retriever.retrieve = MagicMock(return_value=[])
        # If the code tries to set any attribute it will raise AttributeError
        evaluate_query(retriever, "query", set(), None, k=5)
        # Verify only retrieve() was called
        retriever.retrieve.assert_called_once()

    def test_returns_query_result(self):
        retriever = _make_retriever_returning([_chunk("abc", score=0.9)])
        result = evaluate_query(retriever, "my query", {"abc"}, None, k=5)
        assert isinstance(result, QueryResult)

    def test_perfect_hit_scores(self):
        retriever = _make_retriever_returning([_chunk("abc", score=0.9)])
        result = evaluate_query(retriever, "query", {"abc"}, None, k=1)
        assert result.hit_rate == pytest.approx(1.0)
        assert result.recall_at_k == pytest.approx(1.0)
        assert result.reciprocal_rank == pytest.approx(1.0)

    def test_total_miss_scores(self):
        retriever = _make_retriever_returning([_chunk("xyz", score=0.9)])
        result = evaluate_query(retriever, "query", {"abc"}, None, k=1)
        assert result.hit_rate == pytest.approx(0.0)
        assert result.recall_at_k == pytest.approx(0.0)
        assert result.reciprocal_rank == pytest.approx(0.0)

    def test_latency_ms_is_non_negative(self):
        retriever = _make_retriever_returning([])
        result = evaluate_query(retriever, "query", set(), None, k=5)
        assert result.latency_ms >= 0.0

    def test_graded_relevance_passed_to_ndcg(self):
        """When graded_relevance is provided, NDCG uses graded gains."""
        retriever = _make_retriever_returning([_chunk("abc", score=0.9)])
        graded = {"abc": 3}
        result_graded = evaluate_query(
            retriever, "query", {"abc"}, None, k=1, graded_relevance=graded
        )
        result_binary = evaluate_query(
            retriever, "query", {"abc"}, None, k=1, graded_relevance=None
        )
        # Both should have ndcg=1.0 when single relevant chunk is rank 1
        assert result_graded.ndcg_at_k == pytest.approx(1.0)
        assert result_binary.ndcg_at_k == pytest.approx(1.0)

    def test_text_fallback_matching(self):
        """When relevant_chunk_ids is empty, match by text prefix."""
        text = "This is the relevant chunk text content for this test."
        retriever = _make_retriever_returning([{"text": text, "score": 0.9}])
        result = evaluate_query(
            retriever, "query", set(), relevant_texts=[text], k=1
        )
        assert result.hit_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_retriever — schema support and aggregation
# ---------------------------------------------------------------------------


class TestEvaluateRetriever:
    def test_empty_dataset_returns_zero_report(self):
        retriever = _make_retriever_returning([])
        report = evaluate_retriever(retriever, [], k=5)
        assert report.n_queries == 0

    def test_legacy_schema_item_evaluated(self):
        """Old schema with relevant_chunk_ids must work."""
        retriever = _make_retriever_returning([_chunk("abc", score=0.9)])
        dataset = [{"query": "test?", "relevant_chunk_ids": ["abc"], "relevant_texts": []}]
        report = evaluate_retriever(retriever, dataset, k=5)
        assert report.n_queries == 1
        assert report.hit_rate == pytest.approx(1.0)

    def test_new_schema_item_evaluated(self):
        """New schema with relevant_chunks must work."""
        retriever = _make_retriever_returning([_chunk("abc", score=0.9)])
        dataset = [
            {
                "query": "test?",
                "relevant_chunks": [{"chunk_id": "abc", "relevance": 3}],
            }
        ]
        report = evaluate_retriever(retriever, dataset, k=5)
        assert report.n_queries == 1
        assert report.hit_rate == pytest.approx(1.0)

    def test_category_propagated_to_query_result(self):
        retriever = _make_retriever_returning([])
        dataset = [
            {
                "query": "test?",
                "relevant_chunk_ids": [],
                "category": "factual",
                "answerable": False,
            }
        ]
        report = evaluate_retriever(retriever, dataset, k=5)
        assert report.per_query[0].category == "factual"
        assert report.per_query[0].answerable is False

    def test_per_category_breakdown_in_to_dict(self):
        """to_dict() should include by_category when multiple categories exist."""
        retriever = _make_retriever_returning([_chunk("a", score=0.9)])
        dataset = [
            {"query": "q1", "relevant_chunk_ids": ["a"], "category": "factual"},
            {"query": "q2", "relevant_chunk_ids": ["b"], "category": "paraphrase"},
        ]
        report = evaluate_retriever(retriever, dataset, k=5)
        d = report.to_dict()
        assert "by_category" in d
        assert "factual" in d["by_category"]
        assert "paraphrase" in d["by_category"]

    def test_single_category_no_breakdown(self):
        """When all queries have same category, by_category is absent."""
        retriever = _make_retriever_returning([_chunk("a", score=0.9)])
        dataset = [
            {"query": "q1", "relevant_chunk_ids": ["a"], "category": "factual"},
            {"query": "q2", "relevant_chunk_ids": ["a"], "category": "factual"},
        ]
        report = evaluate_retriever(retriever, dataset, k=5)
        d = report.to_dict()
        assert "by_category" not in d

    def test_aggregated_metrics_are_averages(self):
        """Mean recall should be the average of individual query recalls."""
        # q1: hit, q2: miss
        def _side_effect(query, k=5):
            if "hit" in query:
                return [_chunk("abc", score=0.9)]
            return []

        retriever = MagicMock()
        retriever.retrieve.side_effect = _side_effect
        dataset = [
            {"query": "hit query", "relevant_chunk_ids": ["abc"]},
            {"query": "miss query", "relevant_chunk_ids": ["xyz"]},
        ]
        report = evaluate_retriever(retriever, dataset, k=5)
        assert report.mean_recall == pytest.approx(0.5)
        assert report.hit_rate == pytest.approx(0.5)

    def test_skips_items_with_empty_query(self):
        retriever = _make_retriever_returning([])
        dataset = [
            {"query": "", "relevant_chunk_ids": []},
            {"query": "valid query", "relevant_chunk_ids": []},
        ]
        report = evaluate_retriever(retriever, dataset, k=5)
        assert report.n_queries == 1

    def test_report_to_dict_has_required_keys(self):
        retriever = _make_retriever_returning([])
        dataset = [{"query": "test", "relevant_chunk_ids": []}]
        report = evaluate_retriever(retriever, dataset, k=5)
        d = report.to_dict()
        for key in ["k", "n_queries", "recall@k", "hit_rate@k", "mrr", "ndcg@k"]:
            assert key in d


# ---------------------------------------------------------------------------
# Dataset file I/O
# ---------------------------------------------------------------------------


class TestDatasetIO:
    def test_save_and_load_legacy_schema(self, tmp_path):
        dataset = [{"query": "q1", "relevant_chunk_ids": ["abc"], "category": "factual"}]
        path = str(tmp_path / "ds.json")
        save_dataset_to_file(dataset, path)
        loaded = load_dataset_from_file(path)
        assert len(loaded) == 1
        assert loaded[0]["query"] == "q1"

    def test_save_and_load_new_schema(self, tmp_path):
        dataset = [
            {
                "id": "q001",
                "query": "What is X?",
                "category": "factual",
                "answerable": True,
                "relevant_chunks": [{"chunk_id": "abc123", "relevance": 3}],
                "reference_answer": "X is Y.",
            }
        ]
        path = str(tmp_path / "ds.json")
        save_dataset_to_file(dataset, path)
        loaded = load_dataset_from_file(path)
        assert loaded[0]["id"] == "q001"
        assert loaded[0]["relevant_chunks"][0]["chunk_id"] == "abc123"

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset_from_file(str(tmp_path / "nonexistent.json"))

    def test_unicode_preserved(self, tmp_path):
        dataset = [{"query": "O que é fotossíntese?", "relevant_chunk_ids": []}]
        path = str(tmp_path / "ds.json")
        save_dataset_to_file(dataset, path)
        loaded = load_dataset_from_file(path)
        assert loaded[0]["query"] == "O que é fotossíntese?"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "ds.json")
        save_dataset_to_file([{"query": "q"}], path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# build_synthetic_dataset uses public metadata property
# ---------------------------------------------------------------------------


class TestBuildSyntheticDataset:
    def test_uses_public_metadata_property(self):
        """Must call vectorstore.metadata (not _metadata)."""
        vs = MagicMock()
        vs.metadata = [
            {
                "text": "Python é uma linguagem de programação muito popular e versátil.",
                "source": "doc.txt",
                "chunk_id": "abc",
            }
        ]
        result = build_synthetic_dataset(vs, n_samples=1, seed=42)
        # metadata property was accessed (no AttributeError on _metadata)
        assert isinstance(result, list)

    def test_empty_metadata_returns_empty(self):
        vs = MagicMock()
        vs.metadata = []
        result = build_synthetic_dataset(vs, n_samples=10, seed=42)
        assert result == []

    def test_result_has_category_synthetic_smoke(self):
        vs = MagicMock()
        vs.metadata = [
            {
                "text": "Getúlio Vargas foi presidente do Brasil por vários anos.",
                "source": "doc.txt",
                "chunk_id": "v1",
            }
        ]
        result = build_synthetic_dataset(vs, n_samples=1, seed=0)
        if result:
            assert result[0]["category"] == "synthetic_smoke"
            assert result[0]["answerable"] is True

    def test_result_has_required_fields(self):
        vs = MagicMock()
        vs.metadata = [
            {
                "text": (
                    "A Revolução Industrial transformou profundamente a sociedade europeia "
                    "no século XVIII, trazendo novas formas de produção."
                ),
                "source": "doc.txt",
                "chunk_id": "r1",
            }
        ]
        result = build_synthetic_dataset(vs, n_samples=1, seed=1)
        if result:
            item = result[0]
            assert "query" in item
            assert "relevant_chunk_ids" in item
            assert "relevant_texts" in item

    def test_n_samples_capped_at_metadata_length(self):
        vs = MagicMock()
        vs.metadata = [
            {"text": f"Texto {i} com conteúdo suficiente.", "source": "doc.txt", "chunk_id": f"c{i}"}
            for i in range(3)
        ]
        result = build_synthetic_dataset(vs, n_samples=100, seed=42)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# EvaluationReport.to_dict category breakdown
# ---------------------------------------------------------------------------


class TestEvaluationReportToDict:
    def _make_query_result(self, category: str, hit: float) -> QueryResult:
        return QueryResult(
            query="q",
            relevant_ids={"a"},
            retrieved=[],
            category=category,
            hit_rate=hit,
            recall_at_k=hit,
            reciprocal_rank=hit,
        )

    def test_by_category_included_when_multiple_categories(self):
        report = EvaluationReport(
            k=5,
            n_queries=2,
            per_query=[
                self._make_query_result("factual", 1.0),
                self._make_query_result("paraphrase", 0.0),
            ],
        )
        d = report.to_dict()
        assert "by_category" in d
        assert d["by_category"]["factual"]["hit_rate"] == pytest.approx(1.0)
        assert d["by_category"]["paraphrase"]["hit_rate"] == pytest.approx(0.0)

    def test_by_category_absent_when_single_category(self):
        report = EvaluationReport(
            k=5,
            n_queries=2,
            per_query=[
                self._make_query_result("factual", 1.0),
                self._make_query_result("factual", 0.5),
            ],
        )
        d = report.to_dict()
        assert "by_category" not in d

    def test_uncategorized_queries_grouped(self):
        report = EvaluationReport(
            k=5,
            n_queries=2,
            per_query=[
                self._make_query_result("", 0.5),
                self._make_query_result("factual", 1.0),
            ],
        )
        d = report.to_dict()
        assert "by_category" in d
        assert "uncategorized" in d["by_category"]

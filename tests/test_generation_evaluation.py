"""
Tests for rag/generation_evaluation.py — agregação de métricas RAGAS.

Não exercita o pacote `ragas` de verdade (não é uma dependência instalada em
CI, mesmo padrão de llama-cpp-python) — `_score_with_ragas` é substituído por
um stub via monkeypatch, retornando um "DataFrame" falso mínimo (só a fatia
de interface que evaluate_generation() usa: `.columns`, `.iloc[i][col]`,
`df[col].tolist()`), para validar apenas a lógica de agregação/serialização.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.generation_evaluation import (  # noqa: E402
    DEFAULT_METRICS,
    GenerationEvaluationReport,
    evaluate_generation,
    is_eligible_item,
)

# ---------------------------------------------------------------------------
# is_eligible_item
# ---------------------------------------------------------------------------


def test_is_eligible_item_true_when_answerable_with_reference():
    item = {"answerable": True, "reference_answer": "1930."}
    assert is_eligible_item(item) is True


def test_is_eligible_item_false_when_not_answerable():
    item = {"answerable": False, "reference_answer": ""}
    assert is_eligible_item(item) is False


def test_is_eligible_item_false_when_missing_reference_answer():
    item = {"answerable": True}
    assert is_eligible_item(item) is False


def test_is_eligible_item_defaults_answerable_to_true():
    item = {"reference_answer": "algo"}
    assert is_eligible_item(item) is True


# ---------------------------------------------------------------------------
# Fake "DataFrame" — evita depender de ragas/pandas nos testes.
# ---------------------------------------------------------------------------


class _FakeSeries(list):
    def tolist(self):
        return list(self)


class _FakeDataFrame:
    def __init__(self, data: dict):
        self._data = {k: _FakeSeries(v) for k, v in data.items()}
        self.columns = list(data.keys())

    def __getitem__(self, key):
        return self._data[key]

    @property
    def iloc(self):
        data = self._data

        class _ILoc:
            def __getitem__(self, idx):
                return {col: vals[idx] for col, vals in data.items()}

        return _ILoc()


def _make_pipeline(answers):
    """MagicMock de RAGPipeline cujo .chat() retorna respostas na ordem das chamadas."""
    pipeline = MagicMock()
    pipeline.chat.side_effect = answers
    return pipeline


# ---------------------------------------------------------------------------
# evaluate_generation
# ---------------------------------------------------------------------------


def test_evaluate_generation_skips_items_without_context(monkeypatch):
    items = [
        {
            "id": "q1",
            "query": "Pergunta 1",
            "category": "factual",
            "reference_answer": "R1",
        },
        {
            "id": "q2",
            "query": "Pergunta 2",
            "category": "factual",
            "reference_answer": "R2",
        },
    ]
    contexts_by_id = {"q1": [{"text": "contexto 1"}], "q2": []}
    pipeline = _make_pipeline(["Resposta 1"])

    called = {}

    def fake_score(rows, judge_llm, embed_model, metric_names, temp, max_tokens):
        called["rows"] = rows
        return _FakeDataFrame({m: [0.8] for m in metric_names})

    monkeypatch.setattr("rag.generation_evaluation._score_with_ragas", fake_score)

    report, details = evaluate_generation(
        items,
        contexts_by_id,
        pipeline,
        judge_llm=MagicMock(),
        embed_model=MagicMock(),
        model_name="test-model.gguf",
    )

    assert report.n_evaluated == 1
    assert report.n_skipped_no_context == 1
    assert len(details) == 1
    assert details[0]["id"] == "q1"
    assert called["rows"][0]["user_input"] == "Pergunta 1"
    assert called["rows"][0]["retrieved_contexts"] == ["contexto 1"]
    assert called["rows"][0]["reference"] == "R1"
    pipeline.chat.assert_called_once()


def test_evaluate_generation_aggregates_mean_scores(monkeypatch):
    items = [
        {"id": "q1", "query": "P1", "category": "factual", "reference_answer": "R1"},
        {"id": "q2", "query": "P2", "category": "factual", "reference_answer": "R2"},
    ]
    contexts_by_id = {"q1": [{"text": "ctx1"}], "q2": [{"text": "ctx2"}]}
    pipeline = _make_pipeline(["A1", "A2"])

    def fake_score(rows, judge_llm, embed_model, metric_names, temp, max_tokens):
        return _FakeDataFrame(
            {"faithfulness": [1.0, 0.5], "answer_relevancy": [0.9, 0.7]}
        )

    monkeypatch.setattr("rag.generation_evaluation._score_with_ragas", fake_score)

    report, details = evaluate_generation(
        items,
        contexts_by_id,
        pipeline,
        judge_llm=MagicMock(),
        embed_model=MagicMock(),
        metrics=["faithfulness", "answer_relevancy"],
        model_name="test-model.gguf",
    )

    assert report.n_evaluated == 2
    assert report.mean_scores["faithfulness"] == pytest.approx(0.75)
    assert report.mean_scores["answer_relevancy"] == pytest.approx(0.8)
    assert details[0]["scores"]["faithfulness"] == 1.0
    assert details[1]["scores"]["faithfulness"] == 0.5


def test_evaluate_generation_category_breakdown(monkeypatch):
    items = [
        {"id": "q1", "query": "P1", "category": "factual", "reference_answer": "R1"},
        {"id": "q2", "query": "P2", "category": "multi_hop", "reference_answer": "R2"},
        {"id": "q3", "query": "P3", "category": "multi_hop", "reference_answer": "R3"},
    ]
    contexts_by_id = {k: [{"text": f"ctx-{k}"}] for k in ("q1", "q2", "q3")}
    pipeline = _make_pipeline(["A1", "A2", "A3"])

    def fake_score(rows, judge_llm, embed_model, metric_names, temp, max_tokens):
        return _FakeDataFrame({"faithfulness": [1.0, 0.0, 0.4]})

    monkeypatch.setattr("rag.generation_evaluation._score_with_ragas", fake_score)

    report, _ = evaluate_generation(
        items,
        contexts_by_id,
        pipeline,
        judge_llm=MagicMock(),
        embed_model=MagicMock(),
        metrics=["faithfulness"],
        model_name="test-model.gguf",
    )

    by_category = report.to_dict()["by_category"]
    assert by_category["factual"]["n"] == 1
    assert by_category["factual"]["faithfulness"] == 1.0
    assert by_category["multi_hop"]["n"] == 2
    assert by_category["multi_hop"]["faithfulness"] == pytest.approx(0.2)


def test_evaluate_generation_no_eligible_rows_skips_ragas_call(monkeypatch):
    pipeline = _make_pipeline([])

    def fake_score(*args, **kwargs):
        raise AssertionError(
            "não deveria chamar _score_with_ragas sem linhas elegíveis"
        )

    monkeypatch.setattr("rag.generation_evaluation._score_with_ragas", fake_score)

    report, details = evaluate_generation(
        [],
        {},
        pipeline,
        judge_llm=MagicMock(),
        embed_model=MagicMock(),
        model_name="test-model.gguf",
    )

    assert report.n_evaluated == 0
    assert details == []


def test_generation_evaluation_report_to_dict_rounds_scores():
    report = GenerationEvaluationReport(
        model_name="m.gguf",
        n_evaluated=1,
        n_skipped_no_context=0,
        metrics=list(DEFAULT_METRICS),
        mean_scores={"faithfulness": 0.123456},
    )
    d = report.to_dict()
    assert d["metrics"]["faithfulness"] == 0.1235
    assert d["model"] == "m.gguf"
    assert "by_category" not in d

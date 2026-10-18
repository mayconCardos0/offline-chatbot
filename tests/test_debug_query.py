"""
Tests for scripts/debug_query.py — relatório de diagnóstico de uma pergunta.
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.pipeline import RAGPipeline  # noqa: E402
from scripts.debug_query import build_report  # noqa: E402


def _pipeline(max_context_chars):
    return RAGPipeline(
        retriever=MagicMock(),
        llm=None,
        conv_manager=None,
        max_context_chars=max_context_chars,
    )


def _chunk(text, score=0.8, source="docs/livro.pdf"):
    return {"text": text, "score": score, "source": source}


class TestBuildReport:
    def test_no_chunks_reports_default_answer(self):
        report = build_report("Pergunta", [], _pipeline(3500), 3500)
        assert "Nenhum chunk passou nos filtros" in report

    def test_lists_chunks_with_score_and_source(self):
        chunks = [_chunk("A fotossíntese ocorre nos cloroplastos das células." * 3)]
        report = build_report(
            "fotossíntese cloroplastos", chunks, _pipeline(3500), 3500
        )
        assert "[#1] score=0.800" in report
        assert "livro.pdf" in report
        assert "NO PROMPT" in report

    def test_marks_chunk_that_did_not_fit_in_context(self):
        chunks = [
            _chunk("Primeiro trecho sobre cloroplastos. " * 20, score=0.9),
            _chunk("Segundo trecho totalmente diferente sobre Vargas. " * 20, 0.5),
        ]
        report = build_report("cloroplastos", chunks, _pipeline(800), 800)
        assert "[#1] score=0.900  NO PROMPT" in report
        assert "[#2] score=0.500  FORA do prompt" in report

    def test_reports_topical_relevance_failure(self):
        chunks = [_chunk("Texto sobre mitose celular e divisão de células." * 3)]
        report = build_report(
            "República de Saló na guerra", chunks, _pipeline(3500), 3500
        )
        assert "FALHA" in report

    def test_reports_topical_relevance_pass(self):
        chunks = [_chunk("A mitose é a divisão celular que gera células idênticas.")]
        report = build_report("o que é mitose celular", chunks, _pipeline(3500), 3500)
        assert "passa" in report

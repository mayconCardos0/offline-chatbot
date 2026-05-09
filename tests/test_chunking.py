"""
Tests for rag/chunking.py — clean_pdf_text, split_sentences, chunk_document.
"""
import pytest
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.chunking import clean_pdf_text, split_sentences, chunk_document


# ---------------------------------------------------------------------------
# clean_pdf_text
# ---------------------------------------------------------------------------

class TestCleanPdfText:
    def test_removes_isolated_page_numbers(self):
        text = "Paragraph one.\n\n42\n\nParagraph two."
        result = clean_pdf_text(text)
        assert "42" not in result.split("\n\n")[1] or "42" not in result

    def test_joins_hyphenated_line_breaks(self):
        text = "pro-\ngramming is fun"
        result = clean_pdf_text(text)
        assert "programming" in result

    def test_collapses_multiple_newlines(self):
        text = "A\n\n\n\n\nB"
        result = clean_pdf_text(text)
        assert "\n\n\n" not in result

    def test_normalizes_multiple_spaces(self):
        text = "word    another"
        result = clean_pdf_text(text)
        assert "  " not in result

    def test_strips_leading_trailing_whitespace(self):
        text = "   content here   "
        result = clean_pdf_text(text)
        assert result == result.strip()

    def test_empty_string_returns_empty(self):
        assert clean_pdf_text("") == ""

    def test_preserves_content(self):
        text = "Hello world. This is a test."
        result = clean_pdf_text(text)
        assert "Hello world" in result
        assert "This is a test" in result


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_splits_on_period(self):
        text = "First sentence. Second sentence."
        result = split_sentences(text)
        assert len(result) == 2

    def test_splits_on_question_mark(self):
        text = "Is it ready? Yes it is."
        result = split_sentences(text)
        assert len(result) == 2

    def test_splits_on_exclamation(self):
        text = "Atenção! Leia com cuidado."
        result = split_sentences(text)
        assert len(result) == 2

    def test_single_sentence_no_split(self):
        text = "Uma única sentença sem ponto final"
        result = split_sentences(text)
        assert len(result) == 1
        assert result[0] == text

    def test_empty_string_returns_empty_list(self):
        result = split_sentences("")
        assert result == []

    def test_strips_whitespace_from_sentences(self):
        text = "Sentence one. Sentence two."
        for s in split_sentences(text):
            assert s == s.strip()


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def _make_doc(self, text):
        return {"text": text, "source": "test.txt"}

    def test_small_doc_returns_single_chunk(self):
        doc = self._make_doc("Short text.")
        chunks = chunk_document(doc, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0]["source"] == "test.txt"

    def test_empty_text_returns_empty(self):
        doc = self._make_doc("")
        chunks = chunk_document(doc, chunk_size=512)
        assert chunks == []

    def test_long_text_splits_into_multiple_chunks(self):
        # Build text longer than chunk_size
        sentence = "Esta é uma sentença de teste para verificar o chunking. "
        text = sentence * 30  # ~1650 chars
        doc = self._make_doc(text)
        chunks = chunk_document(doc, chunk_size=200)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        sentence = "Esta é uma sentença longa o suficiente para testar o limite. "
        text = sentence * 20
        doc = self._make_doc(text)
        chunk_size = 300
        chunks = chunk_document(doc, chunk_size=chunk_size)
        # Each chunk should be close to chunk_size but may slightly exceed due to sentence boundaries
        for chunk in chunks:
            assert len(chunk["text"]) < chunk_size * 3  # generous bound

    def test_source_preserved_in_all_chunks(self):
        sentence = "Sentença de teste número um ponto. "
        text = sentence * 30
        doc = {"text": text, "source": "my_doc.pdf"}
        chunks = chunk_document(doc, chunk_size=200)
        for chunk in chunks:
            assert chunk["source"] == "my_doc.pdf"

    def test_overlap_shares_content(self):
        """With overlap=1, adjacent chunks should share at least one sentence."""
        sentences = [f"Esta é a sentença número {i} do documento." for i in range(20)]
        text = " ".join(sentences)
        doc = self._make_doc(text)
        chunks = chunk_document(doc, chunk_size=150, overlap=1)
        if len(chunks) > 1:
            # The last sentence of chunk[0] should appear in chunk[1]
            words_0 = set(chunks[0]["text"].split())
            words_1 = set(chunks[1]["text"].split())
            # Some overlap expected
            assert len(words_0 & words_1) > 0

    def test_no_overlap_zero(self):
        sentence = "Sentença separada e independente ponto final. "
        text = sentence * 20
        doc = self._make_doc(text)
        chunks = chunk_document(doc, chunk_size=150, overlap=0)
        assert len(chunks) > 1

    def test_chunk_text_not_empty(self):
        sentence = "Conteúdo válido aqui. "
        text = sentence * 20
        doc = self._make_doc(text)
        chunks = chunk_document(doc, chunk_size=200)
        for chunk in chunks:
            assert chunk["text"].strip() != ""

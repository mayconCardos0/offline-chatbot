"""
Tests for rag/vectorstore.py — assinatura do modelo de embedding gravada no índice.
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("faiss")

from rag.vectorstore import (  # noqa: E402
    _SIGNATURE_FILE,
    EmbeddingMismatchError,
    VectorStore,
)

DIM = 8
SIG_MINILM = "paraphrase-multilingual-MiniLM-L12-v2|query=|passage="
SIG_E5 = "intfloat/multilingual-e5-small|query=query: |passage=passage: "


def _unit(i):
    vec = [0.0] * DIM
    vec[i % DIM] = 1.0
    return vec


def _saved_store(path, signature):
    store = VectorStore(
        index_dir=str(path), embedding_dim=DIM, embed_signature=signature
    )
    store.add(
        [
            {"text": "trecho um", "source": "a.pdf"},
            {"text": "trecho dois", "source": "a.pdf"},
        ],
        [_unit(0), _unit(1)],
    )
    store.save()
    return store


class TestEmbedSignature:
    def test_save_writes_signature_file(self, tmp_path):
        _saved_store(tmp_path, SIG_E5)
        assert (tmp_path / _SIGNATURE_FILE).read_text(encoding="utf-8") == SIG_E5

    def test_same_signature_loads_index(self, tmp_path):
        _saved_store(tmp_path, SIG_E5)
        reloaded = VectorStore(
            index_dir=str(tmp_path), embedding_dim=DIM, embed_signature=SIG_E5
        )
        assert reloaded.size == 2

    def test_different_model_is_rejected_with_clear_message(self, tmp_path):
        _saved_store(tmp_path, SIG_MINILM)
        with pytest.raises(EmbeddingMismatchError) as exc:
            VectorStore(
                index_dir=str(tmp_path), embedding_dim=DIM, embed_signature=SIG_E5
            )
        message = str(exc.value)
        assert SIG_MINILM in message
        assert SIG_E5 in message
        assert "Reindexe" in message

    def test_different_prefix_is_rejected(self, tmp_path):
        _saved_store(tmp_path, SIG_E5)
        with pytest.raises(EmbeddingMismatchError):
            VectorStore(
                index_dir=str(tmp_path),
                embedding_dim=DIM,
                embed_signature="intfloat/multilingual-e5-small|query=|passage=",
            )

    def test_no_signature_given_skips_the_check(self, tmp_path):
        _saved_store(tmp_path, SIG_MINILM)
        reloaded = VectorStore(index_dir=str(tmp_path), embedding_dim=DIM)
        assert reloaded.size == 2

    def test_legacy_index_without_signature_file_still_loads(self, tmp_path, caplog):
        _saved_store(tmp_path, None)
        assert not (tmp_path / _SIGNATURE_FILE).exists()
        with caplog.at_level(logging.WARNING, logger="rag.vectorstore"):
            reloaded = VectorStore(
                index_dir=str(tmp_path), embedding_dim=DIM, embed_signature=SIG_E5
            )
        assert reloaded.size == 2
        assert "não registra o modelo" in caplog.text

    def test_empty_directory_has_nothing_to_check(self, tmp_path):
        store = VectorStore(
            index_dir=str(tmp_path / "novo"), embedding_dim=DIM, embed_signature=SIG_E5
        )
        assert store.size == 0

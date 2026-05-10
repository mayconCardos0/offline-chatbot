"""
Tests for rag/pipeline.py — RAGPipeline chat flow, prompt building, history trimming.
"""

import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.conversation import ConversationManager
from rag.pipeline import (
    _MAX_CONTEXT_CHARS,
    _MAX_HISTORY_TURNS,
    _NO_CONTEXT_RESPONSE,
    RAGPipeline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_conversations():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    yield tmp
    if os.path.exists(tmp):
        os.remove(tmp)


@pytest.fixture
def conv_manager(tmp_conversations):
    return ConversationManager(storage_path=tmp_conversations)


def _make_pipeline(retriever=None, llm=None, conv_manager=None, tmp_conversations=None):
    if conv_manager is None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        conv_manager = ConversationManager(storage_path=tmp)

    mock_retriever = retriever or MagicMock()
    mock_llm = llm or MagicMock()

    if retriever is None:
        mock_retriever.retrieve.return_value = []
    if llm is None:
        mock_llm.chat.return_value = "Default LLM response."

    return (
        RAGPipeline(
            retriever=mock_retriever,
            llm=mock_llm,
            conv_manager=conv_manager,
            max_context_chars=_MAX_CONTEXT_CHARS,
            max_history_turns=_MAX_HISTORY_TURNS,
        ),
        mock_retriever,
        mock_llm,
        conv_manager,
    )


# ---------------------------------------------------------------------------
# chat — basic flow
# ---------------------------------------------------------------------------


class TestPipelineChat:
    def test_returns_string(self):
        pipeline, retriever, llm, _ = _make_pipeline()
        retriever.retrieve.return_value = [
            {"text": "Some relevant content here.", "source": "doc.txt", "score": 0.9}
        ]
        result = pipeline.chat("session-1", "Hello")
        assert isinstance(result, str)

    def test_returns_no_context_response_when_no_chunks(self):
        pipeline, retriever, _, _ = _make_pipeline()
        retriever.retrieve.return_value = []
        result = pipeline.chat("session-x", "Hello")
        assert result == _NO_CONTEXT_RESPONSE

    def test_llm_called_with_chunks(self):
        pipeline, retriever, llm, _ = _make_pipeline()
        retriever.retrieve.return_value = [
            {
                "text": "A fotossíntese ocorre nos cloroplastos das células vegetais.",
                "source": "doc.txt",
                "score": 0.85,
            }
        ]
        llm.chat.return_value = "LLM answer here."
        result = pipeline.chat(
            "session-2", "Como ocorre a fotossíntese nos cloroplastos?"
        )
        assert llm.chat.called
        assert result == "LLM answer here."

    def test_auto_creates_session(self):
        pipeline, retriever, llm, conv_manager = _make_pipeline()
        retriever.retrieve.return_value = []
        pipeline.chat("brand-new-session", "Hi")
        assert conv_manager.get("brand-new-session") is not None

    def test_session_already_exists_no_duplicate(self):
        pipeline, retriever, llm, conv_manager = _make_pipeline()
        retriever.retrieve.return_value = []
        conv_manager.create(session_id="existing")
        pipeline.chat("existing", "Hello again")
        # Only one conversation with this id
        sessions = [c for c in conv_manager.list() if c["id"] == "existing"]
        assert len(sessions) == 1

    def test_messages_persisted_after_chat(self):
        pipeline, retriever, llm, conv_manager = _make_pipeline()
        retriever.retrieve.return_value = []
        pipeline.chat("persist-test", "Test message")
        messages = conv_manager.get("persist-test")["messages"]
        assert len(messages) == 2  # user + assistant
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test message"
        assert messages[1]["role"] == "assistant"

    def test_multiple_turns_accumulate_messages(self):
        pipeline, retriever, llm, conv_manager = _make_pipeline()
        retriever.retrieve.return_value = []
        pipeline.chat("multi-turn", "First message")
        pipeline.chat("multi-turn", "Second message")
        messages = conv_manager.get("multi-turn")["messages"]
        assert len(messages) == 4  # 2 turns × (user + assistant)


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def _pipeline(self):
        return _make_pipeline()[0]

    def test_contains_chunk_text(self):
        pipeline = self._pipeline()
        chunks = [
            {"text": "Conteúdo relevante aqui.", "source": "doc.pdf", "score": 0.8}
        ]
        prompt = pipeline._build_system_prompt(chunks)
        assert "Conteúdo relevante aqui." in prompt

    def test_contains_source_basename(self):
        pipeline = self._pipeline()
        chunks = [{"text": "Texto.", "source": "/path/to/arquivo.pdf", "score": 0.7}]
        prompt = pipeline._build_system_prompt(chunks)
        assert "arquivo.pdf" in prompt

    def test_respects_max_context_chars(self):
        pipeline = self._pipeline()
        long_text = "A" * 5000
        chunks = [{"text": long_text, "source": "big.txt", "score": 0.9}]
        prompt = pipeline._build_system_prompt(chunks)
        # The prompt template itself adds overhead, so we allow some extra
        assert len(prompt) < 5000 + 2000  # generous bound

    def test_multiple_chunks_combined(self):
        pipeline = self._pipeline()
        chunks = [
            {"text": "Primeiro chunk.", "source": "a.txt", "score": 0.9},
            {"text": "Segundo chunk.", "source": "b.txt", "score": 0.8},
        ]
        prompt = pipeline._build_system_prompt(chunks)
        assert "Primeiro chunk." in prompt
        assert "Segundo chunk." in prompt


# ---------------------------------------------------------------------------
# _trim_history
# ---------------------------------------------------------------------------


class TestTrimHistory:
    def _pipeline(self, max_turns=3):
        p, _, _, _ = _make_pipeline()
        p._max_history_turns = max_turns
        return p

    def _turn(self, i):
        return [
            {"role": "user", "content": f"Question {i}", "timestamp": time.time()},
            {"role": "assistant", "content": f"Answer {i}", "timestamp": time.time()},
        ]

    def test_empty_history_returns_empty(self):
        pipeline = self._pipeline()
        assert pipeline._trim_history([]) == []

    def test_short_history_unchanged(self):
        pipeline = self._pipeline(max_turns=5)
        msgs = self._turn(1) + self._turn(2)
        result = pipeline._trim_history(msgs)
        assert len(result) == 4

    def test_long_history_trimmed_to_max_turns(self):
        pipeline = self._pipeline(max_turns=2)
        msgs = self._turn(1) + self._turn(2) + self._turn(3) + self._turn(4)
        result = pipeline._trim_history(msgs)
        assert len(result) == 4  # 2 turns × 2 messages

    def test_keeps_most_recent_messages(self):
        pipeline = self._pipeline(max_turns=1)
        msgs = self._turn(1) + self._turn(2) + self._turn(3)
        result = pipeline._trim_history(msgs)
        assert result[0]["content"] == "Question 3"
        assert result[1]["content"] == "Answer 3"

    def test_filters_out_system_messages(self):
        pipeline = self._pipeline()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = pipeline._trim_history(msgs)
        assert all(m["role"] in ("user", "assistant") for m in result)

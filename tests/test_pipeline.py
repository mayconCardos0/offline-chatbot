"""
Tests for rag/pipeline.py — RAGPipeline chat flow, prompt building, history trimming.
"""

import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.conversation import ConversationManager  # noqa: E402
from rag.pipeline import (  # noqa: E402
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

    def test_strips_model_think_blocks_and_prompt_echoes(self):
        pipeline, retriever, llm, _ = _make_pipeline()
        retriever.retrieve.return_value = [
            {
                "text": "O Iluminismo valorizava a razão.",
                "source": "doc.txt",
                "score": 0.9,
            }
        ]
        llm.chat.return_value = (
            "<think>hidden reasoning</think>\n"
            "O Iluminismo valorizava a razão como guia para o conhecimento."
        )

        result = pipeline.chat("session-clean", "O que foi o Iluminismo?")

        assert "<think>" not in result
        assert result.startswith("O Iluminismo")

    def test_prompt_echo_returns_no_context_response(self):
        pipeline, retriever, llm, _ = _make_pipeline()
        retriever.retrieve.return_value = [
            {
                "text": "Conteúdo relevante suficiente.",
                "source": "doc.txt",
                "score": 0.9,
            }
        ]
        llm.chat.return_value = (
            "Você é um professor particular especializado no Ensino Médio brasileiro."
        )

        result = pipeline.chat("session-echo", "Pergunta qualquer")

        assert result == _NO_CONTEXT_RESPONSE

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
# Título automático da conversa
# ---------------------------------------------------------------------------


class TestTitleGeneration:
    # Par pergunta/chunk com overlap real de keywords (mesmo par usado em
    # test_llm_called_with_chunks) — necessário para passar no filtro de
    # relevância temática do pipeline (_is_topically_relevant) e garantir
    # que o caminho testado é o de resposta normal do LLM, não o canônico
    # de "fora do contexto".
    _QUERY = "Como ocorre a fotossíntese nos cloroplastos?"
    _CHUNK = {
        "text": "A fotossíntese ocorre nos cloroplastos das células vegetais.",
        "source": "doc.txt",
        "score": 0.85,
    }

    def test_first_turn_generates_title(self):
        pipeline, retriever, llm, conv_manager = _make_pipeline()
        retriever.retrieve.return_value = [self._CHUNK]
        llm.chat.side_effect = [
            "Resposta principal do LLM.",
            "Fotossíntese nas plantas",
        ]

        pipeline.chat("title-session", self._QUERY)

        conv = conv_manager.get("title-session")
        assert conv["title"] == "Fotossíntese nas plantas"
        assert conv["title"] != "New Chat"

    def test_second_turn_does_not_regenerate_title(self):
        pipeline, retriever, llm, conv_manager = _make_pipeline()
        retriever.retrieve.return_value = [self._CHUNK]
        llm.chat.side_effect = [
            "Primeira resposta.",
            "Título Gerado",
            "Segunda resposta.",
        ]

        pipeline.chat("multi-turn-title", self._QUERY)
        pipeline.chat("multi-turn-title", self._QUERY)

        assert conv_manager.get("multi-turn-title")["title"] == "Título Gerado"
        assert llm.chat.call_count == 3

    def test_title_generated_even_without_chunks(self):
        pipeline, retriever, llm, conv_manager = _make_pipeline()
        retriever.retrieve.return_value = []
        llm.chat.return_value = "Dúvida Genérica"

        pipeline.chat("no-context-title", "Pergunta qualquer")

        conv = conv_manager.get("no-context-title")
        assert conv["title"] == "Dúvida Genérica"

    def test_title_generation_failure_falls_back_without_breaking_chat(self):
        pipeline, retriever, llm, conv_manager = _make_pipeline()
        retriever.retrieve.return_value = [self._CHUNK]

        def _side_effect(messages, **kwargs):
            if messages[0]["content"].startswith("Gere um título"):
                raise RuntimeError("model error")
            return "Resposta principal."

        llm.chat.side_effect = _side_effect

        result = pipeline.chat("fallback-title", self._QUERY)

        assert result == "Resposta principal."
        conv = conv_manager.get("fallback-title")
        assert conv["title"] != "New Chat"
        assert conv["title"].startswith("Como ocorre")

    def test_title_strips_think_blocks_and_quotes(self):
        pipeline, retriever, llm, conv_manager = _make_pipeline()
        retriever.retrieve.return_value = [
            {
                "text": "A Revolução Francesa foi um marco histórico.",
                "source": "doc.txt",
                "score": 0.9,
            }
        ]
        llm.chat.side_effect = [
            "Resposta principal.",
            '<think>pensando...</think>"Revolução Francesa"',
        ]

        pipeline.chat("clean-title", "O que foi a Revolução Francesa?")

        title = conv_manager.get("clean-title")["title"]
        assert "<think>" not in title
        assert not title.startswith('"')


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


# ---------------------------------------------------------------------------
# _ThinkFilter — remoção incremental de <think> em stream de tokens
# ---------------------------------------------------------------------------


def _filter_tokens(tokens):
    from rag.pipeline import _ThinkFilter

    f = _ThinkFilter()
    out = "".join(f.feed(t) for t in tokens)
    return out + f.flush()


class TestThinkFilter:
    def test_plain_text_passes_through(self):
        assert _filter_tokens(["Olá", ", ", "mundo"]) == "Olá, mundo"

    def test_removes_think_block(self):
        tokens = ["<think>", "raciocínio", "</think>", "Resposta"]
        assert _filter_tokens(tokens).strip() == "Resposta"

    def test_removes_think_block_with_tags_split_across_tokens(self):
        tokens = ["<thi", "nk>pensando", "</th", "ink>\n\nOk", "!"]
        assert _filter_tokens(tokens).strip() == "Ok!"

    def test_unclosed_think_is_discarded(self):
        assert _filter_tokens(["Antes ", "<think>", "nunca fecha"]) == "Antes "

    def test_lone_angle_bracket_is_not_lost(self):
        assert _filter_tokens(["a ", "<", " b"]) == "a < b"

    def test_partial_tag_at_end_is_flushed_as_text(self):
        assert _filter_tokens(["fim ", "<thi"]) == "fim <thi"


# ---------------------------------------------------------------------------
# chat_stream — eventos de streaming
# ---------------------------------------------------------------------------

_STREAM_CHUNK = {
    "text": "A fotossíntese ocorre nos cloroplastos das células vegetais.",
    "source": "doc.txt",
    "score": 0.9,
}
_STREAM_QUERY = "Como ocorre a fotossíntese nos cloroplastos?"


def _stream_llm(tokens, title="Fotossíntese"):
    llm = MagicMock()

    def fake_chat(messages, stream=False, **kwargs):
        return iter(tokens) if stream else title

    llm.chat.side_effect = fake_chat
    return llm


def _stream_pipeline(tokens, chunk=None, title="Fotossíntese"):
    retriever = MagicMock()
    retriever.retrieve.return_value = [chunk or _STREAM_CHUNK]
    return _make_pipeline(retriever=retriever, llm=_stream_llm(tokens, title))


class TestPipelineChatStream:
    def test_emits_tokens_then_done_with_full_response(self):
        pipeline, *_ = _stream_pipeline(["Olá", ", ", "mundo"])
        events = list(pipeline.chat_stream("s1", _STREAM_QUERY))
        tokens = [e["text"] for e in events if e["type"] == "token"]
        done = [e for e in events if e["type"] == "done"]
        assert "".join(tokens) == "Olá, mundo"
        assert len(done) == 1
        assert done[0]["response"] == "Olá, mundo"
        last_token = max(i for i, e in enumerate(events) if e["type"] == "token")
        assert events.index(done[0]) > last_token

    def test_streams_with_llm_stream_flag(self):
        pipeline, _, llm, _ = _stream_pipeline(["x"])
        list(pipeline.chat_stream("s1", _STREAM_QUERY))
        assert llm.chat.call_args_list[0].kwargs["stream"] is True

    def test_persists_turn_in_history(self):
        pipeline, _, _, conv_manager = _stream_pipeline(["Resposta", " final"])
        list(pipeline.chat_stream("s1", _STREAM_QUERY))
        messages = conv_manager.get("s1")["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"] == "Resposta final"

    def test_think_block_hidden_from_tokens_and_response(self):
        tokens = ["<think>", "segredo", "</think>", "\n\n", "Visível"]
        pipeline, *_ = _stream_pipeline(tokens)
        events = list(pipeline.chat_stream("s1", _STREAM_QUERY))
        streamed = "".join(e["text"] for e in events if e["type"] == "token")
        done = next(e for e in events if e["type"] == "done")
        assert streamed == "Visível"
        assert done["response"] == "Visível"

    def test_leading_whitespace_is_not_streamed(self):
        pipeline, *_ = _stream_pipeline(["\n", "  ", "Texto"])
        events = list(pipeline.chat_stream("s1", _STREAM_QUERY))
        first = next(e for e in events if e["type"] == "token")
        assert first["text"] == "Texto"

    def test_first_turn_emits_title_after_done(self):
        pipeline, _, _, conv_manager = _stream_pipeline(["Ok"], title="Meu título")
        events = list(pipeline.chat_stream("s1", _STREAM_QUERY))
        types = [e["type"] for e in events]
        assert types[-2:] == ["done", "title"]
        assert events[-1]["title"] == "Meu título"
        assert conv_manager.get("s1")["title"] == "Meu título"

    def test_second_turn_has_no_title_event(self):
        pipeline, *_ = _stream_pipeline(["Ok"])
        list(pipeline.chat_stream("s1", _STREAM_QUERY))
        events = list(pipeline.chat_stream("s1", _STREAM_QUERY))
        assert "title" not in [e["type"] for e in events]

    def test_no_chunks_returns_canned_response_without_calling_llm(self):
        pipeline, retriever, llm, _ = _make_pipeline()
        retriever.retrieve.return_value = []
        events = list(pipeline.chat_stream("s1", "Pergunta qualquer"))
        done = next(e for e in events if e["type"] == "done")
        assert done["response"] == _NO_CONTEXT_RESPONSE
        # só o título (stream=False) pode chamar o LLM, nunca a resposta
        assert all(c.kwargs.get("stream") is not True for c in llm.chat.call_args_list)

    def test_medium_confidence_prefix_is_streamed_first(self):
        chunk = dict(_STREAM_CHUNK, score=0.5)
        pipeline, *_ = _stream_pipeline(["Resposta"], chunk=chunk)
        events = list(pipeline.chat_stream("s1", _STREAM_QUERY))
        first = next(e for e in events if e["type"] == "token")
        done = next(e for e in events if e["type"] == "done")
        assert first["text"].startswith("Com base no material disponível")
        assert done["response"].endswith("Resposta")

    def test_model_failure_emits_error_and_persists_nothing(self):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("boom")
        retriever = MagicMock()
        retriever.retrieve.return_value = [_STREAM_CHUNK]
        pipeline, _, _, conv_manager = _make_pipeline(retriever=retriever, llm=llm)
        events = list(pipeline.chat_stream("s1", _STREAM_QUERY))
        assert events[-1]["type"] == "error"
        assert not any(e["type"] == "done" for e in events)
        assert conv_manager.get("s1")["messages"] == []

    def test_llm_lock_released_after_stream(self):
        pipeline, *_ = _stream_pipeline(["Ok"])
        list(pipeline.chat_stream("s1", _STREAM_QUERY))
        assert pipeline._llm_lock.acquire(blocking=False)
        pipeline._llm_lock.release()


# ---------------------------------------------------------------------------
# chat_stream — o lock do modelo nunca pode travar a API
# ---------------------------------------------------------------------------


def _wait_lock_free(pipeline, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pipeline._llm_lock.acquire(blocking=False):
            pipeline._llm_lock.release()
            return True
        time.sleep(0.01)
    return False


def _endless_llm():
    llm = MagicMock()

    def endless():
        while True:
            time.sleep(0.01)
            yield "x"

    llm.chat.side_effect = lambda messages, stream=False, **kw: (
        endless() if stream else "título"
    )
    return llm


class TestStreamLockSafety:
    def _pipeline(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [_STREAM_CHUNK]
        return _make_pipeline(retriever=retriever, llm=_endless_llm())

    def test_closing_stream_early_releases_llm_lock(self):
        pipeline, *_ = self._pipeline()
        events = pipeline.chat_stream("s1", _STREAM_QUERY)
        assert next(events)["type"] == "token"
        events.close()
        assert _wait_lock_free(pipeline)

    def test_new_stream_works_after_client_disconnect(self):
        pipeline, _, llm, _ = self._pipeline()
        events = pipeline.chat_stream("s1", _STREAM_QUERY)
        next(events)
        events.close()

        llm.chat.side_effect = _stream_llm(["Ok"]).chat.side_effect
        result = list(pipeline.chat_stream("s2", _STREAM_QUERY))
        assert any(e["type"] == "done" for e in result)

    def test_disconnected_stream_is_not_persisted(self):
        pipeline, _, _, conv_manager = self._pipeline()
        events = pipeline.chat_stream("s1", _STREAM_QUERY)
        next(events)
        events.close()
        assert conv_manager.get("s1")["messages"] == []

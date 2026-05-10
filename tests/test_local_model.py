"""
Tests for llm/local_model.py — LocalModel wrapper (mocked llama_cpp).
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_llm_response(content="Test response"):
    return {"choices": [{"message": {"content": content}}]}


def _make_stream_chunks(tokens):
    return [{"choices": [{"delta": {"content": t}}]} for t in tokens]


@pytest.fixture
def model_file(tmp_path):
    model = tmp_path / "test.gguf"
    model.write_bytes(b"dummy")
    return str(model)


class TestNormalizeMessages:
    def _normalize(self, messages):
        from llm.local_model import LocalModel

        return LocalModel._normalize_messages(messages)

    def test_no_system_message_unchanged(self):
        msgs = [{"role": "user", "content": "Hello"}]
        assert self._normalize(msgs) == msgs

    def test_system_prepended_to_first_user(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
        ]
        result = self._normalize(msgs)
        assert result[0]["role"] == "user"
        assert "You are helpful." in result[0]["content"]
        assert "What is Python?" in result[0]["content"]

    def test_system_only_no_user_returns_empty(self):
        msgs = [{"role": "system", "content": "Just system."}]
        assert self._normalize(msgs) == []

    def test_empty_messages_unchanged(self):
        assert self._normalize([]) == []

    def test_no_system_role_in_result(self):
        msgs = [
            {"role": "system", "content": "Instructions."},
            {"role": "user", "content": "Question"},
        ]
        result = self._normalize(msgs)
        assert all(m["role"] != "system" for m in result)

    def test_multi_turn_only_first_user_gets_system(self):
        msgs = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Turn 2"},
        ]
        result = self._normalize(msgs)
        assert "Sys." in result[0]["content"]
        assert result[2]["content"] == "Turn 2"


class TestLocalModelInit:
    def test_raises_if_model_file_missing(self):
        from llm.local_model import LocalModel

        with pytest.raises(RuntimeError, match="not found"):
            LocalModel(model_path="/nonexistent/model.gguf")


class TestLocalModelChat:
    def _make_model(self, model_file, mock_llama):
        """Build LocalModel with the Llama class pre-patched."""
        # Inject Llama into the module so patching works at import time
        import llm.local_model as lm
        from llm.local_model import LocalModel

        lm.Llama = mock_llama
        model = object.__new__(LocalModel)
        model._llm = mock_llama()
        return model

    def test_chat_returns_string(self, model_file):
        mock_llama_cls = MagicMock()
        mock_instance = mock_llama_cls.return_value
        mock_instance.create_chat_completion.return_value = _make_llm_response("Hello!")

        import llm.local_model as lm

        original = getattr(lm, "Llama", None)
        lm.Llama = mock_llama_cls

        try:
            from llm.local_model import LocalModel

            model = object.__new__(LocalModel)
            model._llm = mock_instance
            result = model.chat([{"role": "user", "content": "Hi"}])
            assert result == "Hello!"
        finally:
            if original is not None:
                lm.Llama = original

    def test_chat_stream_yields_tokens(self, model_file):
        tokens = ["Hello", " ", "world"]
        mock_instance = MagicMock()
        mock_instance.create_chat_completion.return_value = iter(
            _make_stream_chunks(tokens)
        )

        from llm.local_model import LocalModel

        model = object.__new__(LocalModel)
        model._llm = mock_instance
        result = list(model.chat([{"role": "user", "content": "Hi"}], stream=True))
        assert result == tokens

    def test_chat_fallback_on_value_error(self):
        mock_instance = MagicMock()
        mock_instance.create_chat_completion.side_effect = [
            ValueError("system role not supported"),
            _make_llm_response("Fallback response"),
        ]
        from llm.local_model import LocalModel

        model = object.__new__(LocalModel)
        model._llm = mock_instance
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question"},
        ]
        result = model.chat(messages)
        assert result == "Fallback response"
        assert mock_instance.create_chat_completion.call_count == 2

    def test_iter_tokens_skips_empty(self):
        from llm.local_model import LocalModel

        chunks = [
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [{"delta": {}}]},  # no content key
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        result = list(LocalModel._iter_tokens(iter(chunks)))
        assert result == ["hello", " world"]

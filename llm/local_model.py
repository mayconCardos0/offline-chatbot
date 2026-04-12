"""
Local LLM wrapper using llama-cpp-python.
Loads a GGUF model and provides a chat completion interface.
"""
import logging
import os
from typing import Iterator

logger = logging.getLogger(__name__)


class LocalModel:
    """Thin wrapper around llama_cpp.Llama for chat completion."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
    ) -> None:
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model file not found at configured path: {model_path}"
            )

        logger.info("Loading GGUF model from %s", model_path)
        from llama_cpp import Llama  # imported here so missing dep gives a clear error

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info("Model loaded successfully.")

    @staticmethod
    def _normalize_messages(messages: list[dict]) -> list[dict]:
        """
        Some models (e.g. Gemma) don't support a 'system' role.
        If the first message is a system prompt, prepend its content to the
        first user message instead so the context is never lost.
        """
        if not messages or messages[0]["role"] != "system":
            return messages

        system_content = messages[0]["content"]
        rest = list(messages[1:])

        # Find the first user message to inject the system content into
        for i, msg in enumerate(rest):
            if msg["role"] == "user":
                rest[i] = {
                    "role": "user",
                    "content": f"{system_content}\n\n{msg['content']}",
                }
                return rest

        # No user message found — just drop the system turn
        return rest

    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> "str | Iterator[str]":
        """
        Run a chat completion.

        Args:
            messages: List of {role, content} dicts (OpenAI-style).
            stream:   If True, yield tokens incrementally; otherwise return full string.

        Returns:
            Full response string when stream=False, or a token iterator when stream=True.
        """
        # Try with the messages as-is first; fall back to folding the system
        # prompt into the first user message for models that reject 'system'.
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                stream=stream,
            )
        except ValueError:
            logger.debug("Model rejected system role — retrying with folded system prompt.")
            normalized = self._normalize_messages(messages)
            response = self._llm.create_chat_completion(
                messages=normalized,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                stream=stream,
            )

        if stream:
            return self._iter_tokens(response)

        return response["choices"][0]["message"]["content"]

    @staticmethod
    def _iter_tokens(response) -> Iterator[str]:
        """Yield text tokens from a streaming llama-cpp response."""
        for chunk in response:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                token = delta.get("content")
                if token:
                    yield token

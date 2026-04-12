"""
RAGPipeline: orchestrates retrieval → prompt construction → LLM call → session update.
"""
import logging
import os
import time
from typing import TYPE_CHECKING

from core.conversation import ConversationManager

if TYPE_CHECKING:
    from llm.local_model import LocalModel
    from rag.retriever import Retriever

logger = logging.getLogger(__name__)

_SYSTEM_WITH_CONTEXT = (
    "You are a helpful assistant. Use the context below to answer the user's question. "
    "If the context is insufficient, use your own knowledge.\n\n"
    "Context:\n{context}"
)

_SYSTEM_NO_CONTEXT = (
    "You are a helpful assistant. Answer the user's question using your own knowledge."
)


class RAGPipeline:
    """Combines retrieval and LLM inference into a single chat interface."""

    def __init__(
        self,
        retriever: "Retriever",
        llm: "LocalModel",
        conv_manager: ConversationManager,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._conv_manager = conv_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, session_id: str, message: str) -> str:
        """Process one user turn and return the assistant response.

        Creates the session automatically if it does not exist (Requirement 9.3).

        Args:
            session_id: Unique identifier for the conversation session.
            message:    The user's message text.

        Returns:
            The assistant's response as a plain string.
        """
        # Ensure session exists (auto-create per Requirement 9.3)
        conv = self._conv_manager.get(session_id)
        if conv is None:
            conv = self._conv_manager.create(session_id=session_id)
            logger.debug("Auto-created session '%s'", session_id)

        # Retrieve relevant chunks (Requirement 8.1)
        chunks = self._retriever.retrieve(message)

        # Build system prompt (Requirements 8.2, 8.3)
        system_content = self._build_system_prompt(chunks)

        # Assemble messages: system + history + new user turn (Requirement 8.4)
        messages = [{"role": "system", "content": system_content}]
        messages.extend(
            {"role": m["role"], "content": m["content"]}
            for m in conv["messages"]
        )
        messages.append({"role": "user", "content": message})

        # Call LLM
        response_text = self._llm.chat(messages, stream=False)

        # Persist updated history
        updated_messages = list(conv["messages"])
        updated_messages.append({"role": "user", "content": message, "timestamp": time.time()})
        updated_messages.append({"role": "assistant", "content": response_text, "timestamp": time.time()})
        self._conv_manager.update(session_id, updated_messages)

        return response_text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, chunks: list[dict]) -> str:
        """Build the system prompt, injecting retrieved context if available."""
        if not chunks:
            # Requirement 8.3: fall back to no-context prompt
            return _SYSTEM_NO_CONTEXT

        # Requirement 8.2: format context with [source: filename] attribution
        context_lines = []
        for chunk in chunks:
            source = os.path.basename(chunk.get("source", "unknown"))
            context_lines.append(f"[source: {source}]\n{chunk['text']}")

        context_block = "\n\n".join(context_lines)
        return _SYSTEM_WITH_CONTEXT.format(context=context_block)

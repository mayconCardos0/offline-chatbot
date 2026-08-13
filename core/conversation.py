"""
ConversationManager: CRUD for chat sessions, persisted to a JSON file.
Ported from the root-level chat_ai.py ConversationManager.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationManager:
    """Persists conversations to a JSON file and provides CRUD operations."""

    def __init__(self, storage_path: str) -> None:
        self.storage_path = storage_path
        self.conversations: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {c["id"]: c for c in data}
            except Exception as e:
                logger.error(
                    "Failed to load conversations from %s: %s", self.storage_path, e
                )
        return {}

    def _save(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(list(self.conversations.values()), f, indent=2)
        except Exception as e:
            logger.error("Failed to save conversations to %s: %s", self.storage_path, e)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, session_id: Optional[str] = None, title: str = "New Chat") -> dict:
        """Create a new conversation and persist it."""
        conv_id = session_id or str(uuid.uuid4())
        conv = {
            "id": conv_id,
            "title": title,
            "messages": [],
            "updated_at": time.time(),
        }
        self.conversations[conv_id] = conv
        self._save()
        return conv

    def get(self, session_id: str) -> Optional[dict]:
        """Return the conversation dict or None if not found."""
        return self.conversations.get(session_id)

    def update(self, session_id: str, messages: List[dict]) -> None:
        """Replace the message list for a session and persist."""
        if session_id in self.conversations:
            self.conversations[session_id]["messages"] = messages
            self.conversations[session_id]["updated_at"] = time.time()
            self._save()

    def set_title(self, session_id: str, title: str) -> None:
        """Rename a session's title and persist. No-op if it doesn't exist."""
        if session_id in self.conversations:
            self.conversations[session_id]["title"] = title
            self._save()

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed, False otherwise."""
        if session_id in self.conversations:
            del self.conversations[session_id]
            self._save()
            return True
        return False

    def list(self) -> List[dict]:
        """Return all conversations sorted by most-recently updated."""
        return sorted(
            self.conversations.values(),
            key=lambda c: c["updated_at"],
            reverse=True,
        )

"""
Tests for core/conversation.py — ConversationManager CRUD and persistence.
"""
import json
import os
import sys
import tempfile
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.conversation import ConversationManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_file():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    yield tmp
    if os.path.exists(tmp):
        os.remove(tmp)


@pytest.fixture
def manager(tmp_file):
    return ConversationManager(storage_path=tmp_file)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_starts_empty_with_new_file(self, tmp_file):
        # Remove the file so it starts fresh
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        manager = ConversationManager(storage_path=tmp_file)
        assert manager.list() == []

    def test_loads_existing_conversations(self, tmp_file):
        existing = [{"id": "abc", "title": "Test", "messages": [], "updated_at": time.time()}]
        with open(tmp_file, "w") as f:
            json.dump(existing, f)
        manager = ConversationManager(storage_path=tmp_file)
        assert manager.get("abc") is not None

    def test_handles_corrupted_file_gracefully(self, tmp_file):
        with open(tmp_file, "w") as f:
            f.write("not valid json {{{")
        # Should not raise
        manager = ConversationManager(storage_path=tmp_file)
        assert manager.list() == []


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

class TestCreate:
    def test_create_returns_dict_with_id(self, manager):
        conv = manager.create()
        assert "id" in conv
        assert isinstance(conv["id"], str)

    def test_create_with_custom_session_id(self, manager):
        conv = manager.create(session_id="my-session")
        assert conv["id"] == "my-session"

    def test_create_with_title(self, manager):
        conv = manager.create(title="My Chat")
        assert conv["title"] == "My Chat"

    def test_create_default_title(self, manager):
        conv = manager.create()
        assert conv["title"] == "New Chat"

    def test_create_has_empty_messages(self, manager):
        conv = manager.create()
        assert conv["messages"] == []

    def test_create_has_updated_at(self, manager):
        before = time.time()
        conv = manager.create()
        after = time.time()
        assert before <= conv["updated_at"] <= after

    def test_create_persists_to_disk(self, tmp_file):
        manager = ConversationManager(storage_path=tmp_file)
        manager.create(session_id="persist-test")
        # Re-load from disk
        manager2 = ConversationManager(storage_path=tmp_file)
        assert manager2.get("persist-test") is not None

    def test_create_multiple_unique_ids(self, manager):
        ids = {manager.create()["id"] for _ in range(5)}
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_existing_session(self, manager):
        conv = manager.create(session_id="get-test")
        assert manager.get("get-test") == conv

    def test_get_nonexistent_returns_none(self, manager):
        assert manager.get("does-not-exist") is None


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_replaces_messages(self, manager):
        manager.create(session_id="upd")
        messages = [{"role": "user", "content": "hello"}]
        manager.update("upd", messages)
        assert manager.get("upd")["messages"] == messages

    def test_update_nonexistent_session_is_noop(self, manager):
        # Should not raise
        manager.update("ghost", [{"role": "user", "content": "x"}])

    def test_update_changes_updated_at(self, manager):
        manager.create(session_id="ts-test")
        old_ts = manager.get("ts-test")["updated_at"]
        time.sleep(0.01)
        manager.update("ts-test", [])
        new_ts = manager.get("ts-test")["updated_at"]
        assert new_ts >= old_ts

    def test_update_persists_to_disk(self, tmp_file):
        manager = ConversationManager(storage_path=tmp_file)
        manager.create(session_id="upd-persist")
        manager.update("upd-persist", [{"role": "user", "content": "saved"}])

        manager2 = ConversationManager(storage_path=tmp_file)
        assert manager2.get("upd-persist")["messages"][0]["content"] == "saved"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_existing_returns_true(self, manager):
        manager.create(session_id="del-test")
        assert manager.delete("del-test") is True

    def test_delete_removes_session(self, manager):
        manager.create(session_id="gone")
        manager.delete("gone")
        assert manager.get("gone") is None

    def test_delete_nonexistent_returns_false(self, manager):
        assert manager.delete("never-existed") is False

    def test_delete_persists_to_disk(self, tmp_file):
        manager = ConversationManager(storage_path=tmp_file)
        manager.create(session_id="del-persist")
        manager.delete("del-persist")

        manager2 = ConversationManager(storage_path=tmp_file)
        assert manager2.get("del-persist") is None


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

class TestList:
    def test_list_empty(self, manager):
        assert manager.list() == []

    def test_list_returns_all(self, manager):
        manager.create(session_id="a")
        manager.create(session_id="b")
        manager.create(session_id="c")
        assert len(manager.list()) == 3

    def test_list_sorted_by_updated_at_desc(self, manager):
        manager.create(session_id="first")
        time.sleep(0.01)
        manager.create(session_id="second")
        time.sleep(0.01)
        manager.create(session_id="third")

        listing = manager.list()
        timestamps = [c["updated_at"] for c in listing]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_reflects_updates(self, manager):
        manager.create(session_id="x")
        manager.create(session_id="y")
        time.sleep(0.01)
        # Update x so it becomes the most-recent
        manager.update("x", [{"role": "user", "content": "new"}])
        listing = manager.list()
        assert listing[0]["id"] == "x"

"""
Tests for api/routes.py — FastAPI endpoints /chat, /health, DELETE /chat/{id}.
Uses a lightweight test app with mocked pipeline and conversation manager.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api.routes import router  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_test_app(pipeline=None, conv_manager=None):
    """Create a minimal FastAPI app with mocked state."""
    app = FastAPI()
    app.include_router(router)

    mock_pipeline = pipeline or MagicMock()
    mock_conv_manager = conv_manager or MagicMock()

    @app.on_event("startup")
    async def set_state():
        app.state.pipeline = mock_pipeline
        app.state.conv_manager = mock_conv_manager

    return app, mock_pipeline, mock_conv_manager


@pytest.fixture
def client_and_mocks():
    app, pipeline, conv_manager = _build_test_app()
    pipeline.chat.return_value = "Mocked response from pipeline."
    conv_manager.delete.return_value = True
    with TestClient(app) as client:
        yield client, pipeline, conv_manager


@pytest.fixture
def client(client_and_mocks):
    return client_and_mocks[0]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client):
        resp = client.get("/health")
        assert resp.json() == {"status": "ok"}

    def test_content_type_json(self, client):
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    def test_returns_200(self, client):
        payload = {"session_id": "test-session", "message": "Hello"}
        resp = client.post("/chat", json=payload)
        assert resp.status_code == 200

    def test_response_contains_session_id(self, client):
        payload = {"session_id": "abc123", "message": "Hi"}
        resp = client.post("/chat", json=payload)
        assert resp.json()["session_id"] == "abc123"

    def test_response_contains_response_field(self, client):
        payload = {"session_id": "abc123", "message": "Hi"}
        resp = client.post("/chat", json=payload)
        assert "response" in resp.json()

    def test_pipeline_called_with_correct_args(self, client_and_mocks):
        client, pipeline, _ = client_and_mocks
        payload = {"session_id": "sess-1", "message": "What is Python?"}
        client.post("/chat", json=payload)
        pipeline.chat.assert_called_once_with("sess-1", "What is Python?")

    def test_response_text_matches_pipeline_output(self, client_and_mocks):
        client, pipeline, _ = client_and_mocks
        pipeline.chat.return_value = "Python é uma linguagem de programação."
        payload = {"session_id": "sess-2", "message": "What is Python?"}
        resp = client.post("/chat", json=payload)
        assert resp.json()["response"] == "Python é uma linguagem de programação."

    def test_missing_message_returns_422(self, client):
        resp = client.post("/chat", json={"session_id": "abc"})
        assert resp.status_code == 422

    def test_missing_session_id_returns_422(self, client):
        resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_multiple_sessions_are_independent(self, client_and_mocks):
        client, pipeline, _ = client_and_mocks
        pipeline.chat.side_effect = lambda sid, msg: f"Response for {sid}"

        resp1 = client.post("/chat", json={"session_id": "s1", "message": "Hi"})
        resp2 = client.post("/chat", json={"session_id": "s2", "message": "Hi"})

        assert resp1.json()["session_id"] == "s1"
        assert resp2.json()["session_id"] == "s2"


# ---------------------------------------------------------------------------
# PATCH /conversations/{session_id}
# ---------------------------------------------------------------------------


class TestRenameConversationEndpoint:
    def _existing_conversation(self):
        return {
            "id": "s1",
            "title": "New Chat",
            "messages": [],
            "updated_at": 123.0,
        }

    def test_returns_200_when_session_exists(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.get.return_value = self._existing_conversation()
        resp = client.patch("/conversations/s1", json={"title": "Novo Título"})
        assert resp.status_code == 200

    def test_returns_updated_title(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.get.return_value = self._existing_conversation()
        resp = client.patch("/conversations/s1", json={"title": "Novo Título"})
        assert resp.json()["title"] == "Novo Título"

    def test_strips_whitespace_from_title(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.get.return_value = self._existing_conversation()
        resp = client.patch(
            "/conversations/s1", json={"title": "  Título com espaços  "}
        )
        assert resp.json()["title"] == "Título com espaços"

    def test_conv_manager_set_title_called_with_correct_args(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.get.return_value = self._existing_conversation()
        client.patch("/conversations/s1", json={"title": "Novo Título"})
        conv_manager.set_title.assert_called_once_with("s1", "Novo Título")

    def test_returns_404_when_session_not_found(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.get.return_value = None
        resp = client.patch("/conversations/ghost", json={"title": "Qualquer"})
        assert resp.status_code == 404

    def test_returns_400_when_title_empty(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.get.return_value = self._existing_conversation()
        resp = client.patch("/conversations/s1", json={"title": "   "})
        assert resp.status_code == 400

    def test_missing_title_returns_422(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.get.return_value = self._existing_conversation()
        resp = client.patch("/conversations/s1", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /chat/{session_id}
# ---------------------------------------------------------------------------


class TestDeleteSessionEndpoint:
    def test_returns_200_when_session_exists(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.delete.return_value = True
        resp = client.delete("/chat/existing-session")
        assert resp.status_code == 200

    def test_returns_deleted_status(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.delete.return_value = True
        resp = client.delete("/chat/existing-session")
        assert resp.json() == {"status": "deleted"}

    def test_returns_404_when_session_not_found(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.delete.return_value = False
        resp = client.delete("/chat/ghost-session")
        assert resp.status_code == 404

    def test_404_detail_message(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.delete.return_value = False
        resp = client.delete("/chat/ghost-session")
        assert "session not found" in resp.json()["detail"].lower()

    def test_conv_manager_called_with_correct_id(self, client_and_mocks):
        client, _, conv_manager = client_and_mocks
        conv_manager.delete.return_value = True
        client.delete("/chat/my-unique-id")
        conv_manager.delete.assert_called_once_with("my-unique-id")

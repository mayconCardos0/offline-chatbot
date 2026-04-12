"""
API routes for the Offline Chatbot.

Endpoints:
  POST   /chat                 — send a message, get a response (auto-creates session)
  DELETE /chat/{session_id}    — delete a session (404 if not found)
  GET    /health               — liveness check
"""
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

if TYPE_CHECKING:
    from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pipeline(request: Request) -> "RAGPipeline":
    return request.app.state.pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Run the RAG pipeline for one user turn.

    Auto-creates the session if it does not exist (Requirement 9.3).
    """
    pipeline = _get_pipeline(request)
    response_text = pipeline.chat(body.session_id, body.message)
    return ChatResponse(session_id=body.session_id, response=response_text)


@router.delete("/chat/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    """Delete a conversation session.

    Returns 404 if the session does not exist (Requirement 9.2).
    """
    conv_manager = request.app.state.conv_manager
    deleted = conv_manager.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session not found")
    return {"status": "deleted"}


@router.get("/health")
async def health() -> dict:
    """Liveness check (Requirement 9.4)."""
    return {"status": "ok"}

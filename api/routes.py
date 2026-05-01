"""
API routes for the Offline Chatbot.

Endpoints:
  GET    /conversations                     — list all conversations
  GET    /conversations/{session_id}        — get a specific conversation
  POST   /chat                              — send a message, get a response (auto-creates session)
  DELETE /chat/{session_id}                 — delete a session (404 if not found)
  GET    /health                            — liveness check
"""
import logging
from typing import TYPE_CHECKING, List

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


class ConversationPreview(BaseModel):
    id: str
    title: str
    updated_at: float


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: List[dict]
    updated_at: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pipeline(request: Request) -> "RAGPipeline":
    return request.app.state.pipeline


def _get_conv_manager(request: Request):
    return request.app.state.conv_manager


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/conversations", response_model=List[ConversationPreview])
async def list_conversations(request: Request) -> List[ConversationPreview]:
    """List all conversations, sorted by most recent first."""
    conv_manager = _get_conv_manager(request)
    conversations = conv_manager.list()
    return [
        ConversationPreview(
            id=c["id"],
            title=c["title"],
            updated_at=c["updated_at"],
        )
        for c in conversations
    ]


@router.get("/conversations/{session_id}", response_model=ConversationDetail)
async def get_conversation(session_id: str, request: Request) -> ConversationDetail:
    """Get a specific conversation by ID."""
    conv_manager = _get_conv_manager(request)
    conversation = conv_manager.get(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationDetail(
        id=conversation["id"],
        title=conversation["title"],
        messages=conversation["messages"],
        updated_at=conversation["updated_at"],
    )

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

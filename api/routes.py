"""
API routes for the Offline Chatbot.

Endpoints:
  GET    /conversations                     — list all conversations
  GET    /conversations/{session_id}        — get a specific conversation
  PATCH  /conversations/{session_id}        — rename a conversation (404 if not found)
  POST   /chat                              — send a message, get a response (auto-creates session)
  POST   /chat/stream                       — same, streamed as NDJSON events (token/done/title/error)
  DELETE /chat/{session_id}                 — delete a session (404 if not found)
  GET    /battery                           — UPS HAT battery status (available=false without the hardware)
  GET    /health                            — liveness check
"""

import json
import logging
from typing import TYPE_CHECKING, Iterator, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
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


class RenameRequest(BaseModel):
    title: str


class BatteryResponse(BaseModel):
    available: bool
    percent: Optional[float] = None
    voltage: Optional[float] = None
    ac_power: Optional[bool] = None
    charging: Optional[bool] = None


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


@router.patch("/conversations/{session_id}", response_model=ConversationPreview)
async def rename_conversation(
    session_id: str, body: RenameRequest, request: Request
) -> ConversationPreview:
    """Rename a conversation's title (manual override of the auto-generated one)."""
    conv_manager = _get_conv_manager(request)
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title cannot be empty")

    conversation = conv_manager.get(session_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="conversation not found")

    conv_manager.set_title(session_id, title)
    return ConversationPreview(
        id=conversation["id"], title=title, updated_at=conversation["updated_at"]
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Run the RAG pipeline for one user turn.

    Auto-creates the session if it does not exist (Requirement 9.3).
    """
    pipeline = _get_pipeline(request)
    response_text = pipeline.chat(body.session_id, body.message)
    return ChatResponse(session_id=body.session_id, response=response_text)


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    """Run the RAG pipeline for one user turn, streaming events as NDJSON.

    One JSON object per line: {"type": "token", "text"}, then {"type": "done",
    "response"} with the final cleaned text, then (first turn only) {"type":
    "title", "title"}. On model failure a single {"type": "error"} is sent.
    """
    pipeline = _get_pipeline(request)

    def event_lines() -> Iterator[str]:
        for event in pipeline.chat_stream(body.session_id, body.message):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


@router.get("/battery", response_model=BatteryResponse)
def battery(request: Request) -> BatteryResponse:
    """Battery status of the UPS HAT.

    Sync on purpose: the read blocks on I2C / a subprocess, so FastAPI runs it
    in a worker thread. Without the hardware it answers available=false.
    """
    monitor = getattr(request.app.state, "battery", None)
    status = monitor.read() if monitor is not None else None
    if status is None:
        return BatteryResponse(available=False)
    return BatteryResponse(
        available=True,
        percent=status.percent,
        voltage=status.voltage,
        ac_power=status.ac_power,
        charging=status.charging,
    )


@router.get("/health")
async def health() -> dict:
    """Liveness check (Requirement 9.4)."""
    return {"status": "ok"}

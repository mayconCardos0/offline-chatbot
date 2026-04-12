"""
FastAPI application entry point for the Offline Chatbot.

Startup (lifespan) loads all heavy components once:
  LocalModel → EmbeddingModel → VectorStore → Retriever → RAGPipeline → ConversationManager

A global exception handler converts unhandled errors to HTTP 500 responses.
"""
import logging
import os
import sys
from contextlib import asynccontextmanager

# Ensure both the project root and the api/ directory are on sys.path
# so absolute imports work regardless of how uvicorn loads this module.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in (_root, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings, setup_logging
from core.conversation import ConversationManager
from llm.local_model import LocalModel
from rag.embeddings import EmbeddingModel
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.vectorstore import VectorStore
from routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all components at startup; clean up on shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Starting Offline Chatbot API…")

    # Resolve paths relative to this file's parent (offline-chatbot/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))

    # --- LLM (Requirement 7.1, 1.4) ---
    llm = LocalModel(
        model_path=resolve(settings.model_path),
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        n_gpu_layers=settings.n_gpu_layers,
    )

    # --- Embeddings (Requirement 4.1) ---
    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=resolve(settings.embed_cache_dir),
    )

    # Determine embedding dimension from a test encode
    sample_dim = len(embed_model.embed(["test"])[0])

    # --- Vector Store (Requirement 5.1, 5.2) ---
    vectorstore = VectorStore(
        index_dir=resolve(settings.index_dir),
        embedding_dim=sample_dim,
    )

    # --- Retriever (Requirement 6.1) ---
    retriever = Retriever(
        vectorstore=vectorstore,
        embed_model=embed_model,
        top_k=settings.top_k,
    )

    # --- Conversation Manager (Requirement 8.4) ---
    conv_file = resolve(settings.conversations_file)
    os.makedirs(os.path.dirname(conv_file), exist_ok=True)
    conv_manager = ConversationManager(storage_path=conv_file)

    # --- RAG Pipeline (Requirement 8.1) ---
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        conv_manager=conv_manager,
    )

    # Attach to app state so routes can access them
    app.state.pipeline = pipeline
    app.state.conv_manager = conv_manager

    logger.info("All components loaded. API is ready.")
    yield

    logger.info("Shutting down Offline Chatbot API.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Offline Chatbot API",
        description="Fully offline RAG-enabled chatbot backed by a local GGUF model.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins by default; tighten in production via env
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler (Requirement 9.5)
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    app.include_router(router)
    return app


app = create_app()

"""
FastAPI application entry point para o Offline Chatbot.

Startup (lifespan) carrega todos os componentes pesados uma vez:
  LocalModel → EmbeddingModel → VectorStore → Retriever → RAGPipeline → ConversationManager
"""
import logging
import os
import sys
from contextlib import asynccontextmanager

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
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Iniciando Offline Chatbot API…")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))

    # --- LLM ---
    llm = LocalModel(
        model_path=resolve(settings.model_path),
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        n_gpu_layers=settings.n_gpu_layers,
    )

    # --- Embeddings ---
    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=resolve(settings.embed_cache_dir),
        batch_size=settings.embed_batch_size,
        use_disk_cache=settings.embed_disk_cache,
    )

    # --- Vector Store (usa dimensão real do modelo) ---
    vectorstore = VectorStore(
        index_dir=resolve(settings.index_dir),
        embedding_dim=embed_model.dimension,
    )

    # --- Retriever ---
    retriever = Retriever(
        vectorstore=vectorstore,
        embed_model=embed_model,
        top_k=settings.top_k,
        candidate_multiplier=settings.candidate_multiplier,
        min_score=settings.min_score,
        lexical_weight=settings.lexical_weight,
    )

    # --- Conversation Manager ---
    conv_file = resolve(settings.conversations_file)
    os.makedirs(os.path.dirname(conv_file), exist_ok=True)
    conv_manager = ConversationManager(storage_path=conv_file)

    # --- RAG Pipeline ---
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        conv_manager=conv_manager,
        max_context_chars=settings.max_context_chars,
        max_history_turns=settings.max_history_turns,
    )

    app.state.pipeline    = pipeline
    app.state.conv_manager = conv_manager

    logger.info("Todos os componentes carregados. API pronta.")
    yield
    logger.info("Encerrando Offline Chatbot API.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Offline Chatbot API",
        description="RAG chatbot offline com modelo GGUF local.",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Exceção não tratada em %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    app.include_router(router)
    return app


app = create_app()

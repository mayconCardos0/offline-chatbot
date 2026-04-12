"""
Centralized configuration for the Offline Chatbot.
All values are read from environment variables (and an optional .env file).
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass


@dataclass
class Settings:
    # --- Server ---
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8000")))

    # --- LLM ---
    model_path: str = field(
        default_factory=lambda: os.environ.get("MODEL_PATH", "models/Qwen3-0.6B-Q8_0.gguf")
    )
    n_ctx: int = field(default_factory=lambda: int(os.environ.get("N_CTX", "4096")))
    n_threads: int = field(default_factory=lambda: int(os.environ.get("N_THREADS", "4")))
    n_gpu_layers: int = field(default_factory=lambda: int(os.environ.get("N_GPU_LAYERS", "0")))

    # --- Embeddings ---
    embed_model_name: str = field(
        default_factory=lambda: os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    )
    embed_cache_dir: str = field(
        default_factory=lambda: os.environ.get("EMBED_CACHE_DIR", "models")
    )

    # --- RAG ---
    docs_dir: str = field(
        default_factory=lambda: os.environ.get("DOCS_DIR", "data/documents")
    )
    index_dir: str = field(
        default_factory=lambda: os.environ.get("INDEX_DIR", "data/index")
    )
    top_k: int = field(default_factory=lambda: int(os.environ.get("TOP_K", "4")))

    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "100")))

    # --- Conversations ---
    conversations_file: str = field(
        default_factory=lambda: os.environ.get("CONVERSATIONS_FILE", "data/conversations.json")
    )

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))


def get_settings() -> Settings:
    """Return a Settings instance populated from the current environment."""
    return Settings()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger. Call once at app startup."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

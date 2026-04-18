"""
Configuração centralizada do Offline Chatbot.
Todos os valores são lidos de variáveis de ambiente (e arquivo .env opcional).
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
    # --- Servidor ---
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8000")))

    # --- LLM ---
    model_path: str = field(
        default_factory=lambda: os.environ.get("MODEL_PATH", "models/Qwen3-0.6B-Q8_0.gguf")
    )
    n_ctx: int      = field(default_factory=lambda: int(os.environ.get("N_CTX", "4096")))
    n_threads: int  = field(default_factory=lambda: int(os.environ.get("N_THREADS", "4")))
    n_gpu_layers: int = field(default_factory=lambda: int(os.environ.get("N_GPU_LAYERS", "0")))

    # --- Embeddings ---
    # Padrão: modelo multilíngue leve, bom para PT-BR (~120 MB)
    embed_model_name: str = field(
        default_factory=lambda: os.environ.get(
            "EMBED_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    embed_cache_dir: str = field(
        default_factory=lambda: os.environ.get("EMBED_CACHE_DIR", "models")
    )
    embed_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("EMBED_BATCH_SIZE", "32"))
    )
    embed_disk_cache: bool = field(
        default_factory=lambda: os.environ.get("EMBED_DISK_CACHE", "true").lower() == "true"
    )

    # --- RAG ---
    docs_dir: str = field(
        default_factory=lambda: os.environ.get("DOCS_DIR", "data/documents")
    )
    index_dir: str = field(
        default_factory=lambda: os.environ.get("INDEX_DIR", "data/index")
    )
    top_k: int = field(default_factory=lambda: int(os.environ.get("TOP_K", "5")))

    # Quantos candidatos buscar antes do reranking (top_k × multiplier)
    candidate_multiplier: int = field(
        default_factory=lambda: int(os.environ.get("CANDIDATE_MULTIPLIER", "4"))
    )
    # Score mínimo para incluir chunk no contexto (0.0–1.0)
    min_score: float = field(
        default_factory=lambda: float(os.environ.get("MIN_SCORE", "0.25"))
    )
    # Peso do score léxico BM25 na fusão (0.0 = só semântico)
    lexical_weight: float = field(
        default_factory=lambda: float(os.environ.get("LEXICAL_WEIGHT", "0.30"))
    )

    # --- Chunking ---
    chunk_size: int = field(default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "512")))
    # overlap em número de SENTENÇAS (não caracteres)
    chunk_overlap: int = field(default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "1")))

    # --- Pipeline ---
    max_context_chars: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CONTEXT_CHARS", "2000"))
    )
    max_history_turns: int = field(
        default_factory=lambda: int(os.environ.get("MAX_HISTORY_TURNS", "6"))
    )

    # --- Conversas ---
    conversations_file: str = field(
        default_factory=lambda: os.environ.get("CONVERSATIONS_FILE", "data/conversations.json")
    )

    # --- Logging ---
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))


def get_settings() -> Settings:
    return Settings()


def setup_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

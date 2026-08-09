"""
Configuração centralizada do Offline Chatbot.

Todos os valores são lidos de variáveis de ambiente (e arquivo .env opcional).
Este módulo é a ÚNICA fonte de verdade para todos os parâmetros do pipeline.

Valores efetivos do baseline (Phase 2):
  lexical_weight      : 0.40  (corrigido de 0.30 — valor real usado pelo retriever)
  max_context_chars   : 3500  (corrigido de 2000 — valor real usado pelo pipeline)
  chunk_overlap       : 50    (tokens, não sentenças — corrigido doc mismatch)
  bm25_k1             : 1.5
  bm25_b              : 0.75
  hnsw_ef_search      : 64
  hnsw_m              : 32
  hnsw_ef_construction: 200
  adaptive_sigma      : 1.0   (multiplicador do desvio padrão no filtro adaptativo)
  gap_filter_enabled  : true
  keyword_filter_enabled: true
  min_keyword_overlap : 0.15
  temporal_validation_enabled: true
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
        default_factory=lambda: os.environ.get(
            "MODEL_PATH", "models/Qwen3-0.6B-Q8_0.gguf"
        )
    )
    n_ctx: int = field(default_factory=lambda: int(os.environ.get("N_CTX", "4096")))
    n_threads: int = field(
        default_factory=lambda: int(os.environ.get("N_THREADS", "4"))
    )
    n_gpu_layers: int = field(
        default_factory=lambda: int(os.environ.get("N_GPU_LAYERS", "0"))
    )

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
        default_factory=lambda: os.environ.get("EMBED_DISK_CACHE", "true").lower()
        == "true"
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

    # Peso do score léxico BM25 na fusão híbrida (0.0 = só semântico, 1.0 = só BM25).
    # Baseline: 0.40 — alinhado com o valor efetivo do retriever.
    # (Era 0.30 em versões anteriores deste arquivo, mas 0.40 no módulo retriever.py)
    lexical_weight: float = field(
        default_factory=lambda: float(os.environ.get("LEXICAL_WEIGHT", "0.40"))
    )

    # --- Chunking ---
    chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "512"))
    )
    # Overlap em número de TOKENS entre chunks consecutivos.
    # Nota: versões anteriores documentavam como "sentenças" — corrigido.
    # Baseline: 50 tokens (~13% de max_tokens=375).
    chunk_overlap: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "50"))
    )
    # Tokens mínimos por chunk (chunks menores são mesclados).
    chunk_min_tokens: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_MIN_TOKENS", "200"))
    )
    # Tokens máximos por chunk.
    chunk_max_tokens: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_MAX_TOKENS", "375"))
    )

    # --- Pipeline ---
    # Tamanho máximo do contexto enviado ao LLM em caracteres.
    # Baseline: 3500 — alinhado com o valor efetivo de pipeline.py.
    # (Era 2000 em versões anteriores deste arquivo)
    max_context_chars: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CONTEXT_CHARS", "3500"))
    )
    max_history_turns: int = field(
        default_factory=lambda: int(os.environ.get("MAX_HISTORY_TURNS", "6"))
    )

    # --- BM25 ---
    bm25_k1: float = field(
        default_factory=lambda: float(os.environ.get("BM25_K1", "1.5"))
    )
    bm25_b: float = field(
        default_factory=lambda: float(os.environ.get("BM25_B", "0.75"))
    )

    # --- HNSW (FAISS) ---
    hnsw_m: int = field(default_factory=lambda: int(os.environ.get("HNSW_M", "32")))
    hnsw_ef_construction: int = field(
        default_factory=lambda: int(os.environ.get("HNSW_EF_CONSTRUCTION", "200"))
    )
    hnsw_ef_search: int = field(
        default_factory=lambda: int(os.environ.get("HNSW_EF_SEARCH", "64"))
    )

    # --- Filtros do retriever ---
    # Multiplicador de sigma para o filtro adaptativo (threshold = best - sigma * std).
    adaptive_sigma: float = field(
        default_factory=lambda: float(os.environ.get("ADAPTIVE_SIGMA", "1.0"))
    )
    # Ativa/desativa o gap filter (True = ativo no baseline).
    gap_filter_enabled: bool = field(
        default_factory=lambda: os.environ.get("GAP_FILTER_ENABLED", "true").lower()
        == "true"
    )
    # Ativa/desativa o keyword filter (True = ativo no baseline).
    keyword_filter_enabled: bool = field(
        default_factory=lambda: os.environ.get("KEYWORD_FILTER_ENABLED", "true").lower()
        == "true"
    )
    # Limiar mínimo de overlap de keywords entre query e chunks (0.0–1.0).
    min_keyword_overlap: float = field(
        default_factory=lambda: float(os.environ.get("MIN_KEYWORD_OVERLAP", "0.15"))
    )
    # Ativa/desativa a validação temporal pós-geração (True = ativo no baseline).
    temporal_validation_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "TEMPORAL_VALIDATION_ENABLED", "true"
        ).lower()
        == "true"
    )

    # Scores de confiança
    high_confidence_score: float = field(
        default_factory=lambda: float(os.environ.get("HIGH_CONFIDENCE_SCORE", "0.65"))
    )
    low_confidence_score: float = field(
        default_factory=lambda: float(os.environ.get("LOW_CONFIDENCE_SCORE", "0.45"))
    )

    # --- Cross-Encoder Reranking (Phase 3E/3F) ---
    # Enable/disable cross-encoder reranking. When disabled, pipeline uses hybrid only.
    cross_encoder_enabled: bool = field(
        default_factory=lambda: os.environ.get("CROSS_ENCODER_ENABLED", "false").lower()
        == "true"
    )
    # HuggingFace model name for the cross-encoder.
    cross_encoder_model: str = field(
        default_factory=lambda: os.environ.get(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    # Number of hybrid-reranked candidates to pass to the cross-encoder.
    cross_encoder_top_k: int = field(
        default_factory=lambda: int(os.environ.get("CROSS_ENCODER_TOP_K", "20"))
    )
    # Weight of the hybrid score in fusion (hybrid_weight + ce_weight = 1.0).
    # 0.0 = pure CE ranking, 1.0 = pure hybrid ranking.
    # Phase 3F validated: 0.4 hybrid + 0.6 CE is optimal.
    cross_encoder_hybrid_weight: float = field(
        default_factory=lambda: float(
            os.environ.get("CROSS_ENCODER_HYBRID_WEIGHT", "0.4")
        )
    )

    # --- Conversas ---
    conversations_file: str = field(
        default_factory=lambda: os.environ.get(
            "CONVERSATIONS_FILE", "data/conversations.json"
        )
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

"""
Componentes compartilhados pelos scripts de avaliação do RAG
(eval_rag.py, run_rag_experiment.py, analyze_rag_failures.py,
generate_eval_dataset.py).

Antes desta refatoração, cada um desses quatro scripts reimplementava, quase
palavra por palavra, a construção de EmbeddingModel/VectorStore/Retriever a
partir de Settings — inclusive com uma inconsistência real entre eles: só
run_rag_experiment.py e analyze_rag_failures.py resolviam os diretórios
relativos à raiz do projeto (ROOT), então rodar eval_rag.py de um diretório
diferente do root podia carregar o índice errado silenciosamente.

Além disso, nenhum dos três scripts originais habilitava o cross-encoder
reranker mesmo quando CROSS_ENCODER_ENABLED=true (o padrão em .env.example) —
ou seja, a avaliação media um pipeline de retrieval DIFERENTE do que
realmente roda em produção via api/main.py. load_retriever_from_settings()
espelha exatamente a construção do Retriever em api/main.py, incluindo o
cross-encoder, para que os números de avaliação reflitam o sistema real.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.embeddings import EmbeddingModel  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402

# Caminho padrão do dataset de avaliação de RETRIEVAL, usado por eval_rag.py /
# run_rag_experiment.py / analyze_rag_failures.py quando --dataset/--eval-file
# não é passado explicitamente. Não tem reference_answer.
DEFAULT_DATASET_PATH = "data/eval/dataset.json"

_VERSION_DIR_RE = re.compile(r"^v(\d+)$")


def resolve_path(path: str) -> str:
    """Resolve um caminho relativo à raiz do projeto (não ao CWD atual)."""
    p = Path(path)
    return str(p) if p.is_absolute() else str(ROOT / p)


def next_metrics_version(metrics_dir: Path) -> int:
    """Determina o próximo número de versão disponível em metrics_dir/vN/.

    Usado pelos scripts que exportam métricas versionadas de retrieval
    (eval_rag.py) — cada diretório de métricas tem seu próprio contador
    independente.
    """
    if not metrics_dir.exists():
        return 1
    versions = []
    for p in metrics_dir.iterdir():
        if p.is_dir():
            m = _VERSION_DIR_RE.match(p.name)
            if m:
                versions.append(int(m.group(1)))
    return max(versions, default=0) + 1


def load_retriever_from_settings(
    settings,
    *,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    lexical_weight: Optional[float] = None,
    adaptive_sigma: Optional[float] = None,
    gap_filter_enabled: Optional[bool] = None,
    keyword_filter_enabled: Optional[bool] = None,
    candidate_multiplier: Optional[int] = None,
    cross_encoder_enabled: Optional[bool] = None,
    cross_encoder_model: Optional[str] = None,
    cross_encoder_top_k: Optional[int] = None,
    cross_encoder_hybrid_weight: Optional[float] = None,
    cross_encoder_min_score: Optional[float] = None,
) -> tuple[VectorStore, EmbeddingModel, Retriever]:
    """Carrega EmbeddingModel + VectorStore + Retriever a partir de Settings.

    Os parâmetros nomeados (top_k, min_score, ...) permitem que scripts de
    ablação (run_rag_experiment.py) sobrescrevam parâmetros individuais do
    retriever sem duplicar esta função — quando None, usa o valor de settings.

    Sai do processo com um erro claro se o índice não existir ou estiver
    vazio, já que todo script de avaliação precisa de um índice pronto.
    """
    print(f"  Carregando modelo de embeddings: {settings.embed_model_name}")
    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=resolve_path(settings.embed_cache_dir),
        batch_size=settings.embed_batch_size,
        use_disk_cache=settings.embed_disk_cache,
    )

    index_dir = resolve_path(settings.index_dir)
    if not Path(index_dir).exists():
        print(
            f"\n[ERRO] Índice não encontrado em: {index_dir}\n"
            "       Execute primeiro: python scripts/index_documents.py\n"
        )
        sys.exit(1)

    vs = VectorStore(
        index_dir=index_dir,
        embedding_dim=embed_model.dimension,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    if vs.size == 0:
        print("\n[ERRO] VectorStore vazio — nenhum chunk indexado.\n")
        sys.exit(1)

    ce_enabled = (
        cross_encoder_enabled
        if cross_encoder_enabled is not None
        else settings.cross_encoder_enabled
    )
    cross_encoder = None
    if ce_enabled:
        from rag.cross_encoder import CrossEncoderReranker

        ce_model = cross_encoder_model or settings.cross_encoder_model
        ce_top_k = (
            cross_encoder_top_k
            if cross_encoder_top_k is not None
            else settings.cross_encoder_top_k
        )
        ce_hybrid_weight = (
            cross_encoder_hybrid_weight
            if cross_encoder_hybrid_weight is not None
            else settings.cross_encoder_hybrid_weight
        )
        ce_min_score = (
            cross_encoder_min_score
            if cross_encoder_min_score is not None
            else settings.cross_encoder_min_score
        )
        print(f"  Carregando cross-encoder: {ce_model}")
        cross_encoder = CrossEncoderReranker(
            model_name=ce_model,
            hybrid_weight=ce_hybrid_weight,
            ce_top_k=ce_top_k,
            min_score=ce_min_score,
        )

    retriever = Retriever(
        vectorstore=vs,
        embed_model=embed_model,
        top_k=top_k if top_k is not None else settings.top_k,
        candidate_multiplier=(
            candidate_multiplier
            if candidate_multiplier is not None
            else settings.candidate_multiplier
        ),
        min_score=min_score if min_score is not None else settings.min_score,
        lexical_weight=(
            lexical_weight if lexical_weight is not None else settings.lexical_weight
        ),
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        adaptive_sigma=(
            adaptive_sigma if adaptive_sigma is not None else settings.adaptive_sigma
        ),
        gap_filter_enabled=(
            gap_filter_enabled
            if gap_filter_enabled is not None
            else settings.gap_filter_enabled
        ),
        keyword_filter_enabled=(
            keyword_filter_enabled
            if keyword_filter_enabled is not None
            else settings.keyword_filter_enabled
        ),
        high_confidence_score=settings.high_confidence_score,
        low_confidence_score=settings.low_confidence_score,
        cross_encoder=cross_encoder,
    )
    return vs, embed_model, retriever


def parse_bool(value, default: bool) -> bool:
    """Parseia flags de CLI true/false/1/0/yes/no; None mantém o default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")

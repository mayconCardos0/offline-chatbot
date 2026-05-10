"""
CLI para carregar, chunkar, embedar e indexar documentos no FAISS.

Melhorias em relação à versão anterior:
  - Exibe estatísticas de qualidade dos chunks (tokens: min/média/max).
  - Detecta e avisa sobre chunks suspeitos (muito curtos ou muito longos).
  - Suporta --chunk-size e --overlap como argumentos CLI (sem editar .env).
  - Relatório final mais informativo: tokens totais, tamanho médio de chunk,
    distribuição de páginas por documento.
  - Compatível com a nova saída de load_documents() que inclui page e section.

Uso:
    python scripts/index_documents.py [--docs-dir PATH] [--chunk-size N] [--overlap N]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings, setup_logging
from rag.chunking import ChunkConfig, chunk_document, count_tokens
from rag.embeddings import EmbeddingModel
from rag.loader import load_documents
from rag.vectorstore import VectorStore


def _print_chunk_stats(all_chunks: list[dict]) -> None:
    """Exibe estatísticas de qualidade dos chunks gerados."""
    if not all_chunks:
        return

    token_counts = [count_tokens(c["text"]) for c in all_chunks]
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)

    # Chunks suspeitos: muito curtos (possível artefato) ou muito longos
    tiny = [c for c, t in zip(all_chunks, token_counts) if t < 30]
    large = [c for c, t in zip(all_chunks, token_counts) if t > 600]

    print(f"\n  Estatísticas de chunks:")
    print(f"    Total de chunks   : {len(all_chunks)}")
    print(f"    Total de tokens   : {total_tokens:,}")
    print(
        f"    Tokens por chunk  : min={min_tokens}  média={avg_tokens:.0f}  max={max_tokens}"
    )

    if tiny:
        print(
            f"\n  ⚠  {len(tiny)} chunk(s) com menos de 30 tokens (podem ser artefatos):"
        )
        for c in tiny[:3]:
            preview = c["text"][:80].replace("\n", " ")
            print(f"     [{c.get('page', '?')}p] \"{preview}\"")
        if len(tiny) > 3:
            print(f"     ... e mais {len(tiny) - 3} chunks.")

    if large:
        print(
            f"\n  ⚠  {len(large)} chunk(s) com mais de 600 tokens (verifique configuração):"
        )
        for c in large[:3]:
            preview = c["text"][:80].replace("\n", " ")
            print(f"     [{c.get('page', '?')}p] \"{preview}\"")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indexa documentos no FAISS vector store com chunking semântico.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Diretório de documentos (override do .env).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Tamanho máximo do chunk em caracteres (override do .env).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Sobreposição em número de sentenças entre chunks (override do .env).",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    # Resolve parâmetros: CLI > .env > padrão
    docs_dir = args.docs_dir if args.docs_dir is not None else settings.docs_dir
    chunk_size = args.chunk_size if args.chunk_size is not None else settings.chunk_size
    overlap = args.overlap if args.overlap is not None else settings.chunk_overlap

    print(f"\nIndexando documentos de: {docs_dir}")
    print(f"Configuração: chunk_size={chunk_size} chars  overlap={overlap} sentenças")

    # --- 1. Carrega documentos ---
    logger.info("Carregando documentos de '%s'", docs_dir)
    documents = load_documents(docs_dir)

    if not documents:
        print(
            f"\nAVISO: Nenhum documento suportado encontrado em '{docs_dir}'. Nada a indexar."
        )
        sys.exit(0)

    # Resume documentos carregados
    sources = {}
    for doc in documents:
        src = Path(doc["source"]).name
        sources[src] = sources.get(src, 0) + 1

    print(f"\nDocumentos carregados ({len(sources)} arquivo(s)):")
    for name, count in sorted(sources.items()):
        suffix = f" — {count} páginas" if count > 1 else ""
        print(f"  • {name}{suffix}")

    # --- 2. Chunkeia com configuração semântica ---
    # Cria ChunkConfig explícito para garantir parâmetros consistentes
    # entre a conversão chars→tokens e o comportamento do montador de chunks
    max_tokens = max(100, chunk_size // 3)
    chunk_config = ChunkConfig(
        min_tokens=max(50, max_tokens // 3),
        max_tokens=max_tokens,
        overlap_sents=overlap,
    )

    all_chunks: list[dict] = []
    for doc in documents:
        chunks = chunk_document(
            doc,
            chunk_size=chunk_size,
            overlap=overlap,
            config=chunk_config,
        )
        all_chunks.extend(chunks)

        page_info = f" [pág. {doc.get('page')}]" if doc.get("page") else ""
        logger.debug(
            "'%s'%s: %d chunks.", Path(doc["source"]).name, page_info, len(chunks)
        )

    logger.info(
        "%d chunks criados a partir de %d entrada(s).", len(all_chunks), len(documents)
    )

    # Exibe estatísticas de qualidade
    _print_chunk_stats(all_chunks)

    # --- 3. Gera embeddings ---
    print(f"\nCarregando modelo de embedding: {settings.embed_model_name}")
    logger.info("Carregando modelo de embedding '%s'", settings.embed_model_name)
    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=settings.embed_cache_dir,
        batch_size=settings.embed_batch_size,
        use_disk_cache=settings.embed_disk_cache,
    )

    texts = [chunk["text"] for chunk in all_chunks]
    print(f"Gerando embeddings para {len(texts)} chunks...")
    logger.info("Gerando embeddings para %d chunks…", len(texts))
    embeddings = embed_model.embed(texts)

    # --- 4. Salva no vector store ---
    store = VectorStore(
        index_dir=settings.index_dir,
        embedding_dim=embed_model.dimension,
    )
    store.add(all_chunks, embeddings)
    store.save()

    # --- Relatório final ---
    index_path = Path(settings.index_dir).resolve()
    print(f"\n{'─' * 55}")
    print(f"  ✓  Indexação concluída")
    print(f"{'─' * 55}")
    print(f"  Arquivos indexados   : {len(sources)}")
    print(f"  Chunks no índice     : {len(all_chunks)}")
    print(
        f"  Tokens estimados     : {sum(count_tokens(c['text']) for c in all_chunks):,}"
    )
    print(f"  Índice salvo em      : {index_path}")
    print(f"  Modelo de embedding  : {settings.embed_model_name}")
    print(f"  Tipo de índice       : HNSW (busca aproximada, baixa latência)")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    main()

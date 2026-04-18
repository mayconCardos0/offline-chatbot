"""
CLI para carregar, chunkar, embedar e indexar documentos no FAISS.

Uso:
    python scripts/index_documents.py [--docs-dir PATH]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings, setup_logging
from rag.chunking import chunk_document
from rag.embeddings import EmbeddingModel
from rag.loader import load_documents
from rag.vectorstore import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa documentos no FAISS vector store.")
    parser.add_argument("--docs-dir", default=None, help="Diretório de documentos (override do .env).")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    docs_dir = args.docs_dir if args.docs_dir is not None else settings.docs_dir

    # --- Carrega ---
    logger.info("Carregando documentos de '%s'", docs_dir)
    documents = load_documents(docs_dir)

    if not documents:
        print(f"AVISO: Nenhum documento suportado encontrado em '{docs_dir}'. Nada a indexar.")
        sys.exit(0)

    # --- Chunkeia ---
    all_chunks: list[dict] = []
    for doc in documents:
        chunks = chunk_document(
            doc,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        all_chunks.extend(chunks)
        logger.debug("'%s': %d chunks.", Path(doc["source"]).name, len(chunks))

    logger.info("%d chunks criados a partir de %d documento(s).", len(all_chunks), len(documents))

    # --- Embeda ---
    logger.info("Carregando modelo de embedding '%s'", settings.embed_model_name)
    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=settings.embed_cache_dir,
        batch_size=settings.embed_batch_size,
        use_disk_cache=settings.embed_disk_cache,
    )

    texts = [chunk["text"] for chunk in all_chunks]
    logger.info("Gerando embeddings para %d chunks…", len(texts))
    embeddings = embed_model.embed(texts)

    # --- Salva ---
    store = VectorStore(
        index_dir=settings.index_dir,
        embedding_dim=embed_model.dimension,
    )
    store.add(all_chunks, embeddings)
    store.save()

    index_path = Path(settings.index_dir).resolve()
    print(
        f"Indexados {len(documents)} documento(s), "
        f"{len(all_chunks)} chunks → {index_path}"
    )
    print(f"Modelo de embedding: {settings.embed_model_name}")
    print(f"Tipo de índice: HNSW (busca aproximada, baixa latência)")


if __name__ == "__main__":
    main()

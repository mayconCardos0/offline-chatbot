"""
CLI script to load, chunk, embed, and index documents into the FAISS vector store.

Usage:
    python scripts/index_documents.py [--docs-dir PATH]
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow imports from the project root (offline-chatbot/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings, setup_logging
from rag.chunking import chunk_document
from rag.embeddings import EmbeddingModel
from rag.loader import load_documents
from rag.vectorstore import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into the FAISS vector store.")
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Override the configured documents directory for this run.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    docs_dir = args.docs_dir if args.docs_dir is not None else settings.docs_dir

    # --- Load ---
    logger.info("Loading documents from '%s'", docs_dir)
    documents = load_documents(docs_dir)

    if not documents:
        print(f"WARNING: No supported documents found in '{docs_dir}'. Nothing to index.")
        sys.exit(0)

    # --- Chunk ---
    all_chunks: list[dict] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        all_chunks.extend(chunks)

    logger.info("Created %d chunks from %d document(s).", len(all_chunks), len(documents))

    # --- Embed ---
    logger.info("Loading embedding model '%s'", settings.embed_model_name)
    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=settings.embed_cache_dir,
    )

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embed_model.embed(texts)
    embedding_dim = len(embeddings[0])

    # --- Save ---
    store = VectorStore(index_dir=settings.index_dir, embedding_dim=embedding_dim)
    store.add(all_chunks, embeddings)
    store.save()

    index_path = Path(settings.index_dir).resolve()
    print(f"Indexed {len(documents)} document(s), {len(all_chunks)} chunk(s) -> {index_path}")


if __name__ == "__main__":
    main()

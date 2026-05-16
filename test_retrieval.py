"""Testa o retrieval para as perguntas problemáticas."""

import sys
from pathlib import Path

from core.config import get_settings
from rag.embeddings import EmbeddingModel
from rag.retriever import Retriever
from rag.vectorstore import VectorStore

sys.path.insert(0, str(Path.cwd()))

# Inicializa componentes
settings = get_settings()
embed_model = EmbeddingModel(
    model_name=settings.embed_model_name,
    cache_dir=settings.embed_cache_dir,
)
vectorstore = VectorStore(
    index_dir=settings.index_dir,
    embedding_dim=embed_model.dimension,
)
vectorstore.load()

retriever = Retriever(
    vectorstore=vectorstore,
    embed_model=embed_model,
    top_k=5,
    candidate_multiplier=4,
    min_score=0.40,
    lexical_weight=0.30,
)

# Perguntas para testar
queries = [
    "O que foi a República de Saló?",
    "Quem foi Napoleão Bonaparte?",
    "como foi o segundo governo Vargas?",
    "Qual o nome da bomba atômica que atingiu Hiroshima?",
    "O que foi o populismo no Brasil?",
]

print("=" * 80)
print("TESTE DE RETRIEVAL - Análise de Chunks Recuperados")
print("=" * 80)

for query in queries:
    print(f"\n{'─'*80}")
    print(f"QUERY: {query}")
    print(f"{'─'*80}")

    chunks = retriever.retrieve(query)

    if not chunks:
        print("❌ NENHUM CHUNK RECUPERADO")
        continue

    print(f"\n✓ {len(chunks)} chunks recuperados\n")

    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", 0)
        confidence = chunk.get("confidence", "?")
        breadcrumb = chunk.get("breadcrumb", "N/A")
        text = chunk.get("text", "")[:200]

        print(f"[{i}] Score: {score:.3f} | Confidence: {confidence}")
        print(f"    Breadcrumb: {breadcrumb}")
        print(f"    Texto: {text}...")
        print()

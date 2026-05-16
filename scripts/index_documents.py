"""
CLI para indexar documentos com estrutura hierárquica e avaliação do RAG.

Melhorias em relação à versão anterior
───────────────────────────────────────
  Estrutura hierárquica
    - Detecta Unidade > Capítulo > Tópico via rag.structure.detect_structure().
    - Cada chunk carrega breadcrumb ("Unidade X > Cap. Y > Tópico Z") como
      metadado, injetado no contexto do LLM.
    - O contexto injetado no prompt agora lê:
        [fonte: livro.pdf | Unidade 2 > Cap. 3 > 3.1 Fotossíntese | score: 0.87]
      em vez de apenas [fonte: livro.pdf | score: 0.87].

  Avaliação do RAG
    - --eval: após indexar, roda avaliação automática com dataset sintético.
    - --eval-file PATH: usa dataset anotado manualmente (JSON).
    - --eval-k N: profundidade de avaliação (padrão 5).
    - --save-eval PATH: salva o relatório em JSON para comparação futura.
    - --compare PATH: compara com um relatório anterior salvo.

  Relatório de indexação
    - Exibe distribuição por Unidade/Capítulo/Tópico detectados.
    - Mantém todas as estatísticas de tokens da versão anterior.
    - Avisa sobre seções sem heading (possível material sem estrutura).

Uso
───
    # Indexar com estrutura hierárquica (padrão)
    python scripts/index_documents.py

    # Indexar e avaliar o RAG com 30 queries sintéticas, K=5
    python scripts/index_documents.py --eval --eval-k 5

    # Indexar e avaliar com dataset anotado
    python scripts/index_documents.py --eval-file data/eval/dataset.json

    # Salvar relatório para comparação futura
    python scripts/index_documents.py --eval --save-eval data/eval/report_v1.json

    # Comparar com relatório anterior
    python scripts/index_documents.py --eval --compare data/eval/report_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings, setup_logging  # noqa: E402
from rag.chunking import ChunkConfig, chunk_document, count_tokens  # noqa: E402
from rag.embeddings import EmbeddingModel  # noqa: E402
from rag.evaluation import (  # noqa: E402
    EvaluationReport,
    build_synthetic_dataset,
    compare_reports,
    evaluate_retriever,
    load_dataset_from_file,
    print_report,
    save_dataset_to_file,
)
from rag.loader import load_documents  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.structure import detect_structure, sections_to_docs  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relatório de estrutura hierárquica
# ---------------------------------------------------------------------------


def _print_structure_report(all_chunks: list[dict]) -> None:
    """Exibe distribuição hierárquica dos chunks indexados."""
    units: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    no_structure = 0

    for c in all_chunks:
        unit = c.get("unit") or "— (sem unidade)"
        chapter = c.get("chapter") or "— (sem capítulo)"
        topic = c.get("topic") or ""
        if not c.get("unit") and not c.get("chapter"):
            no_structure += 1
        units[unit][chapter].add(topic)

    print("\n  Estrutura hierárquica detectada:")
    for unit, chapters in sorted(units.items()):
        print(f"    📘 {unit}")
        for chapter, topics in sorted(chapters.items()):
            topic_list = [t for t in topics if t]
            print(f"       📄 {chapter}  ({len(topic_list)} tópico(s))")
            for topic in sorted(topic_list)[:5]:
                print(f"          • {topic}")
            if len(topic_list) > 5:
                print(f"          ... e mais {len(topic_list) - 5} tópico(s)")

    if no_structure:
        pct = no_structure / len(all_chunks) * 100
        print(
            f"\n  ⚠  {no_structure} chunk(s) ({pct:.0f}%) sem heading detectado."
        )
        print(
            "     Isso é normal para prefácios, apêndices e páginas de índice."
        )


def _print_chunk_stats(all_chunks: list[dict]) -> None:
    """Exibe estatísticas de qualidade dos chunks."""
    if not all_chunks:
        return

    token_counts = [count_tokens(c["text"]) for c in all_chunks]
    total = sum(token_counts)
    avg = total / len(token_counts)
    mn, mx = min(token_counts), max(token_counts)

    tiny = [c for c, t in zip(all_chunks, token_counts) if t < 30]
    large = [c for c, t in zip(all_chunks, token_counts) if t > 600]

    print("\n  Estatísticas de chunks:")
    print(f"    Total de chunks   : {len(all_chunks)}")
    print(f"    Total de tokens   : {total:,}")
    print(f"    Tokens por chunk  : min={mn}  média={avg:.0f}  max={mx}")

    if tiny:
        print(f"\n  ⚠  {len(tiny)} chunk(s) com < 30 tokens (possível artefato):")
        for c in tiny[:3]:
            preview = c["text"][:80].replace("\n", " ")
            crumb = c.get("breadcrumb", "")
            print(f"     [{crumb or '?'}] \"{preview}\"")

    if large:
        print(f"\n  ⚠  {len(large)} chunk(s) com > 600 tokens (chunk muito grande):")
        for c in large[:3]:
            preview = c["text"][:80].replace("\n", " ")
            print(f"     \"{preview}\"")


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------


def _run_evaluation(
    vectorstore: VectorStore,
    embed_model: EmbeddingModel,
    args: argparse.Namespace,
    settings,
) -> EvaluationReport | None:
    """Roda avaliação do RAG e imprime/salva o relatório."""
    print(f"\n{'─' * 55}")
    print("  Avaliação do RAG")
    print(f"{'─' * 55}")

    # Monta retriever para avaliação (mesmos parâmetros do servidor)
    retriever = Retriever(
        vectorstore=vectorstore,
        embed_model=embed_model,
        top_k=args.eval_k,
        candidate_multiplier=getattr(settings, "candidate_multiplier", 4),
        min_score=getattr(settings, "min_score", 0.25),
        lexical_weight=getattr(settings, "lexical_weight", 0.30),
    )

    # Carrega ou gera dataset
    if args.eval_file:
        print(f"  Carregando dataset: {args.eval_file}")
        dataset = load_dataset_from_file(args.eval_file)
    else:
        n = getattr(args, "eval_n", 30)
        print(f"  Gerando dataset sintético ({n} queries)…")
        dataset = build_synthetic_dataset(vectorstore, n_samples=n)

        # Oferece salvar o dataset gerado
        if getattr(args, "save_dataset", None):
            save_dataset_to_file(dataset, args.save_dataset)
            print(f"  Dataset salvo em: {args.save_dataset}")

    if not dataset:
        print("  Dataset vazio — avaliação ignorada.")
        return None

    print(f"  Avaliando {len(dataset)} queries com K={args.eval_k}…")
    report = evaluate_retriever(retriever, dataset, k=args.eval_k)

    print_report(report, title=f"RAG Evaluation  (K={args.eval_k})")

    # Salva relatório
    if getattr(args, "save_eval", None):
        out = Path(args.save_eval)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"  Relatório salvo em: {out}")

    # Compara com relatório anterior
    if getattr(args, "compare", None):
        prev_path = Path(args.compare)
        if prev_path.exists():
            with open(prev_path, encoding="utf-8") as f:
                prev_data = json.load(f)
            # Reconstrói um EvaluationReport "fake" para display
            prev = EvaluationReport(
                k=prev_data.get("k", args.eval_k),
                n_queries=prev_data.get("n_queries", 0),
                mean_precision=prev_data.get("precision@k", 0),
                mean_recall=prev_data.get("recall@k", 0),
                mean_f1=prev_data.get("f1@k", 0),
                hit_rate=prev_data.get("hit_rate@k", 0),
                mrr=prev_data.get("mrr", 0),
                mean_ndcg=prev_data.get("ndcg@k", 0),
                map_at_k=prev_data.get("map@k", 0),
                mean_latency_ms=prev_data.get("mean_latency_ms", 0),
            )
            compare_reports(
                {"Anterior": prev, "Atual": report},
                title=f"Comparação  (K={args.eval_k})",
            )
        else:
            print(f"  ⚠  Relatório anterior não encontrado: {prev_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Indexa documentos com estrutura hierárquica e avalia o RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Indexação
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument(
        "--no-structure",
        action="store_true",
        help="Desativa detecção hierárquica (comportamento legado).",
    )

    # Avaliação
    eval_group = parser.add_argument_group("avaliação")
    eval_group.add_argument(
        "--eval",
        action="store_true",
        help="Roda avaliação com dataset sintético após indexar.",
    )
    eval_group.add_argument(
        "--eval-file",
        default=None,
        metavar="PATH",
        help="Arquivo JSON com dataset anotado manualmente.",
    )
    eval_group.add_argument(
        "--eval-k",
        type=int,
        default=5,
        metavar="N",
        help="Profundidade de avaliação (número de chunks considerados).",
    )
    eval_group.add_argument(
        "--eval-n",
        type=int,
        default=30,
        metavar="N",
        help="Número de queries no dataset sintético.",
    )
    eval_group.add_argument(
        "--save-eval",
        default=None,
        metavar="PATH",
        help="Salva relatório de avaliação em JSON.",
    )
    eval_group.add_argument(
        "--save-dataset",
        default=None,
        metavar="PATH",
        help="Salva dataset sintético gerado em JSON para reutilização.",
    )
    eval_group.add_argument(
        "--compare",
        default=None,
        metavar="PATH",
        help="Compara com relatório anterior salvo em JSON.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    docs_dir = args.docs_dir or settings.docs_dir
    chunk_size = args.chunk_size or settings.chunk_size
    overlap = args.overlap or settings.chunk_overlap

    print(f"\nIndexando documentos de: {docs_dir}")
    print(f"Configuração: chunk_size={chunk_size} chars  overlap={overlap} sentenças")
    print(
        f"Detecção hierárquica: {'desativada (--no-structure)' if args.no_structure else 'ativada'}"
    )

    # --- 1. Carrega documentos ---
    documents = load_documents(docs_dir)
    if not documents:
        print(f"\nAVISO: Nenhum documento encontrado em '{docs_dir}'.")
        sys.exit(0)

    # Resume por arquivo
    sources: dict[str, int] = {}
    for doc in documents:
        src = Path(doc["source"]).name
        sources[src] = sources.get(src, 0) + 1

    print(f"\nDocumentos carregados ({len(sources)} arquivo(s)):")
    for name, count in sorted(sources.items()):
        suffix = f" — {count} páginas" if count > 1 else ""
        print(f"  • {name}{suffix}")

    # --- 2. Detecção hierárquica por arquivo ---
    all_docs: list[dict] = []

    if args.no_structure:
        # Modo legado: sem hierarquia
        all_docs = documents
    else:
        # Agrupa páginas por source
        pages_by_source: dict[str, list[dict]] = {}
        for doc in documents:
            src = doc["source"]
            pages_by_source.setdefault(src, []).append(doc)

        for source, pages in pages_by_source.items():
            src_name = Path(source).name
            sections = detect_structure(pages, source)
            docs = sections_to_docs(sections)
            all_docs.extend(docs)

            # Estatística de estrutura detectada
            n_units = len({d.get("unit") for d in docs if d.get("unit")})
            n_chapters = len({d.get("chapter") for d in docs if d.get("chapter")})
            n_topics = len({d.get("topic") for d in docs if d.get("topic")})
            print(
                f"  📐 {src_name}: {n_units} unid. | {n_chapters} cap. | {n_topics} tópicos"
            )

    # --- 3. Chunking semântico ---
    max_tokens = max(100, chunk_size // 3)
    chunk_config = ChunkConfig(
        min_tokens=max(50, max_tokens // 3),
        max_tokens=max_tokens,
        overlap_tokens=overlap,
    )

    all_chunks: list[dict] = []
    for doc in all_docs:
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap, config=chunk_config)
        # Mantém apenas text e source nos metadados
        for c in chunks:
            # Remove metadados hierárquicos, mantendo apenas text e source
            c.pop("unit", None)
            c.pop("chapter", None)
            c.pop("topic", None)
            c.pop("breadcrumb", None)
        all_chunks.extend(chunks)

    logger.info("%d chunks criados.", len(all_chunks))

    # Relatórios
    _print_chunk_stats(all_chunks)

    # --- 4. Embeddings ---
    print(f"\nCarregando modelo: {settings.embed_model_name}")
    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=settings.embed_cache_dir,
        batch_size=settings.embed_batch_size,
        use_disk_cache=settings.embed_disk_cache,
    )

    texts = [c["text"] for c in all_chunks]
    print(f"Gerando embeddings para {len(texts)} chunks…")
    embeddings = embed_model.embed(texts)

    # --- 5. Vector store ---
    store = VectorStore(
        index_dir=settings.index_dir,
        embedding_dim=embed_model.dimension,
    )
    store.add(all_chunks, embeddings)
    store.save()

    # --- Relatório final ---
    index_path = Path(settings.index_dir).resolve()
    sep = "─" * 55
    print(f"\n{sep}")
    print("  ✓  Indexação concluída")
    print(sep)
    print(f"  Arquivos indexados   : {len(sources)}")
    print(f"  Chunks no índice     : {len(all_chunks)}")
    total_tok = sum(count_tokens(c["text"]) for c in all_chunks)
    print(f"  Tokens estimados     : {total_tok:,}")
    print(f"  Índice salvo em      : {index_path}")
    print(f"  Modelo de embedding  : {settings.embed_model_name}")
    print(f"  Hierarquia           : {'desativada' if args.no_structure else 'ativada'}")
    print(f"{sep}\n")

    # --- 6. Avaliação ---
    if args.eval or args.eval_file:
        _run_evaluation(store, embed_model, args, settings)


if __name__ == "__main__":
    main()
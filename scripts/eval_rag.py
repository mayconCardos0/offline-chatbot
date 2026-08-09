"""
Script de avaliação de métricas do pipeline RAG.

Métricas avaliadas
──────────────────
  MAP@K       — Mean Average Precision
  Precision@K — Fração dos K chunks recuperados que são relevantes
  Recall@K    — Fração dos chunks relevantes encontrados no top-K
  MRR         — Mean Reciprocal Rank (posição do primeiro chunk relevante)
  NDCG@K      — Normalized Discounted Cumulative Gain
  Hit Rate@K  — Fração de queries com pelo menos 1 acerto no top-K
  F1@K        — Média harmônica entre Precision@K e Recall@K

Modos de uso
────────────
  # 1. Dataset sintético (smoke-test rápido, sem anotação manual):
  python scripts/eval_rag.py --mode synthetic --k 5 --samples 50

  # 2. Dataset JSON anotado manualmente:
  python scripts/eval_rag.py --mode file --dataset data/eval/eval_dataset.json --k 5

  # 3. Criar dataset sintético e salvar para edição manual:
  python scripts/eval_rag.py --mode export --samples 100 --output data/eval/eval_dataset.json

  # 4. Comparar diferentes valores de K:
  python scripts/eval_rag.py --mode synthetic --k 1 3 5 10 --samples 50

  # 5. Exportar relatório em JSON:
  python scripts/eval_rag.py --mode synthetic --k 5 --output-report reports/eval_result.json

Formato do dataset JSON anotado
─────────────────────────────────
  [
    {
      "query": "O que é regressão linear?",
      "relevant_chunk_ids": ["abc123", "def456"],
      "relevant_texts": ["trecho do livro sobre regressão..."]
    },
    ...
  ]

  Dica: use --mode export para gerar o esqueleto e ajuste manualmente.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Garante import relativo ao root do projeto
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings, setup_logging  # noqa: E402
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
from rag.retriever import Retriever  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers de inicialização
# ---------------------------------------------------------------------------


def load_components(settings) -> tuple[VectorStore, EmbeddingModel, Retriever]:
    """Carrega EmbeddingModel, VectorStore e Retriever prontos para uso."""

    # 1. Embedding model (necessário para saber a dimensão e para o Retriever)
    print(f"  Carregando modelo de embeddings: {settings.embed_model_name}")
    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=settings.embed_cache_dir,
        batch_size=settings.embed_batch_size,
        use_disk_cache=settings.embed_disk_cache,
    )

    # 2. VectorStore
    index_path = Path(settings.index_dir)
    if not index_path.exists():
        print(
            f"\n[ERRO] Índice não encontrado em: {index_path}\n"
            "       Execute primeiro: python scripts/index_documents.py\n"
        )
        sys.exit(1)

    vs = VectorStore(
        index_dir=settings.index_dir,
        embedding_dim=embed_model.dimension,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )

    if vs.size == 0:
        print("\n[ERRO] VectorStore vazio — nenhum chunk indexado.\n")
        sys.exit(1)

    # 3. Retriever
    retriever = Retriever(
        vectorstore=vs,
        embed_model=embed_model,
        top_k=settings.top_k,
        candidate_multiplier=settings.candidate_multiplier,
        min_score=settings.min_score,
        lexical_weight=settings.lexical_weight,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        adaptive_sigma=settings.adaptive_sigma,
        gap_filter_enabled=settings.gap_filter_enabled,
        keyword_filter_enabled=settings.keyword_filter_enabled,
        high_confidence_score=settings.high_confidence_score,
        low_confidence_score=settings.low_confidence_score,
    )

    return vs, embed_model, retriever


# ---------------------------------------------------------------------------
# Modo: export (gerar dataset sintético e salvar)
# ---------------------------------------------------------------------------


def mode_export(args, vs: VectorStore) -> None:
    output = Path(args.output or "data/eval/eval_dataset.json")
    print(f"\n[EXPORT] Gerando dataset sintético com {args.samples} amostras...")
    dataset = build_synthetic_dataset(vs, n_samples=args.samples, seed=args.seed)

    if not dataset:
        print("[ERRO] Nenhum chunk disponível para gerar dataset.")
        sys.exit(1)

    save_dataset_to_file(dataset, str(output))
    print(f"[OK] Dataset salvo em: {output}")
    print(
        "\nPróximos passos:\n"
        "  1. Abra o arquivo JSON e ajuste 'relevant_chunk_ids' e 'relevant_texts'\n"
        "     com as anotações corretas para cada query.\n"
        "  2. Execute: python scripts/eval_rag.py --mode file --dataset " + str(output)
    )


# ---------------------------------------------------------------------------
# Modo: avaliação (synthetic ou file)
# ---------------------------------------------------------------------------


def mode_evaluate(args, vs: VectorStore, retriever: Retriever) -> None:
    # Carrega dataset
    if args.mode == "file":
        if not args.dataset:
            print("[ERRO] --dataset é obrigatório no modo 'file'.")
            sys.exit(1)
        print(f"\n[EVAL] Carregando dataset: {args.dataset}")
        dataset = load_dataset_from_file(args.dataset)
    else:
        print(f"\n[EVAL] Gerando dataset sintético com {args.samples} amostras...")
        dataset = build_synthetic_dataset(vs, n_samples=args.samples, seed=args.seed)

    if not dataset:
        print("[ERRO] Dataset vazio — nada a avaliar.")
        sys.exit(1)

    print(f"       {len(dataset)} queries carregadas.")

    k_values: list[int] = args.k

    # ── Avaliação para cada valor de K ──────────────────────────────────────
    reports: dict[str, EvaluationReport] = {}

    for k in k_values:
        print(
            f"\n[EVAL] Avaliando K={k} sobre {len(dataset)} queries...",
            end=" ",
            flush=True,
        )
        t0 = time.perf_counter()
        report = evaluate_retriever(retriever, dataset, k=k)
        elapsed = time.perf_counter() - t0
        print(f"concluído em {elapsed:.1f}s")
        reports[f"K={k}"] = report

    # ── Exibição ─────────────────────────────────────────────────────────────
    if len(k_values) == 1:
        print_report(reports[f"K={k_values[0]}"], title="RAG Evaluation")
    else:
        for label, report in reports.items():
            print_report(report, title=f"RAG Evaluation — {label}")
        compare_reports(reports, title="Comparação por K")

    # ── Detalhe por query (se solicitado) ────────────────────────────────────
    if args.verbose and k_values:
        _print_per_query(reports[f"K={k_values[-1]}"])

    # ── Exportar relatório JSON ───────────────────────────────────────────────
    if args.output_report:
        _save_report(reports, args.output_report)


def _print_per_query(report: EvaluationReport) -> None:
    print(f"\n{'═'*75}")
    print(f"  Detalhe por query  (K={report.k})")
    print(f"{'═'*75}")
    header = f"  {'Query':<45} {'P@K':>6} {'R@K':>6} {'RR':>6} {'AP':>6}"
    print(header)
    print("  " + "─" * 73)
    for qr in sorted(report.per_query, key=lambda r: r.recall_at_k, reverse=True):
        q = qr.query[:43] + ".." if len(qr.query) > 45 else qr.query
        print(
            f"  {q:<45} {qr.precision_at_k:>6.3f} {qr.recall_at_k:>6.3f} "
            f"{qr.reciprocal_rank:>6.3f} {qr.ap_at_k:>6.3f}"
        )
    print(f"{'═'*75}\n")


def _save_report(reports: dict[str, EvaluationReport], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {label: r.to_dict() for label, r in reports.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Relatório salvo em: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_rag",
        description="Avalia métricas do pipeline RAG (MAP, Precision@K, Recall@K, MRR, ...).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["synthetic", "file", "export"],
        default="synthetic",
        help=(
            "synthetic: dataset gerado automaticamente (padrão)\n"
            "file:      dataset JSON anotado manualmente\n"
            "export:    salva dataset sintético para edição manual"
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5],
        metavar="K",
        help="Valor(es) de K para avaliação. Ex: --k 1 3 5 10  (padrão: 5)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        metavar="N",
        help="Número de amostras para o dataset sintético (padrão: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade do dataset sintético (padrão: 42)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="PATH",
        help="Caminho para o dataset JSON (obrigatório no modo 'file')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Arquivo de saída para o dataset exportado (modo 'export')",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Salva relatório de métricas em JSON. Ex: reports/eval.json",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Exibe métricas detalhadas por query",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Nível de log (padrão: WARNING)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    print("\n" + "═" * 55)
    print("  RAG Metrics Evaluator")
    print("═" * 55)
    print(f"  Modo     : {args.mode}")
    print(f"  K        : {args.k}")
    if args.mode != "file":
        print(f"  Amostras : {args.samples}  |  Seed: {args.seed}")
    print("═" * 55)

    # Carrega todos os componentes
    vs, embed_model, retriever = load_components(settings)
    print(f"  Índice carregado: {vs.size} chunks disponíveis.\n")

    if args.mode == "export":
        mode_export(args, vs)
    else:
        mode_evaluate(args, vs, retriever)


if __name__ == "__main__":
    main()

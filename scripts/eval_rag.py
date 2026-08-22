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
  # 1. Dataset JSON anotado (padrão: varre K=5, 10 e 15 automaticamente):
  python scripts/eval_rag.py --mode file --dataset data/eval/dataset.json

  # 2. Dataset sintético (smoke-test rápido, sem anotação manual):
  python scripts/eval_rag.py --mode synthetic --k 5 --samples 50

  # 3. Criar dataset sintético e salvar para edição manual:
  python scripts/eval_rag.py --mode export --samples 100 --output data/eval/eval_dataset.json

  # 4. Sobrescrever os valores de K:
  python scripts/eval_rag.py --mode file --dataset data/eval/dataset.json --k 1 3 5 10

  # 5. Exportar relatório agregado em JSON (caminho arbitrário):
  python scripts/eval_rag.py --mode file --dataset data/eval/dataset.json --output-report reports/eval_result.json

Queries negativas e guardrail
──────────────────────────────
  Itens do dataset com "answerable": false (ex: perguntas fora do domínio)
  nunca entram no cálculo de Precision/Recall/F1/MRR/NDCG/MAP — eles são
  avaliados à parte como um guardrail: "passou" quando o retriever devolveu
  0 chunks para aquela query naquele K. Isso testa só o Retriever (filtros de
  score/keyword), não a cadeia completa de guardrails do RAGPipeline.

Exportação automática em data/metrics/retrieval/
────────────────────────────────────────
  No modo 'file', a cada K avaliado o script grava em data/metrics/retrieval/
  (use --metrics-dir para mudar, --no-metrics para desativar):
    details_k{K}.json    — por query: id, query, k, chunks esperados,
                            chunks retornados NA ORDEM retornada, métricas.
    guardrail_k{K}.json  — por query negativa: retrieved_count, passou ou não.
    summary.json         — métricas agregadas de todos os K's + pass-rate
                            do guardrail, sobrescrito a cada execução.

Formato do dataset JSON anotado
─────────────────────────────────
  Schema legado (binário):
  [
    {
      "query": "O que é regressão linear?",
      "relevant_chunk_ids": ["abc123", "def456"],
      "relevant_texts": ["trecho do livro sobre regressão..."]
    },
    ...
  ]

  Schema curado (rag/evaluation.py, o gerado por generate_eval_dataset.py):
  [
    {
      "id": "q0001", "query": "...", "category": "factual",
      "answerable": true,
      "relevant_chunks": [{"chunk_id": "abc123", "relevance": 3}]
    },
    ...
  ]

  Dica: use --mode export para gerar o esqueleto do schema legado, ou
  scripts/generate_eval_dataset.py para gerar o schema curado a partir do
  índice real.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Garante import relativo ao root do projeto
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings, setup_logging  # noqa: E402
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
from scripts._eval_common import (  # noqa: E402
    load_retriever_from_settings,
    next_metrics_version,
)

logger = logging.getLogger(__name__)


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
# Guardrail de queries negativas (fora do domínio)
# ---------------------------------------------------------------------------


def evaluate_negative_guardrail(
    retriever: Retriever, negative_items: list[dict], k: int
) -> dict:
    """Avalia queries negativas (fora do domínio) contra o retriever.

    Diferente de evaluate_retriever/evaluate_query, isto não é uma métrica de
    ranking — queries negativas têm relevant_chunks vazio por definição, então
    Precision/Recall/NDCG/etc. seriam sempre 0 e só distorceriam a média das
    queries de verdade. Aqui "passou" significa que o retriever devolveu uma
    lista vazia para essa query neste K, isto é, os filtros de score/keyword
    corretamente rejeitaram todos os candidatos. Testa só o Retriever
    isoladamente — não a cadeia completa de guardrails do RAGPipeline
    (checagem de relevância tópica, prompt, validador temporal).
    """
    details = []
    passed = 0
    for item in negative_items:
        query = item.get("query", "")
        if not query:
            continue
        t0 = time.perf_counter()
        retrieved = retriever.retrieve(query, k=k)
        latency_ms = (time.perf_counter() - t0) * 1000
        ok = len(retrieved) == 0
        if ok:
            passed += 1
        details.append(
            {
                "id": item.get("id", ""),
                "query": query,
                "k": k,
                "retrieved_count": len(retrieved),
                "top_score": round(retrieved[0]["score"], 4) if retrieved else None,
                "passed": ok,
                "latency_ms": round(latency_ms, 2),
            }
        )
    n = len(details)
    return {
        "k": k,
        "n": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "details": details,
    }


def _print_guardrail(result: dict) -> None:
    print(
        f"  K={result['k']:<3} {result['passed']}/{result['n']} queries negativas "
        f"sem retorno ({result['pass_rate'] * 100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Exportação de métricas detalhadas (data/metrics/retrieval/)
# ---------------------------------------------------------------------------

_TEXT_PREVIEW_CHARS = 200


def _text_preview(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= _TEXT_PREVIEW_CHARS:
        return text
    return text[:_TEXT_PREVIEW_CHARS] + "…"


def _build_expected_chunks(item: dict, chunk_by_id: dict[str, dict]) -> list[dict]:
    expected = []
    for entry in item.get("relevant_chunks", []):
        cid = entry.get("chunk_id", "")
        chunk = chunk_by_id.get(cid, {})
        expected.append(
            {
                "chunk_id": cid,
                "relevance": entry.get("relevance", 3),
                "text_preview": _text_preview(chunk.get("text", "")),
                "source": chunk.get("source", ""),
                "page": chunk.get("page"),
            }
        )
    return expected


def _build_retrieved_chunks(
    retrieved: list[dict], relevant_ids: set[str]
) -> list[dict]:
    out = []
    for rank, chunk in enumerate(retrieved, start=1):
        cid = chunk.get("chunk_id", "")
        out.append(
            {
                "rank": rank,
                "chunk_id": cid,
                "score": round(chunk.get("score", 0.0), 4),
                "confidence": chunk.get("confidence", ""),
                "text_preview": _text_preview(chunk.get("text", "")),
                "source": chunk.get("source", ""),
                "page": chunk.get("page"),
                "is_relevant": cid in relevant_ids,
            }
        )
    return out


def _save_query_details(
    report: EvaluationReport,
    dataset_by_query: dict[str, dict],
    chunk_by_id: dict[str, dict],
    metrics_dir: Path,
) -> None:
    """Salva, por query, os chunks esperados e os retornados NA ORDEM retornada."""
    entries = []
    for qr in report.per_query:
        item = dataset_by_query.get(qr.query, {})
        entries.append(
            {
                "id": item.get("id", ""),
                "query": qr.query,
                "category": qr.category,
                "k": report.k,
                "expected_chunks": _build_expected_chunks(item, chunk_by_id),
                "retrieved_chunks": _build_retrieved_chunks(
                    qr.retrieved, qr.relevant_ids
                ),
                "metrics": {
                    "precision_at_k": round(qr.precision_at_k, 4),
                    "recall_at_k": round(qr.recall_at_k, 4),
                    "f1_at_k": round(qr.f1_at_k, 4),
                    "hit_rate": qr.hit_rate,
                    "reciprocal_rank": round(qr.reciprocal_rank, 4),
                    "ndcg_at_k": round(qr.ndcg_at_k, 4),
                    "ap_at_k": round(qr.ap_at_k, 4),
                    "latency_ms": round(qr.latency_ms, 2),
                },
            }
        )
    path = metrics_dir / f"details_k{report.k}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Detalhes K={report.k} salvos em: {path}")


def _save_guardrail_details(guardrail: dict, metrics_dir: Path) -> None:
    path = metrics_dir / f"guardrail_k{guardrail['k']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(guardrail["details"], f, ensure_ascii=False, indent=2)
    print(f"  [OK] Guardrail K={guardrail['k']} salvo em: {path}")


def _save_metrics_summary(
    reports: dict[str, EvaluationReport],
    guardrails: dict[str, dict],
    metrics_dir: Path,
) -> None:
    summary: dict = {"positive": {label: r.to_dict() for label, r in reports.items()}}
    if guardrails:
        summary["negative_guardrail"] = {
            label: {k: v for k, v in g.items() if k != "details"}
            for label, g in guardrails.items()
        }
    path = metrics_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Resumo salvo em: {path}")


# ---------------------------------------------------------------------------
# Versionamento de data/metrics/retrieval/ — cada execução vira uma pasta vN/ nova, para
# nunca sobrescrever uma rodada anterior e poder comparar mudanças de config.
# (next_metrics_version vive em scripts/_eval_common.py)
# ---------------------------------------------------------------------------


def _migrate_legacy_metrics(metrics_dir: Path) -> None:
    """Move uma rodada solta (de antes do versionamento existir) para vN=1/.

    Não apaga nada — só reorganiza os arquivos já existentes direto em
    metrics_dir (details_k*.json, guardrail_k*.json, summary.json) para
    v1/, deixando a estrutura consistente com as próximas versões.
    """
    if not metrics_dir.exists():
        return
    loose = list(metrics_dir.glob("details_k*.json")) + list(
        metrics_dir.glob("guardrail_k*.json")
    )
    summary_path = metrics_dir / "summary.json"
    if summary_path.exists():
        loose.append(summary_path)
    if not loose:
        return
    v1_dir = metrics_dir / "v1"
    if v1_dir.exists():
        return
    v1_dir.mkdir(parents=True, exist_ok=True)
    for f in loose:
        f.rename(v1_dir / f.name)
    print(f"  [OK] Rodada anterior (pré-versionamento) movida para: {v1_dir}")


def _save_run_meta(
    version: int,
    dataset_path: str,
    k_values: list[int],
    settings,
    metrics_dir: Path,
) -> None:
    """Registra config + dataset usados nesta versão, para saber depois o que
    mudou de uma vN para outra só olhando os números de summary.json."""
    meta = {
        "version": f"v{version}",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_path,
        "k_values": k_values,
        "config": {
            "top_k": settings.top_k,
            "candidate_multiplier": settings.candidate_multiplier,
            "min_score": settings.min_score,
            "lexical_weight": settings.lexical_weight,
            "adaptive_sigma": settings.adaptive_sigma,
            "gap_filter_enabled": settings.gap_filter_enabled,
            "keyword_filter_enabled": settings.keyword_filter_enabled,
            "cross_encoder_enabled": settings.cross_encoder_enabled,
            "cross_encoder_model": settings.cross_encoder_model,
            "cross_encoder_top_k": settings.cross_encoder_top_k,
            "cross_encoder_hybrid_weight": settings.cross_encoder_hybrid_weight,
            "cross_encoder_min_score": settings.cross_encoder_min_score,
        },
    }
    path = metrics_dir / "run_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Metadados da rodada salvos em: {path}")


# ---------------------------------------------------------------------------
# Modo: avaliação (synthetic ou file)
# ---------------------------------------------------------------------------


def mode_evaluate(args, vs: VectorStore, retriever: Retriever, settings) -> None:
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

    # Queries negativas (answerable=False) não têm ground-truth de chunks —
    # tratadas à parte via guardrail, nunca entram nas métricas de ranking.
    positive_items = [d for d in dataset if d.get("answerable", True)]
    negative_items = [d for d in dataset if not d.get("answerable", True)]

    print(
        f"       {len(dataset)} queries carregadas "
        f"({len(positive_items)} com ground-truth, {len(negative_items)} negativas)."
    )
    if not positive_items:
        print("[ERRO] Nenhuma query com ground-truth — nada a avaliar.")
        sys.exit(1)

    k_values: list[int] = args.k

    # ── Avaliação para cada valor de K ──────────────────────────────────────
    reports: dict[str, EvaluationReport] = {}
    guardrails: dict[str, dict] = {}

    for k in k_values:
        print(
            f"\n[EVAL] Avaliando K={k} sobre {len(positive_items)} queries...",
            end=" ",
            flush=True,
        )
        t0 = time.perf_counter()
        report = evaluate_retriever(retriever, positive_items, k=k)
        elapsed = time.perf_counter() - t0
        print(f"concluído em {elapsed:.1f}s")
        reports[f"K={k}"] = report

        if negative_items:
            guardrails[f"K={k}"] = evaluate_negative_guardrail(
                retriever, negative_items, k=k
            )

    # ── Exibição ─────────────────────────────────────────────────────────────
    if len(k_values) == 1:
        print_report(reports[f"K={k_values[0]}"], title="RAG Evaluation")
    else:
        for label, report in reports.items():
            print_report(report, title=f"RAG Evaluation — {label}")
        compare_reports(reports, title="Comparação por K")

    if guardrails:
        print(f"\n{'═' * 55}")
        print("  Guardrail — queries negativas (fora do domínio)")
        print(f"{'═' * 55}")
        for guardrail in guardrails.values():
            _print_guardrail(guardrail)
        print(f"{'═' * 55}\n")

    # ── Detalhe por query (se solicitado) ────────────────────────────────────
    if args.verbose and k_values:
        _print_per_query(reports[f"K={k_values[-1]}"])

    # ── Exportar relatório agregado em JSON (caminho arbitrário, manual) ─────
    if args.output_report:
        _save_report(reports, args.output_report)

    # ── Exportar métricas detalhadas em data/metrics/retrieval/vN/ (automático, modo file,
    #    nunca sobrescreve uma versão anterior) ────────────────────────────────
    if args.mode == "file" and not args.no_metrics:
        base_metrics_dir = ROOT / args.metrics_dir
        _migrate_legacy_metrics(base_metrics_dir)
        version = next_metrics_version(base_metrics_dir)
        metrics_dir = base_metrics_dir / f"v{version}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        chunk_by_id = {c["chunk_id"]: c for c in vs.metadata if c.get("chunk_id")}
        dataset_by_query = {item["query"]: item for item in positive_items}

        print(f"\n[METRICS] v{version} em: {metrics_dir}")
        for report in reports.values():
            _save_query_details(report, dataset_by_query, chunk_by_id, metrics_dir)
        for guardrail in guardrails.values():
            _save_guardrail_details(guardrail, metrics_dir)
        _save_metrics_summary(reports, guardrails, metrics_dir)
        _save_run_meta(version, args.dataset, k_values, settings, metrics_dir)


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
        default=[5, 10, 15],
        metavar="K",
        help="Valor(es) de K para avaliação. Ex: --k 1 3 5 10  (padrão: 5 10 15)",
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
        "--metrics-dir",
        type=str,
        default="data/metrics/retrieval",
        metavar="PATH",
        help=(
            "Diretório (relativo à raiz do projeto) onde salvar métricas "
            "detalhadas por query no modo 'file' (padrão: data/metrics/retrieval)"
        ),
    )
    parser.add_argument(
        "--no-metrics",
        dest="no_metrics",
        action="store_true",
        help="Desativa a exportação automática para --metrics-dir no modo 'file'",
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

    # Carrega todos os componentes — mesma construção usada em produção
    # (api/main.py), incluindo o cross-encoder quando CROSS_ENCODER_ENABLED=true.
    vs, embed_model, retriever = load_retriever_from_settings(settings)
    print(f"  Índice carregado: {vs.size} chunks disponíveis.\n")

    if args.mode == "export":
        mode_export(args, vs)
    else:
        mode_evaluate(args, vs, retriever, settings)


if __name__ == "__main__":
    main()

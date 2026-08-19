"""
Avaliação de métricas de GERAÇÃO (RAGAS) dos modelos .gguf candidatos.

Diferente de scripts/eval_rag.py (retrieval — Precision/Recall/NDCG/...),
este script pontua a RESPOSTA gerada por cada modelo candidato a chat, usando
um LLM-juiz local fixo (models/ragas/*.gguf) e as métricas clássicas do RAGAS:
faithfulness, answer_relevancy, context_precision, context_recall e
answer_correctness. Tudo local — o juiz é só mais um modelo GGUF via
llama-cpp-python, nenhuma chamada de API externa.

Pré-requisito (não está em requirements.txt — ferramenta dev-only, mesmo
padrão de pytest/flake8/black/isort):
  pip install ragas

Modos de uso
────────────
  # 1. Todos os .gguf soltos em models/ (não desce em models/ragas/, onde
  #    fica o juiz):
  python scripts/eval_generation.py --dataset data/eval/benchmark_v1.json

  # 2. Um modelo candidato isolado:
  python scripts/eval_generation.py --dataset data/eval/benchmark_v1.json \\
      --model models/gemma-2-2b-it-q4_k_m.gguf

  # 3. Diretório de modelos customizado + amostra menor (o juiz 27B em CPU é
  #    lento — cada métrica RAGAS dispara várias chamadas ao juiz por item):
  python scripts/eval_generation.py --dataset data/eval/benchmark_v1.json \\
      --models-dir /outro/dir --samples 30

  # 4. Subconjunto de métricas:
  python scripts/eval_generation.py --dataset data/eval/benchmark_v1.json \\
      --metrics faithfulness answer_relevancy

Elegibilidade e exportação
───────────────────────────
  Só itens do dataset com "answerable": true e "reference_answer" preenchido
  entram na avaliação (RAGAS precisa de gabarito para context_recall/
  answer_correctness). Itens elegíveis sem nenhum chunk recuperado são
  pulados e contados à parte — nesse caso o pipeline teria retornado a
  resposta padrão "não está no material", que não faz sentido pontuar.

  A recuperação de contexto é feita UMA VEZ por query (não depende do modelo
  de chat) e reaproveitada entre todos os modelos candidatos avaliados na
  mesma execução. O modelo-juiz também é carregado uma única vez e fica
  residente durante toda a execução; cada modelo candidato é carregado,
  avaliado e liberado antes do próximo.

  Cada execução grava em data/metrics/generation/vN/ (use --metrics-dir para
  mudar, --no-metrics para desativar), nunca sobrescrevendo uma rodada
  anterior:
    details_{modelo}.json  — por query: contexto, resposta, gabarito, scores.
    summary.json           — métricas médias por modelo + breakdown por
                              categoria + config do juiz.
    run_meta.json           — timestamp, dataset, métricas, config do
                              retriever, do juiz e dos modelos candidatos.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings, setup_logging  # noqa: E402
from core.conversation import ConversationManager  # noqa: E402
from llm.local_model import LocalModel  # noqa: E402
from rag.evaluation import load_dataset_from_file  # noqa: E402
from rag.generation_evaluation import (  # noqa: E402
    DEFAULT_METRICS,
    evaluate_generation,
    is_eligible_item,
)
from rag.pipeline import RAGPipeline  # noqa: E402
from scripts._eval_common import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    load_retriever_from_settings,
    next_metrics_version,
    resolve_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolução de modelos candidatos (--model isolado ou --models-dir inteiro)
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Nome de arquivo seguro a partir do nome do modelo (sem extensão)."""
    stem = Path(name).stem
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem)


def _resolve_candidate_models(args) -> list[Path]:
    if args.model:
        model_path = Path(resolve_path(args.model))
        if not model_path.exists():
            print(f"\n[ERRO] Modelo não encontrado: {model_path}\n")
            sys.exit(1)
        return [model_path]

    models_dir = Path(resolve_path(args.models_dir))
    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        print(f"\n[ERRO] Nenhum .gguf encontrado em: {models_dir}\n")
        sys.exit(1)
    return gguf_files


def _make_temp_conversation_manager() -> ConversationManager:
    """ConversationManager descartável — não polui data/conversations.json."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    Path(tmp.name).write_text("[]", encoding="utf-8")
    return ConversationManager(storage_path=tmp.name)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _load_eligible_items(
    dataset_path: str, samples: "int | None", seed: int
) -> list[dict]:
    dataset = load_dataset_from_file(dataset_path)
    eligible = [item for item in dataset if is_eligible_item(item)]
    print(
        f"       {len(dataset)} itens carregados "
        f"({len(eligible)} elegíveis para RAGAS — respondíveis com reference_answer)."
    )
    if not eligible:
        print("[ERRO] Nenhum item elegível para avaliação de geração.")
        sys.exit(1)

    if samples and samples < len(eligible):
        rng = random.Random(seed)
        eligible = rng.sample(eligible, samples)
        print(f"       Amostra de {samples} itens (seed={seed}).")

    # Garante item["id"] único e presente — datasets legados podem não ter id.
    for idx, item in enumerate(eligible):
        if not item.get("id"):
            item["id"] = f"idx{idx}"

    return eligible


# ---------------------------------------------------------------------------
# Exportação de métricas (data/metrics/generation/)
# ---------------------------------------------------------------------------


def _save_details(model_slug: str, details: list[dict], metrics_dir: Path) -> None:
    path = metrics_dir / f"details_{model_slug}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Detalhes salvos em: {path}")


def _save_summary(per_model: dict, judge_info: dict, metrics_dir: Path) -> None:
    summary = {"judge": judge_info, "models": per_model}
    path = metrics_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Resumo salvo em: {path}")


def _save_run_meta(
    version: int,
    dataset_path: str,
    metrics: list[str],
    settings,
    judge_info: dict,
    candidate_models: list[dict],
    metrics_dir: Path,
) -> None:
    meta = {
        "version": f"v{version}",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_path,
        "metrics": metrics,
        "retriever_config": {
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
        "judge": judge_info,
        "candidate_models": candidate_models,
    }
    path = metrics_dir / "run_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Metadados da rodada salvos em: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_generation",
        description=(
            "Avalia a qualidade de geração (RAGAS) dos modelos .gguf candidatos "
            "a modelo de chat, usando um LLM-juiz local fixo."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        metavar="PATH",
        help=f"Dataset JSON anotado (padrão: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="DIR",
        help="Diretório com .gguf candidatos (padrão: models/, glob não-recursivo)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="PATH",
        help="Avalia só este .gguf (sobrepõe --models-dir)",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=None, help="n_ctx dos modelos candidatos"
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="n_threads dos modelos candidatos"
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help="n_gpu_layers dos modelos candidatos",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        metavar="N",
        help="Amostra aleatória de N itens elegíveis (padrão: todos)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=list(DEFAULT_METRICS),
        metavar="METRIC",
        help=f"Métricas RAGAS a rodar (padrão: {' '.join(DEFAULT_METRICS)})",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="data/metrics/generation",
        metavar="PATH",
        help="Diretório (relativo à raiz) onde versionar métricas (padrão: data/metrics/generation)",
    )
    parser.add_argument(
        "--no-metrics",
        dest="no_metrics",
        action="store_true",
        help="Desativa a exportação automática para --metrics-dir",
    )
    parser.add_argument(
        "--ragas-model-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Override de Settings.ragas_model_path (modelo-juiz)",
    )
    parser.add_argument("--ragas-n-ctx", type=int, default=None)
    parser.add_argument("--ragas-n-threads", type=int, default=None)
    parser.add_argument("--ragas-n-gpu-layers", type=int, default=None)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    print("\n" + "═" * 55)
    print("  RAG Generation Metrics Evaluator (RAGAS)")
    print("═" * 55)
    print(f"  Dataset  : {args.dataset}")
    print(f"  Métricas : {', '.join(args.metrics)}")
    print("═" * 55)

    # Recuperação (retriever) — mesma construção usada em produção (api/main.py)
    vs, embed_model, retriever = load_retriever_from_settings(settings)
    print(f"  Índice carregado: {vs.size} chunks disponíveis.\n")

    print("[EVAL] Carregando dataset...")
    items = _load_eligible_items(resolve_path(args.dataset), args.samples, args.seed)

    print(
        "\n[RETRIEVAL] Recuperando contexto (uma vez, compartilhado entre modelos)..."
    )
    contexts_by_id = {item["id"]: retriever.retrieve(item["query"]) for item in items}
    n_with_context = sum(1 for c in contexts_by_id.values() if c)
    print(f"       {n_with_context}/{len(items)} itens com contexto recuperado.")

    candidate_paths = _resolve_candidate_models(args)
    print(f"\n[MODELOS] {len(candidate_paths)} modelo(s) candidato(s):")
    for p in candidate_paths:
        print(f"  - {p.name}")

    # Modelo-juiz — fixo, carregado uma única vez, reaproveitado entre modelos.
    ragas_model_path_setting = args.ragas_model_path or settings.ragas_model_path
    ragas_n_ctx = (
        args.ragas_n_ctx if args.ragas_n_ctx is not None else settings.ragas_n_ctx
    )
    ragas_n_threads = (
        args.ragas_n_threads
        if args.ragas_n_threads is not None
        else settings.ragas_n_threads
    )
    ragas_n_gpu_layers = (
        args.ragas_n_gpu_layers
        if args.ragas_n_gpu_layers is not None
        else settings.ragas_n_gpu_layers
    )
    print(f"\n[JUIZ] Carregando modelo-juiz: {ragas_model_path_setting}")
    judge_llm = LocalModel(
        model_path=resolve_path(ragas_model_path_setting),
        n_ctx=ragas_n_ctx,
        n_threads=ragas_n_threads,
        n_gpu_layers=ragas_n_gpu_layers,
    )
    judge_info = {
        "model_path": ragas_model_path_setting,
        "n_ctx": ragas_n_ctx,
        "n_threads": ragas_n_threads,
        "n_gpu_layers": ragas_n_gpu_layers,
    }

    n_ctx = args.n_ctx if args.n_ctx is not None else settings.n_ctx
    threads = args.threads if args.threads is not None else settings.n_threads
    gpu_layers = (
        args.gpu_layers if args.gpu_layers is not None else settings.n_gpu_layers
    )

    version = None
    metrics_dir = None
    if not args.no_metrics:
        base_metrics_dir = ROOT / args.metrics_dir
        version = next_metrics_version(base_metrics_dir)
        metrics_dir = base_metrics_dir / f"v{version}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[METRICS] v{version} em: {metrics_dir}")

    per_model_summaries: dict[str, dict] = {}
    candidate_meta: list[dict] = []

    for model_path in candidate_paths:
        print(f"\n{'─' * 55}")
        print(f"  Avaliando: {model_path.name}")
        print(f"{'─' * 55}")
        t0 = time.perf_counter()
        try:
            llm = LocalModel(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_threads=threads,
                n_gpu_layers=gpu_layers,
            )
        except Exception as e:
            print(f"  [ERRO] Falha ao carregar modelo: {e}")
            continue

        conv_manager = _make_temp_conversation_manager()
        pipeline = RAGPipeline(
            retriever=retriever,
            llm=llm,
            conv_manager=conv_manager,
            max_context_chars=settings.max_context_chars,
            max_history_turns=settings.max_history_turns,
        )

        slug = _slugify(model_path.name)
        report, details = evaluate_generation(
            items,
            contexts_by_id,
            pipeline,
            judge_llm=judge_llm,
            embed_model=embed_model,
            metrics=args.metrics,
            session_prefix=f"ragas-{slug}",
            model_name=model_path.name,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"  concluído em {elapsed:.1f}s | avaliados={report.n_evaluated} "
            f"| sem contexto={report.n_skipped_no_context}"
        )
        for metric, value in report.mean_scores.items():
            print(f"    {metric:<20} {value:.4f}")

        per_model_summaries[model_path.name] = report.to_dict()
        candidate_meta.append(
            {
                "model": model_path.name,
                "n_ctx": n_ctx,
                "n_threads": threads,
                "n_gpu_layers": gpu_layers,
            }
        )

        if metrics_dir is not None:
            _save_details(slug, details, metrics_dir)

        del llm, pipeline

    if metrics_dir is not None and per_model_summaries:
        _save_summary(per_model_summaries, judge_info, metrics_dir)
        _save_run_meta(
            version,
            args.dataset,
            args.metrics,
            settings,
            judge_info,
            candidate_meta,
            metrics_dir,
        )
    elif metrics_dir is not None:
        print("\n[AVISO] Nenhum modelo avaliado com sucesso — nada exportado.")


if __name__ == "__main__":
    main()

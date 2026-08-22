"""
Script de avaliação da etapa de GERAÇÃO do pipeline RAG.

Métricas avaliadas
───────────────────
  Faithfulness      — fração de afirmações da resposta sustentadas pelo contexto.
  Answer Relevancy  — similaridade entre a resposta e a pergunta original.

Ambas calculadas localmente via um modelo "juiz" GGUF (models/judge/*.gguf) e o
mesmo modelo de embedding usado pelo RAG — sem depender da lib RAGAS/LangChain,
para caber em hardware restrito (Raspberry Pi 5). Ver rag/generation_evaluation.py
para os prompts do juiz e a lógica de parsing.

O contexto de cada pergunta é FIXO, vindo do dataset (data/eval/dataset_generation.json)
— este script nunca chama o Retriever real, para isolar a qualidade de geração da
qualidade de retrieval (essa é avaliada separadamente por scripts/eval_rag.py).

Modos de uso
────────────
  # 1. Roda todos os modelos .gguf de models/ contra o dataset completo:
  python scripts/eval_generation.py

  # 2. Testa só um modelo:
  python scripts/eval_generation.py --model models/qwen1_5-1_8b-chat-q5_k_m.gguf

  # 3. Iteração rápida (primeiras 5 perguntas), recomendado antes de um sweep completo:
  python scripts/eval_generation.py --model models/qwen1_5-1_8b-chat-q5_k_m.gguf --limit 5 -v

Modelo juiz
───────────
  Deve haver exatamente 1 arquivo .gguf em models/judge/ — usado para julgar
  as respostas de TODOS os modelos candidatos nesta execução, garantindo que
  os scores sejam comparáveis entre eles. Recomenda-se um modelo instruct de
  pelo menos ~2B parâmetros para o juiz — modelos menores tendem a ter mais
  falhas de parsing nas respostas estruturadas exigidas.

Custo em hardware restrito
───────────────────────────
  Um sweep completo (todos os modelos de models/) roda, para cada modelo
  candidato, 1 chamada de geração + até 2 chamadas de juiz para faithfulness
  + até 2 para answer relevancy, por pergunta do dataset — tudo em CPU. Use
  --limit e --model para iterar rápido antes de rodar o sweep completo.
  A RAM de pico é o juiz + 1 candidato por vez (nunca todos os candidatos
  carregados simultaneamente).

Exportação automática em data/metrics/generation/
──────────────────────────────────────────────────
  A cada execução (sweep completo ou --model único) é criada uma nova pasta
  vN/ (use --metrics-dir para mudar, --no-metrics para desativar):
    summary.json              — métricas agregadas por modelo candidato.
    details_{model}.json      — por query: resposta, afirmações, veredito, etc.
    run_meta.json              — config completa da rodada (juiz, candidatos, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Garante import relativo ao root do projeto
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings, setup_logging  # noqa: E402
from llm.local_model import LocalModel  # noqa: E402
from rag.embeddings import EmbeddingModel  # noqa: E402
from rag.generation_evaluation import (  # noqa: E402
    _JUDGE_MAX_TOKENS_FAITHFULNESS,
    _JUDGE_MAX_TOKENS_RELEVANCY,
    _JUDGE_TEMPERATURE_FIRST,
    _JUDGE_TEMPERATURE_RETRY,
    GenerationReport,
    evaluate_generation,
    load_generation_dataset_from_file,
    print_generation_report,
)
from scripts._eval_common import resolve_path  # noqa: E402

logger = logging.getLogger(__name__)

_VERSION_DIR_RE = re.compile(r"^v(\d+)$")


def next_metrics_version(metrics_dir: Path) -> int:
    """Determina o próximo número de versão disponível em metrics_dir/vN/.

    Cópia local da mesma lógica usada por scripts/eval_rag.py (retrieval) —
    duplicada aqui em vez de importada de scripts/_eval_common.py porque a
    versão instalada dessa função ainda não está commitada no branch base
    deste worktree.
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


# ---------------------------------------------------------------------------
# Resolução de modelos
# ---------------------------------------------------------------------------


def resolve_candidate_models(args: argparse.Namespace, models_dir: Path) -> list[Path]:
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.exists():
            print(f"[ERRO] Modelo não encontrado: {model_path}")
            sys.exit(1)
        return [model_path]

    candidates = sorted(models_dir.glob("*.gguf"))
    if not candidates:
        print(f"[ERRO] Nenhum arquivo .gguf encontrado em {models_dir}")
        sys.exit(1)
    return candidates


def resolve_judge_model(judge_dir: Path) -> Path:
    candidates = sorted(judge_dir.glob("*.gguf"))
    if len(candidates) != 1:
        print(
            f"[ERRO] {judge_dir} deve conter exatamente 1 arquivo .gguf "
            f"(encontrado: {len(candidates)}). Coloque o modelo juiz nessa "
            "pasta antes de rodar."
        )
        sys.exit(1)
    return candidates[0]


def make_chat_fn(local_model: LocalModel):
    """Fecha sobre um LocalModel já carregado; kwargs são repassados direto
    para LocalModel.chat() (ex: temperature/max_tokens usados pelo juiz)."""

    def _chat(messages: list[dict], **kwargs) -> str:
        return local_model.chat(messages, stream=False, **kwargs)

    return _chat


# ---------------------------------------------------------------------------
# Execução por modelo candidato
# ---------------------------------------------------------------------------


def run_candidate_model(
    model_path: Path,
    judge_chat_fn,
    embed_fn,
    dataset: list[dict],
    args: argparse.Namespace,
    settings,
) -> GenerationReport:
    n_ctx = args.n_ctx or settings.n_ctx
    n_threads = args.threads or settings.n_threads
    n_gpu_layers = (
        args.gpu_layers if args.gpu_layers is not None else settings.n_gpu_layers
    )

    print(f"\n[MODELO] Carregando candidato: {model_path.name}")
    candidate = LocalModel(
        str(model_path), n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers
    )
    candidate_chat_fn = make_chat_fn(candidate)

    report = evaluate_generation(
        candidate_chat_fn,
        judge_chat_fn,
        embed_fn,
        dataset,
        n_questions=args.n_questions,
        model_name=model_path.stem,
        model_path=str(model_path),
        max_context_chars=settings.max_context_chars,
    )

    # Libera o modelo antes de carregar o próximo candidato — pico de RAM é
    # sempre juiz + 1 candidato, nunca todos os candidatos simultaneamente.
    del candidate
    return report


# ---------------------------------------------------------------------------
# Exportação versionada em data/metrics/generation/vN/
# ---------------------------------------------------------------------------


def save_run_outputs(
    reports: list[GenerationReport],
    version_dir: Path,
    args: argparse.Namespace,
    settings,
    judge_model_path: Path,
    dataset_path: str,
    n_dataset_queries: int,
) -> None:
    version_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "version": version_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_path,
        "n_queries": n_dataset_queries,
        "models": {r.model_name: r.to_dict() for r in reports},
    }
    summary_path = version_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Resumo salvo em: {summary_path}")

    for r in reports:
        details = [qr.to_dict() for qr in r.per_query]
        details_path = version_dir / f"details_{r.model_name}.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        print(f"  [OK] Detalhes de {r.model_name} salvos em: {details_path}")

    meta = {
        "version": version_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_path,
        "n_dataset_queries": n_dataset_queries,
        "limit_applied": args.limit,
        "shuffle_sample": args.shuffle_sample,
        "seed": args.seed,
        "judge_model": str(judge_model_path),
        "judge_config": {
            "n_ctx": args.judge_n_ctx or settings.n_ctx,
            "n_threads": args.judge_threads or settings.n_threads,
            "n_gpu_layers": (
                args.judge_gpu_layers
                if args.judge_gpu_layers is not None
                else settings.n_gpu_layers
            ),
            "temperature_first_attempt": _JUDGE_TEMPERATURE_FIRST,
            "temperature_retry": _JUDGE_TEMPERATURE_RETRY,
            "max_tokens_faithfulness": _JUDGE_MAX_TOKENS_FAITHFULNESS,
            "max_tokens_relevancy": _JUDGE_MAX_TOKENS_RELEVANCY,
        },
        "candidate_models": [r.model_path for r in reports],
        "candidate_config": {
            "n_ctx": args.n_ctx or settings.n_ctx,
            "n_threads": args.threads or settings.n_threads,
            "n_gpu_layers": (
                args.gpu_layers
                if args.gpu_layers is not None
                else settings.n_gpu_layers
            ),
        },
        "n_questions": args.n_questions,
    }
    meta_path = version_dir / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Metadados da rodada salvos em: {meta_path}")


# ---------------------------------------------------------------------------
# Exibição detalhada por query (--verbose)
# ---------------------------------------------------------------------------


def _print_per_query(report: GenerationReport) -> None:
    print(f"\n{'═' * 75}")
    print(f"  Detalhe por query — {report.model_name}")
    print(f"{'═' * 75}")
    print(f"  {'Query':<45} {'Faith':>8} {'Relev':>8}")
    print("  " + "─" * 73)
    for qr in report.per_query:
        q = qr.query[:43] + ".." if len(qr.query) > 45 else qr.query
        f_score = qr.faithfulness.score
        r_score = qr.answer_relevancy.score
        f_str = f"{f_score:.3f}" if f_score is not None else "N/A"
        r_str = f"{r_score:.3f}" if r_score is not None else "N/A"
        print(f"  {q:<45} {f_str:>8} {r_str:>8}")
    print(f"{'═' * 75}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_generation",
        description=(
            "Avalia a qualidade de GERAÇÃO do pipeline RAG "
            "(Faithfulness, Answer Relevancy) — sem RAGAS."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval/dataset_generation.json",
        metavar="PATH",
        help="Dataset de avaliação de geração (padrão: data/eval/dataset_generation.json)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="data/metrics/generation",
        metavar="PATH",
        help="Diretório onde salvar métricas versionadas (padrão: data/metrics/generation)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="PATH",
        help="Testa apenas este modelo (padrão: itera todos os .gguf em --models-dir)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        metavar="PATH",
        help="Diretório com os modelos candidatos (padrão: models)",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=3,
        metavar="N",
        help="Perguntas geradas pelo juiz para Answer Relevancy (padrão: 3)",
    )
    parser.add_argument("--n-ctx", type=int, default=None, help="n_ctx do candidato")
    parser.add_argument(
        "--threads", type=int, default=None, help="n_threads do candidato"
    )
    parser.add_argument(
        "--gpu-layers", type=int, default=None, help="n_gpu_layers do candidato"
    )
    parser.add_argument(
        "--judge-n-ctx", type=int, default=None, help="n_ctx do modelo juiz"
    )
    parser.add_argument(
        "--judge-threads", type=int, default=None, help="n_threads do modelo juiz"
    )
    parser.add_argument(
        "--judge-gpu-layers",
        type=int,
        default=None,
        help="n_gpu_layers do modelo juiz",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Avalia só as N primeiras perguntas do dataset (iteração rápida)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed para --shuffle-sample (padrão: 42)"
    )
    parser.add_argument(
        "--shuffle-sample",
        action="store_true",
        help="Com --limit, sorteia N perguntas em vez de pegar as N primeiras",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Nível de log (padrão: WARNING)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Exibe métricas detalhadas por query",
    )
    parser.add_argument(
        "--no-metrics",
        dest="no_metrics",
        action="store_true",
        help="Desativa a exportação automática para --metrics-dir",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    print("\n" + "═" * 55)
    print("  RAG Generation Metrics Evaluator")
    print("═" * 55)

    judge_path = resolve_judge_model(ROOT / "models" / "judge")
    print(f"  Juiz: {judge_path.name}")
    judge_model = LocalModel(
        str(judge_path),
        n_ctx=args.judge_n_ctx or settings.n_ctx,
        n_threads=args.judge_threads or settings.n_threads,
        n_gpu_layers=(
            args.judge_gpu_layers
            if args.judge_gpu_layers is not None
            else settings.n_gpu_layers
        ),
    )
    judge_chat_fn = make_chat_fn(judge_model)

    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=resolve_path(settings.embed_cache_dir),
        batch_size=settings.embed_batch_size,
        use_disk_cache=settings.embed_disk_cache,
    )
    embed_fn = embed_model.embed

    dataset_path = resolve_path(args.dataset)
    dataset = load_generation_dataset_from_file(dataset_path)
    n_dataset_queries = len(dataset)
    if not dataset:
        print("[ERRO] Dataset vazio — nada a avaliar.")
        sys.exit(1)

    if args.limit is not None:
        if args.shuffle_sample:
            rng = random.Random(args.seed)
            dataset = rng.sample(dataset, min(args.limit, len(dataset)))
        else:
            dataset = dataset[: args.limit]

    print(f"  Dataset: {dataset_path} ({len(dataset)}/{n_dataset_queries} perguntas)")

    candidates = resolve_candidate_models(args, ROOT / args.models_dir)
    print(f"  Modelos candidatos: {', '.join(p.name for p in candidates)}")
    print("═" * 55)

    reports: list[GenerationReport] = []
    for model_path in candidates:
        report = run_candidate_model(
            model_path, judge_chat_fn, embed_fn, dataset, args, settings
        )
        reports.append(report)
        print_generation_report(
            report, title=f"Generation Evaluation — {model_path.stem}"
        )
        if args.verbose:
            _print_per_query(report)

    if not args.no_metrics:
        base_metrics_dir = ROOT / args.metrics_dir
        version = next_metrics_version(base_metrics_dir)
        version_dir = base_metrics_dir / f"v{version}"
        print(f"\n[METRICS] v{version} em: {version_dir}")
        save_run_outputs(
            reports,
            version_dir,
            args,
            settings,
            judge_path,
            args.dataset,
            n_dataset_queries,
        )


if __name__ == "__main__":
    main()

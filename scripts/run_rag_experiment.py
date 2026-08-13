"""
Reproducible RAG experiment runner.

Runs the retriever against a benchmark dataset under a specified configuration
and saves machine-readable results.  Every run is stamped with a timestamp and
the full configuration so results can be compared reliably.

Usage examples
──────────────
  # Baseline (all filters on, default settings):
  python scripts/run_rag_experiment.py \\
      --name baseline \\
      --dataset data/eval/benchmark_v1.json \\
      --output data/eval/results/

  # Ablation: keyword filter disabled:
  python scripts/run_rag_experiment.py \\
      --name no_keyword_filter \\
      --dataset data/eval/benchmark_v1.json \\
      --keyword-filter-enabled false \\
      --output data/eval/results/

  # Ablation: gap filter disabled:
  python scripts/run_rag_experiment.py \\
      --name no_gap_filter \\
      --dataset data/eval/benchmark_v1.json \\
      --gap-filter-enabled false \\
      --output data/eval/results/

  # Compare two saved results:
  python scripts/run_rag_experiment.py \\
      --compare data/eval/results/baseline.json \\
                data/eval/results/no_keyword_filter.json

Ablation flags (all default to Settings/.env — i.e. the real deployed config)
──────────────────────────────────────────────────────────────────────────────
  --gap-filter-enabled          true|false
  --keyword-filter-enabled      true|false
  --min-score                   float
  --lexical-weight              float
  --adaptive-sigma              float
  --top-k                       int
  --candidate-multiplier        int
  --cross-encoder-enabled       true|false
  --cross-encoder-model         str
  --cross-encoder-top-k         int
  --cross-encoder-hybrid-weight float
  --k-eval                      int    (default: 5)  evaluation depth

Queries com "answerable": false (negativas/fora do domínio) nunca entram nas
métricas de ranking — mesmo tratamento de scripts/eval_rag.py, já que elas
não têm relevant_chunks por definição e só distorceriam a média.

Output format
─────────────
  {
    "experiment": "no_keyword_filter",
    "timestamp": "2026-08-08T14:32:01",
    "config": {...},
    "metrics": {
      "recall@1": 0.XX, "recall@3": 0.XX, "recall@5": 0.XX, "recall@10": 0.XX,
      "mrr": 0.XX, "ndcg@5": 0.XX, "hit_rate@5": 0.XX, "precision@5": 0.XX,
      "mean_latency_ms": 0.XX
    },
    "by_category": {...},
    "n_queries": 100,
    "dataset": "benchmark_v1.json"
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings, setup_logging  # noqa: E402
from rag.embeddings import EmbeddingModel  # noqa: E402
from rag.evaluation import (  # noqa: E402
    EvaluationReport,
    evaluate_retriever,
    load_dataset_from_file,
    print_report,
)
from rag.retriever import Retriever  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402
from scripts._eval_common import load_retriever_from_settings, parse_bool  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Components loader — mesma construção usada em produção (api/main.py) e nos
# outros scripts de avaliação, incluindo cross-encoder quando habilitado.
# ---------------------------------------------------------------------------


def load_components(settings, args) -> tuple[VectorStore, EmbeddingModel, Retriever]:
    """Load all RAG components, overriding settings with CLI ablation flags."""
    ce_enabled_arg = getattr(args, "cross_encoder_enabled", None)
    return load_retriever_from_settings(
        settings,
        top_k=getattr(args, "top_k", None),
        min_score=getattr(args, "min_score", None),
        lexical_weight=getattr(args, "lexical_weight", None),
        adaptive_sigma=getattr(args, "adaptive_sigma", None),
        gap_filter_enabled=_parse_bool(
            getattr(args, "gap_filter_enabled", None), settings.gap_filter_enabled
        ),
        keyword_filter_enabled=_parse_bool(
            getattr(args, "keyword_filter_enabled", None),
            settings.keyword_filter_enabled,
        ),
        candidate_multiplier=getattr(args, "candidate_multiplier", None),
        cross_encoder_enabled=(
            _parse_bool(ce_enabled_arg, settings.cross_encoder_enabled)
            if ce_enabled_arg is not None
            else None
        ),
        cross_encoder_model=getattr(args, "cross_encoder_model", None),
        cross_encoder_top_k=getattr(args, "cross_encoder_top_k", None),
        cross_encoder_hybrid_weight=getattr(args, "cross_encoder_hybrid_weight", None),
    )


def _parse_bool(value, default: bool) -> bool:
    return parse_bool(value, default)


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------


def run_experiment(args) -> dict:
    """Run a single experiment and return the result dict."""
    settings = get_settings()

    vs, embed_model, retriever = load_components(settings, args)
    print(f"  Index loaded: {vs.size} chunks\n")

    dataset_path = Path(args.dataset)
    print(f"  Loading dataset: {dataset_path.name} ...", end=" ", flush=True)
    dataset = load_dataset_from_file(str(dataset_path))
    print(f"{len(dataset)} queries")

    # Queries negativas (answerable=False) não têm ground-truth de chunks —
    # nunca entram nas métricas de ranking (mesmo critério de eval_rag.py).
    positive_dataset = [d for d in dataset if d.get("answerable", True)]
    n_negative = len(dataset) - len(positive_dataset)
    if n_negative:
        print(f"  ({n_negative} queries negativas ignoradas nas métricas de ranking)")

    k_eval = getattr(args, "k_eval", 5)

    # Run evaluation at multiple depths for recall curves
    all_k = sorted({1, 3, k_eval, 10})
    reports: dict[int, EvaluationReport] = {}
    for k in all_k:
        t0 = time.perf_counter()
        report = evaluate_retriever(retriever, positive_dataset, k=k)
        elapsed = time.perf_counter() - t0
        reports[k] = report
        print(
            f"  K={k}: Hit Rate={report.hit_rate:.3f}  Recall={report.mean_recall:.3f}  MRR={report.mrr:.3f}  ({elapsed:.1f}s)"
        )

    # Primary report at k_eval
    primary = reports[k_eval]
    print_report(primary, title=f"Experiment: {args.name}  (K={k_eval})")

    # Build result dict
    config_dict = _build_config_dict(args, settings)
    result = {
        "experiment": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_path.name,
        "n_queries": len(positive_dataset),
        "n_negative_queries": n_negative,
        "config": config_dict,
        "metrics": {
            "recall@1": round(reports[1].mean_recall, 4) if 1 in reports else None,
            "recall@3": round(reports[3].mean_recall, 4) if 3 in reports else None,
            f"recall@{k_eval}": round(primary.mean_recall, 4),
            "recall@10": round(reports[10].mean_recall, 4) if 10 in reports else None,
            "mrr": round(primary.mrr, 4),
            f"ndcg@{k_eval}": round(primary.mean_ndcg, 4),
            f"hit_rate@{k_eval}": round(primary.hit_rate, 4),
            f"precision@{k_eval}": round(primary.mean_precision, 4),
            f"map@{k_eval}": round(primary.map_at_k, 4),
            "mean_latency_ms": round(primary.mean_latency_ms, 2),
        },
        "by_category": primary.to_dict().get("by_category", {}),
        "worst_queries": [
            {
                "query": qr.query[:100],
                "category": qr.category,
                "recall": round(qr.recall_at_k, 3),
                "hit_rate": qr.hit_rate,
                "rr": round(qr.reciprocal_rank, 3),
            }
            for qr in sorted(
                primary.per_query, key=lambda r: (r.hit_rate, r.recall_at_k)
            )[:10]
        ],
    }
    return result


def _build_config_dict(args, settings) -> dict:
    top_k = getattr(args, "top_k", None) or settings.top_k
    ce_enabled_arg = getattr(args, "cross_encoder_enabled", None)
    ce_enabled = (
        _parse_bool(ce_enabled_arg, settings.cross_encoder_enabled)
        if ce_enabled_arg is not None
        else settings.cross_encoder_enabled
    )
    return {
        "top_k": top_k,
        "candidate_multiplier": getattr(args, "candidate_multiplier", None)
        or settings.candidate_multiplier,
        "min_score": getattr(args, "min_score", None) or settings.min_score,
        "lexical_weight": getattr(args, "lexical_weight", None)
        or settings.lexical_weight,
        "adaptive_sigma": getattr(args, "adaptive_sigma", None)
        or settings.adaptive_sigma,
        "gap_filter_enabled": _parse_bool(
            getattr(args, "gap_filter_enabled", None), settings.gap_filter_enabled
        ),
        "keyword_filter_enabled": _parse_bool(
            getattr(args, "keyword_filter_enabled", None),
            settings.keyword_filter_enabled,
        ),
        "cross_encoder_enabled": ce_enabled,
        "cross_encoder_model": (
            getattr(args, "cross_encoder_model", None) or settings.cross_encoder_model
            if ce_enabled
            else None
        ),
        "cross_encoder_top_k": (
            getattr(args, "cross_encoder_top_k", None) or settings.cross_encoder_top_k
            if ce_enabled
            else None
        ),
        "cross_encoder_hybrid_weight": (
            getattr(args, "cross_encoder_hybrid_weight", None)
            or settings.cross_encoder_hybrid_weight
            if ce_enabled
            else None
        ),
        "bm25_k1": settings.bm25_k1,
        "bm25_b": settings.bm25_b,
        "embed_model": settings.embed_model_name,
    }


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def compare_results(paths: list[str]) -> None:
    """Print a side-by-side comparison of saved experiment results."""
    loaded = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            loaded.append(json.load(f))

    if not loaded:
        print("No results to compare.")
        return

    names = [r["experiment"] for r in loaded]
    metric_keys = [
        k for k in loaded[0]["metrics"].keys() if loaded[0]["metrics"][k] is not None
    ]

    col_w = 14
    header = f"  {'Metric':<24}" + "".join(f"{n[:col_w]:>{col_w}}" for n in names)
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print("  Experiment Comparison")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    for key in metric_keys:
        vals = [r["metrics"].get(key) for r in loaded]
        valid = [v for v in vals if v is not None]
        best = max(valid) if valid else None
        row = f"  {key:<24}"
        for v in vals:
            if v is None:
                row += f"{'N/A':>{col_w}}"
            else:
                mark = " ★" if v == best and len(valid) > 1 else "  "
                row += f"{v:>{col_w - 2}.4f}{mark}"
        print(row)

    print(f"{'═' * len(header)}\n")

    # Category breakdown if present
    if loaded[0].get("by_category"):
        print("  Per-category Hit Rate:")
        all_cats = sorted({cat for r in loaded for cat in r.get("by_category", {})})
        sub_header = f"  {'Category':<20}" + "".join(
            f"{n[:col_w]:>{col_w}}" for n in names
        )
        print(sub_header)
        print("  " + "─" * (len(sub_header) - 2))
        for cat in all_cats:
            row = f"  {cat:<20}"
            for r in loaded:
                hr = r.get("by_category", {}).get(cat, {}).get("hit_rate")
                if hr is None:
                    row += f"{'N/A':>{col_w}}"
                else:
                    row += f"{hr:>{col_w}.4f}"
            print(row)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_rag_experiment",
        description="Run and compare RAG retrieval experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=False)

    # ── run ──────────────────────────────────────────────────────────────────
    run = sub.add_parser("run", help="Run a single experiment")
    run.add_argument("--name", required=True, help="Experiment identifier")
    run.add_argument(
        "--dataset",
        default="data/eval/benchmark_v1.json",
        help="Path to evaluation dataset JSON",
    )
    run.add_argument(
        "--output",
        default="data/eval/results",
        help="Directory to save result JSON",
    )
    run.add_argument("--k-eval", type=int, default=5, help="Primary evaluation depth")

    # Ablation flags
    abl = run.add_argument_group("ablation")
    abl.add_argument("--top-k", type=int, default=None)
    abl.add_argument("--min-score", type=float, default=None)
    abl.add_argument("--lexical-weight", type=float, default=None)
    abl.add_argument("--adaptive-sigma", type=float, default=None)
    abl.add_argument("--candidate-multiplier", type=int, default=None)
    abl.add_argument(
        "--gap-filter-enabled", default=None, help="true|false (default: from settings)"
    )
    abl.add_argument(
        "--keyword-filter-enabled",
        default=None,
        help="true|false (default: from settings)",
    )
    abl.add_argument(
        "--cross-encoder-enabled",
        default=None,
        help="true|false (default: from settings)",
    )
    abl.add_argument("--cross-encoder-model", default=None, help="HF model name")
    abl.add_argument("--cross-encoder-top-k", type=int, default=None)
    abl.add_argument("--cross-encoder-hybrid-weight", type=float, default=None)

    # ── compare ──────────────────────────────────────────────────────────────
    cmp = sub.add_parser("compare", help="Compare saved experiment results")
    cmp.add_argument("results", nargs="+", help="Paths to result JSON files")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging("WARNING")

    if args.mode == "compare":
        compare_results(args.results)
        return

    if args.mode == "run" or args.mode is None:
        if not hasattr(args, "name"):
            # Called without subcommand — show help
            parser.print_help()
            return

        print("\n" + "═" * 60)
        print(f"  RAG Experiment: {args.name}")
        print("═" * 60)

        result = run_experiment(args)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{args.name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n  Result saved → {out_path}")

        # Print worst queries summary
        worst = result.get("worst_queries", [])
        if worst:
            print(f"\n  Top {len(worst)} worst-performing queries:")
            for i, w in enumerate(worst, 1):
                print(
                    f"    {i:2}. [{w['category']:<14}] hit={w['hit_rate']:.0f}  rr={w['rr']:.3f}  \"{w['query'][:60]}\""
                )


if __name__ == "__main__":
    main()

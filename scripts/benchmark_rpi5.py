"""
Phase 3G — Raspberry Pi 5 Production Benchmark & Readiness Validation.

This script measures retrieval quality, latency, memory, CPU, and stability
on the target hardware. Designed to run on Raspberry Pi 5 (8GB ARM64).

Usage:
    # Full benchmark (both configurations):
    python scripts/benchmark_rpi5.py

    # Hybrid only:
    python scripts/benchmark_rpi5.py --mode hybrid

    # Fusion only:
    python scripts/benchmark_rpi5.py --mode fusion

    # Quick (10 queries instead of full 80):
    python scripts/benchmark_rpi5.py --quick

Output: prints structured report and saves JSON to data/eval/results/
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System info collection
# ---------------------------------------------------------------------------


def get_system_info() -> dict:
    """Collect hardware and software information."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    # Try to get RPi-specific info
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        for line in cpuinfo.splitlines():
            if "Model" in line or "model name" in line:
                info["cpu_model"] = line.split(":")[1].strip()
                break
    except (FileNotFoundError, PermissionError):
        info["cpu_model"] = platform.processor() or "unknown"

    # RAM
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024**3), 2)
        info["available_ram_gb"] = round(mem.available / (1024**3), 2)
    except ImportError:
        info["total_ram_gb"] = "psutil not available"

    # PyTorch
    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["pytorch_version"] = "not installed"

    # sentence-transformers
    try:
        import sentence_transformers

        info["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        info["sentence_transformers_version"] = "not installed"

    # FAISS
    try:
        import faiss

        info["faiss_version"] = getattr(faiss, "__version__", "unknown")
    except ImportError:
        info["faiss_version"] = "not installed"

    return info


def get_process_rss_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback for systems without psutil
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # KB to MB
        except (FileNotFoundError, PermissionError):
            return -1.0
    return -1.0


def get_cpu_temp() -> float:
    """Get CPU temperature (Linux/RPi specific)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, PermissionError, ValueError):
        return -1.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(mode: str, dataset: list[dict], quick: bool = False):
    """Run the full benchmark for a given mode (hybrid/fusion)."""
    from core.config import get_settings, setup_logging
    from rag.embeddings import EmbeddingModel
    from rag.evaluation import _effective_ids_from_item
    from rag.retriever import Retriever
    from rag.vectorstore import VectorStore

    setup_logging("WARNING")
    settings = get_settings()

    results = {
        "mode": mode,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system": get_system_info(),
    }

    # ── Memory: process startup ────────────────────────────────────────────
    gc.collect()
    results["memory"] = {"startup_rss_mb": round(get_process_rss_mb(), 1)}

    # ── Load embedding model ───────────────────────────────────────────────
    print("  Loading embedding model...")
    t0 = time.perf_counter()
    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=(
            str(ROOT / settings.embed_cache_dir)
            if not Path(settings.embed_cache_dir).is_absolute()
            else settings.embed_cache_dir
        ),
        batch_size=settings.embed_batch_size,
        use_disk_cache=settings.embed_disk_cache,
    )
    embed_load_time = time.perf_counter() - t0
    results["load_times"] = {"embedding_model_s": round(embed_load_time, 2)}
    results["memory"]["after_embedding_rss_mb"] = round(get_process_rss_mb(), 1)
    print(
        f"    Loaded in {embed_load_time:.1f}s | RSS={results['memory']['after_embedding_rss_mb']:.0f}MB"
    )

    # ── Load FAISS index ───────────────────────────────────────────────────
    print("  Loading FAISS index...")
    t0 = time.perf_counter()
    index_dir = (
        str(ROOT / settings.index_dir)
        if not Path(settings.index_dir).is_absolute()
        else settings.index_dir
    )
    vs = VectorStore(
        index_dir=index_dir,
        embedding_dim=embed_model.dimension,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    faiss_load_time = time.perf_counter() - t0
    results["load_times"]["faiss_index_s"] = round(faiss_load_time, 2)
    results["memory"]["after_faiss_rss_mb"] = round(get_process_rss_mb(), 1)
    results["index_size"] = vs.size
    print(
        f"    {vs.size} chunks | {faiss_load_time:.1f}s | RSS={results['memory']['after_faiss_rss_mb']:.0f}MB"
    )

    # ── Load CrossEncoder (if fusion mode) ─────────────────────────────────
    cross_encoder = None
    if mode == "fusion":
        from rag.cross_encoder import CrossEncoderReranker

        print("  Loading cross-encoder...")
        t0 = time.perf_counter()
        cross_encoder = CrossEncoderReranker(
            model_name=settings.cross_encoder_model,
            hybrid_weight=settings.cross_encoder_hybrid_weight,
            ce_top_k=settings.cross_encoder_top_k,
            min_score=settings.cross_encoder_min_score,
        )
        # Force load (lazy load triggers on first rerank, but we want to measure)
        cross_encoder._load_model()
        ce_load_time = time.perf_counter() - t0
        results["load_times"]["cross_encoder_s"] = round(ce_load_time, 2)
        results["memory"]["after_ce_rss_mb"] = round(get_process_rss_mb(), 1)
        print(
            f"    Loaded in {ce_load_time:.1f}s | RSS={results['memory']['after_ce_rss_mb']:.0f}MB"
        )

    # ── Build retriever ────────────────────────────────────────────────────
    retriever = Retriever(
        vectorstore=vs,
        embed_model=embed_model,
        top_k=settings.top_k,
        candidate_multiplier=settings.candidate_multiplier,
        min_score=settings.min_score,
        lexical_weight=settings.lexical_weight,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        adaptive_sigma=999.0,
        gap_filter_enabled=False,
        keyword_filter_enabled=True,
        cross_encoder=cross_encoder,
    )

    # ── Configuration snapshot ─────────────────────────────────────────────
    results["config"] = {
        "mode": mode,
        "cross_encoder_enabled": mode == "fusion",
        "cross_encoder_model": (
            settings.cross_encoder_model if mode == "fusion" else None
        ),
        "cross_encoder_top_k": (
            settings.cross_encoder_top_k if mode == "fusion" else None
        ),
        "cross_encoder_hybrid_weight": (
            settings.cross_encoder_hybrid_weight if mode == "fusion" else None
        ),
        "cross_encoder_min_score": (
            settings.cross_encoder_min_score if mode == "fusion" else None
        ),
        "candidate_multiplier": settings.candidate_multiplier,
        "top_k": settings.top_k,
        "lexical_weight": settings.lexical_weight,
        "adaptive_sigma": 999.0,
        "gap_filter_enabled": False,
        "keyword_filter_enabled": True,
    }

    # ── Cold start query ───────────────────────────────────────────────────
    print("\n  Cold-start query...")
    cold_query = dataset[0]["query"]
    t0 = time.perf_counter()
    _ = retriever.retrieve(cold_query, k=5)
    cold_latency = (time.perf_counter() - t0) * 1000
    results["latency"] = {"cold_start_ms": round(cold_latency, 1)}
    print(f"    Cold start: {cold_latency:.0f}ms")

    # ── Warm queries (main benchmark) ──────────────────────────────────────
    n_queries = 10 if quick else len(dataset)
    benchmark_ds = dataset[:n_queries]

    print(f"\n  Running {n_queries} warm queries...")
    latencies = []
    hits = 0
    rr_sum = 0.0
    cats = defaultdict(list)
    sustained = []

    for i, item in enumerate(benchmark_ds):
        query = item["query"]
        expected = _effective_ids_from_item(item)
        cat = item.get("category", "?")

        t0 = time.perf_counter()
        res = retriever.retrieve(query, k=5)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        fids = [c.get("chunk_id", "") for c in res]
        hit = 1.0 if any(f in expected for f in fids) else 0.0
        rr = 0.0
        for j, f in enumerate(fids, 1):
            if f in expected:
                rr = 1.0 / j
                break
        hits += hit
        rr_sum += rr
        cats[cat].append(hit)

        # Sustained-load metrics (every query)
        sustained.append(
            {
                "query_idx": i + 1,
                "latency_ms": round(lat, 1),
                "rss_mb": round(get_process_rss_mb(), 1),
                "cpu_temp": round(get_cpu_temp(), 1),
            }
        )

        if (i + 1) % 10 == 0 or i == 0:
            rss = get_process_rss_mb()
            temp = get_cpu_temp()
            print(
                f"    [{i+1}/{n_queries}] lat={lat:.0f}ms rss={rss:.0f}MB temp={temp:.1f}C"
            )

    # ── Compute metrics ────────────────────────────────────────────────────
    n = len(benchmark_ds)
    sorted_lat = sorted(latencies)

    results["quality"] = {
        "n_queries": n,
        "hit_rate_at_5": round(hits / n, 4),
        "mrr": round(rr_sum / n, 4),
        "by_category": {
            cat: {"n": len(v), "hit_rate": round(sum(v) / len(v), 4)}
            for cat, v in sorted(cats.items())
        },
    }

    results["latency"]["n_queries"] = n
    results["latency"]["mean_ms"] = round(sum(latencies) / n, 1)
    results["latency"]["median_ms"] = round(sorted_lat[n // 2], 1)
    results["latency"]["p50_ms"] = round(sorted_lat[int(n * 0.50)], 1)
    results["latency"]["p95_ms"] = round(sorted_lat[int(n * 0.95)], 1)
    results["latency"]["p99_ms"] = round(sorted_lat[min(int(n * 0.99), n - 1)], 1)
    results["latency"]["min_ms"] = round(sorted_lat[0], 1)
    results["latency"]["max_ms"] = round(sorted_lat[-1], 1)

    results["memory"]["peak_rss_mb"] = round(max(s["rss_mb"] for s in sustained), 1)
    results["sustained_load"] = sustained

    # ── Final summary ──────────────────────────────────────────────────────
    gc.collect()
    results["memory"]["final_rss_mb"] = round(get_process_rss_mb(), 1)

    return results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(results: dict):
    """Print a human-readable report."""
    mode = results["mode"]
    m = results["quality"]
    lat = results["latency"]
    mem = results["memory"]
    sys_info = results["system"]

    print(f"\n{'='*70}")
    print(f"  PHASE 3G BENCHMARK — {mode.upper()}")
    print(f"{'='*70}")

    print("\n  System:")
    print(f"    Platform:   {sys_info.get('platform', '?')}")
    print(f"    Machine:    {sys_info.get('machine', '?')}")
    print(f"    CPU:        {sys_info.get('cpu_model', '?')}")
    print(f"    Cores:      {sys_info.get('cpu_count', '?')}")
    print(f"    RAM:        {sys_info.get('total_ram_gb', '?')} GB")
    print(f"    Python:     {sys_info.get('python_version', '?')}")
    print(f"    PyTorch:    {sys_info.get('pytorch_version', '?')}")
    print(f"    ST:         {sys_info.get('sentence_transformers_version', '?')}")

    print(f"\n  Quality (n={m['n_queries']}):")
    print(f"    HR@5:       {m['hit_rate_at_5']:.4f}")
    print(f"    MRR:        {m['mrr']:.4f}")
    if m.get("by_category"):
        print("    Per-category HR@5:")
        for cat, v in sorted(m["by_category"].items()):
            print(
                f"      {cat:<14} {v['hit_rate']:.3f} ({int(v['hit_rate']*v['n'])}/{v['n']})"
            )

    print("\n  Latency:")
    print(f"    Cold start: {lat['cold_start_ms']:.0f}ms")
    print(f"    Mean:       {lat['mean_ms']:.0f}ms")
    print(f"    Median:     {lat['median_ms']:.0f}ms")
    print(f"    P95:        {lat['p95_ms']:.0f}ms")
    print(f"    P99:        {lat['p99_ms']:.0f}ms")
    print(f"    Min:        {lat['min_ms']:.0f}ms")
    print(f"    Max:        {lat['max_ms']:.0f}ms")

    print("\n  Memory:")
    print(f"    Startup:    {mem.get('startup_rss_mb', '?')} MB")
    print(f"    +Embedding: {mem.get('after_embedding_rss_mb', '?')} MB")
    print(f"    +FAISS:     {mem.get('after_faiss_rss_mb', '?')} MB")
    if mem.get("after_ce_rss_mb"):
        print(f"    +CE:        {mem['after_ce_rss_mb']} MB")
    print(f"    Peak:       {mem.get('peak_rss_mb', '?')} MB")
    print(f"    Final:      {mem.get('final_rss_mb', '?')} MB")

    # Sustained load summary
    sustained = results.get("sustained_load", [])
    if sustained:
        rss_vals = [s["rss_mb"] for s in sustained if s["rss_mb"] > 0]
        temp_vals = [s["cpu_temp"] for s in sustained if s["cpu_temp"] > 0]
        lat_vals = [s["latency_ms"] for s in sustained]
        if rss_vals:
            rss_growth = rss_vals[-1] - rss_vals[0]
            print(f"\n  Sustained load ({len(sustained)} queries):")
            print(f"    RSS growth: {rss_growth:+.1f} MB")
            print(
                f"    Lat trend:  first={lat_vals[0]:.0f}ms last={lat_vals[-1]:.0f}ms"
            )
            if temp_vals:
                print(
                    f"    Temp:       start={temp_vals[0]:.1f}C end={temp_vals[-1]:.1f}C"
                )

    load = results.get("load_times", {})
    if load:
        print("\n  Load times:")
        for k, v in load.items():
            print(f"    {k}: {v:.1f}s")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3G — RPi 5 production benchmark",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "fusion", "both"],
        default="both",
        help="Which configuration to benchmark",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only 10 queries (quick smoke test)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: data/eval/results/phase3g_<mode>.json)",
    )
    args = parser.parse_args()

    from core.config import setup_logging
    from rag.evaluation import load_dataset_from_file

    setup_logging("WARNING")

    dataset_path = ROOT / "data/eval/benchmark_v1_linked.json"
    dataset_full = load_dataset_from_file(str(dataset_path))
    dataset = [
        i
        for i in dataset_full
        if i.get("answerable", True) and i.get("category") != "conversational"
    ]

    print(f"\n{'='*70}")
    print("  PHASE 3G — RASPBERRY PI 5 PRODUCTION BENCHMARK")
    print(f"{'='*70}")
    print(f"  Dataset: {len(dataset)} queries")
    print(f"  Mode:    {args.mode}")
    print(f"  Quick:   {args.quick}")

    modes = ["hybrid", "fusion"] if args.mode == "both" else [args.mode]
    all_results = {}

    for mode in modes:
        print(f"\n{'─'*70}")
        print(f"  BENCHMARKING: {mode.upper()}")
        print(f"{'─'*70}")
        results = run_benchmark(mode, dataset, quick=args.quick)
        all_results[mode] = results
        print_report(results)

        # Save individual result
        out_dir = ROOT / "data/eval/results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"phase3g_{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved → {out_path}")

    # Comparison if both modes
    if len(all_results) == 2:
        h = all_results["hybrid"]
        f = all_results["fusion"]
        print(f"\n{'='*70}")
        print("  COMPARISON: HYBRID vs FUSION")
        print(f"{'='*70}")
        print(f"  {'Metric':<20} {'Hybrid':>12} {'Fusion':>12} {'Delta':>12}")
        print(f"  {'-'*56}")
        comparisons = [
            ("HR@5", h["quality"]["hit_rate_at_5"], f["quality"]["hit_rate_at_5"]),
            ("MRR", h["quality"]["mrr"], f["quality"]["mrr"]),
            ("Mean latency (ms)", h["latency"]["mean_ms"], f["latency"]["mean_ms"]),
            ("P95 latency (ms)", h["latency"]["p95_ms"], f["latency"]["p95_ms"]),
            ("Peak RSS (MB)", h["memory"]["peak_rss_mb"], f["memory"]["peak_rss_mb"]),
        ]
        for name, hv, fv in comparisons:
            delta = fv - hv
            print(f"  {name:<20} {hv:>12.4f} {fv:>12.4f} {delta:>+12.4f}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()

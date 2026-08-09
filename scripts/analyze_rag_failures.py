"""
Per-query RAG failure analysis with stage-by-stage tracing.

For every query in a dataset, traces exactly where expected chunks were lost
in the retrieval pipeline (which filter removed them) and produces a structured
failure report.

Usage examples
──────────────
  # Analyse all failures in the benchmark:
  python scripts/analyze_rag_failures.py \\
      --dataset data/eval/benchmark_v1.json \\
      --output  data/eval/failure_report.json

  # Analyse only queries that failed (recall@5 < 1.0):
  python scripts/analyze_rag_failures.py \\
      --dataset data/eval/benchmark_v1.json \\
      --failures-only

  # Print detailed trace for a specific query index:
  python scripts/analyze_rag_failures.py \\
      --dataset data/eval/benchmark_v1.json \\
      --query-id q042

  # Link benchmark chunk_ids from text matching (run after re-indexing):
  python scripts/analyze_rag_failures.py --link-benchmark \\
      --dataset  data/eval/benchmark_v1.json \\
      --output   data/eval/benchmark_v1_linked.json

Output format per failed query
───────────────────────────────
  {
    "query_id": "q042",
    "query": "...",
    "category": "paraphrase",
    "expected_chunks": ["chunk_id_1"],
    "failure_stage": "AFTER_KEYWORD_FILTER",
    "faiss_rank": 2,
    "stages": {
      "FAISS_SEMANTIC":         {"found": true,  "rank": 2, "score": 0.71},
      "AFTER_ABSOLUTE_FILTER":  {"found": true,  "rank": 2},
      "AFTER_ADAPTIVE_FILTER":  {"found": true,  "rank": 2},
      "AFTER_GAP_FILTER":       {"found": true,  "rank": 2},
      "AFTER_KEYWORD_FILTER":   {"found": false, "rank": null},
      "AFTER_HYBRID_RERANK":    {"found": false, "rank": null}
    },
    "final_results": [{"chunk_id": "...", "score": 0.XX}, ...]
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings, setup_logging  # noqa: E402
from rag.embeddings import EmbeddingModel  # noqa: E402
from rag.evaluation import (  # noqa: E402
    _effective_ids_from_item,
    load_dataset_from_file,
)
from rag.retriever import RetrievalTrace, Retriever  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component loading (same pattern as run_rag_experiment.py)
# ---------------------------------------------------------------------------


def _load_components(settings) -> tuple[VectorStore, EmbeddingModel, Retriever]:
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

    index_dir = (
        str(ROOT / settings.index_dir)
        if not Path(settings.index_dir).is_absolute()
        else settings.index_dir
    )
    if not Path(index_dir).exists():
        print(f"\n[ERROR] Index not found: {index_dir}")
        sys.exit(1)

    vs = VectorStore(
        index_dir=index_dir,
        embedding_dim=embed_model.dimension,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )

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
    )
    return vs, embed_model, retriever


# ---------------------------------------------------------------------------
# Stage-by-stage tracking
# ---------------------------------------------------------------------------


def _find_rank(chunk_ids: set[str], candidates: list[dict]) -> int | None:
    """Return 1-based rank of first expected chunk in candidate list, or None."""
    for i, c in enumerate(candidates, start=1):
        cid = c.get("chunk_id", "") or c.get("text", "")[:64].strip()
        if cid in chunk_ids:
            return i
    return None


def _trace_query(
    retriever: Retriever,
    query: str,
    expected_ids: set[str],
    k: int,
) -> dict:
    """Run retrieve_with_trace and build a structured failure record."""
    results, trace = retriever.retrieve_with_trace(query, k=k)

    if trace is None:
        return {"error": "trace not available"}

    # Per-stage presence info
    stages_info: dict[str, dict] = {}
    failure_stage: str | None = None

    for snap in trace.stages:
        rank = _find_rank(expected_ids, snap.candidates)
        stages_info[snap.stage] = {
            "count": len(snap.candidates),
            "found": rank is not None,
            "rank": rank,
        }
        if rank is None and failure_stage is None and expected_ids:
            # The first stage where expected chunk disappeared
            prev_stages = list(stages_info.keys())[:-1]
            if any(stages_info[s]["found"] for s in prev_stages):
                failure_stage = snap.stage

    # If never found at all, failure is at FAISS level
    if expected_ids and failure_stage is None:
        first = next(iter(stages_info.values()), {})
        if not first.get("found"):
            failure_stage = "FAISS_SEMANTIC"

    final_ids = [
        c.get("chunk_id", "") or c.get("text", "")[:64].strip() for c in results
    ]
    hit = any(fid in expected_ids for fid in final_ids)

    return {
        "failure_stage": failure_stage if not hit else None,
        "hit": hit,
        "stages": stages_info,
        "faiss_rank": stages_info.get("FAISS_SEMANTIC", {}).get("rank"),
        "final_results": [
            {
                "chunk_id": c.get("chunk_id", ""),
                "score": round(c.get("score", 0.0), 4),
                "text_preview": c.get("text", "")[:80].replace("\n", " "),
            }
            for c in results
        ],
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyse(
    retriever: Retriever,
    vs: VectorStore,
    dataset: list[dict],
    k: int,
    failures_only: bool,
    query_id: str | None,
) -> list[dict]:
    """Trace every query and return list of analysis records."""
    records = []
    total = len(dataset)

    for i, item in enumerate(dataset):
        qid = item.get("id", f"q{i+1:03d}")
        if query_id and qid != query_id:
            continue

        query = item.get("query", "")
        if not query:
            continue

        expected_ids = _effective_ids_from_item(item)
        # Also try text matching as fallback
        relevant_texts = item.get("relevant_texts", [])
        if not expected_ids and relevant_texts:
            # Use metadata text-matching to find chunk_ids in the index
            expected_ids = _resolve_ids_from_texts(vs, relevant_texts)

        trace_info = _trace_query(retriever, query, expected_ids, k)
        hit = trace_info.get("hit", False)

        if failures_only and hit:
            continue

        record = {
            "query_id": qid,
            "query": query,
            "category": item.get("category", ""),
            "answerable": item.get("answerable", True),
            "expected_chunk_ids": list(expected_ids),
            "expected_texts_provided": len(relevant_texts),
            **trace_info,
        }
        records.append(record)

        if i % 10 == 0 or query_id:
            status = "HIT" if hit else f"MISS@{trace_info.get('failure_stage', '?')}"
            print(f'  [{i+1:3}/{total}] {qid}  {status:<35}  "{query[:50]}"')

    return records


def _resolve_ids_from_texts(vs: VectorStore, relevant_texts: list[str]) -> set[str]:
    """Attempt to find chunk_ids in the index for the given reference texts.

    Matches by checking if any chunk text starts with the reference text prefix.
    This is a best-effort approximation for benchmarks created before chunk_ids
    were available.
    """
    found: set[str] = set()
    for ref in relevant_texts:
        ref_prefix = ref.strip()[:120]
        for chunk in vs.metadata:
            ctext = chunk.get("text", "").strip()[:120]
            if ctext == ref_prefix or ref_prefix in ctext:
                cid = chunk.get("chunk_id", "")
                if cid:
                    found.add(cid)
    return found


# ---------------------------------------------------------------------------
# Link benchmark (fill chunk_ids from index text matching)
# ---------------------------------------------------------------------------


def link_benchmark(vs: VectorStore, dataset: list[dict]) -> list[dict]:
    """Populate relevant_chunks[].chunk_id by matching relevant_texts against index.

    Writes the matched chunk_id back into the dataset items so future evaluation
    can use ID-based matching (more reliable than text-prefix matching).
    """
    updated = 0
    for item in dataset:
        if item.get("relevant_chunks"):
            # Already linked
            continue
        relevant_texts = item.get("relevant_texts", [])
        if not relevant_texts:
            continue

        matched_ids = _resolve_ids_from_texts(vs, relevant_texts)
        if matched_ids:
            item["relevant_chunks"] = [
                {"chunk_id": cid, "relevance": 3} for cid in sorted(matched_ids)
            ]
            updated += 1

    print(f"  Linked {updated}/{len(dataset)} items with chunk_ids from index.")
    return dataset


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_failure_summary(records: list[dict]) -> None:
    """Print aggregate failure statistics."""
    total = len(records)
    if total == 0:
        print("  No records to summarise.")
        return

    hits = sum(1 for r in records if r.get("hit"))
    misses = total - hits

    print(f"\n{'═' * 65}")
    print(f"  Failure Analysis Summary")
    print(f"{'═' * 65}")
    print(f"  Queries analysed : {total}")
    print(f"  Hits             : {hits}  ({hits/total*100:.1f}%)")
    print(f"  Misses           : {misses}  ({misses/total*100:.1f}%)")

    # Failure stage distribution
    from collections import Counter

    stage_counts: Counter = Counter()
    for r in records:
        if not r.get("hit") and r.get("answerable", True):
            fs = r.get("failure_stage") or "NO_EXPECTED_IDS"
            stage_counts[fs] += 1

    if stage_counts:
        print(f"\n  Failure stage breakdown (answerable queries only):")
        for stage, count in stage_counts.most_common():
            bar = "█" * count
            print(f"    {stage:<35} {count:>3}  {bar}")

    # Category breakdown
    from collections import defaultdict

    cat_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "hits": 0})
    for r in records:
        cat = r.get("category") or "uncategorized"
        cat_stats[cat]["total"] += 1
        if r.get("hit"):
            cat_stats[cat]["hits"] += 1

    print(f"\n  Hit rate by category:")
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        hr = s["hits"] / s["total"] if s["total"] else 0
        bar = "█" * int(hr * 20)
        print(f"    {cat:<16} {hr:.2f}  {bar}  ({s['hits']}/{s['total']})")

    # FAISS rank distribution (where in FAISS results was the expected chunk?)
    faiss_ranks = [
        r.get("faiss_rank") for r in records if r.get("faiss_rank") is not None
    ]
    if faiss_ranks:
        found_in_faiss = len(faiss_ranks)
        top1 = sum(1 for r in faiss_ranks if r == 1)
        top5 = sum(1 for r in faiss_ranks if r <= 5)
        top10 = sum(1 for r in faiss_ranks if r <= 10)
        print(f"\n  FAISS rank of expected chunk (when found in semantic results):")
        print(
            f"    Rank 1     : {top1}/{found_in_faiss} ({top1/found_in_faiss*100:.1f}%)"
        )
        print(
            f"    Rank ≤5    : {top5}/{found_in_faiss} ({top5/found_in_faiss*100:.1f}%)"
        )
        print(
            f"    Rank ≤10   : {top10}/{found_in_faiss} ({top10/found_in_faiss*100:.1f}%)"
        )

    print(f"{'═' * 65}\n")


def print_query_trace(record: dict) -> None:
    """Print a human-readable trace for a single query."""
    print(f"\n{'─' * 70}")
    print(f"  Query ID  : {record.get('query_id', '?')}")
    print(f"  Category  : {record.get('category', '?')}")
    print(f"  Answerable: {record.get('answerable', True)}")
    print(f"  Query     : {record.get('query', '')}")
    print(f"  Expected  : {record.get('expected_chunk_ids', [])}")
    print(f"  Hit       : {record.get('hit', False)}")
    if record.get("failure_stage"):
        print(f"  FAILED AT : *** {record['failure_stage']} ***")
    print()

    for stage, info in record.get("stages", {}).items():
        found = info.get("found", False)
        rank = info.get("rank")
        count = info.get("count", 0)
        mark = "✓" if found else "✗"
        rank_str = f"rank={rank}" if rank else "not found"
        print(f"  [{mark}] {stage:<35} {count:>3} candidates  {rank_str}")

    print(f"\n  Final results ({len(record.get('final_results', []))}):")
    for fr in record.get("final_results", []):
        print(
            f"    {fr['chunk_id']:<20} score={fr['score']:.4f}  \"{fr['text_preview'][:50]}\""
        )
    print(f"{'─' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_rag_failures",
        description="Per-query RAG failure analysis with stage tracing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        default="data/eval/benchmark_v1.json",
        help="Evaluation dataset JSON path",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Save full report to this JSON file",
    )
    p.add_argument(
        "--failures-only",
        action="store_true",
        help="Only trace queries where retrieval failed",
    )
    p.add_argument(
        "--query-id",
        default=None,
        help="Trace only the query with this ID (e.g. q042)",
    )
    p.add_argument(
        "--k",
        type=int,
        default=5,
        help="Retrieval depth for analysis (default: 5)",
    )
    p.add_argument(
        "--link-benchmark",
        action="store_true",
        help="Populate relevant_chunks[].chunk_id by text-matching against current index",
    )
    p.add_argument(
        "--linked-output",
        default=None,
        help="Where to write linked benchmark (default: overwrites --dataset)",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    settings = get_settings()

    print(f"\n{'═' * 65}")
    print(f"  RAG Failure Analyser")
    print(f"{'═' * 65}")

    vs, embed_model, retriever = _load_components(settings)
    print(f"  Index loaded: {vs.size} chunks")

    dataset = load_dataset_from_file(
        str(ROOT / args.dataset)
        if not Path(args.dataset).is_absolute()
        else args.dataset
    )
    print(f"  Dataset: {len(dataset)} queries from {Path(args.dataset).name}\n")

    if args.link_benchmark:
        linked = link_benchmark(vs, dataset)
        out_path = args.linked_output or args.dataset
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(linked, f, ensure_ascii=False, indent=2)
        print(f"  Linked dataset saved → {out_path}")
        return

    records = analyse(
        retriever=retriever,
        vs=vs,
        dataset=dataset,
        k=args.k,
        failures_only=args.failures_only,
        query_id=args.query_id,
    )

    print_failure_summary(records)

    if args.query_id:
        for r in records:
            print_query_trace(r)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  Full report saved → {out}")


if __name__ == "__main__":
    main()

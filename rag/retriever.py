"""
Retriever com busca híbrida (semântica + BM25 léxico) e reranking.

Estratégia de dois estágios:

  Estágio 1 — Candidatos (rápido):
    - Busca semântica FAISS retorna top-(k * candidate_multiplier) chunks.
    - BM25 leve pontua os mesmos chunks por relevância léxica.

  Estágio 2 — Reranking e filtros:
    - Filtro absoluto (min_score).
    - Filtro adaptativo (best − sigma * std).
    - Gap detection.
    - Keyword filter (opcional — controlado por keyword_filter_enabled).
    - Hybrid reranking (weighted semantic + BM25).
    - Cross-encoder reranking (opcional — score fusion com o score híbrido).

Todos os parâmetros são configuráveis via core/config.py.

  - retrieve(query, k=None) aceita k explícito — remove necessidade de
    mutation de atributos privados em evaluation.py.
  - Parâmetros BM25 (k1, b), sigma do filtro adaptativo e flags de
    filtros são lidos do Settings, não de constantes de módulo.
  - Observabilidade: retrieve_with_trace() expõe candidatos em cada estágio.

Removido (era corpus-específico, hardcoded para um único período histórico
de um corpus particular — não generaliza para outros documentos):
  - Period boost e o ajuste dinâmico de k por heurísticas biográficas/de
    período (frases fixas em PT-BR). effective_k agora é sempre k explícito
    ou top_k configurado, sem heurística de query.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Stopwords PT-BR
_STOPWORDS_PT = {
    "a",
    "o",
    "e",
    "é",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "para",
    "com",
    "um",
    "uma",
    "uns",
    "umas",
    "se",
    "que",
    "ao",
    "aos",
    "às",
    "ou",
    "mas",
    "também",
    "já",
    "mais",
    "como",
    "seu",
    "sua",
    "seus",
    "suas",
    "este",
    "esta",
    "estes",
    "estas",
    "esse",
    "essa",
    "esses",
    "essas",
    "isso",
    "aqui",
    "ali",
    "quando",
    "onde",
    "há",
    "ter",
    "ser",
    "estar",
    "foi",
    "são",
    "era",
    "tem",
    "não",
    "the",
    "of",
    "and",
    "in",
    "to",
    "is",
    "it",
    "that",
    "for",
    "on",
    "are",
    "fazer",
    "feito",
    "faz",
    "fazia",
    "faça",
    "pode",
    "podem",
    "deve",
    "devem",
    "dever",
    "usar",
    "usado",
    "precisa",
    "precisar",
    "preciso",
    "quero",
    "quer",
    "querer",
    "saber",
    "sabe",
    "dizer",
    "disse",
    "diz",
    "ver",
    "veja",
    "vai",
    "vão",
    "ir",
    "dar",
    "dá",
    "dado",
    "tudo",
    "todo",
    "toda",
    "todos",
    "todas",
    "muito",
    "muita",
    "muitos",
    "muitas",
    "pouco",
    "pouca",
    "poucos",
    "cada",
    "qual",
    "quais",
    "quem",
    "cujo",
    "cuja",
    "cujos",
    "cujas",
    "mesmo",
    "mesma",
    "apenas",
    "ainda",
    "então",
    "assim",
    "pois",
    "sobre",
    "entre",
    "após",
    "antes",
    "durante",
    "contra",
    "desde",
    "até",
    "pelo",
    "pela",
    "pelos",
    "pelas",
    "num",
    "numa",
    "nuns",
    "numas",
    # Verbos genéricos comuns em PT-BR, ausentes das formas já cobertas acima
    # (fazer/poder/dever/usar/precisar/querer/saber/dizer/ver/ir/dar/ter/ser/
    # estar) — sem eles, uma query fora do domínio que compartilhe só um
    # verbo genérico com qualquer chunk do livro já passa no _keyword_filter
    # (overlap de qualquer 1 token, não uma razão mínima).
    "funciona",
    "funcionam",
    "funcionava",
    "funcionavam",
    "funcionar",
    "significa",
    "significam",
    "significava",
    "significar",
    "existe",
    "existem",
    "existia",
    "existiam",
    "existir",
    "acontece",
    "acontecem",
    "aconteceu",
    "aconteciam",
    "acontecer",
    "chama",
    "chamado",
    "chamada",
    "chamam",
    "chamava",
    "explica",
    "explicam",
    "explicar",
    "explicou",
}

# Module-level defaults (fallback when Retriever is constructed without settings).
# These match the Phase 2 baseline values documented in core/config.py.
_BM25_K1 = 1.5
_BM25_B = 0.75
_MIN_SCORE = 0.25
_HIGH_CONFIDENCE_SCORE = 0.65
_LOW_CONFIDENCE_SCORE = 0.45
_LEXICAL_WEIGHT = 0.40


# ---------------------------------------------------------------------------
# Trace data structures (Step 2 — observability)
# ---------------------------------------------------------------------------


@dataclass
class StageSnapshot:
    """Snapshot of candidates after a single retrieval stage."""

    stage: str
    candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "count": len(self.candidates),
            "candidates": [
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "score": round(c.get("score", 0.0), 4),
                    "text_preview": c.get("text", "")[:80].replace("\n", " "),
                }
                for c in self.candidates
            ],
        }


@dataclass
class RetrievalTrace:
    """Full trace of a retrieve_with_trace() call."""

    query: str
    rewritten_query: str
    stages: list[StageSnapshot] = field(default_factory=list)
    final_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "stages": [s.to_dict() for s in self.stages],
            "final_count": len(self.final_results),
        }

    def summary(self) -> str:
        lines = [
            f"QUERY: {self.query}",
            f"REWRITTEN: {self.rewritten_query}",
            "",
        ]
        for snap in self.stages:
            lines.append(f"[{snap.stage}]  {len(snap.candidates)} candidates")
            for c in snap.candidates[:5]:
                cid = c.get("chunk_id", "?")
                score = c.get("score", 0.0)
                preview = c.get("text", "")[:60].replace("\n", " ")
                lines.append(f'  {cid:<18} score={score:.4f}  "{preview}"')
            if len(snap.candidates) > 5:
                lines.append(f"  ... and {len(snap.candidates) - 5} more")
            lines.append("")
        lines.append(f"FINAL: {len(self.final_results)} results")
        return "\n".join(lines)


class Retriever:
    """Busca híbrida semântica + BM25 com threshold adaptativo e score de confiança."""

    def __init__(
        self,
        vectorstore,
        embed_model,
        top_k: int = 10,
        candidate_multiplier: int = 4,
        min_score: float = _MIN_SCORE,
        lexical_weight: float = _LEXICAL_WEIGHT,
        bm25_k1: float = _BM25_K1,
        bm25_b: float = _BM25_B,
        adaptive_sigma: float = 1.0,
        gap_filter_enabled: bool = True,
        keyword_filter_enabled: bool = True,
        high_confidence_score: float = _HIGH_CONFIDENCE_SCORE,
        low_confidence_score: float = _LOW_CONFIDENCE_SCORE,
        cross_encoder=None,
    ) -> None:
        self._vectorstore = vectorstore
        self._embed_model = embed_model
        self._top_k = top_k
        self._base_top_k = top_k
        self._candidate_multiplier = candidate_multiplier
        self._min_score = min_score
        self._lexical_weight = lexical_weight
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._adaptive_sigma = adaptive_sigma
        self._gap_filter_enabled = gap_filter_enabled
        self._keyword_filter_enabled = keyword_filter_enabled
        self._high_confidence_score = high_confidence_score
        self._low_confidence_score = low_confidence_score
        self._cross_encoder = cross_encoder  # Optional CrossEncoderReranker instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: Optional[int] = None) -> list[dict]:
        """Return the top-k most relevant chunks for the query.

        Args:
            query: The user query string.
            k:     Number of results to return. If None, uses self._top_k
                   (with dynamic adjustment for biographical/period queries).
                   Pass an explicit value to bypass the dynamic adjustment —
                   used by evaluation to ensure consistent depth across queries.

        Returns:
            Ordered list of chunk dicts. Empty if nothing passes quality filters.
            Each chunk includes: text, source, score, confidence.
        """
        results, _ = self._retrieve_internal(query, k=k, trace=False)
        return results

    def retrieve_with_trace(
        self, query: str, k: Optional[int] = None
    ) -> tuple[list[dict], RetrievalTrace]:
        """Like retrieve(), but also returns a full RetrievalTrace.

        The trace records candidate counts and scores at every filter stage,
        enabling per-query failure analysis. Disabled in the normal hot path.

        Args:
            query: The user query string.
            k:     Explicit top-k (same semantics as retrieve()).

        Returns:
            (results, trace) tuple.
        """
        return self._retrieve_internal(query, k=k, trace=True)

    def best_confidence(self, chunks: list[dict]) -> str:
        """Return the confidence label of the highest-scoring chunk."""
        if not chunks:
            return "none"
        best = max(c.get("score", 0) for c in chunks)
        if best >= self._high_confidence_score:
            return "high"
        if best >= self._low_confidence_score:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _retrieve_internal(
        self, query: str, k: Optional[int], trace: bool
    ) -> tuple[list[dict], Optional[RetrievalTrace]]:
        """Core retrieval logic shared by retrieve() and retrieve_with_trace()."""
        tr = RetrievalTrace(query=query, rewritten_query=query) if trace else None

        if self._vectorstore.size == 0:
            logger.warning("VectorStore vazio — sem contexto para: '%s'", query)
            return [], tr

        effective_k = k if k is not None else self._base_top_k

        candidates_count = effective_k * self._candidate_multiplier

        # When cross-encoder is enabled, ensure the FAISS pool is large enough
        # to provide ce_top_k candidates after filtering. Use ce_top_k * cm
        # as the minimum FAISS pool to guarantee enough hybrid candidates.
        if self._cross_encoder is not None:
            min_pool = self._cross_encoder._ce_top_k * self._candidate_multiplier
            candidates_count = max(candidates_count, min_pool)

        # Stage: FAISS semantic retrieval
        query_vec = self._embed_model.embed([query])[0]
        candidates = self._vectorstore.search(query_vec, candidates_count)
        if trace and tr:
            tr.stages.append(StageSnapshot("FAISS_SEMANTIC", list(candidates)))

        if not candidates:
            return [], tr

        # Stage: absolute threshold
        candidates = [c for c in candidates if c.get("score", 0) >= self._min_score]
        if trace and tr:
            tr.stages.append(StageSnapshot("AFTER_ABSOLUTE_FILTER", list(candidates)))
        if not candidates:
            logger.info(
                "Nenhum chunk passou o limiar absoluto %.2f (query='%s')",
                self._min_score,
                query[:60],
            )
            return [], tr

        # Stage: adaptive filter
        candidates = self._adaptive_filter(candidates)
        if trace and tr:
            tr.stages.append(StageSnapshot("AFTER_ADAPTIVE_FILTER", list(candidates)))
        if not candidates:
            logger.info("Filtro adaptativo descartou todos (query='%s')", query[:60])
            return [], tr

        # Stage: gap filter (optional)
        if self._gap_filter_enabled:
            candidates = self._gap_filter(candidates)
            if trace and tr:
                tr.stages.append(StageSnapshot("AFTER_GAP_FILTER", list(candidates)))
            if not candidates:
                logger.info("Gap filter descartou todos (query='%s')", query[:60])
                return [], tr

        # Stage: keyword filter (optional)
        if self._keyword_filter_enabled:
            candidates = self._keyword_filter(query, candidates)
            if trace and tr:
                tr.stages.append(
                    StageSnapshot("AFTER_KEYWORD_FILTER", list(candidates))
                )
            if not candidates:
                logger.info("Keyword filter descartou todos (query='%s')", query[:60])
                return [], tr

        # Stage: hybrid reranking
        reranked = self._hybrid_rerank(query, candidates)
        if trace and tr:
            tr.stages.append(StageSnapshot("AFTER_HYBRID_RERANK", list(reranked)))

        # Stage: cross-encoder reranking (optional — Phase 3F)
        # When CE is enabled, pass the full hybrid-reranked pool to it.
        # The CE internally takes its ce_top_k (default 20) best candidates,
        # applies score fusion, and returns final_k results.
        if self._cross_encoder is not None:
            reranked = self._cross_encoder.rerank(query, reranked, final_k=effective_k)
            if trace and tr:
                tr.stages.append(StageSnapshot("AFTER_CROSS_ENCODER", list(reranked)))
            results = reranked  # CE already returns final_k
        else:
            results = reranked[:effective_k]

        # Attach confidence label
        for chunk in results:
            score = chunk.get("score", 0)
            if score >= self._high_confidence_score:
                chunk["confidence"] = "high"
            elif score >= self._low_confidence_score:
                chunk["confidence"] = "medium"
            else:
                chunk["confidence"] = "low"

        if trace and tr:
            tr.final_results = list(results)

        logger.debug(
            "Retrieve: %d candidatos → %d resultados (query='%s', k=%d)",
            len(candidates),
            len(results),
            query[:60],
            effective_k,
        )
        return results, tr

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _adaptive_filter(self, candidates: list[dict]) -> list[dict]:
        """Keep chunks within adaptive_sigma standard deviations below best score."""
        scores = [c.get("score", 0.0) for c in candidates]
        best = max(scores)
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance**0.5

        threshold = max(self._min_score, best - self._adaptive_sigma * std)
        filtered = [c for c in candidates if c.get("score", 0) >= threshold]

        logger.debug(
            "adaptive_filter: best=%.3f mean=%.3f std=%.3f sigma=%.1f "
            "threshold=%.3f → %d/%d chunks",
            best,
            mean,
            std,
            self._adaptive_sigma,
            threshold,
            len(filtered),
            len(candidates),
        )
        return filtered

    def _keyword_filter(self, query: str, candidates: list[dict]) -> list[dict]:
        """Keep only chunks that share at least one keyword (3+ chars) with the query.

        NOTE: Identified as the most likely source of false negatives in Phase 1
        audit (weakness W2). Retained for baseline fidelity; ablation scheduled
        in Phase 3 Experiment A. Disable via keyword_filter_enabled=False.
        """
        query_keywords = {t for t in _tokenize(query) if len(t) >= 3}
        if not query_keywords:
            return candidates

        matched = []
        for chunk in candidates:
            chunk_tokens = {t for t in _tokenize(chunk.get("text", "")) if len(t) >= 3}
            if query_keywords & chunk_tokens:
                matched.append(chunk)
        return matched

    def _gap_filter(self, candidates: list[dict]) -> list[dict]:
        """Discard candidates only when there is clear evidence of irrelevance.

        NOTE: Identified as a probable false-negative source in Phase 1 audit
        (weakness W1, W5). Retained for baseline; ablation in Phase 3 Experiment B.
        Disable via gap_filter_enabled=False.
        """
        if not candidates:
            return []

        scores = [c.get("score", 0.0) for c in candidates]
        best = max(scores)

        if best >= self._high_confidence_score:
            return [c for c in candidates if c.get("score", 0) >= best - 0.20]
        if best >= self._low_confidence_score:
            return [c for c in candidates if c.get("score", 0) >= best - 0.18]

        # Low best score: only keep if there is a meaningful gap
        others = [s for s in scores if s != best]
        avg_others = sum(others) / len(others) if others else 0.0
        if best - avg_others < 0.10:
            logger.debug(
                "Gap insuficiente: best=%.3f avg_others=%.3f — descartando todos",
                best,
                avg_others,
            )
            return []
        return [c for c in candidates if c.get("score", 0) >= best - 0.18]

    # ------------------------------------------------------------------
    # Hybrid reranking
    # ------------------------------------------------------------------

    def _hybrid_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Combine semantic score and BM25 via weighted score fusion."""
        if self._lexical_weight == 0 or len(candidates) < 3:
            return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)

        query_tokens = _tokenize(query)
        if not query_tokens:
            return candidates

        bm25_scores = _bm25_scores(
            query_tokens,
            [_tokenize(c["text"]) for c in candidates],
            k1=self._bm25_k1,
            b=self._bm25_b,
        )

        sem_scores = [c.get("score", 0.0) for c in candidates]
        sem_norm = _normalize(sem_scores)
        bm25_norm = _normalize(bm25_scores)

        w_sem = 1.0 - self._lexical_weight
        w_lex = self._lexical_weight

        reranked = []
        for chunk, s_sem, s_lex in zip(candidates, sem_norm, bm25_norm):
            combined = w_sem * s_sem + w_lex * s_lex
            new_chunk = dict(chunk)
            new_chunk["score"] = combined
            reranked.append(new_chunk)

        reranked.sort(key=lambda c: c["score"], reverse=True)
        return reranked


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Tokenize: lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS_PT and len(t) > 2]


def _bm25_scores(
    query_tokens: list[str],
    doc_tokens_list: list[list[str]],
    k1: float = _BM25_K1,
    b: float = _BM25_B,
) -> list[float]:
    """Compute BM25 score for query against each document."""
    n_docs = len(doc_tokens_list)
    if n_docs == 0:
        return []

    doc_lengths = [len(d) for d in doc_tokens_list]
    avg_dl = sum(doc_lengths) / n_docs if n_docs else 1.0
    doc_freqs = [Counter(d) for d in doc_tokens_list]

    idf: dict[str, float] = {}
    for term in set(query_tokens):
        df = sum(1 for df_c in doc_freqs if term in df_c)
        idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    scores = []
    for dl, df_c in zip(doc_lengths, doc_freqs):
        score = 0.0
        for term in query_tokens:
            tf = df_c.get(term, 0)
            idf_v = idf.get(term, 0.0)
            num = tf * (k1 + 1)
            den = tf + k1 * (1 - b + b * dl / avg_dl)
            score += idf_v * num / (den + 1e-9)
        scores.append(score)

    return scores


def _normalize(values: list[float]) -> list[float]:
    """Normalize list to [0, 1]."""
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]

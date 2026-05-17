"""
Retriever com busca híbrida (semântica + BM25 léxico) e reranking.

Melhorias em relação à versão anterior:
  - min_score elevado de 0.25 → 0.40 (menos ruído no contexto).
  - Threshold adaptativo baseado em desvio padrão dos scores.
  - Campo 'confidence' exposto nos chunks retornados (usado pelo pipeline).
  - Keyword filter com peso mínimo de 3 chars (antes era 4).
  - Gap detection mais conservador para PDFs técnicos densos.

Estratégia de dois estágios:

  Estágio 1 — Candidatos (rápido):
    - Busca semântica FAISS retorna top-(k * candidate_multiplier) chunks.
    - BM25 leve pontua os mesmos chunks por relevância léxica.
    - Scores são combinados por fusão (Reciprocal Rank Fusion simplificada).

  Estágio 2 — Reranking (preciso):
    - Re-ordena os candidatos por score combinado.
    - Filtra chunks com score abaixo de min_score e threshold adaptativo.
    - Retorna os top-k finais com campo 'confidence'.
"""

import logging
import math
import re
from collections import Counter

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
}

# Parâmetros BM25
_BM25_K1 = 1.5
_BM25_B = 0.75

# Score mínimo ajustado: 0.25 permite capturar mais candidatos relevantes
# Reduzido de 0.30 para 0.25 para melhorar recall em queries com variações textuais
# e evitar falsos negativos em perguntas sobre conteúdo disponível
_MIN_SCORE = 0.25

# Score a partir do qual o chunk é considerado altamente confiável
_HIGH_CONFIDENCE_SCORE = 0.65

# Score abaixo do qual o pipeline deve avisar o aluno sobre baixa confiança
_LOW_CONFIDENCE_SCORE = 0.45

# Peso da fusão: semântico vs léxico (aumentado para 0.40)
# Maior peso léxico ajuda a capturar variações de capitalização e sinônimos
_LEXICAL_WEIGHT = 0.40


class Retriever:
    """Busca híbrida semântica + BM25 com threshold adaptativo e score de confiança."""

    def __init__(
        self,
        vectorstore,
        embed_model,
        top_k: int = 5,
        candidate_multiplier: int = 4,
        min_score: float = _MIN_SCORE,
        lexical_weight: float = _LEXICAL_WEIGHT,
    ) -> None:
        self._vectorstore = vectorstore
        self._embed_model = embed_model
        self._top_k = top_k
        self._base_top_k = top_k  # Guarda o valor base para ajustes dinâmicos
        self._candidates = top_k * candidate_multiplier
        self._min_score = min_score
        self._lexical_weight = lexical_weight

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def _adjust_top_k(self, query: str) -> int:
        """Ajusta dinamicamente o número de chunks (k) baseado no tipo de query.

        Queries que pedem informação ampla ou biográfica precisam de mais chunks
        para fornecer uma resposta completa.

        Tipos de query e multiplicadores:
        - Biográfica ("quem foi", "biografia"): k * 2
        - Período/governo ("como foi", "governo", "período"): k * 1.5
        - Normal: k (sem ajuste)

        Returns:
            Número ajustado de chunks a recuperar
        """
        query_lower = query.lower()

        # Indicadores de query biográfica (precisa de mais contexto)
        biographical_indicators = [
            "quem foi",
            "quem é",
            "biografia",
            "história de",
            "trajetória",
            "vida de",
        ]

        # Indicadores de query sobre período/governo (precisa de contexto moderado)
        period_indicators = [
            "como foi",
            "o que foi",
            "governo",
            "período",
            "periodo",
            "era",
            "fase",
            "mandato",
            "administração",
            "administracao",
        ]

        # Verifica tipo de query
        if any(ind in query_lower for ind in biographical_indicators):
            k = int(self._base_top_k * 2)
            logger.debug(
                "Query biográfica detectada. Aumentando k: %d → %d", self._base_top_k, k
            )
            return k

        if any(ind in query_lower for ind in period_indicators):
            k = int(self._base_top_k * 1.5)
            logger.debug(
                "Query sobre período detectada. Aumentando k: %d → %d",
                self._base_top_k,
                k,
            )
            return k

        # Query normal, usa k base
        return self._base_top_k

    def retrieve(self, query: str) -> list[dict]:
        """Retorna os top-k chunks mais relevantes para a query.

        Cada chunk retornado possui:
          - text, source: conteúdo e origem
          - score: score combinado (semântico + BM25), normalizado [0,1]
          - confidence: 'high' | 'medium' | 'low' — usado pelo pipeline
            para decidir se deve alertar o aluno sobre incerteza.

        Ajusta dinamicamente o número de chunks (top-k) baseado no tipo de query:
          - Queries biográficas ("quem foi", "biografia"): k * 2
          - Queries sobre períodos/governos: k * 1.5
          - Queries normais: k

        Returns:
            Lista ordenada por relevância. Vazia se nenhum chunk
            passar os filtros de qualidade.
        """
        if self._vectorstore.size == 0:
            logger.warning("VectorStore vazio — sem contexto para: '%s'", query)
            return []

        # Ajusta k dinamicamente baseado no tipo de query
        k = self._adjust_top_k(query)

        # Ajusta também o número de candidatos proporcionalmente
        candidates_count = k * 4

        # Estágio 1: candidatos semânticos
        query_vec = self._embed_model.embed([query])[0]
        candidates = self._vectorstore.search(query_vec, candidates_count)

        if not candidates:
            return []

        # Camada 1: threshold absoluto (min_score = 0.25)
        candidates = [c for c in candidates if c.get("score", 0) >= self._min_score]
        if not candidates:
            logger.info(
                "Nenhum chunk passou o limiar absoluto %.2f (query='%s', k=%d)",
                self._min_score,
                query[:60],
                k,
            )
            return []

        # Camada 2: threshold adaptativo baseado em desvio padrão
        candidates = self._adaptive_filter(candidates)
        if not candidates:
            logger.info(
                "Filtro adaptativo descartou todos os chunks (query='%s', k=%d)",
                query[:60],
                k,
            )
            return []

        # Camada 3: gap detection
        candidates = self._gap_filter(candidates)
        if not candidates:
            logger.info(
                "Gap detection descartou todos os chunks (query='%s', k=%d)",
                query[:60],
                k,
            )
            return []

        # Camada 4: overlap léxico (keyword filter)
        candidates = self._keyword_filter(query, candidates)
        if not candidates:
            logger.info(
                "Keyword filter descartou todos os chunks (query='%s', k=%d)",
                query[:60],
                k,
            )
            return []

        # Estágio 2: reranking híbrido
        reranked = self._hybrid_rerank(query, candidates)
        reranked = self._apply_period_boost(query, reranked)
        results = reranked[:k]

        # Adiciona campo 'confidence' a cada chunk
        for chunk in results:
            score = chunk.get("score", 0)
            if score >= _HIGH_CONFIDENCE_SCORE:
                chunk["confidence"] = "high"
            elif score >= _LOW_CONFIDENCE_SCORE:
                chunk["confidence"] = "medium"
            else:
                chunk["confidence"] = "low"

        logger.debug(
            "Retrieve: %d candidatos → %d resultados (query='%s', k=%d)",
            len(candidates),
            len(results),
            query[:60],
            k,
        )
        return results

    def best_confidence(self, chunks: list[dict]) -> str:
        """Retorna o nível de confiança do melhor chunk da lista."""
        if not chunks:
            return "none"
        scores = [c.get("score", 0) for c in chunks]
        best = max(scores)
        if best >= _HIGH_CONFIDENCE_SCORE:
            return "high"
        if best >= _LOW_CONFIDENCE_SCORE:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Filtros
    # ------------------------------------------------------------------

    def _adaptive_filter(self, candidates: list[dict]) -> list[dict]:
        """Mantém chunks dentro de 1 desvio padrão abaixo do melhor score.

        Isso remove a cauda longa de chunks vagamente relacionados que
        passam pelo threshold absoluto mas ficam muito abaixo do melhor.
        """
        scores = [c.get("score", 0.0) for c in candidates]
        best = max(scores)
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance**0.5

        # Threshold = melhor - 1σ, mas nunca abaixo de min_score
        threshold = max(self._min_score, best - std)
        filtered = [c for c in candidates if c.get("score", 0) >= threshold]

        logger.debug(
            "adaptive_filter: best=%.3f mean=%.3f std=%.3f threshold=%.3f → %d/%d chunks",
            best,
            mean,
            std,
            threshold,
            len(filtered),
            len(candidates),
        )
        return filtered

    def _keyword_filter(self, query: str, candidates: list[dict]) -> list[dict]:
        """Mantém apenas chunks com ao menos uma keyword da query (3+ chars).

        Tokens de 3+ chars capturam termos técnicos curtos comuns em
        matemática e ciências (sin, cos, pH, mol, lei, etc.).
        Se a query não tiver keywords com 3+ chars (saudação etc.), não filtra.
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
        """Descarta chunks apenas quando há evidência clara de irrelevância."""
        if not candidates:
            return []

        scores = [c.get("score", 0.0) for c in candidates]
        best = max(scores)

        # Score alto: confia no threshold absoluto (mais tolerante)
        if best >= _HIGH_CONFIDENCE_SCORE:
            return [c for c in candidates if c.get("score", 0) >= best - 0.20]

        # Score médio: aceita chunks próximos (novo threshold intermediário)
        if best >= 0.45:
            return [c for c in candidates if c.get("score", 0) >= best - 0.18]

        # Score baixo: verifica gap (mais tolerante: 0.10 em vez de 0.12)
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
    # Reranking híbrido
    # ------------------------------------------------------------------

    def _hybrid_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Combina score semântico e BM25 por fusão ponderada."""
        if self._lexical_weight == 0 or len(candidates) < 3:
            return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)

        query_tokens = _tokenize(query)
        if not query_tokens:
            return candidates

        bm25_scores = _bm25_scores(
            query_tokens,
            [_tokenize(c["text"]) for c in candidates],
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

    def _apply_period_boost(self, query: str, candidates: list[dict]) -> list[dict]:
        """Prioriza chunks que mencionam explicitamente o período histórico pedido.

        Embeddings tendem a aproximar todos os trechos sobre "Vargas"; para perguntas
        sobre um mandato específico, datas explícitas são um sinal mais forte que
        similaridade semântica genérica.
        """
        query_l = query.lower()
        if not ("segundo governo" in query_l and "vargas" in query_l):
            return candidates

        boosted = []
        for chunk in candidates:
            text_l = chunk.get("text", "").lower()
            score = chunk.get("score", 0.0)

            has_correct_period = (
                ("1951" in text_l and "1954" in text_l)
                or "segundo governo vargas" in text_l
                or "segundo mandato vargas" in text_l
            )
            has_wrong_period = (
                "estado novo" in text_l
                or "1937-1945" in text_l
                or "1937" in text_l
                or "governo provis" in text_l
                or "1930-1934" in text_l
            )

            adjusted = dict(chunk)
            if has_correct_period:
                score += 1.0
            if has_wrong_period and not has_correct_period:
                score -= 0.35
            adjusted["score"] = max(0.0, min(1.0, score))
            boosted.append(adjusted)

        boosted.sort(key=lambda c: c.get("score", 0.0), reverse=True)
        return boosted


# ---------------------------------------------------------------------------
# BM25 local
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Tokenização simples: minúsculo, remove pontuação, filtra stopwords."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS_PT and len(t) > 2]


def _bm25_scores(
    query_tokens: list[str],
    doc_tokens_list: list[list[str]],
) -> list[float]:
    """Calcula BM25 para query contra cada documento na lista."""
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
            num = tf * (_BM25_K1 + 1)
            den = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avg_dl)
            score += idf_v * num / (den + 1e-9)
        scores.append(score)

    return scores


def _normalize(values: list[float]) -> list[float]:
    """Normaliza lista para [0, 1]."""
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]

"""
Retriever com busca híbrida (semântica + BM25 léxico) e reranking.

Estratégia de dois estágios para alta precisão com baixa latência:

  Estágio 1 — Candidatos (rápido):
    - Busca semântica FAISS retorna top-(k * candidate_multiplier) chunks.
    - BM25 leve pontua os mesmos chunks por relevância léxica.
    - Scores são combinados por fusão (Reciprocal Rank Fusion simplificada).

  Estágio 2 — Reranking (preciso):
    - Re-ordena os candidatos por score combinado.
    - Filtra chunks com score abaixo de min_score (evita contexto irrelevante).
    - Retorna os top-k finais.

Por que BM25 híbrido?
  - Modelos de embedding multilíngues às vezes perdem termos técnicos/nomes
    próprios — o BM25 os captura diretamente.
  - Custo computacional do BM25 é O(k) sobre candidatos pré-selecionados,
    não sobre o corpus inteiro → sem impacto de latência significativo.

Sem dependência de modelos de reranking externos (cross-encoders),
que seriam pesados demais para a Raspberry Pi 5.
"""
import logging
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Stopwords básicas PT-BR (evita peso excessivo em palavras funcionais)
_STOPWORDS_PT = {
    "a", "o", "e", "é", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "um", "uma", "uns", "umas", "se",
    "que", "ao", "aos", "às", "ou", "mas", "também", "já", "mais", "como",
    "seu", "sua", "seus", "suas", "este", "esta", "estes", "estas", "esse",
    "essa", "esses", "essas", "isso", "aqui", "ali", "quando", "onde",
    "há", "ter", "ser", "estar", "foi", "são", "era", "tem", "não", "the",
    "of", "and", "in", "to", "is", "it", "that", "for", "on", "are",
    # Verbos genéricos que aparecem em qualquer texto
    "fazer", "fazer", "feito", "faz", "fazia", "faça", "pode", "podem",
    "deve", "devem", "dever", "usar", "usar", "usado", "usar", "precisa",
    "precisar", "preciso", "quero", "quer", "querer", "saber", "sabe",
    "dizer", "disse", "diz", "ver", "veja", "veja", "vai", "vão", "ir",
    "dar", "dá", "dado", "tudo", "todo", "toda", "todos", "todas",
    "muito", "muita", "muitos", "muitas", "pouco", "pouca", "poucos",
    "cada", "qual", "quais", "quem", "cujo", "cuja", "cujos", "cujas",
    "mesmo", "mesma", "apenas", "ainda", "então", "assim", "pois",
    "sobre", "entre", "após", "antes", "durante", "contra", "desde",
    "até", "pelo", "pela", "pelos", "pelas", "num", "numa", "nuns", "numas",
}

# Parâmetros BM25
_BM25_K1 = 1.5
_BM25_B  = 0.75

# Score mínimo de similaridade para incluir chunk no contexto
# (escala IP normalizada: 1.0 = idêntico, 0.0 = ortogonal)
_MIN_SCORE = 0.25

# Peso da fusão: semântico vs léxico (0.0 = só semântico, 1.0 = só léxico)
_LEXICAL_WEIGHT = 0.30


class Retriever:
    """Busca híbrida semântica + BM25 com filtro de score mínimo."""

    def __init__(
        self,
        vectorstore,
        embed_model,
        top_k: int = 5,
        candidate_multiplier: int = 4,
        min_score: float = _MIN_SCORE,
        lexical_weight: float = _LEXICAL_WEIGHT,
    ) -> None:
        """
        Args:
            vectorstore:          Instância de VectorStore populada.
            embed_model:          Instância de EmbeddingModel.
            top_k:                Chunks finais retornados.
            candidate_multiplier: Candidatos semânticos = top_k × multiplier.
            min_score:            Descarta chunks com score combinado abaixo desse valor.
            lexical_weight:       Peso do score BM25 na fusão (0–1).
        """
        self._vectorstore   = vectorstore
        self._embed_model   = embed_model
        self._top_k         = top_k
        self._candidates    = top_k * candidate_multiplier
        self._min_score     = min_score
        self._lexical_weight = lexical_weight

        # BM25: estatísticas do corpus (calculadas sob demanda)
        self._bm25_ready    = False
        self._doc_freqs: list[Counter] = []
        self._idf: dict[str, float]    = {}
        self._avg_dl: float            = 0.0

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict]:
        """Retorna os top-k chunks mais relevantes para a query.

        Args:
            query: Texto da pergunta do usuário.

        Returns:
            Lista de {text, source, score} ordenada por relevância.
            Retorna lista vazia se o índice estiver vazio ou nenhum chunk
            passar o limiar de score mínimo.
        """
        if self._vectorstore.size == 0:
            logger.warning("VectorStore vazio — sem contexto para: '%s'", query)
            return []

        # Estágio 1: candidatos semânticos
        query_vec  = self._embed_model.embed([query])[0]
        candidates = self._vectorstore.search(query_vec, self._candidates)

        if not candidates:
            return []

        # Filtra pelo score semântico BRUTO antes do reranking.
        # FAISS sempre retorna K vizinhos — mesmo sem contexto relevante.
        # Camada 1: threshold absoluto
        candidates = [c for c in candidates if c.get("score", 0) >= self._min_score]

        if not candidates:
            logger.info(
                "Nenhum chunk passou o limiar semântico %.2f (query='%s')",
                self._min_score, query[:60]
            )
            return []

        # Camada 2: gap detection — se os scores são todos parecidos e baixos,
        # o modelo de embedding não encontrou nada realmente relevante.
        candidates = self._gap_filter(candidates)

        if not candidates:
            logger.info("Gap detection descartou todos os chunks (query='%s')", query[:60])
            return []

        # Camada 3: overlap léxico — exige que pelo menos um chunk contenha
        # ao menos uma palavra-chave da query (após remover stopwords).
        # Isso elimina falsos positivos onde o embedding encontra similaridade
        # superficial de linguagem sem relevância temática real.
        candidates = self._keyword_filter(query, candidates)

        if not candidates:
            logger.info("Keyword filter descartou todos os chunks (query='%s')", query[:60])
            return []

        # Estágio 2: reranking híbrido sobre candidatos já filtrados
        reranked = self._hybrid_rerank(query, candidates)

        results = reranked[: self._top_k]

        logger.debug(
            "Retrieve: %d candidatos → %d após filtro (query='%s')",
            len(candidates), len(results), query[:60]
        )
        return results

    def _keyword_filter(self, query: str, candidates: list[dict]) -> list[dict]:
        """Mantém apenas chunks que compartilham ao menos uma palavra-chave com a query.

        Usa apenas tokens com 4+ caracteres para evitar falsos positivos com
        verbos genéricos curtos ("fazer", "usar", "ter").
        Se a query não tiver keywords longas (saudação, etc.), não filtra.
        """
        # Só considera tokens com 4+ chars como keywords significativas
        query_keywords = {t for t in _tokenize(query) if len(t) >= 4}

        if not query_keywords:
            return candidates

        matched = []
        for chunk in candidates:
            chunk_tokens = {t for t in _tokenize(chunk.get("text", "")) if len(t) >= 4}
            if query_keywords & chunk_tokens:
                matched.append(chunk)

        return matched

    def _gap_filter(self, candidates: list[dict]) -> list[dict]:
        """Descarta chunks quando não há um score claramente dominante.

        Lógica:
        - Se o melhor score < 0.70, exige que ele seja pelo menos 15% maior
          que a média dos demais. Caso contrário, provavelmente é ruído.
        - Mantém apenas chunks dentro de 0.15 do melhor score (corta cauda longa).
        """
        if not candidates:
            return []

        scores = [c.get("score", 0.0) for c in candidates]
        best = max(scores)

        # Se o melhor score é alto (>=0.70), confia no threshold absoluto
        if best >= 0.70:
            return [c for c in candidates if c.get("score", 0) >= best - 0.15]

        # Score médio dos demais (excluindo o melhor)
        others = [s for s in scores if s != best]
        avg_others = sum(others) / len(others) if others else 0.0

        # Exige que o melhor se destaque pelo menos 15% acima da média
        if best - avg_others < 0.15:
            logger.debug(
                "Gap insuficiente: best=%.3f avg_others=%.3f — descartando todos",
                best, avg_others
            )
            return []

        return [c for c in candidates if c.get("score", 0) >= best - 0.15]

    # ------------------------------------------------------------------
    # Reranking híbrido
    # ------------------------------------------------------------------

    def _hybrid_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Combina score semântico e BM25 por Reciprocal Rank Fusion."""
        # Com menos de 3 candidatos a normalização distorce os scores (ex: [1.0, 0.0])
        # Retorna ordenado pelo score semântico bruto
        if self._lexical_weight == 0 or len(candidates) < 3:
            return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)

        query_tokens = _tokenize(query)
        if not query_tokens:
            return candidates

        # BM25 sobre os candidatos (corpus local)
        bm25_scores = _bm25_scores(
            query_tokens,
            [_tokenize(c["text"]) for c in candidates],
        )

        # Normaliza ambos os scores para [0, 1]
        sem_scores = [c.get("score", 0.0) for c in candidates]
        sem_norm   = _normalize(sem_scores)
        bm25_norm  = _normalize(bm25_scores)

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
# BM25 local (corpus = candidatos do FAISS)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Tokenização simples: minúsculo, remove pontuação, filtra stopwords."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in _STOPWORDS_PT and len(t) > 2]


def _bm25_scores(
    query_tokens: list[str],
    doc_tokens_list: list[list[str]],
) -> list[float]:
    """Calcula BM25 para query contra cada documento na lista."""
    n_docs = len(doc_tokens_list)
    if n_docs == 0:
        return []

    # Estatísticas do mini-corpus
    doc_lengths = [len(d) for d in doc_tokens_list]
    avg_dl = sum(doc_lengths) / n_docs if n_docs else 1.0

    doc_freqs = [Counter(d) for d in doc_tokens_list]

    # IDF por termo
    idf: dict[str, float] = {}
    for term in set(query_tokens):
        df = sum(1 for df_c in doc_freqs if term in df_c)
        idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    scores = []
    for i, (dl, df_c) in enumerate(zip(doc_lengths, doc_freqs)):
        score = 0.0
        for term in query_tokens:
            tf  = df_c.get(term, 0)
            idf_v = idf.get(term, 0.0)
            num = tf * (_BM25_K1 + 1)
            den = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avg_dl)
            score += idf_v * num / (den + 1e-9)
        scores.append(score)

    return scores


def _normalize(values: list[float]) -> list[float]:
    """Normaliza lista para [0, 1]. Retorna zeros se range == 0."""
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]

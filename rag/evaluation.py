"""
Métricas de avaliação para o pipeline RAG (offline, sem LLM externo).

Métricas implementadas
──────────────────────
  Precision@K  — fração dos K chunks recuperados que são relevantes.
  Recall@K     — fração dos chunks relevantes que foram recuperados entre os K primeiros.
  F1@K         — média harmônica entre Precision@K e Recall@K.
  Hit Rate@K   — 1 se ao menos 1 chunk relevante está no top-K, 0 caso contrário.
  MRR          — Mean Reciprocal Rank: 1/rank do primeiro chunk relevante.
  NDCG@K       — Normalized Discounted Cumulative Gain (lida com graus de relevância).
  MAP@K        — Mean Average Precision over K.

Critério de relevância
──────────────────────
  Um chunk é considerado relevante para uma query se sua string `chunk_id`
  (ou `text` como fallback) estiver no conjunto de chunk_ids anotados como
  ground-truth para aquela query.

  Para PDFs de livro didático sem dataset externo, o módulo inclui
  `build_synthetic_dataset()` que gera pares (query, relevant_chunk_ids)
  a partir de trechos do próprio índice — útil para smoke-test rápido.

Dataset format (JSON)
─────────────────────
  [
    {
      "query": "O que é fotossíntese?",
      "relevant_chunk_ids": ["a1b2c3d4", "e5f6g7h8"],
      "relevant_texts": ["trecho do livro..."]   // alternativa quando não há chunk_id
    },
    ...
  ]

Uso rápido
──────────
  from rag.evaluation import evaluate_retriever, build_synthetic_dataset

  dataset = build_synthetic_dataset(vectorstore, n_samples=30)
  report  = evaluate_retriever(retriever, dataset, k=5)
  print_report(report)
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Resultado de uma única query no dataset de avaliação."""

    query: str
    relevant_ids: set[str]
    retrieved: list[dict]  # chunks retornados pelo retriever (com score)

    # Métricas calculadas por evaluate_query()
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_at_k: float = 0.0
    hit_rate: float = 0.0
    reciprocal_rank: float = 0.0
    ndcg_at_k: float = 0.0
    ap_at_k: float = 0.0
    latency_ms: float = 0.0


@dataclass
class EvaluationReport:
    """Relatório agregado de avaliação do retriever."""

    k: int
    n_queries: int
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    hit_rate: float = 0.0  # fração de queries com ≥1 hit
    mrr: float = 0.0  # Mean Reciprocal Rank
    mean_ndcg: float = 0.0
    map_at_k: float = 0.0  # Mean Average Precision
    mean_latency_ms: float = 0.0
    per_query: list[QueryResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "n_queries": self.n_queries,
            "precision@k": round(self.mean_precision, 4),
            "recall@k": round(self.mean_recall, 4),
            "f1@k": round(self.mean_f1, 4),
            "hit_rate@k": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@k": round(self.mean_ndcg, 4),
            "map@k": round(self.map_at_k, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# Extração de ID de chunk
# ---------------------------------------------------------------------------


def _chunk_id(chunk: dict) -> str:
    """Retorna o identificador do chunk: chunk_id se presente, senão hash do texto."""
    if "chunk_id" in chunk and chunk["chunk_id"]:
        return str(chunk["chunk_id"])
    # Fallback: primeiros 64 chars do texto (compatível com índices antigos)
    return chunk.get("text", "")[:64].strip()


# ---------------------------------------------------------------------------
# Métricas individuais
# ---------------------------------------------------------------------------


def _precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fração dos top-k recuperados que são relevantes."""
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for rid in top if rid in relevant_ids)
    return hits / len(top)


def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fração dos relevantes que aparecem no top-k."""
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for rid in top if rid in relevant_ids)
    return hits / len(relevant_ids)


def _hit_rate(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 se pelo menos 1 relevante está no top-k, 0.0 caso contrário."""
    top = retrieved_ids[:k]
    return 1.0 if any(rid in relevant_ids for rid in top) else 0.0


def _reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / posição do primeiro chunk relevante (1-indexed). 0 se não encontrado."""
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@K com relevância binária (1 se relevante, 0 caso contrário)."""
    top = retrieved_ids[:k]
    # DCG
    dcg = sum(
        1.0 / math.log2(i + 2)  # i+2 porque i é 0-indexed, log2(1) = 0
        for i, rid in enumerate(top)
        if rid in relevant_ids
    )
    # IDCG: todos os relevantes no topo
    n_ideal = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _average_precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Average Precision (AP) truncada em K."""
    top = retrieved_ids[:k]
    hits, precision_sum = 0, 0.0
    for i, rid in enumerate(top, start=1):
        if rid in relevant_ids:
            hits += 1
            precision_sum += hits / i
    return precision_sum / min(len(relevant_ids), k) if relevant_ids else 0.0


# ---------------------------------------------------------------------------
# Avaliação de query individual
# ---------------------------------------------------------------------------


def evaluate_query(
    retriever,
    query: str,
    relevant_ids: set[str],
    relevant_texts: Optional[list[str]],
    k: int,
) -> QueryResult:
    """Avalia uma única query contra o retriever.

    Args:
        retriever:      Instância de Retriever.
        query:          String da pergunta.
        relevant_ids:   Conjunto de chunk_ids relevantes (ground-truth).
        relevant_texts: Textos alternativos para matching quando chunk_id
                        não está disponível (fallback).
        k:              Profundidade de avaliação.

    Returns:
        QueryResult com todas as métricas preenchidas.

    IMPORTANTE: o retriever é chamado com top_k=k para garantir que ele
    devolva exatamente k resultados. Sem isso, Precision@10 == Precision@5
    porque retrieved[:10] == retrieved[:5] quando o retriever retorna só 5.
    """
    # Salva top_k original e força o retriever a devolver k resultados
    original_top_k = retriever._top_k
    original_base_top_k = retriever._base_top_k
    retriever._top_k = k
    retriever._base_top_k = k
    try:
        t0 = time.perf_counter()
        retrieved = retriever.retrieve(query)
        latency_ms = (time.perf_counter() - t0) * 1000
    finally:
        retriever._top_k = original_top_k
        retriever._base_top_k = original_base_top_k

    retrieved_ids = [_chunk_id(c) for c in retrieved]

    # Se relevant_ids está vazio mas temos relevant_texts, construir IDs pelo texto.
    # Compara usando o mesmo critério de _chunk_id: primeiros 64 chars do text.
    effective_ids = set(relevant_ids)
    if not effective_ids and relevant_texts:
        effective_ids = {t[:64].strip() for t in relevant_texts}

    # Quando relevant_ids estão presentes mas nenhum retrieved_id bateu,
    # tenta fallback por prefixo de texto (cobre índices sem chunk_id explícito)
    if effective_ids and not any(rid in effective_ids for rid in retrieved_ids):
        retrieved_text_ids = {c.get("text", "")[:64].strip() for c in retrieved}
        if relevant_texts:
            text_relevant = {t[:64].strip() for t in relevant_texts}
            if retrieved_text_ids & text_relevant:
                effective_ids = text_relevant

    result = QueryResult(
        query=query,
        relevant_ids=effective_ids,
        retrieved=retrieved,
        latency_ms=latency_ms,
    )

    result.precision_at_k = _precision_at_k(retrieved_ids, effective_ids, k)
    result.recall_at_k = _recall_at_k(retrieved_ids, effective_ids, k)
    p, r = result.precision_at_k, result.recall_at_k
    result.f1_at_k = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    result.hit_rate = _hit_rate(retrieved_ids, effective_ids, k)
    result.reciprocal_rank = _reciprocal_rank(retrieved_ids, effective_ids)
    result.ndcg_at_k = _ndcg_at_k(retrieved_ids, effective_ids, k)
    result.ap_at_k = _average_precision_at_k(retrieved_ids, effective_ids, k)

    return result


# ---------------------------------------------------------------------------
# Avaliação sobre dataset completo
# ---------------------------------------------------------------------------


def evaluate_retriever(
    retriever,
    dataset: list[dict],
    k: int = 5,
) -> EvaluationReport:
    """Avalia o retriever sobre um dataset de queries anotadas.

    Args:
        retriever: Instância de Retriever.
        dataset:   Lista de dicts com campos:
                   - "query" (str)
                   - "relevant_chunk_ids" (list[str]) — pode ser vazio
                   - "relevant_texts" (list[str])     — fallback
        k:         Profundidade de avaliação (número de chunks considerados).

    Returns:
        EvaluationReport com médias de todas as métricas.
    """
    if not dataset:
        logger.warning("Dataset vazio — nada a avaliar.")
        return EvaluationReport(k=k, n_queries=0)

    results: list[QueryResult] = []
    for i, item in enumerate(dataset):
        query = item.get("query", "")
        relevant_ids = set(item.get("relevant_chunk_ids", []))
        relevant_texts = item.get("relevant_texts", [])

        if not query:
            continue

        logger.debug("Avaliando query %d/%d: '%s'", i + 1, len(dataset), query[:60])
        qr = evaluate_query(retriever, query, relevant_ids, relevant_texts, k)
        results.append(qr)

    if not results:
        return EvaluationReport(k=k, n_queries=0)

    n = len(results)
    report = EvaluationReport(
        k=k,
        n_queries=n,
        per_query=results,
        mean_precision=sum(r.precision_at_k for r in results) / n,
        mean_recall=sum(r.recall_at_k for r in results) / n,
        mean_f1=sum(r.f1_at_k for r in results) / n,
        hit_rate=sum(r.hit_rate for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        mean_ndcg=sum(r.ndcg_at_k for r in results) / n,
        map_at_k=sum(r.ap_at_k for r in results) / n,
        mean_latency_ms=sum(r.latency_ms for r in results) / n,
    )
    return report


# ---------------------------------------------------------------------------
# Geração de dataset sintético (smoke-test sem anotação manual)
# ---------------------------------------------------------------------------


def build_synthetic_dataset(
    vectorstore,
    n_samples: int = 30,
    seed: int = 42,
) -> list[dict]:
    """Gera um dataset de avaliação sintético a partir do índice existente.

    Estratégia (em ordem de preferência por chunk):

      1. Extrai uma frase completa e informativa do próprio chunk como query.
         Frases reais têm máxima sobreposição semântica e léxica com o chunk
         de origem, garantindo que o retriever consiga recuperá-lo.

      2. Expande os relevant_chunk_ids para incluir chunks vizinhos (mesmo
         source e posição adjacente no índice). O retriever frequentemente
         retorna o chunk anterior ou posterior ao esperado — ambos são
         semanticamente relevantes e não devem penalizar as métricas.

      3. Usa o texto completo do chunk (não apenas 64 chars) como fallback
         de matching quando chunk_id não está disponível.

    Limitações:
      - Ainda é um teste de regressão/configuração, não substitui anotação humana.
      - Para avaliação real, use `load_dataset_from_file()` com perguntas reais.

    Returns:
        Lista de dicts compatíveis com evaluate_retriever().
    """
    rng = random.Random(seed)
    metadata: list[dict] = vectorstore._metadata  # acesso direto à lista interna

    if not metadata:
        logger.warning("VectorStore vazio — dataset sintético vazio.")
        return []

    # Monta índice de vizinhança por source para expansão de relevantes
    source_index: dict[str, list[int]] = {}
    for pos, chunk in enumerate(metadata):
        src = chunk.get("source", "")
        source_index.setdefault(src, []).append(pos)

    n_samples = min(n_samples, len(metadata))
    sampled_positions = rng.sample(range(len(metadata)), n_samples)

    dataset = []
    for pos in sampled_positions:
        chunk = metadata[pos]
        text = chunk.get("text", "")
        cid = _chunk_id(chunk)

        if not text.strip():
            continue

        # --- Query: frase real extraída do chunk ----------------------------
        query = _extract_query_sentence(text, chunk)
        if not query:
            continue

        # --- Relevant IDs: chunk alvo + vizinhos do mesmo source -----------
        src = chunk.get("source", "")
        neighbor_positions = source_index.get(src, [])
        current_offset = (
            neighbor_positions.index(pos) if pos in neighbor_positions else -1
        )

        relevant_ids: list[str] = [cid]
        if current_offset >= 0:
            for delta in (-1, 1):  # chunk anterior e posterior
                neighbor_offset = current_offset + delta
                if 0 <= neighbor_offset < len(neighbor_positions):
                    neighbor_chunk = metadata[neighbor_positions[neighbor_offset]]
                    relevant_ids.append(_chunk_id(neighbor_chunk))

        dataset.append(
            {
                "query": query,
                "relevant_chunk_ids": relevant_ids,
                # Fallback: texto completo (não truncado) para matching por texto
                "relevant_texts": [text.strip()],
                "breadcrumb": chunk.get("breadcrumb", ""),
                "source": src,
            }
        )

    logger.info("Dataset sintético gerado: %d queries.", len(dataset))
    return dataset


def _extract_query_sentence(text: str, chunk: dict) -> str:
    """Extrai a frase mais informativa do chunk para usar como query.

    Prefere frases do meio do chunk (evita frases de transição no início
    e conclusões no final). Filtra frases muito curtas ou que sejam apenas
    números/símbolos.

    Fallback: usa as top keywords para montar uma query descritiva.
    """
    # Divide em frases por pontuação
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 40]

    if sentences:
        # Pega a frase do meio (mais representativa do conteúdo)
        mid = len(sentences) // 2
        # Prefere frases que contenham verbos ou palavras de definição
        definition_markers = [
            "é ",
            "são ",
            "consiste",
            "define",
            "significa",
            "representa",
            "refere",
            "denomina",
            "caracteriza",
            "utiliza",
            "permite",
            "ocorre",
            "resulta",
        ]
        scored = []
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            has_definition = any(m in sent_lower for m in definition_markers)
            # Pontuação: prefere centro do chunk e frases com marcadores de definição
            distance_from_mid = abs(i - mid)
            score = (1 if has_definition else 0) * 3 - distance_from_mid
            scored.append((score, i, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_sentence = scored[0][2]

        # Limpa a frase: remove referências de página, colchetes, etc.
        best_sentence = re.sub(r"\[.*?\]|\(p\.\s*\d+\)", "", best_sentence).strip()

        if len(best_sentence) >= 30:
            return best_sentence

    # Fallback: monta query a partir de keywords
    keywords = _top_keywords(text, n=3)
    if not keywords:
        return ""
    return _make_query_from_keywords(keywords, chunk)


def load_dataset_from_file(path: str) -> list[dict]:
    """Carrega dataset de avaliação de um arquivo JSON.

    Formato esperado:
        [{"query": "...", "relevant_chunk_ids": [...], "relevant_texts": [...]}, ...]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Dataset carregado: %d queries de '%s'.", len(data), path)
    return data


def save_dataset_to_file(dataset: list[dict], path: str) -> None:
    """Salva dataset em JSON para reutilização."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info("Dataset salvo: %d queries em '%s'.", len(dataset), path)


# ---------------------------------------------------------------------------
# Exibição de relatório
# ---------------------------------------------------------------------------


def print_report(report: EvaluationReport, title: str = "RAG Evaluation") -> None:
    """Imprime o relatório de avaliação em formato tabular legível."""
    sep = "─" * 55
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"  K={report.k}  |  Queries: {report.n_queries}")
    print(f"{'═' * 55}")
    print(f"  {'Métrica':<25} {'Valor':>10}")
    print(sep)

    metrics = report.to_dict()
    labels = {
        "precision@k": f"Precision@{report.k}",
        "recall@k": f"Recall@{report.k}",
        "f1@k": f"F1@{report.k}",
        "hit_rate@k": f"Hit Rate@{report.k}",
        "mrr": "MRR",
        "ndcg@k": f"NDCG@{report.k}",
        "map@k": f"MAP@{report.k}",
        "mean_latency_ms": "Latência média (ms)",
    }
    for key, label in labels.items():
        val = metrics.get(key, 0)
        if key == "mean_latency_ms":
            print(f"  {label:<25} {val:>10.1f}")
        else:
            bar = "█" * int(val * 20)
            print(f"  {label:<25} {val:>8.4f}  {bar}")

    print(sep)

    # Queries com pior desempenho
    if report.per_query:
        worst = sorted(report.per_query, key=lambda r: r.recall_at_k)[:3]
        if worst and worst[0].recall_at_k < 1.0:
            print(f"\n  Queries com menor Recall@{report.k}:")
            for qr in worst:
                print(f"    [{qr.recall_at_k:.2f}] {qr.query[:60]}")

    print(f"{'═' * 55}\n")


def compare_reports(
    reports: dict[str, EvaluationReport],
    title: str = "Comparação de configurações",
) -> None:
    """Compara múltiplos relatórios lado a lado."""
    names = list(reports.keys())
    if not names:
        return

    k = next(iter(reports.values())).k
    metrics_keys = [
        "precision@k",
        "recall@k",
        "f1@k",
        "hit_rate@k",
        "mrr",
        "ndcg@k",
        "map@k",
    ]

    col_w = 12
    header = f"  {'Métrica':<22}" + "".join(f"{n[:col_w]:>{col_w}}" for n in names)
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print(f"  {title}  (K={k})")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    for key in metrics_keys:
        label = key.replace("@k", f"@{k}")
        row = f"  {label:<22}"
        vals = [reports[n].to_dict().get(key, 0) for n in names]
        best = max(vals)
        for v in vals:
            mark = " ★" if v == best and len(names) > 1 else "  "
            row += f"{v:>{col_w - 2}.4f}{mark}"
        print(row)

    print(f"{'═' * len(header)}\n")


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

_STOPWORDS_EVAL = {
    "a",
    "o",
    "e",
    "é",
    "de",
    "do",
    "da",
    "em",
    "no",
    "na",
    "por",
    "para",
    "com",
    "um",
    "uma",
    "se",
    "que",
    "ao",
    "ou",
    "mas",
    "como",
    "não",
    "foi",
    "são",
    "era",
    "tem",
    "ser",
    "ter",
    "isso",
    "esse",
    "esta",
    "the",
    "of",
    "and",
    "in",
    "to",
    "is",
    "it",
    "for",
    "on",
    "are",
    "que",
    "dos",
    "das",
    "seu",
    "sua",
    "seus",
    "suas",
    "este",
    "esta",
    "estes",
    "estas",
    "esse",
    "isso",
    "aqui",
    "ali",
    "quando",
    "onde",
    "há",
    "dizer",
    "fazer",
    "ver",
}


def _top_keywords(text: str, n: int = 3) -> list[str]:
    """Retorna as n palavras mais frequentes (sem stopwords, 4+ chars)."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    from collections import Counter

    freq = Counter(t for t in tokens if len(t) >= 4 and t not in _STOPWORDS_EVAL)
    return [w for w, _ in freq.most_common(n)]


def _make_query_from_keywords(keywords: list[str], chunk: dict) -> str:
    """Constrói uma query a partir de keywords e metadados do chunk."""
    breadcrumb = chunk.get("breadcrumb", "")

    # Prefixo contextual se tiver breadcrumb
    if breadcrumb:
        kw_str = " e ".join(keywords[:2])
        return f"O que o material diz sobre {kw_str} em: {breadcrumb}?"

    if len(keywords) >= 2:
        return f"Explique a relação entre {keywords[0]} e {keywords[1]}."
    return f"O que é {keywords[0]}?"

"""
Métricas de avaliação para a etapa de GERAÇÃO do pipeline RAG (offline, sem RAGAS).

Métricas implementadas
───────────────────────
  Faithfulness      — fração das afirmações atômicas da resposta que são
                       sustentadas pelo contexto fornecido. Calculada por um
                       LLM "juiz" local (decomposição + verificação em uma
                       única chamada).
  Answer Relevancy  — o quão bem a resposta aborda a pergunta original.
                       O juiz gera N perguntas que a resposta responderia;
                       a métrica é a similaridade de cosseno média entre
                       essas perguntas geradas e a pergunta original
                       (embeddings via o mesmo modelo de embedding do RAG).

Ambas seguem a metodologia real do RAGAS (LLM-as-judge + embedding
similarity), reimplementada localmente sem depender da lib ragas/langchain,
para caber em hardware restrito (Raspberry Pi 5).

Uma resposta noncommittal/recusa (ex: "Esse conteúdo não está no material
disponível.") recebe Faithfulness=1.0 (nada de errado foi afirmado) e Answer
Relevancy=0.0 (convenção do RAGAS: uma recusa não aborda a pergunta) — em
ambos os casos sem chamar o juiz.

Dataset format (JSON) — data/eval/dataset_generation.json
───────────────────────────────────────────────────────────
  [
    {
      "id": "g0001", "query": "...", "category": "factual",
      "answerable": true,
      "context": ["trecho real do material didático..."],
      "source_breadcrumb": "Unidade 2: ... > Capitulo 3: ..."
    },
    ...
  ]

Uso rápido
──────────
  from rag.generation_evaluation import evaluate_generation

  report = evaluate_generation(candidate_chat_fn, judge_chat_fn, embed_fn, dataset)
  print_generation_report(report)
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rag.pipeline import _SYSTEM_WITH_CONTEXT, _USER_SUFFIX

logger = logging.getLogger(__name__)

# (messages, **kwargs) -> resposta do modelo. kwargs repassados direto para
# LocalModel.chat() (ex: temperature, max_tokens) — permite ao juiz variar
# temperatura/limite de tokens por chamada sem mudar a assinatura.
ChatFn = Callable[..., str]
EmbedFn = Callable[[list[str]], list[list[float]]]

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_REFUSAL_MARKER = "nao esta no material"  # comparado sem acento/case
_MIN_COMMITTAL_CHARS = 15
_CHUNK_SEPARATOR = "\n\n---\n\n"
_DEFAULT_MAX_CONTEXT_CHARS = 3500
_DEFAULT_N_QUESTIONS = 3

# Parâmetros de geração do juiz: determinístico na 1a tentativa; na retentativa
# (só ocorre se o parsing falhar) varia tanto o prompt quanto a temperatura —
# repetir o mesmo prompt em temperatura 0 reproduziria a mesma saída malformada.
_JUDGE_TEMPERATURE_FIRST = 0.0
_JUDGE_TEMPERATURE_RETRY = 0.4
_JUDGE_MAX_TOKENS_FAITHFULNESS = 700
_JUDGE_MAX_TOKENS_RELEVANCY = 200

_CLAIM_LINE_RE = re.compile(
    r"^\s*\d+[.)]\s*(.+?)\s*\|\s*(SIM|NAO|NÃO)\.?\s*$", re.IGNORECASE
)
_VERDICT_TOKEN_RE = re.compile(r"\b(SIM|NAO|NÃO)\b", re.IGNORECASE)
_QUESTION_LINE_RE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")

_FAITHFULNESS_PROMPT_TEMPLATE = """\
Você é um verificador de fatos rigoroso. Sua tarefa tem duas etapas:

ETAPA 1 — Divida a RESPOSTA abaixo em afirmações atômicas: frases curtas, \
cada uma contendo exatamente UM fato verificável. Ignore saudações, convites \
a perguntar mais coisas ou frases sem conteúdo factual.

ETAPA 2 — Para cada afirmação atômica, verifique se ela é DIRETAMENTE \
sustentada pelo CONTEXTO abaixo (use apenas o que está escrito no CONTEXTO, \
não conhecimento externo).

CONTEXTO:
\"\"\"
{context}
\"\"\"

RESPOSTA A VERIFICAR:
\"\"\"
{answer}
\"\"\"

Responda EXCLUSIVAMENTE no formato abaixo, uma linha por afirmação, sem \
nenhum texto antes ou depois:

1. [afirmação atômica] | [SIM ou NAO]
2. [afirmação atômica] | [SIM ou NAO]
...

Regras do formato:
- Use "SIM" se a afirmação está sustentada pelo CONTEXTO, "NAO" caso contrário.
- Não use markdown, não use JSON, não explique o veredito.
- Se a RESPOSTA não tiver nenhuma afirmação factual verificável, responda \
apenas: SEM_AFIRMACOES
"""

_FAITHFULNESS_RETRY_SUFFIX = """

ATENÇÃO: uma tentativa anterior não seguiu esse formato exatamente. Siga o \
formato à risca: uma linha por afirmação, no padrão "N. afirmação | SIM" ou \
"N. afirmação | NAO", nada além disso.
"""

_RELEVANCY_PROMPT_TEMPLATE = """\
Você é um gerador de perguntas. Dada a RESPOSTA abaixo, gere exatamente {n} \
perguntas em português brasileiro que essa RESPOSTA responderia de forma \
completa e direta — ou seja, perguntas para as quais essa RESPOSTA seria uma \
resposta adequada.

RESPOSTA:
\"\"\"
{answer}
\"\"\"

Responda EXCLUSIVAMENTE com {n} perguntas, uma por linha, numeradas, sem \
nenhum texto antes ou depois:

1. [pergunta]
...
{n}. [pergunta]

Regras:
- Cada pergunta deve ser uma pergunta completa em português, terminada em "?".
- Não repita a mesma pergunta com palavras diferentes.
- Não inclua explicações, apenas as perguntas numeradas.
"""

_RELEVANCY_RETRY_SUFFIX = """

ATENÇÃO: uma tentativa anterior não seguiu esse formato exatamente. Siga o \
formato à risca: {n} perguntas numeradas, uma por linha, nada além disso.
"""


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool


@dataclass
class FaithfulnessResult:
    score: Optional[float]  # None = não avaliável, excluído da média
    claims: list[ClaimVerdict] = field(default_factory=list)
    is_refusal: bool = False  # score=1.0 sem chamar o juiz
    parse_error: bool = False
    raw_judge_output: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4) if self.score is not None else None,
            "is_refusal": self.is_refusal,
            "parse_error": self.parse_error,
            "claims": [
                {"claim": c.claim, "supported": c.supported} for c in self.claims
            ],
        }


@dataclass
class AnswerRelevancyResult:
    score: Optional[float]  # None = não avaliável, excluído da média
    generated_questions: list[str] = field(default_factory=list)
    similarities: list[float] = field(default_factory=list)
    is_noncommittal: bool = False  # score=0.0 sem chamar o juiz
    parse_error: bool = False
    raw_judge_output: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4) if self.score is not None else None,
            "is_noncommittal": self.is_noncommittal,
            "parse_error": self.parse_error,
            "generated_questions": self.generated_questions,
            "similarities": [round(s, 4) for s in self.similarities],
        }


@dataclass
class GenerationQueryResult:
    id: str
    query: str
    category: str
    answer: str
    latency_ms: float
    faithfulness: FaithfulnessResult
    answer_relevancy: AnswerRelevancyResult

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "category": self.category,
            "answer": self.answer,
            "latency_ms": round(self.latency_ms, 2),
            "faithfulness": self.faithfulness.to_dict(),
            "answer_relevancy": self.answer_relevancy.to_dict(),
        }


@dataclass
class GenerationReport:
    model_name: str
    model_path: str
    n_queries: int
    mean_faithfulness: Optional[float]
    mean_answer_relevancy: Optional[float]
    n_faithfulness_parse_failures: int
    n_relevancy_parse_failures: int
    mean_latency_ms: float
    per_query: list[GenerationQueryResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        base = {
            "model_path": self.model_path,
            "n_queries": self.n_queries,
            "faithfulness": {
                "mean": (
                    round(self.mean_faithfulness, 4)
                    if self.mean_faithfulness is not None
                    else None
                ),
                "n_scored": sum(
                    1 for q in self.per_query if q.faithfulness.score is not None
                ),
                "n_parse_failures": self.n_faithfulness_parse_failures,
            },
            "answer_relevancy": {
                "mean": (
                    round(self.mean_answer_relevancy, 4)
                    if self.mean_answer_relevancy is not None
                    else None
                ),
                "n_scored": sum(
                    1 for q in self.per_query if q.answer_relevancy.score is not None
                ),
                "n_parse_failures": self.n_relevancy_parse_failures,
            },
            "mean_latency_ms": round(self.mean_latency_ms, 2),
        }
        if self.per_query:
            categories: dict[str, list[GenerationQueryResult]] = {}
            for qr in self.per_query:
                categories.setdefault(qr.category or "uncategorized", []).append(qr)
            if len(categories) > 1:
                by_category = {}
                for cat, qrs in sorted(categories.items()):
                    f_scores = [
                        q.faithfulness.score
                        for q in qrs
                        if q.faithfulness.score is not None
                    ]
                    r_scores = [
                        q.answer_relevancy.score
                        for q in qrs
                        if q.answer_relevancy.score is not None
                    ]
                    by_category[cat] = {
                        "n": len(qrs),
                        "faithfulness": (
                            round(sum(f_scores) / len(f_scores), 4)
                            if f_scores
                            else None
                        ),
                        "answer_relevancy": (
                            round(sum(r_scores) / len(r_scores), 4)
                            if r_scores
                            else None
                        ),
                    }
                base["by_category"] = by_category
        return base


# ---------------------------------------------------------------------------
# Heurística compartilhada de recusa/resposta não-comprometida
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def is_noncommittal_or_refusal(answer: str) -> bool:
    """Detecta respostas vazias, recusas ("não está no material") ou
    degeneradas — usada por Faithfulness e Answer Relevancy para decidir se
    o juiz LLM precisa ser chamado."""
    stripped = (answer or "").strip()
    if not stripped:
        return True
    normalized = _normalize(stripped)
    if _REFUSAL_MARKER in normalized:
        return True
    committal_chars = re.sub(r"[^\w]", "", stripped, flags=re.UNICODE)
    return len(committal_chars) < _MIN_COMMITTAL_CHARS


# ---------------------------------------------------------------------------
# Prompt de geração (candidato) — reaproveita o formato de rag/pipeline.py
# ---------------------------------------------------------------------------


def build_candidate_prompt(
    context: list[str],
    query: str,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> list[dict]:
    """Monta o prompt de geração reaproveitando o formato usado em produção
    (rag/pipeline.py), mas com contexto FIXO (curado no dataset) em vez de
    recuperado — não passa por Retriever nem por ConversationManager."""
    blocks = [f"[trecho {i}]\n{chunk}" for i, chunk in enumerate(context, start=1)]
    context_block = _CHUNK_SEPARATOR.join(blocks)
    if len(context_block) > max_context_chars:
        context_block = context_block[:max_context_chars]
    system_content = _SYSTEM_WITH_CONTEXT.format(context=context_block)
    user_content = query + _USER_SUFFIX
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Prompts do juiz + parsing
# ---------------------------------------------------------------------------


def build_faithfulness_prompt(
    context: str, answer: str, retry: bool = False
) -> list[dict]:
    prompt = _FAITHFULNESS_PROMPT_TEMPLATE.format(context=context, answer=answer)
    if retry:
        prompt += _FAITHFULNESS_RETRY_SUFFIX
    return [{"role": "user", "content": prompt}]


def build_relevancy_prompt(
    answer: str, n_questions: int, retry: bool = False
) -> list[dict]:
    prompt = _RELEVANCY_PROMPT_TEMPLATE.format(answer=answer, n=n_questions)
    if retry:
        prompt += _RELEVANCY_RETRY_SUFFIX.format(n=n_questions)
    return [{"role": "user", "content": prompt}]


def parse_faithfulness_output(raw: str) -> tuple[list[ClaimVerdict], bool]:
    """Retorna (claims, parse_ok). Lista vazia + parse_ok=True significa que
    o juiz declarou explicitamente SEM_AFIRMACOES (parse bem-sucedido)."""
    text = (raw or "").strip()
    if not text:
        return [], False

    normalized = _normalize(text).replace(" ", "")
    if normalized == "sem_afirmacoes":
        return [], True

    claims: list[ClaimVerdict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _CLAIM_LINE_RE.match(line)
        if m:
            claim_text, verdict = m.group(1).strip(), m.group(2)
            claims.append(
                ClaimVerdict(
                    claim=claim_text, supported=verdict.upper().startswith("SIM")
                )
            )
    if claims:
        return claims, True

    # Fallback frouxo: procura o token de veredito em qualquer lugar da linha
    # (cobre casos em que o modelo trocou a numeração/o separador "|").
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        vm = _VERDICT_TOKEN_RE.search(line)
        if vm:
            claim_text = (line[: vm.start()] + line[vm.end() :]).strip(" .|-\t")
            if claim_text:
                claims.append(
                    ClaimVerdict(
                        claim=claim_text,
                        supported=vm.group(1).upper().startswith("SIM"),
                    )
                )

    return claims, bool(claims)


def parse_relevancy_output(raw: str, n_questions: int) -> tuple[list[str], bool]:
    text = (raw or "").strip()
    if not text:
        return [], False

    questions: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _QUESTION_LINE_RE.match(line)
        if m:
            questions.append(m.group(1).strip())

    if not questions:
        # Fallback frouxo: qualquer linha não vazia, sem numeração.
        for line in text.splitlines():
            line = line.strip(" \t-*")
            if line:
                questions.append(line)

    questions = questions[:n_questions]
    return questions, bool(questions)


def _cosine(a: list[float], b: list[float]) -> float:
    """Embeddings já normalizados (L2) por EmbeddingModel — cosseno = dot product."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_faithfulness(
    judge_chat_fn: ChatFn, context: str, answer: str
) -> FaithfulnessResult:
    if is_noncommittal_or_refusal(answer):
        return FaithfulnessResult(score=1.0, is_refusal=True)

    raw = judge_chat_fn(
        build_faithfulness_prompt(context, answer),
        temperature=_JUDGE_TEMPERATURE_FIRST,
        max_tokens=_JUDGE_MAX_TOKENS_FAITHFULNESS,
    )
    claims, parse_ok = parse_faithfulness_output(raw)

    if not parse_ok:
        raw = judge_chat_fn(
            build_faithfulness_prompt(context, answer, retry=True),
            temperature=_JUDGE_TEMPERATURE_RETRY,
            max_tokens=_JUDGE_MAX_TOKENS_FAITHFULNESS,
        )
        claims, parse_ok = parse_faithfulness_output(raw)

    if not parse_ok:
        return FaithfulnessResult(score=None, parse_error=True, raw_judge_output=raw)

    if not claims:
        # SEM_AFIRMACOES numa resposta não-recusa: indefinido, exclui da média
        # (mesma convenção do RAGAS para zero afirmações).
        return FaithfulnessResult(score=None, claims=[], raw_judge_output=raw)

    supported = sum(1 for c in claims if c.supported)
    return FaithfulnessResult(
        score=supported / len(claims), claims=claims, raw_judge_output=raw
    )


def score_answer_relevancy(
    judge_chat_fn: ChatFn,
    embed_fn: EmbedFn,
    query: str,
    answer: str,
    n_questions: int = _DEFAULT_N_QUESTIONS,
) -> AnswerRelevancyResult:
    if is_noncommittal_or_refusal(answer):
        return AnswerRelevancyResult(score=0.0, is_noncommittal=True)

    raw = judge_chat_fn(
        build_relevancy_prompt(answer, n_questions),
        temperature=_JUDGE_TEMPERATURE_FIRST,
        max_tokens=_JUDGE_MAX_TOKENS_RELEVANCY,
    )
    questions, parse_ok = parse_relevancy_output(raw, n_questions)

    if not parse_ok:
        raw = judge_chat_fn(
            build_relevancy_prompt(answer, n_questions, retry=True),
            temperature=_JUDGE_TEMPERATURE_RETRY,
            max_tokens=_JUDGE_MAX_TOKENS_RELEVANCY,
        )
        questions, parse_ok = parse_relevancy_output(raw, n_questions)

    if not parse_ok:
        return AnswerRelevancyResult(score=None, parse_error=True, raw_judge_output=raw)

    vectors = embed_fn([query] + questions)
    query_vec, question_vecs = vectors[0], vectors[1:]
    similarities = [_cosine(query_vec, qv) for qv in question_vecs]
    score = sum(similarities) / len(similarities) if similarities else None

    return AnswerRelevancyResult(
        score=score,
        generated_questions=questions,
        similarities=similarities,
        raw_judge_output=raw,
    )


# ---------------------------------------------------------------------------
# Avaliação do dataset completo
# ---------------------------------------------------------------------------


def _empty_report(model_name: str, model_path: str) -> GenerationReport:
    return GenerationReport(
        model_name=model_name,
        model_path=model_path,
        n_queries=0,
        mean_faithfulness=None,
        mean_answer_relevancy=None,
        n_faithfulness_parse_failures=0,
        n_relevancy_parse_failures=0,
        mean_latency_ms=0.0,
    )


def evaluate_generation(
    candidate_chat_fn: ChatFn,
    judge_chat_fn: ChatFn,
    embed_fn: EmbedFn,
    dataset: list[dict],
    n_questions: int = _DEFAULT_N_QUESTIONS,
    model_name: str = "",
    model_path: str = "",
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> GenerationReport:
    """Avalia um modelo candidato sobre o dataset de geração completo.

    Args:
        candidate_chat_fn: callable(messages) -> resposta do modelo avaliado.
        judge_chat_fn:     callable(messages, **kwargs) -> resposta do juiz.
        embed_fn:          callable(list[str]) -> list[vetor] (já normalizado L2).
        dataset:           itens no schema de data/eval/dataset_generation.json.
    """
    if not dataset:
        logger.warning("Dataset vazio — nada a avaliar.")
        return _empty_report(model_name, model_path)

    results: list[GenerationQueryResult] = []
    for i, item in enumerate(dataset):
        query = item.get("query", "")
        context = item.get("context", [])
        if not query:
            continue

        logger.debug("Avaliando query %d/%d: '%s'", i + 1, len(dataset), query[:60])

        prompt = build_candidate_prompt(
            context, query, max_context_chars=max_context_chars
        )
        t0 = time.perf_counter()
        answer = candidate_chat_fn(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        context_text = _CHUNK_SEPARATOR.join(context)
        faithfulness = score_faithfulness(judge_chat_fn, context_text, answer)
        relevancy = score_answer_relevancy(
            judge_chat_fn, embed_fn, query, answer, n_questions
        )

        results.append(
            GenerationQueryResult(
                id=item.get("id", ""),
                query=query,
                category=item.get("category", ""),
                answer=answer,
                latency_ms=latency_ms,
                faithfulness=faithfulness,
                answer_relevancy=relevancy,
            )
        )

    if not results:
        return _empty_report(model_name, model_path)

    f_scores = [
        r.faithfulness.score for r in results if r.faithfulness.score is not None
    ]
    r_scores = [
        r.answer_relevancy.score
        for r in results
        if r.answer_relevancy.score is not None
    ]
    n_f_failures = sum(1 for r in results if r.faithfulness.parse_error)
    n_r_failures = sum(1 for r in results if r.answer_relevancy.parse_error)

    return GenerationReport(
        model_name=model_name,
        model_path=model_path,
        n_queries=len(results),
        mean_faithfulness=sum(f_scores) / len(f_scores) if f_scores else None,
        mean_answer_relevancy=sum(r_scores) / len(r_scores) if r_scores else None,
        n_faithfulness_parse_failures=n_f_failures,
        n_relevancy_parse_failures=n_r_failures,
        mean_latency_ms=sum(r.latency_ms for r in results) / len(results),
        per_query=results,
    )


# ---------------------------------------------------------------------------
# I/O do dataset e exibição de relatório
# ---------------------------------------------------------------------------


def load_generation_dataset_from_file(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Dataset de geração carregado: %d queries de '%s'.", len(data), path)
    return data


def print_generation_report(
    report: GenerationReport, title: str = "Generation Evaluation"
) -> None:
    sep = "─" * 55
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"  Modelo: {report.model_name}  |  Queries: {report.n_queries}")
    print(f"{'═' * 55}")
    print(f"  {'Métrica':<25} {'Valor':>10}")
    print(sep)

    d = report.to_dict()
    f_mean = d["faithfulness"]["mean"]
    r_mean = d["answer_relevancy"]["mean"]
    f_str = f"{f_mean:.4f}" if f_mean is not None else "N/A"
    r_str = f"{r_mean:.4f}" if r_mean is not None else "N/A"
    print(f"  {'Faithfulness':<25} {f_str:>10}")
    print(f"  {'Answer Relevancy':<25} {r_str:>10}")
    print(f"  {'Latência média (ms)':<25} {report.mean_latency_ms:>10.1f}")
    print(sep)
    print(
        f"  Falhas de parsing — faithfulness: {report.n_faithfulness_parse_failures}, "
        f"relevancy: {report.n_relevancy_parse_failures}"
    )
    print(f"{'═' * 55}\n")

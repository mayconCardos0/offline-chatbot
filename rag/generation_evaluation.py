"""
Métricas de avaliação da qualidade de *geração* do pipeline RAG, via RAGAS.

Diferente de `rag/evaluation.py` (métricas de retrieval, sem LLM externo),
este módulo pontua a RESPOSTA gerada pelo LLM — faithfulness, relevância da
resposta, precisão/recall do contexto usado, correção factual vs. gabarito —
usando um segundo modelo local (GGUF, via llama-cpp-python) como juiz. Nada
sai da máquina: o juiz é só mais um `LocalModel`, adaptado para as interfaces
LangChain/RAGAS esperam via um wrapper mínimo.

`ragas` e `langchain-core` são dependências pesadas, não instaladas por
padrão neste projeto (mesmo motivo de pytest/flake8/black/isort não estarem
em requirements.txt: não pertencem ao runtime da Raspberry Pi). Por isso
todos os imports desses pacotes são feitos sob demanda (dentro de
`evaluate_generation()`), para que este módulo continue importável — e sua
lógica de agregação testável — mesmo sem `ragas` instalado.

Uso rápido
──────────
  from rag.generation_evaluation import evaluate_generation, is_eligible_item

  eligible = [item for item in dataset if is_eligible_item(item)]
  contexts_by_id = {item["id"]: retriever.retrieve(item["query"]) for item in eligible}
  report, details = evaluate_generation(
      eligible, contexts_by_id, pipeline, judge_llm=judge_model, embed_model=embed_model,
      model_name="gemma-2-2b-it-q4_k_m.gguf",
  )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# As 5 métricas clássicas do RAGAS, todas pontuadas pelo LLM-juiz local
# (answer_relevancy também usa embeddings). Nomes espelham
# ragas.metrics.{Faithfulness,AnswerRelevancy,ContextPrecision,ContextRecall,AnswerCorrectness}.
DEFAULT_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]


# ---------------------------------------------------------------------------
# Elegibilidade de itens do dataset
# ---------------------------------------------------------------------------


def is_eligible_item(item: dict) -> bool:
    """Um item só entra na avaliação RAGAS se for respondível e tiver gabarito.

    Queries negativas (`answerable: false`) não têm `reference_answer` e não
    fazem sentido para métricas de fidelidade/correção — já são cobertas pelo
    guardrail de `scripts/eval_rag.py`.
    """
    return bool(item.get("answerable", True)) and bool(item.get("reference_answer"))


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------


@dataclass
class GenerationQueryResult:
    """Resultado de uma única query avaliada para um modelo candidato."""

    id: str
    query: str
    category: str
    contexts: list[str]
    answer: str
    reference_answer: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class GenerationEvaluationReport:
    """Relatório agregado de avaliação de geração para um modelo candidato."""

    model_name: str
    n_evaluated: int = 0
    n_skipped_no_context: int = 0
    metrics: list[str] = field(default_factory=list)
    mean_scores: dict[str, float] = field(default_factory=dict)
    per_query: list[GenerationQueryResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        base: dict[str, Any] = {
            "model": self.model_name,
            "n_evaluated": self.n_evaluated,
            "n_skipped_no_context": self.n_skipped_no_context,
            "metrics": {k: round(v, 4) for k, v in self.mean_scores.items()},
        }
        if self.per_query:
            categories: dict[str, list[GenerationQueryResult]] = {}
            for qr in self.per_query:
                cat = qr.category or "uncategorized"
                categories.setdefault(cat, []).append(qr)
            if len(categories) > 1:
                by_category = {}
                for cat, qrs in sorted(categories.items()):
                    cat_means = {}
                    for metric in self.metrics:
                        vals = [
                            q.scores[metric]
                            for q in qrs
                            if metric in q.scores and _is_finite(q.scores[metric])
                        ]
                        if vals:
                            cat_means[metric] = round(sum(vals) / len(vals), 4)
                    by_category[cat] = {"n": len(qrs), **cat_means}
                base["by_category"] = by_category
        return base


def _is_finite(value: Optional[float]) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


# ---------------------------------------------------------------------------
# Adaptadores LangChain (lazy — só importa langchain_core quando chamado)
# ---------------------------------------------------------------------------


def _build_local_chat_model(local_model: Any, temperature: float, max_tokens: int):
    """Constrói um BaseChatModel (LangChain) mínimo que delega para LocalModel.chat().

    Usado só para o modelo-juiz do RAGAS (via ragas.llms.LangchainLLMWrapper).
    """
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    def _to_chat_messages(messages) -> list[dict]:
        role_by_type = {
            SystemMessage: "system",
            HumanMessage: "user",
            AIMessage: "assistant",
        }
        out = []
        for m in messages:
            role = "user"
            for cls, r in role_by_type.items():
                if isinstance(m, cls):
                    role = r
                    break
            out.append({"role": role, "content": m.content})
        return out

    class LocalGGUFChatModel(BaseChatModel):
        """Wrapper LangChain mínimo sobre `llm.local_model.LocalModel`."""

        local_model: Any
        temperature: float = 0.0
        max_tokens: int = 1024

        def _generate(
            self, messages, stop=None, run_manager=None, **kwargs
        ) -> "ChatResult":
            chat_messages = _to_chat_messages(messages)
            text = self.local_model.chat(
                chat_messages,
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            )

        @property
        def _llm_type(self) -> str:
            return "local-gguf-llama-cpp"

    return LocalGGUFChatModel(
        local_model=local_model, temperature=temperature, max_tokens=max_tokens
    )


def _build_local_embeddings(embed_model: Any):
    """Adapta `rag.embeddings.EmbeddingModel` para a interface `Embeddings` do LangChain.

    Reaproveita o mesmo modelo de embeddings já usado no retrieval — nenhum
    modelo extra é carregado para o RAGAS.
    """
    from langchain_core.embeddings import Embeddings

    class LocalEmbeddingsAdapter(Embeddings):
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._inner.embed(list(texts))

        def embed_query(self, text: str) -> list[float]:
            return self._inner.embed([text])[0]

    return LocalEmbeddingsAdapter(embed_model)


def _score_with_ragas(
    rows: list[dict],
    judge_llm: Any,
    embed_model: Any,
    metric_names: list[str],
    ragas_temperature: float,
    ragas_max_tokens: int,
):
    """Roda ragas.evaluate() sobre as linhas montadas e retorna um DataFrame.

    Todos os imports de ragas/langchain_core ficam aqui, isolados — este é o
    único ponto do módulo que exige esses pacotes instalados.
    """
    try:
        from ragas import EvaluationDataset
        from ragas import evaluate as ragas_evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            "O pacote 'ragas' (e dependências, incl. langchain-core) não está "
            "instalado. Instale com: pip install ragas"
        ) from exc

    registry = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": ContextPrecision,
        "context_recall": ContextRecall,
        "answer_correctness": AnswerCorrectness,
    }
    unknown = [m for m in metric_names if m not in registry]
    if unknown:
        raise ValueError(
            f"Métricas RAGAS desconhecidas: {unknown}. Disponíveis: {sorted(registry)}"
        )

    chat_model = _build_local_chat_model(judge_llm, ragas_temperature, ragas_max_tokens)
    embeddings = _build_local_embeddings(embed_model)

    dataset = EvaluationDataset.from_list(rows)
    metric_instances = [registry[name]() for name in metric_names]
    result = ragas_evaluate(
        dataset=dataset,
        metrics=metric_instances,
        llm=LangchainLLMWrapper(chat_model),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )
    return result.to_pandas()


# ---------------------------------------------------------------------------
# Avaliação de um modelo candidato
# ---------------------------------------------------------------------------


def evaluate_generation(
    items: list[dict],
    contexts_by_id: dict[str, list[dict]],
    pipeline: Any,
    judge_llm: Any,
    embed_model: Any,
    metrics: Optional[list[str]] = None,
    session_prefix: str = "ragas",
    model_name: str = "",
    ragas_temperature: float = 0.0,
    ragas_max_tokens: int = 1024,
) -> tuple[GenerationEvaluationReport, list[dict]]:
    """Avalia um modelo candidato (via `pipeline`) sobre `items` usando RAGAS.

    Args:
        items:           Itens elegíveis do dataset (ver `is_eligible_item`) —
                          já filtrados/amostrados pelo chamador.
        contexts_by_id:  Chunks recuperados por `item["id"]`, computados uma
                          única vez fora do loop de modelos candidatos (a
                          recuperação não depende do modelo de chat).
        pipeline:        RAGPipeline já construído com o modelo candidato.
        judge_llm:       LocalModel do modelo-juiz (fixo entre modelos candidatos).
        embed_model:     EmbeddingModel do projeto (reaproveitado para o RAGAS).
        metrics:         Nomes das métricas RAGAS a rodar (default: DEFAULT_METRICS).
        session_prefix:  Prefixo do session_id usado no RAGPipeline (sessões
                          isoladas por query, sem histórico entre elas).
        model_name:      Nome do modelo candidato, só para identificação no relatório.

    Returns:
        (GenerationEvaluationReport, lista de dicts com detalhe por query).
    """
    metrics = list(metrics) if metrics else list(DEFAULT_METRICS)

    rows: list[dict] = []
    query_results: list[GenerationQueryResult] = []
    n_skipped_no_context = 0

    for item in items:
        item_id = item.get("id", "")
        query = item.get("query", "")
        if not query:
            continue

        chunks = contexts_by_id.get(item_id, [])
        if not chunks:
            n_skipped_no_context += 1
            continue

        session_id = f"{session_prefix}-{item_id or len(query_results)}"
        answer = pipeline.chat(session_id, query)
        contexts = [c.get("text", "") for c in chunks if c.get("text")]

        qr = GenerationQueryResult(
            id=item_id,
            query=query,
            category=item.get("category", ""),
            contexts=contexts,
            answer=answer,
            reference_answer=item["reference_answer"],
        )
        query_results.append(qr)
        rows.append(
            {
                "user_input": query,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": item["reference_answer"],
            }
        )

    mean_scores: dict[str, float] = {}

    if rows:
        df = _score_with_ragas(
            rows, judge_llm, embed_model, metrics, ragas_temperature, ragas_max_tokens
        )
        for i, qr in enumerate(query_results):
            for metric in metrics:
                if metric not in df.columns:
                    continue
                value = df.iloc[i][metric]
                if _is_finite(value):
                    qr.scores[metric] = float(value)
        for metric in metrics:
            if metric not in df.columns:
                continue
            vals = [v for v in df[metric].tolist() if _is_finite(v)]
            if vals:
                mean_scores[metric] = sum(vals) / len(vals)

    report = GenerationEvaluationReport(
        model_name=model_name,
        n_evaluated=len(rows),
        n_skipped_no_context=n_skipped_no_context,
        metrics=metrics,
        mean_scores=mean_scores,
        per_query=query_results,
    )
    details = [
        {
            "id": qr.id,
            "query": qr.query,
            "category": qr.category,
            "contexts": qr.contexts,
            "answer": qr.answer,
            "reference_answer": qr.reference_answer,
            "scores": {k: round(v, 4) for k, v in qr.scores.items()},
        }
        for qr in query_results
    ]
    return report, details

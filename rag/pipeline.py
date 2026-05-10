"""
RAGPipeline otimizado para PT-BR com foco em redução de alucinações.

Melhorias nesta versão:
  - Verificação de relevância temática antes de chamar o LLM:
    compara keywords da query com keywords dos chunks recuperados.
    Se a sobreposição for menor que o limiar, retorna _NO_CONTEXT_RESPONSE
    sem nem chamar o modelo — elimina alucinações por contexto irrelevante.
  - Guardrail duplicado: a restrição de "use apenas o contexto" é repetida
    no final da mensagem do usuário (técnica necessária para Gemma e outros
    modelos que ignoram instruções somente no system prompt).
  - Prefixo de confiança proporcional ao score dos chunks.
  - Contexto ampliado (3500 chars) e histórico reduzido (4 turnos).
"""

import logging
import os
import re
import time
from typing import TYPE_CHECKING

from core.conversation import ConversationManager

if TYPE_CHECKING:
    from llm.local_model import LocalModel
    from rag.retriever import Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords leves para comparação de keywords (sem dependência externa)
# ---------------------------------------------------------------------------
_STOPWORDS = {
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
    "me",
    "eu",
    "tu",
    "ele",
    "ela",
    "nos",
    "eles",
    "elas",
    "você",
    "vocês",
    "qual",
    "quais",
    "quem",
    "quando",
    "onde",
    "como",
    "quanto",
    "seu",
    "sua",
    "meu",
    "minha",
    "nosso",
    "nossa",
}

# ---------------------------------------------------------------------------
# Templates de prompt
# ---------------------------------------------------------------------------
_SYSTEM_WITH_CONTEXT = """\
Você é um professor particular especializado no Ensino Médio brasileiro. \
Seu objetivo é explicar conteúdos de forma clara, completa e didática para estudantes.

REGRAS:
Responda SEMPRE em português brasileiro correto.
Use o contexto fornecido como base principal da resposta.
Se o contexto não contiver a informação, diga: "Esse conteúdo não está no material disponível."
Seja completo: cubra todos os pontos importantes da pergunta, sem deixar lacunas.
Seja objetivo: evite enrolação, repetição e parágrafos desnecessários.
Use exemplos concretos, comparações ou analogias quando ajudarem a entender.
Quando houver comparações (ex: A vs B), use estrutura com tópicos para facilitar a leitura.
Ao final de respostas complexas, inclua um parágrafo curto de resumo/conclusão.
Nunca repita o mesmo ponto com palavras diferentes só para parecer mais completo.
Adapte o nível da linguagem: acessível, mas sem ser simplista.
Não utilize emojis nas respostas.

CONTEXTO:
{context}
"""

# Sufixo adicionado à mensagem do USUÁRIO — reforça o guardrail para modelos
# que tendem a ignorar instruções somente no system prompt (ex: Gemma).
_USER_SUFFIX = (
    "\n\n[INSTRUÇÃO IMPORTANTE: responda APENAS com base no contexto fornecido "
    "no system prompt. Se a resposta não estiver lá, diga que não está no material.]"
)

_NO_CONTEXT_RESPONSE = (
    "Esse conteúdo não está no material disponível para consulta. "
    "Anote sua dúvida e pergunte ao seu professor(a) — assim você garante "
    "uma explicação completa e precisa!"
)

_LOW_CONFIDENCE_PREFIX = (
    "Atenção: as informações encontradas no material têm baixa correspondência "
    "com sua pergunta. Verifique com seu professor(a) se a resposta abaixo está completa.\n\n"
)

_MEDIUM_CONFIDENCE_PREFIX = "Com base no material disponível:\n\n"

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------

_MAX_CONTEXT_CHARS = 3500
_MAX_HISTORY_TURNS = 4
_MIN_CHUNK_CHARS = 80

# Limiar mínimo de overlap de keywords entre query e chunks recuperados.
# Se a sobreposição for menor, os chunks são considerados off-topic e o
# pipeline retorna _NO_CONTEXT_RESPONSE sem chamar o LLM.
# Valor: fração de keywords da query presentes nos chunks (0.0–1.0).
_MIN_KEYWORD_OVERLAP = 0.25

# Scores de confiança
_MEDIUM_CONFIDENCE_THRESHOLD = 0.55
_HIGH_CONFIDENCE_THRESHOLD = 0.70


class RAGPipeline:
    """Combina recuperação e inferência LLM em uma interface de chat."""

    def __init__(
        self,
        retriever: "Retriever",
        llm: "LocalModel",
        conv_manager: ConversationManager,
        max_context_chars: int = _MAX_CONTEXT_CHARS,
        max_history_turns: int = _MAX_HISTORY_TURNS,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._conv_manager = conv_manager
        self._max_context_chars = max_context_chars
        self._max_history_turns = max_history_turns

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def chat(self, session_id: str, message: str) -> str:
        """Processa um turno do usuário e retorna a resposta do assistente."""
        conv = self._conv_manager.get(session_id)
        if conv is None:
            conv = self._conv_manager.create(session_id=session_id)
            logger.debug("Sessão criada automaticamente: '%s'", session_id)

        # Recupera chunks relevantes
        chunks = self._retriever.retrieve(message)

        # --- Verificação 1: nenhum chunk passou os filtros do retriever ---
        if not chunks:
            logger.info(
                "Nenhum chunk relevante — retornando resposta padrão (sessão='%s')",
                session_id,
            )
            return self._persist_and_return(conv, message, _NO_CONTEXT_RESPONSE)

        # --- Verificação 2: relevância temática ---
        # Compara keywords da query com keywords dos chunks.
        # Evita que chunks vagamente relacionados (ex: "história da Europa")
        # sejam usados para responder perguntas específicas não cobertas
        # pelo material (ex: "República de Saló").
        if not self._is_topically_relevant(message, chunks):
            logger.info(
                "Chunks off-topic para a query — retornando resposta padrão (sessão='%s')",
                session_id,
            )
            return self._persist_and_return(conv, message, _NO_CONTEXT_RESPONSE)

        # Prefixo de confiança baseado no score dos chunks
        confidence_prefix = self._confidence_prefix(chunks)

        # Monta mensagens para o LLM
        system_content = self._build_system_prompt(chunks)
        history = self._trim_history(conv["messages"])

        # Guardrail duplicado: repete a instrução na mensagem do usuário
        # Necessário para modelos como Gemma que ignoram o system prompt
        user_message = message + _USER_SUFFIX

        messages = [{"role": "system", "content": system_content}]
        messages.extend({"role": m["role"], "content": m["content"]} for m in history)
        messages.append({"role": "user", "content": user_message})

        # Chama o LLM
        t0 = time.perf_counter()
        response_text = self._llm.chat(messages, stream=False)
        elapsed = time.perf_counter() - t0
        logger.info(
            "LLM respondeu em %.2fs | chunks=%d | sessão='%s'",
            elapsed,
            len(chunks),
            session_id,
        )

        final_response = (
            confidence_prefix + response_text if confidence_prefix else response_text
        )
        return self._persist_and_return(conv, message, final_response)

    # ------------------------------------------------------------------
    # Verificação de relevância temática
    # ------------------------------------------------------------------

    def _is_topically_relevant(self, query: str, chunks: list[dict]) -> bool:
        """Verifica se os chunks são tematicamente relevantes para a query.

        Calcula a fração de keywords significativas da query (4+ chars,
        fora de stopwords) que aparecem em pelo menos um dos chunks.
        Se essa fração for menor que _MIN_KEYWORD_OVERLAP, os chunks são
        considerados off-topic e o pipeline não chama o LLM.

        Exemplos:
          - query="República de Saló", chunks sobre Rev. Francesa:
            overlap baixo → False → _NO_CONTEXT_RESPONSE (sem alucinação)
          - query="fotossíntese cloroplasto", chunks sobre fotossíntese:
            overlap alto → True → LLM gera resposta ancorada
        """
        query_keywords = self._extract_keywords(query)

        # Query sem keywords significativas (saudações, etc.): não filtra
        if not query_keywords:
            return True

        # Agrega todos os tokens dos chunks
        all_chunk_tokens: set[str] = set()
        for chunk in chunks:
            all_chunk_tokens.update(self._extract_keywords(chunk.get("text", "")))

        matched = query_keywords & all_chunk_tokens
        overlap = len(matched) / len(query_keywords)

        logger.debug(
            "Relevância temática: query_kw=%s | matched=%s | overlap=%.2f (limiar=%.2f)",
            query_keywords,
            matched,
            overlap,
            _MIN_KEYWORD_OVERLAP,
        )

        return overlap >= _MIN_KEYWORD_OVERLAP

    def _extract_keywords(self, text: str) -> set[str]:
        """Extrai keywords significativas (4+ chars, sem stopwords)."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return {t for t in tokens if len(t) >= 4 and t not in _STOPWORDS}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist_and_return(self, conv: dict, message: str, response: str) -> str:
        """Persiste o par user/assistant no histórico e retorna a resposta."""
        ts = time.time()
        updated = list(conv["messages"])
        updated.append({"role": "user", "content": message, "timestamp": ts})
        updated.append({"role": "assistant", "content": response, "timestamp": ts})
        self._conv_manager.update(conv["id"], updated)
        return response

    def _confidence_prefix(self, chunks: list[dict]) -> str:
        """Retorna prefixo de aviso proporcional ao score dos chunks."""
        if not chunks:
            return ""
        best = max(c.get("score", 0) for c in chunks)
        if best < _MEDIUM_CONFIDENCE_THRESHOLD:
            return _LOW_CONFIDENCE_PREFIX
        if best < _HIGH_CONFIDENCE_THRESHOLD:
            return _MEDIUM_CONFIDENCE_PREFIX
        return ""

    def _build_system_prompt(self, chunks: list[dict]) -> str:
        """Monta o prompt de sistema com contexto filtrado e comprimido."""
        usable = [c for c in chunks if len(c.get("text", "")) >= _MIN_CHUNK_CHARS]
        if not usable:
            usable = chunks

        context_parts: list[str] = []
        total_chars = 0

        for chunk in usable:
            source = os.path.basename(chunk.get("source", "desconhecido"))
            score = chunk.get("score", 0.0)
            header = f"[fonte: {source} | relevância: {score:.2f}]"
            body = chunk["text"].strip()
            entry = f"{header}\n{body}"

            if total_chars + len(entry) > self._max_context_chars:
                remaining = self._max_context_chars - total_chars
                if remaining > 150:
                    entry = entry[:remaining] + "…"
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            total_chars += len(entry)

        context_block = "\n\n---\n\n".join(context_parts)
        return _SYSTEM_WITH_CONTEXT.format(context=context_block)

    def _trim_history(self, messages: list[dict]) -> list[dict]:
        """Mantém apenas os últimos N turnos de histórico."""
        pairs = [m for m in messages if m["role"] in ("user", "assistant")]
        max_msgs = self._max_history_turns * 2
        return pairs[-max_msgs:]

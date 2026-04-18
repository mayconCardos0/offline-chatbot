"""
RAGPipeline otimizado para PT-BR com:
  - Compressão de contexto: limita total de tokens injetados.
  - Prompt mais diretivo para modelos pequenos (GGUF quantizados).
  - Histórico com janela deslizante (evita estouro de contexto na RPi).
  - Logging de score dos chunks para diagnóstico.
"""
import logging
import os
import time
from typing import TYPE_CHECKING

from core.conversation import ConversationManager

if TYPE_CHECKING:
    from llm.local_model import LocalModel
    from rag.retriever import Retriever

logger = logging.getLogger(__name__)

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

_NO_CONTEXT_RESPONSE = "No momento não consigo te responder com certeza, mas você pode anotá-la e perguntar ao seu professor(a) depois — assim você garante uma explicação completa!"

# Número máximo de caracteres de contexto injetado no prompt
# ~2 000 chars ≈ ~500 tokens — seguro para modelos com N_CTX=4096
_MAX_CONTEXT_CHARS = 2000

# Número máximo de turnos de histórico mantidos (user+assistant = 1 turno)
_MAX_HISTORY_TURNS = 6


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
        self._retriever         = retriever
        self._llm               = llm
        self._conv_manager      = conv_manager
        self._max_context_chars = max_context_chars
        self._max_history_turns = max_history_turns

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def chat(self, session_id: str, message: str) -> str:
        """Processa um turno do usuário e retorna a resposta do assistente.

        Cria a sessão automaticamente se não existir.

        Args:
            session_id: Identificador único da conversa.
            message:    Mensagem do usuário.

        Returns:
            Resposta do assistente como string.
        """
        # Garante que a sessão existe
        conv = self._conv_manager.get(session_id)
        if conv is None:
            conv = self._conv_manager.create(session_id=session_id)
            logger.debug("Sessão criada automaticamente: '%s'", session_id)

        # Recupera chunks relevantes
        chunks = self._retriever.retrieve(message)
        if chunks:
            scores_str = ", ".join(f"{c.get('score', 0):.3f}" for c in chunks)
            logger.info("Chunks recuperados (scores): [%s]", scores_str)
        else:
            logger.info("Nenhum chunk relevante encontrado para sessão='%s'", session_id)
            # Persiste a troca mesmo sem contexto
            ts = time.time()
            updated = list(conv["messages"])
            updated.append({"role": "user",      "content": message,              "timestamp": ts})
            updated.append({"role": "assistant", "content": _NO_CONTEXT_RESPONSE, "timestamp": ts})
            self._conv_manager.update(session_id, updated)
            return _NO_CONTEXT_RESPONSE

        # Monta prompt de sistema com contexto comprimido
        system_content = self._build_system_prompt(chunks)

        # Monta histórico com janela deslizante
        history = self._trim_history(conv["messages"])

        messages = [{"role": "system", "content": system_content}]
        messages.extend({"role": m["role"], "content": m["content"]} for m in history)
        messages.append({"role": "user", "content": message})

        # Chama o LLM
        t0 = time.perf_counter()
        response_text = self._llm.chat(messages, stream=False)
        elapsed = time.perf_counter() - t0
        logger.info(
            "LLM respondeu em %.2fs | chunks=%d | sessão='%s'",
            elapsed, len(chunks), session_id
        )

        # Persiste histórico atualizado
        ts = time.time()
        updated = list(conv["messages"])
        updated.append({"role": "user",      "content": message,       "timestamp": ts})
        updated.append({"role": "assistant", "content": response_text, "timestamp": ts})
        self._conv_manager.update(session_id, updated)

        return response_text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, chunks: list[dict]) -> str:
        """Monta o prompt de sistema com contexto comprimido."""
        context_parts: list[str] = []
        total_chars = 0

        for chunk in chunks:
            source = os.path.basename(chunk.get("source", "desconhecido"))
            score  = chunk.get("score", 0.0)
            header = f"[fonte: {source} | relevância: {score:.2f}]"
            body   = chunk["text"].strip()
            entry  = f"{header}\n{body}"

            if total_chars + len(entry) > self._max_context_chars:
                # Trunca o último chunk se necessário
                remaining = self._max_context_chars - total_chars
                if remaining > 100:
                    entry = entry[:remaining] + "…"
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            total_chars += len(entry)

        context_block = "\n\n".join(context_parts)
        return _SYSTEM_WITH_CONTEXT.format(context=context_block)

    def _trim_history(self, messages: list[dict]) -> list[dict]:
        """Mantém apenas os últimos N turnos de histórico."""
        # Filtra apenas roles user/assistant
        pairs = [m for m in messages if m["role"] in ("user", "assistant")]
        # Mantém os últimos max_history_turns * 2 mensagens
        max_msgs = self._max_history_turns * 2
        return pairs[-max_msgs:]

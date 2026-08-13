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
Seu objetivo é explicar conteúdos de forma clara, completa e didática para estudantes, \
de um jeito que resolva completamente a dúvida do aluno.

REGRAS:
Responda SEMPRE em português brasileiro correto.
Use EXCLUSIVAMENTE as informações do CONTEXTO fornecido.
VERIFICAÇÃO OBRIGATÓRIA: Antes de mencionar QUALQUER data, nome ou evento, verifique se está EXPLICITAMENTE no contexto.
Se o contexto não contiver a informação, diga: "Esse conteúdo não está no material disponível."
Seja completo: cubra todos os pontos importantes da pergunta, sem deixar lacunas — prefira sempre uma resposta mais longa e completa a uma resposta breve e incompleta.
"Objetivo" aqui significa não repetir a mesma ideia com palavras diferentes só para parecer mais completo — NÃO significa dar respostas curtas.
Se a pergunta tiver mais de uma parte (ex: causa e consequência, comparação entre dois temas), responda cada parte explicitamente, sem pular nenhuma.
Não se limite a listar fatos: explique as causas, o contexto e as consequências de cada evento ou conceito, sempre que o CONTEXTO trouxer essa informação.
Ao explicar um conceito, mostre como ele se conecta ao que está descrito no CONTEXTO, em vez de apenas afirmar algo — isso ajuda o aluno a entender de onde vem a explicação.
Se o CONTEXTO trouxer informações relacionadas que ajudem a entender melhor o tema perguntado, inclua-as — o objetivo é que o aluno não precise perguntar de novo sobre o mesmo assunto.
Defina termos técnicos ou históricos que aparecerem na resposta, mesmo que brevemente.
Use exemplos concretos, comparações ou analogias quando ajudarem a entender.
Sempre que a resposta tiver mais de um aspecto (causas, consequências, características, etapas, comparações), organize com tópicos ou subtítulos para facilitar o estudo.
Nunca repita o mesmo ponto com palavras diferentes só para parecer mais completo.
Adapte o nível da linguagem: acessível, mas sem ser simplista.
Não utilize emojis nas respostas.
NÃO inclua raciocínio interno ou textos como <think>. Responda apenas com a resposta final.

CONTEXTO:
{context}
"""

# Sufixo adicionado à mensagem do USUÁRIO — reforça o guardrail para modelos
# que tendem a ignorar instruções somente no system prompt (ex: Gemma).
_USER_SUFFIX = (
    "\n\n[INSTRUÇÃO IMPORTANTE: responda APENAS com base no contexto fornecido "
    "no system prompt. Se a resposta não estiver lá, diga que não está no material. "
    "Dê uma resposta completa e aprofundada, cobrindo causas, contexto e consequências "
    "quando o material trouxer essa informação — não dê uma resposta curta.]"
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

# Geração automática de título da conversa (a partir da primeira pergunta)
_TITLE_SYSTEM_PROMPT = (
    "Gere um título curto (no máximo 6 palavras) em português para a "
    "pergunta a seguir, resumindo o assunto. Responda APENAS com o título, "
    "sem aspas, sem pontuação final e sem explicações."
)
_TITLE_MAX_TOKENS = 20
_TITLE_MAX_CHARS = 60

# Limiar mínimo de overlap de keywords entre query e chunks recuperados.
# Se a sobreposição for menor, os chunks são considerados off-topic e o
# pipeline retorna _NO_CONTEXT_RESPONSE sem chamar o LLM.
# Valor: fração de keywords da query presentes nos chunks (0.0–1.0).
# Reduzido de 0.20 para 0.15 para ser mais tolerante a variações textuais
# e evitar falsos negativos em perguntas sobre conteúdo disponível
_MIN_KEYWORD_OVERLAP = 0.15

# Scores de confiança (ajustados para alinhar com retriever)
_MEDIUM_CONFIDENCE_THRESHOLD = 0.45
_HIGH_CONFIDENCE_THRESHOLD = 0.65


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

        # Detecta se é uma pergunta de follow-up
        is_followup = self._is_followup_question(message, conv["messages"])

        # Se for follow-up, enriquece a query com contexto da conversa anterior
        enriched_query = message
        if is_followup:
            context_from_history = self._extract_context_from_history(conv["messages"])
            if context_from_history:
                enriched_query = f"{message} {context_from_history}"
                logger.debug(
                    "Follow-up detectado. Query enriquecida: '%s'", enriched_query[:100]
                )

        # Log da transformação da query
        if enriched_query != message:
            logger.debug(
                "Query transformada: '%s' → '%s'", message[:80], enriched_query[:80]
            )

        # Recupera chunks relevantes (usa query enriquecida se for follow-up)
        chunks = self._retriever.retrieve(enriched_query)

        # --- Verificação 1: nenhum chunk passou os filtros do retriever ---
        if not chunks:
            logger.info(
                "Nenhum chunk relevante — retornando resposta padrão (sessão='%s', query='%s')",
                session_id,
                enriched_query[:100],
            )
            return self._persist_and_return(conv, message, _NO_CONTEXT_RESPONSE)

        # Log dos chunks recuperados
        logger.debug(
            "Chunks recuperados: %d | Scores: %s",
            len(chunks),
            [f"{c.get('score', 0):.3f}" for c in chunks[:5]],
        )

        # --- Verificação 2: relevância temática ---
        # Compara keywords da query com keywords dos chunks.
        # Evita que chunks vagamente relacionados (ex: "história da Europa")
        # sejam usados para responder perguntas específicas não cobertas
        # pelo material (ex: "República de Saló").
        if not self._is_topically_relevant(enriched_query, chunks):
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
        response_text = self._clean_model_response(response_text, message)
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

    def _is_followup_question(self, message: str, history: list[dict]) -> bool:
        """Detecta se a mensagem é uma pergunta de follow-up.

        Follow-ups são perguntas que referenciam a conversa anterior,
        pedindo mais detalhes ou esclarecimentos sobre o tópico em discussão.
        """
        if not history or len(history) < 2:
            return False

        followup_indicators = [
            "explique melhor",
            "e sobre",
            "e quanto",
            "nos estavamos falando",
            "nós estávamos falando",
            "você disse",
            "você mencionou",
            "continue",
            "mais detalhes",
            "aprofunde",
            "detalhe",
            "e a",
            "e o",
            "como assim",
            "o que você quis dizer",
            "pode explicar",
            # Novos indicadores para melhorar detecção
            "como foi",
            "o que foi",
            "quem foi",
            "quando foi",
            "onde foi",
            "por que",
            "porque",
            "qual foi",
            "quais foram",
            "me fale mais",
            "conte mais",
            "explique",
            "esclareça",
            "desenvolva",
        ]

        message_lower = message.lower()
        return any(indicator in message_lower for indicator in followup_indicators)

    def _extract_context_from_history(self, history: list[dict]) -> str:
        """Extrai tópicos principais das últimas mensagens para enriquecer follow-ups.

        Pega as últimas respostas do assistente e extrai keywords principais
        para adicionar contexto à query de follow-up.
        """
        if not history:
            return ""

        # Pega as últimas 2 mensagens do assistente
        recent_responses = [
            msg["content"]
            for msg in history[-4:]
            if msg["role"] == "assistant"
            and "Esse conteúdo não está no material" not in msg["content"]
        ][-2:]

        if not recent_responses:
            return ""

        # Extrai keywords principais (palavras com 5+ chars, capitalizadas ou técnicas)
        context_keywords = set()
        for response in recent_responses:
            # Pega palavras capitalizadas (nomes próprios, eventos históricos)
            words = response.split()
            for word in words:
                # Remove pontuação
                clean_word = word.strip('.,;:!?"()[]')
                if clean_word and len(clean_word) >= 5:
                    # Adiciona se for capitalizada ou se for uma palavra técnica comum
                    if clean_word[0].isupper() or clean_word.lower() in {
                        "revolução",
                        "revolucao",
                        "guerra",
                        "governo",
                        "industrial",
                        "trabalhadores",
                        "economia",
                        "política",
                        "politica",
                        "estado",
                        "brasil",
                        "brasileiro",
                        "brasileira",
                        "período",
                        "periodo",
                        "ditadura",
                        "democracia",
                        "constitucional",
                        "provisório",
                        "provisorio",
                    }:
                        context_keywords.add(clean_word)

        # Limita a 8 keywords mais relevantes
        return " ".join(list(context_keywords)[:8])

    # ------------------------------------------------------------------
    # Verificação de relevância temática
    # ------------------------------------------------------------------

    def _is_topically_relevant(self, query: str, chunks: list[dict]) -> bool:
        """Verifica se os chunks são tematicamente relevantes para a query.

        Calcula a fração de keywords significativas da query (4+ chars,
        fora de stopwords) que aparecem em pelo menos um dos chunks.
        Se essa fração for menor que _MIN_KEYWORD_OVERLAP, os chunks são
        considerados off-topic e o pipeline não chama o LLM.

        Melhorias:
        - Tratamento especial para nomes próprios (palavras capitalizadas)
        - Se um nome próprio da query aparece nos chunks, considera relevante
        - Isso ajuda a capturar "Segundo Governo Vargas" mesmo com variações

        Exemplos:
          - query="República de Saló", chunks sobre Rev. Francesa:
            overlap baixo → False → _NO_CONTEXT_RESPONSE (sem alucinação)
          - query="fotossíntese cloroplasto", chunks sobre fotossíntese:
            overlap alto → True → LLM gera resposta ancorada
          - query="Segundo Governo Vargas", chunks com "Vargas":
            nome próprio encontrado → True → LLM gera resposta
        """
        query_keywords = self._extract_keywords(query)

        # Query sem keywords significativas (saudações, etc.): não filtra
        if not query_keywords:
            return True

        # Extrai nomes próprios da query (palavras capitalizadas com 3+ chars)
        proper_nouns = set(
            word
            for word in query.split()
            if word and word[0].isupper() and len(word) > 3
        )

        # Agrega todos os tokens dos chunks
        all_chunk_tokens: set[str] = set()
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            all_chunk_tokens.update(self._extract_keywords(chunk_text))
            # Adiciona também palavras capitalizadas do chunk (lowercased para comparação)
            all_chunk_tokens.update(
                word.lower()
                for word in chunk_text.split()
                if word and word[0].isupper() and len(word) > 3
            )

        matched = query_keywords & all_chunk_tokens

        # Se há nomes próprios na query, verifica se pelo menos um está nos chunks
        if proper_nouns:
            proper_matched = any(pn.lower() in all_chunk_tokens for pn in proper_nouns)
            if proper_matched:
                logger.debug(
                    "Nome próprio encontrado nos chunks: %s → relevante",
                    proper_nouns
                    & {pn for pn in proper_nouns if pn.lower() in all_chunk_tokens},
                )
                return True  # Nome próprio encontrado = relevante

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

    def _clean_model_response(self, response: str, user_message: str = "") -> str:
        """Remove raciocínio interno e ecos de prompt comuns em modelos pequenos."""
        text = (response or "").strip()
        if not text:
            return _NO_CONTEXT_RESPONSE

        text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
        text = re.sub(r"(?is)<think>.*$", "", text).strip()

        # Alguns modelos repetem a mensagem de usuário com o sufixo de guardrail.
        text = re.sub(
            r"(?is)^.*?\[INSTRUÇÃO IMPORTANTE:.*?\]\s*",
            "",
            text,
            count=1,
        ).strip()

        if user_message and text.lower().startswith(user_message.lower()):
            text = text[len(user_message) :].strip(" \n\r\t:-")

        prompt_echo_starts = (
            "você é um professor",
            "voce e um professor",
            "regras:",
            "contexto:",
        )
        if text.lower().startswith(prompt_echo_starts):
            return _NO_CONTEXT_RESPONSE

        return text or _NO_CONTEXT_RESPONSE

    def _persist_and_return(self, conv: dict, message: str, response: str) -> str:
        """Persiste o par user/assistant no histórico e retorna a resposta."""
        is_first_turn = len(conv["messages"]) == 0

        ts = time.time()
        updated = list(conv["messages"])
        updated.append({"role": "user", "content": message, "timestamp": ts})
        updated.append({"role": "assistant", "content": response, "timestamp": ts})
        self._conv_manager.update(conv["id"], updated)

        if is_first_turn:
            title = self._generate_title(message)
            self._conv_manager.set_title(conv["id"], title)

        return response

    def _generate_title(self, message: str) -> str:
        """Gera um título curto para a conversa a partir da primeira pergunta.

        Chamada extra ao LLM, isolada da resposta principal — nunca deve
        derrubar o turno do usuário. Em qualquer falha (exceção do modelo ou
        saída vazia após limpeza), cai de volta na própria pergunta truncada.
        """
        fallback = self._clean_title(message)
        try:
            raw = self._llm.chat(
                [
                    {"role": "system", "content": _TITLE_SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                stream=False,
                max_tokens=_TITLE_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("Falha ao gerar título da conversa: %s", exc)
            return fallback

        title = self._clean_title(raw)
        return title or fallback

    @staticmethod
    def _clean_title(text: str) -> str:
        """Normaliza um título bruto: remove <think>, aspas, quebras de linha."""
        text = (text or "").strip()
        text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
        text = re.sub(r"(?is)<think>.*$", "", text).strip()
        text = text.replace("\n", " ").strip(" \"'“”'.-")
        if len(text) > _TITLE_MAX_CHARS:
            text = text[:_TITLE_MAX_CHARS].rstrip() + "…"
        return text

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
            crumb = chunk.get("breadcrumb", "")
            header = f"[fonte: {source}{' | ' + crumb if crumb else ''} | relevância: {score:.2f}]"
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

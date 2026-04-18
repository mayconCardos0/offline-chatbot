"""
Local LLM wrapper using llama-cpp-python.

Melhorias em relação à versão anterior:
  - Temperatura padrão reduzida de 0.7 → 0.1 (determinismo para conteúdo factual).
  - repeat_penalty=1.1 para evitar loops de repetição em modelos GGUF pequenos.
  - top_p ajustado para 0.95 (compensa a temperatura baixa sem perder fluência).
  - Parâmetros de geração expostos no método chat() para override por chamador.
"""
import logging
import os
from typing import Iterator

logger = logging.getLogger(__name__)

# Parâmetros de geração padrão otimizados para RAG factual em PT-BR
_DEFAULT_TEMPERATURE    = 0.1   # Baixo = respostas determinísticas e ancoradas no contexto
_DEFAULT_TOP_P          = 0.95  # Um pouco mais amplo para compensar temperatura baixa
_DEFAULT_REPEAT_PENALTY = 1.1   # Evita loops de repetição comuns em modelos GGUF pequenos
_DEFAULT_MAX_TOKENS     = 2048


class LocalModel:
    """Thin wrapper around llama_cpp.Llama for chat completion."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
    ) -> None:
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model file not found at configured path: {model_path}"
            )

        logger.info("Loading GGUF model from %s", model_path)
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info("Model loaded successfully.")

    @staticmethod
    def _normalize_messages(messages: list[dict]) -> list[dict]:
        """
        Alguns modelos (ex: Gemma) não suportam role 'system'.
        Se a primeira mensagem for um system prompt, injeta seu conteúdo
        na primeira mensagem de usuário para não perder o contexto.
        """
        if not messages or messages[0]["role"] != "system":
            return messages

        system_content = messages[0]["content"]
        rest = list(messages[1:])

        for i, msg in enumerate(rest):
            if msg["role"] == "user":
                rest[i] = {
                    "role": "user",
                    "content": f"{system_content}\n\n{msg['content']}",
                }
                return rest

        return rest

    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        temperature: float = _DEFAULT_TEMPERATURE,
        top_p: float = _DEFAULT_TOP_P,
        repeat_penalty: float = _DEFAULT_REPEAT_PENALTY,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> "str | Iterator[str]":
        """
        Executa uma chat completion.

        Args:
            messages:       Lista de {role, content} (estilo OpenAI).
            stream:         Se True, retorna iterator de tokens.
            temperature:    Controla aleatoriedade (0.1 = determinístico).
                            Mantenha baixo para respostas factuais RAG.
            top_p:          Nucleus sampling threshold.
            repeat_penalty: Penalidade para tokens repetidos (≥1.0).
            max_tokens:     Máximo de tokens gerados.

        Returns:
            String completa (stream=False) ou iterator de tokens (stream=True).
        """
        gen_kwargs = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stream=stream,
        )

        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                **gen_kwargs,
            )
        except ValueError:
            logger.debug("Model rejected system role — retrying with folded system prompt.")
            normalized = self._normalize_messages(messages)
            response   = self._llm.create_chat_completion(
                messages=normalized,
                **gen_kwargs,
            )

        if stream:
            return self._iter_tokens(response)

        return response["choices"][0]["message"]["content"]

    @staticmethod
    def _iter_tokens(response) -> Iterator[str]:
        """Yield text tokens from a streaming llama-cpp response."""
        for chunk in response:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                token = delta.get("content")
                if token:
                    yield token
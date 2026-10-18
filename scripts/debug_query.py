"""
Diagnóstico de uma pergunta: o que a busca entrega ao modelo.

Roda só a recuperação (não carrega o LLM), então é rápido e usa pouca memória.
Mostra cada chunk recuperado, quais cabem no contexto do prompt e se a
pergunta passaria na verificação de relevância temática.

Uso (na Pi, de preferência com a API parada para liberar memória):
    python scripts/debug_query.py "Qual foi o papel de Getúlio Vargas?"
    python scripts/debug_query.py "..." --top-k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings, setup_logging  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

_PREVIEW_CHARS = 350


def _prompt_status(text: str, prompt: str) -> str:
    """Diz se o chunk entrou inteiro, cortado ou ficou fora do prompt."""
    text = text.strip()
    if text and text in prompt:
        return "NO PROMPT"
    if text[:60] and text[:60] in prompt:
        return "CORTADO no prompt"
    return "FORA do prompt"


def build_report(
    query: str, chunks: list[dict], pipeline: RAGPipeline, max_context_chars: int
) -> str:
    lines = [f"PERGUNTA: {query}", f"Chunks recuperados: {len(chunks)}", ""]
    if not chunks:
        lines.append(
            "Nenhum chunk passou nos filtros: a resposta seria a padrão "
            "('não está no material')."
        )
        return "\n".join(lines)

    prompt = pipeline._build_system_prompt(chunks)
    for rank, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip()
        source = Path(chunk.get("source", "desconhecido")).name
        crumb = chunk.get("breadcrumb", "")
        lines.append(
            f"[#{rank}] score={chunk.get('score', 0.0):.3f}  "
            f"{_prompt_status(text, prompt)}  ({len(text)} chars)"
        )
        lines.append(f"     fonte: {source}{' | ' + crumb if crumb else ''}")
        preview = text[:_PREVIEW_CHARS].replace("\n", " ")
        lines.append(f"     {preview}{'…' if len(text) > _PREVIEW_CHARS else ''}")
        lines.append("")

    relevant = pipeline._is_topically_relevant(query, chunks)
    lines.append(f"MAX_CONTEXT_CHARS = {max_context_chars}")
    lines.append(
        "Verificação de relevância temática: "
        + ("passa (o LLM seria chamado)" if relevant else "FALHA (resposta padrão)")
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("query", help="Pergunta a diagnosticar")
    parser.add_argument(
        "--top-k", type=int, default=None, help="Sobrescreve TOP_K (opcional)"
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("WARNING")

    from scripts._eval_common import load_retriever_from_settings

    _, _, retriever = load_retriever_from_settings(settings)
    chunks = retriever.retrieve(args.query, k=args.top_k)

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=None,
        conv_manager=None,
        max_context_chars=settings.max_context_chars,
        max_history_turns=settings.max_history_turns,
    )
    print()
    print(build_report(args.query, chunks, pipeline, settings.max_context_chars))


if __name__ == "__main__":
    main()

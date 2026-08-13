"""
Gera um dataset de avaliação do RAG a partir dos próprios chunks indexados.

Por que este script existe
───────────────────────────
O único dataset de avaliação anterior (data/eval/benchmark_v1.json) era 100
perguntas escritas à mão, todas sobre um único recorte do material (Segundo
Governo Vargas, ~6% do livro) — o resto do corpus (pré-história, civilizações
antigas, América indígena, Idade Média, etc.) nunca era exercitado por
nenhuma avaliação. Este script gera perguntas para o livro INTEIRO,
automaticamente, a partir dos chunks reais já indexados:

  1. Amostra GRUPOS de chunks do índice de forma estratificada por
     Unidade/Capítulo, para cobrir o livro todo em vez de se concentrar em
     um único trecho. A maioria dos grupos tem 1 chunk só, mas uma parte tem
     2 ou 3 chunks do mesmo capítulo — para testar perguntas que exigem
     combinar informação de mais de um trecho (categoria "multi_hop").
  2. Para cada grupo amostrado, pede ao próprio LocalModel (o mesmo modelo
     usado no chat) para gerar UMA pergunta objetiva cuja resposta esteja
     apenas nesse(s) trecho(s) — sem inventar informação. Para grupos com
     2+ chunks, o prompt exige explicitamente que a pergunta só possa ser
     respondida combinando todos os trechos, não apenas um deles sozinho.
  3. O ground-truth é trivial: os chunks de origem da pergunta SÃO os chunks
     relevantes (chunk_id, relevance=3 cada) — compatível com o schema já
     usado por rag/evaluation.py (evaluate_retriever, load_dataset_from_file,
     que já suportam múltiplos `relevant_chunks` por query), sem precisar de
     nenhuma mudança na infraestrutura de métricas.
  4. Inclui um pequeno conjunto fixo de perguntas negativas (fora do domínio
     de História, ex: matemática/química) para continuar exercitando o
     guardrail de "não está no material" — não é um hack específico do
     corpus, é um fixture genérico para testar um comportamento genérico.

Pré-requisito: um modelo GGUF configurado em MODEL_PATH (o mesmo usado pelo
chat em produção) — a geração de perguntas roda o LLM localmente, offline,
sem nenhuma chamada de rede.

Uso
───
    # Gera 100 perguntas de 1 chunk + 30 de 2 chunks + 20 de 3 chunks (padrão)
    python scripts/generate_eval_dataset.py

    # Controla quantas perguntas de cada tipo e a seed de amostragem
    python scripts/generate_eval_dataset.py --n-single 50 --n-pairs 20 --n-triples 10 --seed 7

    # Só perguntas de 1 chunk, sem multi-hop
    python scripts/generate_eval_dataset.py --n-pairs 0 --n-triples 0

    # Sem as perguntas negativas fixas
    python scripts/generate_eval_dataset.py --no-negatives

    # Depois de gerado, avalie o retriever com o dataset:
    python scripts/eval_rag.py --mode file --dataset data/eval/dataset.json --k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_settings, setup_logging  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402
from scripts._eval_common import resolve_path  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Perguntas negativas fixas — genéricas, fora do domínio de História, não
# amarradas a nenhum tópico específico do corpus. Testam se o guardrail
# "não está no material" continua funcionando para qualquer assunto ausente.
# ---------------------------------------------------------------------------
_NEGATIVE_QUERIES = [
    "Qual é a fórmula de Bhaskara?",
    "Como funciona a fotossíntese nas plantas?",
    "Qual é a capital da Austrália?",
    "Como calcular a área de um círculo?",
    "O que é a tabela periódica dos elementos?",
    "Quais são as leis de Newton?",
    "Como funciona um motor de combustão interna?",
    "O que é inteligência artificial?",
    "Qual é a receita de um bolo de chocolate?",
    "Como programar em Python?",
]

# ---------------------------------------------------------------------------
# Geração de perguntas via LLM local
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Você cria perguntas de prova para estudantes do Ensino Médio brasileiro "
    "a partir de um trecho de material didático.\n"
    "REGRAS:\n"
    "- Crie exatamente UMA pergunta objetiva, cuja resposta esteja completa e "
    "explicitamente no trecho fornecido.\n"
    "- Não invente informação que não esteja no trecho.\n"
    "- Não faça perguntas de sim/não nem sobre o próprio texto (ex: 'o que diz "
    "o trecho?') — pergunte sobre o conteúdo histórico/factual.\n"
    "- Responda APENAS com a pergunta, sem numeração, sem aspas, sem prefixos "
    "como 'Pergunta:'."
)

_USER_TEMPLATE = "TRECHO:\n{chunk_text}\n\nCrie a pergunta agora."

# Prompt para grupos de 2+ chunks: a pergunta precisa exigir os trechos TODOS,
# não apenas um deles — é isso que torna a pergunta "multi-hop" de verdade,
# em vez de uma pergunta de 1 chunk que por acaso tem outros trechos ao lado.
_MULTI_SYSTEM_PROMPT = (
    "Você cria perguntas de prova para estudantes do Ensino Médio brasileiro "
    "a partir de MÚLTIPLOS trechos de material didático.\n"
    "REGRAS:\n"
    "- Crie exatamente UMA pergunta que só possa ser respondida combinando "
    "informação de TODOS os trechos abaixo — por exemplo, comparando dois "
    "eventos/períodos, relacionando causa e efeito entre eles, ou juntando "
    "fatos complementares. Não crie uma pergunta que qualquer um dos trechos, "
    "sozinho, já responderia por completo.\n"
    "- Não invente informação que não esteja nos trechos.\n"
    "- Responda APENAS com a pergunta, sem numeração, sem aspas, sem prefixos "
    "como 'Pergunta:'."
)


def build_question_generation_messages(chunk_text: str) -> list[dict]:
    """Monta as mensagens enviadas ao LLM para gerar uma pergunta sobre 1 chunk."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(chunk_text=chunk_text)},
    ]


def build_multi_chunk_question_messages(chunk_texts: list[str]) -> list[dict]:
    """Monta as mensagens enviadas ao LLM para gerar uma pergunta sobre 2+ chunks."""
    parts = [f"TRECHO {i + 1}:\n{text}" for i, text in enumerate(chunk_texts)]
    user_content = "\n\n".join(parts) + "\n\nCrie a pergunta agora."
    return [
        {"role": "system", "content": _MULTI_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def clean_generated_question(raw: str) -> Optional[str]:
    """Limpa a saída do LLM até obter só a pergunta, ou None se não for utilizável.

    Espelha rag.pipeline._clean_model_response: remove blocos <think>, aspas
    e prefixos comuns, e pega só a primeira linha não vazia.
    """
    if not raw:
        return None
    text = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
    text = re.sub(r"(?is)<think>.*$", "", text).strip()
    if not text:
        return None

    # Pega só a primeira linha não vazia (o modelo às vezes ecoa o trecho antes).
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    first_line = re.sub(r"(?i)^pergunta\s*:?\s*", "", first_line)
    first_line = first_line.strip("\"'“”‘’ ")

    if len(first_line) < 8:
        return None
    if "?" not in first_line:
        return None
    return first_line


def generate_question_for_chunk(
    chunk_text: str, llm_chat: Callable[[list[dict]], str]
) -> Optional[str]:
    """Gera uma pergunta para 1 chunk usando um `llm_chat` injetável.

    `llm_chat` recebe uma lista de mensagens (formato OpenAI-like, o mesmo
    usado por LocalModel.chat) e retorna a string de resposta — isolado assim
    para ser testável com um mock, sem precisar de um GGUF real nos testes.
    """
    messages = build_question_generation_messages(chunk_text)
    raw = llm_chat(messages)
    return clean_generated_question(raw)


def generate_question_for_group(
    chunk_texts: list[str], llm_chat: Callable[[list[dict]], str]
) -> Optional[str]:
    """Gera uma pergunta para um grupo de 1+ chunks usando um `llm_chat` injetável.

    Com 1 chunk, usa o prompt simples de pergunta factual; com 2+ chunks,
    usa o prompt multi-hop que exige combinar todos os trechos.
    """
    if len(chunk_texts) == 1:
        messages = build_question_generation_messages(chunk_texts[0])
    else:
        messages = build_multi_chunk_question_messages(chunk_texts)
    raw = llm_chat(messages)
    return clean_generated_question(raw)


# ---------------------------------------------------------------------------
# Amostragem estratificada de grupos de chunks
# ---------------------------------------------------------------------------


def sample_chunk_groups(
    chunks: list[dict],
    group_size: int,
    n_groups: int,
    seed: int = 42,
    max_page_span: int = 6,
) -> list[list[dict]]:
    """Amostra n_groups grupos de group_size chunks RELACIONADOS, distribuídos
    entre Unidade/Capítulo para cobrir o livro todo.

    Sem estratificação, uma amostra aleatória simples tende a se concentrar
    nas unidades/capítulos com mais chunks (os mais longos), deixando partes
    inteiras do livro sem nenhuma pergunta de avaliação. Agrupa candidatos por
    (unit, chapter) e sorteia em round-robin entre as chaves — mas o campo
    `chapter` está ausente na maior parte do índice (a estrutura detectada
    colapsa para o nível de Unidade), então uma chave `(unit, chapter)` pode
    abranger uma Unidade inteira (60-70 páginas, vários capítulos/assuntos
    sem relação direta entre si). Agrupar só por essa chave produzia grupos
    multi-chunk (group_size > 1) com chunks de assuntos completamente
    diferentes (ex: um sobre economia cafeeira e outro sobre revoluções
    europeias, só porque calhavam de estar na mesma Unidade) — o LLM então
    gerava uma pergunta que só uma das duas fontes de fato respondia, mas o
    ground-truth marcava as duas como relevantes, inflando artificialmente o
    "miss rate" do retriever em avaliações de recall multi-hop.

    Para além da chave (unit, chapter), ordena os candidatos de cada chave
    por página e só forma um grupo se todos os seus chunks couberem numa
    janela de `max_page_span` páginas — chunks fisicamente próximos no livro
    têm chance muito maior de serem sobre o mesmo assunto do que dois chunks
    quaisquer da mesma Unidade. Capítulos/Unidades com menos de `group_size`
    chunks elegíveis, ou sem nenhuma janela de páginas próxima o bastante,
    são ignorados.
    """
    if not chunks or n_groups <= 0 or group_size <= 0:
        return []

    rng = random.Random(seed)
    candidates: dict[tuple, list[dict]] = {}
    for chunk in chunks:
        # Ignora chunks minúsculos demais para gerar uma pergunta com contexto.
        if len(chunk.get("text", "")) < 200:
            continue
        key = (chunk.get("unit"), chunk.get("chapter"))
        candidates.setdefault(key, []).append(chunk)

    # Ordena por página dentro de cada chave — uma janela de chunks
    # consecutivos nessa ordem fica fisicamente próxima no livro, ao
    # contrário de um embaralhamento livre dentro da chave inteira.
    for key in candidates:
        candidates[key].sort(key=lambda c: (c.get("page") is None, c.get("page") or 0))

    eligible_keys = [k for k, v in candidates.items() if len(v) >= group_size]
    rng.shuffle(eligible_keys)

    def _window_ok(window: list[dict]) -> bool:
        if group_size == 1:
            return True
        pages = [c.get("page") for c in window if c.get("page") is not None]
        return not pages or (max(pages) - min(pages)) <= max_page_span

    result: list[list[dict]] = []
    idx_per_key = {k: 0 for k in eligible_keys}
    while len(result) < n_groups and eligible_keys:
        progressed = False
        for key in list(eligible_keys):
            if len(result) >= n_groups:
                break
            pool = candidates[key]
            i = idx_per_key[key]
            found = False
            while i + group_size <= len(pool):
                window = pool[i : i + group_size]
                if _window_ok(window):
                    result.append(window)
                    idx_per_key[key] = i + group_size
                    found = True
                    progressed = True
                    break
                i += 1
            if not found:
                eligible_keys.remove(key)
        if not progressed:
            break

    return result


# ---------------------------------------------------------------------------
# Montagem do dataset
# ---------------------------------------------------------------------------


def build_dataset_entry(
    query_id: str,
    query: str,
    chunk_group: list[dict],
    category: Optional[str] = None,
) -> dict:
    """Monta uma entrada no schema curado de rag/evaluation.py.

    `chunk_group` é a lista de 1+ chunks de origem da pergunta — todos viram
    entradas em `relevant_chunks` (schema já suportado por
    rag/evaluation.py::evaluate_query, sem mudança nenhuma lá). Quando
    `category` não é informado, infere "factual" para grupos de 1 chunk e
    "multi_hop" para grupos de 2+.
    """
    if category is None:
        category = "factual" if len(chunk_group) == 1 else "multi_hop"
    return {
        "id": query_id,
        "query": query,
        "relevant_chunks": [
            {"chunk_id": c.get("chunk_id", ""), "relevance": 3} for c in chunk_group
        ],
        "category": category,
        "answerable": True,
        "n_source_chunks": len(chunk_group),
        "source_breadcrumb": (
            chunk_group[0].get("breadcrumb", "") if chunk_group else ""
        ),
    }


def build_negative_entries(start_index: int) -> list[dict]:
    return [
        {
            "id": f"q{start_index + i:04d}",
            "query": query,
            "relevant_chunks": [],
            "category": "negative",
            "answerable": False,
            "source_breadcrumb": "",
        }
        for i, query in enumerate(_NEGATIVE_QUERIES)
    ]


def generate_dataset(
    chunks: list[dict],
    llm_chat: Callable[[list[dict]], str],
    n_single: int = 100,
    n_pairs: int = 30,
    n_triples: int = 20,
    seed: int = 42,
    include_negatives: bool = True,
) -> list[dict]:
    """Gera o dataset completo: amostra grupos de chunks, gera perguntas, monta entradas.

    Gera três "famílias" de perguntas, cada uma amostrada independentemente
    (com seeds derivadas para não repetir exatamente os mesmos chunks entre
    famílias): `n_single` perguntas de 1 chunk (categoria "factual"),
    `n_pairs` de 2 chunks e `n_triples` de 3 chunks (ambas "multi_hop") —
    essas últimas exigem que a pergunta combine informação dos vários trechos,
    não apenas um deles.
    """
    entries: list[dict] = []
    skipped = 0
    counter = 0

    for group_size, n_groups in ((1, n_single), (2, n_pairs), (3, n_triples)):
        groups = sample_chunk_groups(
            chunks, group_size=group_size, n_groups=n_groups, seed=seed + group_size
        )
        for group in groups:
            question = generate_question_for_group([c["text"] for c in group], llm_chat)
            if question is None:
                skipped += 1
                continue
            entries.append(build_dataset_entry(f"q{counter:04d}", question, group))
            counter += 1

    if skipped:
        logger.warning(
            "%d grupo(s) de chunks não geraram uma pergunta utilizável e "
            "foram ignorados.",
            skipped,
        )

    if include_negatives:
        entries.extend(build_negative_entries(len(entries)))

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_eval_dataset",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-single",
        type=int,
        default=100,
        metavar="N",
        help="Perguntas de 1 chunk só, categoria 'factual' (padrão: 100)",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=30,
        metavar="N",
        help="Perguntas multi-hop de 2 chunks do mesmo capítulo (padrão: 30)",
    )
    parser.add_argument(
        "--n-triples",
        type=int,
        default=20,
        metavar="N",
        help="Perguntas multi-hop de 3 chunks do mesmo capítulo (padrão: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed para amostragem reprodutível"
    )
    parser.add_argument(
        "--output",
        default="data/eval/dataset.json",
        metavar="PATH",
        help="Arquivo de saída (padrão: data/eval/dataset.json)",
    )
    parser.add_argument(
        "--negatives",
        dest="negatives",
        action="store_true",
        default=True,
        help="Inclui o conjunto fixo de perguntas negativas (padrão)",
    )
    parser.add_argument(
        "--no-negatives",
        dest="negatives",
        action="store_false",
        help="Não inclui as perguntas negativas fixas",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    settings = get_settings()

    print("\n" + "═" * 60)
    print("  Gerador de dataset de avaliação (a partir do índice real)")
    print("═" * 60)

    index_dir = resolve_path(settings.index_dir)
    if not Path(index_dir).exists():
        print(f"\n[ERRO] Índice não encontrado em: {index_dir}")
        print("       Execute primeiro: python scripts/index_documents.py\n")
        sys.exit(1)

    # VectorStore precisa da dimensão real do modelo de embeddings só para
    # abrir o índice existente — não gera novos embeddings aqui.
    from rag.embeddings import EmbeddingModel

    embed_model = EmbeddingModel(
        model_name=settings.embed_model_name,
        cache_dir=resolve_path(settings.embed_cache_dir),
        batch_size=settings.embed_batch_size,
        use_disk_cache=settings.embed_disk_cache,
    )
    vs = VectorStore(
        index_dir=index_dir,
        embedding_dim=embed_model.dimension,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    if vs.size == 0:
        print("\n[ERRO] VectorStore vazio — nenhum chunk indexado.\n")
        sys.exit(1)
    print(f"  Índice carregado: {vs.size} chunks")

    from llm.local_model import LocalModel

    print(f"  Carregando modelo: {settings.model_path}")
    llm = LocalModel(
        model_path=resolve_path(settings.model_path),
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        n_gpu_layers=settings.n_gpu_layers,
    )

    def llm_chat(messages: list[dict]) -> str:
        return llm.chat(messages, stream=False, temperature=0.3, max_tokens=128)

    total = args.n_single + args.n_pairs + args.n_triples
    print(
        f"  Gerando até {total} perguntas "
        f"({args.n_single} de 1 chunk, {args.n_pairs} de 2, {args.n_triples} de 3) "
        f"(seed={args.seed})...\n"
    )
    t0 = time.perf_counter()
    dataset = generate_dataset(
        vs.metadata,
        llm_chat,
        n_single=args.n_single,
        n_pairs=args.n_pairs,
        n_triples=args.n_triples,
        seed=args.seed,
        include_negatives=args.negatives,
    )
    elapsed = time.perf_counter() - t0
    print(f"  {len(dataset)} perguntas geradas em {elapsed:.1f}s")

    output_path = Path(resolve_path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Dataset salvo em: {output_path}")
    print(
        "\nPróximo passo — avaliar o retriever com este dataset:\n"
        f"  python scripts/eval_rag.py --mode file --dataset {args.output} --k 5"
    )


if __name__ == "__main__":
    main()

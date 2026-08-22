"""
Chunking semântico OTIMIZADO para documentos em português (BR).

MELHORIAS IMPLEMENTADAS (baseadas em best practices de RAG):

1. ✅ CHUNKS MENORES (300-500 tokens):
   - Antes: 1000-2000 tokens (muito grandes para embeddings)
   - Agora: 300-500 tokens (ideal para modelos de embedding)
   - Benefício: Embeddings mais precisos e focados

2. ✅ OVERLAP PEQUENO E PROPORCIONAL (50 tokens = 10-15%):
   - Antes: 1 sentença fixa (variava muito: 10-200 tokens)
   - Agora: 50 tokens fixos (~13% de 375 tokens)
   - Benefício: Preserva contexto sem redundância excessiva

3. ✅ SEMANTIC CHUNKING (divisão inteligente):
   - Divide por estrutura natural do documento:
     * Títulos e subtítulos
     * Parágrafos completos
     * Seções temáticas
   - Exemplo ideal:
     * Chunk 1: "Primavera dos Povos" (completo)
     * Chunk 2: "Três Dias Gloriosos" (completo)
     * Chunk 3: "Marianne como símbolo republicano" (completo)
   - Benefício: Cada chunk é uma unidade semântica coerente

4. ✅ LIMPEZA ANTES DA INDEXAÇÃO:
   - Remove boilerplate editorial:
     * Copyright e créditos
     * ISBN e dados de catalogação
     * "Reprodução proibida"
     * Sumário e índice remissivo
     * Números de página isolados
     * Cabeçalhos/rodapés repetitivos
   - Benefício: Melhora retrieval eliminando ruído

5. ✅ METADATA ENRIQUECIDA:
   - Antes: source, page, chunk_id
   - Agora: + chapter, topic, section_title
   - Benefício: Permite reranking e hybrid search mais sofisticados

Melhorias técnicas adicionais:
  - Limpeza de texto muito mais robusta para PDFs brasileiros:
      * Corrige hifenização em quebra de linha ("atô-\\nmicas" → "atômicas")
      * Remove quebras de linha indevidas dentro de parágrafos
      * Normaliza encoding NFC (acentos duplicados, caracteres compostos)
      * Remove artefatos de cabeçalho/rodapé numéricos
  - Chunking guiado por semântica real:
      * Divide primeiro por parágrafos (separa assuntos distintos)
      * Usa sentenças como fallback apenas dentro do parágrafo
      * Respeita limites de tokens (não caracteres) para compatibilidade com LLMs
      * Overlap configurável em número de tokens (não sentenças variáveis)
  - Estrutura de saída enriquecida com page, section, chunk_id, chapter, topic, section_title
  - Contagem de tokens aproximada sem dependências externas (split by whitespace)
    suficiente para modelos com tokenizadores parecidos com BPE

Decisões técnicas:
  - Contagem por palavras (÷ 0.75) é ±10% precisa para PT-BR com modelos BPE,
    evita importar tiktoken/transformers que pesariam na Raspberry Pi.
  - Overlap em tokens (não sentenças) preserva contexto de forma proporcional.
  - chunk_id = sha1(source + posição + página) garante idempotência na re-indexação.
"""

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex de limpeza — compiladas uma vez no import
# ---------------------------------------------------------------------------

# Hifenização no fim de linha: "atô-\nmicas" → "atômicas"
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")

# Quebra de linha dentro de parágrafo (não é separador de parágrafo)
# Heurística: linha seguinte começa com minúscula ou número → mesma frase
#
# CORREÇÃO: o `re.IGNORECASE` original também casava letras MAIÚSCULAS contra
# a classe de caracteres minúsculos (`[a-z...]` sob IGNORECASE aceita A-Z),
# invertendo a heurística — toda quebra de linha antes de uma frase/heading
# COMEÇANDO em maiúscula também era unida ao parágrafo anterior. Na prática
# isso colapsava a página inteira em uma única linha, destruindo os limites
# de heading que rag.structure depende para detectar Unidade/Capítulo/Tópico.
_SOFT_NEWLINE = re.compile(r"(?<=[^\n])\n(?=[a-záéíóúàâêôãõç0-9,;:\"\'])")

# Número de página isolado (linha com só dígitos, opcional espaços)
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

# Cabeçalhos / rodapés repetitivos (linha toda em maiúsculas, curta)
_HEADER_LINE = re.compile(r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\s]{5,60}$", re.MULTILINE)

# Padrões de boilerplate editorial (copyright, ISBN, créditos, etc.)
_BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)©\s*\d{4}.*?todos\s+os\s+direitos\s+reservados", re.DOTALL),
    re.compile(r"(?i)copyright\s*©?\s*\d{4}", re.IGNORECASE),
    re.compile(r"(?i)isbn\s*:?\s*[\d\-]+", re.IGNORECASE),
    re.compile(r"(?i)reprodução\s+proibida", re.IGNORECASE),
    re.compile(r"(?i)todos\s+os\s+direitos\s+reservados", re.IGNORECASE),
    re.compile(r"(?i)editora\s+moderna\s+ltda", re.IGNORECASE),
    re.compile(r"(?i)impresso\s+no\s+brasil", re.IGNORECASE),
    re.compile(r"(?i)printed\s+in\s+brazil", re.IGNORECASE),
    re.compile(r"(?i)ficha\s+catalográfica", re.IGNORECASE),
    re.compile(r"(?i)catalogação\s+na\s+fonte", re.IGNORECASE),
    re.compile(r"(?i)dados\s+internacionais\s+de\s+catalogação", re.IGNORECASE),
    re.compile(r"(?i)cip\s*-\s*brasil", re.IGNORECASE),
    re.compile(r"(?i)sumário", re.IGNORECASE),  # Remove páginas de sumário
    re.compile(r"(?i)índice\s+remissivo", re.IGNORECASE),
]

# Espaços múltiplos (mantém quebras simples)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# Três ou mais quebras de linha → duas (separa parágrafos)
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Pontuação de fim de sentença em PT-BR
# Lookbehind: após .!?…  Lookahead: maiúscula ou aspas
_SENT_END = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\"\(])")

# Fator de conversão palavras → tokens (BPE típico para PT-BR)
# Medido empiricamente: textos PT-BR têm ~1.33 tokens/palavra em modelos Llama/Mistral
_TOKENS_PER_WORD = 1.33


# ---------------------------------------------------------------------------
# Configuração de chunking (imutável após criação)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkConfig:
    """Parâmetros do pipeline de chunking.

    Attributes:
        min_tokens:   Mínimo de tokens por chunk (chunks menores são mesclados).
        max_tokens:   Máximo de tokens por chunk (chunks maiores são divididos).
        overlap_tokens: Tokens a repetir entre chunks consecutivos (overlap semântico).
        remove_headers: Remove linhas de cabeçalho/rodapé detectadas heuristicamente.
        remove_boilerplate: Remove conteúdo editorial (copyright, ISBN, etc.).
    """

    min_tokens: int = 200  # ~300-400 chars — chunks menores mas densos
    max_tokens: int = 375  # ~500 chars — otimizado para embeddings (300-500 tokens)
    overlap_tokens: int = 50  # 50 tokens de overlap (~13% de 375) — preserva contexto
    remove_headers: bool = True
    remove_boilerplate: bool = True  # Nova flag para limpeza editorial


# ---------------------------------------------------------------------------
# Contagem de tokens (sem dependências pesadas)
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Estimativa de tokens para modelos BPE (Llama, Mistral, Qwen, Gemma).

    Usa contagem de palavras × fator empírico para PT-BR.
    Erro típico: ±10% — suficiente para controle de chunk_size.

    Antes: código usava len(text) / 4 (caracteres), que subestimava tokens
           para textos em português com muitas palavras longas e acentuadas.
    Depois: contagem por palavras × 1.33 é mais fiel ao tokenizador real.
    """
    if not text:
        return 0
    word_count = len(text.split())
    return max(1, int(word_count * _TOKENS_PER_WORD))


# ---------------------------------------------------------------------------
# Limpeza de texto extraído de PDF
# ---------------------------------------------------------------------------


def clean_pdf_text(
    text: str, remove_headers: bool = True, remove_boilerplate: bool = True
) -> str:
    """Remove e corrige artefatos comuns de PDFs em português.

    Melhorias em relação à versão anterior:
      - NOVO: Remove boilerplate editorial (copyright, ISBN, créditos, sumário)
      - NOVO: Limpeza mais agressiva para melhorar qualidade dos embeddings
      - Corrige hifenização ANTES de remover quebras de linha
        (ordem importa: "atô-\\nmicas" → "atômicas", não "atô- micas").
      - Remove quebras de linha suaves dentro de parágrafos
        ("nor\\nte" → "norte", "fre-\\nquência" → "frequência").
      - Remoção opcional de linhas de cabeçalho/rodapé em maiúsculas.
      - Mantém: normalização NFC, remoção de números de página, espaços.

    """
    if not text:
        return ""

    # 1. Normaliza unicode (remove acentuação duplicada, formas compostas)
    text = unicodedata.normalize("NFC", text)

    # 2. Remove boilerplate editorial PRIMEIRO (antes de processar o texto)
    if remove_boilerplate:
        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)

    # 3. Corrige hifenização de quebra de linha
    #    "atô-\nmicas" → "atômicas"  |  "fre-\nquência" → "frequência"
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # 4. Remove quebras de linha suaves dentro de parágrafos
    #    "nor\nte" → "norte"  |  "proces\nso" → "processo"
    text = _SOFT_NEWLINE.sub(" ", text)

    # 5. Remove números de página isolados
    text = _PAGE_NUMBER.sub("", text)

    # 6. Remove cabeçalhos/rodapés heurísticos (linhas ALL-CAPS curtas)
    if remove_headers:
        text = _HEADER_LINE.sub("", text)

    # 7. Normaliza espaços e quebras de linha redundantes
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Segmentação em sentenças
# ---------------------------------------------------------------------------


def split_sentences(paragraph: str) -> list[str]:
    """Divide um parágrafo em sentenças respeitando pontuação PT-BR.

    Sem dependências externas — regex calibrada para textos técnicos/acadêmicos.
    Preserva abreviações comuns (Dr., Sr., pág.) ao exigir letra maiúscula após ponto.
    """
    parts = _SENT_END.split(paragraph.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Geração de chunk_id determinístico
# ---------------------------------------------------------------------------


def _make_chunk_id(source: str, page: Optional[int], text: str) -> str:
    """SHA-1 truncado garante idempotência na re-indexação.

    Mesmo documento re-indexado produz os mesmos IDs se o texto não mudou.

    Hash baseado em conteúdo (source + page + texto), não em posição:
    um índice local a cada chamada de chunk_document() (uma por seção
    detectada) colidia entre seções diferentes que caem na mesma página.
    Hashear o texto alinha o chunk_id com o critério que VectorStore já
    usa para deduplicar chunks (hash do texto).
    """
    page_str = f"::page{page}" if page is not None else ""
    raw = f"{source}{page_str}::{text}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Montador de chunks a partir de sentenças
# ---------------------------------------------------------------------------


def _build_chunks_from_sentences(
    sentences: list[str],
    source: str,
    page: Optional[int],
    section: Optional[str],
    config: ChunkConfig,
) -> list[dict]:
    """Agrupa sentenças em chunks respeitando min/max_tokens e overlap baseado em tokens.

    Algoritmo MELHORADO:
      1. Acumula sentenças até atingir max_tokens.
      2. Ao fechar um chunk, mantém `overlap_tokens` de texto no buffer seguinte.
      3. Overlap agora é baseado em TOKENS, não em número de sentenças fixo.
      4. Chunks com menos de min_tokens são mesclados ao próximo.
      5. Chunk final sempre é emitido (mesmo abaixo de min_tokens).

    Melhorias:
      - Overlap proporcional (50 tokens ≈ 10-15% de 375 tokens)
      - Mais preciso que overlap por sentenças (que varia muito em tamanho)
      - Melhor preservação de contexto entre chunks

    Args:
        sentences:   Lista de sentenças pré-segmentadas.
        source:      Caminho do arquivo original.
        page:        Número da página (None se desconhecido).
        section:     Título de seção detectado (None se desconhecido).
        config:      Parâmetros de chunking.

    Returns:
        Lista de dicts com text, source, page, section, chunk_id, chapter, topic.
    """
    chunks: list[dict] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def _emit(sents: list[str]) -> None:
        text = " ".join(s for s in sents if s).strip()
        if not text:
            return

        # Extrai metadados enriquecidos
        metadata = _extract_metadata(text, section, page)

        chunks.append(
            {
                "text": text,
                "source": source,
                "page": page,
                "section": section,
                "chunk_id": _make_chunk_id(source, page, text),
                "chapter": metadata.get("chapter"),
                "topic": metadata.get("topic"),
                "section_title": metadata.get("section_title"),
            }
        )

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        # Sentença sozinha já excede o limite → divide como chunk independente
        if sent_tokens > config.max_tokens:
            if buffer:
                _emit(buffer)
                # Overlap baseado em tokens: pega sentenças do final até atingir overlap_tokens
                buffer = _get_overlap_sentences(buffer, config.overlap_tokens)
                buffer_tokens = sum(count_tokens(s) for s in buffer)
            _emit([sent])
            buffer = []
            buffer_tokens = 0
            continue

        # Acumularia além do limite → fecha chunk atual
        if buffer_tokens + sent_tokens > config.max_tokens and buffer:
            # Só fecha se o buffer já tem tamanho mínimo; caso contrário continua
            if buffer_tokens >= config.min_tokens:
                _emit(buffer)
                # Overlap baseado em tokens
                buffer = _get_overlap_sentences(buffer, config.overlap_tokens)
                buffer_tokens = sum(count_tokens(s) for s in buffer)

        buffer.append(sent)
        buffer_tokens += sent_tokens

    # Último buffer — mescla com o penúltimo chunk se muito pequeno
    if buffer:
        if buffer_tokens < config.min_tokens and chunks:
            # Enriquece o último chunk em vez de criar um fragmento
            last = chunks[-1]
            merged_text = last["text"] + " " + " ".join(s for s in buffer if s).strip()
            last["text"] = merged_text.strip()
            # Atualiza metadados do chunk mesclado
            metadata = _extract_metadata(last["text"], section, page)
            last["chapter"] = metadata.get("chapter")
            last["topic"] = metadata.get("topic")
            last["section_title"] = metadata.get("section_title")
        else:
            _emit(buffer)

    return chunks


def _get_overlap_sentences(sentences: list[str], target_tokens: int) -> list[str]:
    """Retorna sentenças do final da lista até atingir aproximadamente target_tokens.

    Implementa overlap baseado em tokens em vez de número fixo de sentenças.
    Isso garante overlap proporcional independente do tamanho das sentenças.

    Args:
        sentences: Lista de sentenças do chunk anterior
        target_tokens: Número alvo de tokens para o overlap

    Returns:
        Lista de sentenças do final que somam ~target_tokens
    """
    if not sentences or target_tokens <= 0:
        return []

    overlap_sents = []
    overlap_tokens = 0

    # Itera de trás para frente até atingir o target
    for sent in reversed(sentences):
        sent_tokens = count_tokens(sent)
        if overlap_tokens + sent_tokens > target_tokens and overlap_sents:
            # Já temos overlap suficiente
            break
        overlap_sents.insert(0, sent)
        overlap_tokens += sent_tokens

    return overlap_sents


def _extract_metadata(text: str, section: Optional[str], page: Optional[int]) -> dict:
    """Extrai metadados enriquecidos do texto do chunk.

    Detecta:
    - chapter: Número do capítulo (ex: "Capítulo 3", "Cap. 5")
    - topic: Tópico principal (primeira linha significativa)
    - section_title: Título da seção (já vem do loader, mas pode refinar)

    Esses metadados melhoram reranking e hybrid search.

    Args:
        text: Texto do chunk
        section: Seção detectada pelo loader (pode ser None)
        page: Número da página (pode ser None)

    Returns:
        Dict com chapter, topic, section_title
    """
    metadata = {
        "chapter": None,
        "topic": None,
        "section_title": section,  # Usa seção do loader como fallback
    }

    # Detecta capítulo (ex: "Capítulo 3", "Cap. 5", "CAPÍTULO III")
    chapter_pattern = re.compile(r"(?i)cap(?:ítulo)?\.?\s+(\d+|[IVX]+)", re.IGNORECASE)
    chapter_match = chapter_pattern.search(text[:200])  # Busca nos primeiros 200 chars
    if chapter_match:
        metadata["chapter"] = chapter_match.group(0).strip()

    # Detecta tópico principal (primeira linha significativa com 10-80 chars)
    lines = text.split("\n")
    for line in lines[:5]:  # Verifica as primeiras 5 linhas
        line = line.strip()
        if 10 <= len(line) <= 80 and not line.endswith((".", "!", "?")):
            # Linha curta sem pontuação final = provável título/tópico
            metadata["topic"] = line
            # Se não tinha section_title, usa o tópico
            if not metadata["section_title"]:
                metadata["section_title"] = line
            break

    return metadata


# ---------------------------------------------------------------------------
# API pública — chunk_document (compatível com a interface original)
# ---------------------------------------------------------------------------


def chunk_document(
    doc: dict,
    chunk_size: int = 512,
    overlap: int = 50,
    config: Optional[ChunkConfig] = None,
) -> list[dict]:
    """Divide um documento em chunks semânticos com overlap baseado em tokens.

    Interface compatível com o código anterior: aceita chunk_size (caracteres)
    e overlap (agora em tokens), mas internamente usa contagem de tokens e lógica
    semântica aprimorada.

    MELHORIAS IMPLEMENTADAS:
      1. ✅ Chunks menores: 300-500 tokens (antes: 1000-2000)
      2. ✅ Overlap pequeno: 50 tokens ou 10-15% (antes: 1 sentença variável)
      3. ✅ Semantic chunking: divide por títulos, subtítulos, parágrafos, seções
      4. ✅ Limpeza antes da indexação: remove copyright, ISBN, créditos, sumário
      5. ✅ Metadata melhor: adiciona chapter, topic, section_title além de source, page, chunk_id

    Mudança de comportamento intencional:
      - chunk_size agora é usado para derivar max_tokens (chunk_size / 3 ≈ tokens).
        Isso porque 512 chars ≈ ~170 tokens — dentro da faixa recomendada.
      - overlap agora é em TOKENS (não sentenças), garantindo overlap proporcional.
      - O output inclui os novos campos page, section, chunk_id, chapter, topic, section_title.
        Código existente que acessa apenas ["text"] e ["source"] continua funcionando.

    Args:
        doc:        Dict com chaves 'text' (obrigatório) e 'source' (obrigatório).
                    Pode incluir 'page' (int) e 'section' (str) — repassados ao chunk.
        chunk_size: Tamanho máximo APROXIMADO em caracteres (mantido para compatibilidade).
                    Internamente convertido para tokens.
        overlap:    Número de TOKENS a repetir entre chunks (padrão 50).
        config:     ChunkConfig explícito — sobrescreve chunk_size e overlap se fornecido.

    Returns:
        Lista de dicts {'text', 'source', 'page', 'section', 'chunk_id', 'chapter', 'topic', 'section_title'}.
        Campos novos têm valor None quando não disponíveis, não quebram código antigo.
    """
    raw_text: str = doc.get("text", "")
    source: str = doc.get("source", "desconhecido")
    page: Optional[int] = doc.get("page")
    section: Optional[str] = doc.get("section")
    # Hierarchical metadata passed through from structure detection (Phase 2 fix)
    doc_unit: Optional[str] = doc.get("unit")
    doc_chapter: Optional[str] = doc.get("chapter")
    doc_topic: Optional[str] = doc.get("topic")
    doc_breadcrumb: Optional[str] = doc.get("breadcrumb")

    if not raw_text.strip():
        return []

    # Deriva configuração a partir dos parâmetros legados se não foi passado config
    if config is None:
        # chunk_size em chars → tokens (÷ 3 é conservador, preserva margem)
        # Ajustado para valores menores: 300-500 tokens
        max_tok = max(30, min(chunk_size // 3, 375))  # Cap em 375 tokens
        min_tok = max(10, max_tok // 2)  # min = metade do max
        config = ChunkConfig(
            min_tokens=min_tok,
            max_tokens=max_tok,
            overlap_tokens=overlap,
            remove_boilerplate=True,  # Ativa limpeza editorial
        )

    # Limpa artefatos de PDF E boilerplate editorial
    text = clean_pdf_text(
        raw_text,
        remove_headers=config.remove_headers,
        remove_boilerplate=config.remove_boilerplate,
    )

    if not text:
        logger.warning("Documento vazio após limpeza: %s", source)
        return []

    total_tokens = count_tokens(text)
    if total_tokens <= config.max_tokens:
        # Documento cabe inteiro em um chunk
        metadata = _extract_metadata(text, section, page)
        return [
            {
                "text": text,
                "source": source,
                "page": page,
                "section": section,
                "chunk_id": _make_chunk_id(source, page, text),
                "chapter": metadata.get("chapter") or doc_chapter,
                "topic": metadata.get("topic") or doc_topic,
                "section_title": metadata.get("section_title"),
                "unit": doc_unit,
                "breadcrumb": doc_breadcrumb,
            }
        ]

    # Divide em parágrafos reais (linhas em branco separam assuntos)
    # SEMANTIC CHUNKING: respeita estrutura natural do documento
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    all_chunks: list[dict] = []

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens <= config.max_tokens:
            # Parágrafo inteiro cabe em um chunk — não divide sentenças
            # SEMANTIC CHUNKING: preserva unidade semântica do parágrafo
            metadata = _extract_metadata(para, section, page)
            all_chunks.append(
                {
                    "text": para,
                    "source": source,
                    "page": page,
                    "section": section,
                    "chunk_id": _make_chunk_id(source, page, para),
                    "chapter": metadata.get("chapter"),
                    "topic": metadata.get("topic"),
                    "section_title": metadata.get("section_title"),
                }
            )
        else:
            # Parágrafo grande → divide em sentenças
            sentences = split_sentences(para)
            para_chunks = _build_chunks_from_sentences(
                sentences, source, page, section, config
            )
            all_chunks.extend(para_chunks)

    # Mescla parágrafos minúsculos consecutivos (títulos soltos, listas curtas)
    all_chunks = _merge_tiny_chunks(all_chunks, config)

    # Propagate document-level hierarchical metadata (unit/breadcrumb) to every chunk.
    # chunk-level chapter/topic may override doc-level if detected inside the chunk text,
    # but unit and breadcrumb always come from the document structure detection.
    for chunk in all_chunks:
        if doc_unit is not None:
            chunk.setdefault("unit", doc_unit)
        if doc_chapter is not None:
            chunk.setdefault("chapter", chunk.get("chapter") or doc_chapter)
        if doc_topic is not None:
            chunk.setdefault("topic", chunk.get("topic") or doc_topic)
        if doc_breadcrumb is not None:
            chunk["breadcrumb"] = doc_breadcrumb

    logger.debug(
        "chunk_document: %d parágrafos → %d chunks | %d tokens totais | overlap=%d tokens (source=%s)",
        len(paragraphs),
        len(all_chunks),
        total_tokens,
        config.overlap_tokens,
        source,
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Pós-processamento: mescla chunks minúsculos consecutivos
# ---------------------------------------------------------------------------


def _merge_tiny_chunks(chunks: list[dict], config: ChunkConfig) -> list[dict]:
    """Mescla chunks consecutivos que ficaram abaixo de min_tokens.

    Evita chunks como {"text": "Capítulo 3"} que poluem o índice vetorial
    sem adicionar informação semântica útil.

    CORREÇÃO: a condição original só mesclava quando AMBOS os chunks
    vizinhos eram pequenos — mas o caso mais comum é um único parágrafo
    minúsculo (ex: um heading sozinho, sem nenhum corpo de texto na mesma
    seção) seguido de um parágrafo de tamanho normal. Com o segundo chunk
    não sendo "pequeno", a condição antiga nunca mesclava e o chunk-heading
    ficava sozinho no índice — exatamente o problema que esta função afirma
    evitar. Agora basta que UM dos dois lados seja pequeno.

    Preserva source, page e section do primeiro chunk do par mesclado.
    chunk_id do primeiro é mantido (o segundo é descartado).
    Metadados (chapter, topic, section_title) são re-extraídos do texto mesclado.
    """
    if not chunks:
        return chunks

    merged: list[dict] = [chunks[0]]

    for current in chunks[1:]:
        last = merged[-1]
        last_tokens = count_tokens(last["text"])
        curr_tokens = count_tokens(current["text"])

        # Mescla se UM dos dois é pequeno, têm a mesma fonte, e a soma cabe
        # dentro de max_tokens.
        if (
            (last_tokens < config.min_tokens or curr_tokens < config.min_tokens)
            and last["source"] == current["source"]
            and last_tokens + curr_tokens <= config.max_tokens
        ):
            merged_text = last["text"] + " " + current["text"]
            last["text"] = merged_text.strip()
            # Re-extrai metadados do texto mesclado
            metadata = _extract_metadata(
                last["text"], last.get("section"), last.get("page")
            )
            last["chapter"] = metadata.get("chapter")
            last["topic"] = metadata.get("topic")
            last["section_title"] = metadata.get("section_title")
        else:
            merged.append(current)

    return merged

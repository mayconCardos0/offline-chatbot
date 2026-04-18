"""
Chunking semântico para documentos em português (BR).

Estratégia:
  1. Divide o texto em parágrafos reais (linhas em branco).
  2. Dentro de cada parágrafo, respeita limites de sentença usando pontuação PT-BR.
  3. Mescla parágrafos pequenos com o seguinte para evitar chunks minúsculos.
  4. Aplica overlap em nível de sentença (não de caracter) para preservar contexto.
  5. Limpa artefatos comuns de PDF (hifenação, cabeçalhos/rodapés numéricos).
"""
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Pontuação de fim de sentença em português
_SENT_END = re.compile(r'(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\"\'])')

# Artefatos de PDF: número de página isolado, hifenação no fim de linha
_PAGE_NUMBER   = re.compile(r'^\s*\d{1,4}\s*$', re.MULTILINE)
_HYPHEN_BREAK  = re.compile(r'(\w)-\n(\w)')
_MULTI_SPACE   = re.compile(r'[ \t]{2,}')
_MULTI_NEWLINE = re.compile(r'\n{3,}')


# ---------------------------------------------------------------------------
# Limpeza de texto extraído de PDF
# ---------------------------------------------------------------------------

def clean_pdf_text(text: str) -> str:
    """Remove artefatos comuns de PDFs em português."""
    # Normaliza unicode (remove acentuação duplicada etc.)
    text = unicodedata.normalize("NFC", text)
    # Remove números de página isolados
    text = _PAGE_NUMBER.sub('\n', text)
    # Junta palavras hifenadas que quebraram na linha
    text = _HYPHEN_BREAK.sub(r'\1\2', text)
    # Normaliza espaços
    text = _MULTI_SPACE.sub(' ', text)
    # Máximo duas quebras de linha consecutivas
    text = _MULTI_NEWLINE.sub('\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Segmentação em sentenças (sem dependência externa)
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Divide um parágrafo em sentenças respeitando pontuação PT-BR."""
    parts = _SENT_END.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Chunking principal
# ---------------------------------------------------------------------------

def chunk_document(doc: dict, chunk_size: int = 512, overlap: int = 1) -> list[dict]:
    """Divide um documento em chunks semânticos com overlap em nível de sentença.

    Args:
        doc:        Dict com chaves 'text' e 'source'.
        chunk_size: Tamanho máximo do chunk em caracteres.
        overlap:    Número de **sentenças** a repetir entre chunks consecutivos
                    (padrão 1 sentença de sobreposição).

    Returns:
        Lista de dicts {'text': str, 'source': str}.
    """
    raw_text: str = doc["text"]
    source: str   = doc["source"]

    if not raw_text:
        return []

    text = clean_pdf_text(raw_text)

    if len(text) <= chunk_size:
        return [{"text": text, "source": source}]

    # 1. Divide em parágrafos
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

    # 2. Divide cada parágrafo em sentenças
    all_sentences: list[str] = []
    for para in paragraphs:
        sents = split_sentences(para)
        all_sentences.extend(sents)
        # Marca fim de parágrafo para o montador não misturar contextos distintos
        # (usa sentinela vazia — filtrada depois)
        all_sentences.append("")

    # 3. Monta chunks respeitando chunk_size
    chunks: list[dict] = []
    current_sents: list[str] = []
    current_len = 0

    def _flush(sents: list[str]) -> None:
        text_chunk = " ".join(s for s in sents if s).strip()
        if text_chunk:
            chunks.append({"text": text_chunk, "source": source})

    for sent in all_sentences:
        if not sent:
            # Fim de parágrafo: fecha chunk se estiver razoavelmente cheio
            if current_len >= chunk_size // 2:
                _flush(current_sents)
                # Mantém overlap de sentenças não-vazias
                non_empty = [s for s in current_sents if s]
                current_sents = non_empty[-overlap:] if overlap else []
                current_len   = sum(len(s) for s in current_sents)
            continue

        sent_len = len(sent)

        if current_len + sent_len > chunk_size and current_sents:
            _flush(current_sents)
            non_empty = [s for s in current_sents if s]
            current_sents = non_empty[-overlap:] if overlap else []
            current_len   = sum(len(s) for s in current_sents)

        current_sents.append(sent)
        current_len += sent_len

    # Último chunk
    if current_sents:
        _flush(current_sents)

    logger.debug(
        "chunk_document: %d sentenças → %d chunks (source=%s)",
        len(all_sentences), len(chunks), source
    )
    return chunks

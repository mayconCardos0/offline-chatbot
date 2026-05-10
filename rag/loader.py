"""
Document loader com suporte a .txt, .md, .pdf e .json.

Melhorias em relação à versão anterior:
  - PDFs: rastreia número de página por chunk (campo 'page' no output).
  - PDFs: detecta títulos/seções heuristicamente (campo 'section' no output).
  - PDFs: retorna lista de dicts por página em vez de texto único concatenado,
    permitindo que chunk_document saiba a origem de cada pedaço.
  - .txt/.md: detecção automática de encoding (utf-8 → latin-1 → cp1252).
  - .json: extração recursiva de todas as strings aninhadas (sem mudança).
  - Separação de responsabilidades: _load_pdf_pages() retorna páginas estruturadas;
    load_documents() decide se concatena ou não conforme o tipo de arquivo.

Decisão técnica — por que retornar por página e não por documento inteiro?
  Saber a página permite:
    (a) citações mais precisas na resposta do LLM ("ver pág. 12")
    (b) debug facilitado quando um chunk tem texto corrompido
    (c) futuro suporte a highlighting no frontend
  Custo: load_documents() agora pode retornar mais de um dict por PDF.
  Compatibilidade: cada dict ainda tem 'text' e 'source', então o pipeline
  existente funciona sem alteração.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json"}

# Heurística de detecção de título/seção:
#   linha com até 80 chars, sem ponto final, que começa com maiúscula ou número
import re

_SECTION_PATTERN = re.compile(
    r"^(?:\d+[\.\)]\s+)?[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][^\n]{3,79}(?<![.!?,;:])$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Loaders individuais
# ---------------------------------------------------------------------------


def _load_txt(path: Path) -> str:
    """Carrega arquivo de texto com detecção automática de encoding."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    # Último recurso: substitui bytes inválidos
    return path.read_text(encoding="utf-8", errors="replace")


def _detect_section(text: str) -> Optional[str]:
    """Extrai o primeiro título/seção encontrado em um bloco de texto.

    Heurística: linha curta, sem pontuação final, começa com maiúscula ou número.
    Retorna None se nenhum título for encontrado.

    Exemplo:
      "3. Energia Nuclear\\nA fissão é o processo..." → "3. Energia Nuclear"
    """
    match = _SECTION_PATTERN.search(text)
    if match:
        candidate = match.group(0).strip()
        # Ignora linhas que são claramente sentenças incompletas (têm vírgula no meio)
        if len(candidate.split()) <= 12:
            return candidate
    return None


def _load_pdf_pages(path: Path) -> list[dict]:
    """Extrai texto de PDF página a página usando PyMuPDF (fitz).

    Mantém a mesma estrutura da versão anterior, porém com melhorias:
      - Melhor suporte a texto rotacionado
      - Melhor preservação da ordem de leitura
      - Extração mais confiável para RAG
    """
    import fitz  # PyMuPDF
    from rag.chunking import clean_pdf_text

    pages: list[dict] = []

    # Abre o documento
    doc = fitz.open(str(path))

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1  # manter 1-based

        try:
            # Extração principal (mais robusta que pypdf)
            raw_text = page.get_text("text") or ""
        except Exception:
            # fallback simples
            raw_text = page.get_text() or ""

        # Limpeza (mantém seu pipeline atual)
        cleaned = clean_pdf_text(raw_text)

        if not cleaned:
            continue

        # Detecta seção (mantido)
        section = _detect_section(cleaned)

        pages.append(
            {
                "text": cleaned,
                "source": str(path),
                "page": page_num,
                "section": section,
            }
        )

    doc.close()

    return pages


def _load_json(path: Path) -> str:
    """Extrai todas as strings de um JSON aninhado (sem alteração)."""
    data = json.loads(path.read_text(encoding="utf-8"))

    def _extract(obj) -> list[str]:
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, dict):
            return [s for v in obj.values() for s in _extract(v)]
        if isinstance(obj, list):
            return [s for item in obj for s in _extract(item)]
        return []

    return " ".join(_extract(data))


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def load_documents(docs_dir: str) -> list[dict]:
    """Carrega todos os documentos suportados de docs_dir.

    Mudança em relação à versão anterior:
      - PDFs agora retornam múltiplos dicts (um por página) em vez de um único
        dict com todo o texto concatenado. Isso permite rastrear page e section
        por chunk, melhorando a qualidade das citações no RAG.
      - Outros formatos (.txt, .md, .json) continuam retornando um dict por arquivo.

    Returns:
        Lista de dicts com chaves:
          'text'    (str)       — conteúdo textual,
          'source'  (str)       — caminho do arquivo,
          'page'    (int|None)  — número de página (PDFs) ou None,
          'section' (str|None)  — título de seção detectado ou None.
    """
    docs_path = Path(docs_dir)
    results: list[dict] = []

    if not docs_path.exists():
        logger.warning("Diretório de documentos não existe: %s", docs_dir)
        return results

    for entry in sorted(docs_path.iterdir()):
        if not entry.is_file():
            continue

        ext = entry.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning("Formato não suportado '%s': %s", ext, entry)
            continue

        try:
            if ext in {".txt", ".md"}:
                text = _load_txt(entry)
                if not text.strip():
                    logger.warning("Documento vazio após extração: %s", entry)
                    continue
                results.append(
                    {
                        "text": text,
                        "source": str(entry),
                        "page": None,
                        "section": None,
                    }
                )
                logger.info("Carregado: %s (%d chars)", entry.name, len(text))

            elif ext == ".pdf":
                # PDFs retornam uma entrada por página para rastrear metadados
                pages = _load_pdf_pages(entry)
                if not pages:
                    logger.warning("PDF sem texto extraível: %s", entry)
                    continue
                results.extend(pages)
                total_chars = sum(len(p["text"]) for p in pages)
                logger.info(
                    "Carregado PDF: %s — %d páginas, %d chars totais",
                    entry.name,
                    len(pages),
                    total_chars,
                )

            elif ext == ".json":
                text = _load_json(entry)
                if not text.strip():
                    logger.warning("JSON sem strings extraíveis: %s", entry)
                    continue
                results.append(
                    {
                        "text": text,
                        "source": str(entry),
                        "page": None,
                        "section": None,
                    }
                )
                logger.info("Carregado JSON: %s (%d chars)", entry.name, len(text))

        except Exception as exc:
            logger.error("Falha ao ler '%s': %s", entry, exc)

    return results

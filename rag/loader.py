"""
Document loader com suporte a .txt, .md, .pdf e .json.

Melhorias para PDFs em PT-BR:
  - Usa pypdf com extração de layout para melhor ordem de leitura.
  - Aplica limpeza de artefatos (hifenação, cabeçalhos, números de página)
    via rag.chunking.clean_pdf_text.
  - Detecta automaticamente encoding em arquivos .txt.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json"}


# ---------------------------------------------------------------------------
# Loaders individuais
# ---------------------------------------------------------------------------

def _load_txt(path: Path) -> str:
    """Carrega arquivo de texto com detecção de encoding."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    """Extrai texto de PDF preservando ordem de leitura natural."""
    from pypdf import PdfReader
    from rag.chunking import clean_pdf_text

    reader = PdfReader(str(path))
    pages: list[str] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            # extract_text com layout_mode_space_vertically=False
            # melhora extração em colunas e tabelas
            text = page.extract_text(
                extraction_mode="layout",
                layout_mode_space_vertically=False,
            ) or ""
        except Exception:
            # Fallback para modo simples se o modo layout falhar
            text = page.extract_text() or ""

        if text.strip():
            pages.append(text)

    full_text = "\n\n".join(pages)
    return clean_pdf_text(full_text)


def _load_json(path: Path) -> str:
    """Extrai todas as strings de um JSON aninhado."""
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

    Returns:
        Lista de dicts com chaves 'text' (str) e 'source' (str caminho).
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
            elif ext == ".pdf":
                text = _load_pdf(entry)
            elif ext == ".json":
                text = _load_json(entry)
            else:
                continue

            if not text.strip():
                logger.warning("Documento vazio após extração: %s", entry)
                continue

            results.append({"text": text, "source": str(entry)})
            logger.info("Carregado: %s (%d chars)", entry.name, len(text))

        except Exception as exc:
            logger.error("Falha ao ler '%s': %s", entry, exc)

    return results

"""
Document loader supporting .txt, .md, .pdf, and .json formats.
Returns a list of {text: str, source: str} dicts.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json"}


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))

    def _extract_strings(obj) -> list[str]:
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, dict):
            return [s for v in obj.values() for s in _extract_strings(v)]
        if isinstance(obj, list):
            return [s for item in obj for s in _extract_strings(item)]
        return []

    return " ".join(_extract_strings(data))


def load_documents(docs_dir: str) -> list[dict]:
    """Load all supported documents from docs_dir.

    Returns:
        List of dicts with keys 'text' (str) and 'source' (str file path).
    """
    docs_path = Path(docs_dir)
    results: list[dict] = []

    if not docs_path.exists():
        logger.warning("Documents directory does not exist: %s", docs_dir)
        return results

    for entry in sorted(docs_path.iterdir()):
        if not entry.is_file():
            continue

        ext = entry.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file type '%s': %s", ext, entry)
            continue

        try:
            if ext in {".txt", ".md"}:
                text = _load_txt(entry)
            elif ext == ".pdf":
                text = _load_pdf(entry)
            elif ext == ".json":
                text = _load_json(entry)
            else:
                continue  # unreachable, but keeps linters happy

            results.append({"text": text, "source": str(entry)})
        except Exception as exc:
            logger.error("Failed to read file %s: %s", entry, exc)

    return results

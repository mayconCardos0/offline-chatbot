"""
Sentence-aware text chunking with configurable size and overlap.
"""
import logging

logger = logging.getLogger(__name__)

SENTENCE_ENDINGS = {".", "!", "?"}


def _find_sentence_boundary(text: str, end: int) -> int:
    """Search backwards from `end` for a sentence boundary within the last 20% of the window.

    Returns the index just after the boundary character, or `end` if none found.
    """
    window = int(end * 0.20)
    search_start = end - window

    for i in range(end - 1, search_start - 1, -1):
        if text[i] in SENTENCE_ENDINGS:
            return i + 1  # cut after the punctuation

    return end  # fall back to hard cut


def chunk_document(doc: dict, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Split a document into sentence-aware overlapping chunks.

    Args:
        doc: Dict with 'text' and 'source' keys.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of dicts with 'text' and 'source' keys.
    """
    text: str = doc["text"]
    source: str = doc["source"]

    if not text:
        return []

    # Short-text edge case: return as single chunk
    if len(text) <= chunk_size:
        return [{"text": text, "source": source}]

    chunks: list[dict] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take whatever remains
            chunks.append({"text": text[start:], "source": source})
            break

        # Try to break at a sentence boundary
        cut = _find_sentence_boundary(text, end)
        chunks.append({"text": text[start:cut], "source": source})

        # Advance start, stepping back by overlap
        start = cut - overlap
        if start <= 0:
            start = cut  # safety: avoid infinite loop on very small texts

    return chunks

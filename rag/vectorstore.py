"""
FAISS vector store otimizado para Raspberry Pi 5.

Melhorias em relação à versão original:
  - Índice HNSW (IndexHNSWFlat) em vez de IndexFlatL2:
      * Busca aproximada O(log n) — latência ~5-10× menor em índices grandes.
      * Adequado para um PDF de 500 páginas (~2 000–5 000 chunks).
  - Inner-Product (IP) em vez de L2: funciona como cosseno quando os vetores
    estão L2-normalizados (o EmbeddingModel já faz isso).
  - Deduplicação por hash de texto: evita chunks duplicados ao re-indexar.
  - API pública limpa (sem acesso a _index.ntotal externo).
"""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.faiss"
_META_FILE  = "metadata.json"
_HASH_FILE  = "hashes.json"

# Parâmetros HNSW
_HNSW_M           = 32   # Conexões por nó — mais = melhor recall, mais RAM
_HNSW_EF_CONSTRUCTION = 200  # Qualidade da construção
_HNSW_EF_SEARCH   = 64   # Qualidade da busca (pode ajustar em runtime)


class VectorStore:
    """FAISS HNSW vector store com deduplicação e persistência em disco."""

    def __init__(self, index_dir: str, embedding_dim: int) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu não está instalado. Execute: pip install faiss-cpu"
            ) from exc

        self._faiss        = faiss
        self._index_dir    = Path(index_dir)
        self._embedding_dim = embedding_dim
        self._metadata: list[dict] = []
        self._hashes: set[str]     = set()
        self._index = self._build_empty_index()

        self.load()

    # ------------------------------------------------------------------
    # Tamanho (substitui acesso externo a _index.ntotal)
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Número de vetores no índice."""
        return self._index.ntotal

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Adiciona chunks e embeddings ao índice, ignorando duplicatas.

        Args:
            chunks:     Lista de {text, source}.
            embeddings: Vetores paralelos a chunks (L2-normalizados).
        """
        if not chunks:
            return

        new_chunks: list[dict]       = []
        new_vecs:   list[list[float]] = []

        for chunk, vec in zip(chunks, embeddings):
            h = _text_hash(chunk["text"])
            if h in self._hashes:
                continue
            self._hashes.add(h)
            new_chunks.append(chunk)
            new_vecs.append(vec)

        if not new_vecs:
            logger.debug("Todos os chunks já estavam indexados (deduplicação).")
            return

        vectors = np.array(new_vecs, dtype=np.float32)
        self._index.add(vectors)
        self._metadata.extend(new_chunks)

        logger.debug(
            "Adicionados %d chunks (total: %d, ignorados: %d).",
            len(new_chunks), len(self._metadata), len(chunks) - len(new_chunks)
        )

    def save(self) -> None:
        """Persiste índice FAISS, metadados e hashes em disco."""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self._index_dir / _INDEX_FILE))
        (self._index_dir / _META_FILE).write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self._index_dir / _HASH_FILE).write_text(
            json.dumps(list(self._hashes)), encoding="utf-8"
        )
        logger.info(
            "VectorStore salvo em '%s' (%d chunks).", self._index_dir, len(self._metadata)
        )

    def load(self) -> bool:
        """Carrega índice existente do disco, se disponível."""
        index_path = self._index_dir / _INDEX_FILE
        meta_path  = self._index_dir / _META_FILE
        hash_path  = self._index_dir / _HASH_FILE

        if not (index_path.exists() and meta_path.exists()):
            return False

        try:
            self._index    = self._faiss.read_index(str(index_path))
            self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if hash_path.exists():
                self._hashes = set(json.loads(hash_path.read_text(encoding="utf-8")))
            else:
                # Reconstrói hashes a partir dos metadados (compatibilidade)
                self._hashes = {_text_hash(c["text"]) for c in self._metadata}

            # Garante ef_search no índice carregado
            _set_ef_search(self._index, _HNSW_EF_SEARCH)

            logger.info(
                "VectorStore carregado de '%s' (%d chunks).",
                self._index_dir, len(self._metadata)
            )
            return True
        except Exception as exc:
            logger.error("Falha ao carregar VectorStore de '%s': %s", self._index_dir, exc)
            self._index    = self._build_empty_index()
            self._metadata = []
            self._hashes   = set()
            return False

    def search(self, query_vec: list[float], k: int) -> list[dict]:
        """Retorna os top-k chunks mais similares.

        Args:
            query_vec: Vetor de consulta (deve estar L2-normalizado).
            k:         Número de resultados.

        Returns:
            Lista de {text, source, score} ordenada por similaridade decrescente.
        """
        total = len(self._metadata)
        if total == 0:
            return []

        k = min(k, total)
        query = np.array([query_vec], dtype=np.float32)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < total:
                chunk = dict(self._metadata[idx])
                chunk["score"] = float(score)
                results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_empty_index(self):
        """Cria um índice HNSW-IP vazio."""
        index = self._faiss.IndexHNSWFlat(self._embedding_dim, _HNSW_M, self._faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = _HNSW_EF_CONSTRUCTION
        _set_ef_search(index, _HNSW_EF_SEARCH)
        return index


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _set_ef_search(index, ef: int) -> None:
    """Define ef_search se o índice for HNSW (silencioso caso contrário)."""
    try:
        index.hnsw.efSearch = ef
    except AttributeError:
        pass

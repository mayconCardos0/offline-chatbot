"""
Embedding model wrapper com:
  - Cache de embeddings em disco (evita recomputação entre restarts).
  - Suporte a modelos multilíngues (padrão: paraphrase-multilingual-MiniLM-L12-v2).
  - Batch encoding eficiente para uso controlado de memória na Raspberry Pi.
  - Normalização L2 para melhorar qualidade do produto interno (cosine via FAISS IP).
"""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Modelo padrão: multilíngue, leve (~120 MB), bom para PT-BR
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Tamanho do lote para encoding — limita pico de RAM na RPi
DEFAULT_BATCH_SIZE = 32


class EmbeddingModel:
    """Wraps SentenceTransformer com cache em disco e batch encoding."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_disk_cache: bool = True,
    ) -> None:
        """
        Args:
            model_name:     Nome do modelo HuggingFace.
            cache_dir:      Diretório local onde o modelo está armazenado.
            batch_size:     Tamanho do lote para encoding (controla uso de RAM).
            use_disk_cache: Se True, cacheia embeddings já computados em disco.

        Raises:
            FileNotFoundError: Se cache_dir não existir.
            ImportError:       Se sentence-transformers não estiver instalado.
        """
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Diretório de cache do modelo não encontrado: '{cache_dir}'. "
                f"Esperado o modelo '{model_name}' nesse diretório."
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers não está instalado. Execute: pip install sentence-transformers"
            ) from exc

        # Resolve caminho local do modelo (offline-first)
        local_model_dir = cache_path / f"models--sentence-transformers--{model_name}"
        if local_model_dir.exists():
            snapshots_dir = local_model_dir / "snapshots"
            snapshots = list(snapshots_dir.glob("*")) if snapshots_dir.exists() else []
            load_path = str(snapshots[0]) if snapshots else str(local_model_dir)
        else:
            logger.warning(
                "Modelo '%s' não encontrado localmente em '%s'. Tentando download…",
                model_name,
                cache_dir,
            )
            load_path = model_name

        logger.info(
            "Carregando modelo de embedding '%s' de '%s'", model_name, load_path
        )
        self._model = SentenceTransformer(load_path, cache_folder=str(cache_dir))
        self._model_name = model_name
        self._batch_size = batch_size

        # Cache em disco
        self._use_disk_cache = use_disk_cache
        self._disk_cache_dir = cache_path / "embedding_cache"
        if use_disk_cache:
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Modelo de embedding '%s' carregado.", model_name)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para uma lista de textos.

        Usa cache em disco quando disponível. Processa em lotes para
        controlar uso de memória.

        Args:
            texts: Lista de strings para embedar.

        Returns:
            Lista de vetores float, um por texto de entrada.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # 1. Verifica cache em disco
        if self._use_disk_cache:
            for i, text in enumerate(texts):
                cached = self._load_from_cache(text)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = list(texts)

        # 2. Computa embeddings faltantes em lotes
        if uncached_texts:
            computed = self._encode_batched(uncached_texts)
            for idx, vec, text in zip(uncached_indices, computed, uncached_texts):
                results[idx] = vec
                if self._use_disk_cache:
                    self._save_to_cache(text, vec)

        return results  # type: ignore[return-value]

    @property
    def dimension(self) -> int:
        """Dimensão dos vetores produzidos pelo modelo."""
        return int(self._model.get_sentence_embedding_dimension())

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _encode_batched(self, texts: list[str]) -> list[list[float]]:
        """Encoding em lotes com normalização L2."""
        all_vecs: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vecs = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2-norm → cosine via IP index
                show_progress_bar=False,
            )
            all_vecs.extend(v.tolist() for v in vecs)
        return all_vecs

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha256(f"{self._model_name}::{text}".encode("utf-8")).hexdigest()
        return h

    def _cache_path(self, text: str) -> Path:
        key = self._cache_key(text)
        # Subpastas por prefixo (evita diretório enorme)
        return self._disk_cache_dir / key[:2] / f"{key}.json"

    def _load_from_cache(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return None

    def _save_to_cache(self, text: str, vec: list[float]) -> None:
        path = self._cache_path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(vec), encoding="utf-8")
        except Exception as exc:
            logger.debug("Falha ao salvar cache de embedding: %s", exc)

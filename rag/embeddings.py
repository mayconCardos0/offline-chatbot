"""
Embedding model wrapper using sentence-transformers.
Model is loaded once and cached in memory for subsequent calls.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps a sentence-transformers SentenceTransformer for local embedding generation."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        """Load and cache the embedding model.

        Args:
            model_name: HuggingFace model name (e.g. 'all-MiniLM-L6-v2').
            cache_dir: Local directory where model files are stored / downloaded.

        Raises:
            FileNotFoundError: If the model directory does not exist at cache_dir.
            ImportError: If sentence-transformers is not installed.
        """
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Embedding model cache directory not found: '{cache_dir}'. "
                f"Expected model '{model_name}' to be present there."
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model '%s' from '%s'", model_name, cache_dir)
        # Prefer loading directly from the local cached model directory to stay fully offline.
        # sentence-transformers stores models as: <cache_dir>/models--<org>--<name>/snapshots/<hash>/
        local_model_dir = cache_path / f"models--sentence-transformers--{model_name}"
        if local_model_dir.exists():
            # Find the snapshot directory (there's typically one)
            snapshots = list((local_model_dir / "snapshots").glob("*")) if (local_model_dir / "snapshots").exists() else []
            load_path = str(snapshots[0]) if snapshots else str(local_model_dir)
        else:
            load_path = model_name  # fall back to HuggingFace download

        self._model = SentenceTransformer(load_path, cache_folder=str(cache_dir))
        self._model_name = model_name
        logger.info("Embedding model '%s' loaded and cached.", model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.
        """
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

"""
Baixa os modelos de busca para o cache local — precisa de internet, UMA vez.

Depois disso a API e os scripts rodam 100% offline (OFFLINE_MODE=true, o padrão):
os modelos são carregados direto da pasta local, sem consultar o Hugging Face.

Uso (com internet, antes de levar o dispositivo para um lugar sem rede):
    python scripts/download_models.py

Baixa o modelo de embeddings (EMBED_MODEL_NAME) e, se CROSS_ENCODER_ENABLED=true,
o cross-encoder (CROSS_ENCODER_MODEL), em EMBED_CACHE_DIR. É seguro repetir:
o que já está completo no cache não é baixado de novo.
"""

import os
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Este é o único script que precisa de rede: libera o Hugging Face antes de importá-lo.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from core.config import get_settings  # noqa: E402
from core.offline import find_local_snapshot  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Formatos alternativos pesados que o carregamento (PyTorch) não usa.
_IGNORE_PATTERNS = [
    "onnx/*",
    "openvino/*",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.tflite",
]


def _repo_id(model_name: str) -> str:
    return model_name if "/" in model_name else f"sentence-transformers/{model_name}"


def ensure_model(
    model_name: str,
    cache_dir: Path,
    downloader: Callable[..., str] | None = None,
) -> bool:
    """Garante um snapshot completo de `model_name` em `cache_dir`.

    Retorna True se o modelo está pronto (já estava, ou foi baixado agora).
    """
    if find_local_snapshot(model_name, [cache_dir]) is not None:
        print(f"  ✓ {model_name}: já está no cache local")
        return True

    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download

    print(f"  ↓ {model_name}: baixando para {cache_dir} …")
    try:
        downloader(
            repo_id=_repo_id(model_name),
            cache_dir=str(cache_dir),
            ignore_patterns=_IGNORE_PATTERNS,
        )
    except Exception as exc:
        print(f"  ✗ {model_name}: falha no download ({type(exc).__name__}: {exc})")
        return False

    if find_local_snapshot(model_name, [cache_dir]) is None:
        print(f"  ✗ {model_name}: o download terminou, mas o modelo ficou incompleto")
        return False
    print(f"  ✓ {model_name}: pronto")
    return True


def main() -> int:
    settings = get_settings()
    cache_dir = Path(settings.embed_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    models = [settings.embed_model_name]
    if settings.cross_encoder_enabled:
        models.append(settings.cross_encoder_model)

    print(f"Modelos para o cache local ({cache_dir}):")
    results = [ensure_model(name, cache_dir) for name in models]

    if all(results):
        print("\nTudo pronto: a API roda sem internet (OFFLINE_MODE=true).")
        return 0
    print(
        "\nAlgum modelo não foi baixado. Confira a conexão e rode de novo — "
        "o que já está completo não é baixado outra vez."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

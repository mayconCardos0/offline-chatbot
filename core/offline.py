"""
Carregamento de modelos sem internet.

O projeto roda offline. Por padrão, porém, o Hugging Face é consultado a cada
carga de modelo ("o arquivo mudou?"), e sem rede isso trava por minutos ou
falha, mesmo com o modelo já salvo em disco — a API nunca chega a subir.

Aqui o modelo é resolvido direto para a pasta do cache local, e carregar de um
caminho de pasta não faz nenhuma requisição de rede. Só se o modelo não estiver
no cache e OFFLINE_MODE=false o nome é repassado para o download.
"""

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


class ModelNotAvailableOffline(RuntimeError):
    """O modelo não está completo no cache local e o modo offline está ativo."""


def apply_offline_env(offline: bool) -> None:
    """Impede bibliotecas do Hugging Face de abrir conexões (e telemetria).

    Usa setdefault: uma variável já definida no ambiente (ex: HF_HUB_OFFLINE=0 no
    script de download) tem prioridade.
    """
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def default_hub_cache() -> Path:
    """Cache padrão do Hugging Face (onde modelos baixados sem cache_dir ficam)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:  # inexistente ou symlink quebrado
        return 0


def _is_complete(snapshot: Path, required_files: Sequence[str]) -> bool:
    """Um download interrompido deixa arquivos vazios; esses não servem."""
    if any(_file_size(snapshot / name) == 0 for name in required_files):
        return False
    if not any(_file_size(snapshot / name) > 0 for name in _WEIGHT_FILES):
        return False
    return all(_file_size(p) > 0 for p in snapshot.glob("*.json"))


def _repo_folders(model_name: str) -> list[str]:
    """Nomes de pasta do cache do HF para o modelo (`org/nome` → `models--org--nome`).

    Nomes sem organização são resolvidos pelo sentence-transformers como
    `sentence-transformers/<nome>`.
    """
    if "/" in model_name:
        return ["models--" + model_name.replace("/", "--")]
    return [f"models--sentence-transformers--{model_name}", f"models--{model_name}"]


def find_local_snapshot(
    model_name: str,
    cache_dirs: Iterable[Optional[object]],
    required_files: Sequence[str] = ("config.json",),
) -> Optional[Path]:
    """Procura um snapshot completo do modelo nas pastas de cache dadas."""
    for cache_dir in cache_dirs:
        if cache_dir is None:
            continue
        for folder in _repo_folders(model_name):
            repo_dir = Path(str(cache_dir)) / folder
            snapshots_dir = repo_dir / "snapshots"
            if not snapshots_dir.is_dir():
                continue

            candidates: list[Path] = []
            ref = repo_dir / "refs" / "main"
            if ref.is_file():
                pinned = snapshots_dir / ref.read_text(encoding="utf-8").strip()
                if pinned.is_dir():
                    candidates.append(pinned)
            others = sorted(
                (d for d in snapshots_dir.iterdir() if d.is_dir()),
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            candidates.extend(d for d in others if d not in candidates)

            for snapshot in candidates:
                if _is_complete(snapshot, required_files):
                    return snapshot
                logger.warning("Snapshot incompleto ignorado: %s", snapshot)
    return None


def resolve_model_path(
    model_name: str,
    cache_dirs: Iterable[Optional[object]],
    required_files: Sequence[str] = ("config.json",),
    offline: bool = True,
) -> str:
    """Devolve o que passar para o carregador do modelo.

    - Um caminho de pasta existente é usado como está.
    - Um snapshot completo no cache local vira o caminho dessa pasta (sem rede).
    - Sem cópia local: no modo offline falha com instruções; no modo online devolve
      o próprio nome, e o download acontece no carregamento.
    """
    if Path(model_name).is_dir():
        return model_name

    cache_dirs = list(cache_dirs)
    snapshot = find_local_snapshot(model_name, cache_dirs, required_files)
    if snapshot is not None:
        return str(snapshot)

    searched = ", ".join(str(d) for d in cache_dirs if d is not None) or "(nenhuma)"
    if offline:
        raise ModelNotAvailableOffline(
            f"Modelo '{model_name}' não encontrado (ou incompleto) no cache local. "
            f"Pastas verificadas: {searched}. O modo offline (OFFLINE_MODE=true) não "
            "baixa nada. Com internet, baixe uma vez com: python scripts/download_models.py"
        )
    logger.warning(
        "Modelo '%s' não encontrado localmente (%s); será baixado.",
        model_name,
        searched,
    )
    return model_name

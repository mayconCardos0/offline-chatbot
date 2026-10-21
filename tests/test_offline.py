"""
Tests for core/offline.py, rag/embeddings.py, rag/cross_encoder.py e
scripts/download_models.py — carregamento de modelos sem internet.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import Settings  # noqa: E402
from core.offline import (  # noqa: E402
    ModelNotAvailableOffline,
    apply_offline_env,
    find_local_snapshot,
    resolve_model_path,
)

_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
_saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
from scripts.download_models import ensure_model  # noqa: E402

for _k, _v in _saved_env.items():  # o script libera a rede ao ser importado
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v

E5 = "intfloat/multilingual-e5-small"
MINILM = "paraphrase-multilingual-MiniLM-L12-v2"


def make_snapshot(cache_dir, folder, rev="rev1", weights=True, files=None, ref=True):
    """Cria um snapshot no formato do cache do Hugging Face."""
    repo = cache_dir / folder
    snap = repo / "snapshots" / rev
    snap.mkdir(parents=True)
    content = {"config.json": "{}", "modules.json": "[]"}
    content.update(files or {})
    for name, text in content.items():
        (snap / name).write_text(text, encoding="utf-8")
    if weights:
        (snap / "model.safetensors").write_bytes(b"pesos")
    if ref:
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(rev, encoding="utf-8")
    return snap


class TestFindLocalSnapshot:
    def test_finds_model_from_another_organization(self, tmp_path):
        # Regressão: só `models--sentence-transformers--*` era reconhecido, então o
        # e5 (org `intfloat`) ia para a internet a cada boot.
        snap = make_snapshot(tmp_path, "models--intfloat--multilingual-e5-small")
        assert find_local_snapshot(E5, [tmp_path]) == snap

    def test_name_without_org_maps_to_sentence_transformers(self, tmp_path):
        snap = make_snapshot(tmp_path, f"models--sentence-transformers--{MINILM}")
        assert find_local_snapshot(MINILM, [tmp_path]) == snap

    def test_returns_none_when_not_cached(self, tmp_path):
        assert find_local_snapshot(E5, [tmp_path]) is None

    def test_interrupted_download_with_empty_file_is_ignored(self, tmp_path):
        # Foi o que derrubou a API na Pi: modules.json vazio → JSONDecodeError.
        make_snapshot(
            tmp_path,
            "models--intfloat--multilingual-e5-small",
            files={"modules.json": ""},
        )
        assert find_local_snapshot(E5, [tmp_path]) is None

    def test_snapshot_without_weights_is_ignored(self, tmp_path):
        make_snapshot(
            tmp_path, "models--intfloat--multilingual-e5-small", weights=False
        )
        assert find_local_snapshot(E5, [tmp_path]) is None

    def test_empty_weights_file_is_ignored(self, tmp_path):
        snap = make_snapshot(tmp_path, "models--intfloat--multilingual-e5-small")
        (snap / "model.safetensors").write_bytes(b"")
        assert find_local_snapshot(E5, [tmp_path]) is None

    def test_pytorch_bin_weights_are_accepted(self, tmp_path):
        snap = make_snapshot(
            tmp_path, "models--intfloat--multilingual-e5-small", weights=False
        )
        (snap / "pytorch_model.bin").write_bytes(b"pesos")
        assert find_local_snapshot(E5, [tmp_path]) == snap

    def test_prefers_revision_pinned_by_refs_main(self, tmp_path):
        folder = "models--intfloat--multilingual-e5-small"
        make_snapshot(tmp_path, folder, rev="velha", ref=False)
        pinned = make_snapshot(tmp_path, folder, rev="nova", ref=False)
        (tmp_path / folder / "refs").mkdir()
        (tmp_path / folder / "refs" / "main").write_text("nova", encoding="utf-8")
        assert find_local_snapshot(E5, [tmp_path]) == pinned

    def test_falls_back_to_next_cache_dir(self, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        first.mkdir()
        second.mkdir()
        snap = make_snapshot(second, "models--intfloat--multilingual-e5-small")
        assert find_local_snapshot(E5, [first, None, second]) == snap

    def test_missing_required_file_invalidates_the_snapshot(self, tmp_path):
        snap = make_snapshot(tmp_path, "models--intfloat--multilingual-e5-small")
        assert find_local_snapshot(E5, [tmp_path]) == snap
        assert find_local_snapshot(E5, [tmp_path], required_files=("x.json",)) is None


class TestResolveModelPath:
    def test_cached_model_resolves_to_local_folder(self, tmp_path):
        snap = make_snapshot(tmp_path, "models--intfloat--multilingual-e5-small")
        assert resolve_model_path(E5, [tmp_path], offline=True) == str(snap)

    def test_existing_directory_is_used_as_is(self, tmp_path):
        assert resolve_model_path(str(tmp_path), [], offline=True) == str(tmp_path)

    def test_offline_without_local_copy_fails_with_instructions(self, tmp_path):
        with pytest.raises(ModelNotAvailableOffline) as exc:
            resolve_model_path(E5, [tmp_path], offline=True)
        message = str(exc.value)
        assert E5 in message
        assert "download_models.py" in message
        assert str(tmp_path) in message

    def test_online_without_local_copy_returns_the_name_for_download(self, tmp_path):
        assert resolve_model_path(E5, [tmp_path], offline=False) == E5

    def test_local_copy_wins_even_when_online(self, tmp_path):
        snap = make_snapshot(tmp_path, "models--intfloat--multilingual-e5-small")
        assert resolve_model_path(E5, [tmp_path], offline=False) == str(snap)


class TestApplyOfflineEnv:
    def test_offline_sets_hub_variables(self, monkeypatch):
        for key in (*_ENV_KEYS, "HF_HUB_DISABLE_TELEMETRY"):
            monkeypatch.delenv(key, raising=False)
        apply_offline_env(True)
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"

    def test_online_leaves_network_enabled_but_disables_telemetry(self, monkeypatch):
        for key in (*_ENV_KEYS, "HF_HUB_DISABLE_TELEMETRY"):
            monkeypatch.delenv(key, raising=False)
        apply_offline_env(False)
        assert "HF_HUB_OFFLINE" not in os.environ
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"

    def test_explicit_environment_wins(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        apply_offline_env(True)
        assert os.environ["HF_HUB_OFFLINE"] == "0"


class TestOfflineSetting:
    def test_offline_by_default(self, monkeypatch):
        monkeypatch.delenv("OFFLINE_MODE", raising=False)
        assert Settings().offline_mode is True

    @pytest.mark.parametrize("value", ["false", "FALSE", "0", "no"])
    def test_can_be_turned_off(self, monkeypatch, value):
        monkeypatch.setenv("OFFLINE_MODE", value)
        assert Settings().offline_mode is False


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Substitui sentence_transformers por classes que só registram como foram chamadas."""
    calls = {"embedding": [], "cross": []}

    class FakeSentenceTransformer:
        def __init__(self, path, **kwargs):
            calls["embedding"].append((path, kwargs))

        def get_sentence_embedding_dimension(self):
            return 4

    class FakeCrossEncoder:
        def __init__(self, path, **kwargs):
            calls["cross"].append((path, kwargs))

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return calls


class TestEmbeddingModelOffline:
    def _model(self, cache_dir, offline=True):
        from rag.embeddings import EmbeddingModel

        return EmbeddingModel(
            model_name=E5,
            cache_dir=str(cache_dir),
            use_disk_cache=False,
            offline=offline,
        )

    def test_loads_from_local_snapshot_path_never_from_the_hub_name(
        self, tmp_path, fake_sentence_transformers, monkeypatch
    ):
        monkeypatch.setattr("rag.embeddings.default_hub_cache", lambda: tmp_path / "x")
        snap = make_snapshot(tmp_path, "models--intfloat--multilingual-e5-small")
        self._model(tmp_path)
        path, _ = fake_sentence_transformers["embedding"][0]
        assert path == str(snap)

    def test_offline_without_model_raises_instead_of_hitting_the_network(
        self, tmp_path, fake_sentence_transformers, monkeypatch
    ):
        monkeypatch.setattr("rag.embeddings.default_hub_cache", lambda: tmp_path / "x")
        with pytest.raises(ModelNotAvailableOffline):
            self._model(tmp_path)
        assert fake_sentence_transformers["embedding"] == []

    def test_online_without_model_falls_back_to_download_by_name(
        self, tmp_path, fake_sentence_transformers, monkeypatch
    ):
        monkeypatch.setattr("rag.embeddings.default_hub_cache", lambda: tmp_path / "x")
        self._model(tmp_path, offline=False)
        path, _ = fake_sentence_transformers["embedding"][0]
        assert path == E5

    def test_corrupt_partial_download_is_not_loaded(
        self, tmp_path, fake_sentence_transformers, monkeypatch
    ):
        monkeypatch.setattr("rag.embeddings.default_hub_cache", lambda: tmp_path / "x")
        make_snapshot(
            tmp_path,
            "models--intfloat--multilingual-e5-small",
            files={"modules.json": ""},
        )
        with pytest.raises(ModelNotAvailableOffline):
            self._model(tmp_path)


class TestCrossEncoderOffline:
    CE = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    def _reranker(self, cache_dir, offline=True):
        from rag.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(
            model_name=self.CE, cache_dir=str(cache_dir), offline=offline
        )

    def test_warm_up_loads_from_local_snapshot(
        self, tmp_path, fake_sentence_transformers, monkeypatch
    ):
        monkeypatch.setattr(
            "rag.cross_encoder.default_hub_cache", lambda: tmp_path / "x"
        )
        snap = make_snapshot(
            tmp_path, "models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1"
        )
        self._reranker(tmp_path).warm_up()
        path, _ = fake_sentence_transformers["cross"][0]
        assert path == str(snap)

    def test_model_is_not_loaded_until_needed(
        self, tmp_path, fake_sentence_transformers
    ):
        self._reranker(tmp_path)
        assert fake_sentence_transformers["cross"] == []

    def test_warm_up_offline_without_model_fails_at_startup(
        self, tmp_path, fake_sentence_transformers, monkeypatch
    ):
        monkeypatch.setattr(
            "rag.cross_encoder.default_hub_cache", lambda: tmp_path / "x"
        )
        with pytest.raises(ModelNotAvailableOffline):
            self._reranker(tmp_path).warm_up()
        assert fake_sentence_transformers["cross"] == []


class TestDownloadModelsScript:
    def _downloader(self, cache_dir, calls, complete=True):
        def download(repo_id, cache_dir, ignore_patterns):
            calls.append((repo_id, ignore_patterns))
            if complete:
                from pathlib import Path

                folder = "models--" + repo_id.replace("/", "--")
                make_snapshot(Path(cache_dir), folder)
            return cache_dir

        return download

    def test_skips_download_when_model_is_already_complete(self, tmp_path):
        make_snapshot(tmp_path, "models--intfloat--multilingual-e5-small")
        calls = []
        assert ensure_model(E5, tmp_path, self._downloader(tmp_path, calls)) is True
        assert calls == []

    def test_downloads_missing_model_and_verifies_it(self, tmp_path):
        calls = []
        assert ensure_model(E5, tmp_path, self._downloader(tmp_path, calls)) is True
        assert calls[0][0] == E5
        assert find_local_snapshot(E5, [tmp_path]) is not None

    def test_skips_heavy_alternative_formats(self, tmp_path):
        calls = []
        ensure_model(E5, tmp_path, self._downloader(tmp_path, calls))
        assert "onnx/*" in calls[0][1]

    def test_name_without_org_downloads_from_sentence_transformers(self, tmp_path):
        calls = []
        ensure_model(MINILM, tmp_path, self._downloader(tmp_path, calls))
        assert calls[0][0] == f"sentence-transformers/{MINILM}"

    def test_download_failure_returns_false(self, tmp_path):
        def failing(**kwargs):
            raise ConnectionError("sem rede")

        assert ensure_model(E5, tmp_path, failing) is False

    def test_incomplete_download_returns_false(self, tmp_path):
        calls = []
        downloader = self._downloader(tmp_path, calls, complete=False)
        assert ensure_model(E5, tmp_path, downloader) is False

    def test_corrupt_partial_download_is_downloaded_again(self, tmp_path):
        snap = make_snapshot(
            tmp_path,
            "models--intfloat--multilingual-e5-small",
            files={"modules.json": ""},
        )
        calls = []

        def repair(repo_id, cache_dir, ignore_patterns):
            calls.append(repo_id)
            (snap / "modules.json").write_text("[]", encoding="utf-8")

        # um snapshot corrompido não conta como "já está no cache"
        assert ensure_model(E5, tmp_path, repair) is True
        assert calls == [E5]

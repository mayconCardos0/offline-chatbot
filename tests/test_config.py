"""
Tests for core/config.py — Settings dataclass and setup_logging.
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import Settings, get_settings, setup_logging  # noqa: E402


class TestSettingsDefaults:
    def setup_method(self):
        self._original = {}
        for key in ["PORT", "MODEL_PATH", "N_CTX", "N_THREADS", "TOP_K"]:
            self._original[key] = os.environ.pop(key, None)

    def teardown_method(self):
        for key, val in self._original.items():
            if val is not None:
                os.environ[key] = val

    def test_default_port(self):
        assert Settings().port == 8000

    def test_default_n_ctx(self):
        assert Settings().n_ctx == 4096

    def test_default_n_threads(self):
        assert Settings().n_threads == 4

    def test_default_n_gpu_layers(self):
        assert Settings().n_gpu_layers == 0

    def test_default_top_k(self):
        assert Settings().top_k == 5

    def test_default_chunk_size(self):
        assert Settings().chunk_size == 512

    def test_default_log_level(self):
        assert Settings().log_level == "INFO"

    def test_default_embed_disk_cache_true(self):
        assert Settings().embed_disk_cache is True


class TestSettingsFromEnv:
    def test_port_from_env(self, monkeypatch):
        monkeypatch.setenv("PORT", "9000")
        assert Settings().port == 9000

    def test_n_ctx_from_env(self, monkeypatch):
        monkeypatch.setenv("N_CTX", "2048")
        assert Settings().n_ctx == 2048

    def test_n_threads_from_env(self, monkeypatch):
        monkeypatch.setenv("N_THREADS", "8")
        assert Settings().n_threads == 8

    def test_top_k_from_env(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "10")
        assert Settings().top_k == 10

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        assert Settings().log_level == "DEBUG"

    def test_embed_disk_cache_false_from_env(self, monkeypatch):
        monkeypatch.setenv("EMBED_DISK_CACHE", "false")
        assert Settings().embed_disk_cache is False

    def test_model_path_from_env(self, monkeypatch):
        monkeypatch.setenv("MODEL_PATH", "models/my_model.gguf")
        assert Settings().model_path == "models/my_model.gguf"

    def test_min_score_from_env(self, monkeypatch):
        monkeypatch.setenv("MIN_SCORE", "0.5")
        assert Settings().min_score == pytest.approx(0.5)

    def test_lexical_weight_from_env(self, monkeypatch):
        monkeypatch.setenv("LEXICAL_WEIGHT", "0.4")
        assert Settings().lexical_weight == pytest.approx(0.4)


class TestGetSettings:
    def test_returns_settings_instance(self):
        assert isinstance(get_settings(), Settings)

    def test_returns_new_instance_each_call(self):
        assert get_settings() is not get_settings()


class TestSetupLogging:
    def _reset(self):
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_info_level_set(self):
        self._reset()
        setup_logging("INFO")
        assert logging.getLogger().level == logging.INFO

    def test_debug_level_set(self):
        self._reset()
        setup_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_warning_level_set(self):
        self._reset()
        setup_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_invalid_level_defaults_to_info(self):
        self._reset()
        setup_logging("NOTAVALIDLEVEL")
        assert logging.getLogger().level == logging.INFO

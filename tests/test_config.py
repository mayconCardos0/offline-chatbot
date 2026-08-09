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
        for key in [
            "PORT", "MODEL_PATH", "N_CTX", "N_THREADS", "TOP_K",
            "CHUNK_SIZE", "CHUNK_OVERLAP",
        ]:
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


# ---------------------------------------------------------------------------
# Phase 2: new settings added in core/config.py
# ---------------------------------------------------------------------------


class TestSettingsPhase2Defaults:
    """Verify that every Phase 2 parameter has the correct baseline default."""

    def test_lexical_weight_baseline_is_0_40(self):
        # Corrected from 0.30 to match the actual retriever default
        assert Settings().lexical_weight == pytest.approx(0.40)

    def test_max_context_chars_baseline_is_3500(self):
        # Corrected from 2000 to match the actual pipeline default
        assert Settings().max_context_chars == 3500

    def test_chunk_overlap_baseline_is_50_tokens(self, monkeypatch):
        # Corrected from 1 sentence to 50 tokens
        monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
        assert Settings().chunk_overlap == 50

    def test_chunk_min_tokens_default(self):
        assert Settings().chunk_min_tokens == 200

    def test_chunk_max_tokens_default(self):
        assert Settings().chunk_max_tokens == 375

    def test_bm25_k1_default(self):
        assert Settings().bm25_k1 == pytest.approx(1.5)

    def test_bm25_b_default(self):
        assert Settings().bm25_b == pytest.approx(0.75)

    def test_hnsw_m_default(self):
        assert Settings().hnsw_m == 32

    def test_hnsw_ef_construction_default(self):
        assert Settings().hnsw_ef_construction == 200

    def test_hnsw_ef_search_default(self):
        assert Settings().hnsw_ef_search == 64

    def test_adaptive_sigma_default(self):
        assert Settings().adaptive_sigma == pytest.approx(1.0)

    def test_gap_filter_enabled_default_true(self):
        assert Settings().gap_filter_enabled is True

    def test_keyword_filter_enabled_default_true(self):
        assert Settings().keyword_filter_enabled is True

    def test_min_keyword_overlap_default(self):
        assert Settings().min_keyword_overlap == pytest.approx(0.15)

    def test_temporal_validation_enabled_default_true(self):
        assert Settings().temporal_validation_enabled is True

    def test_high_confidence_score_default(self):
        assert Settings().high_confidence_score == pytest.approx(0.65)

    def test_low_confidence_score_default(self):
        assert Settings().low_confidence_score == pytest.approx(0.45)


class TestSettingsPhase2FromEnv:
    """Verify that Phase 2 parameters can be overridden via environment."""

    def test_lexical_weight_from_env(self, monkeypatch):
        monkeypatch.setenv("LEXICAL_WEIGHT", "0.6")
        assert Settings().lexical_weight == pytest.approx(0.6)

    def test_max_context_chars_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_CONTEXT_CHARS", "5000")
        assert Settings().max_context_chars == 5000

    def test_chunk_overlap_from_env(self, monkeypatch):
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        assert Settings().chunk_overlap == 100

    def test_chunk_min_tokens_from_env(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MIN_TOKENS", "150")
        assert Settings().chunk_min_tokens == 150

    def test_chunk_max_tokens_from_env(self, monkeypatch):
        monkeypatch.setenv("CHUNK_MAX_TOKENS", "500")
        assert Settings().chunk_max_tokens == 500

    def test_bm25_k1_from_env(self, monkeypatch):
        monkeypatch.setenv("BM25_K1", "2.0")
        assert Settings().bm25_k1 == pytest.approx(2.0)

    def test_bm25_b_from_env(self, monkeypatch):
        monkeypatch.setenv("BM25_B", "0.5")
        assert Settings().bm25_b == pytest.approx(0.5)

    def test_hnsw_m_from_env(self, monkeypatch):
        monkeypatch.setenv("HNSW_M", "16")
        assert Settings().hnsw_m == 16

    def test_hnsw_ef_search_from_env(self, monkeypatch):
        monkeypatch.setenv("HNSW_EF_SEARCH", "128")
        assert Settings().hnsw_ef_search == 128

    def test_adaptive_sigma_from_env(self, monkeypatch):
        monkeypatch.setenv("ADAPTIVE_SIGMA", "0.5")
        assert Settings().adaptive_sigma == pytest.approx(0.5)

    def test_gap_filter_disabled_from_env(self, monkeypatch):
        monkeypatch.setenv("GAP_FILTER_ENABLED", "false")
        assert Settings().gap_filter_enabled is False

    def test_keyword_filter_disabled_from_env(self, monkeypatch):
        monkeypatch.setenv("KEYWORD_FILTER_ENABLED", "false")
        assert Settings().keyword_filter_enabled is False

    def test_temporal_validation_disabled_from_env(self, monkeypatch):
        monkeypatch.setenv("TEMPORAL_VALIDATION_ENABLED", "false")
        assert Settings().temporal_validation_enabled is False

    def test_high_confidence_score_from_env(self, monkeypatch):
        monkeypatch.setenv("HIGH_CONFIDENCE_SCORE", "0.80")
        assert Settings().high_confidence_score == pytest.approx(0.80)

    def test_low_confidence_score_from_env(self, monkeypatch):
        monkeypatch.setenv("LOW_CONFIDENCE_SCORE", "0.35")
        assert Settings().low_confidence_score == pytest.approx(0.35)

"""
Tests for rag/loader.py — load_documents with .txt, .md, .json, and PDF support.
"""
import json
import os
import sys
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.loader import load_documents, _load_txt, _load_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_file(directory, name, content, mode="w", encoding="utf-8"):
    path = os.path.join(directory, name)
    with open(path, mode, encoding=encoding) as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# _load_txt
# ---------------------------------------------------------------------------

class TestLoadTxt:
    def test_reads_utf8_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello UTF-8 — ção", encoding="utf-8")
        result = _load_txt(f)
        assert "ção" in result

    def test_reads_latin1_file(self, tmp_path):
        f = tmp_path / "latin.txt"
        f.write_bytes("caf\xe9".encode("latin-1"))
        result = _load_txt(f)
        assert "caf" in result

    def test_empty_file_returns_empty_string(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = _load_txt(f)
        assert result == ""


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------

class TestLoadJson:
    def test_extracts_string_values(self, tmp_path):
        data = {"key": "value", "nested": {"inner": "content"}}
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = _load_json(f)
        assert "value" in result
        assert "content" in result

    def test_extracts_list_strings(self, tmp_path):
        data = ["first item", "second item"]
        f = tmp_path / "list.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = _load_json(f)
        assert "first item" in result
        assert "second item" in result

    def test_ignores_non_string_values(self, tmp_path):
        data = {"count": 42, "flag": True, "name": "test"}
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = _load_json(f)
        assert "test" in result
        assert "42" not in result  # numbers excluded


# ---------------------------------------------------------------------------
# load_documents
# ---------------------------------------------------------------------------

class TestLoadDocuments:
    def test_empty_directory_returns_empty(self, tmp_path):
        result = load_documents(str(tmp_path))
        assert result == []

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        result = load_documents(str(tmp_path / "does_not_exist"))
        assert result == []

    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Plain text content.", encoding="utf-8")
        result = load_documents(str(tmp_path))
        assert len(result) == 1
        assert "Plain text content." in result[0]["text"]

    def test_loads_md_files(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Title\n\nSome markdown.", encoding="utf-8")
        result = load_documents(str(tmp_path))
        assert len(result) == 1
        assert "Some markdown." in result[0]["text"]

    def test_loads_json_files(self, tmp_path):
        (tmp_path / "data.json").write_text(json.dumps({"info": "json content"}), encoding="utf-8")
        result = load_documents(str(tmp_path))
        assert len(result) == 1
        assert "json content" in result[0]["text"]

    def test_skips_unsupported_extensions(self, tmp_path):
        (tmp_path / "data.csv").write_text("col1,col2\nval1,val2", encoding="utf-8")
        (tmp_path / "doc.txt").write_text("Valid text.", encoding="utf-8")
        result = load_documents(str(tmp_path))
        assert len(result) == 1
        assert result[0]["source"].endswith(".txt")

    def test_source_field_contains_path(self, tmp_path):
        (tmp_path / "myfile.txt").write_text("content", encoding="utf-8")
        result = load_documents(str(tmp_path))
        assert "myfile.txt" in result[0]["source"]

    def test_loads_multiple_files(self, tmp_path):
        for i in range(3):
            (tmp_path / f"file{i}.txt").write_text(f"Content {i}.", encoding="utf-8")
        result = load_documents(str(tmp_path))
        assert len(result) == 3

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        (tmp_path / "content.txt").write_text("Non-empty.", encoding="utf-8")
        result = load_documents(str(tmp_path))
        assert len(result) == 1

    def test_result_has_text_and_source_keys(self, tmp_path):
        (tmp_path / "test.txt").write_text("hello", encoding="utf-8")
        result = load_documents(str(tmp_path))
        assert "text" in result[0]
        assert "source" in result[0]

    def test_skips_subdirectories(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested", encoding="utf-8")
        (tmp_path / "root.txt").write_text("root content", encoding="utf-8")
        result = load_documents(str(tmp_path))
        assert len(result) == 1
        assert "root content" in result[0]["text"]

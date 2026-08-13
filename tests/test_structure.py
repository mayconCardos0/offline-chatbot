"""
Tests for rag/structure.py — hierarchical Unidade/Capítulo/Tópico detection.

The bold/font-size-aware branch of `_classify_line` is exercised end-to-end
using tiny, real PDFs built in-memory with PyMuPDF's own writer API
(`page.insert_text(..., fontname="hebo")` for bold, `"helv"` for regular
text) rather than mocks — `pymupdf` is a real (non-mocked) dependency of this
test suite already (see rag/loader.py), so this keeps the tests honest about
what PyMuPDF actually reports for bold spans.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.loader import _load_pdf_pages  # noqa: E402
from rag.structure import (  # noqa: E402
    DocumentSection,
    _classify_line,
    _looks_like_heading_text,
    detect_structure,
    merge_tiny_sections,
    sections_to_docs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pdf(tmp_path, lines):
    """Builds a minimal real PDF from (text, fontsize, bold) tuples.

    Each tuple becomes its own line, spaced 20pt apart, using PyMuPDF's
    built-in Helvetica ("helv") / Helvetica-Bold ("hebo") fonts.
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for text, fontsize, bold in lines:
        fontname = "hebo" if bold else "helv"
        page.insert_text((72, y), text, fontsize=fontsize, fontname=fontname)
        y += 20
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# _looks_like_heading_text
# ---------------------------------------------------------------------------


class TestLooksLikeHeadingText:
    def test_accepts_short_title_case_phrase(self):
        assert _looks_like_heading_text("O Paleolítico") is True

    def test_accepts_text_starting_with_digit(self):
        assert _looks_like_heading_text("1822 e a Independência") is True

    def test_rejects_too_short(self):
        assert _looks_like_heading_text("Ok") is False

    def test_rejects_too_long(self):
        assert _looks_like_heading_text("x" * 121) is False

    def test_rejects_lowercase_start(self):
        assert _looks_like_heading_text("naturais") is False

    def test_rejects_too_many_words(self):
        assert _looks_like_heading_text(" ".join(["Palavra"] * 21)) is False


# ---------------------------------------------------------------------------
# _classify_line
# ---------------------------------------------------------------------------


class TestClassifyLine:
    def test_unit_pattern_matches_regardless_of_bold(self):
        level, _ = _classify_line("Unidade 1: Natureza em transformação")
        assert level == 1

    def test_chapter_pattern_with_colon_separator(self):
        level, _ = _classify_line("Capitulo 1: A origem da humanidade")
        assert level == 2

    def test_numbered_topic_pattern(self):
        level, _ = _classify_line("1.1 Alcanos")
        assert level == 3

    def test_parte_prefix_does_not_false_match_ordinary_sentences(self):
        """Regression: [IVXLCDM]+ without a word boundary let a single roman
        letter (e.g. the 'd' in 'dos') satisfy the numeral group, so any
        sentence starting with a common PT-BR word like "Parte dos..." /
        "Parte da..." / "Parte do..." false-matched as a Unidade heading."""
        for text in [
            "Parte dos líderes das revoltas foram presos",
            "Parte da população migrou para as cidades",
            "Parte do território foi anexado",
        ]:
            level, _ = _classify_line(text)
            assert level == 0, text

    def test_parte_with_real_numeral_still_matches(self):
        level, _ = _classify_line("Parte 1 — Introdução")
        assert level == 1
        level, _ = _classify_line("Parte II: O mundo antigo")
        assert level == 1

    def test_bold_body_size_text_is_subtitle(self):
        level, text = _classify_line(
            "As fontes históricas e a análise histórica",
            is_bold=True,
            font_size=12.0,
            body_size=12.0,
        )
        assert level == 3
        assert text == "As fontes históricas e a análise histórica"

    def test_bold_and_much_larger_than_body_is_unit_level(self):
        level, _ = _classify_line(
            "Alguma Unidade Sem Padrão Textual",
            is_bold=True,
            font_size=16.0,
            body_size=12.0,
        )
        assert level == 1

    def test_bold_large_font_run_on_sentence_is_not_a_unit_heading(self):
        """Regression: a bold, larger-than-body pull-quote/highlight paragraph
        must NOT be classified as a Unidade just because of font size — that
        corrupts the breadcrumb for every chunk until the next real Unidade
        heading is found. Real unit titles are short; a multi-clause run-on
        sentence (>20 words) must fail the heading-shape sanity filter."""
        long_sentence = (
            "Parte dos líderes das revoltas era influenciada pelo pensamento "
            "de Wycliffe, e o rei Ricardo II culpou os lolardos pelo levante. "
            "Como consequência, expulsou"
        )
        level, _ = _classify_line(
            long_sentence, is_bold=True, font_size=14.0, body_size=12.0
        )
        assert level == 0

    def test_non_bold_body_text_is_level_zero(self):
        level, _ = _classify_line(
            "Este é um parágrafo comum de corpo de texto.",
            is_bold=False,
            font_size=0.0,
            body_size=12.0,
        )
        assert level == 0

    def test_bold_but_fails_heading_sanity_filter_is_level_zero(self):
        # Starts lowercase — not a plausible heading, even though bold.
        level, _ = _classify_line(
            "naturais", is_bold=True, font_size=12.0, body_size=12.0
        )
        assert level == 0

    def test_empty_line_is_level_zero(self):
        level, text = _classify_line("   ")
        assert level == 0
        assert text == ""

    def test_isolated_page_number_is_level_zero(self):
        level, _ = _classify_line("42")
        assert level == 0


# ---------------------------------------------------------------------------
# merge_tiny_sections
# ---------------------------------------------------------------------------


def _section(text, source="doc.pdf", topic=None):
    return DocumentSection(
        topic=topic, text=text, source=source, level=3 if topic else 0
    )


class TestMergeTinySections:
    def test_merges_consecutive_tiny_sections_same_source(self):
        sections = [
            _section("short", topic="A"),
            _section("also short", topic="B"),
        ]
        merged = merge_tiny_sections(sections, min_tokens=1000)
        assert len(merged) == 1
        assert "short" in merged[0].text
        assert "also short" in merged[0].text
        # Keeps the FIRST section's heading metadata.
        assert merged[0].topic == "A"

    def test_does_not_merge_across_different_sources(self):
        sections = [
            _section("short", source="a.pdf", topic="A"),
            _section("short", source="b.pdf", topic="B"),
        ]
        merged = merge_tiny_sections(sections, min_tokens=1000)
        assert len(merged) == 2

    def test_does_not_merge_sections_already_above_threshold(self):
        long_text = " ".join(["palavra"] * 500)
        sections = [_section(long_text, topic="A"), _section("short", topic="B")]
        merged = merge_tiny_sections(sections, min_tokens=10)
        assert len(merged) == 2

    def test_empty_input_returns_empty(self):
        assert merge_tiny_sections([], min_tokens=100) == []


# ---------------------------------------------------------------------------
# detect_structure — end-to-end against a real (in-memory) PDF
# ---------------------------------------------------------------------------


class TestDetectStructureWithRealPdf:
    def test_detects_unit_chapter_and_bold_only_subtitle(self, tmp_path):
        body_a = (
            "Este e um paragrafo de corpo de texto explicando o primeiro "
            "topico em bastante detalhe para simular um paragrafo real."
        )
        body_b = (
            "Este e outro paragrafo de corpo, agora sob o segundo subtitulo, "
            "tambem com texto suficiente para nao ser considerado minusculo."
        )
        pdf_path = _build_pdf(
            tmp_path,
            [
                ("Unidade 1: Teste", 16, True),
                ("Capitulo 1: Introducao", 12, True),
                ("Um Subtitulo Qualquer", 12, True),
                (body_a, 12, False),
                ("Outro Subtitulo Aqui", 12, True),
                (body_b, 12, False),
            ],
        )

        pages = _load_pdf_pages(pdf_path)
        sections = detect_structure(pages, str(pdf_path))

        levels = [s.level for s in sections]
        assert 1 in levels  # Unidade
        assert 2 in levels  # Capitulo
        assert levels.count(3) >= 2  # os dois subtitulos em negrito

        topics = {s.topic for s in sections if s.topic}
        assert "Um Subtitulo Qualquer" in topics
        assert "Outro Subtitulo Aqui" in topics

        # sections_to_docs continua produzindo o formato esperado pelo pipeline
        docs = sections_to_docs(sections)
        assert all(
            {"text", "source", "unit", "chapter", "topic"} <= d.keys() for d in docs
        )

    def test_non_bold_document_still_detects_textual_patterns_only(self, tmp_path):
        """Sem informação de negrito (ex: fonte .txt), apenas Unidade/Capítulo
        via padrão textual continuam funcionando — subtítulos sem numeração
        não são detectados, o que é o comportamento esperado sem font info."""
        pages = [
            {
                "text": "Unidade 1: Teste\nCapitulo 1: Introducao\nTexto qualquer aqui.",
                "source": "doc.txt",
                "page": None,
            }
        ]
        sections = detect_structure(pages, "doc.txt")
        levels = [s.level for s in sections]
        assert 1 in levels
        assert 2 in levels

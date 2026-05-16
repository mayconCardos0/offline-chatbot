"""
Detecção de estrutura hierárquica em documentos didáticos PT-BR.

Hierarquia detectada:
    Unidade  (nível 1) — ex: "UNIDADE 3", "Unidade III — Química Orgânica"
    Capítulo (nível 2) — ex: "Capítulo 5", "5. Hidrocarbonetos"
    Tópico   (nível 3) — ex: "5.1 Alcanos", "1.2.3 Propriedades Físicas"

Estratégia:
    1. Cada linha é classificada como heading (1/2/3) ou body via regex.
    2. Um DocumentOutline é construído linha a linha mantendo o nó corrente
       de cada nível (unidade, capítulo, tópico).
    3. O texto body é associado ao tópico ativo (ou capítulo/unidade se
       ainda não foi aberto um tópico).
    4. O resultado é uma lista de DocumentSection prontas para chunking,
       com os campos unit, chapter, topic, level e breadcrumb.

Por que isso melhora o RAG?
    - Chunks passam a ter contexto hierárquico como metadado (breadcrumb),
      que o pipeline pode usar para prefixar o contexto injetado no LLM:
      "Em [Unidade 2 > Cap. 3 > 3.1 Fotossíntese]: ..."
    - O retriever pode filtrar por unidade/capítulo quando a query contém
      esses marcadores (ex: "capítulo 2 questão 1").
    - O indexador pode criar um nó especial "título de seção" com alta
      prioridade para queries sobre estrutura do documento.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Padrões de heading
# ---------------------------------------------------------------------------

# Nível 1 — Unidade (mais flexível para capturar variações)
_UNIT_PATTERNS = [
    # "UNIDADE 1" ou "Unidade 1" (case insensitive)
    re.compile(
        r"^\s*UNIDADE\s+(?P<num>[IVXLCDM]+|\d+)",
        re.IGNORECASE,
    ),
    # "MÓDULO 1" ou "Módulo 1"
    re.compile(
        r"^\s*M[ÓO]DULO\s+(?P<num>[IVXLCDM]+|\d+)",
        re.IGNORECASE,
    ),
    # "PARTE 1" ou "Parte 1"
    re.compile(
        r"^\s*PARTE\s+(?P<num>[IVXLCDM]+|\d+)",
        re.IGNORECASE,
    ),
]

# Nível 2 — Capítulo (mais flexível)
_CHAPTER_PATTERNS = [
    # "CAPÍTULO 1" ou "Capítulo 1" (com ou sem acento) - DEVE estar isolado ou com título curto
    re.compile(
        r"^\s*CAP[ÍI]TULO\s+(?P<num>[IVXLCDM]+|\d+)(?:\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][^\n]{0,80})?$",
        re.IGNORECASE,
    ),
    # Padrão "CAPÍTULO X Nome do capítulo" em linha separada
    re.compile(
        r"^\s*CAP[ÍI]TULO\s+\d+\s*$",
        re.IGNORECASE,
    ),
]

# Nível 3 — Tópico (seções numeradas)
_TOPIC_PATTERNS = [
    # Seções numeradas: "1.1", "2.3.1", etc
    re.compile(
        r"^\s*(?P<num>\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ]",
    ),
    # Letras de seção: "a)", "b)"
    re.compile(
        r"^\s*[a-z]\)\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][^\n]{8,}$",
    ),
    # Seções especiais: "EM PAUTA", "TRABALHO COM FONTES", etc
    re.compile(
        r"^\s*(?:EM PAUTA|TRABALHO COM FONTES|TRABALHO E JUVENTUDES|ESTRATÉGIA DE ESTUDO)\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ]",
        re.IGNORECASE,
    ),
]

# Heading genérico (linha toda em maiúsculas, 4-100 chars, sem ponto final)
_ALLCAPS_HEADING = re.compile(r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ\s\-–—:,0-9]{4,100}$")

# Linha que claramente é corpo de texto (comprimento e pontuação interna)
_BODY_SIGNALS = re.compile(r"[,;:\(\)\[\]]|\.{2,}")


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------


@dataclass
class DocumentSection:
    """Um trecho de texto com hierarquia detectada.

    Attributes:
        unit:      Título da unidade (nível 1), ou None.
        chapter:   Título do capítulo (nível 2), ou None.
        topic:     Título do tópico (nível 3), ou None.
        level:     Nível mais específico detectado (1=unit, 2=chapter, 3=topic, 0=body).
        text:      Conteúdo textual da seção.
        source:    Caminho do arquivo original.
        page:      Página inicial da seção (pode ser None).
        breadcrumb: String legível: "Unidade X > Cap. Y > Tópico Z"
    """

    unit: Optional[str] = None
    chapter: Optional[str] = None
    topic: Optional[str] = None
    level: int = 0
    text: str = ""
    source: str = ""
    page: Optional[int] = None

    @property
    def breadcrumb(self) -> str:
        parts = []
        if self.unit:
            parts.append(self.unit)
        if self.chapter:
            parts.append(self.chapter)
        if self.topic:
            parts.append(self.topic)
        return " > ".join(parts) if parts else ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "unit": self.unit,
            "chapter": self.chapter,
            "topic": self.topic,
            "level": self.level,
            "breadcrumb": self.breadcrumb,
            # Keep backward-compat field used by old pipeline
            "section": self.topic or self.chapter or self.unit,
        }


# ---------------------------------------------------------------------------
# Classificador de linha
# ---------------------------------------------------------------------------


def _classify_line(line: str) -> tuple[int, str]:
    """Retorna (nível, texto_limpo). Nível 0 = corpo de texto.

    Returns:
        (level, clean_text) onde level é 0=body, 1=unit, 2=chapter, 3=topic.
    """
    stripped = line.strip()
    if not stripped:
        return 0, stripped

    # Ignora linhas muito curtas (< 3 chars) ou números de página isolados
    if len(stripped) < 3 or (stripped.isdigit() and len(stripped) <= 4):
        return 0, stripped

    # Nível 1 — Unidade
    for pat in _UNIT_PATTERNS:
        match = pat.match(stripped)
        if match:
            return 1, stripped

    # Nível 2 — Capítulo
    for pat in _CHAPTER_PATTERNS:
        match = pat.match(stripped)
        if match:
            return 2, stripped

    # Nível 3 — Tópico
    for pat in _TOPIC_PATTERNS:
        match = pat.match(stripped)
        if match:
            return 3, stripped

    # Heading ALL-CAPS genérico: linha curta sem sinais de corpo
    # Mas ignora se tem muitos números (provavelmente é página/data)
    if (
        _ALLCAPS_HEADING.match(stripped)
        and not _BODY_SIGNALS.search(stripped)
        and len(stripped.split()) <= 12
        and len(stripped) >= 10  # Mínimo de 10 chars para ser título
        and sum(c.isdigit() for c in stripped) < len(stripped) * 0.3  # < 30% números
    ):
        # Heurística: se parece cabeçalho mas não sabemos o nível, usa 2
        return 2, stripped

    return 0, stripped


# ---------------------------------------------------------------------------
# Construtor de outline
# ---------------------------------------------------------------------------


@dataclass
class _OutlineState:
    """Estado mutável do parser linha a linha."""

    current_unit: Optional[str] = None
    current_chapter: Optional[str] = None
    current_topic: Optional[str] = None
    buffer_lines: list[str] = field(default_factory=list)
    buffer_page: Optional[int] = None
    sections: list[DocumentSection] = field(default_factory=list)

    def flush(self, source: str) -> None:
        """Emite a seção acumulada no buffer."""
        text = "\n".join(self.buffer_lines).strip()
        if not text:
            self.buffer_lines = []
            return

        # Determina o nível ativo mais específico
        level = 0
        if self.current_topic:
            level = 3
        elif self.current_chapter:
            level = 2
        elif self.current_unit:
            level = 1

        self.sections.append(
            DocumentSection(
                unit=self.current_unit,
                chapter=self.current_chapter,
                topic=self.current_topic,
                level=level,
                text=text,
                source=source,
                page=self.buffer_page,
            )
        )
        self.buffer_lines = []
        self.buffer_page = None

    def push_line(self, line: str, page: Optional[int] = None) -> None:
        if self.buffer_page is None and page is not None:
            self.buffer_page = page
        self.buffer_lines.append(line)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def detect_structure(
    pages: list[dict],
    source: str,
) -> list[DocumentSection]:
    """Detecta a estrutura hierárquica unit/chapter/topic de uma lista de páginas.

    Args:
        pages:  Lista de dicts {text, page} como retornados por load_documents().
                O campo 'page' pode ser None.
        source: Caminho do arquivo (para preencher DocumentSection.source).

    Returns:
        Lista de DocumentSection com texto, hierarquia e metadados.
        Seções sem texto são descartadas.

    Exemplo de uso::

        pages = load_pdf_pages("livro.pdf")
        sections = detect_structure(pages, "livro.pdf")
        for sec in sections:
            print(sec.breadcrumb, "->", sec.text[:80])
    """
    state = _OutlineState()

    for page_dict in pages:
        raw_text: str = page_dict.get("text", "")
        page_num: Optional[int] = page_dict.get("page")

        # Ignora páginas de sumário/índice (detecta por densidade de pontos)
        # Sumários têm muitos "..." consecutivos
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        if len(lines) > 5:
            dots_lines = sum(1 for line in lines if "....." in line or "…" in line)
            # Se mais de 40% das linhas têm muitos pontos, é sumário
            if dots_lines > len(lines) * 0.4:
                continue

        for raw_line in raw_text.splitlines():
            level, line = _classify_line(raw_line)

            if level == 0:
                state.push_line(line, page_num)
                continue

            # Heading encontrado → fecha buffer anterior
            state.flush(source)

            # Limpa o título (remove números de página e pontos no final)
            clean_title = re.sub(r"\s*\.{3,}.*$", "", line)  # Remove "... 123"
            clean_title = re.sub(
                r"\s+\d{1,4}\s*$", "", clean_title
            )  # Remove número de página no final
            clean_title = clean_title.strip()

            if level == 1:
                state.current_unit = clean_title
                state.current_chapter = None
                state.current_topic = None
            elif level == 2:
                state.current_chapter = clean_title
                state.current_topic = None
            elif level == 3:
                state.current_topic = clean_title

            # Inclui o próprio heading no buffer como primeira linha da seção
            state.push_line(clean_title, page_num)

    # Último buffer
    state.flush(source)

    return state.sections


def sections_to_docs(sections: list[DocumentSection]) -> list[dict]:
    """Converte DocumentSection para o formato de dict esperado pelo pipeline.

    Compatível com o formato {text, source, page, section} usado pelo
    chunk_document() e vectorstore.add().
    """
    return [s.to_dict() for s in sections if s.text.strip()]

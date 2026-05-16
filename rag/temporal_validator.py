"""
Validador temporal para respostas históricas.

Este módulo detecta e valida datas mencionadas em respostas do chatbot,
garantindo que períodos históricos sejam corretamente identificados e
que não haja confusão entre eventos de diferentes épocas.

Funcionalidades:
- Extração de datas e períodos de textos
- Validação de consistência temporal
- Detecção de conflitos entre datas mencionadas
- Base de conhecimento de períodos históricos brasileiros
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Base de conhecimento de períodos históricos importantes
HISTORICAL_PERIODS = {
    # Getúlio Vargas
    "governo provisório vargas": {
        "start": 1930,
        "end": 1934,
        "aliases": ["governo provisório", "revolução de 1930"],
    },
    "governo constitucional vargas": {
        "start": 1934,
        "end": 1937,
        "aliases": ["governo constitucional"],
    },
    "estado novo": {
        "start": 1937,
        "end": 1945,
        "aliases": ["estado novo vargas", "ditadura vargas"],
    },
    "era vargas": {
        "start": 1930,
        "end": 1945,
        "aliases": ["primeiro período vargas", "primeira era vargas"],
    },
    "segundo governo vargas": {
        "start": 1951,
        "end": 1954,
        "aliases": ["segundo mandato vargas", "governo democrático vargas"],
    },
    # Outros períodos importantes
    "primeira guerra mundial": {
        "start": 1914,
        "end": 1918,
        "aliases": ["primeira guerra", "grande guerra"],
    },
    "segunda guerra mundial": {
        "start": 1939,
        "end": 1945,
        "aliases": ["segunda guerra", "wwii"],
    },
    "revolução industrial": {
        "start": 1760,
        "end": 1840,
        "aliases": ["industrialização"],
    },
    "império napoleônico": {
        "start": 1804,
        "end": 1814,
        "aliases": ["napoleão imperador", "império francês"],
    },
    "consulado napoleão": {
        "start": 1799,
        "end": 1804,
        "aliases": ["consulado francês", "napoleão cônsul"],
    },
}


def extract_years(text: str) -> list[int]:
    """Extrai anos mencionados no texto (formato: 4 dígitos entre 1000-2100).

    Args:
        text: Texto para análise

    Returns:
        Lista de anos encontrados (ordenada, sem duplicatas)
    """
    # Padrão para anos de 4 dígitos
    year_pattern = r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b"
    years = [int(y) for y in re.findall(year_pattern, text)]
    return sorted(set(years))


def extract_periods(text: str) -> list[tuple[int, int]]:
    """Extrai períodos no formato YYYY-YYYY do texto.

    Args:
        text: Texto para análise

    Returns:
        Lista de tuplas (ano_inicio, ano_fim)
    """
    # Padrão para períodos: 1930-1945, 1951-1954, etc.
    period_pattern = r"\b(1[0-9]{3}|20[0-9]{2})[-–—](1[0-9]{3}|20[0-9]{2})\b"
    matches = re.findall(period_pattern, text)
    return [(int(start), int(end)) for start, end in matches]


def identify_historical_period(text: str) -> Optional[dict]:
    """Identifica qual período histórico está sendo discutido no texto.

    Args:
        text: Texto para análise

    Returns:
        Dicionário com informações do período ou None se não identificado
    """
    text_lower = text.lower()

    for period_name, period_info in HISTORICAL_PERIODS.items():
        # Verifica nome principal
        if period_name in text_lower:
            return {
                "name": period_name,
                "start": period_info["start"],
                "end": period_info["end"],
            }

        # Verifica aliases
        for alias in period_info.get("aliases", []):
            if alias.lower() in text_lower:
                return {
                    "name": period_name,
                    "start": period_info["start"],
                    "end": period_info["end"],
                }

    return None


def validate_temporal_consistency(query: str, response: str) -> dict:
    """Valida se a resposta é temporalmente consistente com a pergunta.

    Verifica:
    1. Se as datas mencionadas na resposta são consistentes entre si
    2. Se o período identificado na query corresponde ao período na resposta
    3. Se há conflitos temporais óbvios

    Args:
        query: Pergunta do usuário
        response: Resposta do chatbot

    Returns:
        Dicionário com resultado da validação:
        {
            "valid": bool,
            "issues": list[str],  # Lista de problemas encontrados
            "query_period": dict | None,
            "response_period": dict | None,
            "years_mentioned": list[int],
        }
    """
    issues = []

    # Identifica períodos na query e na resposta
    query_period = identify_historical_period(query)
    response_period = identify_historical_period(response)

    # Extrai anos mencionados na resposta
    years = extract_years(response)
    periods = extract_periods(response)

    # Validação 1: Conflito entre período identificado e datas mencionadas
    if response_period and years:
        period_start = response_period["start"]
        period_end = response_period["end"]

        for year in years:
            if not (period_start <= year <= period_end):
                issues.append(
                    f"Ano {year} mencionado está fora do período "
                    f"{response_period['name']} ({period_start}-{period_end})"
                )

    # Validação 2: Query e resposta falam de períodos diferentes
    if query_period and response_period:
        if query_period["name"] != response_period["name"]:
            # Verifica se não são períodos relacionados (ex: Era Vargas contém Governo Provisório)
            query_range = (query_period["start"], query_period["end"])
            response_range = (response_period["start"], response_period["end"])

            # Verifica se um período contém o outro
            query_contains_response = (
                query_range[0] <= response_range[0]
                and query_range[1] >= response_range[1]
            )
            response_contains_query = (
                response_range[0] <= query_range[0]
                and response_range[1] >= query_range[1]
            )

            if not (query_contains_response or response_contains_query):
                issues.append(
                    f"Query menciona '{query_period['name']}' "
                    f"({query_period['start']}-{query_period['end']}) "
                    f"mas resposta fala sobre '{response_period['name']}' "
                    f"({response_period['start']}-{response_period['end']})"
                )

    # Validação 3: Períodos extraídos são consistentes
    for start, end in periods:
        if start > end:
            issues.append(f"Período inválido: {start}-{end} (início > fim)")

    # Validação 4: Caso específico - "Segundo Governo Vargas"
    if "segundo governo" in query.lower() and "vargas" in query.lower():
        # Deve mencionar 1951-1954, não 1934-1937
        if any(1934 <= y <= 1937 for y in years):
            issues.append(
                "Resposta menciona período 1934-1937 (Governo Constitucional) "
                "mas query pergunta sobre Segundo Governo Vargas (1951-1954)"
            )

        # Se não menciona o período correto, é um problema
        if not any(1951 <= y <= 1954 for y in years) and not periods:
            issues.append(
                "Resposta não menciona o período correto do Segundo Governo Vargas (1951-1954)"
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "query_period": query_period,
        "response_period": response_period,
        "years_mentioned": years,
        "periods_mentioned": periods,
    }


def get_period_context(period_name: str) -> Optional[str]:
    """Retorna contexto adicional sobre um período histórico.

    Args:
        period_name: Nome do período (deve estar em HISTORICAL_PERIODS)

    Returns:
        String com contexto adicional ou None se período não encontrado
    """
    period_name_lower = period_name.lower()

    for name, info in HISTORICAL_PERIODS.items():
        if name == period_name_lower or period_name_lower in info.get("aliases", []):
            return f"{name.title()} ({info['start']}-{info['end']})"

    return None

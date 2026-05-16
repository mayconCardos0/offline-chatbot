"""
Testes para o módulo de validação temporal.
"""

import pytest

from rag.temporal_validator import (
    extract_periods,
    extract_years,
    identify_historical_period,
    validate_temporal_consistency,
)


class TestExtractYears:
    """Testes para extração de anos."""

    def test_extract_single_year(self):
        text = "O evento ocorreu em 1954"
        years = extract_years(text)
        assert years == [1954]

    def test_extract_multiple_years(self):
        text = "Entre 1930 e 1945, houve mudanças significativas"
        years = extract_years(text)
        assert years == [1930, 1945]

    def test_extract_no_years(self):
        text = "Não há anos mencionados aqui"
        years = extract_years(text)
        assert years == []

    def test_extract_years_removes_duplicates(self):
        text = "Em 1930, 1930 e 1930 aconteceu a revolução"
        years = extract_years(text)
        assert years == [1930]

    def test_extract_years_sorted(self):
        text = "Anos importantes: 1954, 1930, 1945"
        years = extract_years(text)
        assert years == [1930, 1945, 1954]


class TestExtractPeriods:
    """Testes para extração de períodos."""

    def test_extract_single_period(self):
        text = "O governo durou de 1951-1954"
        periods = extract_periods(text)
        assert periods == [(1951, 1954)]

    def test_extract_multiple_periods(self):
        text = "Primeiro período: 1930-1945, segundo: 1951-1954"
        periods = extract_periods(text)
        assert periods == [(1930, 1945), (1951, 1954)]

    def test_extract_no_periods(self):
        text = "Sem períodos aqui"
        periods = extract_periods(text)
        assert periods == []

    def test_extract_periods_with_different_separators(self):
        text = "Períodos: 1930-1945, 1951–1954, 1960—1964"
        periods = extract_periods(text)
        assert len(periods) == 3


class TestIdentifyHistoricalPeriod:
    """Testes para identificação de períodos históricos."""

    def test_identify_segundo_governo_vargas(self):
        text = "O segundo governo Vargas foi importante"
        period = identify_historical_period(text)
        assert period is not None
        assert period["name"] == "segundo governo vargas"
        assert period["start"] == 1951
        assert period["end"] == 1954

    def test_identify_era_vargas(self):
        text = "Durante a Era Vargas houve mudanças"
        period = identify_historical_period(text)
        assert period is not None
        assert period["name"] == "era vargas"
        assert period["start"] == 1930
        assert period["end"] == 1945

    def test_identify_by_alias(self):
        text = "No governo provisório de Vargas"
        period = identify_historical_period(text)
        assert period is not None
        assert period["name"] == "governo provisório vargas"

    def test_identify_no_period(self):
        text = "Texto sem período histórico conhecido"
        period = identify_historical_period(text)
        assert period is None


class TestValidateTemporalConsistency:
    """Testes para validação de consistência temporal."""

    def test_valid_segundo_governo_vargas(self):
        query = "Como foi o Segundo Governo Vargas?"
        response = "O Segundo Governo Vargas (1951-1954) foi marcado pelo nacionalismo."
        
        validation = validate_temporal_consistency(query, response)
        
        assert validation["valid"] is True
        assert len(validation["issues"]) == 0
        assert validation["query_period"]["name"] == "segundo governo vargas"

    def test_invalid_segundo_governo_wrong_dates(self):
        query = "Como foi o Segundo Governo Vargas?"
        response = "O Segundo Governo Vargas, entre 1934 e 1937, foi centralizador."
        
        validation = validate_temporal_consistency(query, response)
        
        assert validation["valid"] is False
        assert len(validation["issues"]) > 0
        # Deve detectar que menciona 1934-1937 quando deveria ser 1951-1954
        assert any("1934" in issue and "1937" in issue for issue in validation["issues"])

    def test_invalid_period_conflict(self):
        query = "Fale sobre o Estado Novo"
        response = "O Estado Novo (1930-1934) foi um período importante."
        
        validation = validate_temporal_consistency(query, response)
        
        assert validation["valid"] is False
        # Estado Novo foi 1937-1945, não 1930-1934

    def test_valid_no_specific_period(self):
        query = "O que foi a Revolução Industrial?"
        response = "A Revolução Industrial transformou a sociedade."
        
        validation = validate_temporal_consistency(query, response)
        
        # Sem períodos específicos conflitantes, deve ser válido
        assert validation["valid"] is True

    def test_years_outside_period(self):
        query = "Fale sobre o Estado Novo"
        response = "O Estado Novo foi importante. Em 1930 houve mudanças."
        
        validation = validate_temporal_consistency(query, response)
        
        # 1930 está fora do Estado Novo (1937-1945)
        assert validation["valid"] is False
        assert any("1930" in issue for issue in validation["issues"])

    def test_invalid_period_format(self):
        query = "Teste"
        response = "Período inválido: 1954-1930"
        
        validation = validate_temporal_consistency(query, response)
        
        assert validation["valid"] is False
        assert any("inválido" in issue.lower() for issue in validation["issues"])


class TestEdgeCases:
    """Testes de casos extremos."""

    def test_empty_texts(self):
        validation = validate_temporal_consistency("", "")
        assert validation["valid"] is True
        assert validation["query_period"] is None
        assert validation["response_period"] is None

    def test_only_query_has_period(self):
        query = "Como foi o Segundo Governo Vargas?"
        response = "Foi um período importante para o Brasil."
        
        validation = validate_temporal_consistency(query, response)
        
        # Sem datas na resposta, mas query tem período
        # Deve alertar que não menciona o período correto
        assert validation["valid"] is False

    def test_overlapping_periods(self):
        query = "Fale sobre a Era Vargas"
        response = "Durante o Governo Provisório (1930-1934) houve mudanças."
        
        validation = validate_temporal_consistency(query, response)
        
        # Governo Provisório está contido na Era Vargas, deve ser válido
        assert validation["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

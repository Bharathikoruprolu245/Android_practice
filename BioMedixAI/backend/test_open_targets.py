"""
Unit tests for the Open Targets switch: OpenTargetsProvider and
FallbackGeneProvider (data_ingestor.py).

Run with:  pytest test_open_targets_provider.py -v

Mocks requests.post so this runs offline and fast, same convention as the
rest of the suite. No database or network required.
"""

from unittest.mock import patch, MagicMock

import pytest

from data_ingestor import (
    OpenTargetsProvider,
    FallbackGeneProvider,
    MockGeneProvider,
    DiseaseGeneProvider,
)


def _mock_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


SEARCH_HIT = {"data": {"search": {"hits": [{"id": "EFO_0000249", "name": "Alzheimer disease", "entity": "disease"}]}}}
SEARCH_MISS = {"data": {"search": {"hits": []}}}
TARGETS_HIT = {
    "data": {
        "disease": {
            "id": "EFO_0000249",
            "name": "Alzheimer disease",
            "associatedTargets": {
                "rows": [
                    {"score": 0.9, "target": {"approvedSymbol": "APP"}},
                    {"score": 0.8, "target": {"approvedSymbol": "PSEN1"}},
                ]
            },
        }
    }
}
TARGETS_EMPTY = {"data": {"disease": {"id": "EFO_0000249", "name": "X", "associatedTargets": {"rows": []}}}}


class TestOpenTargetsProvider:
    def test_implements_abstract_base(self):
        assert isinstance(OpenTargetsProvider(), DiseaseGeneProvider)

    def test_happy_path_returns_genes_ranked_by_score(self):
        provider = OpenTargetsProvider()
        with patch("requests.post", side_effect=[_mock_response(SEARCH_HIT), _mock_response(TARGETS_HIT)]):
            genes = provider.get_genes_for_disease("alzheimer's disease", limit=2)
        assert genes == ["APP", "PSEN1"]

    def test_unknown_disease_raises_value_error(self):
        provider = OpenTargetsProvider()
        with patch("requests.post", return_value=_mock_response(SEARCH_MISS)):
            with pytest.raises(ValueError, match="no disease matching"):
                provider.get_genes_for_disease("not a real disease xyz")

    def test_disease_with_no_associations_raises_value_error(self):
        provider = OpenTargetsProvider()
        with patch(
            "requests.post", side_effect=[_mock_response(SEARCH_HIT), _mock_response(TARGETS_EMPTY)]
        ):
            with pytest.raises(ValueError, match="no gene associations"):
                provider.get_genes_for_disease("alzheimer's disease")

    def test_graphql_error_payload_raises_value_error(self):
        provider = OpenTargetsProvider()
        error_resp = _mock_response({"errors": [{"message": "boom"}]})
        with patch("requests.post", return_value=error_resp):
            with pytest.raises(ValueError):
                provider.get_genes_for_disease("alzheimer's disease")


class TestFallbackGeneProvider:
    def test_uses_primary_when_it_succeeds(self):
        primary = MagicMock(spec=DiseaseGeneProvider)
        primary.get_genes_for_disease.return_value = ["APP", "PSEN1"]
        fallback = MockGeneProvider()
        provider = FallbackGeneProvider(primary=primary, fallback=fallback)

        genes = provider.get_genes_for_disease("alzheimer's disease", limit=2)
        assert genes == ["APP", "PSEN1"]

    def test_falls_back_when_primary_raises(self):
        primary = MagicMock(spec=DiseaseGeneProvider)
        primary.get_genes_for_disease.side_effect = ValueError("Open Targets found no disease matching X")
        fallback = MockGeneProvider()
        provider = FallbackGeneProvider(primary=primary, fallback=fallback)

        # non-human trait Open Targets can't know about -> should still work via mock
        genes = provider.get_genes_for_disease("rice cadmium toxicity")
        assert "OsNramp5" in genes

    def test_falls_back_when_primary_returns_empty_list(self):
        primary = MagicMock(spec=DiseaseGeneProvider)
        primary.get_genes_for_disease.return_value = []
        fallback = MockGeneProvider()
        provider = FallbackGeneProvider(primary=primary, fallback=fallback)

        genes = provider.get_genes_for_disease("alzheimer's disease")
        assert "APP" in genes  # from MockGeneProvider's curated list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Unit tests for Module 1 (TargetDiscoveryPipeline and its components).

Run with:  pytest test_pipeline.py -v

These use mocks/fixtures for external APIs so the suite runs offline and
fast — no dependency on STRING DB or DisGeNET being reachable. A local
PostgreSQL instance IS required (migrated from the SQLite tempfile-per-test
approach, since Postgres doesn't have a throwaway-file equivalent).

Point TEST_DATABASE_URL at a scratch database before running, e.g.:
    export TEST_DATABASE_URL="postgresql://biomedix_user:password@localhost:5432/biomedix_test_db"
The fixture truncates all Module 1 tables before each test via
Database.reset(), so tests remain isolated and repeatable even though
they share one physical database.
"""

import os
import pytest
from unittest.mock import patch

from data_ingestor import (
    DataIngestor,
    MockGeneProvider,
    DiseaseGeneProvider,
    resolve_species,
    SPECIES_TAXONOMY_IDS,
)
from network_analyzer import NetworkAnalyzer
from target_discovery import TargetDiscoveryPipeline, HubResult
from db import Database

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://biomedix_user:password@localhost:5432/biomedix_test_db",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db():
    """
    A clean slate on the shared Postgres test database for every test.
    Truncates Module 1 tables (with identity reset) before and after, so
    each test sees an empty DB regardless of execution order.
    """
    db = Database(TEST_DATABASE_URL)
    db.reset()
    yield db
    db.reset()
    db.close()


@pytest.fixture
def pipeline(temp_db):
    provider = MockGeneProvider()
    ingestor = DataIngestor(gene_provider=provider)
    analyzer = NetworkAnalyzer()
    return TargetDiscoveryPipeline(ingestor=ingestor, analyzer=analyzer, db=temp_db)


FAKE_RICE_EDGES = [
    ("OsNramp5", "OsLsi2", 900),
    ("OsNramp5", "OsHMA3", 850),
    ("OsHMA3", "OsHMA2", 700),
]


# ---------------------------------------------------------------------------
# MockGeneProvider
# ---------------------------------------------------------------------------
class TestMockGeneProvider:
    def test_returns_known_disease(self):
        provider = MockGeneProvider()
        genes = provider.get_genes_for_disease("rice cadmium toxicity")
        assert "OsNramp5" in genes
        assert "OsLsi2" in genes

    def test_respects_limit(self):
        provider = MockGeneProvider()
        genes = provider.get_genes_for_disease("alzheimer's disease", limit=3)
        assert len(genes) == 3

    def test_case_insensitive(self):
        provider = MockGeneProvider()
        genes_lower = provider.get_genes_for_disease("rice cadmium toxicity")
        genes_upper = provider.get_genes_for_disease("RICE CADMIUM TOXICITY")
        assert genes_lower == genes_upper

    def test_unknown_disease_raises(self):
        provider = MockGeneProvider()
        with pytest.raises(ValueError):
            provider.get_genes_for_disease("completely made up disease xyz123")

    def test_implements_abstract_base(self):
        assert isinstance(MockGeneProvider(), DiseaseGeneProvider)


# ---------------------------------------------------------------------------
# resolve_species
# ---------------------------------------------------------------------------
class TestResolveSpecies:
    def test_known_common_name(self):
        assert resolve_species("rice") == 4530
        assert resolve_species("human") == 9606

    def test_case_insensitive(self):
        assert resolve_species("Rice") == 4530
        assert resolve_species("HUMAN") == 9606

    def test_passthrough_int(self):
        assert resolve_species(9606) == 9606
        assert resolve_species(12345) == 12345  # unknown ID still passes through

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError):
            resolve_species("dinosaur")

    def test_all_species_map_to_ints(self):
        for name, taxid in SPECIES_TAXONOMY_IDS.items():
            assert isinstance(taxid, int)
            assert resolve_species(name) == taxid


# ---------------------------------------------------------------------------
# NetworkAnalyzer
# ---------------------------------------------------------------------------
class TestNetworkAnalyzer:
    def test_build_graph_creates_correct_nodes_and_edges(self):
        analyzer = NetworkAnalyzer()
        analyzer.build_graph(FAKE_RICE_EDGES)
        assert analyzer.graph.number_of_nodes() == 4
        assert analyzer.graph.number_of_edges() == 3

    def test_empty_edges_gives_empty_graph(self):
        analyzer = NetworkAnalyzer()
        analyzer.build_graph([])
        assert analyzer.graph.number_of_nodes() == 0

    def test_degree_centrality_picks_most_connected_gene(self):
        analyzer = NetworkAnalyzer()
        analyzer.build_graph(FAKE_RICE_EDGES)
        hub, scores = analyzer.get_top_hub(method="degree")
        # OsNramp5 and OsHMA3 both have degree 2 (tied for most connections)
        assert hub in ("OsNramp5", "OsHMA3")

    def test_get_top_hub_on_empty_graph_raises(self):
        analyzer = NetworkAnalyzer()
        analyzer.build_graph([])
        with pytest.raises(ValueError):
            analyzer.get_top_hub()

    def test_invalid_method_raises(self):
        analyzer = NetworkAnalyzer()
        analyzer.build_graph(FAKE_RICE_EDGES)
        with pytest.raises(ValueError):
            analyzer.compute_centrality(method="not_a_real_method")

    def test_graph_summary_shape(self):
        analyzer = NetworkAnalyzer()
        analyzer.build_graph(FAKE_RICE_EDGES)
        summary = analyzer.graph_summary()
        assert summary["num_nodes"] == 4
        assert summary["num_edges"] == 3
        assert "OsNramp5" in summary["nodes"]


# ---------------------------------------------------------------------------
# TargetDiscoveryPipeline (end-to-end, STRING DB mocked out)
# ---------------------------------------------------------------------------
class TestTargetDiscoveryPipeline:
    def test_run_returns_hub_result(self, pipeline):
        with patch.object(
            pipeline.ingestor, "fetch_string_interactions", return_value=FAKE_RICE_EDGES
        ):
            result = pipeline.run("rice cadmium toxicity", species="rice")
        assert isinstance(result, HubResult)
        assert result.disease_name == "rice cadmium toxicity"
        assert result.hub_gene_symbol in ("OsNramp5", "OsHMA3")

    def test_run_persists_to_database(self, pipeline, temp_db):
        with patch.object(
            pipeline.ingestor, "fetch_string_interactions", return_value=FAKE_RICE_EDGES
        ):
            pipeline.run("rice cadmium toxicity", species="rice")

        diseases = temp_db.fetch_all("DISEASE")
        genes = temp_db.fetch_all("GENE")
        assocs = temp_db.fetch_all("DISEASE_GENE_ASSOC")
        ppis = temp_db.fetch_all("PPI_INTERACTION")
        runs = temp_db.fetch_all("NETWORK_ANALYSIS_RUN")

        assert len(diseases) == 1
        assert len(genes) == 8  # full mock gene list for rice cadmium toxicity
        assert len(assocs) == 8
        assert len(ppis) == 3
        assert len(runs) == 1

    def test_ppi_scores_persist_correctly_not_zero(self, pipeline, temp_db):
        """Regression test for the int(0.9)==0 truncation bug."""
        with patch.object(
            pipeline.ingestor, "fetch_string_interactions", return_value=FAKE_RICE_EDGES
        ):
            pipeline.run("rice cadmium toxicity", species="rice")

        ppis = temp_db.fetch_all("PPI_INTERACTION")
        scores = [row["combined_score"] for row in ppis]
        assert all(s > 0 for s in scores), (
            "PPI combined_score values must not be zero — "
            "check for float truncation bugs."
        )

    def test_no_edges_falls_back_gracefully(self, pipeline):
        """When STRING returns nothing, pipeline should still produce a
        usable HubResult instead of crashing."""
        with patch.object(
            pipeline.ingestor, "fetch_string_interactions", return_value=[]
        ):
            result = pipeline.run("rice cadmium toxicity", species="rice")
        assert isinstance(result, HubResult)
        assert result.hub_gene_symbol == "OsNramp5"  # first gene in mock list
        assert result.graph_summary["num_edges"] == 0

    def test_running_twice_does_not_duplicate_disease_or_gene_rows(
        self, pipeline, temp_db
    ):
        """Upsert logic should prevent duplicate DISEASE/GENE rows on reruns."""
        with patch.object(
            pipeline.ingestor, "fetch_string_interactions", return_value=FAKE_RICE_EDGES
        ):
            pipeline.run("rice cadmium toxicity", species="rice")
            pipeline.run("rice cadmium toxicity", species="rice")

        diseases = temp_db.fetch_all("DISEASE")
        genes = temp_db.fetch_all("GENE")
        runs = temp_db.fetch_all("NETWORK_ANALYSIS_RUN")

        assert len(diseases) == 1  # not duplicated
        assert len(genes) == 8  # not duplicated
        assert len(runs) == 2  # each run DOES get its own row (that's by design)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
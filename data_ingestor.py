"""
Module 1 — Data Ingestion Layer
Implements: DiseaseGeneProvider (abstract), MockGeneProvider, DisGeNETProvider, DataIngestor

Matches class diagram:
    <<abstract>> DiseaseGeneProvider
        +get_genes_for_disease(name, limit)
    DisGeNETProvider(-api_key: str)
    MockGeneProvider(-curated: Dict)
    DataIngestor(-gene_provider: DiseaseGeneProvider)
        +fetch_disease_genes(name)
        +fetch_string_interactions(genes)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import requests


# STRING DB uses NCBI taxonomy IDs for species. This lookup lets callers
# write pipeline.run(..., species="rice") instead of memorizing "4530".
SPECIES_TAXONOMY_IDS = {
    "human": 9606,
    "rice": 4530,
    "mouse": 10090,
    "arabidopsis": 3702,
    "yeast": 4932,
    "zebrafish": 7955,
    "fruit fly": 7227,
    "e. coli": 511145,
}


def resolve_species(species) -> int:
    """
    Accepts either a common-name string ("rice", "human") or a raw NCBI
    taxonomy ID (int, e.g. 4530) and always returns the integer ID that
    STRING DB's API expects.
    """
    if isinstance(species, int):
        return species
    key = str(species).strip().lower()
    if key in SPECIES_TAXONOMY_IDS:
        return SPECIES_TAXONOMY_IDS[key]
    raise ValueError(
        f"Unknown species '{species}'. Use a taxonomy ID (int) or one of: "
        f"{list(SPECIES_TAXONOMY_IDS.keys())}"
    )


# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------
class DiseaseGeneProvider(ABC):
    """Abstract base — any gene-source (real API or mock) must implement this."""

    @abstractmethod
    def get_genes_for_disease(self, name: str, limit: int = 10) -> List[str]:
        """Return a list of gene symbols associated with a disease/trait name."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock provider — no API key needed, works offline
# ---------------------------------------------------------------------------
class MockGeneProvider(DiseaseGeneProvider):
    """
    Curated lookup table so you can build/test the whole pipeline without
    needing a DisGeNET API key. Seeded with your rice heavy-metal genes
    (OsNramp5, OsLsi2) plus a few generic human-disease examples so the
    pipeline is demonstrably disease-agnostic.
    """

    def __init__(self):
        self.curated: Dict[str, List[str]] = {
            "rice cadmium toxicity": [
                "OsNramp5", "OsLsi2", "OsHMA3", "OsHMA2",
                "OsLCT1", "OsCAL1", "OsABCC1", "OsPCS1",
            ],
            "rice heavy metal accumulation": [
                "OsNramp5", "OsLsi2", "OsHMA3", "OsHMA2",
                "OsNramp1", "OsIRT1", "OsFRDL1", "OsZIP5",
            ],
            "alzheimer's disease": [
                "APP", "PSEN1", "PSEN2", "APOE", "MAPT",
                "TREM2", "CLU", "BIN1", "ABCA7", "SORL1",
            ],
            "type 2 diabetes": [
                "TCF7L2", "PPARG", "KCNJ11", "INS", "IRS1",
                "GCK", "SLC30A8", "HNF4A", "ABCC8", "WFS1",
            ],
        }

    def get_genes_for_disease(self, name: str, limit: int = 10) -> List[str]:
        key = name.strip().lower()
        genes = self.curated.get(key)
        if genes is None:
            # graceful fallback so the pipeline never hard-crashes on typos
            close_matches = [k for k in self.curated if key in k or k in key]
            if close_matches:
                genes = self.curated[close_matches[0]]
            else:
                raise ValueError(
                    f"No mock data for '{name}'. Available: {list(self.curated.keys())}"
                )
        return genes[:limit]


# ---------------------------------------------------------------------------
# Real provider — DisGeNET REST API
#
# NOTE (deprecated as of the Open Targets switch): DisGeNET's free/trial API
# key is rate-limited and restricted for evaluation use only, which made it
# unreliable for a pipeline that needs to run repeatedly during development
# and demos. Kept here for reference / in case a paid key is obtained later,
# but OpenTargetsProvider below is now the default real provider — see
# FallbackGeneProvider for how the two "real vs. mock" providers combine.
# ---------------------------------------------------------------------------
class DisGeNETProvider(DiseaseGeneProvider):
    """
    Wraps the DisGeNET API (https://www.disgenet.org/api/).
    Requires an API key (free academic registration).
    """

    BASE_URL = "https://www.disgenet.org/api"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_genes_for_disease(self, name: str, limit: int = 10) -> List[str]:
        # Step 1: resolve free-text disease name -> UMLS CUI
        search_resp = requests.get(
            f"{self.BASE_URL}/disease/name/{name}",
            headers={"Authorization": self.api_key},
            timeout=15,
        )
        search_resp.raise_for_status()
        results = search_resp.json()
        if not results:
            raise ValueError(f"DisGeNET found no disease matching '{name}'")
        cui = results[0]["diseaseId"]

        # Step 2: fetch gene-disease associations for that CUI
        gda_resp = requests.get(
            f"{self.BASE_URL}/gda/disease/{cui}",
            params={"source": "CURATED"},
            headers={"Authorization": self.api_key},
            timeout=15,
        )
        gda_resp.raise_for_status()
        associations = gda_resp.json()

        # sort by association score, descending
        associations.sort(key=lambda a: a.get("score", 0), reverse=True)
        return [a["gene_symbol"] for a in associations[:limit]]


# ---------------------------------------------------------------------------
# Real provider — Open Targets Platform GraphQL API (replaces DisGeNET)
# ---------------------------------------------------------------------------
class OpenTargetsProvider(DiseaseGeneProvider):
    """
    Wraps the Open Targets Platform API
    (https://platform-docs.opentargets.org/api). No API key required, no
    rate-limit tier to worry about — this is the same public GraphQL
    endpoint used by the Open Targets website itself.

    Two GraphQL calls per lookup, same two-step shape as DisGeNETProvider:
      1. `search` — resolve the free-text disease name to an EFO disease ID.
      2. `disease.associatedTargets` — fetch genes associated with that EFO
         ID, already ranked by Open Targets' own association score.

    NOTE: Open Targets' disease-target associations are human-only. For
    non-human diseases/traits (e.g. "rice cadmium toxicity"), the search
    step will simply find no match, this raises ValueError, and
    FallbackGeneProvider (below) drops through to MockGeneProvider's
    curated plant-gene entries — exactly the graceful degradation SRS UC-2
    describes, just triggered by "wrong domain" instead of "API down".
    """

    BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def _graphql(self, query: str, variables: Dict) -> Dict:
        resp = requests.post(
            self.BASE_URL,
            json={"query": query, "variables": variables},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload and payload["errors"]:
            raise ValueError(f"Open Targets GraphQL error: {payload['errors']}")
        return payload["data"]

    def _search_disease(self, name: str) -> Optional[str]:
        query = """
        query SearchDisease($q: String!) {
          search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 1}) {
            hits { id name entity }
          }
        }
        """
        data = self._graphql(query, {"q": name})
        hits = data.get("search", {}).get("hits", [])
        return hits[0]["id"] if hits else None

    def _fetch_associated_targets(self, efo_id: str, limit: int) -> List[Dict]:
        query = """
        query DiseaseTargets($efoId: String!, $size: Int!) {
          disease(efoId: $efoId) {
            id
            name
            associatedTargets(page: {index: 0, size: $size}) {
              rows {
                score
                target { approvedSymbol }
              }
            }
          }
        }
        """
        data = self._graphql(query, {"efoId": efo_id, "size": limit})
        disease = data.get("disease")
        if disease is None:
            return []
        return disease.get("associatedTargets", {}).get("rows", [])

    def get_genes_for_disease(self, name: str, limit: int = 10) -> List[str]:
        try:
            efo_id = self._search_disease(name)
        except requests.RequestException as e:
            raise ValueError(f"Open Targets unreachable while searching for '{name}': {e}")

        if efo_id is None:
            raise ValueError(f"Open Targets found no disease matching '{name}'")

        try:
            rows = self._fetch_associated_targets(efo_id, limit)
        except requests.RequestException as e:
            raise ValueError(f"Open Targets unreachable while fetching targets for '{name}': {e}")

        if not rows:
            raise ValueError(f"Open Targets found no gene associations for '{name}'")

        # Already sorted by Open Targets' own association score, descending.
        return [row["target"]["approvedSymbol"] for row in rows]


# ---------------------------------------------------------------------------
# FallbackGeneProvider — composite that actually implements SRS UC-2's
# exception flow (real API unreachable/no match -> curated offline dataset)
# ---------------------------------------------------------------------------
class FallbackGeneProvider(DiseaseGeneProvider):
    """
    Wraps a primary (real) provider and a fallback (usually MockGeneProvider)
    behind the same DiseaseGeneProvider interface, so DataIngestor doesn't
    need to know which one actually answered. Any exception from the
    primary — network error, no disease match, no associations — triggers
    a logged fallback instead of propagating up and failing the whole
    pipeline run (NFR-3/NFR-4).
    """

    def __init__(self, primary: DiseaseGeneProvider, fallback: DiseaseGeneProvider):
        self.primary = primary
        self.fallback = fallback

    def get_genes_for_disease(self, name: str, limit: int = 10) -> List[str]:
        try:
            genes = self.primary.get_genes_for_disease(name, limit)
            if not genes:
                raise ValueError("primary provider returned an empty gene list")
            return genes
        except Exception as e:
            print(
                f"[FallbackGeneProvider] {type(self.primary).__name__} failed for "
                f"'{name}' ({e}) — falling back to {type(self.fallback).__name__}."
            )
            return self.fallback.get_genes_for_disease(name, limit)


# ---------------------------------------------------------------------------
# DataIngestor — fetches genes, then fetches PPI edges from STRING DB
# ---------------------------------------------------------------------------
class DataIngestor:
    """
    Orchestrates: gene lookup (via injected provider) -> STRING DB PPI edges.
    STRING's API is free and keyless, so this half needs no credentials.
    """

    STRING_API_URL = "https://string-db.org/api/json/network"

    def __init__(self, gene_provider: DiseaseGeneProvider):
        self.gene_provider = gene_provider

    def fetch_disease_genes(self, name: str, limit: int = 10) -> List[str]:
        return self.gene_provider.get_genes_for_disease(name, limit)

    def fetch_string_interactions(
        self, genes: List[str], species=9606
    ) -> List[Tuple[str, str, float]]:
        """
        Query STRING DB for PPI edges among the given gene symbols.
        species can be a common name ("rice", "human") or a raw NCBI
        taxonomy ID (int). See SPECIES_TAXONOMY_IDS for the built-in list.

        Returns list of (gene_a, gene_b, combined_score) tuples.
        Falls back to an empty-but-valid structure on network failure so
        the pipeline degrades gracefully instead of crashing.
        """
        species_id = resolve_species(species)
        try:
            resp = requests.get(
                self.STRING_API_URL,
                params={
                    "identifiers": "%0d".join(genes),
                    "species": species_id,
                    "caller_identity": "biomedix-ai",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            edges = [
                (
                    row["preferredName_A"],
                    row["preferredName_B"],
                    # STRING returns "score" as a 0-1 decimal (e.g. 0.983).
                    # Scale to STRING's conventional 0-1000 combined_score
                    # range so it survives being stored as an INT downstream.
                    round(float(row["score"]) * 1000),
                )
                for row in data
            ]
            return edges
        except (requests.RequestException, ValueError) as e:
            print(f"[DataIngestor] STRING DB fetch failed ({e}); returning no edges.")
            return []
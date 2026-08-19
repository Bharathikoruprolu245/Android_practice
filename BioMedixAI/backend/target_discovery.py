"""
Module 1 — Orchestration Layer
Implements: HubResult (dataclass), TargetDiscoveryPipeline

Matches class diagram:
    <<dataclass>> HubResult
        +disease_name: str
        +hub_gene_symbol: str
        +centrality_scores: Dict
        +graph_summary: Dict
    TargetDiscoveryPipeline(-ingestor: DataIngestor, -analyzer: NetworkAnalyzer)
        +run(disease_name): HubResult
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from data_ingestor import DataIngestor, DiseaseGeneProvider
from network_analyzer import NetworkAnalyzer
from db import Database


@dataclass
class HubResult:
    disease_name: str
    hub_gene_symbol: str
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    graph_summary: Dict = field(default_factory=dict)
    # Set only when persisted (db is not None) — the NETWORK_ANALYSIS_RUN
    # row id, needed by Module 4 to link its INTEGRATED_REPORT row back to
    # this run. None for in-memory/no-db runs, same optionality pattern as
    # DruggabilityResult.result_id / CrisprSafetyResult.result_id.
    run_id: Optional[int] = None

    def __str__(self) -> str:
        top5 = sorted(self.centrality_scores.items(), key=lambda x: -x[1])[:5]
        lines = [
            f"Disease:        {self.disease_name}",
            f"Hub gene:       {self.hub_gene_symbol}",
            f"Graph size:     {self.graph_summary.get('num_nodes', 0)} nodes, "
            f"{self.graph_summary.get('num_edges', 0)} edges",
            "Top 5 by centrality:",
        ]
        for gene, score in top5:
            lines.append(f"  {gene:<12} {score:.4f}")
        return "\n".join(lines)


class TargetDiscoveryPipeline:
    """
    Wires together DataIngestor -> NetworkAnalyzer per the activity diagram:
      Enter Disease Name
        -> Fetch Disease-Associated Genes (DisGeNET/Mock)
        -> Fetch PPI Edges (STRING DB)
        -> Build NetworkX Graph
        -> Compute Centrality & Select Hub Gene
    """

    def __init__(
        self,
        ingestor: DataIngestor,
        analyzer: NetworkAnalyzer,
        db: Optional[Database] = None,
    ):
        self.ingestor = ingestor
        self.analyzer = analyzer
        self.db = db  # if None, pipeline runs in-memory only (no persistence)

    def run(
        self,
        disease_name: str,
        gene_limit: int = 10,
        species="human",
        centrality_method: str = "degree",
        source_label: str = "mock",
    ) -> HubResult:
        # 1. Fetch disease-associated genes
        genes = self.ingestor.fetch_disease_genes(disease_name, limit=gene_limit)
        print(f"[Pipeline] {len(genes)} genes fetched for '{disease_name}': {genes}")

        # 2. Fetch PPI edges from STRING DB
        edges = self.ingestor.fetch_string_interactions(genes, species=species)
        print(f"[Pipeline] {len(edges)} PPI edges fetched from STRING DB")

        # 3. Build graph
        self.analyzer.build_graph(edges)

        # 4. Compute centrality & select hub gene
        if edges:
            hub_gene, scores = self.analyzer.get_top_hub(method=centrality_method)
        else:
            # No PPI edges found (offline, obscure genes, etc.) — fall back
            # to the first gene in the list so the pipeline still produces
            # a usable HubResult instead of crashing.
            hub_gene = genes[0]
            scores = {g: 0.0 for g in genes}
            print("[Pipeline] No edges found — falling back to first gene as hub.")

        result = HubResult(
            disease_name=disease_name,
            hub_gene_symbol=hub_gene,
            centrality_scores=scores,
            graph_summary=self.analyzer.graph_summary(),
        )

        # 5. Persist to DB, if one was provided
        if self.db is not None:
            result.run_id = self._persist(result, genes, edges, centrality_method, source_label)

        return result

    def _persist(
        self,
        result: HubResult,
        genes,
        edges,
        centrality_method: str,
        source_label: str,
    ) -> int:
        """Writes DISEASE, GENE, DISEASE_GENE_ASSOC, PPI_INTERACTION, and
        NETWORK_ANALYSIS_RUN rows per the ER diagram."""
        disease_id = self.db.upsert_disease(result.disease_name)

        gene_id_by_symbol = {}
        for gene_symbol in genes:
            gene_id = self.db.upsert_gene(gene_symbol)
            gene_id_by_symbol[gene_symbol] = gene_id
            self.db.insert_disease_gene_assoc(
                disease_id, gene_id, association_score=None, source=source_label
            )

        for gene_a, gene_b, score in edges:
            # STRING may return genes not in our original disease-gene list
            # (case normalization, aliases) — upsert them too so the FK holds.
            id_a = gene_id_by_symbol.get(gene_a) or self.db.upsert_gene(gene_a)
            id_b = gene_id_by_symbol.get(gene_b) or self.db.upsert_gene(gene_b)
            self.db.insert_ppi_interaction(id_a, id_b, combined_score=int(score))

        hub_gene_id = gene_id_by_symbol.get(
            result.hub_gene_symbol
        ) or self.db.upsert_gene(result.hub_gene_symbol)

        run_id = self.db.insert_network_analysis_run(
            disease_id=disease_id,
            hub_gene_id=hub_gene_id,
            centrality_method=centrality_method,
            centrality_score=result.centrality_scores.get(result.hub_gene_symbol, 0.0),
        )
        print(f"[Pipeline] Persisted run_id={run_id} to database.")
        return run_id
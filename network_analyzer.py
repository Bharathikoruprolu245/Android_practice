"""
Module 1 — Network Analysis Layer
Implements: NetworkAnalyzer

Matches class diagram:
    NetworkAnalyzer(-graph: nx.Graph)
        +build_graph(edges)
        +compute_centrality(method)
        +get_top_hub(method)
"""

from typing import List, Tuple, Dict
import networkx as nx


class NetworkAnalyzer:
    """
    Builds a protein-protein interaction graph from STRING DB edges and
    identifies the most "central" (hub) gene using a chosen centrality
    algorithm — this hub gene is the candidate therapeutic target that
    Modules 2 and 3 will investigate further.
    """

    SUPPORTED_METHODS = {"degree", "betweenness", "eigenvector", "closeness"}

    def __init__(self):
        self.graph: nx.Graph = nx.Graph()

    def build_graph(self, edges: List[Tuple[str, str, float]]) -> nx.Graph:
        """
        edges: list of (gene_a, gene_b, combined_score) from STRING DB.
        Edge weight = combined_score (higher = more confident interaction).
        """
        self.graph = nx.Graph()
        for gene_a, gene_b, score in edges:
            self.graph.add_edge(gene_a, gene_b, weight=score)
        return self.graph

    def compute_centrality(self, method: str = "degree") -> Dict[str, float]:
        """
        Returns {gene_symbol: centrality_score} for every node in the graph.
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown centrality method '{method}'. Choose from {self.SUPPORTED_METHODS}"
            )

        if self.graph.number_of_nodes() == 0:
            return {}

        if method == "degree":
            return nx.degree_centrality(self.graph)
        elif method == "betweenness":
            return nx.betweenness_centrality(self.graph, weight="weight")
        elif method == "eigenvector":
            try:
                return nx.eigenvector_centrality(self.graph, weight="weight", max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                # graceful fallback if the graph is too sparse/disconnected
                return nx.degree_centrality(self.graph)
        elif method == "closeness":
            return nx.closeness_centrality(self.graph)

    def get_top_hub(self, method: str = "degree") -> Tuple[str, Dict[str, float]]:
        """
        Returns (hub_gene_symbol, full_centrality_dict).
        The hub gene is the single highest-scoring node — this becomes
        HubResult.hub_gene_symbol, which Modules 2 & 3 consume next.
        """
        scores = self.compute_centrality(method)
        if not scores:
            raise ValueError(
                "Cannot determine hub gene: graph is empty. "
                "Check that STRING DB returned interaction edges."
            )
        hub_gene = max(scores, key=scores.get)
        return hub_gene, scores

    def graph_summary(self) -> Dict:
        """
        Returns a metadata block summarizing the properties of the constructed network,
        extending it to provide an explicit edge list for visualization layers like Plotly.
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return {"num_nodes": 0, "num_edges": 0, "nodes": [], "edges": []}
            
        # Extract the source, target, and confidence weight for every connection
        edges_list = [
            {"source": u, "target": v, "weight": d.get("weight", 1.0)} 
            for u, v, d in self.graph.edges(data=True)
        ]
        
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "nodes": list(self.graph.nodes()),
            "edges": edges_list
        }
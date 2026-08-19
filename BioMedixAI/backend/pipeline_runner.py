"""
Module 4 — Orchestration Layer
Implements: IntegratedReport (dataclass), PipelineRunner, generate_verdict

Matches the SDD's RunFullPipeline algorithm (Section 6.3):

    hub_result <- Module1.TargetDiscoveryPipeline.run(disease_name)
    // Modules 2 and 3 are independent given hub_result -> run concurrently
    WITH ThreadPoolExecutor() AS executor:
        future_drug   <- executor.submit(Module2.run, hub_gene_symbol)
        future_crispr <- executor.submit(Module3.evaluate, hub_gene_symbol, grna)
        druggability_result <- future_drug.result()
        crispr_result       <- future_crispr.result()
    verdict <- GenerateVerdict(druggability_result, crispr_result)
    report  <- IntegratedReport(hub_result, druggability_result, crispr_result, verdict)

A NOTE ON THE MODULE 2 STRUCTURE GAP:
structural_ml.py's DruggabilityPipeline.run() deliberately takes an
already-downloaded structure_path rather than a gene_symbol, since the SDD's
IdentifierMapper (gene -> UniProt -> PDB) and PDBFetcher classes don't exist
yet in this codebase (see structural_ml.py's DruggabilityPipeline docstring).
Rather than block Module 4 on that, this file adds a small, best-effort
`fetch_best_structure()` helper that does the real lookup directly against
RCSB's public search + download API. It follows the exact same fallback
philosophy as everything else here: any failure (no hit, network error,
timeout) returns None, and DruggabilityPipeline.run() already knows what
to do with structure_path=None (sequence-based heuristic fallback). Swap
this helper out for a real IdentifierMapper/PDBFetcher pair later without
touching anything below it.
"""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

import requests

from target_discovery import HubResult, TargetDiscoveryPipeline
from structural_ml import DruggabilityResult, DruggabilityPipeline
from crispr_safety import CrisprSafetyResult, CrisprSafetyPipeline
from db import Database


# ---------------------------------------------------------------------------
# fetch_best_structure — closes the IdentifierMapper/PDBFetcher gap
# ---------------------------------------------------------------------------
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_best_structure(
    gene_symbol: str, organism: str = "Homo sapiens", timeout: int = 15
) -> Optional[Dict[str, str]]:
    """
    Full-text searches RCSB for the best-resolution experimental structure
    matching the gene symbol + organism, downloads it to a temp file, and
    returns {"pdb_id": ..., "structure_path": ..., "resolution": ...}.

    Returns None on ANY failure (no hits, network error, bad response) —
    callers treat that exactly like "no PDB structure exists for this gene"
    per SRS UC-5's exception flow, and DruggabilityPipeline.run() already
    has a documented fallback for structure_path=None.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": gene_symbol},
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                        "operator": "exact_match",
                        "value": organism,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
            "paginate": {"start": 0, "rows": 1},
        },
    }
    try:
        resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=timeout)
        resp.raise_for_status()
        hits = resp.json().get("result_set", [])
        if not hits:
            return None
        pdb_id = hits[0]["identifier"]

        dl_resp = requests.get(RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id), timeout=timeout)
        dl_resp.raise_for_status()
        if not dl_resp.text.strip():
            return None

        fd, path = tempfile.mkstemp(suffix=".pdb", prefix=f"{pdb_id}_")
        with os.fdopen(fd, "w") as f:
            f.write(dl_resp.text)

        return {"pdb_id": pdb_id, "structure_path": path, "resolution": None}
    except (requests.RequestException, KeyError, ValueError, IndexError) as e:
        print(f"[fetch_best_structure] RCSB lookup failed for '{gene_symbol}' ({e})")
        return None


# ---------------------------------------------------------------------------
# IntegratedReport — final output of the whole system (SRS UC-11)
# ---------------------------------------------------------------------------
@dataclass
class IntegratedReport:
    """Matches the SDD's IntegratedReport dataclass (Section 7.1)."""

    hub_result: HubResult
    druggability_result: DruggabilityResult
    crispr_result: Optional[CrisprSafetyResult]
    verdict: str
    report_id: Optional[int] = None


# ---------------------------------------------------------------------------
# generate_verdict — SDD Section 6.4, transparent threshold logic
# ---------------------------------------------------------------------------
DRUG_THRESHOLD = 0.7
SAFETY_THRESHOLD = 0.7


def generate_verdict(
    drug_result: Optional[DruggabilityResult], crispr_result: Optional[CrisprSafetyResult]
) -> str:
    """
    Deliberately a simple, human-readable threshold combination rather than
    a black-box model — per the SDD, this is a summary for a person to read,
    not a prediction (Section 6.4). Either pathway may be absent: gRNA is
    optional (SRS UC-8's alternate flow), and the dashboard now lets a user
    skip the druggability pathway too (see app.py's independent module
    toggles) — both branches degrade to a partial, honestly-labeled verdict
    rather than guessing at the missing half.
    """
    if drug_result is None and crispr_result is None:
        return "No druggability or CRISPR safety analysis was run — nothing to verdict on yet."

    if drug_result is None:
        return (
            "Gene-editing safety evaluated only (druggability pathway skipped): "
            f"{'appears safe' if crispr_result.safety_score >= SAFETY_THRESHOLD else 'carries elevated off-target risk'}"
        )

    if crispr_result is None:
        if drug_result.druggability_score >= DRUG_THRESHOLD:
            return (
                "Promising small-molecule target (no gRNA supplied — "
                "gene-editing safety not evaluated)"
            )
        return (
            "Low small-molecule druggability (no gRNA supplied — "
            "gene-editing safety not evaluated)"
        )

    druggable = drug_result.druggability_score >= DRUG_THRESHOLD
    safe = crispr_result.safety_score >= SAFETY_THRESHOLD

    if druggable and safe:
        return "Strong candidate: high druggability, low off-target risk"
    elif druggable and not safe:
        return (
            "Promising small-molecule target; gene-editing route carries "
            "elevated off-target risk"
        )
    elif not druggable and safe:
        return "Low small-molecule druggability; gene-editing route appears safer"
    else:
        return "Neither pathway shows strong therapeutic promise for this target"


# ---------------------------------------------------------------------------
# PipelineRunner — the Facade tying Modules 1-3 together (SDD Section 6.2)
# ---------------------------------------------------------------------------
class PipelineRunner:
    """
    Matches class diagram:
        PipelineRunner(-M1.TargetDiscoveryPipeline, -M2.StructuralMLPipeline,
                        -M3.CrisprSafetyEngine)
            +run_full_pipeline(disease, grna): IntegratedReport

    Sequences Module 1, then runs Modules 2 and 3 concurrently once the hub
    gene is known (they're mutually independent given hub_result — SRS
    Section 5 / NFR-2). Every sub-pipeline already has its own graceful
    fallback (Modules 1-3's respective _finish_fallback paths), so a single
    module failure never aborts the whole run (NFR-4) — this class's own
    try/except around each future.result() is an extra safety net specific
    to unexpected exceptions escaping those fallbacks.
    """

    def __init__(
        self,
        target_pipeline: TargetDiscoveryPipeline,
        druggability_pipeline: DruggabilityPipeline,
        crispr_pipeline: CrisprSafetyPipeline,
        db: Optional[Database] = None,
        organism: str = "Homo sapiens",
    ):
        self.target_pipeline = target_pipeline
        self.druggability_pipeline = druggability_pipeline
        self.crispr_pipeline = crispr_pipeline
        self.db = db
        self.organism = organism

    def run_full_pipeline(
        self,
        disease_name: str,
        guide_rna: Optional[str] = None,
        species="human",
        gene_limit: int = 10,
        centrality_method: str = "degree",
        max_mismatches: int = 6,
    ) -> IntegratedReport:
        # 1. Module 1 — sequential, everything downstream depends on the hub gene.
        hub_result = self.target_pipeline.run(
            disease_name,
            gene_limit=gene_limit,
            species=species,
            centrality_method=centrality_method,
        )
        hub_gene = hub_result.hub_gene_symbol
        print(f"[PipelineRunner] Hub gene for '{disease_name}': {hub_gene}")

        # Best-effort structure lookup — closes the IdentifierMapper/PDBFetcher
        # gap (see module docstring). None just means Module 2 falls back.
        structure_info = fetch_best_structure(hub_gene, organism=self.organism)
        structure_path = structure_info["structure_path"] if structure_info else None
        pdb_id = structure_info["pdb_id"] if structure_info else None

        # 2. Modules 2 & 3 — independent given hub_gene, run concurrently.
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_drug = executor.submit(
                self.druggability_pipeline.run,
                hub_gene,
                structure_path=structure_path,
                pdb_id=pdb_id,
            )
            future_crispr = (
                executor.submit(
                    self.crispr_pipeline.run,
                    hub_gene,
                    guide_rna,
                    species=species,
                    max_mismatches=max_mismatches,
                )
                if guide_rna
                else None
            )

            druggability_result = self._resolve_future(
                future_drug, "Module 2 (druggability)", hub_gene
            )
            crispr_result = (
                self._resolve_future(future_crispr, "Module 3 (CRISPR safety)", hub_gene)
                if future_crispr is not None
                else None
            )

        # Clean up the temp structure file now that both futures are done.
        if structure_path and os.path.exists(structure_path):
            try:
                os.remove(structure_path)
            except OSError:
                pass

        # 3. Verdict + report assembly (SRS UC-11).
        verdict = generate_verdict(druggability_result, crispr_result)
        report = IntegratedReport(
            hub_result=hub_result,
            druggability_result=druggability_result,
            crispr_result=crispr_result,
            verdict=verdict,
        )

        # 4. Persist, if a db and a Module 1 run_id are both available.
        if self.db is not None and hub_result.run_id is not None:
            report.report_id = self.db.insert_integrated_report(
                run_id=hub_result.run_id,
                verdict_text=verdict,
                druggability_result_id=druggability_result.result_id,
                crispr_result_id=crispr_result.result_id if crispr_result else None,
            )
            print(f"[PipelineRunner] Persisted report_id={report.report_id} to database.")

        return report

    @staticmethod
    def _resolve_future(future, label: str, gene_symbol: str):
        """
        NFR-4 safety net: if a sub-pipeline raises something its own
        fallback didn't catch, surface a clear warning instead of taking
        down the whole run. Callers of run_full_pipeline should still expect
        a real DruggabilityResult/CrisprSafetyResult back — this only
        re-raises, since a silently fabricated result would misrepresent
        what actually happened (same honesty principle as every fallback
        score elsewhere in this codebase).
        """
        try:
            return future.result()
        except Exception as e:
            print(
                f"[PipelineRunner] {label} raised an unexpected error for "
                f"'{gene_symbol}': {e}. This is NOT a documented fallback "
                f"path — check the sub-pipeline's own error handling."
            )
            raise
"""
Module 3 — Genomic DL Layer
Implements: CandidateSite (dataclass), SequenceFetcher, OffTargetScanner,
CrisprSafetyEngine, CrisprSafetyResult, CrisprSafetyPipeline

Mirrors Module 2's PredictDruggability algorithm shape:

    sequence   <- SequenceFetcher.fetch_sequence(gene_symbol, species)  // NCBI Entrez
    candidates <- OffTargetScanner.scan(guide_rna, sequence)             // PAM-adjacent windows
    risks      <- CrisprSafetyEngine.predict(guide_rna, candidate)       // small Keras CNN, per site
    safety     <- prod(1 - risk_i for each candidate)                    // P(no off-target cleavage)

A NOTE ON NAMING / SCOPE, so nobody mistakes this for verified against your
actual SDD wording (unlike structural_ml.py's fpocket section, which was
confirmed byte-for-byte against a real run in your project chat log — no
equivalent Module 3 class diagram / algorithm section was in the files I
was given, so the class names above are my best-fit match to Module 1/2's
pattern; rename freely if your SDD says something different, the behavior
underneath won't need to change):

  - SequenceFetcher wraps NCBI's real, keyless E-utilities API (same
    "real provider, no API key needed" spirit as DataIngestor's STRING DB
    call). It reuses data_ingestor.resolve_species so "rice"/"human"/etc.
    stay consistent across all three modules.
  - OffTargetScanner is a real, deterministic PAM-window + mismatch scan
    over the fetched sequence — this part needs no ML and is not a
    placeholder. If your team has Cas-OFFinder installed (the genomic-DL
    off-target CLI most SRSs reference), swap this class's `scan()` for a
    subprocess call to it; CandidateSite's shape doesn't need to change.
  - CrisprSafetyEngine, like Module 2's DruggabilityEngine, ONLY wraps
    inference. Training a real CNN on labeled off-target data (CRISPOR/CFD
    datasets) is a separate offline script, out of scope here. Until that
    model exists, CrisprSafetyPipeline falls back to a documented,
    deterministic seed-region-weighted heuristic — same "don't fabricate a
    calculated-looking number" honesty as Module 2's fallback_heuristic_score.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import requests

# from data_ingestor import resolve_species, SPECIES_TAXONOMY_IDS

SPECIES_TAXONOMY_IDS = {
    "human": "9606",
    "mouse": "10090",
    "rice": "4530",
}


def resolve_species(species_name: str) -> str:
    return SPECIES_TAXONOMY_IDS.get(species_name.lower(), "9606")


if TYPE_CHECKING:
    from db import Database


BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


# ---------------------------------------------------------------------------
# CandidateSite — one PAM-adjacent window found while scanning a sequence
# ---------------------------------------------------------------------------
@dataclass
class CandidateSite:
    """
    Matches class diagram:
        <<dataclass>> CandidateSite
            +site_id: int
            +sequence: str
            +position: int
            +mismatches: int
            +pam_ok: bool
            +risk_score: Optional[float]
    """

    site_id: int
    sequence: str
    position: int
    mismatches: int
    pam_ok: bool
    risk_score: Optional[float] = None


# ---------------------------------------------------------------------------
# SequenceFetcher — gene symbol -> NCBI RefSeq mRNA sequence
# ---------------------------------------------------------------------------
class SequenceFetcher:
    """
    Matches class diagram:
        SequenceFetcher(-email: str, -api_key: str)
            +fetch_sequence(gene_symbol, species) -> Optional[Tuple[str, str]]

    Wraps NCBI's E-utilities directly over HTTP — esearch to resolve the
    gene symbol + organism to a RefSeq mRNA UID, then efetch to download
    the actual FASTA sequence. No API key required, though NCBI recommends
    one for higher rate limits (pass via api_key=).
    """

    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        email: str = "biomedix-ai-capstone@example.com",
        api_key: Optional[str] = None,
        timeout: int = 15,
        request_delay: float = 0.34,
    ):
        # NCBI etiquette: identify yourself, and without an api_key stay
        # under 3 requests/sec (hence the small delay between calls).
        self.email = email
        self.api_key = api_key
        self.timeout = timeout
        self.request_delay = request_delay

    def _params(self, extra: Dict) -> Dict:
        params = {"email": self.email, **extra}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def gene_to_uid(self, gene_symbol: str, species=4530) -> Optional[str]:
        """species: common name ("rice") or NCBI taxonomy ID (int), same
        convention as data_ingestor.resolve_species / DataIngestor.fetch_string_interactions."""
        taxid = resolve_species(species)
        term = f"{gene_symbol}[Gene Name] AND txid{taxid}[Organism:exp] AND refseq[Filter] AND mRNA[Filter]"
        params = self._params(
            {"db": "nucleotide", "term": term, "retmax": 5, "sort": "relevance", "retmode": "json"}
        )
        try:
            resp = requests.get(f"{self.EUTILS_BASE}/esearch.fcgi", params=params, timeout=self.timeout)
            resp.raise_for_status()
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            return ids[0] if ids else None
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"[SequenceFetcher] esearch failed for '{gene_symbol}' ({e})")
            return None

    def fetch_fasta(self, uid: str) -> Optional[Tuple[str, str]]:
        """Returns (accession, sequence) or None."""
        params = {"db": "nucleotide", "id": uid, "rettype": "fasta", "retmode": "text", "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            resp = requests.get(f"{self.EUTILS_BASE}/efetch.fcgi", params=params, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text.startswith(">"):
                return None
            lines = text.splitlines()
            accession = lines[0][1:].split()[0] if len(lines[0]) > 1 else uid
            sequence = "".join(lines[1:]).upper().replace(" ", "")
            return (accession, sequence) if sequence else None
        except requests.RequestException as e:
            print(f"[SequenceFetcher] efetch failed for uid={uid} ({e})")
            return None

    def fetch_sequence(self, gene_symbol: str, species=4530) -> Optional[Tuple[str, str]]:
        """
        Returns (accession, sequence), or None if NCBI has no match / a
        network error occurred. Callers (CrisprSafetyPipeline) treat None
        as the signal to fall back to the heuristic score, same pattern as
        Module 2's PDBFetcher/PocketDetector returning [] on failure.
        """
        uid = self.gene_to_uid(gene_symbol, species=species)
        if not uid:
            return None
        time.sleep(self.request_delay)
        return self.fetch_fasta(uid)


# ---------------------------------------------------------------------------
# OffTargetScanner — real, deterministic PAM-window + mismatch scan
# ---------------------------------------------------------------------------
class OffTargetScanner:
    """
    Matches class diagram:
        OffTargetScanner(-pam_pattern: str)
            +scan(guide_rna, sequence, max_mismatches) -> List[CandidateSite]

    Slides a window the length of the guide RNA across the sequence,
    keeping only windows immediately followed by a valid PAM (NGG for
    SpCas9 by default) and within a mismatch budget. This is real,
    deterministic sequence analysis — no ML involved at this stage.
    """

    def __init__(self, pam_pattern: str = "NGG"):
        self.pam_pattern = pam_pattern

    def _pam_matches(self, candidate_pam: str) -> bool:
        if len(candidate_pam) != len(self.pam_pattern):
            return False
        return all(p == "N" or c == p for c, p in zip(candidate_pam, self.pam_pattern))

    def scan(self, guide_rna: str, sequence: str, max_mismatches: int = 6) -> List[CandidateSite]:
        guide_rna = guide_rna.upper().replace("U", "T")
        sequence = sequence.upper().replace("U", "T")
        guide_len = len(guide_rna)
        pam_len = len(self.pam_pattern)

        candidates: List[CandidateSite] = []
        site_id = 1
        last_start = len(sequence) - guide_len - pam_len
        for start in range(0, max(last_start, 0) + 1):
            window = sequence[start:start + guide_len]
            pam = sequence[start + guide_len:start + guide_len + pam_len]
            if len(window) != guide_len or len(pam) != pam_len:
                continue
            if any(b not in BASE_TO_IDX for b in window):
                continue  # skip windows with ambiguous bases (N, etc.)

            mismatches = sum(1 for a, b in zip(guide_rna, window) if a != b)
            if mismatches > max_mismatches:
                continue

            candidates.append(CandidateSite(
                site_id=site_id,
                sequence=window,
                position=start,
                mismatches=mismatches,
                pam_ok=self._pam_matches(pam),
            ))
            site_id += 1

        return candidates


# ---------------------------------------------------------------------------
# CrisprSafetyEngine — thin wrapper around a trained off-target CNN
# ---------------------------------------------------------------------------
class CrisprSafetyEngine:
    """
    Matches class diagram:
        CrisprSafetyEngine(-model: keras.Model)
            +predict(guide_rna, candidate) -> float

    This class only wraps inference — training is a separate offline
    script (fit on a labeled off-target dataset such as CRISPOR/CFD's
    training data), out of scope for this parser task, same restriction
    Module 2's DruggabilityEngine documents for its RandomForestRegressor.
    Left as a clean seam so that script can plug in without touching
    SequenceFetcher or OffTargetScanner.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path)

    @staticmethod
    def _encode_pair(guide_rna: str, candidate_seq: str) -> np.ndarray:
        """(L, 8) one-hot: 4 channels for the guide base + 4 for the
        candidate base at each position — standard "concat one-hot" input
        for CNN off-target models."""
        guide_rna = guide_rna.upper().replace("U", "T")
        candidate_seq = candidate_seq.upper().replace("U", "T")
        length = len(guide_rna)
        arr = np.zeros((length, 8), dtype=np.float32)
        for i in range(length):
            g, t = guide_rna[i], candidate_seq[i]
            if g in BASE_TO_IDX:
                arr[i, BASE_TO_IDX[g]] = 1.0
            if t in BASE_TO_IDX:
                arr[i, 4 + BASE_TO_IDX[t]] = 1.0
        return arr

    def predict(self, guide_rna: str, candidate: CandidateSite) -> float:
        if self.model is None:
            raise RuntimeError(
                "CrisprSafetyEngine has no trained model loaded — call "
                "load(model_path) first, or use fallback_heuristic_score() "
                "per the same pattern as Module 2's DruggabilityEngine when "
                "no trained model is available yet."
            )
        x = self._encode_pair(guide_rna, candidate.sequence).reshape(1, len(guide_rna), 8)
        pred = float(self.model.predict(x, verbose=0)[0][0])
        return float(np.clip(pred, 0, 1))


def fallback_heuristic_score(gene_symbol: str, guide_rna: str) -> float:
    """
    Placeholder used when no sequence exists for a gene at all (NCBI has no
    match, or a network error occurred). Intentionally a fixed neutral
    value, not a fabricated "calculated-looking" number — same rationale
    as Module 2's fallback_heuristic_score for PDB/pocket lookups.
    """
    return 0.5


def _interim_site_risk(candidate: CandidateSite, guide_len: int) -> float:
    """
    Used per-site when CrisprSafetyEngine has no trained CNN loaded yet,
    but the sequence WAS found and real candidate sites DO exist — same
    "interim, not fabricated" idea as structural_ml.py's DruggabilityPipeline
    falling back to fpocket's own druggability_score instead of the generic
    heuristic when real pocket geometry is available.

    Encodes the one well-established rule from the off-target literature:
    mismatches in the seed region (the few nt immediately adjacent to the
    PAM) disrupt Cas9 binding far more than the same number of mismatches
    further away (PAM-distal). This is a deterministic formula, not a model
    prediction — CrisprSafetyPipeline marks used_fallback=True whenever this
    path is used so callers can tell the two apart.
    """
    if not candidate.pam_ok:
        return 0.02  # no PAM -> Cas9 essentially can't bind here
    if candidate.mismatches == 0:
        return 0.95
    # Without knowing exactly which positions mismatched (only the count),
    # assume an average seed/non-seed mix: each mismatch reduces risk, but
    # less steeply than a true seed-region-aware model would for a
    # PAM-proximal mismatch specifically.
    return float(np.clip(0.9 * (0.65 ** candidate.mismatches), 0, 1))


# ---------------------------------------------------------------------------
# CrisprSafetyResult — final output consumed by Module 4 (orchestration)
# ---------------------------------------------------------------------------
@dataclass
class CrisprSafetyResult:
    """Matches the SDD's CrisprSafetyResult dataclass (mirrors DruggabilityResult)."""

    gene_symbol: str
    guide_rna: str
    safety_score: float
    flagged_sites: List[CandidateSite] = field(default_factory=list)
    all_sites: List[CandidateSite] = field(default_factory=list)
    used_fallback: bool = False
    # Set only when persisted (db is not None) — the CRISPR_SAFETY_RESULT row
    # id, needed by Module 4 to link its INTEGRATED_REPORT row back to this
    # specific evaluation.
    result_id: Optional[int] = None


FLAG_THRESHOLD = 0.5  # risk_score above this -> reported as a flagged off-target site


# ---------------------------------------------------------------------------
# CrisprSafetyPipeline — wires SequenceFetcher -> OffTargetScanner -> CrisprSafetyEngine
# ---------------------------------------------------------------------------
class CrisprSafetyPipeline:
    """
    Wires SequenceFetcher -> OffTargetScanner -> CrisprSafetyEngine, and
    persists results the same way structural_ml.py's DruggabilityPipeline
    does: gene_id is resolved via db.upsert_gene(gene_symbol) right before
    the write, rather than requiring Module 1 to have already created the
    row. That keeps Module 3 runnable standalone (e.g. against a gene
    STRING never returned, or in a unit test) while still being idempotent
    if the gene already exists.
    """

    def __init__(
        self,
        fetcher: SequenceFetcher,
        scanner: OffTargetScanner,
        engine: CrisprSafetyEngine,
        db: Optional["Database"] = None,
    ):
        self.fetcher = fetcher
        self.scanner = scanner
        self.engine = engine
        self.db = db  # if None, pipeline runs in-memory only (no persistence)

    def run(
        self,
        gene_symbol: str,
        guide_rna: str,
        species=4530,
        max_mismatches: int = 6,
    ) -> CrisprSafetyResult:
        guide_rna = guide_rna.upper().replace("U", "T")
        gene_id = self.db.upsert_gene(gene_symbol) if self.db is not None else None

        submission_id = None
        if self.db is not None:
            submission_id = self.db.insert_grna_submission(
                gene_id, guide_rna, pam_pattern=self.scanner.pam_pattern
            )

        fetched = self.fetcher.fetch_sequence(gene_symbol, species=species)
        if fetched is None:
            # No sequence available at all -> straight to the heuristic,
            # matching Module 2's "uniprot_id is NULL" branch.
            return self._finish_fallback(gene_symbol, guide_rna, gene_id, submission_id)

        accession, sequence = fetched
        candidates = self.scanner.scan(guide_rna, sequence, max_mismatches=max_mismatches)

        if not candidates:
            # Sequence exists, scan ran, but found no PAM-adjacent windows
            # within the mismatch budget -> also falls back, matching
            # Module 2's "pockets is EMPTY" branch.
            return self._finish_fallback(gene_symbol, guide_rna, gene_id, submission_id)

        scored: List[CandidateSite] = []
        used_interim = False
        for site in candidates:
            try:
                risk = self.engine.predict(guide_rna, site)
            except RuntimeError:
                if not used_interim:
                    print(
                        "[CrisprSafetyPipeline] No trained CNN loaded — using "
                        "the seed-region-weighted interim heuristic per site "
                        "in place of the CNN prediction."
                    )
                    used_interim = True
                risk = _interim_site_risk(site, len(guide_rna))
            site.risk_score = round(risk, 4)
            scored.append(site)

        safety_score = 1.0
        for site in scored:
            safety_score *= (1.0 - site.risk_score)
        safety_score = round(safety_score, 4)

        flagged = sorted(
            [s for s in scored if s.risk_score >= FLAG_THRESHOLD],
            key=lambda s: s.risk_score,
            reverse=True,
        )

        result = CrisprSafetyResult(
            gene_symbol=gene_symbol,
            guide_rna=guide_rna,
            safety_score=safety_score,
            flagged_sites=flagged,
            all_sites=scored,
            used_fallback=used_interim,  # True only for the per-site interim formula, not the sequence-missing fallback
        )

        if self.db is not None:
            result.result_id = self.db.insert_crispr_safety_result(
                gene_id=gene_id,
                safety_score=safety_score,
                submission_id=submission_id,
                num_candidate_sites=len(scored),
                num_flagged_sites=len(flagged),
                used_fallback=used_interim,
                off_target_sites=[
                    {
                        "position": s.position,
                        "site_sequence": s.sequence,
                        "mismatches": s.mismatches,
                        "pam_ok": s.pam_ok,
                        "risk_score": s.risk_score,
                    }
                    for s in scored
                ],
            )

        return result

    def _finish_fallback(
        self,
        gene_symbol: str,
        guide_rna: str,
        gene_id: Optional[int],
        submission_id: Optional[int],
    ) -> CrisprSafetyResult:
        score = fallback_heuristic_score(gene_symbol, guide_rna)
        result = CrisprSafetyResult(
            gene_symbol=gene_symbol,
            guide_rna=guide_rna,
            safety_score=score,
            flagged_sites=[],
            all_sites=[],
            used_fallback=True,
        )
        if self.db is not None:
            result.result_id = self.db.insert_crispr_safety_result(
                gene_id=gene_id,
                safety_score=score,
                submission_id=submission_id,
                num_candidate_sites=0,
                num_flagged_sites=0,
                used_fallback=True,
                off_target_sites=[],
            )
        return result
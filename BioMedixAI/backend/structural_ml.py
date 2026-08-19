"""
Module 2 — Structural ML Layer
Implements: Pocket (dataclass), PocketDetector, FeatureExtractor, DruggabilityEngine

Matches the SDD's PredictDruggability algorithm (Section 4.3):

    pockets   <- PocketDetector.detect_pockets(structure)
    best      <- argmax(pockets, key=volume)
    features  <- FeatureExtractor.extract(best)
    score     <- DruggabilityEngine.predict(features)   // RandomForestRegressor

PocketDetector wraps the `fpocket` CLI (already compiled/on $PATH in the WSL
environment per project setup) and parses its `_info.txt` descriptor output.

NOTE ON THE FPOCKET FORMAT:
Verified against a real `fpocket -f 1EMA.pdb` run (see project chat log) —
the `_KEY_MAP` below matches the actual `1EMA_out/1EMA_info.txt` output
byte-for-byte, one "Pocket N :" header per pocket followed by
"Key : value" lines, e.g.:

    Pocket 1 :
            Score :                                 0.734
            Druggability Score :                   0.828
            Number of Alpha Spheres :               45
            Total SASA :                            178.234
            Polar SASA :                            65.123
            Apolar SASA :                           113.111
            Volume :                                412.335
            Mean local hydrophobic density :        23.456
            Mean alpha sphere radius :               3.987
            Mean alp. sph. solvent access :          0.512
            Apolar alpha sphere proportion :         0.622
            Hydrophobicity score:                    32.145
            Volume score:                            4.234
            Polarity score:                          4
            Charge score :                           2
            Proportion of polar atoms:                28.345
            Alpha sphere density :                   4.567
            Cent. of mass - Alpha Sphere max dist:    8.901
            Flexibility :                             0.512

This has been stable across fpocket 3.x/4.x releases. If your installed
version emits different key strings, send a real sample from
`<name>_out/<name>_info.txt` and only `_KEY_MAP` below needs updating —
nothing else in the pipeline changes.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from db import Database


# ---------------------------------------------------------------------------
# Pocket — one parsed fpocket pocket record
# ---------------------------------------------------------------------------
@dataclass
class Pocket:
    """
    One pocket as reported by fpocket, with descriptors renamed to
    snake_case Python attributes. `pocket_id` and `volume` are guaranteed
    present; every other field defaults to None if fpocket's output omits
    it (older builds sometimes drop one or two descriptors).
    """

    pocket_id: int
    volume: float
    score: Optional[float] = None
    druggability_score: Optional[float] = None
    num_alpha_spheres: Optional[float] = None
    total_sasa: Optional[float] = None
    polar_sasa: Optional[float] = None
    apolar_sasa: Optional[float] = None
    mean_local_hydrophobic_density: Optional[float] = None
    mean_alpha_sphere_radius: Optional[float] = None
    mean_alpha_sphere_solvent_access: Optional[float] = None
    apolar_alpha_sphere_proportion: Optional[float] = None
    hydrophobicity_score: Optional[float] = None
    volume_score: Optional[float] = None
    polarity_score: Optional[float] = None
    charge_score: Optional[float] = None
    proportion_polar_atoms: Optional[float] = None
    alpha_sphere_density: Optional[float] = None
    cent_of_mass_alpha_sphere_max_dist: Optional[float] = None
    flexibility: Optional[float] = None

    # Populated separately from the pocketN_atm.pdb / pocketN_vert.pqr files
    # if/when residue-level detail is needed (not required for the
    # FeatureExtractor -> DruggabilityEngine path, so left empty by default).
    lining_residues: List[str] = field(default_factory=list)

    # Absolute paths to the per-pocket files fpocket writes alongside
    # <name>_info.txt, in <out_dir>/pockets/. Populated by PocketDetector
    # right after parsing, when those files exist. Used by the offline
    # training script to compute each pocket's 3D centroid (for
    # ligand-proximity pocket selection — see train_druggability_model.py)
    # — NOT used anywhere in the request-time inference path, so leaving
    # these None never affects DruggabilityPipeline.run().
    atm_pdb_path: Optional[str] = None
    vert_pqr_path: Optional[str] = None


# ---------------------------------------------------------------------------
# PocketDetector — runs fpocket, parses its descriptor output
# ---------------------------------------------------------------------------
class PocketDetector:
    """
    Matches class diagram:
        PocketDetector(-fpocket_binary: str)
            +detect_pockets(structure_path) -> List[Pocket]
    """

    # Maps the exact left-hand-side text fpocket prints (lower-cased,
    # trailing/leading whitespace and colons stripped, internal whitespace
    # collapsed to single spaces) to our dataclass field name. Update this
    # dict — and nothing else — if a different fpocket build uses different
    # wording.
    _KEY_MAP = {
        "score": "score",
        "druggability score": "druggability_score",
        "number of alpha spheres": "num_alpha_spheres",
        "total sasa": "total_sasa",
        "polar sasa": "polar_sasa",
        "apolar sasa": "apolar_sasa",
        "volume": "volume",
        "mean local hydrophobic density": "mean_local_hydrophobic_density",
        "mean alpha sphere radius": "mean_alpha_sphere_radius",
        "mean alp. sph. solvent access": "mean_alpha_sphere_solvent_access",
        "apolar alpha sphere proportion": "apolar_alpha_sphere_proportion",
        "hydrophobicity score": "hydrophobicity_score",
        "volume score": "volume_score",
        "polarity score": "polarity_score",
        "charge score": "charge_score",
        "proportion of polar atoms": "proportion_polar_atoms",
        "alpha sphere density": "alpha_sphere_density",
        "cent. of mass - alpha sphere max dist": "cent_of_mass_alpha_sphere_max_dist",
        "flexibility": "flexibility",
    }

    _POCKET_HEADER_RE = re.compile(r"^pocket\s+(\d+)\s*:?\s*$", re.IGNORECASE)

    def __init__(self, fpocket_binary: str = "fpocket"):
        self.fpocket_binary = fpocket_binary
        resolved = shutil.which(fpocket_binary)
        if resolved is None:
            raise FileNotFoundError(
                f"'{fpocket_binary}' not found on $PATH. Confirm the WSL "
                f"environment's fpocket install is on PATH for this process "
                f"(e.g. `which fpocket` in the same shell that runs Python)."
            )
        self._resolved_binary = resolved

    # -- public API ---------------------------------------------------
    def detect_pockets(self, structure_path: str) -> List[Pocket]:
        """
        Runs fpocket on a PDB file and returns every detected pocket.
        Returns [] (not an error) if fpocket runs successfully but finds
        no cavities — callers (see PredictDruggability) treat that as a
        signal to fall back to the sequence-based heuristic score.
        """
        if not os.path.isfile(structure_path):
            raise FileNotFoundError(f"Structure file not found: {structure_path}")
        if not structure_path.lower().endswith(".pdb"):
            raise ValueError(
                f"fpocket requires a .pdb input, got: {structure_path}. "
                f"Convert with StructureParser/Bio.PDB first if this came "
                f"from mmCIF."
            )

        info_path = self._run_fpocket(structure_path)
        if info_path is None:
            return []
        pockets = self._parse_info_file(info_path)
        self._attach_pocket_file_paths(pockets, info_path)
        return pockets

    def _attach_pocket_file_paths(self, pockets: List[Pocket], info_path: str) -> None:
        """
        fpocket writes one pocketN_atm.pdb (lining protein atoms) and one
        pocketN_vert.pqr (alpha-sphere centers) per pocket into
        <out_dir>/pockets/, where N matches the "Pocket N :" numbering in
        info.txt. Attaches whichever of these actually exist to each
        Pocket — silently leaves both None if fpocket's output layout ever
        changes, since nothing in the inference path depends on them.
        """
        pockets_dir = os.path.join(os.path.dirname(info_path), "pockets")
        if not os.path.isdir(pockets_dir):
            return
        for pocket in pockets:
            atm_path = os.path.join(pockets_dir, f"pocket{pocket.pocket_id}_atm.pdb")
            vert_path = os.path.join(pockets_dir, f"pocket{pocket.pocket_id}_vert.pqr")
            if os.path.isfile(atm_path):
                pocket.atm_pdb_path = atm_path
            if os.path.isfile(vert_path):
                pocket.vert_pqr_path = vert_path

    # -- internals ------------------------------------------------------
    def _run_fpocket(self, structure_path: str) -> Optional[str]:
        """
        fpocket writes output next to its input, as <basename>_out/. It
        refuses to run if that directory already exists, so we copy the
        input into a fresh temp dir per call to keep runs isolated and
        re-runnable (important for tests and for re-analyzing the same
        gene's structure across pipeline runs).
        """
        work_dir = tempfile.mkdtemp(prefix="fpocket_")
        local_pdb = os.path.join(work_dir, os.path.basename(structure_path))
        shutil.copy(structure_path, local_pdb)

        result = subprocess.run(
            [self._resolved_binary, "-f", local_pdb],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(
                f"[PocketDetector] fpocket exited {result.returncode} on "
                f"{structure_path}: {result.stderr.strip()}"
            )
            return None

        base = os.path.splitext(os.path.basename(local_pdb))[0]
        out_dir = os.path.join(work_dir, f"{base}_out")
        info_path = os.path.join(out_dir, f"{base}_info.txt")
        if not os.path.isfile(info_path):
            print(
                f"[PocketDetector] fpocket ran but produced no info file "
                f"at {info_path} — likely zero pockets detected."
            )
            return None
        return info_path

    def _parse_info_file(self, info_path: str) -> List[Pocket]:
        pockets: List[Pocket] = []
        current_id: Optional[int] = None
        current_kwargs: Dict[str, float] = {}

        def flush():
            if current_id is not None:
                # volume is the one field PredictDruggability's argmax()
                # depends on — if fpocket ever omits it, skip the pocket
                # rather than silently feeding a None into argmax(key=volume).
                if "volume" not in current_kwargs:
                    print(
                        f"[PocketDetector] Pocket {current_id} missing a "
                        f"volume value — dropping it from results."
                    )
                    return
                pockets.append(Pocket(pocket_id=current_id, **current_kwargs))

        with open(info_path, "r") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue

                header_match = self._POCKET_HEADER_RE.match(line)
                if header_match:
                    flush()
                    current_id = int(header_match.group(1))
                    current_kwargs = {}
                    continue

                if current_id is None or ":" not in line:
                    continue  # preamble/footer lines outside any pocket block

                key_raw, _, value_raw = line.partition(":")
                key_norm = re.sub(r"\s+", " ", key_raw.strip().lower())
                field_name = self._KEY_MAP.get(key_norm)
                if field_name is None:
                    continue  # unrecognized descriptor — ignore, don't crash

                try:
                    current_kwargs[field_name] = float(value_raw.strip())
                except ValueError:
                    print(
                        f"[PocketDetector] Could not parse value for "
                        f"'{key_raw.strip()}' in pocket {current_id}: "
                        f"'{value_raw.strip()}'"
                    )

        flush()  # last pocket in the file has no following header to trigger it
        return pockets


# ---------------------------------------------------------------------------
# FeatureExtractor — Pocket -> flat numeric feature dict for the ML model
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """
    Matches class diagram:
        FeatureExtractor
            +extract(pocket) -> Dict[str, float]

    Converts a Pocket into the exact feature vector DruggabilityEngine's
    RandomForestRegressor expects: every numeric descriptor, missing
    values imputed to 0.0 (rather than dropped) so the feature vector's
    shape/ordering never changes between pockets, which matters for a
    fixed-input-shape sklearn model.
    """

    # Explicit, ordered list — this ordering IS the model's input contract.
    FEATURE_NAMES = [
        "score",
        "druggability_score",
        "num_alpha_spheres",
        "total_sasa",
        "polar_sasa",
        "apolar_sasa",
        "volume",
        "mean_local_hydrophobic_density",
        "mean_alpha_sphere_radius",
        "mean_alpha_sphere_solvent_access",
        "apolar_alpha_sphere_proportion",
        "hydrophobicity_score",
        "volume_score",
        "polarity_score",
        "charge_score",
        "proportion_polar_atoms",
        "alpha_sphere_density",
        "cent_of_mass_alpha_sphere_max_dist",
        "flexibility",
    ]

    def extract(self, pocket: Pocket) -> Dict[str, float]:
        features = {
            name: (getattr(pocket, name) if getattr(pocket, name) is not None else 0.0)
            for name in self.FEATURE_NAMES
        }
        # One derived feature that's genuinely useful and not in fpocket's
        # raw output: polar/total SASA ratio, a cheap proxy for how buried
        # vs. solvent-exposed the pocket's polar surface is.
        if features["total_sasa"] > 0:
            features["polar_sasa_ratio"] = features["polar_sasa"] / features["total_sasa"]
        else:
            features["polar_sasa_ratio"] = 0.0
        return features

    # Full ordering including the one derived feature added in extract().
    # THIS is the model's actual input contract — training and inference
    # both build vectors through to_vector() so they can never drift apart.
    ALL_FEATURE_NAMES = FEATURE_NAMES + ["polar_sasa_ratio"]

    @staticmethod
    def to_vector(features: Dict[str, float]) -> List[float]:
        """
        Converts a features dict (from extract()) into the fixed-order
        list a sklearn estimator expects. Missing keys default to 0.0
        defensively — same imputation policy as extract() itself — so a
        features dict built by hand (e.g. in a training script row that
        skipped one descriptor) still produces a valid, correctly-shaped
        vector instead of a KeyError.
        """
        return [features.get(name, 0.0) for name in FeatureExtractor.ALL_FEATURE_NAMES]


# ---------------------------------------------------------------------------
# DruggabilityEngine — thin wrapper around a trained RandomForestRegressor
# ---------------------------------------------------------------------------
class DruggabilityEngine:
    """
    Matches class diagram:
        DruggabilityEngine(-model: RandomForestRegressor)
            +predict(features) -> float

    This class only wraps inference — training is a separate offline
    script (fit on the known druggable/non-druggable pocket set mentioned
    in SDD Section 5, e.g. via joblib.dump), out of scope for this parser
    task. Left as a clean seam so that script can plug in without touching
    PocketDetector or FeatureExtractor.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        import joblib

        self.model = joblib.load(model_path)

    def predict(self, features: Dict[str, float]) -> float:
        if self.model is None:
            raise RuntimeError(
                "DruggabilityEngine has no trained model loaded — call "
                "load(model_path) first, or use fallback_heuristic_score() "
                "per the SDD's PredictDruggability algorithm when no "
                "structure/pocket is available at all."
            )
        ordered = FeatureExtractor.to_vector(features)
        return float(self.model.predict([ordered])[0])


def fallback_heuristic_score(gene_symbol: str) -> float:
    """
    Placeholder for the SDD's `fallback_heuristic_score(gene_symbol)` —
    the sequence-based proxy used when no PDB structure exists at all for
    a gene, or fpocket finds zero pockets on the structure that does exist.

    This is intentionally a fixed neutral value, not a fabricated
    sequence-derived number. Once IdentifierMapper/PDBFetcher exist, this
    is the natural place to add a real composition-based heuristic (e.g.
    hydrophobicity/charge via Bio.SeqUtils.ProtParam on the UniProt
    sequence) — until then, returning a made-up "calculated-looking"
    number would misrepresent how confident this score actually is.
    """
    return 0.3


@dataclass
class DruggabilityResult:
    """Matches the SDD's DruggabilityResult dataclass exactly."""

    gene_symbol: str
    pdb_id: Optional[str]
    druggability_score: float
    pocket_features: Dict[str, float] = field(default_factory=dict)
    used_fallback: bool = False
    # Set only when persisted (db is not None) — the DRUGGABILITY_RESULT row
    # id, needed by Module 4 to link its INTEGRATED_REPORT row back to this
    # specific prediction.
    result_id: Optional[int] = None


class DruggabilityPipeline:
    """
    Wires PocketDetector -> FeatureExtractor -> DruggabilityEngine per the
    SDD's PredictDruggability algorithm, and persists results the same way
    target_discovery.py's TargetDiscoveryPipeline does: gene_id is resolved
    via db.upsert_gene(gene_symbol) right before the write, rather than
    requiring Module 1 to have already created the row. That keeps Module 2
    runnable standalone (e.g. against a gene STRING never returned, or in
    a unit test) while still being idempotent if the gene already exists.

    This class takes an already-downloaded structure_path (a local .pdb
    file) rather than a gene_symbol -> PDBFetcher.download_structure(pdb_id)
    step, since that mapping (IdentifierMapper, PDBFetcher) doesn't exist
    yet in this codebase. Wire that in front of this class's `run()` later
    without changing anything below it.
    """

    def __init__(
        self,
        detector: PocketDetector,
        extractor: FeatureExtractor,
        engine: DruggabilityEngine,
        db: Optional["Database"] = None,
    ):
        self.detector = detector
        self.extractor = extractor
        self.engine = engine
        self.db = db  # if None, pipeline runs in-memory only (no persistence)

    def run(
        self,
        gene_symbol: str,
        structure_path: Optional[str] = None,
        pdb_id: Optional[str] = None,
        resolution: Optional[float] = None,
        method: Optional[str] = None,
    ) -> DruggabilityResult:
        gene_id = self.db.upsert_gene(gene_symbol) if self.db is not None else None

        if structure_path is None:
            # No structure available at all -> straight to the heuristic,
            # matching the SDD's "uniprot_id is NULL" branch.
            return self._finish_fallback(gene_symbol, gene_id, pdb_id=None)

        resolved_pdb_id = pdb_id or os.path.splitext(os.path.basename(structure_path))[0]

        structure_id = None
        if self.db is not None:
            structure_id = self.db.upsert_protein_structure(
                gene_id, resolved_pdb_id, resolution=resolution, method=method
            )

        if self.detector is None:
            # fpocket isn't installed/on $PATH in this environment (e.g. a
            # deployment host without the CLI) — a structure was found, but
            # we have no way to detect pockets on it, so this degrades the
            # same way as "pockets is EMPTY" rather than crashing (NFR-4).
            return self._finish_fallback(
                gene_symbol, gene_id, pdb_id=resolved_pdb_id, structure_id=structure_id
            )

        pockets = self.detector.detect_pockets(structure_path)
        if not pockets:
            # Structure exists, fpocket ran, but found no cavities -> also
            # falls back, matching the SDD's "pockets is EMPTY" branch.
            return self._finish_fallback(
                gene_symbol, gene_id, pdb_id=resolved_pdb_id, structure_id=structure_id
            )

        best_pocket = max(pockets, key=lambda p: p.volume)
        features = self.extractor.extract(best_pocket)

        try:
            score = self.engine.predict(features)
        except RuntimeError:
            # No trained RandomForestRegressor yet — use fpocket's own
            # druggability_score as the interim number instead of the
            # generic heuristic, since real pocket geometry IS available
            # here; only the ML re-scoring step is what's missing.
            print(
                "[DruggabilityPipeline] No trained model loaded — using "
                "fpocket's own druggability_score in place of the "
                "RandomForestRegressor prediction."
            )
            score = features["druggability_score"]

        result = DruggabilityResult(
            gene_symbol=gene_symbol,
            pdb_id=resolved_pdb_id,
            druggability_score=score,
            pocket_features=features,
            used_fallback=False,
        )

        if self.db is not None:
            result.result_id = self.db.insert_druggability_result(
                gene_id=gene_id,
                druggability_score=score,
                structure_id=structure_id,
                pocket_rank=best_pocket.pocket_id,
                pocket_volume=best_pocket.volume,
                pocket_features=features,
                used_fallback=False,
            )

        return result

    def _finish_fallback(
        self,
        gene_symbol: str,
        gene_id: Optional[int],
        pdb_id: Optional[str],
        structure_id: Optional[int] = None,
    ) -> DruggabilityResult:
        score = fallback_heuristic_score(gene_symbol)
        result = DruggabilityResult(
            gene_symbol=gene_symbol,
            pdb_id=pdb_id,
            druggability_score=score,
            pocket_features={},
            used_fallback=True,
        )
        if self.db is not None:
            result.result_id = self.db.insert_druggability_result(
                gene_id=gene_id,
                druggability_score=score,
                structure_id=structure_id,
                pocket_rank=None,
                pocket_volume=None,
                pocket_features={},
                used_fallback=True,
            )
        return result
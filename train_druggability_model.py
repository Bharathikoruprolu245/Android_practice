"""
Module 2 — offline training script for DruggabilityEngine's model.

This is deliberately NOT part of the request-time pipeline (structural_ml.py
/ pipeline_runner.py). You run this once (or occasionally, as your labeled
set grows) to produce a .joblib file, then point DruggabilityEngine.load()
at it. Matches the SDD's note that model training is out of scope for the
inference-time classes.

------------------------------------------------------------------------
WHERE TO GET REAL LABELS (read this before running with --labels-csv)
------------------------------------------------------------------------
The standard academic benchmark for exactly this task is the NRDLD set:

    Krasowski, A., Muthas, D., Sarkar, A., Schmitt, S., & Sotriffer, C.
    (2011). DrugPred: A Structure-Based Approach To Predict Protein
    Druggability Developed Using an Extensive Nonredundant Data Set.
    J. Chem. Inf. Model., 51(11), 2829-2842.

~115 non-redundant structures, ~71 labeled "druggable" and ~44 "less
druggable" — the same benchmark fpocket's own Druggability Score and
follow-up tools (PockDrug, DrugPred, TRAPP) were trained/validated on.
The per-PDB-ID label table is in that paper's Supporting Information; I'm
not hard-coding it here since I can't currently verify the exact table
contents accurately enough to trust it for something you're training a
model on. Pull it from the SI PDF, or from a later paper that republishes
it (e.g. the PockDrug-Server SI), and reshape it into the CSV format below.

For a smaller MVP while you build out the rest of the pipeline, you can
hand-pick ~10-20 structures yourself using well-documented druggable
targets (kinases, proteases, nuclear hormone receptors — anything with an
approved drug bound in the PDB entry) vs. known-difficult flat/shallow
sites (protein-protein interaction interfaces are the classic example in
this literature), and label them from domain knowledge + literature you
cite in your own methods section. Either way, this script doesn't care
where the CSV came from.

------------------------------------------------------------------------
CSV FORMAT (--labels-csv)
------------------------------------------------------------------------
    pdb_id,label
    1HVR,druggable
    1M47,less_druggable
    ...

`label` accepts: druggable/less_druggable/non_druggable (case-insensitive),
OR a numeric value in [0, 1] directly (for a continuous consensus score
instead of a binary call). Binary labels are converted to 1.0 / 0.0 and
regressed exactly the way fpocket's own Druggability Score was originally
trained (a RandomForest regressed against binary labels behaves like a
soft probability — the fraction of trees voting "druggable" — rather than
a hard classifier), which is why DruggabilityEngine.predict() returns a
float instead of True/False.

------------------------------------------------------------------------
STRUCTURE FILES (--structures-dir)
------------------------------------------------------------------------
Expects one <pdb_id>.pdb per row already downloaded into this directory
(e.g. via `wget https://files.rcsb.org/download/<PDB_ID>.pdb`, same as the
1EMA test run). Not automated here since this script has no network
access requirement beyond what you already need for fpocket itself.

Usage:
    python train_druggability_model.py \\
        --labels-csv nrdld_labels.csv \\
        --structures-dir ./structures \\
        --model-out druggability_model.joblib
"""

import argparse
import csv
import os
import sys
from collections import Counter
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import matthews_corrcoef, mean_absolute_error, r2_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.inspection import permutation_importance

from structural_ml import FeatureExtractor, Pocket, PocketDetector

# Common crystallization additives, cryoprotectants, and ions that show up
# as HETATM records but are NOT the biologically relevant bound ligand a
# druggability label refers to. Excluded when identifying "the ligand" for
# proximity matching. Not exhaustive — if a structure's true ligand isn't
# recognized because of a residue name missing here, that row falls back
# to volume-argmax (see select_training_pocket) rather than silently
# picking a buffer molecule as if it were the real binding site.
_NON_LIGAND_HETATM = {
    "HOH", "WAT", "DOD",  # water
    "SO4", "PO4", "GOL", "EDO", "ACT", "DMS", "TRS", "PEG", "MPD", "IPA",
    "MES", "HEPES", "CIT", "FMT", "ACY", "BME", "IOD", "PG4", "1PE", "BOG",
    "NA", "CL", "MG", "ZN", "CA", "K", "MN", "FE", "CO", "NI", "CU", "CD",
    "HG", "PB", "BR", "LI", "CS", "BA", "SR", "PT", "AU", "AG", "NH4",
    "UNK", "UNX",
}


def parse_label(raw: str) -> Optional[float]:
    """Converts a CSV label cell to a float in [0, 1], or None if unparseable."""
    text = raw.strip().lower()
    if text in ("druggable", "1", "true", "yes"):
        return 1.0
    if text in ("less_druggable", "non_druggable", "undruggable", "0", "false", "no"):
        return 0.0
    try:
        value = float(text)
    except ValueError:
        return None
    if not (0.0 <= value <= 1.0):
        print(f"[train] Warning: label '{raw}' outside [0,1] — clamping.")
        value = max(0.0, min(1.0, value))
    return value


def mcc_scorer(estimator, X, y) -> float:
    """
    Custom scorer matching the (estimator, X, y) -> float signature that
    GridSearchCV / permutation_importance expect. Needed because our model
    is a REGRESSOR (continuous output) but the metric we actually care
    about, MCC, is a binary-classification metric — this thresholds both
    sides at 0.5 before scoring, consistently with cross_validate() and
    evaluate_on_original_split() elsewhere in this file.
    """
    preds = estimator.predict(X)
    return matthews_corrcoef(np.asarray(y) > 0.5, preds > 0.5)


# Feature groups worth testing in isolation, based on real collinearity in
# fpocket's raw output (see FeatureExtractor.FEATURE_NAMES):
#   - "fpocket_own_scores": fpocket already computes its OWN druggability
#     opinion. Including it as an input feature risks our model just
#     learning to imitate fpocket rather than adding independent signal.
#   - "hydrophobicity_cluster": hydrophobicity is measured five different,
#     highly-correlated ways in fpocket's output. Impurity-based
#     feature_importances_ tends to arbitrarily concentrate credit on one
#     representative of a correlated cluster rather than reflecting the
#     cluster's TRUE combined contribution — this ablation tests the whole
#     cluster's contribution directly instead of trusting single-feature
#     importance under collinearity.
ABLATION_GROUPS = {
    "fpocket_own_scores": ["score", "druggability_score"],
    "hydrophobicity_cluster": [
        "hydrophobicity_score",
        "apolar_sasa",
        "apolar_alpha_sphere_proportion",
        "mean_local_hydrophobic_density",
    ],
}


def drop_features(
    X: np.ndarray, feature_names: List[str], drop_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Returns (X_with_columns_removed, remaining_feature_names)."""
    keep_idx = [i for i, name in enumerate(feature_names) if name not in drop_names]
    kept_names = [feature_names[i] for i in keep_idx]
    return X[:, keep_idx], kept_names


def tune_hyperparameters(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
) -> dict:
    """
    Leak-free hyperparameter search: GridSearchCV's internal cross-validation
    is built from StratifiedKFold applied ONLY to X_train/y_train (the
    original paper's 't' rows). It never sees X_val/y_val (the 'v' rows) —
    those stay untouched until evaluate_on_original_split() does the single,
    honest, held-out evaluation AFTER the best hyperparameters are already
    locked in. This is what makes the reported validation MCC still mean
    what it claims to mean.

    Grid kept intentionally small: NRDLD's 't' split is ~90 structures, so
    an aggressive grid searched via CV on that few rows would itself start
    overfitting the hyperparameter choice to CV noise.
    """
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, None],
    }
    base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    # StratifiedKFold needs DISCRETE labels to stratify on, but y_train is
    # the continuous regression target GridSearchCV will actually fit
    # against. Materialize the fold (train_idx, test_idx) pairs up front
    # from the binarized labels, then hand GridSearchCV that fixed list —
    # this sidesteps both the stratification-needs-discrete-y problem AND
    # the fact that a bare generator would be exhausted after the first
    # hyperparameter combination (GridSearchCV re-walks the CV splits once
    # per candidate).
    y_train_binary = (y_train > 0.5).astype(int)
    cv_splits = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state).split(
            X_train, y_train_binary
        )
    )

    search = GridSearchCV(
        base_model,
        param_grid,
        scoring=mcc_scorer,
        cv=cv_splits,
        n_jobs=-1,
        refit=False,  # we refit ourselves in evaluate_on_original_split, on X_train only
    )
    search.fit(X_train, y_train)

    print(
        f"\n[train] Hyperparameter search complete "
        f"(best inner-CV MCC on training rows only: {search.best_score_:.4f}):"
    )
    for k, v in search.best_params_.items():
        print(f"    {k:<20} {v}")

    return search.best_params_


# ---------------------------------------------------------------------------
# Ligand-proximity pocket selection (TRAINING DATA ONLY — see module docstring
# at the bottom of this file for why this can't apply at inference time)
# ---------------------------------------------------------------------------
def find_ligand_centroid(pdb_path: str) -> Optional[np.ndarray]:
    """
    Parses a holo PDB file and returns the centroid (x, y, z) of the bound
    small-molecule ligand's heavy atoms, or None if no plausible ligand is
    found (apo structure, or only crystallization additives present).

    Heuristic: among HETATM residues NOT in _NON_LIGAND_HETATM, picks the
    one with the most atoms (the real ligand is almost always larger than
    stray ions/buffer components that slip past the exclude list). This is
    an approximation, not a curated per-structure ligand annotation — for
    NRDLD specifically, the label already implies "this structure has one
    known relevant binding site with a ligand in it", so picking the
    largest non-excluded HETATM residue is a reasonable proxy rather than
    something we're inventing from nothing.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("training_structure", pdb_path)
    except Exception as e:
        print(f"[train] Bio.PDB failed to parse {pdb_path}: {e}")
        return None

    candidates = []  # (atom_count, coords_array)
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                het_flag = residue.get_id()[0]
                is_hetatm = het_flag != " "
                if not is_hetatm or resname in _NON_LIGAND_HETATM:
                    continue
                coords = np.array([atom.get_coord() for atom in residue])
                if len(coords) > 0:
                    candidates.append((len(coords), coords))
        break  # first model only (NMR ensembles: model 1 is representative)

    if not candidates:
        return None

    _, best_coords = max(candidates, key=lambda pair: pair[0])
    return best_coords.mean(axis=0)


def _pocket_centroid(pocket: Pocket) -> Optional[np.ndarray]:
    """
    Computes a pocket's 3D centroid from whichever fpocket output file is
    available: prefers pocketN_vert.pqr (alpha-sphere centers — a tighter,
    more precise description of the cavity itself) and falls back to
    pocketN_atm.pdb (the lining protein atoms) if the .pqr isn't present.
    Returns None if neither file was captured by PocketDetector (e.g. an
    older fpocket build with a different output layout).
    """
    if pocket.vert_pqr_path and os.path.isfile(pocket.vert_pqr_path):
        coords = []
        with open(pocket.vert_pqr_path) as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                parts = line.split()
                # PQR whitespace-delimited columns: ... x y z charge radius
                # (fpocket's vert.pqr is space-delimited, not fixed-column,
                # so simple split() is correct here — unlike standard PDB.)
                try:
                    x, y, z = float(parts[-5]), float(parts[-4]), float(parts[-3])
                    coords.append((x, y, z))
                except (ValueError, IndexError):
                    continue
        if coords:
            return np.array(coords).mean(axis=0)

    if pocket.atm_pdb_path and os.path.isfile(pocket.atm_pdb_path):
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("pocket_atm", pocket.atm_pdb_path)
            coords = np.array(
                [atom.get_coord() for atom in structure.get_atoms()]
            )
            if len(coords) > 0:
                return coords.mean(axis=0)
        except Exception:
            pass

    return None


def select_training_pocket(
    pockets: List[Pocket], ligand_centroid: Optional[np.ndarray]
) -> Tuple[Pocket, str]:
    """
    Returns (selected_pocket, method) where method is one of:
        "ligand_proximity" - matched to the real bound ligand's location
        "volume_fallback"  - no ligand centroid available, or no pocket
                              centroids could be computed; fell back to
                              the original argmax(volume) behavior

    THIS FUNCTION IS TRAINING-ONLY. At real inference time (a new gene's
    structure via DruggabilityPipeline.run()), there IS no known ligand —
    predicting druggability for a site with no ligand yet is the entire
    point. DruggabilityPipeline.run() correctly keeps using
    max(pockets, key=lambda p: p.volume) and must not be changed to call
    this function.
    """
    if ligand_centroid is not None:
        distances = []
        for pocket in pockets:
            centroid = _pocket_centroid(pocket)
            if centroid is not None:
                distances.append((np.linalg.norm(centroid - ligand_centroid), pocket))
        if distances:
            _, best = min(distances, key=lambda pair: pair[0])
            return best, "ligand_proximity"

    return max(pockets, key=lambda p: p.volume), "volume_fallback"


def build_dataset(
    labels_csv: str, structures_dir: str, detector: PocketDetector, extractor: FeatureExtractor
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Optional[str]]]:
    """
    Reads the labels CSV, runs fpocket on each structure via PocketDetector,
    selects the training pocket via ligand-proximity matching (falling back
    to volume-argmax only when no ligand/pocket-centroid data is available),
    and returns (X, y, pdb_ids, splits) — skipping (with a printed reason)
    any row that can't be turned into a valid training example.

    `splits` carries through the CSV's own 'nrdld_split' column if present
    (e.g. Krasowski et al.'s original 't'/'v' train/validation partition),
    or None per row if the CSV doesn't have that column — see
    evaluate_on_original_split() for why this matters for literature
    comparability, separately from the random-split CV below.

    Prints a summary of how many rows used each pocket-selection method,
    since a high volume_fallback rate would mean the fix is only partially
    engaging and the resulting metrics should be read with that in mind.
    """
    X, y, ids, splits = [], [], [], []
    method_counts = Counter()

    with open(labels_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "pdb_id" not in reader.fieldnames:
            raise ValueError(
                f"'{labels_csv}' must have a header row with at least "
                f"'pdb_id' and 'label' columns."
            )
        has_split_column = "nrdld_split" in reader.fieldnames

        for row in reader:
            pdb_id = row["pdb_id"].strip()
            label = parse_label(row["label"])
            if label is None:
                print(f"[train] Skipping {pdb_id}: unparseable label '{row['label']}'")
                continue

            structure_path = os.path.join(structures_dir, f"{pdb_id}.pdb")
            if not os.path.isfile(structure_path):
                print(f"[train] Skipping {pdb_id}: no file at {structure_path}")
                continue

            try:
                pockets = detector.detect_pockets(structure_path)
            except Exception as e:
                print(f"[train] Skipping {pdb_id}: fpocket failed ({e})")
                continue

            if not pockets:
                print(f"[train] Skipping {pdb_id}: fpocket found no pockets")
                continue

            ligand_centroid = find_ligand_centroid(structure_path)
            best_pocket, method = select_training_pocket(pockets, ligand_centroid)
            method_counts[method] += 1

            features = extractor.extract(best_pocket)
            X.append(FeatureExtractor.to_vector(features))
            y.append(label)
            ids.append(pdb_id)
            splits.append(row["nrdld_split"].strip() if has_split_column else None)

    print(f"\n[train] Pocket selection method breakdown across {len(ids)} usable structures:")
    for method, count in method_counts.most_common():
        print(f"    {method:<20} {count}")
    if method_counts["volume_fallback"] > 0.3 * max(len(ids), 1):
        print(
            f"[train] NOTE: {method_counts['volume_fallback']}/{len(ids)} structures "
            f"({100 * method_counts['volume_fallback'] / len(ids):.0f}%) fell back to "
            f"volume-argmax — the ligand-proximity fix only partially engaged on this "
            f"dataset. Read the metrics below with that in mind."
        )

    if len(X) < 10:
        print(
            f"[train] Warning: only {len(X)} usable examples. A "
            f"RandomForestRegressor with cross-validation needs considerably "
            f"more than this to be trustworthy — treat any metrics below as "
            f"a pipeline sanity check, not a real evaluation, until the "
            f"dataset is bigger."
        )

    return np.array(X), np.array(y), ids, splits


def evaluate_on_original_split(
    X: np.ndarray,
    y: np.ndarray,
    ids: List[str],
    splits: List[Optional[str]],
    random_state: int = 42,
    n_estimators: int = 200,
    train_label: str = "t",
    val_label: str = "v",
    tune: bool = False,
    run_ablation: bool = False,
    feature_names: Optional[List[str]] = None,
) -> Tuple[bool, Optional[dict]]:
    """
    Trains strictly on rows marked as the original paper's training split
    and evaluates strictly on rows marked as its validation split — e.g.
    Krasowski et al. (2011)'s own 't'/'v' NRDLD partition, if the labels
    CSV carries an 'nrdld_split' column (build_dataset() passes this
    through as `splits`).

    THIS is the number directly comparable to DrugPred/PockDrug/etc.'s
    published MCC, since they evaluate on this exact held-out set after
    training on this exact training set. The random-split k-fold CV
    elsewhere in this script is a general robustness check across
    arbitrary splits — useful, but NOT the number to cite against
    published benchmarks, since a different split answers a different
    question (how stable is this approach across many random partitions,
    vs. how does it perform on the specific partition the field
    standardized on for comparability).

    tune=True runs GridSearchCV strictly on the training rows (see
    tune_hyperparameters — never touches X_val/y_val for model selection)
    before the one honest evaluation on the validation rows.

    run_ablation=True additionally reports MCC on this same held-out split
    with each ABLATION_GROUPS feature cluster removed, so you can see
    whether e.g. the hydrophobicity-proxy cluster is actually load-bearing
    or just absorbing impurity-importance credit from correlated features.

    Returns (found, best_params): found is True if the split was present and
    evaluated, False if the CSV had no nrdld_split column (nothing to do
    here in that case). best_params is the dict GridSearchCV selected when
    tune=True (so callers can apply the SAME hyperparameters to
    cross_validate/repeated_cross_validate for a fair, internally
    consistent comparison instead of silently mixing model configurations
    under one summary) — or None when tune=False or found=False.
    """
    if all(s is None for s in splits):
        print(
            "\n[train] No 'nrdld_split' column found in the labels CSV — "
            "skipping original-split evaluation. Only the random-split CV "
            "above is available, which is NOT directly comparable to "
            "published NRDLD benchmark numbers."
        )
        return False, None

    train_idx = [i for i, s in enumerate(splits) if s == train_label]
    val_idx = [i for i, s in enumerate(splits) if s == val_label]
    unmatched = [i for i, s in enumerate(splits) if s not in (train_label, val_label)]

    if unmatched:
        unmatched_values = sorted({splits[i] for i in unmatched})
        print(
            f"[train] Warning: {len(unmatched)} rows have an nrdld_split value "
            f"other than '{train_label}'/'{val_label}' ({unmatched_values}) — "
            f"excluded from this evaluation."
        )
    if not train_idx or not val_idx:
        print(
            f"[train] Warning: original-split evaluation needs both a "
            f"'{train_label}' and a '{val_label}' group — found "
            f"{len(train_idx)} train / {len(val_idx)} val. Skipping."
        )
        return False, None

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if tune:
        best_params = tune_hyperparameters(X_train, y_train, random_state=random_state)
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **best_params)
    else:
        best_params = None
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    mcc = matthews_corrcoef(y_val > 0.5, preds > 0.5)
    mae = mean_absolute_error(y_val, preds)
    cm = confusion_matrix((y_val > 0.5).astype(int), (preds > 0.5).astype(int))

    print(
        f"\n[train] Evaluation on ORIGINAL paper split "
        f"(train='{train_label}': {len(train_idx)}, val='{val_label}': {len(val_idx)})"
        f"{' [tuned]' if tune else ''}:"
    )
    print(f"    MCC: {mcc:.4f}   MAE: {mae:.4f}")
    print("    Confusion matrix (validation set only):")
    print("                     pred_less_drug  pred_druggable")
    print(f"        true_less_drug        {cm[0][0]:<8}      {cm[0][1]}")
    print(f"        true_druggable        {cm[1][0]:<8}      {cm[1][1]}")
    print(
        "    ^ THIS is the number directly comparable to DrugPred/PockDrug's "
        "published MCC on NRDLD — the random-split CV above answers a "
        "different question (robustness across arbitrary splits)."
    )

    # -- Permutation importance on the held-out validation set --------------
    # More trustworthy than RandomForestRegressor's built-in impurity-based
    # feature_importances_ when features are correlated (see
    # hydrophobicity_cluster below) — this measures actual held-out MCC
    # drop when a feature is shuffled, rather than tree-split greediness.
    if feature_names is not None:
        perm_result = permutation_importance(
            model, X_val, y_val, scoring=mcc_scorer,
            n_repeats=30, random_state=random_state, n_jobs=-1,
        )
        order = np.argsort(-perm_result.importances_mean)
        print(
            "\n[train] Permutation importance on held-out validation set "
            "(mean MCC drop when feature is shuffled, ± std across 30 repeats):"
        )
        for i in order:
            print(
                f"    {feature_names[i]:<38} "
                f"{perm_result.importances_mean[i]:+.4f}  (± {perm_result.importances_std[i]:.4f})"
            )

    # -- Feature-group ablation, evaluated on this same held-out split ------
    if run_ablation and feature_names is not None:
        print(f"\n[train] Feature-group ablation (baseline MCC: {mcc:.4f}):")
        for group_name, drop_names in ABLATION_GROUPS.items():
            X_train_abl, kept_names = drop_features(X_train, feature_names, drop_names)
            X_val_abl, _ = drop_features(X_val, feature_names, drop_names)

            abl_model = RandomForestRegressor(
                **(best_params if tune else {"n_estimators": n_estimators}),
                random_state=random_state,
                n_jobs=-1,
            )
            abl_model.fit(X_train_abl, y_train)
            abl_preds = abl_model.predict(X_val_abl)
            abl_mcc = matthews_corrcoef(y_val > 0.5, abl_preds > 0.5)

            delta = abl_mcc - mcc
            verdict = (
                "barely changed -> not load-bearing on its own"
                if abs(delta) < 0.03
                else ("HURT by removing -> genuinely useful" if delta < 0 else
                      "IMPROVED by removing -> may be adding noise/redundancy")
            )
            print(
                f"    without {group_name:<22} ({', '.join(drop_names)}): "
                f"MCC={abl_mcc:.4f}  (delta {delta:+.4f})  -- {verdict}"
            )

    return True, best_params


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 200,
    model_params: Optional[dict] = None,
) -> None:
    """
    Proper stratified 5-fold CV, reporting per-fold MCC (not just an
    aggregate) so a single unusually good/bad fold doesn't get averaged
    into a falsely reassuring or falsely damning single number. Also
    reports a pooled confusion matrix across all folds' held-out
    predictions.

    model_params: if provided (e.g. the dict returned by
    evaluate_on_original_split when tune=True), used INSTEAD of
    n_estimators for every fold's model — so this CV reflects the exact
    same hyperparameters as the tuned t/v evaluation, rather than silently
    comparing two different model configurations under one summary.
    """
    y_binary = (y > 0.5).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_mccs = []
    all_true, all_pred = [], []

    tuned_note = " [using tuned hyperparameters]" if model_params else ""
    print(f"\n[train] {n_splits}-fold stratified cross-validation{tuned_note}:")
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_binary), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_params:
            model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params)
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_state, n_jobs=-1
            )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_mcc = matthews_corrcoef(y_test > 0.5, preds > 0.5)
        fold_mccs.append(fold_mcc)
        all_true.extend((y_test > 0.5).astype(int))
        all_pred.extend((preds > 0.5).astype(int))

        print(
            f"    Fold {fold_idx}: n_test={len(test_idx):<3} "
            f"MCC={fold_mcc:.4f}  MAE={mean_absolute_error(y_test, preds):.4f}"
        )

    print(f"\n[train] CV MCC:  mean={np.mean(fold_mccs):.4f}  std={np.std(fold_mccs):.4f}")
    print(f"[train] CV MCC per fold: {[round(m, 4) for m in fold_mccs]}")

    cm = confusion_matrix(all_true, all_pred)
    print("\n[train] Pooled confusion matrix (across all folds' held-out predictions):")
    print("                 pred_less_drug  pred_druggable")
    print(f"    true_less_drug        {cm[0][0]:<8}      {cm[0][1]}")
    print(f"    true_druggable        {cm[1][0]:<8}      {cm[1][1]}")


def repeated_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
    n_estimators: int = 200,
    model_params: Optional[dict] = None,
) -> Tuple[float, float]:
    """
    Same idea as cross_validate(), but repeats the k-fold split n_repeats
    times with different random partitions and pools every fold's MCC
    together before reporting mean/std.

    Why this matters here specifically: with ~113 examples, a single 5-fold
    split leaves only ~23 held-out examples per fold — flipping 2-3
    predictions swings MCC by ~0.15-0.2. That's sampling noise, not a
    real signal about the model. Repeating the split (default: 10x, i.e.
    50 total fold evaluations instead of 5) gives a mean/std that reflects
    the approach's actual stability rather than one split's luck.

    model_params: if provided (e.g. the dict returned by
    evaluate_on_original_split when tune=True), used INSTEAD of
    n_estimators for every fold's model — see cross_validate's docstring
    for why this matters for a fair comparison across all three reported
    evaluations.

    Returns (mean_mcc, std_mcc) for convenience; also prints the full
    per-repeat breakdown so you can see whether variance is shrinking as
    expected (it should, roughly like 1/sqrt(n_repeats)) or whether
    something structural is going on instead.
    """
    y_binary = (y > 0.5).astype(int)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    all_fold_mccs = []
    all_true, all_pred = [], []

    tuned_note = " [using tuned hyperparameters]" if model_params else ""
    print(
        f"\n[train] Repeated {n_splits}-fold stratified CV{tuned_note} "
        f"({n_repeats} repeats, {n_splits * n_repeats} total fold evaluations):"
    )
    for train_idx, test_idx in rskf.split(X, y_binary):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_params:
            model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params)
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_state, n_jobs=-1
            )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_mcc = matthews_corrcoef(y_test > 0.5, preds > 0.5)
        all_fold_mccs.append(fold_mcc)
        all_true.extend((y_test > 0.5).astype(int))
        all_pred.extend((preds > 0.5).astype(int))

    mean_mcc = float(np.mean(all_fold_mccs))
    std_mcc = float(np.std(all_fold_mccs))

    # Break the per-repeat means out too, so you can see repeat-to-repeat
    # stability separately from fold-to-fold noise within a single repeat.
    per_repeat_means = [
        np.mean(all_fold_mccs[i : i + n_splits])
        for i in range(0, len(all_fold_mccs), n_splits)
    ]
    print(f"[train] Per-repeat mean MCC: {[round(m, 4) for m in per_repeat_means]}")
    print(
        f"[train] Pooled across all {len(all_fold_mccs)} folds: "
        f"mean={mean_mcc:.4f}  std={std_mcc:.4f}"
    )
    print(
        f"[train] (compare to single-run 5-fold CV above — this is the "
        f"more trustworthy number when they disagree)"
    )

    cm = confusion_matrix(all_true, all_pred)
    print("\n[train] Pooled confusion matrix (across ALL repeats' held-out predictions):")
    print("                 pred_less_drug  pred_druggable")
    print(f"    true_less_drug        {cm[0][0]:<8}      {cm[0][1]}")
    print(f"    true_druggable        {cm[1][0]:<8}      {cm[1][1]}")

    return mean_mcc, std_mcc


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 200,
    model_params: Optional[dict] = None,
) -> RandomForestRegressor:
    """
    Trains on ALL available data for the model artifact that actually gets
    saved and used at inference time. This is standard practice once CV has
    already given an honest out-of-sample performance estimate — CV's job
    is to tell us how good the approach is, not to produce the deployed
    model itself, so re-splitting off another held-out set here would only
    throw away training data for no benefit.

    model_params: if provided (e.g. the dict returned by
    evaluate_on_original_split when tune=True), used INSTEAD of
    n_estimators. IMPORTANT: this must match whatever configuration was
    actually evaluated above — otherwise the .joblib file saved at the end
    of this script silently reverts to untuned defaults regardless of what
    --tune-hyperparams found, which would mean the deployed model was never
    actually the one whose MCC you reported.
    """
    if model_params:
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **model_params)
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
    model.fit(X, y)

    print("\n[train] Final model (trained on 100% of data) feature importances:")
    for name, importance in sorted(
        zip(FeatureExtractor.ALL_FEATURE_NAMES, model.feature_importances_),
        key=lambda pair: -pair[1],
    ):
        print(f"    {name:<38} {importance:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--structures-dir", required=True)
    parser.add_argument("--model-out", default="druggability_model.joblib")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10,
                         help="Repeats of the k-fold CV split, pooled for a "
                              "stable mean/std (see repeated_cross_validate).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200,
                         help="Ignored for the original-split evaluation if "
                              "--tune-hyperparams is set (tuning picks its own).")
    parser.add_argument("--fpocket-binary", default="fpocket")
    parser.add_argument("--tune-hyperparams", action="store_true",
                         help="Run leak-free GridSearchCV on the 't' rows only "
                              "before the held-out 'v' evaluation.")
    parser.add_argument("--ablation", action="store_true",
                         help="Additionally report MCC with each "
                              "ABLATION_GROUPS feature cluster removed, on "
                              "the same held-out 'v' split.")
    args = parser.parse_args()

    detector = PocketDetector(fpocket_binary=args.fpocket_binary)
    extractor = FeatureExtractor()

    print(f"[train] Building dataset from {args.labels_csv} / {args.structures_dir} ...")
    X, y, ids, splits = build_dataset(args.labels_csv, args.structures_dir, detector, extractor)

    if len(X) < args.n_splits * 2:
        print(
            f"[train] Only {len(X)} usable examples found — need at least "
            f"~{args.n_splits * 2} for a {args.n_splits}-fold split to be "
            f"meaningful. Nothing trained."
        )
        sys.exit(1)

    _, best_params = evaluate_on_original_split(
        X, y, ids, splits,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        tune=args.tune_hyperparams,
        run_ablation=args.ablation,
        feature_names=FeatureExtractor.ALL_FEATURE_NAMES,
    )

    if args.tune_hyperparams and best_params:
        print(
            f"\n[train] Applying tuned hyperparameters {best_params} to the "
            f"CV routines below and the final saved model, so every "
            f"reported number reflects the SAME model configuration."
        )

    cross_validate(
        X, y, n_splits=args.n_splits, random_state=args.random_state,
        n_estimators=args.n_estimators, model_params=best_params,
    )

    repeated_cross_validate(
        X, y, n_splits=args.n_splits, n_repeats=args.n_repeats,
        random_state=args.random_state, n_estimators=args.n_estimators,
        model_params=best_params,
    )

    model = train_final_model(
        X, y, random_state=args.random_state, n_estimators=args.n_estimators,
        model_params=best_params,
    )

    joblib.dump(model, args.model_out)
    print(f"\n[train] Saved trained model to {args.model_out}")
    print(f"[train] Load it with: DruggabilityEngine(model_path='{args.model_out}')")


if __name__ == "__main__":
    main()
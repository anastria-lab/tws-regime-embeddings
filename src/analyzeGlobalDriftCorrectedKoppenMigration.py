# analyzeGlobalDriftCorrectedKoppenMigration.py
#
# Global drift-corrected climate-signature migration audit
#
# PURPOSE
# -------
# Use the EXISTING frozen 16-D hydrological representation to identify basins
# whose recent hydrological dynamics depart from the historical signature
# associated with their dominant Köppen-Geiger background and become more
# similar to another historically validated climate-associated signature.
#
# This script DOES NOT retrain the autoencoder.
# This script DOES NOT claim formal Köppen reclassification.
#
# Main safeguards
# ---------------
# 1. Historical reference: windows ending <= 2020-12.
# 2. Fully recent windows: windows starting >= 2021-01, so every 24-month
#    sequence is entirely post-2020.
# 3. Statistical source unit: basin, not overlapping windows.
# 4. Common global latent drift is estimated robustly at each time and removed.
# 5. A basin can be a transition candidate only if:
#       a) its static dominant KG fraction is >= 0.80,
#       b) its historical out-of-fold linear-probe prediction agrees with that
#          source class,
#       c) its recent drift-corrected probe signature changes to another class,
#       d) that source-target pair has historical cross-validated separability
#          from the previous probe.
# 6. Pairwise AUC is carried continuously. AUC >= 0.80 is only the PRIMARY
#    interpretation screen; sensitivity is reported at 0.75 / 0.80 / 0.85.
# 7. Quantitative analysis remains in the original 16-D latent space.
#
# Interpretation:
# "moves toward hydrological behavior historically associated with class X"
# is allowed.
# "the Köppen class changed to X" is NOT allowed without independent climate
# variable validation.

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# PATHS
# =============================================================================

EMBEDDINGS_FILE = Path(
    "data/processed/embeddings_window_level_continuous_v3.parquet"
)
STATIC_SUMMARY_FILE = Path(
    "data/processed/koppen_basin_static_composition_summary_labeled.csv"
)
PAIRWISE_AUC_FILE = Path(
    "results/tables/koppen_frozen_latent_probe/"
    "exact_class_pairwise_auc_anchor80.csv"
)

OUT_DIR = Path("results/tables/global_koppen_migration")
FIG_DIR = Path("results/figures/global_koppen_migration")

DRIFT_OUT = OUT_DIR / "global_latent_drift_by_window_end.csv"
BASIN_SUMMARY_OUT = OUT_DIR / "basin_recent_drift_corrected_summary.csv"
CLASS_SUMMARY_OUT = OUT_DIR / "source_class_migration_summary.csv"
PAIR_SCORES_OUT = OUT_DIR / "candidate_pair_axis_scores.csv"
CANDIDATES_OUT = OUT_DIR / "global_supported_migration_candidates.csv"
SENSITIVITY_OUT = OUT_DIR / "migration_candidate_auc_sensitivity.csv"
QC_OUT = OUT_DIR / "global_koppen_migration_qc.csv"
CONFIG_OUT = OUT_DIR / "global_koppen_migration_config.json"

HIST_END = pd.Timestamp("2020-12-01")
RECENT_START = pd.Timestamp("2021-01-01")

SOURCE_ANCHOR_THRESHOLD = 0.80
PRIMARY_PAIR_AUC = 0.80
PAIR_AUC_SENSITIVITY = [0.75, 0.80, 0.85]

MIN_CLASS_BASINS_FOR_PROBE = 10
MAX_CV_SPLITS = 5
RANDOM_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def detect_basin_column(df):
    for c in ["basin", "basin_id", "HYBAS_ID"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect basin identifier column.")


def latent_columns(df):
    cols = [c for c in df.columns if c.startswith("z")]
    try:
        cols = sorted(cols, key=lambda x: int(x[1:]))
    except Exception:
        cols = sorted(cols)
    if len(cols) != 16:
        raise ValueError(
            f"Expected 16 latent dimensions, found {len(cols)}: {cols}"
        )
    return cols


def adaptive_cv(y):
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return None
    n_splits = int(min(MAX_CV_SPLITS, counts.min()))
    if n_splits < 2:
        return None
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED,
    )


def make_logistic():
    return LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )


def pair_auc_lookup(pairwise):
    lut = {}
    for _, r in pairwise.iterrows():
        a = str(r["class_A"])
        b = str(r["class_B"])
        auc = float(r["roc_auc"])
        lut[(a, b)] = auc
        lut[(b, a)] = auc
    return lut


# =============================================================================
# LOAD
# =============================================================================

def load_inputs():
    if not PAIRWISE_AUC_FILE.exists():
        raise FileNotFoundError(
            f"Missing validated pairwise probe table: {PAIRWISE_AUC_FILE}\n"
            "Run probeKoppenInFrozenLatentSpace.py first."
        )

    emb = pd.read_parquet(EMBEDDINGS_FILE).copy()
    static = pd.read_csv(STATIC_SUMMARY_FILE)
    pairwise = pd.read_csv(PAIRWISE_AUC_FILE)

    bcol = detect_basin_column(emb)
    emb["basin_id"] = pd.to_numeric(
        emb[bcol], errors="raise"
    ).astype("int64")
    emb["start_time"] = pd.to_datetime(emb["start_time"])
    emb["end_time"] = pd.to_datetime(emb["end_time"])

    static["HYBAS_ID"] = pd.to_numeric(
        static["HYBAS_ID"], errors="raise"
    ).astype("int64")

    zcols = latent_columns(emb)

    required_static = {
        "HYBAS_ID",
        "KG_qc_pass_90pct_valid",
        "KG_dom_abbr",
        "KG_dom_frac",
    }
    missing = required_static - set(static.columns)
    if missing:
        raise ValueError(f"Static summary missing: {sorted(missing)}")

    required_pair = {"class_A", "class_B", "roc_auc"}
    missing = required_pair - set(pairwise.columns)
    if missing:
        raise ValueError(f"Pairwise table missing: {sorted(missing)}")

    print(f"✅ Continuous windows:    {len(emb):,}")
    print(f"✅ Embedding basins:      {emb['basin_id'].nunique():,}")
    print(f"✅ Static basin rows:     {len(static):,}")
    print(f"✅ Validated pair tests:  {len(pairwise):,}")
    print(f"✅ Latent dimensions:     {len(zcols)}")

    return emb, static, pairwise, zcols


# =============================================================================
# HISTORICAL STANDARDIZATION + BASIN BASELINES
# =============================================================================

def standardize_from_historical_windows(emb, zcols):
    hist_mask = emb["end_time"] <= HIST_END
    hist = emb.loc[hist_mask].copy()
    if hist.empty:
        raise ValueError("No historical windows ending <= 2020-12.")

    scaler = StandardScaler()
    scaler.fit(hist[zcols].to_numpy(dtype=float))

    Z = scaler.transform(emb[zcols].to_numpy(dtype=float))
    szcols = [f"sz{i}" for i in range(len(zcols))]

    out = emb.copy()
    out[szcols] = Z

    hist = out[out["end_time"] <= HIST_END].copy()

    print(f"✅ Historical windows:    {len(hist):,}")
    print(f"✅ Historical basins:     {hist['basin_id'].nunique():,}")

    return out, hist, szcols, scaler


def build_historical_basin_baselines(hist, static, szcols):
    h = (
        hist.groupby("basin_id")[szcols]
        .mean()
        .reset_index()
    )

    n = (
        hist.groupby("basin_id")
        .size()
        .rename("n_historical_windows")
        .reset_index()
    )

    h = h.merge(n, on="basin_id", how="left", validate="one_to_one")
    h = h.merge(
        static,
        left_on="basin_id",
        right_on="HYBAS_ID",
        how="left",
        validate="one_to_one",
    )

    print(f"✅ Historical basin baselines: {len(h):,}")
    return h


# =============================================================================
# GLOBAL COMMON DRIFT
# =============================================================================

def estimate_global_drift(emb_scaled, hist_basins, szcols):
    """
    For each basin-window:
        residual = current standardized latent vector - basin historical mean

    For each end time:
        common drift = median residual across basins

    Then center that time-varying drift by its historical time mean so the
    historical reference has approximately zero common drift.
    """
    base_cols = ["basin_id"] + szcols
    base = hist_basins[base_cols].copy()
    base = base.rename(columns={c: f"h_{c}" for c in szcols})

    d = emb_scaled.merge(
        base,
        on="basin_id",
        how="inner",
        validate="many_to_one",
    )

    delta_cols = []
    for c in szcols:
        dc = f"delta_{c}"
        d[dc] = d[c] - d[f"h_{c}"]
        delta_cols.append(dc)

    drift = (
        d.groupby("end_time")[delta_cols]
        .median()
        .reset_index()
        .sort_values("end_time")
    )

    historical_times = drift["end_time"] <= HIST_END
    if not historical_times.any():
        raise ValueError("No historical time points available for drift centering.")

    g0 = (
        drift.loc[historical_times, delta_cols]
        .mean(axis=0)
        .to_numpy(dtype=float)
    )

    gcorr_cols = []
    for i, dc in enumerate(delta_cols):
        gc = f"global_drift_{szcols[i]}"
        drift[gc] = drift[dc] - g0[i]
        gcorr_cols.append(gc)

    G = drift[gcorr_cols].to_numpy(dtype=float)
    drift["global_drift_magnitude"] = np.linalg.norm(G, axis=1)

    keep = ["end_time"] + gcorr_cols + ["global_drift_magnitude"]
    drift = drift[keep].copy()

    # Apply the common drift correction to every window with a historical
    # basin baseline.
    d = d.merge(
        drift,
        on="end_time",
        how="left",
        validate="many_to_one",
    )

    corr_cols = []
    for i, c in enumerate(szcols):
        cc = f"corr_{c}"
        d[cc] = d[c] - d[gcorr_cols[i]]
        corr_cols.append(cc)

    return d, drift, corr_cols, gcorr_cols


# =============================================================================
# RECENT BASIN SUMMARIES
# =============================================================================

def build_recent_basin_summary(
    corrected_windows,
    hist_basins,
    szcols,
    corr_cols,
    gcorr_cols,
):
    recent = corrected_windows[
        corrected_windows["start_time"] >= RECENT_START
    ].copy()

    if recent.empty:
        raise ValueError("No fully post-2020 windows starting >= 2021-01.")

    raw_mean = (
        recent.groupby("basin_id")[szcols]
        .mean()
        .reset_index()
        .rename(columns={c: f"recent_raw_{c}" for c in szcols})
    )

    corr_mean = (
        recent.groupby("basin_id")[corr_cols]
        .mean()
        .reset_index()
        .rename(
            columns={
                corr_cols[i]: f"recent_corr_{szcols[i]}"
                for i in range(len(szcols))
            }
        )
    )

    count = (
        recent.groupby("basin_id")
        .agg(
            n_recent_fully_post2020_windows=("end_time", "size"),
            recent_first_start=("start_time", "min"),
            recent_last_end=("end_time", "max"),
        )
        .reset_index()
    )

    drift_mag = (
        recent.groupby("basin_id")["global_drift_magnitude"]
        .mean()
        .rename("mean_recent_global_drift_magnitude")
        .reset_index()
    )

    out = hist_basins.copy()
    out = out.merge(raw_mean, on="basin_id", how="inner")
    out = out.merge(corr_mean, on="basin_id", how="inner")
    out = out.merge(count, on="basin_id", how="inner")
    out = out.merge(drift_mag, on="basin_id", how="inner")

    H = out[szcols].to_numpy(dtype=float)
    Rraw = out[
        [f"recent_raw_{c}" for c in szcols]
    ].to_numpy(dtype=float)
    Rcorr = out[
        [f"recent_corr_{c}" for c in szcols]
    ].to_numpy(dtype=float)

    out["raw_recent_displacement_magnitude"] = np.linalg.norm(
        Rraw - H,
        axis=1,
    )
    out["drift_corrected_recent_displacement_magnitude"] = np.linalg.norm(
        Rcorr - H,
        axis=1,
    )

    print(
        "✅ Basins with historical + fully post-2020 summaries: "
        f"{len(out):,}"
    )
    return out


# =============================================================================
# MULTICLASS HISTORICAL-SIGNATURE PROBE
# =============================================================================

def fit_multiclass_probe(basin_summary, szcols):
    anchors = basin_summary[
        basin_summary["KG_qc_pass_90pct_valid"]
        .fillna(False)
        .astype(bool)
        & (basin_summary["KG_dom_frac"] >= SOURCE_ANCHOR_THRESHOLD)
    ].copy()

    counts = anchors["KG_dom_abbr"].value_counts()
    eligible = sorted(
        counts[counts >= MIN_CLASS_BASINS_FOR_PROBE].index.astype(str)
    )

    anchors = anchors[
        anchors["KG_dom_abbr"].astype(str).isin(eligible)
    ].copy()

    X = anchors[szcols].to_numpy(dtype=float)
    y = anchors["KG_dom_abbr"].astype(str).to_numpy()

    cv = adaptive_cv(y)
    if cv is None:
        raise ValueError("Insufficient class support for multiclass probe.")

    # Historical out-of-fold prediction: this determines whether each candidate
    # was historically hydrologically consistent with its static source class.
    oof = cross_val_predict(
        make_logistic(),
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=None,
    )
    anchors["historical_oof_probe_class"] = oof
    anchors["historical_oof_matches_static"] = (
        anchors["historical_oof_probe_class"]
        == anchors["KG_dom_abbr"].astype(str)
    )

    # Final model represents historical climate-associated signatures.
    model = make_logistic()
    model.fit(X, y)

    return anchors, model, eligible


def apply_multiclass_probe(basin_summary, anchors_oof, model, eligible, szcols):
    out = basin_summary.copy()

    # Attach OOF historical probe result only for eligible 80% anchors.
    oof = anchors_oof[
        [
            "basin_id",
            "historical_oof_probe_class",
            "historical_oof_matches_static",
        ]
    ].copy()

    out = out.merge(oof, on="basin_id", how="left", validate="one_to_one")

    H = out[szcols].to_numpy(dtype=float)
    Rraw = out[
        [f"recent_raw_{c}" for c in szcols]
    ].to_numpy(dtype=float)
    Rcorr = out[
        [f"recent_corr_{c}" for c in szcols]
    ].to_numpy(dtype=float)

    out["historical_full_probe_class"] = model.predict(H)
    out["recent_raw_probe_class"] = model.predict(Rraw)
    out["recent_corrected_probe_class"] = model.predict(Rcorr)

    proba_corr = model.predict_proba(Rcorr)
    class_names = model.classes_.astype(str)

    out["recent_corrected_probe_max_probability"] = proba_corr.max(axis=1)

    source_probs = np.full(len(out), np.nan)
    target_probs = np.full(len(out), np.nan)

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for i, row in out.iterrows():
        source = str(row["KG_dom_abbr"])
        target = str(row["recent_corrected_probe_class"])
        if source in class_to_idx:
            source_probs[out.index.get_loc(i)] = proba_corr[
                out.index.get_loc(i),
                class_to_idx[source],
            ]
        if target in class_to_idx:
            target_probs[out.index.get_loc(i)] = proba_corr[
                out.index.get_loc(i),
                class_to_idx[target],
            ]

    out["recent_corrected_source_probability"] = source_probs
    out["recent_corrected_target_probability"] = target_probs
    out["eligible_probe_class_count"] = len(eligible)

    return out


# =============================================================================
# PAIRWISE SOURCE -> TARGET AXES
# =============================================================================

def fit_binary_pair_axis(
    anchor_table,
    source,
    target,
    szcols,
):
    """
    Fit historical binary linear probe and normalize its decision axis so that:

        mean source-anchor score = 0
        mean target-anchor score = 1

    The axis is used for temporal interpretation only after the independent
    historical pairwise CV AUC has already validated the class distinction.
    """
    d = anchor_table[
        anchor_table["KG_dom_abbr"].astype(str).isin([source, target])
    ].copy()

    X = d[szcols].to_numpy(dtype=float)
    y = (d["KG_dom_abbr"].astype(str) == target).astype(int).to_numpy()

    if len(np.unique(y)) != 2:
        return None

    model = make_logistic()
    model.fit(X, y)

    score = model.decision_function(X)
    source_mean = float(
        np.mean(score[d["KG_dom_abbr"].astype(str).to_numpy() == source])
    )
    target_mean = float(
        np.mean(score[d["KG_dom_abbr"].astype(str).to_numpy() == target])
    )

    # Numerical/orientation safeguard.
    if target_mean <= source_mean:
        score = -score
        source_mean = -source_mean
        target_mean = -target_mean

        # Flip model coefficients too, so future scoring has same orientation.
        model.coef_ *= -1.0
        model.intercept_ *= -1.0

    denom = target_mean - source_mean
    if denom <= 1e-12:
        return None

    return model, source_mean, target_mean


def normalized_pair_position(X, model, source_mean, target_mean):
    s = model.decision_function(np.asarray(X, dtype=float))
    return (s - source_mean) / (target_mean - source_mean)


def build_pair_scores(
    basin_summary,
    anchors_oof,
    pairwise,
    szcols,
):
    lut = pair_auc_lookup(pairwise)

    # Candidate source basins must be strong static anchors and historically
    # recovered by the OOF probe.
    sources = basin_summary[
        basin_summary["KG_qc_pass_90pct_valid"]
        .fillna(False)
        .astype(bool)
        & (basin_summary["KG_dom_frac"] >= SOURCE_ANCHOR_THRESHOLD)
        & basin_summary["historical_oof_matches_static"]
        .fillna(False)
        .astype(bool)
    ].copy()

    # Only evaluate the recent corrected class proposed by the multiclass probe.
    sources = sources[
        sources["recent_corrected_probe_class"].notna()
    ].copy()
    sources = sources[
        sources["recent_corrected_probe_class"].astype(str)
        != sources["KG_dom_abbr"].astype(str)
    ].copy()

    pair_models = {}
    rows = []

    for _, r in sources.iterrows():
        source = str(r["KG_dom_abbr"])
        target = str(r["recent_corrected_probe_class"])

        auc = lut.get((source, target), np.nan)
        if not np.isfinite(auc):
            # No historical pairwise validation for this source-target pair.
            continue

        key = (source, target)
        if key not in pair_models:
            pair_models[key] = fit_binary_pair_axis(
                anchors_oof,
                source,
                target,
                szcols,
            )

        fitted = pair_models[key]
        if fitted is None:
            continue

        model, source_mean, target_mean = fitted

        H = r[szcols].to_numpy(dtype=float).reshape(1, -1)
        Rraw = r[
            [f"recent_raw_{c}" for c in szcols]
        ].to_numpy(dtype=float).reshape(1, -1)
        Rcorr = r[
            [f"recent_corr_{c}" for c in szcols]
        ].to_numpy(dtype=float).reshape(1, -1)

        ph = float(
            normalized_pair_position(
                H, model, source_mean, target_mean
            )[0]
        )
        pr = float(
            normalized_pair_position(
                Rraw, model, source_mean, target_mean
            )[0]
        )
        pc = float(
            normalized_pair_position(
                Rcorr, model, source_mean, target_mean
            )[0]
        )

        rows.append({
            "basin_id": int(r["basin_id"]),
            "source_KG": source,
            "target_signature": target,
            "KG_dom_frac": float(r["KG_dom_frac"]),
            "pairwise_historical_cv_auc": float(auc),
            "historical_pair_axis_position": ph,
            "recent_raw_pair_axis_position": pr,
            "recent_corrected_pair_axis_position": pc,
            "raw_delta_toward_target": pr - ph,
            "corrected_delta_toward_target": pc - ph,
            "historically_source_side": bool(ph < 0.5),
            "raw_recent_target_side": bool(pr >= 0.5),
            "corrected_recent_target_side": bool(pc >= 0.5),
            "raw_crossed_pair_midpoint": bool(ph < 0.5 and pr >= 0.5),
            "corrected_crossed_pair_midpoint": bool(
                ph < 0.5 and pc >= 0.5
            ),
            "recent_raw_probe_class": str(r["recent_raw_probe_class"]),
            "recent_corrected_probe_class": str(
                r["recent_corrected_probe_class"]
            ),
            "recent_corrected_source_probability": float(
                r["recent_corrected_source_probability"]
            ),
            "recent_corrected_target_probability": float(
                r["recent_corrected_target_probability"]
            ),
            "mean_recent_global_drift_magnitude": float(
                r["mean_recent_global_drift_magnitude"]
            ),
            "raw_recent_displacement_magnitude": float(
                r["raw_recent_displacement_magnitude"]
            ),
            "drift_corrected_recent_displacement_magnitude": float(
                r["drift_corrected_recent_displacement_magnitude"]
            ),
            "n_historical_windows": int(r["n_historical_windows"]),
            "n_recent_fully_post2020_windows": int(
                r["n_recent_fully_post2020_windows"]
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARIES
# =============================================================================

def build_candidate_table(pair_scores):
    if pair_scores.empty:
        return pair_scores.copy()

    c = pair_scores.copy()

    c["primary_auc_supported"] = (
        c["pairwise_historical_cv_auc"] >= PRIMARY_PAIR_AUC
    )

    # Primary candidate:
    # - pair historically distinguishable at the primary AUC screen
    # - basin was historically on source side of that pair axis
    # - recent corrected position is on target side
    # - movement is positive after global drift correction
    c["primary_supported_transition_candidate"] = (
        c["primary_auc_supported"]
        & c["historically_source_side"]
        & c["corrected_recent_target_side"]
        & (c["corrected_delta_toward_target"] > 0)
    )

    # Evidence-strength ranking. This is a ranking aid, NOT a probability.
    c["candidate_strength_score"] = (
        c["corrected_delta_toward_target"].clip(lower=0)
        * (2.0 * c["pairwise_historical_cv_auc"] - 1.0)
    )

    return c.sort_values(
        [
            "primary_supported_transition_candidate",
            "candidate_strength_score",
        ],
        ascending=[False, False],
    )


def class_summary(basin_summary, candidate_table):
    anchors = basin_summary[
        basin_summary["KG_qc_pass_90pct_valid"]
        .fillna(False)
        .astype(bool)
        & (basin_summary["KG_dom_frac"] >= SOURCE_ANCHOR_THRESHOLD)
        & basin_summary["historical_oof_matches_static"]
        .fillna(False)
        .astype(bool)
    ].copy()

    rows = []

    candidate_ids = set()
    if not candidate_table.empty:
        candidate_ids = set(
            candidate_table.loc[
                candidate_table["primary_supported_transition_candidate"],
                "basin_id",
            ].astype(int)
        )

    for source, g in anchors.groupby("KG_dom_abbr"):
        source = str(source)
        n = len(g)

        raw_stay = float(
            (g["recent_raw_probe_class"].astype(str) == source).mean()
        )
        corr_stay = float(
            (
                g["recent_corrected_probe_class"].astype(str)
                == source
            ).mean()
        )

        n_supported_transition = int(
            g["basin_id"].astype(int).isin(candidate_ids).sum()
        )

        rows.append({
            "source_KG": source,
            "n_historically_consistent_anchor_basins": n,
            "fraction_recent_raw_probe_stays_source": raw_stay,
            "fraction_recent_corrected_probe_stays_source": corr_stay,
            "n_primary_supported_transition_candidates": n_supported_transition,
            "fraction_primary_supported_transition_candidates": (
                n_supported_transition / n if n else np.nan
            ),
            "median_raw_displacement_magnitude": float(
                g["raw_recent_displacement_magnitude"].median()
            ),
            "median_corrected_displacement_magnitude": float(
                g["drift_corrected_recent_displacement_magnitude"].median()
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "n_historically_consistent_anchor_basins",
        ascending=False,
    )


def auc_sensitivity(candidate_table):
    rows = []

    for t in PAIR_AUC_SENSITIVITY:
        if candidate_table.empty:
            n_rows = 0
            n_basins = 0
            n_cross = 0
        else:
            s = candidate_table[
                (candidate_table["pairwise_historical_cv_auc"] >= t)
                & candidate_table["historically_source_side"]
                & candidate_table["corrected_recent_target_side"]
                & (candidate_table["corrected_delta_toward_target"] > 0)
            ]
            n_rows = len(s)
            n_basins = s["basin_id"].nunique()
            n_cross = int(s["corrected_crossed_pair_midpoint"].sum())

        rows.append({
            "pairwise_auc_screen": t,
            "candidate_rows": n_rows,
            "unique_candidate_basins": n_basins,
            "midpoint_crossing_candidates": n_cross,
        })

    return pd.DataFrame(rows)


# =============================================================================
# FIGURES
# =============================================================================

def plot_global_drift(drift):
    plt.figure(figsize=(12, 5))
    plt.plot(
        drift["end_time"],
        drift["global_drift_magnitude"],
        linewidth=1.4,
    )
    plt.axvline(HIST_END, linestyle="--", linewidth=1)
    plt.xlabel("24-month window end")
    plt.ylabel("Common latent-drift magnitude")
    plt.title(
        "Common global displacement in historical-standardized 16-D latent space"
    )
    plt.tight_layout()
    out = FIG_DIR / "global_latent_drift_magnitude_through_time.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_raw_vs_corrected_displacement(basin_summary):
    d = basin_summary[
        np.isfinite(basin_summary["raw_recent_displacement_magnitude"])
        & np.isfinite(
            basin_summary["drift_corrected_recent_displacement_magnitude"]
        )
    ].copy()

    plt.figure(figsize=(7, 7))
    plt.scatter(
        d["raw_recent_displacement_magnitude"],
        d["drift_corrected_recent_displacement_magnitude"],
        s=18,
        alpha=0.55,
    )
    lim = float(max(
        d["raw_recent_displacement_magnitude"].max(),
        d["drift_corrected_recent_displacement_magnitude"].max(),
    ))
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    plt.xlabel("Raw recent displacement")
    plt.ylabel("Drift-corrected recent displacement")
    plt.title("Effect of removing common global latent drift")
    plt.tight_layout()
    out = FIG_DIR / "raw_vs_drift_corrected_recent_displacement.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# =============================================================================
# QC / CONSOLE
# =============================================================================

def build_qc(
    emb,
    hist_basins,
    basin_summary,
    anchors_oof,
    pair_scores,
    candidates,
):
    return pd.DataFrame([
        {
            "check": "continuous_windows",
            "value": len(emb),
        },
        {
            "check": "embedding_basins",
            "value": emb["basin_id"].nunique(),
        },
        {
            "check": "historical_basin_baselines",
            "value": len(hist_basins),
        },
        {
            "check": "basins_with_fully_post2020_summary",
            "value": len(basin_summary),
        },
        {
            "check": "eligible_80pct_anchor_basins",
            "value": len(anchors_oof),
        },
        {
            "check": "historically_OOF_consistent_anchor_basins",
            "value": int(
                anchors_oof["historical_oof_matches_static"].sum()
            ),
        },
        {
            "check": "source_target_pair_score_rows",
            "value": len(pair_scores),
        },
        {
            "check": "primary_supported_transition_candidates",
            "value": int(
                candidates[
                    "primary_supported_transition_candidate"
                ].sum()
            ) if not candidates.empty else 0,
        },
    ])


def print_console(
    drift,
    basin_summary,
    eligible,
    anchors_oof,
    class_stats,
    candidates,
    sensitivity,
):
    print("\n" + "=" * 104)
    print("GLOBAL DRIFT-CORRECTED KÖPPEN-ASSOCIATED HYDROLOGICAL MIGRATION AUDIT")
    print("=" * 104)

    print("\nA) Coverage and historical probe reference")
    print(f"Eligible exact classes in 80% anchor probe: {len(eligible)}")
    print("Classes: " + ", ".join(eligible))
    print(f"80% anchor basins in multiclass probe: {len(anchors_oof):,}")
    print(
        "Historical OOF probe agrees with static source class: "
        f"{int(anchors_oof['historical_oof_matches_static'].sum()):,} "
        f"({anchors_oof['historical_oof_matches_static'].mean():.3f})"
    )
    print(
        "Basins with historical + fully post-2020 summaries: "
        f"{len(basin_summary):,}"
    )

    print("\nB) Common global post-2020 latent drift")
    recent_drift = drift[drift["end_time"] >= pd.Timestamp("2022-12-01")]
    # 2022-12 is approximately the earliest end date for a sequence that began
    # in 2021; this is descriptive only.
    print(
        f"Recent drift magnitude median: "
        f"{recent_drift['global_drift_magnitude'].median():.3f}"
    )
    print(
        f"Recent drift magnitude mean:   "
        f"{recent_drift['global_drift_magnitude'].mean():.3f}"
    )
    print(
        f"Recent drift magnitude max:    "
        f"{recent_drift['global_drift_magnitude'].max():.3f}"
    )
    print(
        "Median basin displacement before correction: "
        f"{basin_summary['raw_recent_displacement_magnitude'].median():.3f}"
    )
    print(
        "Median basin displacement after correction:  "
        f"{basin_summary['drift_corrected_recent_displacement_magnitude'].median():.3f}"
    )

    print("\nC) Source-class stability after common-drift correction")
    print(
        class_stats.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        "\nD) Top primary supported climate-signature migration candidates"
    )
    if candidates.empty:
        print("No source-target candidate rows.")
    else:
        top = candidates[
            candidates["primary_supported_transition_candidate"]
        ].head(25)

        if top.empty:
            print(
                "No candidates passed the primary AUC>=0.80 + historical "
                "source-side + corrected target-side screen."
            )
        else:
            cols = [
                "basin_id",
                "source_KG",
                "target_signature",
                "KG_dom_frac",
                "pairwise_historical_cv_auc",
                "historical_pair_axis_position",
                "recent_raw_pair_axis_position",
                "recent_corrected_pair_axis_position",
                "corrected_delta_toward_target",
                "raw_crossed_pair_midpoint",
                "corrected_crossed_pair_midpoint",
                "candidate_strength_score",
            ]
            print(
                top[cols].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.3f}",
                )
            )

    print("\nE) Pairwise-AUC sensitivity")
    print(
        sensitivity.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nInterpretation guardrails:")
    print(
        "- These are changes in hydrological similarity within the frozen "
        "16-D representation, not formal Köppen reclassifications."
    )
    print(
        "- The common global latent drift is removed before the primary "
        "migration interpretation."
    )
    print(
        "- Only strong static source anchors whose historical OOF hydrological "
        "signature agreed with the source class can become candidates."
    )
    print(
        "- AUC>=0.80 is an interpretation screen, not a natural law; "
        "0.75/0.80/0.85 sensitivity is reported."
    )
    print(
        "- A transition candidate is still a hypothesis for independent "
        "physical/climate validation, not a confirmed climate-class change."
    )
    print(
        "- Overlapping 24-month windows are trajectory samples, not "
        "independent inferential observations."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Global drift-corrected Köppen migration audit ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    emb, static, pairwise, zcols = load_inputs()

    emb_scaled, hist, szcols, scaler = (
        standardize_from_historical_windows(emb, zcols)
    )

    hist_basins = build_historical_basin_baselines(
        hist,
        static,
        szcols,
    )

    corrected_windows, drift, corr_cols, gcorr_cols = (
        estimate_global_drift(
            emb_scaled,
            hist_basins,
            szcols,
        )
    )

    drift.to_csv(DRIFT_OUT, index=False)
    print(f"✅ Saved: {DRIFT_OUT}")

    basin_summary = build_recent_basin_summary(
        corrected_windows,
        hist_basins,
        szcols,
        corr_cols,
        gcorr_cols,
    )

    anchors_oof, multiclass_model, eligible = fit_multiclass_probe(
        basin_summary,
        szcols,
    )

    basin_summary = apply_multiclass_probe(
        basin_summary,
        anchors_oof,
        multiclass_model,
        eligible,
        szcols,
    )

    basin_summary.to_csv(BASIN_SUMMARY_OUT, index=False)
    print(f"✅ Saved: {BASIN_SUMMARY_OUT}")

    pair_scores = build_pair_scores(
        basin_summary,
        anchors_oof,
        pairwise,
        szcols,
    )
    pair_scores.to_csv(PAIR_SCORES_OUT, index=False)
    print(f"✅ Saved: {PAIR_SCORES_OUT}")

    candidates = build_candidate_table(pair_scores)
    candidates.to_csv(CANDIDATES_OUT, index=False)
    print(f"✅ Saved: {CANDIDATES_OUT}")

    class_stats = class_summary(
        basin_summary,
        candidates,
    )
    class_stats.to_csv(CLASS_SUMMARY_OUT, index=False)
    print(f"✅ Saved: {CLASS_SUMMARY_OUT}")

    sensitivity = auc_sensitivity(candidates)
    sensitivity.to_csv(SENSITIVITY_OUT, index=False)
    print(f"✅ Saved: {SENSITIVITY_OUT}")

    qc = build_qc(
        emb,
        hist_basins,
        basin_summary,
        anchors_oof,
        pair_scores,
        candidates,
    )
    qc.to_csv(QC_OUT, index=False)
    print(f"✅ Saved: {QC_OUT}")

    config = {
        "historical_window_end_max": str(HIST_END.date()),
        "fully_recent_window_start_min": str(RECENT_START.date()),
        "source_anchor_threshold": SOURCE_ANCHOR_THRESHOLD,
        "primary_pair_auc_screen": PRIMARY_PAIR_AUC,
        "pair_auc_sensitivity": PAIR_AUC_SENSITIVITY,
        "common_drift_estimator": (
            "per-end-time median of basin-centered standardized latent residuals"
        ),
        "common_drift_historical_centering": True,
        "representation_frozen": True,
        "latent_dimensions": 16,
        "quantitative_space": "historical-standardized 16D latent space",
        "formal_koppen_reclassification_claim": False,
    }
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved: {CONFIG_OUT}")

    plot_global_drift(drift)
    plot_raw_vs_corrected_displacement(basin_summary)

    print_console(
        drift,
        basin_summary,
        eligible,
        anchors_oof,
        class_stats,
        candidates,
        sensitivity,
    )

    print("\n--- Done ---")
    print(
        "STOP here. Interpret sections A-E before selecting any basin "
        "trajectory for physical validation or the poster."
    )


if __name__ == "__main__":
    main()

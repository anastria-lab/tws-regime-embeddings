# finalValidateSourceRecentTargetPhysicalConvergence.py
#
# FINAL INTERNAL PHYSICAL VALIDATION
# ----------------------------------
# For every duration-matched strict latent migration candidate, compare:
#
#     candidate historical physical fingerprint
#                    ->
#     candidate recent physical fingerprint
#                    ->
#     historical TARGET-class physical fingerprint
#
# The comparison uses exactly the same duration-matched blocks defined by the
# completed latent support audit. Historical and recent physical fingerprints
# therefore cover the same temporal scale.
#
# This script does NOT retrain the autoencoder and does NOT use UMAP distances.
# It is a physical-consistency validation using variables already in the
# GRACE/ERA5 modeling dataset. It is not independent proof of formal Köppen
# reclassification.
#
# Conservative final outcomes:
#   - confirmed_physical_convergence
#   - physical_support_overlap
#   - partial_physical_convergence
#   - physically_novel_beyond_source_and_target
#   - latent_migration_not_physically_supported
#   - physical_pair_not_distinguishable

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_STRATIFIED_GROUP = True
except Exception:
    HAVE_STRATIFIED_GROUP = False


# =============================================================================
# PATHS
# =============================================================================

STRICT_FILE = Path(
    "results/tables/koppen_duration_matched_support/"
    "duration_matched_strict_migrations_q95.csv"
)
HIST_SEGMENTS_FILE = Path(
    "results/tables/koppen_duration_matched_support/"
    "historical_duration_matched_segments.parquet"
)
RECENT_SEGMENTS_FILE = Path(
    "results/tables/koppen_duration_matched_support/"
    "recent_duration_matched_basin_means.csv"
)

PHYSICAL_FILE = Path(
    "data/processed/basin_dataset_anomaly_normalized.parquet"
)
WAVELET_FILE = Path(
    "data/processed/basin_dataset_wavelet_multiscale.parquet"
)

OUT_DIR = Path("results/tables/final_physical_convergence")
FIG_DIR = Path("results/figures/final_physical_convergence")

HIST_PHYSICAL_SEGMENTS_OUT = OUT_DIR / "historical_duration_matched_physical_fingerprints.csv"
RECENT_PHYSICAL_OUT = OUT_DIR / "recent_duration_matched_physical_fingerprints.csv"
CLASS_SUPPORT_OUT = OUT_DIR / "historical_physical_class_support_q95.csv"
PAIR_AUC_OUT = OUT_DIR / "historical_physical_pair_cv_auc.csv"
FINAL_OUT = OUT_DIR / "final_source_recent_target_physical_validation.csv"
SUMMARY_OUT = OUT_DIR / "final_physical_validation_summary.csv"
SENSITIVITY_OUT = OUT_DIR / "final_physical_support_quantile_sensitivity.csv"
AVAILABILITY_OUT = OUT_DIR / "final_physical_feature_availability.csv"
QC_OUT = OUT_DIR / "final_physical_validation_qc.csv"
CONFIG_OUT = OUT_DIR / "final_physical_validation_config.json"

PRIMARY_SUPPORT_QUANTILE = 0.95
SUPPORT_QUANTILES = [0.90, 0.95, 0.99]
K_NEIGHBORS = 3

# A physical pair needs at least moderate out-of-basin historical separation
# before we call target convergence "confirmed".
PHYSICAL_PAIR_AUC_SCREEN = 0.75

MAX_CV_SPLITS = 5
RANDOM_SEED = 42


# =============================================================================
# PHYSICAL FEATURES
# =============================================================================

# Core variables with relatively direct physical interpretation.
MONTHLY_VARIABLE_OPTIONS = {
    "TWSA": [
        "lwe_thickness_anomaly_normalized",
        "lwe_thickness_anomaly",
        "lwe_thickness",
    ],
    "P": [
        "tp_anomaly_normalized",
        "tp_anomaly",
        "tp",
    ],
    "SOIL4": [
        "swvl4_anomaly_normalized",
        "swvl4_anomaly",
        "swvl4",
    ],
    "SRO": [
        "sro_anomaly_normalized",
        "sro_anomaly",
        "sro",
    ],
    "SSRO": [
        "ssro_anomaly_normalized",
        "ssro_anomaly",
        "ssro",
    ],
}

# Long components used in the learned multiscale representation.
WAVELET_LONG_OPTIONS = {
    "TWSA": ["lwe_thickness_long"],
    "P": ["tp_long"],
    "SOIL4": ["swvl4_long"],
    "SRO": ["sro_long"],
    "SSRO": ["ssro_long"],
}

# Seasonal RMS is included for the three most interpretable storage/climate
# variables. Mean seasonal wavelet components are often near zero, so RMS is
# more informative than their signed mean.
WAVELET_SEASONAL_OPTIONS = {
    "TWSA": ["lwe_thickness_seasonal"],
    "P": ["tp_seasonal"],
    "SOIL4": ["swvl4_seasonal"],
}


# =============================================================================
# HELPERS
# =============================================================================

def detect_basin_column(df):
    for c in ["basin", "basin_id", "HYBAS_ID"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect basin identifier column.")


def canonical_time(df):
    d = df.copy()
    if "time" in d.columns:
        d["time"] = pd.to_datetime(d["time"])
    elif "time_era5" in d.columns:
        d["time"] = pd.to_datetime(d["time_era5"])
    elif {"year", "month"}.issubset(d.columns):
        d["time"] = pd.to_datetime(
            dict(year=d["year"], month=d["month"], day=1)
        )
    elif "time_grace" in d.columns:
        d["time"] = pd.to_datetime(d["time_grace"])
    else:
        raise ValueError("Could not construct a monthly time column.")
    return d


def choose_column(columns, options):
    cols = set(columns)
    return next((c for c in options if c in cols), None)


def linear_slope(values):
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def mean_k_nearest_distance(point, reference, k):
    point = np.asarray(point, dtype=float).reshape(1, -1)
    reference = np.asarray(reference, dtype=float)
    if len(reference) == 0:
        return np.nan
    d = np.linalg.norm(reference - point, axis=1)
    k_eff = min(k, len(d))
    return float(np.mean(np.partition(d, k_eff - 1)[:k_eff]))


def centroid_axis_position(x, source_centroid, target_centroid):
    x = np.asarray(x, dtype=float)
    s = np.asarray(source_centroid, dtype=float)
    t = np.asarray(target_centroid, dtype=float)
    v = t - s
    denom = float(np.dot(v, v))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(x - s, v) / denom)


def make_binary_probe():
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_SEED,
            ),
        ),
    ])


# =============================================================================
# LOAD
# =============================================================================

def load_inputs():
    strict = pd.read_csv(STRICT_FILE)
    hist_seg = pd.read_parquet(HIST_SEGMENTS_FILE)
    recent_seg = pd.read_csv(RECENT_SEGMENTS_FILE)

    physical = pd.read_parquet(PHYSICAL_FILE)
    wavelet = pd.read_parquet(WAVELET_FILE) if WAVELET_FILE.exists() else None

    for d in [strict, hist_seg, recent_seg]:
        if "basin_id" not in d.columns:
            raise ValueError("Expected basin_id in duration-matched audit outputs.")
        d["basin_id"] = pd.to_numeric(
            d["basin_id"], errors="raise"
        ).astype("int64")

    hist_seg["block_start"] = pd.to_datetime(hist_seg["block_start"])
    hist_seg["block_end"] = pd.to_datetime(hist_seg["block_end"])
    recent_seg["recent_block_start"] = pd.to_datetime(
        recent_seg["recent_block_start"]
    )
    recent_seg["recent_block_end"] = pd.to_datetime(
        recent_seg["recent_block_end"]
    )

    pb = detect_basin_column(physical)
    physical["basin_id"] = pd.to_numeric(
        physical[pb], errors="raise"
    ).astype("int64")
    physical = canonical_time(physical)

    if wavelet is not None:
        wb = detect_basin_column(wavelet)
        wavelet["basin_id"] = pd.to_numeric(
            wavelet[wb], errors="raise"
        ).astype("int64")
        wavelet = canonical_time(wavelet)

    print(f"✅ Strict latent migration candidates: {len(strict):,}")
    print(f"✅ Historical matched latent segments: {len(hist_seg):,}")
    print(f"✅ Recent matched basin segments:      {len(recent_seg):,}")
    print(f"✅ Physical monthly rows:              {len(physical):,}")
    print(
        f"✅ Wavelet monthly rows:               "
        f"{len(wavelet):,}" if wavelet is not None
        else "⚠️ Wavelet table unavailable"
    )

    return strict, hist_seg, recent_seg, physical, wavelet


# =============================================================================
# DISCOVER PHYSICAL INPUTS
# =============================================================================

def discover_physical_columns(physical, wavelet):
    monthly = {}
    long = {}
    seasonal = {}
    rows = []

    for key, opts in MONTHLY_VARIABLE_OPTIONS.items():
        col = choose_column(physical.columns, opts)
        monthly[key] = col
        rows.append({
            "feature_family": "monthly",
            "concept": key,
            "selected_column": col,
            "available": col is not None,
        })

    for key, opts in WAVELET_LONG_OPTIONS.items():
        col = (
            choose_column(wavelet.columns, opts)
            if wavelet is not None
            else None
        )
        long[key] = col
        rows.append({
            "feature_family": "wavelet_long",
            "concept": key,
            "selected_column": col,
            "available": col is not None,
        })

    for key, opts in WAVELET_SEASONAL_OPTIONS.items():
        col = (
            choose_column(wavelet.columns, opts)
            if wavelet is not None
            else None
        )
        seasonal[key] = col
        rows.append({
            "feature_family": "wavelet_seasonal_rms",
            "concept": key,
            "selected_column": col,
            "available": col is not None,
        })

    if monthly["TWSA"] is None or monthly["P"] is None or monthly["SOIL4"] is None:
        raise ValueError(
            "Need at least TWSA, precipitation, and deep-soil-moisture "
            "monthly physical columns for final validation."
        )

    return monthly, long, seasonal, pd.DataFrame(rows)


# =============================================================================
# EXTRACT A DURATION-MATCHED PHYSICAL FINGERPRINT
# =============================================================================

def extract_fingerprint(
    basin_id,
    start,
    end,
    physical_by_basin,
    wavelet_by_basin,
    monthly_cols,
    long_cols,
    seasonal_cols,
):
    p = physical_by_basin.get(int(basin_id))
    if p is None:
        return None

    g = p[(p["time"] >= start) & (p["time"] <= end)].copy()
    if g.empty:
        return None

    row = {
        "basin_id": int(basin_id),
        "segment_start": pd.Timestamp(start),
        "segment_end": pd.Timestamp(end),
        "n_physical_months": len(g),
    }

    # Monthly means + temporal slopes for all five core concepts.
    for concept, col in monthly_cols.items():
        if col is None:
            continue
        x = pd.to_numeric(g[col], errors="coerce")
        row[f"{concept}_mean"] = float(x.mean())
        row[f"{concept}_slope"] = linear_slope(x.to_numpy(dtype=float))

        # Extremes for TWSA, precipitation, deep soil moisture only.
        if concept in {"TWSA", "P", "SOIL4"}:
            valid = x.dropna()
            if len(valid):
                row[f"{concept}_frac_lt_m1"] = float(
                    (valid < -1.0).mean()
                )
                row[f"{concept}_frac_gt_p1"] = float(
                    (valid > 1.0).mean()
                )
            else:
                row[f"{concept}_frac_lt_m1"] = np.nan
                row[f"{concept}_frac_gt_p1"] = np.nan

    w = (
        wavelet_by_basin.get(int(basin_id))
        if wavelet_by_basin is not None
        else None
    )

    if w is not None:
        wg = w[(w["time"] >= start) & (w["time"] <= end)].copy()

        for concept, col in long_cols.items():
            if col is None or col not in wg.columns:
                continue
            x = pd.to_numeric(wg[col], errors="coerce")
            row[f"{concept}_long_mean"] = float(x.mean())

        for concept, col in seasonal_cols.items():
            if col is None or col not in wg.columns:
                continue
            x = pd.to_numeric(wg[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x)
            row[f"{concept}_seasonal_rms"] = (
                float(np.sqrt(np.mean(x[ok] ** 2)))
                if ok.any()
                else np.nan
            )

    return row


def build_physical_segments(
    hist_seg,
    recent_seg,
    strict,
    physical,
    wavelet,
    monthly_cols,
    long_cols,
    seasonal_cols,
):
    # Need historical segments only from source/target classes represented by
    # strict candidates.
    relevant_classes = sorted(
        set(strict["source_KG"].astype(str))
        | set(strict["best_supported_validated_target"].astype(str))
    )

    hist = hist_seg[
        hist_seg["KG_class"].astype(str).isin(relevant_classes)
    ].copy()

    # Recent: only strict candidate basins.
    recent = recent_seg[
        recent_seg["basin_id"].isin(set(strict["basin_id"]))
    ].copy()

    physical_by_basin = {
        int(b): g.sort_values("time").copy()
        for b, g in physical.groupby("basin_id")
    }
    wavelet_by_basin = (
        {
            int(b): g.sort_values("time").copy()
            for b, g in wavelet.groupby("basin_id")
        }
        if wavelet is not None
        else None
    )

    hrows = []
    for _, r in hist.iterrows():
        fp = extract_fingerprint(
            r["basin_id"],
            r["block_start"],
            r["block_end"],
            physical_by_basin,
            wavelet_by_basin,
            monthly_cols,
            long_cols,
            seasonal_cols,
        )
        if fp is None:
            continue
        fp["KG_class"] = str(r["KG_class"])
        fp["KG_dom_frac"] = float(r["KG_dom_frac"])
        fp["run_id"] = int(r["run_id"])
        fp["block_id"] = int(r["block_id"])
        hrows.append(fp)

    rrows = []
    for _, r in recent.iterrows():
        fp = extract_fingerprint(
            r["basin_id"],
            r["recent_block_start"],
            r["recent_block_end"],
            physical_by_basin,
            wavelet_by_basin,
            monthly_cols,
            long_cols,
            seasonal_cols,
        )
        if fp is None:
            continue
        fp["source_KG"] = str(r["source_KG"])
        fp["KG_dom_frac"] = float(r["KG_dom_frac"])
        rrows.append(fp)

    h = pd.DataFrame(hrows)
    rr = pd.DataFrame(rrows)

    if h.empty or rr.empty:
        raise ValueError(
            "Could not construct duration-matched physical fingerprints."
        )

    return h, rr


# =============================================================================
# FEATURE MATRIX
# =============================================================================

def choose_feature_columns(hist_physical):
    meta = {
        "basin_id",
        "segment_start",
        "segment_end",
        "n_physical_months",
        "KG_class",
        "KG_dom_frac",
        "run_id",
        "block_id",
    }

    cols = [
        c for c in hist_physical.columns
        if c not in meta
        and pd.api.types.is_numeric_dtype(hist_physical[c])
    ]

    # Remove columns with too much missingness or virtually no historical
    # variation.
    keep = []
    for c in cols:
        x = pd.to_numeric(hist_physical[c], errors="coerce")
        if x.notna().mean() < 0.95:
            continue
        if float(x.std(ddof=0)) < 1e-8:
            continue
        keep.append(c)

    if len(keep) < 8:
        raise ValueError(
            f"Too few robust physical fingerprint features: {keep}"
        )

    return keep


# =============================================================================
# HISTORICAL STANDARDIZATION + SUPPORT
# =============================================================================

def standardize_physical(hist_phy, recent_phy, feature_cols):
    scaler = StandardScaler()
    H = scaler.fit_transform(
        hist_phy[feature_cols].to_numpy(dtype=float)
    )
    R = scaler.transform(
        recent_phy[feature_cols].to_numpy(dtype=float)
    )

    sfeatures = [f"sf{i}" for i in range(len(feature_cols))]

    h = hist_phy.copy()
    r = recent_phy.copy()
    h[sfeatures] = H
    r[sfeatures] = R

    mapping = pd.DataFrame({
        "standardized_feature": sfeatures,
        "physical_feature": feature_cols,
    })

    return h, r, sfeatures, scaler, mapping


def calibrate_class_support(hist_phy, sfeatures):
    refs = {}
    rows = []

    for cls, g in hist_phy.groupby("KG_class"):
        X = g[sfeatures].to_numpy(dtype=float)
        basin_ids = g["basin_id"].to_numpy(dtype=np.int64)

        loo = []
        for i in range(len(g)):
            # Use other basins only.
            ref = X[basin_ids != basin_ids[i]]
            if len(ref) == 0:
                continue
            loo.append(
                mean_k_nearest_distance(
                    X[i], ref, K_NEIGHBORS
                )
            )

        loo = np.asarray(loo, dtype=float)
        if len(loo) == 0:
            continue

        thresholds = {
            q: float(np.quantile(loo, q))
            for q in SUPPORT_QUANTILES
        }

        refs[str(cls)] = {
            "X": X,
            "basin_ids": basin_ids,
            "centroid": X.mean(axis=0),
            "thresholds": thresholds,
            "n_basins": g["basin_id"].nunique(),
            "n_segments": len(g),
        }

        row = {
            "KG_class": str(cls),
            "n_historical_basins": g["basin_id"].nunique(),
            "n_historical_segments": len(g),
            "loo_other_basin_knn_median": float(np.median(loo)),
        }
        for q in SUPPORT_QUANTILES:
            row[f"support_q{int(q*100)}"] = thresholds[q]
        rows.append(row)

    return refs, pd.DataFrame(rows)


# =============================================================================
# HISTORICAL PHYSICAL PAIR SEPARABILITY
# =============================================================================

def pairwise_physical_auc(hist_phy, strict, sfeatures):
    pathways = (
        strict[
            ["source_KG", "best_supported_validated_target"]
        ]
        .drop_duplicates()
    )

    rows = []

    for _, p in pathways.iterrows():
        source = str(p["source_KG"])
        target = str(p["best_supported_validated_target"])

        d = hist_phy[
            hist_phy["KG_class"].astype(str).isin([source, target])
        ].copy()

        if d["KG_class"].nunique() != 2:
            continue

        n_source_basins = d.loc[
            d["KG_class"].astype(str) == source,
            "basin_id",
        ].nunique()
        n_target_basins = d.loc[
            d["KG_class"].astype(str) == target,
            "basin_id",
        ].nunique()

        n_splits = int(
            min(
                MAX_CV_SPLITS,
                n_source_basins,
                n_target_basins,
            )
        )
        if n_splits < 2:
            continue

        X = d[sfeatures].to_numpy(dtype=float)
        y = (
            d["KG_class"].astype(str).to_numpy() == target
        ).astype(int)
        groups = d["basin_id"].to_numpy()

        if HAVE_STRATIFIED_GROUP:
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_SEED,
            )
        else:
            cv = GroupKFold(n_splits=n_splits)

        try:
            proba = cross_val_predict(
                make_binary_probe(),
                X,
                y,
                groups=groups,
                cv=cv,
                method="predict_proba",
                n_jobs=None,
            )[:, 1]
            auc = float(roc_auc_score(y, proba))
        except Exception as e:
            print(
                f"⚠️ Physical pair CV failed for {source}->{target}: {e}"
            )
            auc = np.nan

        rows.append({
            "source_KG": source,
            "target_KG": target,
            "n_source_basins": n_source_basins,
            "n_target_basins": n_target_basins,
            "n_segments": len(d),
            "n_cv_splits": n_splits,
            "physical_pair_cv_auc": auc,
        })

    return pd.DataFrame(rows)


# =============================================================================
# CANDIDATE SOURCE -> RECENT -> TARGET VALIDATION
# =============================================================================

def evaluate_candidates(
    strict,
    hist_phy,
    recent_phy,
    refs,
    pair_auc,
    sfeatures,
    support_q,
):
    auc_lut = {
        (str(r["source_KG"]), str(r["target_KG"])):
        float(r["physical_pair_cv_auc"])
        for _, r in pair_auc.iterrows()
    }
    # symmetric lookup
    auc_lut.update({
        (b, a): v
        for (a, b), v in list(auc_lut.items())
    })

    recent_lut = {
        int(r["basin_id"]): r
        for _, r in recent_phy.iterrows()
    }

    rows = []

    for _, c in strict.iterrows():
        basin_id = int(c["basin_id"])
        source = str(c["source_KG"])
        target = str(c["best_supported_validated_target"])

        if source not in refs or target not in refs:
            continue
        if basin_id not in recent_lut:
            continue

        own = hist_phy[
            (hist_phy["basin_id"] == basin_id)
            & (hist_phy["KG_class"].astype(str) == source)
        ]
        if own.empty:
            continue

        h = own[sfeatures].mean(axis=0).to_numpy(dtype=float)
        r = recent_lut[basin_id][sfeatures].to_numpy(dtype=float)

        src = refs[source]
        tgt = refs[target]

        d_h_src = float(np.linalg.norm(h - src["centroid"]))
        d_h_tgt = float(np.linalg.norm(h - tgt["centroid"]))
        d_r_src = float(np.linalg.norm(r - src["centroid"]))
        d_r_tgt = float(np.linalg.norm(r - tgt["centroid"]))

        src_knn = mean_k_nearest_distance(
            r, src["X"], K_NEIGHBORS
        )
        tgt_knn = mean_k_nearest_distance(
            r, tgt["X"], K_NEIGHBORS
        )

        src_thr = src["thresholds"][support_q]
        tgt_thr = tgt["thresholds"][support_q]

        src_ratio = src_knn / src_thr if src_thr > 0 else np.nan
        tgt_ratio = tgt_knn / tgt_thr if tgt_thr > 0 else np.nan

        h_axis = centroid_axis_position(
            h, src["centroid"], tgt["centroid"]
        )
        r_axis = centroid_axis_position(
            r, src["centroid"], tgt["centroid"]
        )

        auc = auc_lut.get((source, target), np.nan)

        moved_closer_to_target = bool(d_r_tgt < d_h_tgt)
        moved_away_from_source = bool(d_r_src > d_h_src)
        axis_moved_toward_target = bool(
            np.isfinite(h_axis)
            and np.isfinite(r_axis)
            and r_axis > h_axis
        )
        crossed_midpoint = bool(
            np.isfinite(h_axis)
            and np.isfinite(r_axis)
            and h_axis < 0.5
            and r_axis >= 0.5
        )
        source_inside = bool(np.isfinite(src_ratio) and src_ratio <= 1.0)
        target_inside = bool(np.isfinite(tgt_ratio) and tgt_ratio <= 1.0)

        pair_distinguishable = bool(
            np.isfinite(auc)
            and auc >= PHYSICAL_PAIR_AUC_SCREEN
        )

        if not pair_distinguishable:
            outcome = "physical_pair_not_distinguishable"
        elif (
            (not source_inside)
            and target_inside
            and moved_closer_to_target
            and moved_away_from_source
            and axis_moved_toward_target
            and crossed_midpoint
        ):
            outcome = "confirmed_physical_convergence"
        elif source_inside and target_inside:
            outcome = "physical_support_overlap"
        elif (
            moved_closer_to_target
            and axis_moved_toward_target
        ):
            outcome = "partial_physical_convergence"
        elif (not source_inside) and (not target_inside):
            outcome = "physically_novel_beyond_source_and_target"
        else:
            outcome = "latent_migration_not_physically_supported"

        rows.append({
            "basin_id": basin_id,
            "source_KG": source,
            "target_KG": target,
            "KG_dom_frac": float(c["KG_dom_frac"]),
            "latent_source_support_ratio": float(
                c["source_support_ratio"]
            ),
            "latent_target_support_ratio": float(
                c["target_support_ratio"]
            ),
            "latent_pairwise_auc": float(
                c["source_target_pairwise_auc"]
            ),
            "latent_pair_axis_delta": float(
                c["pair_axis_delta_toward_target"]
            ),
            "physical_pair_cv_auc": auc,
            "historical_distance_to_source_centroid": d_h_src,
            "historical_distance_to_target_centroid": d_h_tgt,
            "recent_distance_to_source_centroid": d_r_src,
            "recent_distance_to_target_centroid": d_r_tgt,
            "recent_minus_historical_target_distance": (
                d_r_tgt - d_h_tgt
            ),
            "recent_minus_historical_source_distance": (
                d_r_src - d_h_src
            ),
            "physical_source_support_ratio": src_ratio,
            "physical_target_support_ratio": tgt_ratio,
            "historical_physical_axis_position": h_axis,
            "recent_physical_axis_position": r_axis,
            "physical_axis_delta_toward_target": (
                r_axis - h_axis
                if np.isfinite(h_axis) and np.isfinite(r_axis)
                else np.nan
            ),
            "moved_closer_to_target": moved_closer_to_target,
            "moved_away_from_source": moved_away_from_source,
            "axis_moved_toward_target": axis_moved_toward_target,
            "crossed_physical_midpoint": crossed_midpoint,
            "source_inside_physical_support": source_inside,
            "target_inside_physical_support": target_inside,
            "final_physical_validation_outcome": outcome,
        })

    return pd.DataFrame(rows)


# =============================================================================
# SENSITIVITY
# =============================================================================

def support_sensitivity(
    strict,
    hist_phy,
    recent_phy,
    refs,
    pair_auc,
    sfeatures,
):
    rows = []

    for q in SUPPORT_QUANTILES:
        d = evaluate_candidates(
            strict,
            hist_phy,
            recent_phy,
            refs,
            pair_auc,
            sfeatures,
            q,
        )
        counts = d["final_physical_validation_outcome"].value_counts()

        row = {
            "support_quantile": q,
            "n_candidates": len(d),
        }
        for outcome in [
            "confirmed_physical_convergence",
            "physical_support_overlap",
            "partial_physical_convergence",
            "physically_novel_beyond_source_and_target",
            "latent_migration_not_physically_supported",
            "physical_pair_not_distinguishable",
        ]:
            row[outcome] = int(counts.get(outcome, 0))
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# FIGURES
# =============================================================================

def plot_axis_changes(final):
    if final.empty:
        return

    d = final.sort_values(
        "physical_axis_delta_toward_target",
        ascending=False,
    ).reset_index(drop=True)

    y = np.arange(len(d))

    plt.figure(figsize=(11, max(7, 0.34 * len(d))))
    plt.scatter(
        d["historical_physical_axis_position"],
        y,
        s=34,
        label="Candidate historical",
    )
    plt.scatter(
        d["recent_physical_axis_position"],
        y,
        s=34,
        label="Candidate recent",
    )

    for i, r in d.iterrows():
        plt.plot(
            [
                r["historical_physical_axis_position"],
                r["recent_physical_axis_position"],
            ],
            [i, i],
            linewidth=1,
        )

    labels = [
        f"{int(r.basin_id)}  {r.source_KG}→{r.target_KG}"
        for _, r in d.iterrows()
    ]
    plt.yticks(y, labels, fontsize=8)
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.axvline(1, linestyle=":", linewidth=1)
    plt.axvline(0.5, linestyle="--", linewidth=1)
    plt.xlabel(
        "Physical source→target centroid-axis position "
        "(source centroid=0, target centroid=1)"
    )
    plt.title(
        "Duration-matched physical movement of strict latent migration candidates"
    )
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "physical_source_recent_target_axis.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_target_distance_change(final):
    if final.empty:
        return

    d = final.sort_values(
        "recent_minus_historical_target_distance",
        ascending=True,
    ).copy()

    labels = [
        f"{int(r.basin_id)} {r.source_KG}→{r.target_KG}"
        for _, r in d.iterrows()
    ]

    plt.figure(figsize=(11, max(7, 0.34 * len(d))))
    y = np.arange(len(d))
    plt.barh(
        y,
        d["recent_minus_historical_target_distance"],
    )
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel(
        "Recent minus historical distance to target physical centroid\n"
        "(negative = convergence toward target)"
    )
    plt.title("Physical convergence toward proposed target signatures")
    plt.tight_layout()

    out = FIG_DIR / "target_physical_distance_change.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# =============================================================================
# CONSOLE
# =============================================================================

def print_console(
    feature_map,
    class_support,
    pair_auc,
    final,
    sensitivity,
):
    print("\n" + "=" * 108)
    print("FINAL SOURCE → RECENT → TARGET PHYSICAL-FINGERPRINT VALIDATION")
    print("=" * 108)

    print("\nA) Physical fingerprint features")
    print(feature_map.to_string(index=False))

    print("\nB) Historical physical class support")
    print(
        class_support.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nC) Historical source-target PHYSICAL separability")
    print(
        pair_auc.sort_values(
            "physical_pair_cv_auc",
            ascending=False,
        ).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nD) Final candidate validation")
    cols = [
        "basin_id",
        "source_KG",
        "target_KG",
        "physical_pair_cv_auc",
        "historical_physical_axis_position",
        "recent_physical_axis_position",
        "physical_axis_delta_toward_target",
        "recent_minus_historical_target_distance",
        "physical_source_support_ratio",
        "physical_target_support_ratio",
        "final_physical_validation_outcome",
    ]
    print(
        final.sort_values(
            [
                "final_physical_validation_outcome",
                "physical_pair_cv_auc",
            ],
            ascending=[True, False],
        )[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nE) Primary outcome counts")
    counts = final[
        "final_physical_validation_outcome"
    ].value_counts()
    summary = pd.DataFrame({
        "outcome": counts.index,
        "n_candidates": counts.values,
    })
    summary["fraction"] = summary["n_candidates"] / len(final)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nF) 90/95/99% physical-support sensitivity")
    print(
        sensitivity.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nFinal interpretation rules:")
    print(
        "- 'confirmed_physical_convergence' means the recent basin fingerprint "
        "leaves source support, enters target support, moves closer to the "
        "target centroid, and crosses the physical source→target midpoint."
    )
    print(
        "- 'partial_physical_convergence' means the physical trajectory moves "
        "toward the target but does not satisfy the full support/crossing test."
    )
    print(
        "- 'physically_novel_beyond_source_and_target' means the recent "
        "fingerprint is outside both historical physical supports."
    )
    print(
        "- This validation uses the same GRACE/ERA5 variables that underpin "
        "the learned representation, so it is physical-consistency evidence, "
        "not an independent formal Köppen reclassification test."
    )
    print(
        "- 2-m temperature is not part of this validation. Formal "
        "temperature-defined Köppen change remains a separate question."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Final source-recent-target physical validation ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    strict, hist_seg, recent_seg, physical, wavelet = load_inputs()

    monthly_cols, long_cols, seasonal_cols, availability = (
        discover_physical_columns(physical, wavelet)
    )
    availability.to_csv(AVAILABILITY_OUT, index=False)
    print(f"✅ Saved: {AVAILABILITY_OUT}")

    hist_phy, recent_phy = build_physical_segments(
        hist_seg,
        recent_seg,
        strict,
        physical,
        wavelet,
        monthly_cols,
        long_cols,
        seasonal_cols,
    )

    feature_cols = choose_feature_columns(hist_phy)

    hist_phy, recent_phy, sfeatures, scaler, feature_map = (
        standardize_physical(
            hist_phy,
            recent_phy,
            feature_cols,
        )
    )

    hist_phy.to_csv(HIST_PHYSICAL_SEGMENTS_OUT, index=False)
    recent_phy.to_csv(RECENT_PHYSICAL_OUT, index=False)
    print(f"✅ Saved: {HIST_PHYSICAL_SEGMENTS_OUT}")
    print(f"✅ Saved: {RECENT_PHYSICAL_OUT}")

    class_refs, class_support = calibrate_class_support(
        hist_phy,
        sfeatures,
    )
    class_support.to_csv(CLASS_SUPPORT_OUT, index=False)
    print(f"✅ Saved: {CLASS_SUPPORT_OUT}")

    pair_auc = pairwise_physical_auc(
        hist_phy,
        strict,
        sfeatures,
    )
    pair_auc.to_csv(PAIR_AUC_OUT, index=False)
    print(f"✅ Saved: {PAIR_AUC_OUT}")

    final = evaluate_candidates(
        strict,
        hist_phy,
        recent_phy,
        class_refs,
        pair_auc,
        sfeatures,
        PRIMARY_SUPPORT_QUANTILE,
    )
    final.to_csv(FINAL_OUT, index=False)
    print(f"✅ Saved: {FINAL_OUT}")

    counts = final[
        "final_physical_validation_outcome"
    ].value_counts()
    summary = pd.DataFrame({
        "outcome": counts.index,
        "n_candidates": counts.values,
    })
    summary["fraction"] = summary["n_candidates"] / len(final)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"✅ Saved: {SUMMARY_OUT}")

    sensitivity = support_sensitivity(
        strict,
        hist_phy,
        recent_phy,
        class_refs,
        pair_auc,
        sfeatures,
    )
    sensitivity.to_csv(SENSITIVITY_OUT, index=False)
    print(f"✅ Saved: {SENSITIVITY_OUT}")

    qc = pd.DataFrame([
        {
            "check": "strict_latent_candidates_input",
            "value": len(strict),
        },
        {
            "check": "historical_physical_segments",
            "value": len(hist_phy),
        },
        {
            "check": "recent_candidate_physical_segments",
            "value": len(recent_phy),
        },
        {
            "check": "physical_fingerprint_features",
            "value": len(feature_cols),
        },
        {
            "check": "historical_physical_pair_tests",
            "value": len(pair_auc),
        },
        {
            "check": "final_candidates_evaluated",
            "value": len(final),
        },
        {
            "check": "confirmed_physical_convergence_q95",
            "value": int(
                (
                    final["final_physical_validation_outcome"]
                    == "confirmed_physical_convergence"
                ).sum()
            ),
        },
    ])
    qc.to_csv(QC_OUT, index=False)
    print(f"✅ Saved: {QC_OUT}")

    config = {
        "autoencoder_retrained": False,
        "umap_used_for_quantitative_distance": False,
        "duration_matching": (
            "physical fingerprints use the exact historical/recent block "
            "start and end dates from the 38-window duration-matched audit"
        ),
        "physical_variables": monthly_cols,
        "wavelet_long_variables": long_cols,
        "wavelet_seasonal_variables": seasonal_cols,
        "n_physical_features": len(feature_cols),
        "physical_pair_auc_screen": PHYSICAL_PAIR_AUC_SCREEN,
        "support_quantiles": SUPPORT_QUANTILES,
        "primary_support_quantile": PRIMARY_SUPPORT_QUANTILE,
        "k_neighbors": K_NEIGHBORS,
        "formal_koppen_reclassification_claim": False,
    }
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved: {CONFIG_OUT}")

    plot_axis_changes(final)
    plot_target_distance_change(final)

    print_console(
        feature_map,
        class_support,
        pair_auc,
        final,
        sensitivity,
    )

    print("\n--- Done ---")
    print(
        "STOP here. This is the final internal physical-consistency validation "
        "for the Köppen-associated migration branch."
    )


if __name__ == "__main__":
    main()

# auditDfbDfcLatentTransition.py
#
# Dfb–Dfc latent-transition audit
#
# Goal
# ----
# Test whether the EXISTING 16-D hydrological representation contains a
# historical Dfb-vs-Dfc organization and, only if it does, ask whether basins
# with a Dfc background move after 2020 toward hydrological behavior that is
# more similar to the historical Dfb reference.
#
# IMPORTANT INTERPRETATION
# ------------------------
# This does NOT claim that a basin's Köppen class has changed.
# The autoencoder represents GRACE/ERA5 hydrological dynamics and does not
# directly reproduce the formal Köppen classification criteria.  In particular,
# a Dfc -> Dfb climate-class change would require independent temperature
# validation later.
#
# No model retraining. No new KMeans. No UMAP distances.
# All similarity calculations are done in the standardized 16-D latent space.

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# =============================================================================
# PATHS
# =============================================================================

EMBEDDINGS_FILE = Path(
    "data/processed/embeddings_window_level_continuous_v3.parquet"
)
STATIC_SUMMARY_FILE = Path(
    "data/processed/koppen_basin_static_composition_summary_labeled.csv"
)
STATIC_LONG_FILE = Path(
    "data/processed/koppen_basin_class_fractions_long_labeled.csv"
)

OUT_DIR = Path("results/tables/dfb_dfc_latent_transition")
FIG_DIR = Path("results/figures/dfb_dfc_latent_transition")

WINDOW_SCORES_OUT = Path(
    "data/processed/dfb_dfc_latent_window_scores.parquet"
)
BASIN_HIST_OUT = OUT_DIR / "dfb_dfc_historical_basin_scores.csv"
VALIDATION_OUT = OUT_DIR / "dfb_dfc_historical_validation_sensitivity.csv"
SHIFT_OUT = OUT_DIR / "dfb_dfc_post2020_basin_shift.csv"
GROUP_SHIFT_OUT = OUT_DIR / "dfb_dfc_post2020_group_shift_sensitivity.csv"
REFERENCE_OUT = OUT_DIR / "dfb_dfc_reference_centroids.csv"
QC_OUT = OUT_DIR / "dfb_dfc_latent_audit_qc.csv"

# Historical reference is frozen at the end of 2020.
HIST_END = pd.Timestamp("2020-12-01")

# A "fully recent" 24-month window must BEGIN after 2020, so no pre-2021
# months enter its 24-month sequence.
RECENT_START = pd.Timestamp("2021-01-01")

# Sensitivity thresholds are used only to describe strong Dfb/Dfc anchors.
# We do NOT select one as the scientific truth.
ANCHOR_THRESHOLDS = [0.70, 0.80, 0.90]

# For the continuum validation, require the Dfb+Dfc pair to account for at
# least these fractions of the basin's valid Köppen composition.
PAIR_COVERAGE_THRESHOLDS = [0.50, 0.75, 0.90]

RANDOM_SEED = 42
BOOTSTRAP_REPEATS = 2000

DFB_CODE = 26
DFC_CODE = 27
DFB_ABBR = "Dfb"
DFC_ABBR = "Dfc"


# =============================================================================
# BASIC HELPERS
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
            f"Expected 16 latent dimensions z0..z15; found {len(cols)}: {cols}"
        )
    return cols


def spearman_corr(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan

    xr = x[ok].rank(method="average")
    yr = y[ok].rank(method="average")
    if xr.std(ddof=0) == 0 or yr.std(ddof=0) == 0:
        return np.nan

    return float(np.corrcoef(xr.to_numpy(), yr.to_numpy())[0, 1])


def bootstrap_mean_ci(values, n_boot=BOOTSTRAP_REPEATS):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(np.mean(v))
    rng = np.random.default_rng(RANDOM_SEED)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(v, size=len(v), replace=True)
        boots.append(np.mean(sample))

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return mean, float(lo), float(hi), len(v)


def bootstrap_median_ci(values, n_boot=BOOTSTRAP_REPEATS):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0

    med = float(np.median(v))
    rng = np.random.default_rng(RANDOM_SEED)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(v, size=len(v), replace=True)
        boots.append(np.median(sample))

    lo, hi = np.quantile(boots, [0.025, 0.975])
    return med, float(lo), float(hi), len(v)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_embeddings():
    e = pd.read_parquet(EMBEDDINGS_FILE).copy()
    bcol = detect_basin_column(e)
    e["basin_id"] = pd.to_numeric(e[bcol], errors="raise").astype("int64")

    required = {"start_time", "end_time"}
    missing = required - set(e.columns)
    if missing:
        raise ValueError(f"Embeddings missing columns: {sorted(missing)}")

    e["start_time"] = pd.to_datetime(e["start_time"])
    e["end_time"] = pd.to_datetime(e["end_time"])
    zcols = latent_columns(e)

    if e[zcols].isna().any().any():
        raise ValueError("NaN found in latent dimensions.")

    print(f"✅ Continuous embeddings: {len(e):,} windows")
    print(f"✅ Basins in embeddings:  {e['basin_id'].nunique():,}")
    print(f"✅ Latent dimensions:     {len(zcols)}")

    return e, zcols


def load_static():
    s = pd.read_csv(STATIC_SUMMARY_FILE)
    l = pd.read_csv(STATIC_LONG_FILE)

    s["HYBAS_ID"] = pd.to_numeric(s["HYBAS_ID"], errors="raise").astype("int64")
    l["HYBAS_ID"] = pd.to_numeric(l["HYBAS_ID"], errors="raise").astype("int64")
    l["KG_code"] = pd.to_numeric(l["KG_code"], errors="raise").astype(int)

    if s["HYBAS_ID"].duplicated().any():
        raise ValueError("Static summary has duplicate HYBAS_ID rows.")

    required_s = {
        "HYBAS_ID",
        "KG_valid_coverage",
        "KG_qc_pass_90pct_valid",
        "KG_dom_code",
        "KG_dom_abbr",
        "KG_dom_frac",
    }
    missing = required_s - set(s.columns)
    if missing:
        raise ValueError(f"Static summary missing: {sorted(missing)}")

    required_l = {"HYBAS_ID", "KG_code", "KG_abbr", "KG_fraction"}
    missing = required_l - set(l.columns)
    if missing:
        raise ValueError(f"Static long table missing: {sorted(missing)}")

    # Wide Dfb/Dfc fractions. Missing class means fraction 0.
    pair = (
        l[l["KG_code"].isin([DFB_CODE, DFC_CODE])]
        .pivot_table(
            index="HYBAS_ID",
            columns="KG_code",
            values="KG_fraction",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )

    if DFB_CODE not in pair.columns:
        pair[DFB_CODE] = 0.0
    if DFC_CODE not in pair.columns:
        pair[DFC_CODE] = 0.0

    pair = pair.rename(columns={
        DFB_CODE: "KG_Dfb_frac",
        DFC_CODE: "KG_Dfc_frac",
    })

    s = s.merge(pair, on="HYBAS_ID", how="left", validate="one_to_one")
    s[["KG_Dfb_frac", "KG_Dfc_frac"]] = (
        s[["KG_Dfb_frac", "KG_Dfc_frac"]].fillna(0.0)
    )
    s["KG_Dfb_Dfc_pair_frac"] = s["KG_Dfb_frac"] + s["KG_Dfc_frac"]

    denom = s["KG_Dfb_Dfc_pair_frac"]
    s["KG_Dfb_share_within_pair"] = np.where(
        denom > 0,
        s["KG_Dfb_frac"] / denom,
        np.nan,
    )

    print(f"✅ Static Köppen basins:  {len(s):,}")
    print(
        "✅ >=90% valid KG QC:    "
        f"{int(s['KG_qc_pass_90pct_valid'].astype(bool).sum()):,}"
    )
    print(
        "✅ Basins with Dfb/Dfc:  "
        f"{int((s['KG_Dfb_Dfc_pair_frac'] > 0).sum()):,}"
    )

    return s, l


# =============================================================================
# HISTORICAL LATENT STANDARDIZATION + BASIN MEANS
# =============================================================================

def historical_standardization(emb, zcols):
    hist = emb[emb["end_time"] <= HIST_END].copy()
    if hist.empty:
        raise ValueError("No historical windows ending <= 2020-12.")

    # Fit latent standardization ONLY on historical windows. This avoids using
    # post-2020 distribution information to define the reference geometry.
    scaler = StandardScaler()
    scaler.fit(hist[zcols].to_numpy(dtype=float))

    Z_all = scaler.transform(emb[zcols].to_numpy(dtype=float))

    out = emb.copy()
    sz = [f"sz{i}" for i in range(len(zcols))]
    out[sz] = Z_all

    hist = out[out["end_time"] <= HIST_END].copy()

    print(
        f"✅ Historical reference windows (end <= {HIST_END.date()}): "
        f"{len(hist):,}"
    )
    print(f"✅ Historical reference basins: {hist['basin_id'].nunique():,}")

    return out, hist, sz, scaler


def historical_basin_means(hist, szcols, static):
    means = (
        hist.groupby("basin_id")[szcols]
        .mean()
        .reset_index()
    )

    counts = (
        hist.groupby("basin_id")
        .size()
        .rename("n_historical_windows")
        .reset_index()
    )
    means = means.merge(counts, on="basin_id", how="left")

    means = means.merge(
        static,
        left_on="basin_id",
        right_on="HYBAS_ID",
        how="left",
        validate="one_to_one",
    )

    means = means[
        means["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
    ].copy()

    return means


# =============================================================================
# SOFT DFB / DFC HISTORICAL REFERENCES
# =============================================================================

def weighted_centroid(X, w):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(w) & (w > 0)

    if ok.sum() == 0:
        raise ValueError("No positive weights available for centroid.")

    return np.average(X[ok], axis=0, weights=w[ok])


def build_soft_centroids(basin_hist, szcols):
    relevant = basin_hist[
        basin_hist["KG_Dfb_Dfc_pair_frac"] > 0
    ].copy()

    X = relevant[szcols].to_numpy(dtype=float)
    w_dfb = relevant["KG_Dfb_frac"].to_numpy(dtype=float)
    w_dfc = relevant["KG_Dfc_frac"].to_numpy(dtype=float)

    c_dfb = weighted_centroid(X, w_dfb)
    c_dfc = weighted_centroid(X, w_dfc)

    sep = float(np.linalg.norm(c_dfb - c_dfc))
    if sep <= 1e-12:
        raise ValueError("Dfb and Dfc historical centroids are indistinguishable.")

    rows = []
    for name, code, centroid in [
        ("Dfb", DFB_CODE, c_dfb),
        ("Dfc", DFC_CODE, c_dfc),
    ]:
        row = {
            "reference": name,
            "KG_code": code,
        }
        for i, v in enumerate(centroid):
            row[f"sz{i}_centroid"] = float(v)
        rows.append(row)

    ref = pd.DataFrame(rows)
    ref["centroid_separation_euclidean"] = sep

    print(f"✅ Historical Dfb/Dfc centroid separation: {sep:.4f}")

    return c_dfb, c_dfc, ref, relevant


def axis_position(X, c_dfc, c_dfb):
    """
    Projection onto the Dfc -> Dfb centroid axis.

    position = 0 at the historical Dfc centroid
    position = 1 at the historical Dfb centroid

    Values can legitimately fall below 0 or above 1.
    """
    X = np.asarray(X, dtype=float)
    v = c_dfb - c_dfc
    denom = float(np.dot(v, v))
    return ((X - c_dfc) @ v) / denom


def distance_scores(X, c_dfc, c_dfb):
    X = np.asarray(X, dtype=float)
    d_dfc = np.linalg.norm(X - c_dfc, axis=1)
    d_dfb = np.linalg.norm(X - c_dfb, axis=1)

    denom = d_dfc + d_dfb
    rel = np.divide(
        d_dfc - d_dfb,
        denom,
        out=np.full_like(d_dfc, np.nan, dtype=float),
        where=denom > 0,
    )
    return d_dfc, d_dfb, rel


# =============================================================================
# LEAVE-ONE-BASIN-OUT HISTORICAL VALIDATION
# =============================================================================

def loo_historical_scores(relevant, szcols):
    """
    Validate historical Dfb/Dfc organization without scoring a basin against
    centroids that include its own latent mean.

    Centroid sums are updated analytically rather than refit N times.
    """
    X = relevant[szcols].to_numpy(dtype=float)
    wfb = relevant["KG_Dfb_frac"].to_numpy(dtype=float)
    wfc = relevant["KG_Dfc_frac"].to_numpy(dtype=float)

    sum_fb = (X * wfb[:, None]).sum(axis=0)
    sum_fc = (X * wfc[:, None]).sum(axis=0)
    den_fb = float(wfb.sum())
    den_fc = float(wfc.sum())

    rows = []

    for i in range(len(relevant)):
        den_fb_i = den_fb - wfb[i]
        den_fc_i = den_fc - wfc[i]

        if den_fb_i <= 0 or den_fc_i <= 0:
            continue

        cfb = (sum_fb - X[i] * wfb[i]) / den_fb_i
        cfc = (sum_fc - X[i] * wfc[i]) / den_fc_i

        pos = float(axis_position(X[i:i+1], cfc, cfb)[0])
        d_dfc, d_dfb, rel = distance_scores(X[i:i+1], cfc, cfb)

        r = relevant.iloc[i]
        rows.append({
            "basin_id": int(r["basin_id"]),
            "KG_Dfb_frac": float(r["KG_Dfb_frac"]),
            "KG_Dfc_frac": float(r["KG_Dfc_frac"]),
            "KG_Dfb_Dfc_pair_frac": float(r["KG_Dfb_Dfc_pair_frac"]),
            "KG_Dfb_share_within_pair": float(r["KG_Dfb_share_within_pair"]),
            "historical_axis_position_LOO": pos,
            "historical_distance_to_Dfc_LOO": float(d_dfc[0]),
            "historical_distance_to_Dfb_LOO": float(d_dfb[0]),
            "historical_Dfb_likeness_rel_LOO": float(rel[0]),
        })

    return pd.DataFrame(rows)


def validation_sensitivity(loo):
    rows = []

    # 1) Continuous composition-gradient validation.
    for cov in PAIR_COVERAGE_THRESHOLDS:
        sub = loo[loo["KG_Dfb_Dfc_pair_frac"] >= cov].copy()
        rho_axis = spearman_corr(
            sub["KG_Dfb_share_within_pair"],
            sub["historical_axis_position_LOO"],
        )
        rho_rel = spearman_corr(
            sub["KG_Dfb_share_within_pair"],
            sub["historical_Dfb_likeness_rel_LOO"],
        )

        rows.append({
            "test_type": "composition_gradient",
            "threshold": cov,
            "n_basins": len(sub),
            "metric": "Spearman(Dfb_share, axis_position_LOO)",
            "value": rho_axis,
        })
        rows.append({
            "test_type": "composition_gradient",
            "threshold": cov,
            "n_basins": len(sub),
            "metric": "Spearman(Dfb_share, relative_likeness_LOO)",
            "value": rho_rel,
        })

    # 2) Strong-anchor discrimination sensitivity.
    for t in ANCHOR_THRESHOLDS:
        sub = loo[
            (loo["KG_Dfb_frac"] >= t)
            | (loo["KG_Dfc_frac"] >= t)
        ].copy()

        # Label 1 = Dfb anchor, 0 = Dfc anchor.
        sub["label_Dfb"] = (sub["KG_Dfb_frac"] >= t).astype(int)

        # Exclude pathological overlap if any numerical/multi-class situation
        # makes both class fractions pass threshold.
        overlap = (
            (sub["KG_Dfb_frac"] >= t)
            & (sub["KG_Dfc_frac"] >= t)
        )
        sub = sub[~overlap].copy()

        if sub["label_Dfb"].nunique() == 2:
            auc_axis = roc_auc_score(
                sub["label_Dfb"],
                sub["historical_axis_position_LOO"],
            )
            auc_rel = roc_auc_score(
                sub["label_Dfb"],
                sub["historical_Dfb_likeness_rel_LOO"],
            )
            accuracy_axis = float(np.mean(
                (sub["historical_axis_position_LOO"] >= 0.5).astype(int)
                == sub["label_Dfb"]
            ))
        else:
            auc_axis = np.nan
            auc_rel = np.nan
            accuracy_axis = np.nan

        rows.append({
            "test_type": "strong_anchor_discrimination",
            "threshold": t,
            "n_basins": len(sub),
            "metric": "ROC_AUC_axis_position_LOO",
            "value": auc_axis,
        })
        rows.append({
            "test_type": "strong_anchor_discrimination",
            "threshold": t,
            "n_basins": len(sub),
            "metric": "ROC_AUC_relative_likeness_LOO",
            "value": auc_rel,
        })
        rows.append({
            "test_type": "strong_anchor_discrimination",
            "threshold": t,
            "n_basins": len(sub),
            "metric": "nearest_axis_centroid_accuracy_LOO",
            "value": accuracy_axis,
        })

    return pd.DataFrame(rows)


# =============================================================================
# SCORE ALL WINDOWS AGAINST FROZEN HISTORICAL REFERENCES
# =============================================================================

def score_windows(emb_scaled, szcols, c_dfc, c_dfb, static):
    X = emb_scaled[szcols].to_numpy(dtype=float)

    pos = axis_position(X, c_dfc, c_dfb)
    d_dfc, d_dfb, rel = distance_scores(X, c_dfc, c_dfb)

    keep = [
        c for c in [
            "sample_id",
            "basin_id",
            "start_time",
            "end_time",
            "split",
        ]
        if c in emb_scaled.columns
    ]

    out = emb_scaled[keep].copy()
    out["dfb_dfc_axis_position"] = pos
    out["distance_to_historical_Dfc"] = d_dfc
    out["distance_to_historical_Dfb"] = d_dfb
    out["Dfb_likeness_relative"] = rel

    static_keep = [
        "HYBAS_ID",
        "KG_valid_coverage",
        "KG_qc_pass_90pct_valid",
        "KG_dom_code",
        "KG_dom_abbr",
        "KG_dom_frac",
        "KG_Dfb_frac",
        "KG_Dfc_frac",
        "KG_Dfb_Dfc_pair_frac",
        "KG_Dfb_share_within_pair",
    ]

    out = out.merge(
        static[static_keep],
        left_on="basin_id",
        right_on="HYBAS_ID",
        how="left",
        validate="many_to_one",
    )

    return out


# =============================================================================
# PRE-vs-POST BASIN SHIFTS
# =============================================================================

def basin_shift_table(scores):
    good = scores[
        scores["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
        & (scores["KG_Dfb_Dfc_pair_frac"] > 0)
    ].copy()

    hist = good[good["end_time"] <= HIST_END].copy()
    recent = good[good["start_time"] >= RECENT_START].copy()

    h = (
        hist.groupby("basin_id")
        .agg(
            historical_axis_position=("dfb_dfc_axis_position", "mean"),
            historical_Dfb_likeness_rel=("Dfb_likeness_relative", "mean"),
            n_historical_windows=("dfb_dfc_axis_position", "size"),
        )
        .reset_index()
    )

    r = (
        recent.groupby("basin_id")
        .agg(
            recent_axis_position=("dfb_dfc_axis_position", "mean"),
            recent_Dfb_likeness_rel=("Dfb_likeness_relative", "mean"),
            n_recent_fully_post2020_windows=("dfb_dfc_axis_position", "size"),
        )
        .reset_index()
    )

    static_cols = [
        "basin_id",
        "KG_dom_code",
        "KG_dom_abbr",
        "KG_dom_frac",
        "KG_Dfb_frac",
        "KG_Dfc_frac",
        "KG_Dfb_Dfc_pair_frac",
        "KG_Dfb_share_within_pair",
    ]
    meta = (
        good[static_cols]
        .drop_duplicates("basin_id")
    )

    out = h.merge(r, on="basin_id", how="inner", validate="one_to_one")
    out = out.merge(meta, on="basin_id", how="left", validate="one_to_one")

    out["delta_axis_position_recent_minus_historical"] = (
        out["recent_axis_position"] - out["historical_axis_position"]
    )
    out["delta_Dfb_likeness_recent_minus_historical"] = (
        out["recent_Dfb_likeness_rel"]
        - out["historical_Dfb_likeness_rel"]
    )

    out["moved_toward_Dfb_axis"] = (
        out["delta_axis_position_recent_minus_historical"] > 0
    )

    return out


def group_shift_sensitivity(shifts):
    rows = []

    for t in ANCHOR_THRESHOLDS:
        groups = {
            "Dfc_anchor": shifts[shifts["KG_Dfc_frac"] >= t],
            "Dfb_anchor": shifts[shifts["KG_Dfb_frac"] >= t],
        }

        # Mixed transitional Dfb/Dfc background: pair dominates basin but neither
        # individual member exceeds the anchor threshold.
        mixed = shifts[
            (shifts["KG_Dfb_Dfc_pair_frac"] >= t)
            & (shifts["KG_Dfb_frac"] < t)
            & (shifts["KG_Dfc_frac"] < t)
        ]
        groups["Dfb_Dfc_mixed"] = mixed

        for group_name, g in groups.items():
            delta = g["delta_axis_position_recent_minus_historical"].dropna()
            mean, mean_lo, mean_hi, n = bootstrap_mean_ci(delta)
            med, med_lo, med_hi, _ = bootstrap_median_ci(delta)

            rows.append({
                "anchor_threshold": t,
                "group": group_name,
                "n_basins": n,
                "mean_delta_axis": mean,
                "mean_ci95_low": mean_lo,
                "mean_ci95_high": mean_hi,
                "median_delta_axis": med,
                "median_ci95_low": med_lo,
                "median_ci95_high": med_hi,
                "fraction_positive_delta": (
                    float((delta > 0).mean()) if n else np.nan
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# FIGURES
# =============================================================================

def plot_historical_gradient(loo):
    sub = loo[loo["KG_Dfb_Dfc_pair_frac"] >= 0.75].copy()
    if sub.empty:
        return

    plt.figure(figsize=(8.5, 6))
    plt.scatter(
        sub["KG_Dfb_share_within_pair"],
        sub["historical_axis_position_LOO"],
        s=22,
        alpha=0.55,
    )
    plt.axhline(0, linewidth=1, linestyle="--")
    plt.axhline(1, linewidth=1, linestyle="--")
    plt.xlabel("Static Dfb share within Dfb+Dfc composition")
    plt.ylabel(
        "Historical latent Dfc→Dfb axis position\n"
        "(leave-one-basin-out)"
    )
    plt.title(
        "Does the historical latent representation recover the Dfc–Dfb continuum?"
    )
    plt.tight_layout()

    out = FIG_DIR / "historical_Dfb_fraction_vs_latent_axis.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_group_time_series(scores, threshold=0.80):
    df = scores[
        scores["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
    ].copy()

    groups = []
    for label, mask in [
        ("Dfc anchor", df["KG_Dfc_frac"] >= threshold),
        ("Dfb anchor", df["KG_Dfb_frac"] >= threshold),
    ]:
        sub = df[mask].copy()
        if sub.empty:
            continue

        # Each basin contributes one monthly score, then take median across basins.
        monthly_basin = (
            sub.groupby(["end_time", "basin_id"])["dfb_dfc_axis_position"]
            .mean()
            .reset_index()
        )
        monthly = (
            monthly_basin.groupby("end_time")["dfb_dfc_axis_position"]
            .median()
            .reset_index()
        )
        monthly["group"] = label
        groups.append(monthly)

    if not groups:
        return

    plot = pd.concat(groups, ignore_index=True)

    plt.figure(figsize=(12, 5.5))
    for label, g in plot.groupby("group"):
        plt.plot(
            g["end_time"],
            g["dfb_dfc_axis_position"],
            linewidth=1.5,
            label=label,
        )

    plt.axvline(HIST_END, linewidth=1, linestyle="--")
    plt.axhline(0, linewidth=0.8, linestyle=":")
    plt.axhline(1, linewidth=0.8, linestyle=":")
    plt.xlabel("24-month window end")
    plt.ylabel("Median Dfc→Dfb latent-axis position")
    plt.title(
        f"Historical anchors through time (fraction ≥ {threshold:.0%})"
    )
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "Dfb_Dfc_anchor_axis_through_time.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_top_dfc_shifts(scores, shifts, threshold=0.80, n_top=8):
    cand = shifts[
        shifts["KG_Dfc_frac"] >= threshold
    ].sort_values(
        "delta_axis_position_recent_minus_historical",
        ascending=False,
    ).head(n_top)

    if cand.empty:
        return

    plt.figure(figsize=(12, 6.5))

    for basin_id in cand["basin_id"]:
        g = scores[scores["basin_id"] == basin_id].sort_values("end_time")
        plt.plot(
            g["end_time"],
            g["dfb_dfc_axis_position"],
            linewidth=1.1,
            alpha=0.8,
            label=str(int(basin_id)),
        )

    plt.axvline(HIST_END, linewidth=1, linestyle="--")
    plt.axhline(0, linewidth=0.8, linestyle=":")
    plt.axhline(1, linewidth=0.8, linestyle=":")
    plt.xlabel("24-month window end")
    plt.ylabel("Dfc→Dfb latent-axis position")
    plt.title(
        "Largest recent shifts toward the historical Dfb hydrological reference\n"
        f"among Dfc anchors (Dfc fraction ≥ {threshold:.0%})"
    )
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out = FIG_DIR / "top_Dfc_to_Dfb_like_candidate_trajectories.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# =============================================================================
# QC / INTERPRETATION
# =============================================================================

def build_qc(emb, hist, basin_hist, relevant, loo, scores, shifts):
    rows = [
        {"check": "continuous_embedding_windows", "value": len(emb)},
        {"check": "continuous_embedding_basins", "value": emb["basin_id"].nunique()},
        {"check": "historical_windows", "value": len(hist)},
        {"check": "historical_basins_with_static_qc", "value": len(basin_hist)},
        {"check": "historical_basins_with_Dfb_or_Dfc", "value": len(relevant)},
        {"check": "LOO_validation_basins", "value": len(loo)},
        {"check": "scored_windows", "value": len(scores)},
        {"check": "basins_with_pre_and_fully_post2020_scores", "value": len(shifts)},
        {
            "check": "recent_windows_min_start",
            "value": str(
                scores.loc[
                    scores["start_time"] >= RECENT_START,
                    "start_time"
                ].min()
            ),
        },
    ]
    return pd.DataFrame(rows)


def print_console(validation, group_shift, shifts, ref):
    print("\n" + "=" * 94)
    print("DFB–DFC LATENT TRANSITION AUDIT")
    print("=" * 94)

    print("\nA) Historical reference geometry")
    print(
        f"Dfc centroid = axis position 0; "
        f"Dfb centroid = axis position 1"
    )
    print(
        "Centroid separation in historical-standardized 16-D space: "
        f"{ref['centroid_separation_euclidean'].iloc[0]:.4f}"
    )

    print("\nB) Historical validation sensitivity")
    print(
        validation.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nC) Post-2020 group-shift sensitivity")
    print(
        group_shift.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nD) Top 15 Dfc-background basins shifting toward Dfb-like behavior")
    # Use the middle 80% threshold for the shortlist only.
    top = (
        shifts[shifts["KG_Dfc_frac"] >= 0.80]
        .sort_values(
            "delta_axis_position_recent_minus_historical",
            ascending=False,
        )
        .head(15)
    )

    cols = [
        "basin_id",
        "KG_Dfc_frac",
        "KG_Dfb_frac",
        "historical_axis_position",
        "recent_axis_position",
        "delta_axis_position_recent_minus_historical",
        "n_historical_windows",
        "n_recent_fully_post2020_windows",
    ]
    print(
        top[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nDecision rule:")
    print(
        "1. FIRST require evidence that pre-2021 Dfb/Dfc composition is "
        "recoverable from the latent geometry:"
    )
    print(
        "   - positive composition-gradient correlation, and"
    )
    print(
        "   - anchor ROC AUC meaningfully above 0.5 across sensitivity thresholds."
    )
    print(
        "2. ONLY if that passes, interpret positive post-2020 axis shifts in "
        "Dfc-background basins as"
    )
    print(
        "   'hydrological dynamics becoming more similar to the historical "
        "Dfb reference.'"
    )
    print(
        "3. Do NOT call this a Dfc→Dfb Köppen reclassification. Formal climate-"
        "class change requires"
    )
    print(
        "   independent temperature-based validation."
    )
    print(
        "4. Recent summaries use windows starting >= 2021-01, so every recent "
        "24-month sequence is fully post-2020."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Dfb–Dfc latent-transition audit ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    WINDOW_SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)

    emb, zcols = load_embeddings()
    static, static_long = load_static()

    emb_scaled, hist, szcols, scaler = historical_standardization(emb, zcols)
    basin_hist = historical_basin_means(hist, szcols, static)

    c_dfb, c_dfc, ref, relevant = build_soft_centroids(
        basin_hist,
        szcols,
    )
    ref.to_csv(REFERENCE_OUT, index=False)
    print(f"✅ Saved: {REFERENCE_OUT}")

    loo = loo_historical_scores(relevant, szcols)

    # Add the full-centroid historical basin score too for downstream inspection.
    Xhist = relevant[szcols].to_numpy(dtype=float)
    relevant = relevant.copy()
    relevant["historical_axis_position_fullref"] = axis_position(
        Xhist, c_dfc, c_dfb
    )
    d_dfc, d_dfb, rel = distance_scores(Xhist, c_dfc, c_dfb)
    relevant["historical_distance_to_Dfc_fullref"] = d_dfc
    relevant["historical_distance_to_Dfb_fullref"] = d_dfb
    relevant["historical_Dfb_likeness_rel_fullref"] = rel

    hist_out = relevant.merge(
        loo,
        on=[
            "basin_id",
            "KG_Dfb_frac",
            "KG_Dfc_frac",
            "KG_Dfb_Dfc_pair_frac",
            "KG_Dfb_share_within_pair",
        ],
        how="left",
        validate="one_to_one",
    )
    hist_out.to_csv(BASIN_HIST_OUT, index=False)
    print(f"✅ Saved: {BASIN_HIST_OUT}")

    validation = validation_sensitivity(loo)
    validation.to_csv(VALIDATION_OUT, index=False)
    print(f"✅ Saved: {VALIDATION_OUT}")

    scores = score_windows(
        emb_scaled,
        szcols,
        c_dfc,
        c_dfb,
        static,
    )
    scores.to_parquet(WINDOW_SCORES_OUT, index=False)
    print(f"✅ Saved: {WINDOW_SCORES_OUT}")

    shifts = basin_shift_table(scores)
    shifts.to_csv(SHIFT_OUT, index=False)
    print(f"✅ Saved: {SHIFT_OUT}")

    group_shift = group_shift_sensitivity(shifts)
    group_shift.to_csv(GROUP_SHIFT_OUT, index=False)
    print(f"✅ Saved: {GROUP_SHIFT_OUT}")

    qc = build_qc(
        emb,
        hist,
        basin_hist,
        relevant,
        loo,
        scores,
        shifts,
    )
    qc.to_csv(QC_OUT, index=False)
    print(f"✅ Saved: {QC_OUT}")

    plot_historical_gradient(loo)
    plot_group_time_series(scores, threshold=0.80)
    plot_top_dfc_shifts(scores, shifts, threshold=0.80, n_top=8)

    print_console(validation, group_shift, shifts, ref)

    print("\n--- Done ---")
    print(
        "STOP after this audit. Do not add temperature or redesign the "
        "autoencoder until the historical Dfb/Dfc validation is interpreted."
    )


if __name__ == "__main__":
    main()

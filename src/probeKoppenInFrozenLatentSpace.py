# probeKoppenInFrozenLatentSpace.py
#
# Global Köppen probe of the FROZEN 16-D hydrological representation.
#
# PURPOSE
# -------
# Test which aspects of Köppen-Geiger climate background are recoverable from
# the existing GRACE/ERA5 hydrological latent representation WITHOUT retraining
# the autoencoder and WITHOUT feeding Köppen labels into representation learning.
#
# Scientific logic:
#   GRACE/ERA5 -> frozen autoencoder -> 16-D latent representation
#                                      |
#                                      +-> post-hoc Köppen probe
#
# Köppen labels are used ONLY after representation learning as an external
# diagnostic. This is a standard "linear probe" design.
#
# IMPORTANT:
# - Quantitative analysis is in the 16-D latent space.
# - UMAP is created only for visualization.
# - One historical latent mean per basin is used for probing, avoiding
#   pseudo-replication from strongly overlapping 24-month windows.
# - Cross-validation is across BASINS.
# - Mixed Köppen basins are preserved in the source tables. Strong dominant
#   fractions are used only as sensitivity-defined anchor subsets for the
#   classification probe.
# - This script does NOT yet analyze post-2020 migration. It only establishes
#   which climate distinctions are actually supported by the frozen latent space.

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

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
STATIC_LONG_FILE = Path(
    "data/processed/koppen_basin_class_fractions_long_labeled.csv"
)

OUT_DIR = Path("results/tables/koppen_frozen_latent_probe")
FIG_DIR = Path("results/figures/koppen_frozen_latent_probe")

HIST_BASIN_OUT = OUT_DIR / "historical_basin_latent_means_with_koppen.csv"
FAMILY_CV_OUT = OUT_DIR / "family_probe_cv_sensitivity.csv"
CLASS_CV_OUT = OUT_DIR / "exact_class_probe_cv_sensitivity.csv"
PAIRWISE_OUT = OUT_DIR / "exact_class_pairwise_auc_anchor80.csv"
COUNTS_OUT = OUT_DIR / "probe_anchor_counts.csv"
QC_OUT = OUT_DIR / "koppen_frozen_latent_probe_qc.csv"
CONFIG_OUT = OUT_DIR / "koppen_frozen_latent_probe_config.json"

# Historical reference only.
HIST_END = pd.Timestamp("2020-12-01")

# Sensitivity anchors. These thresholds are NOT new climate classes.
ANCHOR_THRESHOLDS = [0.70, 0.80, 0.90]

# Minimum basins required for an exact class to enter the multiclass probe.
MIN_CLASS_BASINS = 10

# Pairwise matrix uses the middle threshold and slightly stricter support.
PAIRWISE_THRESHOLD = 0.80
PAIRWISE_MIN_PER_CLASS = 12

MAX_CV_SPLITS = 5
RANDOM_SEED = 42

# Small permutation diagnostic for each multiclass probe.
# This is intentionally modest for runtime; balanced accuracy already has a
# clear chance interpretation of ~1/K.
N_PERMUTATIONS = 20


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


def make_probe():
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


def family_from_abbr(abbr):
    if pd.isna(abbr):
        return pd.NA
    s = str(abbr).strip()
    if not s:
        return pd.NA
    first = s[0].upper()
    if first in {"A", "B", "C", "D", "E"}:
        return first
    return pd.NA


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


def permutation_balanced_accuracy(X, y, cv, n_perm=N_PERMUTATIONS):
    rng = np.random.default_rng(RANDOM_SEED)
    vals = []

    for _ in range(n_perm):
        yp = rng.permutation(np.asarray(y))
        pred = cross_val_predict(
            make_probe(),
            X,
            yp,
            cv=cv,
            method="predict",
            n_jobs=None,
        )
        vals.append(balanced_accuracy_score(yp, pred))

    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def evaluate_multiclass(df, feature_cols, label_col):
    d = df.dropna(subset=[label_col]).copy()
    y = d[label_col].astype(str).to_numpy()
    X = d[feature_cols].to_numpy(dtype=float)

    cv = adaptive_cv(y)
    if cv is None:
        return None, None, None

    model = make_probe()

    pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=None,
    )

    proba = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )

    classes = np.sort(np.unique(y))

    bal = balanced_accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro")

    auc = np.nan
    try:
        auc = roc_auc_score(
            y,
            proba,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        pass

    perm_mean, perm_sd = permutation_balanced_accuracy(X, y, cv)

    metrics = {
        "n_basins": len(d),
        "n_classes": len(classes),
        "n_cv_splits": cv.n_splits,
        "balanced_accuracy": float(bal),
        "macro_f1": float(macro_f1),
        "macro_ovr_roc_auc": float(auc) if np.isfinite(auc) else np.nan,
        "chance_balanced_accuracy_1_over_K": 1.0 / len(classes),
        "permutation_balanced_accuracy_mean": perm_mean,
        "permutation_balanced_accuracy_sd": perm_sd,
    }

    cm = confusion_matrix(y, pred, labels=classes, normalize="true")
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    pred_df = d[["basin_id", label_col]].copy()
    pred_df["predicted_label"] = pred

    return metrics, cm_df, pred_df


# =============================================================================
# LOAD + BUILD BASIN-LEVEL HISTORICAL REPRESENTATION
# =============================================================================

def load_inputs():
    emb = pd.read_parquet(EMBEDDINGS_FILE).copy()
    static = pd.read_csv(STATIC_SUMMARY_FILE)
    long = pd.read_csv(STATIC_LONG_FILE)

    bcol = detect_basin_column(emb)
    emb["basin_id"] = pd.to_numeric(
        emb[bcol], errors="raise"
    ).astype("int64")
    emb["start_time"] = pd.to_datetime(emb["start_time"])
    emb["end_time"] = pd.to_datetime(emb["end_time"])

    static["HYBAS_ID"] = pd.to_numeric(
        static["HYBAS_ID"], errors="raise"
    ).astype("int64")
    long["HYBAS_ID"] = pd.to_numeric(
        long["HYBAS_ID"], errors="raise"
    ).astype("int64")
    long["KG_code"] = pd.to_numeric(
        long["KG_code"], errors="raise"
    ).astype(int)

    zcols = latent_columns(emb)

    print(f"✅ Continuous embeddings: {len(emb):,}")
    print(f"✅ Embedding basins:      {emb['basin_id'].nunique():,}")
    print(f"✅ Static basin rows:     {len(static):,}")
    print(f"✅ Köppen long rows:      {len(long):,}")
    print(f"✅ Latent dimensions:     {len(zcols)}")

    return emb, static, long, zcols


def add_family_fractions(static, long):
    l = long.copy()
    l["KG_family"] = l["KG_abbr"].map(family_from_abbr)

    fam = (
        l.dropna(subset=["KG_family"])
        .groupby(["HYBAS_ID", "KG_family"], as_index=False)["KG_fraction"]
        .sum()
    )

    wide = (
        fam.pivot(
            index="HYBAS_ID",
            columns="KG_family",
            values="KG_fraction",
        )
        .fillna(0.0)
        .reset_index()
    )

    for f in ["A", "B", "C", "D", "E"]:
        if f not in wide.columns:
            wide[f] = 0.0

    wide = wide.rename(columns={
        "A": "KG_family_A_frac",
        "B": "KG_family_B_frac",
        "C": "KG_family_C_frac",
        "D": "KG_family_D_frac",
        "E": "KG_family_E_frac",
    })

    out = static.merge(
        wide,
        on="HYBAS_ID",
        how="left",
        validate="one_to_one",
    )

    fam_cols = [
        "KG_family_A_frac",
        "KG_family_B_frac",
        "KG_family_C_frac",
        "KG_family_D_frac",
        "KG_family_E_frac",
    ]
    out[fam_cols] = out[fam_cols].fillna(0.0)

    vals = out[fam_cols].to_numpy(dtype=float)
    idx = np.argmax(vals, axis=1)
    letters = np.array(["A", "B", "C", "D", "E"], dtype=object)

    out["KG_dom_family"] = letters[idx]
    out["KG_dom_family_frac"] = vals[np.arange(len(out)), idx]

    return out


def build_historical_basin_means(emb, static, zcols):
    hist = emb[emb["end_time"] <= HIST_END].copy()

    if hist.empty:
        raise ValueError("No historical windows ending <= 2020-12.")

    # One row per basin: mean of all historical latent windows.
    # This prevents thousands of overlapping windows from being treated as
    # statistically independent samples in the probe.
    means = (
        hist.groupby("basin_id")[zcols]
        .mean()
        .reset_index()
    )

    counts = (
        hist.groupby("basin_id")
        .agg(
            n_historical_windows=("end_time", "size"),
            first_historical_window_start=("start_time", "min"),
            last_historical_window_end=("end_time", "max"),
        )
        .reset_index()
    )

    means = means.merge(
        counts,
        on="basin_id",
        how="left",
        validate="one_to_one",
    )

    means = means.merge(
        static,
        left_on="basin_id",
        right_on="HYBAS_ID",
        how="left",
        validate="one_to_one",
    )

    means["has_static"] = means["HYBAS_ID"].notna()

    print(f"✅ Historical windows:     {len(hist):,}")
    print(f"✅ Historical basin means: {len(means):,}")

    return hist, means


# =============================================================================
# FAMILY PROBE
# =============================================================================

def run_family_probe(hist_basins, zcols):
    rows = []
    confusion_saved = False

    for t in ANCHOR_THRESHOLDS:
        sub = hist_basins[
            hist_basins["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
            & (hist_basins["KG_dom_family_frac"] >= t)
        ].copy()

        counts = sub["KG_dom_family"].value_counts()

        # Keep only families with enough examples for CV.
        eligible = counts[counts >= MAX_CV_SPLITS].index
        sub = sub[sub["KG_dom_family"].isin(eligible)].copy()

        result = evaluate_multiclass(sub, zcols, "KG_dom_family")
        if result[0] is None:
            continue

        metrics, cm, pred = result
        metrics["anchor_threshold"] = t
        metrics["eligible_labels"] = ";".join(sorted(cm.index.astype(str)))
        rows.append(metrics)

        if abs(t - 0.80) < 1e-9:
            cm.to_csv(OUT_DIR / "family_probe_confusion_matrix_anchor80.csv")
            pred.to_csv(
                OUT_DIR / "family_probe_predictions_anchor80.csv",
                index=False,
            )
            plot_confusion(
                cm,
                "Köppen family probe — historical basin means",
                FIG_DIR / "family_probe_confusion_matrix_anchor80.png",
            )
            confusion_saved = True

    return pd.DataFrame(rows), confusion_saved


# =============================================================================
# EXACT 30-CLASS PROBE
# =============================================================================

def run_exact_class_probe(hist_basins, zcols):
    rows = []

    for t in ANCHOR_THRESHOLDS:
        sub = hist_basins[
            hist_basins["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
            & (hist_basins["KG_dom_frac"] >= t)
        ].copy()

        counts = sub["KG_dom_abbr"].value_counts()
        eligible = counts[counts >= MIN_CLASS_BASINS].index
        sub = sub[sub["KG_dom_abbr"].isin(eligible)].copy()

        result = evaluate_multiclass(sub, zcols, "KG_dom_abbr")
        if result[0] is None:
            continue

        metrics, cm, pred = result
        metrics["anchor_threshold"] = t
        metrics["eligible_labels"] = ";".join(sorted(cm.index.astype(str)))
        rows.append(metrics)

        if abs(t - 0.80) < 1e-9:
            cm.to_csv(
                OUT_DIR / "exact_class_probe_confusion_matrix_anchor80.csv"
            )
            pred.to_csv(
                OUT_DIR / "exact_class_probe_predictions_anchor80.csv",
                index=False,
            )
            plot_confusion(
                cm,
                "Exact Köppen-class probe — historical basin means",
                FIG_DIR / "exact_class_probe_confusion_matrix_anchor80.png",
                fontsize=7,
            )

    return pd.DataFrame(rows)


# =============================================================================
# PAIRWISE EXACT-CLASS SEPARABILITY
# =============================================================================

def pairwise_auc(hist_basins, zcols):
    sub = hist_basins[
        hist_basins["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
        & (hist_basins["KG_dom_frac"] >= PAIRWISE_THRESHOLD)
    ].copy()

    counts = sub["KG_dom_abbr"].value_counts()
    eligible = sorted(
        counts[counts >= PAIRWISE_MIN_PER_CLASS].index.astype(str)
    )

    rows = []

    for i, a in enumerate(eligible):
        for b in eligible[i + 1:]:
            d = sub[sub["KG_dom_abbr"].isin([a, b])].copy()

            y = (d["KG_dom_abbr"].astype(str) == b).astype(int).to_numpy()
            X = d[zcols].to_numpy(dtype=float)

            cv = adaptive_cv(y)
            if cv is None:
                continue

            proba = cross_val_predict(
                make_probe(),
                X,
                y,
                cv=cv,
                method="predict_proba",
                n_jobs=None,
            )[:, 1]

            auc = roc_auc_score(y, proba)

            rows.append({
                "class_A": a,
                "class_B": b,
                "anchor_threshold": PAIRWISE_THRESHOLD,
                "n_A": int((d["KG_dom_abbr"].astype(str) == a).sum()),
                "n_B": int((d["KG_dom_abbr"].astype(str) == b).sum()),
                "n_cv_splits": cv.n_splits,
                "roc_auc": float(auc),
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        plot_pairwise_heatmap(
            out,
            eligible,
            FIG_DIR / "exact_class_pairwise_auc_anchor80.png",
        )

    return out


# =============================================================================
# ANCHOR COUNTS
# =============================================================================

def anchor_counts(hist_basins):
    rows = []

    for level, label_col, frac_col in [
        ("family", "KG_dom_family", "KG_dom_family_frac"),
        ("exact_class", "KG_dom_abbr", "KG_dom_frac"),
    ]:
        for t in ANCHOR_THRESHOLDS:
            sub = hist_basins[
                hist_basins["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
                & (hist_basins[frac_col] >= t)
            ]

            c = sub[label_col].value_counts(dropna=False)
            for label, n in c.items():
                rows.append({
                    "level": level,
                    "anchor_threshold": t,
                    "label": label,
                    "n_basins": int(n),
                })

    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION ONLY: HISTORICAL BASIN UMAP
# =============================================================================

def historical_basin_umap(hist_basins, zcols):
    """
    Visualization only. Quantitative probe metrics never use UMAP coordinates.
    """
    try:
        import umap
    except Exception as e:
        print(f"⚠️ UMAP unavailable; skipping visualization: {e}")
        return

    d = hist_basins[
        hist_basins["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
    ].copy()

    X = d[zcols].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.15,
        metric="euclidean",
        random_state=RANDOM_SEED,
    )
    U = reducer.fit_transform(X)

    d["u1"] = U[:, 0]
    d["u2"] = U[:, 1]

    d[
        [
            "basin_id",
            "u1",
            "u2",
            "KG_dom_family",
            "KG_dom_family_frac",
            "KG_dom_abbr",
            "KG_dom_frac",
        ]
    ].to_csv(
        OUT_DIR / "historical_basin_umap_coordinates.csv",
        index=False,
    )

    # Family view.
    plt.figure(figsize=(9, 7))
    for fam in ["A", "B", "C", "D", "E"]:
        g = d[d["KG_dom_family"] == fam]
        if len(g):
            plt.scatter(
                g["u1"],
                g["u2"],
                s=24,
                alpha=0.65,
                label=fam,
            )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(
        "Historical basin-level latent atlas colored by Köppen family\n"
        "(visualization only; probe statistics use the 16-D space)"
    )
    plt.legend(title="Köppen family")
    plt.tight_layout()
    out = FIG_DIR / "historical_basin_umap_by_KG_family.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")

    # D-family exact-class view because it contains the existing Dfb/Dfc test.
    dd = d[d["KG_dom_family"] == "D"].copy()
    if len(dd):
        plt.figure(figsize=(9, 7))
        for abbr in sorted(dd["KG_dom_abbr"].dropna().unique()):
            g = dd[dd["KG_dom_abbr"] == abbr]
            plt.scatter(
                g["u1"],
                g["u2"],
                s=26,
                alpha=0.70,
                label=abbr,
            )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(
            "Historical D-family basins in the latent atlas\n"
            "(visualization only)"
        )
        plt.legend(title="Dominant KG class", fontsize=8)
        plt.tight_layout()
        out = FIG_DIR / "historical_basin_umap_D_family_exact_classes.png"
        plt.savefig(out, dpi=240, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {out}")


# =============================================================================
# PLOTS
# =============================================================================

def plot_confusion(cm, title, out, fontsize=9):
    arr = cm.to_numpy(dtype=float)

    plt.figure(figsize=(8.5, 7.5))
    im = plt.imshow(arr, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, label="Row-normalized fraction")
    plt.xticks(
        np.arange(len(cm.columns)),
        cm.columns,
        rotation=90,
        fontsize=fontsize,
    )
    plt.yticks(
        np.arange(len(cm.index)),
        cm.index,
        fontsize=fontsize,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_pairwise_heatmap(pairwise, labels, out):
    mat = pd.DataFrame(
        np.nan,
        index=labels,
        columns=labels,
        dtype=float,
    )

    # Pandas/NumPy combinations with Copy-on-Write can expose mat.values as a
    # read-only view. Assign through pandas instead of np.fill_diagonal().
    for i in range(len(labels)):
        mat.iat[i, i] = 1.0

    for _, r in pairwise.iterrows():
        a = r["class_A"]
        b = r["class_B"]
        mat.loc[a, b] = r["roc_auc"]
        mat.loc[b, a] = r["roc_auc"]

    plt.figure(figsize=(10, 9))
    im = plt.imshow(
        mat.to_numpy(dtype=float, copy=True),
        vmin=0.5,
        vmax=1.0,
        aspect="auto",
    )
    plt.colorbar(im, label="Cross-validated ROC AUC")
    plt.xticks(
        np.arange(len(labels)),
        labels,
        rotation=90,
        fontsize=8,
    )
    plt.yticks(
        np.arange(len(labels)),
        labels,
        fontsize=8,
    )
    plt.title(
        "Historical exact-class separability in frozen 16-D latent space\n"
        f"(dominant class fraction ≥ {PAIRWISE_THRESHOLD:.0%})"
    )
    plt.tight_layout()
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# =============================================================================
# QC + CONSOLE
# =============================================================================

def build_qc(emb, hist, hist_basins, family_cv, class_cv, pairwise):
    return pd.DataFrame([
        {
            "check": "continuous_embedding_windows",
            "value": len(emb),
        },
        {
            "check": "continuous_embedding_basins",
            "value": emb["basin_id"].nunique(),
        },
        {
            "check": "historical_windows_end_le_2020_12",
            "value": len(hist),
        },
        {
            "check": "historical_basin_means",
            "value": len(hist_basins),
        },
        {
            "check": "historical_basin_means_with_static",
            "value": int(hist_basins["has_static"].sum()),
        },
        {
            "check": "historical_basin_means_KG_qc90",
            "value": int(
                hist_basins["KG_qc_pass_90pct_valid"]
                .fillna(False)
                .astype(bool)
                .sum()
            ),
        },
        {
            "check": "family_probe_sensitivity_rows",
            "value": len(family_cv),
        },
        {
            "check": "exact_class_probe_sensitivity_rows",
            "value": len(class_cv),
        },
        {
            "check": "pairwise_exact_class_tests",
            "value": len(pairwise),
        },
    ])


def print_summary(family_cv, class_cv, pairwise, counts):
    print("\n" + "=" * 96)
    print("GLOBAL KÖPPEN PROBE OF THE FROZEN 16-D HYDROLOGICAL REPRESENTATION")
    print("=" * 96)

    print("\nA) Broad Köppen-family probe")
    if len(family_cv):
        print(
            family_cv.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )
    else:
        print("No valid family probe result.")

    print("\nB) Exact Köppen-class probe")
    if len(class_cv):
        print(
            class_cv.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )
    else:
        print("No valid exact-class probe result.")

    print("\nC) Pairwise exact-class separability — 15 highest AUC pairs")
    if len(pairwise):
        print(
            pairwise.sort_values("roc_auc", ascending=False)
            .head(15)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

        print("\nD) Pairwise exact-class separability — 15 lowest AUC pairs")
        print(
            pairwise.sort_values("roc_auc", ascending=True)
            .head(15)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )
    else:
        print("No eligible pairwise exact-class tests.")

    print("\nE) Exact-class anchor counts at 80% dominant fraction")
    show = counts[
        (counts["level"] == "exact_class")
        & np.isclose(counts["anchor_threshold"], 0.80)
    ].sort_values("n_basins", ascending=False)
    print(show.to_string(index=False))

    print("\nInterpretation guardrails:")
    print(
        "- The autoencoder remains frozen and was NOT trained with Köppen labels."
    )
    print(
        "- Probe success therefore means Köppen-related information is present "
        "in hydrological dynamics; it does not mean the autoencoder learned the "
        "formal Köppen definition."
    )
    print(
        "- Basin-level historical means are the statistical units, not the "
        "overlapping 24-month windows."
    )
    print(
        "- UMAP figures are descriptive visualizations only. All probe metrics "
        "come from the original 16-D latent vectors."
    )
    print(
        "- Exact-class transition analyses should only be attempted for class "
        "pairs that show reproducible historical separability."
    )
    print(
        "- No post-2020 migration is tested in this script. STOP after "
        "interpreting the probe."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Global Köppen probe of frozen latent representation ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    emb, static, long, zcols = load_inputs()
    static = add_family_fractions(static, long)

    hist, hist_basins = build_historical_basin_means(
        emb,
        static,
        zcols,
    )

    hist_basins.to_csv(HIST_BASIN_OUT, index=False)
    print(f"✅ Saved: {HIST_BASIN_OUT}")

    counts = anchor_counts(hist_basins)
    counts.to_csv(COUNTS_OUT, index=False)
    print(f"✅ Saved: {COUNTS_OUT}")

    family_cv, _ = run_family_probe(hist_basins, zcols)
    family_cv.to_csv(FAMILY_CV_OUT, index=False)
    print(f"✅ Saved: {FAMILY_CV_OUT}")

    class_cv = run_exact_class_probe(hist_basins, zcols)
    class_cv.to_csv(CLASS_CV_OUT, index=False)
    print(f"✅ Saved: {CLASS_CV_OUT}")

    pairwise = pairwise_auc(hist_basins, zcols)
    pairwise.to_csv(PAIRWISE_OUT, index=False)
    print(f"✅ Saved: {PAIRWISE_OUT}")

    historical_basin_umap(hist_basins, zcols)

    qc = build_qc(
        emb,
        hist,
        hist_basins,
        family_cv,
        class_cv,
        pairwise,
    )
    qc.to_csv(QC_OUT, index=False)
    print(f"✅ Saved: {QC_OUT}")

    config = {
        "historical_window_end_max": str(HIST_END.date()),
        "latent_dimensions": 16,
        "anchor_thresholds": ANCHOR_THRESHOLDS,
        "minimum_exact_class_basins": MIN_CLASS_BASINS,
        "pairwise_anchor_threshold": PAIRWISE_THRESHOLD,
        "pairwise_minimum_per_class": PAIRWISE_MIN_PER_CLASS,
        "max_cv_splits": MAX_CV_SPLITS,
        "n_permutations": N_PERMUTATIONS,
        "representation_frozen": True,
        "koppen_used_in_autoencoder_training": False,
        "quantitative_space": "16D latent",
        "umap_role": "visualization only",
    }
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved: {CONFIG_OUT}")

    print_summary(family_cv, class_cv, pairwise, counts)

    print("\n--- Done ---")
    print(
        "STOP here. Send the A-E console output before any global "
        "post-2020 climate-signature migration analysis."
    )


if __name__ == "__main__":
    main()

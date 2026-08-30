# validateKoppenMigrationDurationMatchedSupport.py
#
# Duration-matched historical-support audit.
#
# WHY THIS CORRECTION IS NECESSARY
# --------------------------------
# The previous support audit compared:
#
#   recent mean = ~38 rolling-window embeddings
#
# against a historical support distribution built from:
#
#   one whole-history mean per basin = ~168 rolling-window embeddings
#
# Those objects have very different temporal averaging scales. Whole-history
# basin means are much smoother, so a short recent mean will look artificially
# far from them. That can inflate "novel" and "departure" outcomes.
#
# This audit compares LIKE WITH LIKE:
#
#   recent L-window segment mean
#       versus
#   historical L-window segment means
#
# where L is inferred from the modal number of fully post-2020 windows among
# historically consistent 80%-dominant Köppen anchors (expected ~38 here).
#
# Historical support is calibrated from NON-OVERLAPPING, monthly-contiguous
# L-window blocks to reduce pseudo-replication.
#
# No autoencoder retraining. No UMAP distances. No formal Köppen reclassification.

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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
PREVIOUS_BASIN_SUMMARY_FILE = Path(
    "results/tables/global_koppen_migration/"
    "basin_recent_drift_corrected_summary.csv"
)
GLOBAL_DRIFT_FILE = Path(
    "results/tables/global_koppen_migration/"
    "global_latent_drift_by_window_end.csv"
)

OUT_DIR = Path("results/tables/koppen_duration_matched_support")
FIG_DIR = Path("results/figures/koppen_duration_matched_support")

CALIBRATION_OUT = OUT_DIR / "duration_matched_class_support_calibration.csv"
HIST_SEGMENTS_OUT = OUT_DIR / "historical_duration_matched_segments.parquet"
RECENT_SEGMENTS_OUT = OUT_DIR / "recent_duration_matched_basin_means.csv"
PRIMARY_OUT = OUT_DIR / "duration_matched_support_outcomes_q95.csv"
STRICT_OUT = OUT_DIR / "duration_matched_strict_migrations_q95.csv"
SENSITIVITY_OUT = OUT_DIR / "duration_matched_support_sensitivity.csv"
QC_OUT = OUT_DIR / "duration_matched_support_qc.csv"
CONFIG_OUT = OUT_DIR / "duration_matched_support_config.json"

HIST_END = pd.Timestamp("2020-12-01")
RECENT_START = pd.Timestamp("2021-01-01")

SOURCE_ANCHOR_THRESHOLD = 0.80
PAIRWISE_AUC_SCREEN = 0.80
MIN_CLASS_BASINS = 10

K_NEIGHBORS = 3
SUPPORT_QUANTILES = [0.90, 0.95, 0.99]
PRIMARY_SUPPORT_QUANTILE = 0.95

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
        raise ValueError(f"Expected 16 latent dimensions; found {cols}")
    return cols


def month_diff(t1, t2):
    return (t2.year - t1.year) * 12 + (t2.month - t1.month)


def split_contiguous_runs(df, time_col="start_time"):
    """Return monthly-contiguous runs of a basin's window-start sequence."""
    d = df.sort_values(time_col).reset_index(drop=True).copy()
    if d.empty:
        return []

    breaks = [0]
    times = d[time_col].tolist()
    for i in range(1, len(times)):
        if month_diff(times[i - 1], times[i]) != 1:
            breaks.append(i)
    breaks.append(len(d))

    runs = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        runs.append(d.iloc[a:b].copy())
    return runs


def mean_k_nearest_distance(point, reference, k):
    point = np.asarray(point, dtype=float).reshape(1, -1)
    reference = np.asarray(reference, dtype=float)
    if len(reference) == 0:
        return np.nan
    d = np.linalg.norm(reference - point, axis=1)
    k_eff = min(k, len(d))
    return float(np.mean(np.partition(d, k_eff - 1)[:k_eff]))


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
# LOAD + RECONSTRUCT DRIFT-CORRECTED WINDOW VECTORS
# =============================================================================

def load_inputs():
    emb = pd.read_parquet(EMBEDDINGS_FILE).copy()
    static = pd.read_csv(STATIC_SUMMARY_FILE)
    pairwise = pd.read_csv(PAIRWISE_AUC_FILE)
    prev = pd.read_csv(PREVIOUS_BASIN_SUMMARY_FILE)
    drift = pd.read_csv(GLOBAL_DRIFT_FILE)

    bcol = detect_basin_column(emb)
    emb["basin_id"] = pd.to_numeric(emb[bcol], errors="raise").astype("int64")
    emb["start_time"] = pd.to_datetime(emb["start_time"])
    emb["end_time"] = pd.to_datetime(emb["end_time"])

    static["HYBAS_ID"] = pd.to_numeric(
        static["HYBAS_ID"], errors="raise"
    ).astype("int64")
    prev["basin_id"] = pd.to_numeric(
        prev["basin_id"], errors="raise"
    ).astype("int64")
    drift["end_time"] = pd.to_datetime(drift["end_time"])

    zcols = latent_columns(emb)

    print(f"✅ Continuous windows: {len(emb):,}")
    print(f"✅ Embedding basins:   {emb['basin_id'].nunique():,}")
    print(f"✅ Previous basin summaries: {len(prev):,}")
    print(f"✅ Pairwise AUC rows:  {len(pairwise):,}")

    return emb, static, pairwise, prev, drift, zcols


def reconstruct_corrected_windows(emb, drift, zcols):
    hist = emb[emb["end_time"] <= HIST_END].copy()
    scaler = StandardScaler()
    scaler.fit(hist[zcols].to_numpy(dtype=float))

    Z = scaler.transform(emb[zcols].to_numpy(dtype=float))
    szcols = [f"sz{i}" for i in range(len(zcols))]

    d = emb[["basin_id", "start_time", "end_time"]].copy()
    d[szcols] = Z

    gcols = [f"global_drift_sz{i}" for i in range(len(zcols))]
    missing = [c for c in gcols if c not in drift.columns]
    if missing:
        raise ValueError(
            "Global drift file is missing expected columns: "
            + ", ".join(missing)
        )

    d = d.merge(
        drift[["end_time"] + gcols],
        on="end_time",
        how="left",
        validate="many_to_one",
    )

    if d[gcols].isna().any().any():
        raise ValueError("Some windows could not be matched to global drift.")

    ccols = []
    for i, s in enumerate(szcols):
        c = f"corr_{s}"
        d[c] = d[s] - d[gcols[i]]
        ccols.append(c)

    return d, szcols, ccols


# =============================================================================
# INFER THE RECENT SUMMARY DURATION
# =============================================================================

def infer_segment_length(corrected, prev):
    consistent = prev[
        prev["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
        & (prev["KG_dom_frac"] >= SOURCE_ANCHOR_THRESHOLD)
        & prev["historical_oof_matches_static"].fillna(False).astype(bool)
    ]["basin_id"].astype("int64")

    recent = corrected[
        (corrected["start_time"] >= RECENT_START)
        & corrected["basin_id"].isin(set(consistent))
    ].copy()

    counts = recent.groupby("basin_id").size()
    if counts.empty:
        raise ValueError("No recent windows for historically consistent anchors.")

    mode_vals = counts.mode()
    if mode_vals.empty:
        raise ValueError("Could not infer modal recent window count.")

    L = int(mode_vals.iloc[0])

    print("\n✅ Duration matching")
    print(f"   Modal fully post-2020 window count: {L}")
    print(
        f"   Basins at modal count: {(counts == L).sum():,}/"
        f"{len(counts):,}"
    )
    print(
        f"   Recent count range: {counts.min()}–{counts.max()}"
    )

    return L


# =============================================================================
# HISTORICAL NON-OVERLAPPING L-WINDOW SEGMENTS
# =============================================================================

def make_historical_segments(corrected, static, ccols, L):
    d = corrected[corrected["end_time"] <= HIST_END].copy()

    meta = static[
        [
            "HYBAS_ID",
            "KG_qc_pass_90pct_valid",
            "KG_dom_abbr",
            "KG_dom_frac",
        ]
    ].copy()

    d = d.merge(
        meta,
        left_on="basin_id",
        right_on="HYBAS_ID",
        how="left",
        validate="many_to_one",
    )

    anchors = d[
        d["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
        & (d["KG_dom_frac"] >= SOURCE_ANCHOR_THRESHOLD)
    ].copy()

    basin_class = (
        anchors[
            ["basin_id", "KG_dom_abbr", "KG_dom_frac"]
        ]
        .drop_duplicates("basin_id")
    )
    counts = basin_class["KG_dom_abbr"].value_counts()
    eligible = sorted(
        counts[counts >= MIN_CLASS_BASINS].index.astype(str)
    )

    anchors = anchors[
        anchors["KG_dom_abbr"].astype(str).isin(eligible)
    ].copy()

    rows = []
    for basin_id, b in anchors.groupby("basin_id"):
        cls = str(b["KG_dom_abbr"].iloc[0])
        frac = float(b["KG_dom_frac"].iloc[0])

        for run_id, run in enumerate(split_contiguous_runs(b, "start_time")):
            if len(run) < L:
                continue

            # Non-overlapping blocks of exactly L window starts.
            # Start from the END of the historical run so retained blocks are
            # closest in time scale to the recent period while still independent
            # from each other at the window-start level.
            n_blocks = len(run) // L
            start0 = len(run) - n_blocks * L

            for j in range(n_blocks):
                block = run.iloc[
                    start0 + j * L : start0 + (j + 1) * L
                ]

                row = {
                    "basin_id": int(basin_id),
                    "KG_class": cls,
                    "KG_dom_frac": frac,
                    "run_id": run_id,
                    "block_id": j,
                    "n_windows": len(block),
                    "block_start": block["start_time"].iloc[0],
                    "block_end": block["end_time"].iloc[-1],
                }
                vals = block[ccols].mean(axis=0).to_numpy(dtype=float)
                for i, v in enumerate(vals):
                    row[f"seg{i}"] = float(v)
                rows.append(row)

    seg = pd.DataFrame(rows)
    if seg.empty:
        raise ValueError("No historical duration-matched segments were built.")

    segcols = [f"seg{i}" for i in range(len(ccols))]

    print(f"✅ Historical matched segments: {len(seg):,}")
    print(f"✅ Basins represented:          {seg['basin_id'].nunique():,}")
    print(f"✅ Eligible classes:            {len(eligible)}")

    return seg, segcols, eligible


# =============================================================================
# RECENT L-WINDOW SEGMENT MEANS
# =============================================================================

def make_recent_segments(corrected, prev, ccols, L):
    recent = corrected[corrected["start_time"] >= RECENT_START].copy()

    source_meta = prev[
        [
            "basin_id",
            "KG_qc_pass_90pct_valid",
            "KG_dom_abbr",
            "KG_dom_frac",
            "historical_oof_probe_class",
            "historical_oof_matches_static",
        ]
    ].copy()

    recent = recent.merge(
        source_meta,
        on="basin_id",
        how="left",
        validate="many_to_one",
    )

    source = recent[
        recent["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
        & (recent["KG_dom_frac"] >= SOURCE_ANCHOR_THRESHOLD)
        & recent["historical_oof_matches_static"].fillna(False).astype(bool)
    ].copy()

    rows = []
    for basin_id, b in source.groupby("basin_id"):
        runs = split_contiguous_runs(b, "start_time")
        if not runs:
            continue

        # Use the most recent contiguous run. Require at least L windows, then
        # take exactly the latest L to match historical calibration duration.
        run = max(runs, key=lambda x: x["start_time"].max())
        if len(run) < L:
            continue
        block = run.iloc[-L:].copy()

        row = {
            "basin_id": int(basin_id),
            "source_KG": str(b["KG_dom_abbr"].iloc[0]),
            "KG_dom_frac": float(b["KG_dom_frac"].iloc[0]),
            "historical_oof_probe_class": str(
                b["historical_oof_probe_class"].iloc[0]
            ),
            "n_windows": len(block),
            "recent_block_start": block["start_time"].iloc[0],
            "recent_block_end": block["end_time"].iloc[-1],
        }

        vals = block[ccols].mean(axis=0).to_numpy(dtype=float)
        for i, v in enumerate(vals):
            row[f"seg{i}"] = float(v)
        rows.append(row)

    out = pd.DataFrame(rows)
    print(f"✅ Recent matched basin segments: {len(out):,}")
    return out


# =============================================================================
# DURATION-MATCHED SUPPORT CALIBRATION
# =============================================================================

def calibrate_support(hist_seg, segcols):
    refs = {}
    rows = []

    for cls, g in hist_seg.groupby("KG_class"):
        X = g[segcols].to_numpy(dtype=float)
        basin_ids = g["basin_id"].to_numpy(dtype=np.int64)

        loo = []
        for i in range(len(g)):
            # Critical: calibrate each historical segment against OTHER basins
            # from the same class, so overlap within a basin cannot create an
            # artificially tiny support distance.
            ref_mask = basin_ids != basin_ids[i]
            ref = X[ref_mask]
            if len(ref) == 0:
                continue
            loo.append(mean_k_nearest_distance(X[i], ref, K_NEIGHBORS))

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
            "n_segments": len(X),
            "n_basins": g["basin_id"].nunique(),
            "thresholds": thresholds,
        }

        row = {
            "KG_class": str(cls),
            "n_historical_anchor_basins": g["basin_id"].nunique(),
            "n_historical_matched_segments": len(g),
            "loo_other_basin_knn_distance_median": float(np.median(loo)),
        }
        for q in SUPPORT_QUANTILES:
            row[f"support_distance_q{int(q*100)}"] = thresholds[q]
        rows.append(row)

    cal = pd.DataFrame(rows).sort_values(
        "n_historical_anchor_basins",
        ascending=False,
    )

    return refs, cal


# =============================================================================
# DURATION-MATCHED PAIR AXIS
# =============================================================================

def fit_pair_axis(hist_seg, source, target, segcols):
    d = hist_seg[
        hist_seg["KG_class"].astype(str).isin([source, target])
    ].copy()

    if d["KG_class"].nunique() != 2:
        return None

    X = d[segcols].to_numpy(dtype=float)
    y = (d["KG_class"].astype(str).to_numpy() == target).astype(int)

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)

    s = model.decision_function(X)
    src = d["KG_class"].astype(str).to_numpy() == source
    tgt = ~src
    sm = float(np.mean(s[src]))
    tm = float(np.mean(s[tgt]))

    if tm <= sm:
        model.coef_ *= -1.0
        model.intercept_ *= -1.0
        sm = -sm
        tm = -tm

    if tm - sm <= 1e-12:
        return None

    return model, sm, tm


def axis_position(point, fitted):
    model, sm, tm = fitted
    score = float(
        model.decision_function(
            np.asarray(point, dtype=float).reshape(1, -1)
        )[0]
    )
    return (score - sm) / (tm - sm)


# =============================================================================
# SCORE RECENT SUPPORT
# =============================================================================

def score_recent(
    recent,
    hist_seg,
    refs,
    pairwise,
    segcols,
    support_q,
):
    auc_lut = pair_auc_lookup(pairwise)
    pair_models = {}
    rows = []

    for _, r in recent.iterrows():
        source = str(r["source_KG"])
        point = r[segcols].to_numpy(dtype=float)

        if source not in refs:
            continue

        scored = []
        for cls, info in refs.items():
            # For source support, including the basin's own HISTORICAL segments
            # is scientifically appropriate: we explicitly want to know whether
            # recent behavior still resembles its own historical class behavior.
            dist = mean_k_nearest_distance(
                point,
                info["X"],
                K_NEIGHBORS,
            )
            threshold = info["thresholds"][support_q]
            ratio = dist / threshold if threshold > 0 else np.nan

            auc = (
                1.0 if cls == source
                else auc_lut.get((source, cls), np.nan)
            )

            scored.append({
                "class": cls,
                "distance": dist,
                "threshold": threshold,
                "ratio": ratio,
                "inside": bool(np.isfinite(ratio) and ratio <= 1.0),
                "auc": auc,
            })

        sdf = pd.DataFrame(scored).sort_values(["ratio", "distance"])
        src = sdf[sdf["class"] == source].iloc[0]

        supported = sdf[sdf["inside"]].copy()
        supported_classes = supported["class"].astype(str).tolist()

        alt = sdf[
            (sdf["class"] != source)
            & sdf["inside"]
            & (sdf["auc"] >= PAIRWISE_AUC_SCREEN)
        ].copy()

        if len(alt):
            target_row = alt.iloc[0]
            target = str(target_row["class"])
            target_ratio = float(target_row["ratio"])
            auc = float(target_row["auc"])
        else:
            target = None
            target_ratio = np.nan
            auc = np.nan

        hist_axis = np.nan
        recent_axis = np.nan
        crossed = False
        moved = False

        if target is not None:
            key = (source, target)
            if key not in pair_models:
                pair_models[key] = fit_pair_axis(
                    hist_seg,
                    source,
                    target,
                    segcols,
                )
            fitted = pair_models[key]
            if fitted is not None:
                # Historical source position = median position of this basin's
                # own matched historical segments on this pair axis.
                own_hist = hist_seg[
                    hist_seg["basin_id"] == int(r["basin_id"])
                ]
                own_hist = own_hist[
                    own_hist["KG_class"].astype(str) == source
                ]

                if len(own_hist):
                    hpos = [
                        axis_position(x, fitted)
                        for x in own_hist[segcols].to_numpy(dtype=float)
                    ]
                    hist_axis = float(np.median(hpos))

                recent_axis = axis_position(point, fitted)

                if np.isfinite(hist_axis):
                    moved = bool(recent_axis > hist_axis)
                    crossed = bool(hist_axis < 0.5 and recent_axis >= 0.5)

        source_inside = bool(src["inside"])
        any_support = len(supported_classes) > 0

        if source_inside and target is None:
            outcome = "source_consistent"
        elif source_inside and target is not None:
            outcome = "historical_support_overlap"
        elif (not source_inside) and target is not None:
            if (
                moved
                and np.isfinite(hist_axis)
                and hist_axis < 0.5
                and recent_axis >= 0.5
            ):
                outcome = "strict_supported_migration"
            else:
                outcome = "supported_alternative_but_no_clean_crossing"
        elif not any_support:
            outcome = "novel_outside_all_historical_support"
        else:
            outcome = "departure_without_validated_target"

        rows.append({
            "basin_id": int(r["basin_id"]),
            "source_KG": source,
            "KG_dom_frac": float(r["KG_dom_frac"]),
            "support_quantile": support_q,
            "source_support_ratio": float(src["ratio"]),
            "source_inside_historical_support": source_inside,
            "n_supported_historical_classes": len(supported_classes),
            "supported_classes": ";".join(supported_classes),
            "best_supported_validated_target": target,
            "target_support_ratio": target_ratio,
            "source_target_pairwise_auc": auc,
            "historical_pair_axis_position": hist_axis,
            "recent_corrected_pair_axis_position": recent_axis,
            "pair_axis_delta_toward_target": (
                recent_axis - hist_axis
                if np.isfinite(hist_axis) and np.isfinite(recent_axis)
                else np.nan
            ),
            "moved_toward_target": moved,
            "crossed_pair_midpoint": crossed,
            "outcome": outcome,
        })

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARIES / PLOTS
# =============================================================================

def sensitivity_table(results):
    rows = []
    for q, d in results.items():
        c = d["outcome"].value_counts()
        rows.append({
            "support_quantile": q,
            "n_source_basins": len(d),
            "source_consistent": int(c.get("source_consistent", 0)),
            "historical_support_overlap": int(
                c.get("historical_support_overlap", 0)
            ),
            "strict_supported_migration": int(
                c.get("strict_supported_migration", 0)
            ),
            "supported_alternative_but_no_clean_crossing": int(
                c.get("supported_alternative_but_no_clean_crossing", 0)
            ),
            "departure_without_validated_target": int(
                c.get("departure_without_validated_target", 0)
            ),
            "novel_outside_all_historical_support": int(
                c.get("novel_outside_all_historical_support", 0)
            ),
        })
    return pd.DataFrame(rows)


def plot_support_ratios(primary):
    d = primary[np.isfinite(primary["source_support_ratio"])].copy()
    if d.empty:
        return

    vals, labels = [], []
    for cls, g in d.groupby("source_KG"):
        if len(g) >= 3:
            vals.append(g["source_support_ratio"].to_numpy(dtype=float))
            labels.append(cls)

    if not vals:
        return

    plt.figure(figsize=(11, 6))
    plt.boxplot(vals, tick_labels=labels, showfliers=False)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Historical source class")
    plt.ylabel(
        "Recent matched-segment kNN distance / historical 95% support threshold"
    )
    plt.title("Duration-matched historical source support")
    plt.tight_layout()
    out = FIG_DIR / "duration_matched_source_support_ratio.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def print_console(L, cal, primary, sensitivity):
    print("\n" + "=" * 104)
    print("DURATION-MATCHED KÖPPEN MIGRATION SUPPORT AUDIT")
    print("=" * 104)

    print("\nA) Matching scale")
    print(f"Historical and recent summaries both use exactly {L} consecutive windows.")
    print(
        "Historical calibration blocks are non-overlapping within each "
        "monthly-contiguous run."
    )

    print("\nB) Duration-matched historical support calibration")
    cols = [
        "KG_class",
        "n_historical_anchor_basins",
        "n_historical_matched_segments",
        "loo_other_basin_knn_distance_median",
        "support_distance_q90",
        "support_distance_q95",
        "support_distance_q99",
    ]
    print(
        cal[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nC) Primary 95% support outcomes")
    counts = primary["outcome"].value_counts()
    out = pd.DataFrame({
        "outcome": counts.index,
        "n_basins": counts.values,
    })
    out["fraction"] = out["n_basins"] / len(primary)
    print(
        out.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nD) Strict supported migrations at 95%")
    strict = primary[
        primary["outcome"] == "strict_supported_migration"
    ].copy()
    if strict.empty:
        print("No strict supported migrations.")
    else:
        cols = [
            "basin_id",
            "source_KG",
            "best_supported_validated_target",
            "KG_dom_frac",
            "source_support_ratio",
            "target_support_ratio",
            "source_target_pairwise_auc",
            "historical_pair_axis_position",
            "recent_corrected_pair_axis_position",
            "pair_axis_delta_toward_target",
        ]
        print(
            strict.sort_values(
                ["source_target_pairwise_auc", "target_support_ratio"],
                ascending=[False, True],
            )[cols]
            .head(30)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    print("\nE) Support-quantile sensitivity")
    print(
        sensitivity.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nInterpretation:")
    print(
        "- THIS audit, not the previous whole-history-mean support audit, is "
        "the valid comparison for recent-vs-historical support because the "
        "temporal averaging duration is matched."
    )
    print(
        "- If source consistency rises substantially, the previous high novelty "
        "rate was mainly a scale-mismatch artifact."
    )
    print(
        "- If novelty/departure remains high even after matching duration, that "
        "is stronger evidence of genuine recent hydrological reorganization."
    )
    print(
        "- Strict migrations remain hypotheses of hydrological-signature "
        "migration, not formal Köppen changes."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Duration-matched Köppen support audit ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    emb, static, pairwise, prev, drift, zcols = load_inputs()
    corrected, szcols, ccols = reconstruct_corrected_windows(
        emb, drift, zcols
    )

    L = infer_segment_length(corrected, prev)

    hist_seg, segcols, eligible = make_historical_segments(
        corrected, static, ccols, L
    )
    hist_seg.to_parquet(HIST_SEGMENTS_OUT, index=False)
    print(f"✅ Saved: {HIST_SEGMENTS_OUT}")

    recent = make_recent_segments(corrected, prev, ccols, L)
    recent.to_csv(RECENT_SEGMENTS_OUT, index=False)
    print(f"✅ Saved: {RECENT_SEGMENTS_OUT}")

    refs, cal = calibrate_support(hist_seg, segcols)
    cal.to_csv(CALIBRATION_OUT, index=False)
    print(f"✅ Saved: {CALIBRATION_OUT}")

    results = {}
    for q in SUPPORT_QUANTILES:
        print(f"🚀 Scoring duration-matched support q={q:.2f}")
        results[q] = score_recent(
            recent,
            hist_seg,
            refs,
            pairwise,
            segcols,
            q,
        )

    primary = results[PRIMARY_SUPPORT_QUANTILE].copy()
    primary.to_csv(PRIMARY_OUT, index=False)
    print(f"✅ Saved: {PRIMARY_OUT}")

    strict = primary[
        primary["outcome"] == "strict_supported_migration"
    ].copy()
    strict.to_csv(STRICT_OUT, index=False)
    print(f"✅ Saved: {STRICT_OUT}")

    sensitivity = sensitivity_table(results)
    sensitivity.to_csv(SENSITIVITY_OUT, index=False)
    print(f"✅ Saved: {SENSITIVITY_OUT}")

    qc = pd.DataFrame([
        {"check": "matched_segment_window_count", "value": L},
        {"check": "eligible_classes", "value": len(eligible)},
        {
            "check": "historical_matched_segments",
            "value": len(hist_seg),
        },
        {
            "check": "historical_matched_basins",
            "value": hist_seg["basin_id"].nunique(),
        },
        {
            "check": "recent_matched_source_basins",
            "value": len(recent),
        },
        {
            "check": "strict_supported_migrations_q95",
            "value": len(strict),
        },
    ])
    qc.to_csv(QC_OUT, index=False)
    print(f"✅ Saved: {QC_OUT}")

    config = {
        "representation_frozen": True,
        "historical_window_end_max": str(HIST_END.date()),
        "recent_window_start_min": str(RECENT_START.date()),
        "segment_length_windows": L,
        "historical_segments": "non-overlapping within monthly-contiguous runs",
        "drift_correction": "same global drift correction as previous audit",
        "source_anchor_threshold": SOURCE_ANCHOR_THRESHOLD,
        "pairwise_auc_screen": PAIRWISE_AUC_SCREEN,
        "k_neighbors": K_NEIGHBORS,
        "support_quantiles": SUPPORT_QUANTILES,
        "primary_support_quantile": PRIMARY_SUPPORT_QUANTILE,
        "unknown_novel_allowed": True,
        "formal_koppen_reclassification_claim": False,
    }
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved: {CONFIG_OUT}")

    plot_support_ratios(primary)
    print_console(L, cal, primary, sensitivity)

    print("\n--- Done ---")
    print(
        "STOP here. This corrected duration-matched result replaces the "
        "previous whole-history-mean support audit for interpretation."
    )


if __name__ == "__main__":
    main()

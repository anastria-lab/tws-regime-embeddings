# validateKoppenMigrationTargetSupport.py
#
# Historical-support audit for the global Köppen-associated migration analysis.
#
# WHY THIS STEP EXISTS
# --------------------
# A multiclass classifier MUST assign every recent basin to one of the known
# classes even when the recent latent vector lies far outside the historical
# support of ALL classes. The previous migration audit produced many extreme
# source->target axis positions (>1, often >2-5), which is a warning that
# recent points may be extrapolating beyond historical target signatures.
#
# This script adds the essential "unknown / novel" option.
#
# It asks:
#   Is the recent drift-corrected basin vector actually inside the empirical
#   historical support of its source class or any historically validated
#   alternative class?
#
# Quantitative space: historical-standardized 16-D latent basin means.
# No autoencoder retraining. No UMAP distances. No formal Köppen reclassification.

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


# =============================================================================
# PATHS
# =============================================================================

BASIN_SUMMARY_FILE = Path(
    "results/tables/global_koppen_migration/"
    "basin_recent_drift_corrected_summary.csv"
)
PAIRWISE_AUC_FILE = Path(
    "results/tables/koppen_frozen_latent_probe/"
    "exact_class_pairwise_auc_anchor80.csv"
)

OUT_DIR = Path("results/tables/koppen_migration_target_support")
FIG_DIR = Path("results/figures/koppen_migration_target_support")

CLASS_SUPPORT_OUT = OUT_DIR / "historical_class_knn_support_calibration.csv"
ALL_BASIN_OUT = OUT_DIR / "basin_recent_historical_support_summary.csv"
STRICT_CANDIDATES_OUT = OUT_DIR / "strict_supported_migration_candidates.csv"
OUTCOME_SUMMARY_OUT = OUT_DIR / "migration_support_outcome_summary.csv"
SENSITIVITY_OUT = OUT_DIR / "historical_support_quantile_sensitivity.csv"
QC_OUT = OUT_DIR / "target_support_audit_qc.csv"
CONFIG_OUT = OUT_DIR / "target_support_audit_config.json"

SOURCE_ANCHOR_THRESHOLD = 0.80
PAIRWISE_AUC_SCREEN = 0.80
MIN_CLASS_BASINS = 10

# Empirical support calibration:
# distance = mean distance to K nearest historical basin means of that class.
K_NEIGHBORS = 3
SUPPORT_QUANTILES = [0.90, 0.95, 0.99]
PRIMARY_SUPPORT_QUANTILE = 0.95

RANDOM_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def latent_cols(df):
    cols = [c for c in df.columns if c.startswith("sz") and c[2:].isdigit()]
    cols = sorted(cols, key=lambda x: int(x[2:]))
    if len(cols) != 16:
        raise ValueError(
            f"Expected historical standardized columns sz0..sz15; found {cols}"
        )
    return cols


def recent_corr_cols(szcols):
    cols = [f"recent_corr_{c}" for c in szcols]
    missing = [c for c in cols if c not in _DF_COLUMNS]
    if missing:
        raise ValueError(f"Missing recent corrected latent columns: {missing}")
    return cols


def pair_auc_lookup(pairwise):
    lut = {}
    for _, r in pairwise.iterrows():
        a = str(r["class_A"])
        b = str(r["class_B"])
        auc = float(r["roc_auc"])
        lut[(a, b)] = auc
        lut[(b, a)] = auc
    return lut


def mean_k_nearest_distance(point, reference, k):
    point = np.asarray(point, dtype=float).reshape(1, -1)
    reference = np.asarray(reference, dtype=float)

    d = np.linalg.norm(reference - point, axis=1)
    if len(d) == 0:
        return np.nan
    k_eff = min(k, len(d))
    return float(np.mean(np.partition(d, k_eff - 1)[:k_eff]))


def leave_one_out_knn_distances(X, k):
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2:
        return np.array([], dtype=float)

    vals = np.empty(n, dtype=float)
    for i in range(n):
        ref = np.delete(X, i, axis=0)
        vals[i] = mean_k_nearest_distance(X[i], ref, k)
    return vals


# =============================================================================
# LOAD
# =============================================================================

def load_inputs():
    basin = pd.read_csv(BASIN_SUMMARY_FILE)
    pairwise = pd.read_csv(PAIRWISE_AUC_FILE)

    global _DF_COLUMNS
    _DF_COLUMNS = set(basin.columns)

    required = {
        "basin_id",
        "KG_qc_pass_90pct_valid",
        "KG_dom_abbr",
        "KG_dom_frac",
        "historical_oof_probe_class",
        "historical_oof_matches_static",
        "recent_corrected_probe_class",
    }
    missing = required - set(basin.columns)
    if missing:
        raise ValueError(f"Basin summary missing columns: {sorted(missing)}")

    required_pair = {"class_A", "class_B", "roc_auc"}
    missing = required_pair - set(pairwise.columns)
    if missing:
        raise ValueError(f"Pairwise table missing columns: {sorted(missing)}")

    z = latent_cols(basin)
    rz = recent_corr_cols(z)

    basin["basin_id"] = pd.to_numeric(
        basin["basin_id"], errors="raise"
    ).astype("int64")

    print(f"✅ Basin recent-summary rows: {len(basin):,}")
    print(f"✅ Historical pairwise tests: {len(pairwise):,}")
    print(f"✅ Latent dimensions:         {len(z)}")

    return basin, pairwise, z, rz


# =============================================================================
# HISTORICAL CLASS REFERENCE SETS
# =============================================================================

def build_reference_sets(basin, zcols):
    ref = basin[
        basin["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
        & (basin["KG_dom_frac"] >= SOURCE_ANCHOR_THRESHOLD)
    ].copy()

    counts = ref["KG_dom_abbr"].value_counts()
    eligible = sorted(
        counts[counts >= MIN_CLASS_BASINS].index.astype(str)
    )
    ref = ref[ref["KG_dom_abbr"].astype(str).isin(eligible)].copy()

    refs = {}
    calibration_rows = []

    for cls in eligible:
        g = ref[ref["KG_dom_abbr"].astype(str) == cls].copy()
        X = g[zcols].to_numpy(dtype=float)

        loo = leave_one_out_knn_distances(X, K_NEIGHBORS)

        if len(loo) == 0:
            continue

        thresholds = {
            q: float(np.quantile(loo, q))
            for q in SUPPORT_QUANTILES
        }

        refs[cls] = {
            "X": X,
            "n": len(X),
            "loo_knn": loo,
            "thresholds": thresholds,
        }

        row = {
            "KG_class": cls,
            "n_historical_anchor_basins": len(X),
            "k_neighbors": min(K_NEIGHBORS, len(X) - 1),
            "loo_knn_distance_median": float(np.median(loo)),
            "loo_knn_distance_mean": float(np.mean(loo)),
        }
        for q in SUPPORT_QUANTILES:
            row[f"support_distance_q{int(q*100)}"] = thresholds[q]
        calibration_rows.append(row)

    calibration = pd.DataFrame(calibration_rows).sort_values(
        "n_historical_anchor_basins",
        ascending=False,
    )

    return ref, refs, calibration, eligible


# =============================================================================
# BINARY PAIR AXIS (SECONDARY DIRECTIONAL CHECK)
# =============================================================================

def fit_pair_axis(ref, source, target, zcols):
    d = ref[
        ref["KG_dom_abbr"].astype(str).isin([source, target])
    ].copy()

    if d["KG_dom_abbr"].nunique() != 2:
        return None

    X = d[zcols].to_numpy(dtype=float)
    y = (
        d["KG_dom_abbr"].astype(str).to_numpy() == target
    ).astype(int)

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)

    s = model.decision_function(X)
    source_mask = d["KG_dom_abbr"].astype(str).to_numpy() == source
    target_mask = ~source_mask

    sm = float(np.mean(s[source_mask]))
    tm = float(np.mean(s[target_mask]))

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
    s = float(model.decision_function(
        np.asarray(point, dtype=float).reshape(1, -1)
    )[0])
    return (s - sm) / (tm - sm)


# =============================================================================
# SUPPORT SCORING
# =============================================================================

def score_support(
    basin,
    ref,
    refs,
    pairwise,
    zcols,
    recent_cols,
    support_q,
):
    auc_lut = pair_auc_lookup(pairwise)
    eligible = sorted(refs.keys())

    candidate_source = basin[
        basin["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
        & (basin["KG_dom_frac"] >= SOURCE_ANCHOR_THRESHOLD)
        & basin["historical_oof_matches_static"].fillna(False).astype(bool)
        & basin["KG_dom_abbr"].astype(str).isin(eligible)
    ].copy()

    pair_models = {}
    rows = []

    for _, r in candidate_source.iterrows():
        source = str(r["KG_dom_abbr"])
        h = r[zcols].to_numpy(dtype=float)
        recent = r[recent_cols].to_numpy(dtype=float)

        class_scores = []
        for cls in eligible:
            info = refs[cls]
            d = mean_k_nearest_distance(
                recent,
                info["X"],
                K_NEIGHBORS,
            )
            threshold = info["thresholds"][support_q]
            ratio = d / threshold if threshold > 0 else np.nan

            auc = (
                1.0 if cls == source
                else auc_lut.get((source, cls), np.nan)
            )

            class_scores.append({
                "class": cls,
                "distance": d,
                "threshold": threshold,
                "support_ratio": ratio,
                "inside_support": bool(
                    np.isfinite(ratio) and ratio <= 1.0
                ),
                "pairwise_auc_from_source": auc,
            })

        score_df = pd.DataFrame(class_scores).sort_values(
            ["support_ratio", "distance"],
            ascending=True,
        )

        source_row = score_df[score_df["class"] == source].iloc[0]
        supported = score_df[score_df["inside_support"]].copy()
        supported_classes = supported["class"].astype(str).tolist()

        # Alternative target must:
        # - be inside historical empirical support
        # - differ from source
        # - have a historically validated pairwise AUC >= screen
        alt = score_df[
            (score_df["class"] != source)
            & score_df["inside_support"]
            & (
                score_df["pairwise_auc_from_source"]
                >= PAIRWISE_AUC_SCREEN
            )
        ].copy()

        if len(alt):
            target_row = alt.iloc[0]
            target = str(target_row["class"])
            target_ratio = float(target_row["support_ratio"])
            target_dist = float(target_row["distance"])
            pair_auc = float(target_row["pairwise_auc_from_source"])
        else:
            target = None
            target_ratio = np.nan
            target_dist = np.nan
            pair_auc = np.nan

        # Directional source->target check only if a supported alternative exists.
        hist_axis = np.nan
        recent_axis = np.nan
        crossed_midpoint = False
        moved_toward_target = False

        if target is not None:
            key = (source, target)
            if key not in pair_models:
                pair_models[key] = fit_pair_axis(
                    ref,
                    source,
                    target,
                    zcols,
                )
            fitted = pair_models[key]

            if fitted is not None:
                hist_axis = axis_position(h, fitted)
                recent_axis = axis_position(recent, fitted)
                moved_toward_target = bool(recent_axis > hist_axis)
                crossed_midpoint = bool(
                    hist_axis < 0.5 and recent_axis >= 0.5
                )

        source_inside = bool(source_row["inside_support"])
        any_support = len(supported_classes) > 0
        n_supported = len(supported_classes)

        # Conservative outcome taxonomy.
        if source_inside and target is None:
            outcome = "source_consistent"
        elif source_inside and target is not None:
            outcome = "historical_support_overlap"
        elif (not source_inside) and target is not None:
            if moved_toward_target and (
                np.isfinite(hist_axis)
                and np.isfinite(recent_axis)
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
            "source_support_ratio": float(source_row["support_ratio"]),
            "source_inside_historical_support": source_inside,
            "n_supported_historical_classes": n_supported,
            "supported_classes": ";".join(supported_classes),
            "best_supported_validated_target": target,
            "target_support_ratio": target_ratio,
            "target_knn_distance": target_dist,
            "source_target_pairwise_auc": pair_auc,
            "historical_pair_axis_position": hist_axis,
            "recent_corrected_pair_axis_position": recent_axis,
            "pair_axis_delta_toward_target": (
                recent_axis - hist_axis
                if np.isfinite(hist_axis) and np.isfinite(recent_axis)
                else np.nan
            ),
            "moved_toward_target": moved_toward_target,
            "crossed_pair_midpoint": crossed_midpoint,
            "previous_forced_recent_class": str(
                r["recent_corrected_probe_class"]
            ),
            "previous_forced_class_is_supported": bool(
                str(r["recent_corrected_probe_class"])
                in supported_classes
            ),
            "outcome": outcome,
        })

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARIES / FIGURES
# =============================================================================

def outcome_summary(primary):
    if primary.empty:
        return pd.DataFrame()

    return (
        primary.groupby(["source_KG", "outcome"])
        .size()
        .rename("n_basins")
        .reset_index()
        .sort_values(["source_KG", "n_basins"], ascending=[True, False])
    )


def sensitivity_table(results_by_q):
    rows = []
    for q, d in results_by_q.items():
        counts = d["outcome"].value_counts()
        rows.append({
            "support_quantile": q,
            "n_source_basins": len(d),
            "source_consistent": int(counts.get("source_consistent", 0)),
            "historical_support_overlap": int(
                counts.get("historical_support_overlap", 0)
            ),
            "strict_supported_migration": int(
                counts.get("strict_supported_migration", 0)
            ),
            "supported_alternative_but_no_clean_crossing": int(
                counts.get(
                    "supported_alternative_but_no_clean_crossing", 0
                )
            ),
            "departure_without_validated_target": int(
                counts.get("departure_without_validated_target", 0)
            ),
            "novel_outside_all_historical_support": int(
                counts.get("novel_outside_all_historical_support", 0)
            ),
            "previous_forced_recent_class_supported_fraction": float(
                d["previous_forced_class_is_supported"].mean()
            ),
        })
    return pd.DataFrame(rows)


def plot_source_support(primary):
    d = primary[np.isfinite(primary["source_support_ratio"])].copy()
    if d.empty:
        return

    vals = []
    labels = []
    for cls, g in d.groupby("source_KG"):
        if len(g) >= 3:
            vals.append(g["source_support_ratio"].to_numpy(dtype=float))
            labels.append(cls)

    if not vals:
        return

    plt.figure(figsize=(11, 6))
    plt.boxplot(vals, tick_labels=labels, showfliers=False)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.ylabel(
        "Recent corrected kNN distance / historical 95% support threshold"
    )
    plt.xlabel("Historical source class")
    plt.title(
        "Does recent hydrological behavior remain inside historical source support?"
    )
    plt.tight_layout()
    out = FIG_DIR / "source_historical_support_ratio_by_class.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def print_console(primary, sensitivity, calibration):
    print("\n" + "=" * 100)
    print("KÖPPEN MIGRATION HISTORICAL-TARGET-SUPPORT AUDIT")
    print("=" * 100)

    print("\nA) Historical empirical support calibration")
    show = calibration[
        [
            "KG_class",
            "n_historical_anchor_basins",
            "loo_knn_distance_median",
            "support_distance_q90",
            "support_distance_q95",
            "support_distance_q99",
        ]
    ]
    print(
        show.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nB) Primary 95% support outcomes")
    counts = primary["outcome"].value_counts()
    summary = pd.DataFrame({
        "outcome": counts.index,
        "n_basins": counts.values,
    })
    summary["fraction"] = summary["n_basins"] / len(primary)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nC) Did the previous forced multiclass target lie inside historical support?")
    frac = primary["previous_forced_class_is_supported"].mean()
    print(
        f"Previous forced recent class supported: "
        f"{int(primary['previous_forced_class_is_supported'].sum())}/"
        f"{len(primary)} ({frac:.3f})"
    )

    print("\nD) Strict supported migrations at 95% historical support")
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

    print("\nDecision rule:")
    print(
        "- If most forced recent classes are OUTSIDE their historical target "
        "support, the previous 114 'transitions' were classifier extrapolation, "
        "not supported climate-signature migration."
    )
    print(
        "- 'Novel outside all historical support' is a valid and potentially "
        "important result; the analysis must not force it into a known class."
    )
    print(
        "- Only 'strict_supported_migration' is eligible for later basin-level "
        "physical/climate validation."
    )
    print(
        "- This still does NOT establish formal Köppen class change."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Historical target-support audit ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    basin, pairwise, zcols, rcols = load_inputs()

    ref, refs, calibration, eligible = build_reference_sets(
        basin,
        zcols,
    )
    calibration.to_csv(CLASS_SUPPORT_OUT, index=False)
    print(f"✅ Saved: {CLASS_SUPPORT_OUT}")

    results = {}
    for q in SUPPORT_QUANTILES:
        print(f"🚀 Scoring historical support at q={q:.2f}")
        results[q] = score_support(
            basin,
            ref,
            refs,
            pairwise,
            zcols,
            rcols,
            q,
        )

    primary = results[PRIMARY_SUPPORT_QUANTILE].copy()
    primary.to_csv(ALL_BASIN_OUT, index=False)
    print(f"✅ Saved: {ALL_BASIN_OUT}")

    strict = primary[
        primary["outcome"] == "strict_supported_migration"
    ].copy()
    strict.to_csv(STRICT_CANDIDATES_OUT, index=False)
    print(f"✅ Saved: {STRICT_CANDIDATES_OUT}")

    summary = outcome_summary(primary)
    summary.to_csv(OUTCOME_SUMMARY_OUT, index=False)
    print(f"✅ Saved: {OUTCOME_SUMMARY_OUT}")

    sensitivity = sensitivity_table(results)
    sensitivity.to_csv(SENSITIVITY_OUT, index=False)
    print(f"✅ Saved: {SENSITIVITY_OUT}")

    qc = pd.DataFrame([
        {"check": "eligible_reference_classes", "value": len(eligible)},
        {"check": "historical_reference_basins", "value": len(ref)},
        {"check": "primary_source_basins_scored", "value": len(primary)},
        {
            "check": "strict_supported_migrations_q95",
            "value": len(strict),
        },
        {
            "check": "forced_recent_target_supported_q95",
            "value": int(
                primary["previous_forced_class_is_supported"].sum()
            ),
        },
    ])
    qc.to_csv(QC_OUT, index=False)
    print(f"✅ Saved: {QC_OUT}")

    config = {
        "representation_frozen": True,
        "quantitative_space": "historical-standardized 16D latent basin means",
        "source_anchor_threshold": SOURCE_ANCHOR_THRESHOLD,
        "pairwise_auc_screen": PAIRWISE_AUC_SCREEN,
        "k_nearest_historical_basins": K_NEIGHBORS,
        "support_quantiles": SUPPORT_QUANTILES,
        "primary_support_quantile": PRIMARY_SUPPORT_QUANTILE,
        "unknown_novel_class_allowed": True,
        "formal_koppen_reclassification_claim": False,
    }
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved: {CONFIG_OUT}")

    plot_source_support(primary)
    print_console(primary, sensitivity, calibration)

    print("\n--- Done ---")
    print(
        "STOP here. Interpret A-E before selecting any climate-transition "
        "examples for the poster."
    )


if __name__ == "__main__":
    main()

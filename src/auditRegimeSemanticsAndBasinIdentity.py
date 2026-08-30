# auditRegimeSemanticsAndBasinIdentity.py
# Diagnostic audit BEFORE any further ENSO/poster interpretation.
#
# Questions
# ---------
# A) Do final k=3 regimes have physically consistent meaning ACROSS basins?
# B) How much of the learned 16-D representation is explained by persistent
#    between-basin differences versus within-basin temporal variation?
#
# This script DOES NOT retrain, recluster, rename regimes, or overwrite
# production outputs.
#
# It deliberately keeps regime labels neutral: Regime 0 / 1 / 2.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_continuous_v3.parquet"
CLUSTERS_FILE = "data/processed/window_clusters.parquet"
WAVELET_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"
SPEI_FILE = "data/processed/window_clusters_spei12_validation.parquet"

OUT_DIR = "results/tables/regime_semantic_audit"
FIG_DIR = "results/figures/regime_semantic_audit"

WINDOW_METRICS_OUTPUT = os.path.join(OUT_DIR, "window_physical_metrics_final_k3.parquet")
BASIN_REGIME_OUTPUT = os.path.join(OUT_DIR, "basin_regime_physical_fingerprints.csv")
GLOBAL_PROFILE_OUTPUT = os.path.join(OUT_DIR, "global_basin_balanced_regime_profiles.csv")
PAIRWISE_OUTPUT = os.path.join(OUT_DIR, "regime_pairwise_semantic_consistency.csv")
ORDER_OUTPUT = os.path.join(OUT_DIR, "regime_full_ordering_consistency.csv")
LATENT_DIM_OUTPUT = os.path.join(OUT_DIR, "latent_variance_decomposition_by_dimension.csv")
LATENT_OVERALL_OUTPUT = os.path.join(OUT_DIR, "latent_variance_decomposition_overall.csv")
BASIN_OCCUPANCY_OUTPUT = os.path.join(OUT_DIR, "basin_regime_occupancy.csv")

FINAL_K = 3
WINDOW_LENGTH = 24

# These are descriptive physical diagnostics. No one metric defines a regime.
MEAN_VARIABLES = [
    "lwe_thickness_anomaly_normalized",
    "lwe_thickness_long",
    "tp_anomaly_normalized",
    "tp_long",
    "pev_anomaly_normalized",
    "e_anomaly_normalized",
    "sro_anomaly_normalized",
    "ssro_anomaly_normalized",
    "swvl1_anomaly_normalized",
    "swvl4_anomaly_normalized",
    "swvl4_long",
]

RMS_VARIABLES = [
    "lwe_thickness_short",
    "lwe_thickness_seasonal",
    "tp_short",
    "tp_seasonal",
    "swvl1_short",
    "swvl1_seasonal",
]

KEY_METRICS_FOR_CONSOLE = [
    "mean_lwe_thickness_anomaly_normalized",
    "mean_lwe_thickness_long",
    "mean_tp_anomaly_normalized",
    "mean_swvl4_anomaly_normalized",
    "mean_swvl4_long",
    "spei12_window_mean",
    "spei12_drought_fraction_le_m1",
    "spei12_wet_fraction_ge_p1",
]


def latent_columns(df):
    cols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def month_serial(s):
    return s.dt.year * 12 + s.dt.month


def load_core():
    emb = pd.read_parquet(EMBEDDINGS_FILE)
    clu = pd.read_parquet(CLUSTERS_FILE)

    zcols = latent_columns(emb)
    if len(zcols) != 16:
        raise ValueError(f"Expected 16 latent dimensions, found {len(zcols)}")

    if "basin" not in emb.columns:
        raise ValueError("Continuous embeddings require 'basin' column.")

    required = {"sample_id", "basin_id", "start_time", "end_time", "cluster", "k"}
    missing = required - set(clu.columns)
    if missing:
        raise ValueError(f"Final cluster file missing: {missing}")

    kvals = sorted(clu["k"].dropna().unique().tolist())
    if kvals != [FINAL_K]:
        raise ValueError(f"Expected final k=3 clusters; found k={kvals}")

    emb = emb.copy()
    emb["start_time"] = pd.to_datetime(emb["start_time"])
    emb["end_time"] = pd.to_datetime(emb["end_time"])

    joined = emb.merge(
        clu[["sample_id", "cluster"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if joined["cluster"].isna().any():
        raise ValueError("Some continuous embeddings lack final regime labels.")

    joined["basin_id"] = joined["basin"].astype("int64")

    print(f"✅ Continuous windows: {len(joined):,}")
    print(f"✅ Basins: {joined['basin_id'].nunique():,}")
    print(f"✅ Final regimes: {sorted(joined['cluster'].unique().tolist())}")

    return joined, zcols


def latent_variance_decomposition(df, zcols):
    """
    Descriptive ANOVA-style variance decomposition in standardized 16-D space.

    total SS = between-basin SS + within-basin SS

    between_fraction tells us how much latent variance can be associated with
    persistent differences among basin mean positions.

    This is NOT a causal statistic and NOT a hypothesis test.
    """
    X = df[zcols].to_numpy(dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)

    work = pd.DataFrame(Xs, columns=zcols)
    work["basin_id"] = df["basin_id"].to_numpy()

    grand = Xs.mean(axis=0)
    total_ss_dim = np.sum((Xs - grand) ** 2, axis=0)

    between_ss_dim = np.zeros(len(zcols), dtype=np.float64)
    within_ss_dim = np.zeros(len(zcols), dtype=np.float64)

    for _, g in work.groupby("basin_id", sort=False):
        G = g[zcols].to_numpy(dtype=np.float64)
        mean_b = G.mean(axis=0)
        n = len(G)

        between_ss_dim += n * (mean_b - grand) ** 2
        within_ss_dim += np.sum((G - mean_b) ** 2, axis=0)

    dim = pd.DataFrame({
        "latent_dimension": zcols,
        "total_ss": total_ss_dim,
        "between_basin_ss": between_ss_dim,
        "within_basin_ss": within_ss_dim,
    })
    dim["between_basin_fraction"] = dim["between_basin_ss"] / dim["total_ss"]
    dim["within_basin_fraction"] = dim["within_basin_ss"] / dim["total_ss"]
    dim["decomposition_error"] = (
        dim["total_ss"] - dim["between_basin_ss"] - dim["within_basin_ss"]
    )

    total = float(total_ss_dim.sum())
    between = float(between_ss_dim.sum())
    within = float(within_ss_dim.sum())

    overall = pd.DataFrame([{
        "latent_dimensions": len(zcols),
        "n_windows": len(df),
        "n_basins": df["basin_id"].nunique(),
        "total_ss": total,
        "between_basin_ss": between,
        "within_basin_ss": within,
        "between_basin_fraction": between / total,
        "within_basin_fraction": within / total,
        "decomposition_error": total - between - within,
    }])

    return dim, overall


def load_wavelet():
    w = pd.read_parquet(WAVELET_FILE).copy()

    if "basin" not in w.columns:
        raise ValueError("Wavelet file requires basin column.")

    if "time" in w.columns:
        w["time"] = pd.to_datetime(w["time"])
    elif {"year", "month"}.issubset(w.columns):
        w["time"] = pd.to_datetime(
            dict(year=w["year"], month=w["month"], day=1)
        )
    else:
        raise ValueError("Wavelet file requires time or year/month.")

    return w.sort_values(["basin", "time"]).reset_index(drop=True)


def rolling_features_for_segment(g):
    """
    Compute only within a strictly consecutive monthly segment, so no rolling
    metric can accidentally bridge a real data gap.
    """
    out = g[["basin", "time"]].copy()

    for c in MEAN_VARIABLES:
        if c in g.columns:
            out[f"mean_{c}"] = (
                g[c].astype(float)
                .rolling(WINDOW_LENGTH, min_periods=WINDOW_LENGTH)
                .mean()
            )

    for c in RMS_VARIABLES:
        if c in g.columns:
            out[f"rms_{c}"] = np.sqrt(
                g[c].astype(float).pow(2)
                .rolling(WINDOW_LENGTH, min_periods=WINDOW_LENGTH)
                .mean()
            )

    tws = "lwe_thickness_anomaly_normalized"
    if tws in g.columns:
        s = g[tws].astype(float)

        drought = (s < -1.0).astype(float).where(s.notna(), np.nan)
        wet = (s > 1.0).astype(float).where(s.notna(), np.nan)

        out["tws_drought_fraction_lt_minus1"] = drought.rolling(
            WINDOW_LENGTH, min_periods=WINDOW_LENGTH
        ).mean()
        out["tws_wet_fraction_gt_plus1"] = wet.rolling(
            WINDOW_LENGTH, min_periods=WINDOW_LENGTH
        ).mean()

    return out


def build_window_physical_metrics(wavelet):
    pieces = []

    print("🚀 Building gap-safe 24-month physical metrics...")
    for basin, g0 in wavelet.groupby("basin", sort=False):
        g = g0.sort_values("time").copy()
        serial = month_serial(g["time"])
        segment = serial.diff().fillna(1).ne(1).cumsum()
        g["_segment"] = segment.to_numpy()

        for _, seg in g.groupby("_segment", sort=False):
            if len(seg) < WINDOW_LENGTH:
                continue
            pieces.append(rolling_features_for_segment(seg))

    if not pieces:
        raise ValueError("No physical rolling metrics constructed.")

    phys = pd.concat(pieces, ignore_index=True)
    phys = phys.rename(columns={"basin": "basin_id", "time": "end_time"})
    phys["basin_id"] = phys["basin_id"].astype("int64")

    metric_cols = [
        c for c in phys.columns if c not in ["basin_id", "end_time"]
    ]

    print(f"✅ Constructed {len(metric_cols)} physical rolling metrics")
    return phys, metric_cols


def attach_physical_and_spei(core, phys, metric_cols):
    base = core[
        ["sample_id", "basin_id", "start_time", "end_time", "cluster", "split"]
    ].copy()

    out = base.merge(
        phys,
        on=["basin_id", "end_time"],
        how="left",
        validate="many_to_one",
    )

    spei_cols = []
    if os.path.exists(SPEI_FILE):
        s = pd.read_parquet(SPEI_FILE)
        candidates = [
            "spei12_end",
            "spei12_window_mean",
            "spei12_window_min",
            "spei12_drought_fraction_le_m1",
            "spei12_severe_drought_fraction_le_m1p5",
            "spei12_wet_fraction_ge_p1",
            "spei12_severe_wet_fraction_ge_p1p5",
        ]
        spei_cols = [c for c in candidates if c in s.columns]

        if spei_cols:
            out = out.merge(
                s[["sample_id"] + spei_cols].drop_duplicates("sample_id"),
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            print(f"✅ Added {len(spei_cols)} independent SPEI-12 metrics")
    else:
        print("⚠️ SPEI validation file not found; audit will continue without SPEI.")

    return out, metric_cols + spei_cols


def basin_regime_fingerprints(window_df, metrics):
    usable = [m for m in metrics if m in window_df.columns]

    agg = (
        window_df.groupby(["basin_id", "cluster"], observed=True)[usable]
        .mean()
        .reset_index()
    )

    counts = (
        window_df.groupby(["basin_id", "cluster"], observed=True)
        .size()
        .rename("n_windows")
        .reset_index()
    )

    agg = agg.merge(
        counts,
        on=["basin_id", "cluster"],
        how="left",
        validate="one_to_one",
    )
    return agg, usable


def basin_regime_occupancy(core):
    counts = (
        core.groupby(["basin_id", "cluster"])
        .size()
        .rename("n_windows")
        .reset_index()
    )
    total = counts.groupby("basin_id")["n_windows"].transform("sum")
    counts["occupancy_fraction"] = counts["n_windows"] / total
    return counts


def global_profiles(br, metrics):
    rows = []
    for cluster, g in br.groupby("cluster"):
        row = {
            "cluster": int(cluster),
            "n_basin_regime_members": len(g),
            "n_unique_basins": g["basin_id"].nunique(),
        }
        for m in metrics:
            row[m] = g[m].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def pairwise_semantic_consistency(br, global_profile, metrics):
    """
    For each metric and regime pair:
      - infer global direction from basin-balanced regime means
      - calculate within-basin paired differences
      - report how often basin-level direction agrees with global direction

    This directly tests whether a global regime meaning reverses locally.
    """
    rows = []
    clusters = sorted(global_profile["cluster"].tolist())

    for m in metrics:
        gmean = global_profile.set_index("cluster")[m]

        pivot = br.pivot(index="basin_id", columns="cluster", values=m)

        for i, a in enumerate(clusters):
            for b in clusters[i + 1:]:
                if a not in pivot.columns or b not in pivot.columns:
                    continue

                pair = pivot[[a, b]].dropna()
                if pair.empty:
                    continue

                global_diff = float(gmean.loc[a] - gmean.loc[b])
                diff = (pair[a] - pair[b]).to_numpy(dtype=float)

                if global_diff > 0:
                    same = diff > 0
                    opposite = diff < 0
                elif global_diff < 0:
                    same = diff < 0
                    opposite = diff > 0
                else:
                    same = np.isclose(diff, 0)
                    opposite = ~same

                nonzero = ~np.isclose(diff, 0)

                rows.append({
                    "metric": m,
                    "cluster_a": int(a),
                    "cluster_b": int(b),
                    "n_paired_basins": len(diff),
                    "global_mean_a": float(gmean.loc[a]),
                    "global_mean_b": float(gmean.loc[b]),
                    "global_a_minus_b": global_diff,
                    "basin_mean_a_minus_b": float(np.mean(diff)),
                    "basin_median_a_minus_b": float(np.median(diff)),
                    "same_global_direction_fraction": float(np.mean(same)),
                    "opposite_global_direction_fraction": float(np.mean(opposite)),
                    "zero_difference_fraction": float(np.mean(~nonzero)),
                })

    return pd.DataFrame(rows)


def rank_tuple(values):
    """
    Highest value first. Regime IDs are returned in descending metric order.
    """
    return tuple(values.sort_values(ascending=False).index.astype(int).tolist())


def spearman_three(a, b):
    """
    Spearman correlation between two length-3 vectors, no scipy dependency.
    """
    a = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    b = pd.Series(b).rank(method="average").to_numpy(dtype=float)

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def full_ordering_consistency(br, global_profile, metrics):
    """
    Among basins that contain all 3 regimes, compare complete regime ordering
    for each metric against the basin-balanced global ordering.
    """
    rows = []
    gp = global_profile.set_index("cluster")

    for m in metrics:
        global_values = gp[m].dropna()

        if len(global_values) != FINAL_K:
            continue

        global_order = rank_tuple(global_values)

        pivot = br.pivot(index="basin_id", columns="cluster", values=m)
        required = list(range(FINAL_K))
        if not set(required).issubset(pivot.columns):
            continue

        complete = pivot[required].dropna()
        if complete.empty:
            continue

        exact = []
        rho = []

        for _, row in complete.iterrows():
            local = row[required]
            local_order = rank_tuple(local)
            exact.append(local_order == global_order)

            gv = global_values.reindex(required).to_numpy(dtype=float)
            lv = local.reindex(required).to_numpy(dtype=float)
            rho.append(spearman_three(lv, gv))

        rho = np.asarray(rho, dtype=float)

        rows.append({
            "metric": m,
            "global_order_high_to_low": ">".join(map(str, global_order)),
            "n_basins_with_all_3_regimes": len(complete),
            "exact_global_order_fraction": float(np.mean(exact)),
            "mean_regime_rank_spearman": float(np.nanmean(rho)),
            "median_regime_rank_spearman": float(np.nanmedian(rho)),
            "fraction_positive_rank_agreement": float(np.nanmean(rho > 0)),
            "fraction_negative_rank_agreement": float(np.nanmean(rho < 0)),
        })

    return pd.DataFrame(rows)


def plot_latent_variance(dim, overall):
    d = dim.sort_values("latent_dimension").copy()
    x = np.arange(len(d))

    plt.figure(figsize=(10, 5))
    plt.bar(x, d["between_basin_fraction"])
    plt.axhline(
        overall["between_basin_fraction"].iloc[0],
        linewidth=1.2,
        linestyle="--",
        label="16-D overall",
    )
    plt.xticks(x, d["latent_dimension"], rotation=45)
    plt.ylim(0, 1)
    plt.ylabel("Fraction of latent variance between basins")
    plt.xlabel("Latent dimension")
    plt.title("Basin-Identity Audit: Between- vs Within-Basin Latent Variance")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "latent_between_basin_variance_fraction.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_order_consistency(order):
    if order.empty:
        return

    sub = order.sort_values("exact_global_order_fraction")
    y = np.arange(len(sub))

    plt.figure(figsize=(10, max(5, 0.42 * len(sub))))
    plt.barh(y, sub["exact_global_order_fraction"])
    plt.yticks(y, sub["metric"])
    plt.xlim(0, 1)
    plt.xlabel("Fraction of basins matching global 3-regime ordering")
    plt.title("Are Global Regime Meanings Consistent Within Individual Basins?")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "regime_full_order_consistency.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_pairwise_consistency(pairwise):
    key = pairwise[
        pairwise["metric"].isin(KEY_METRICS_FOR_CONSOLE)
    ].copy()

    if key.empty:
        return

    key["comparison"] = (
        key["metric"]
        + " | R"
        + key["cluster_a"].astype(str)
        + "-R"
        + key["cluster_b"].astype(str)
    )
    key = key.sort_values("same_global_direction_fraction")
    y = np.arange(len(key))

    plt.figure(figsize=(11, max(6, 0.34 * len(key))))
    plt.barh(y, key["same_global_direction_fraction"])
    plt.yticks(y, key["comparison"])
    plt.xlim(0, 1)
    plt.xlabel("Fraction of paired basins agreeing with global direction")
    plt.title("Pairwise Cross-Basin Regime Semantic Consistency")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "regime_pairwise_semantic_consistency.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def compact_metric_name(m):
    return (
        m.replace("mean_lwe_thickness_anomaly_normalized", "TWSA_mean")
         .replace("mean_lwe_thickness_long", "TWS_long")
         .replace("mean_tp_anomaly_normalized", "P_mean")
         .replace("mean_swvl4_anomaly_normalized", "deep_soil_mean")
         .replace("mean_swvl4_long", "deep_soil_long")
         .replace("spei12_window_mean", "SPEI12_mean")
         .replace("spei12_drought_fraction_le_m1", "SPEI12_drought_frac")
         .replace("spei12_wet_fraction_ge_p1", "SPEI12_wet_frac")
    )


def print_summary(overall, dim, global_profile, pairwise, order):
    print("\n" + "=" * 82)
    print("REGIME SEMANTIC CONSISTENCY + BASIN-IDENTITY AUDIT")
    print("=" * 82)

    ob = overall.iloc[0]
    print("\nA) 16-D latent variance decomposition")
    print(f"   Between-basin fraction: {ob['between_basin_fraction']:.3f}")
    print(f"   Within-basin fraction:  {ob['within_basin_fraction']:.3f}")
    print(
        f"   Per-dimension between-basin fraction range: "
        f"{dim['between_basin_fraction'].min():.3f} .. "
        f"{dim['between_basin_fraction'].max():.3f}"
    )

    print("\nB) Basin-balanced global physical profiles")
    keys = [m for m in KEY_METRICS_FOR_CONSOLE if m in global_profile.columns]
    show = global_profile[["cluster", "n_unique_basins"] + keys].copy()
    show = show.rename(columns={m: compact_metric_name(m) for m in keys})
    print(show.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nC) Complete 3-regime ordering consistency within basins")
    key_order = order[order["metric"].isin(keys)].copy()
    if key_order.empty:
        print("   No key metrics had enough complete 3-regime basin fingerprints.")
    else:
        key_order["metric"] = key_order["metric"].map(compact_metric_name)
        print(
            key_order[
                [
                    "metric",
                    "global_order_high_to_low",
                    "n_basins_with_all_3_regimes",
                    "exact_global_order_fraction",
                    "mean_regime_rank_spearman",
                    "fraction_negative_rank_agreement",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x: .3f}")
        )

    print("\nD) Critical pairwise consistency: R0 vs R2")
    critical = pairwise[
        (pairwise["cluster_a"] == 0)
        & (pairwise["cluster_b"] == 2)
        & (pairwise["metric"].isin(keys))
    ].copy()

    if critical.empty:
        print("   No R0-vs-R2 paired results available.")
    else:
        critical["metric"] = critical["metric"].map(compact_metric_name)
        print(
            critical[
                [
                    "metric",
                    "n_paired_basins",
                    "global_a_minus_b",
                    "same_global_direction_fraction",
                    "opposite_global_direction_fraction",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x: .3f}")
        )

    print("\nHow to read this audit:")
    print("- High same-direction / exact-order fractions => regime semantics generalize across basins.")
    print("- Values near 0.5 for pairwise direction => global regime meaning frequently reverses locally.")
    print("- A high between-basin latent fraction => persistent basin identity strongly structures the embedding.")
    print("- Do NOT use the arbitrary labels 'wet/intermediate/dry' until we inspect these numbers.")
    print("- This audit is descriptive; overlapping windows are not treated as independent hypothesis-test samples.")


def main():
    print("--- Audit: regime semantics and basin identity ---")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    core, zcols = load_core()

    # Part A: latent variance audit
    dim, overall = latent_variance_decomposition(core, zcols)
    dim.to_csv(LATENT_DIM_OUTPUT, index=False)
    overall.to_csv(LATENT_OVERALL_OUTPUT, index=False)

    # Part B: physical semantic consistency
    wavelet = load_wavelet()
    phys, physical_metrics = build_window_physical_metrics(wavelet)
    windows, metrics = attach_physical_and_spei(
        core, phys, physical_metrics
    )

    windows.to_parquet(WINDOW_METRICS_OUTPUT, index=False)

    br, usable_metrics = basin_regime_fingerprints(windows, metrics)
    br.to_csv(BASIN_REGIME_OUTPUT, index=False)

    occupancy = basin_regime_occupancy(core)
    occupancy.to_csv(BASIN_OCCUPANCY_OUTPUT, index=False)

    gp = global_profiles(br, usable_metrics)
    gp.to_csv(GLOBAL_PROFILE_OUTPUT, index=False)

    pairwise = pairwise_semantic_consistency(br, gp, usable_metrics)
    pairwise.to_csv(PAIRWISE_OUTPUT, index=False)

    order = full_ordering_consistency(br, gp, usable_metrics)
    order.to_csv(ORDER_OUTPUT, index=False)

    print(f"✅ Saved: {LATENT_DIM_OUTPUT}")
    print(f"✅ Saved: {LATENT_OVERALL_OUTPUT}")
    print(f"✅ Saved: {WINDOW_METRICS_OUTPUT}")
    print(f"✅ Saved: {BASIN_REGIME_OUTPUT}")
    print(f"✅ Saved: {GLOBAL_PROFILE_OUTPUT}")
    print(f"✅ Saved: {PAIRWISE_OUTPUT}")
    print(f"✅ Saved: {ORDER_OUTPUT}")
    print(f"✅ Saved: {BASIN_OCCUPANCY_OUTPUT}")

    plot_latent_variance(dim, overall)
    plot_order_consistency(order)
    plot_pairwise_consistency(pairwise)

    print_summary(overall, dim, gp, pairwise, order)

    print("\n--- Done ---")
    print("STOP after this audit and inspect the summary before changing the model.")


if __name__ == "__main__":
    main()

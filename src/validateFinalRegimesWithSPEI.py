# validateFinalRegimesWithSPEI.py
# Validation A2: Independent SPEI-12 validation of the FINAL k=3 regimes.
#
# Inputs:
#   data/processed/window_clusters.parquet              (final k=3)
#   data/interim/spei12_basin_means_level04.nc
#   data/processed/embeddings_umap_continuous.parquet  (optional, for figure)
#
# Outputs:
#   data/processed/window_clusters_spei12_validation.parquet
#   results/tables/spei12_validation/*
#   results/figures/spei12_validation/*
#
# Statistical principle:
# Adjacent 24-month windows overlap by 23 months, so this script avoids naive
# window-level p-values. Main comparisons use basin-balanced summaries and
# paired basin-level differences between regimes.

import os
from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

CLUSTERS_FILE = "data/processed/window_clusters.parquet"
SPEI_BASIN_FILE = "data/interim/spei12_basin_means_level04.nc"
UMAP_FILE = "data/processed/embeddings_umap_continuous.parquet"

ENRICHED_OUTPUT = "data/processed/window_clusters_spei12_validation.parquet"

TABLE_DIR = "results/tables/spei12_validation"
FIG_DIR = "results/figures/spei12_validation"

BASIN_CLUSTER_OUTPUT = os.path.join(TABLE_DIR, "spei12_basin_cluster_summary.csv")
CLUSTER_OUTPUT = os.path.join(TABLE_DIR, "spei12_cluster_summary_basin_balanced.csv")
PAIRWISE_OUTPUT = os.path.join(TABLE_DIR, "spei12_pairwise_basin_differences.csv")
COVERAGE_OUTPUT = os.path.join(TABLE_DIR, "spei12_validation_coverage.csv")
RANK_OUTPUT = os.path.join(TABLE_DIR, "spei12_regime_climate_ranking.csv")

WINDOW_LENGTH = 24
N_BOOT = 1000
RANDOM_STATE = 42

PRIMARY_METRICS = [
    "spei12_end",
    "spei12_window_mean",
    "spei12_window_min",
    "spei12_drought_fraction_le_m1",
    "spei12_severe_drought_fraction_le_m1p5",
    "spei12_wet_fraction_ge_p1",
    "spei12_severe_wet_fraction_ge_p1p5",
]


def load_clusters():
    df = pd.read_parquet(CLUSTERS_FILE)
    required = {
        "sample_id", "basin_id", "start_time", "end_time", "split", "cluster", "k"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Final cluster file missing required columns: {missing}")

    ks = sorted(df["k"].dropna().unique().tolist())
    if ks != [3]:
        raise ValueError(
            f"Expected finalized k=3 production clusters, found k values: {ks}"
        )

    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    print(f"✅ Loaded FINAL k=3 clusters: {len(df):,} windows")
    print(f"   Basins: {df['basin_id'].nunique():,}")
    return df


def load_spei():
    ds = xr.open_dataset(SPEI_BASIN_FILE)
    if "spei12" not in ds.data_vars:
        raise ValueError(
            f"'spei12' not found in {SPEI_BASIN_FILE}. "
            f"Variables: {list(ds.data_vars)}"
        )
    if "time" not in ds.coords or "basin" not in ds.coords:
        raise ValueError("SPEI basin file requires time and basin coordinates.")

    df = ds["spei12"].to_dataframe().reset_index()
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"basin": "basin_id"})
    df = df.sort_values(["basin_id", "time"]).reset_index(drop=True)

    print(f"✅ Loaded basin SPEI-12: {len(df):,} basin-month rows")
    print(
        f"   Period: {df['time'].min().date()} to {df['time'].max().date()}"
    )
    return df


def rolling_fraction(series, threshold, mode):
    if mode == "le":
        flag = (series <= threshold).astype(float)
    elif mode == "ge":
        flag = (series >= threshold).astype(float)
    else:
        raise ValueError(mode)
    flag = flag.where(series.notna(), np.nan)
    return flag.rolling(WINDOW_LENGTH, min_periods=WINDOW_LENGTH).mean()


def build_window_metrics(spei):
    pieces = []

    for basin_id, g0 in spei.groupby("basin_id", sort=False):
        g = g0.sort_values("time").copy()
        s = g["spei12"].astype(float)

        p = g[["basin_id", "time"]].copy()
        p["spei12_end"] = s
        p["spei12_window_mean"] = s.rolling(
            WINDOW_LENGTH, min_periods=WINDOW_LENGTH
        ).mean()
        p["spei12_window_min"] = s.rolling(
            WINDOW_LENGTH, min_periods=WINDOW_LENGTH
        ).min()
        p["spei12_window_max"] = s.rolling(
            WINDOW_LENGTH, min_periods=WINDOW_LENGTH
        ).max()
        p["spei12_drought_fraction_le_m1"] = rolling_fraction(s, -1.0, "le")
        p["spei12_severe_drought_fraction_le_m1p5"] = rolling_fraction(
            s, -1.5, "le"
        )
        p["spei12_wet_fraction_ge_p1"] = rolling_fraction(s, 1.0, "ge")
        p["spei12_severe_wet_fraction_ge_p1p5"] = rolling_fraction(
            s, 1.5, "ge"
        )

        pieces.append(p)

    metrics = pd.concat(pieces, ignore_index=True)
    metrics = metrics.rename(columns={"time": "end_time"})
    return metrics


def attach_spei(clusters, metrics):
    out = clusters.merge(
        metrics,
        on=["basin_id", "end_time"],
        how="left",
        validate="many_to_one",
    )
    out["has_spei12_validation"] = out["spei12_end"].notna()

    print(
        f"✅ SPEI-validated windows: "
        f"{out['has_spei12_validation'].sum():,}/{len(out):,} "
        f"({out['has_spei12_validation'].mean():.1%})"
    )
    return out


def build_coverage_table(df):
    rows = []

    for label, g in [("ALL", df)] + list(df.groupby("split")):
        rows.append({
            "period": label,
            "n_windows": len(g),
            "n_validated": int(g["has_spei12_validation"].sum()),
            "validation_fraction": float(g["has_spei12_validation"].mean()),
            "earliest_window_end": str(g["end_time"].min().date()),
            "latest_window_end": str(g["end_time"].max().date()),
            "latest_validated_window_end": (
                str(g.loc[g["has_spei12_validation"], "end_time"].max().date())
                if g["has_spei12_validation"].any()
                else None
            ),
        })

    return pd.DataFrame(rows)


def basin_cluster_summary(valid):
    # Average the overlapping monthly windows first within basin × regime.
    # This prevents basins with more windows from dominating the main summary.
    agg = (
        valid.groupby(["basin_id", "cluster"], observed=True)[PRIMARY_METRICS]
        .mean()
        .reset_index()
    )
    return agg


def bootstrap_mean_ci(values, rng):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan, np.nan

    means = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(x), size=len(x))
        means[i] = x[idx].mean()

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def cluster_summary(bc):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for cluster, g in bc.groupby("cluster", observed=True):
        for metric in PRIMARY_METRICS:
            x = g[metric].dropna().to_numpy(dtype=float)
            if not len(x):
                continue

            lo, hi = bootstrap_mean_ci(x, rng)
            rows.append({
                "cluster": int(cluster),
                "metric": metric,
                "n_basins": int(len(x)),
                "basin_balanced_mean": float(np.mean(x)),
                "basin_balanced_median": float(np.median(x)),
                "q25": float(np.quantile(x, 0.25)),
                "q75": float(np.quantile(x, 0.75)),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
            })

    return pd.DataFrame(rows)


def paired_difference_ci(diffs, rng):
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]

    if len(d) < 2:
        return np.nan, np.nan

    means = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(d), size=len(d))
        means[i] = d[idx].mean()

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def pairwise_basin_differences(bc):
    rng = np.random.default_rng(RANDOM_STATE + 1)
    clusters = sorted(bc["cluster"].unique())
    rows = []

    for metric in PRIMARY_METRICS:
        pivot = bc.pivot(index="basin_id", columns="cluster", values=metric)

        for a, b in combinations(clusters, 2):
            if a not in pivot.columns or b not in pivot.columns:
                continue

            pair = pivot[[a, b]].dropna()
            if pair.empty:
                continue

            diff = (pair[a] - pair[b]).to_numpy(dtype=float)
            lo, hi = paired_difference_ci(diff, rng)
            sd = np.std(diff, ddof=1)
            effect = float(np.mean(diff) / sd) if np.isfinite(sd) and sd > 0 else np.nan

            rows.append({
                "metric": metric,
                "cluster_a": int(a),
                "cluster_b": int(b),
                "n_paired_basins": int(len(diff)),
                "mean_a_minus_b": float(np.mean(diff)),
                "median_a_minus_b": float(np.median(diff)),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "paired_standardized_effect": effect,
            })

    return pd.DataFrame(rows)


def climate_ranking(summary):
    sub = summary[summary["metric"] == "spei12_window_mean"].copy()
    sub = sub.sort_values("basin_balanced_mean").reset_index(drop=True)

    labels = ["driest_association", "intermediate_association", "wettest_association"]
    if len(sub) != 3:
        raise ValueError(
            f"Expected three final clusters in SPEI ranking, found {len(sub)}"
        )

    sub["spei12_climate_rank"] = labels
    return sub[
        [
            "cluster",
            "basin_balanced_mean",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "spei12_climate_rank",
        ]
    ]


def plot_basin_boxplot(bc):
    clusters = sorted(bc["cluster"].unique())
    data = [
        bc.loc[bc["cluster"] == c, "spei12_window_mean"].dropna().to_numpy()
        for c in clusters
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=[str(c) for c in clusters], showfliers=False)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Final k=3 regime")
    plt.ylabel("Basin-level mean SPEI-12 across assigned windows")
    plt.title("Independent SPEI-12 Climate Association by Regime")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "spei12_basin_balanced_boxplot.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_mean_ci(summary):
    sub = summary[summary["metric"] == "spei12_window_mean"].sort_values("cluster")
    x = np.arange(len(sub))
    mean = sub["basin_balanced_mean"].to_numpy()
    lower = mean - sub["bootstrap_ci95_low"].to_numpy()
    upper = sub["bootstrap_ci95_high"].to_numpy() - mean

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x,
        mean,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
    )
    plt.axhline(0, linewidth=1)
    plt.xticks(x, sub["cluster"].astype(str))
    plt.xlabel("Final k=3 regime")
    plt.ylabel("Basin-balanced mean SPEI-12")
    plt.title("SPEI-12 Association with 95% Basin-Bootstrap CI")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "spei12_cluster_mean_ci.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_drought_wet(summary):
    drought = summary[
        summary["metric"] == "spei12_drought_fraction_le_m1"
    ].sort_values("cluster")
    wet = summary[
        summary["metric"] == "spei12_wet_fraction_ge_p1"
    ].sort_values("cluster")

    merged = drought[["cluster", "basin_balanced_mean"]].merge(
        wet[["cluster", "basin_balanced_mean"]],
        on="cluster",
        suffixes=("_drought", "_wet"),
    )

    x = np.arange(len(merged))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(
        x - width / 2,
        merged["basin_balanced_mean_drought"],
        width,
        label="SPEI-12 ≤ -1",
    )
    plt.bar(
        x + width / 2,
        merged["basin_balanced_mean_wet"],
        width,
        label="SPEI-12 ≥ +1",
    )
    plt.xticks(x, merged["cluster"].astype(str))
    plt.xlabel("Final k=3 regime")
    plt.ylabel("Mean fraction of months within 24-month windows")
    plt.title("Drought/Wet SPEI-12 Exposure by Regime")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "spei12_drought_wet_exposure.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_umap(valid):
    if not os.path.exists(UMAP_FILE):
        print(f"⚠️ UMAP file not found; skipping SPEI UMAP: {UMAP_FILE}")
        return

    umap = pd.read_parquet(UMAP_FILE)
    p = umap[["sample_id", "u1", "u2"]].merge(
        valid[["sample_id", "spei12_window_mean"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    p = p.dropna(subset=["spei12_window_mean"])

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        p["u1"],
        p["u2"],
        c=p["spei12_window_mean"],
        s=4,
        alpha=0.55,
        cmap="coolwarm",
        vmin=-2,
        vmax=2,
    )
    plt.colorbar(sc, label="Mean SPEI-12 across 24-month window")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Continuous Hydrological State Space Colored by Independent SPEI-12")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "umap_by_spei12_window_mean.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def print_summary(summary, ranking, coverage, pairwise):
    print("\n" + "=" * 76)
    print("SPEI-12 VALIDATION SUMMARY — FINAL k=3 REGIMES")
    print("=" * 76)

    focus_metrics = [
        "spei12_window_mean",
        "spei12_end",
        "spei12_drought_fraction_le_m1",
        "spei12_wet_fraction_ge_p1",
    ]

    for metric in focus_metrics:
        print(f"\n{metric}:")
        sub = summary[summary["metric"] == metric][
            [
                "cluster",
                "n_basins",
                "basin_balanced_mean",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            ]
        ].sort_values("cluster")
        print(sub.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nSPEI-12 climate ranking (association only; not final names):")
    print(ranking.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nValidation coverage:")
    print(
        coverage[
            ["period", "n_windows", "n_validated", "validation_fraction",
             "latest_validated_window_end"]
        ].to_string(index=False)
    )

    focus_pair = pairwise[
        pairwise["metric"].isin(
            ["spei12_window_mean", "spei12_drought_fraction_le_m1"]
        )
    ][
        [
            "metric", "cluster_a", "cluster_b", "n_paired_basins",
            "mean_a_minus_b", "bootstrap_ci95_low", "bootstrap_ci95_high",
            "paired_standardized_effect",
        ]
    ]
    print("\nPaired basin-level regime differences:")
    print(focus_pair.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nInterpretation guardrails:")
    print("- This is independent retrospective climate validation, not model input.")
    print("- Adjacent rolling windows overlap by 23/24 months; no naive window p-values.")
    print("- SPEI-12 association can support regime naming only after these results are inspected.")
    print("- SPEIbase coverage ending in 2024 means later 2025/2026 atlas windows are expected to be unvalidated.")


def main():
    print("--- Validation A2: Independent SPEI-12 validation of FINAL k=3 regimes ---")
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ENRICHED_OUTPUT), exist_ok=True)

    clusters = load_clusters()
    spei = load_spei()
    metrics = build_window_metrics(spei)
    enriched = attach_spei(clusters, metrics)

    coverage = build_coverage_table(enriched)
    valid = enriched[enriched["has_spei12_validation"]].copy()

    bc = basin_cluster_summary(valid)
    summary = cluster_summary(bc)
    pairwise = pairwise_basin_differences(bc)
    ranking = climate_ranking(summary)

    enriched.to_parquet(ENRICHED_OUTPUT, index=False)
    bc.to_csv(BASIN_CLUSTER_OUTPUT, index=False)
    summary.to_csv(CLUSTER_OUTPUT, index=False)
    pairwise.to_csv(PAIRWISE_OUTPUT, index=False)
    coverage.to_csv(COVERAGE_OUTPUT, index=False)
    ranking.to_csv(RANK_OUTPUT, index=False)

    print(f"✅ Saved: {ENRICHED_OUTPUT}")
    print(f"✅ Saved: {BASIN_CLUSTER_OUTPUT}")
    print(f"✅ Saved: {CLUSTER_OUTPUT}")
    print(f"✅ Saved: {PAIRWISE_OUTPUT}")
    print(f"✅ Saved: {COVERAGE_OUTPUT}")
    print(f"✅ Saved: {RANK_OUTPUT}")

    plot_basin_boxplot(bc)
    plot_mean_ci(summary)
    plot_drought_wet(summary)
    plot_umap(valid)

    print_summary(summary, ranking, coverage, pairwise)
    print("--- Done ---")


if __name__ == "__main__":
    main()

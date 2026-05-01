# buildBasinRegimeProfiles.py
# Step 9.3: Basin-level regime occupancy profiles (split-aware)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CLUSTERS_FILE = "data/processed/window_clusters.parquet"

OUTPUT_MAIN = "data/processed/basin_cluster_profiles.parquet"
OUTPUT_SPLIT = "data/processed/basin_cluster_profiles_split.parquet"
OUTPUT_SUMMARY = "results/tables/basin_cluster_profiles_summary.csv"

FIG_DIR = "results/figures/regime_profiles/"


def load_clusters():
    df = pd.read_parquet(CLUSTERS_FILE)
    print(f"✅ Loaded clusters: {CLUSTERS_FILE}")
    print(f"   Rows: {len(df):,}")
    return df


def compute_profiles(df):
    """
    Compute:
    1. Overall basin regime fractions
    2. Split-aware basin regime fractions
    """

    # --------------------------
    # Overall profiles
    # --------------------------
    overall = (
        df.groupby(["basin_id", "cluster"])
        .size()
        .rename("count")
        .reset_index()
    )

    total_per_basin = overall.groupby("basin_id")["count"].transform("sum")
    overall["fraction"] = overall["count"] / total_per_basin

    overall_pivot = overall.pivot_table(
        index="basin_id",
        columns="cluster",
        values="fraction",
        fill_value=0
    )

    overall_pivot.columns = [f"cluster_{c}_frac" for c in overall_pivot.columns]
    overall_pivot = overall_pivot.reset_index()

    # --------------------------
    # Split-aware profiles
    # --------------------------
    split_df = (
        df.groupby(["basin_id", "split", "cluster"])
        .size()
        .rename("count")
        .reset_index()
    )

    total_per_split = split_df.groupby(["basin_id", "split"])["count"].transform("sum")
    split_df["fraction"] = split_df["count"] / total_per_split

    split_pivot = split_df.pivot_table(
        index=["basin_id", "split"],
        columns="cluster",
        values="fraction",
        fill_value=0
    )

    split_pivot.columns = [f"cluster_{c}_frac" for c in split_pivot.columns]
    split_pivot = split_pivot.reset_index()

    return overall_pivot, split_pivot


def save_outputs(overall, split):
    os.makedirs(os.path.dirname(OUTPUT_MAIN), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_SUMMARY), exist_ok=True)

    overall.to_parquet(OUTPUT_MAIN, index=False)
    split.to_parquet(OUTPUT_SPLIT, index=False)

    print(f"✅ Saved overall profiles: {OUTPUT_MAIN}")
    print(f"✅ Saved split profiles: {OUTPUT_SPLIT}")

    summary = overall.describe()
    summary.to_csv(OUTPUT_SUMMARY)
    print(f"✅ Saved summary stats: {OUTPUT_SUMMARY}")


# --------------------------
# FIGURES
# --------------------------

def plot_global_distribution(df):
    os.makedirs(FIG_DIR, exist_ok=True)

    global_dist = df["cluster"].value_counts(normalize=True).sort_index()

    plt.figure(figsize=(6, 4))
    global_dist.plot(kind="bar")
    plt.title("Global regime distribution")
    plt.ylabel("Fraction")
    plt.xlabel("Cluster")
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "regime_global_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"✅ Saved: {path}")


def plot_split_distribution(df):
    dist = (
        df.groupby(["split", "cluster"])
        .size()
        .groupby(level=0)
        .apply(lambda x: x / x.sum())
        .unstack()
        .fillna(0)
    )

    plt.figure(figsize=(8, 5))
    dist.plot(kind="bar", stacked=True)

    plt.title("Regime distribution by split")
    plt.ylabel("Fraction")
    plt.xlabel("Split")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "regime_split_comparison.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"✅ Saved: {path}")


def plot_example_basins(overall, n=10):
    """
    Show regime profiles for a few basins
    """
    sample = overall.sample(n=min(n, len(overall)), random_state=42)

    cluster_cols = [c for c in overall.columns if "cluster_" in c]

    plt.figure(figsize=(10, 6))

    for i, (_, row) in enumerate(sample.iterrows()):
        plt.plot(cluster_cols, row[cluster_cols], marker="o", label=str(row["basin_id"]))

    plt.title("Example basin regime profiles")
    plt.ylabel("Fraction")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "top_basins_regime_distribution.png")
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"✅ Saved: {path}")


def main():
    print("--- Step 9.3: Basin-level regime profiles ---")

    df = load_clusters()

    overall, split = compute_profiles(df)
    save_outputs(overall, split)

    # Figures
    plot_global_distribution(df)
    plot_split_distribution(df)
    plot_example_basins(overall)

    print("\nInterpretation:")
    print("Each basin now has a distribution over dynamic hydrological regimes.")
    print("These are NOT fixed classes — basins can shift regimes over time.")

    print("--- Done ---")


if __name__ == "__main__":
    main()
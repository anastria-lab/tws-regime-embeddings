# mapBasinRegimeProfiles.py
# Step 9.3.1: Map basin-level hydrological regime profiles

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

BASINS_FILE = "data/interim/hydrobasins_l05_global.gpkg"
PROFILE_FILE = "data/processed/basin_cluster_profiles.parquet"
SPLIT_PROFILE_FILE = "data/processed/basin_cluster_profiles_split.parquet"

FIG_DIR = "results/figures/regime_maps"

BASIN_ID_COLUMN = "HYBAS_ID"


def load_inputs():
    basins = gpd.read_file(BASINS_FILE)
    profiles = pd.read_parquet(PROFILE_FILE)
    split_profiles = pd.read_parquet(SPLIT_PROFILE_FILE)

    print(f"✅ Loaded basins: {BASINS_FILE}")
    print(f"   Basins: {len(basins):,}")

    print(f"✅ Loaded overall profiles: {PROFILE_FILE}")
    print(f"   Rows: {len(profiles):,}")

    print(f"✅ Loaded split profiles: {SPLIT_PROFILE_FILE}")
    print(f"   Rows: {len(split_profiles):,}")

    return basins, profiles, split_profiles


def prepare_profiles(profiles):
    cluster_cols = [c for c in profiles.columns if c.startswith("cluster_") and c.endswith("_frac")]

    if not cluster_cols:
        raise ValueError("No cluster fraction columns found.")

    profiles = profiles.copy()

    profiles["dominant_cluster"] = (
        profiles[cluster_cols]
        .idxmax(axis=1)
        .str.extract(r"cluster_(\d+)_frac")
        .astype(int)
    )

    profiles["dominant_cluster_frac"] = profiles[cluster_cols].max(axis=1)

    return profiles, cluster_cols


def prepare_split_profiles(split_profiles):
    cluster_cols = [c for c in split_profiles.columns if c.startswith("cluster_") and c.endswith("_frac")]

    split_profiles = split_profiles.copy()

    split_profiles["dominant_cluster"] = (
        split_profiles[cluster_cols]
        .idxmax(axis=1)
        .str.extract(r"cluster_(\d+)_frac")
        .astype(int)
    )

    split_profiles["dominant_cluster_frac"] = split_profiles[cluster_cols].max(axis=1)

    return split_profiles, cluster_cols


def join_to_basins(basins, profiles):
    basins = basins[[BASIN_ID_COLUMN, "geometry"]].copy()

    basins[BASIN_ID_COLUMN] = basins[BASIN_ID_COLUMN].astype(str)
    profiles = profiles.copy()
    profiles["basin_id"] = profiles["basin_id"].astype(str)

    gdf = basins.merge(
        profiles,
        left_on=BASIN_ID_COLUMN,
        right_on="basin_id",
        how="left"
    )

    return gdf


def plot_dominant_cluster(gdf):
    os.makedirs(FIG_DIR, exist_ok=True)

    plt.figure(figsize=(16, 9))
    ax = plt.gca()

    gdf.plot(
        column="dominant_cluster",
        ax=ax,
        categorical=True,
        legend=True,
        cmap="tab10",
        linewidth=0.05,
        edgecolor="black",
        missing_kwds={
            "color": "lightgrey",
            "label": "No data"
        }
    )

    ax.set_title("Dominant Hydrological Regime by Basin")
    ax.set_axis_off()
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "dominant_regime_by_basin.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {out}")


def plot_dominant_strength(gdf):
    plt.figure(figsize=(16, 9))
    ax = plt.gca()

    gdf.plot(
        column="dominant_cluster_frac",
        ax=ax,
        legend=True,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidth=0.05,
        edgecolor="black",
        missing_kwds={
            "color": "lightgrey",
            "label": "No data"
        }
    )

    ax.set_title("Dominant Regime Occupancy Fraction by Basin")
    ax.set_axis_off()
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "dominant_regime_strength_by_basin.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {out}")


def plot_cluster_fraction_maps(gdf, cluster_cols):
    for col in cluster_cols:
        cluster_id = col.replace("cluster_", "").replace("_frac", "")

        plt.figure(figsize=(16, 9))
        ax = plt.gca()

        gdf.plot(
            column=col,
            ax=ax,
            legend=True,
            cmap="magma",
            vmin=0,
            vmax=1,
            linewidth=0.05,
            edgecolor="black",
            missing_kwds={
                "color": "lightgrey",
                "label": "No data"
            }
        )

        ax.set_title(f"Fraction of Windows in Regime {cluster_id}")
        ax.set_axis_off()
        plt.tight_layout()

        out = os.path.join(FIG_DIR, f"cluster_{cluster_id}_fraction_by_basin.png")
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {out}")


def plot_split_dominant_maps(basins, split_profiles):
    splits = ["train", "val", "test"]

    for split in splits:
        split_df = split_profiles[split_profiles["split"] == split].copy()

        if split_df.empty:
            print(f"⚠️ No split profile rows for {split}")
            continue

        gdf = join_to_basins(basins, split_df)

        plt.figure(figsize=(16, 9))
        ax = plt.gca()

        gdf.plot(
            column="dominant_cluster",
            ax=ax,
            categorical=True,
            legend=True,
            cmap="tab10",
            linewidth=0.05,
            edgecolor="black",
            missing_kwds={
                "color": "lightgrey",
                "label": "No data"
            }
        )

        ax.set_title(f"Dominant Hydrological Regime by Basin — {split}")
        ax.set_axis_off()
        plt.tight_layout()

        out = os.path.join(FIG_DIR, f"dominant_regime_by_basin_{split}.png")
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {out}")


def save_mapped_profiles(gdf):
    out_file = "data/processed/basin_cluster_profiles_mapped.gpkg"

    try:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        gdf.to_file(out_file, driver="GPKG")
        print(f"✅ Saved mapped basin profiles: {out_file}")
    except Exception as e:
        print(f"⚠️ Could not save mapped GeoPackage: {e}")


def plot_train_only_baseline_regime_map(basins, split_profiles):
    """
    Step 9.3.1.1:
    Plot train-only dominant hydrological regime map.

    This is the baseline climatological regime atlas.
    """
    train_df = split_profiles[split_profiles["split"] == "train"].copy()

    if train_df.empty:
        print("⚠️ No train split profile rows found.")
        return

    gdf = join_to_basins(basins, train_df)

    plt.figure(figsize=(16, 9))
    ax = plt.gca()

    gdf.plot(
        column="dominant_cluster",
        ax=ax,
        categorical=True,
        legend=True,
        cmap="tab10",
        linewidth=0.05,
        edgecolor="black",
        missing_kwds={
            "color": "lightgrey",
            "label": "No data"
        }
    )

    ax.set_title("Baseline Hydrological Regime Atlas — Train Period Only")
    ax.set_axis_off()
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "baseline_train_only_dominant_regime_map.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved train-only baseline regime map: {out}")

def main():
    print("--- Step 9.3.1: Map basin regime profiles ---")

    os.makedirs(FIG_DIR, exist_ok=True)

    basins, profiles, split_profiles = load_inputs()

    profiles, cluster_cols = prepare_profiles(profiles)
    split_profiles, split_cluster_cols = prepare_split_profiles(split_profiles)

    gdf = join_to_basins(basins, profiles)

    matched = gdf["basin_id"].notna().sum()
    print(f"✅ Basins matched to profiles: {matched:,}/{len(gdf):,}")

    plot_dominant_cluster(gdf)
    plot_dominant_strength(gdf)
    plot_cluster_fraction_maps(gdf, cluster_cols)
    plot_split_dominant_maps(basins, split_profiles)

    # Step 9.3.1.1 — baseline climatological regime atlas
    plot_train_only_baseline_regime_map(basins, split_profiles)

    save_mapped_profiles(gdf)

    print("--- Done ---")


if __name__ == "__main__":
    main()
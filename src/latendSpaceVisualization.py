# visualizeLatentSpace.py
# Step 9.1: Latent space visualization + diagnostics + basin trajectories

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_v3.parquet"
WAVELET_DATA_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"

UMAP_OUTPUT = "data/processed/embeddings_umap.parquet"
FIG_DIR = "results/figures/latent_space_NEW"

RANDOM_STATE = 42

# Replace / extend these as you identify representative basins
TRAJECTORY_BASINS = {
    "egypt": 1050000010,
    "amazon": 6050298170,
    "iceland": 2050058330,
    "tailand": 4050018280,
}


def load_embeddings():
    df = pd.read_parquet(EMBEDDINGS_FILE)
    print(f"✅ Loaded embeddings: {EMBEDDINGS_FILE}")
    print(f"   Rows: {len(df):,}")
    return df


def load_wavelet_data():
    df = pd.read_parquet(WAVELET_DATA_FILE)
    print(f"✅ Loaded wavelet data: {WAVELET_DATA_FILE}")
    print(f"   Rows: {len(df):,}")
    return df


def get_latent_columns(df):
    latent_cols = [c for c in df.columns if c.startswith("z")]
    latent_cols = sorted(latent_cols, key=lambda x: int(x.replace("z", "")))

    if not latent_cols:
        raise ValueError("No latent columns found.")

    print(f"✅ Latent dimensions: {len(latent_cols)}")
    return latent_cols


def compute_umap(df, latent_cols):
    X = df[latent_cols].to_numpy(dtype=np.float32)

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.15,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )

    U = reducer.fit_transform(X)

    out = df.copy()
    out["u1"] = U[:, 0]
    out["u2"] = U[:, 1]

    return out


def save_umap(df):
    os.makedirs(os.path.dirname(UMAP_OUTPUT), exist_ok=True)
    df.to_parquet(UMAP_OUTPUT, index=False)
    print(f"✅ Saved UMAP embeddings: {UMAP_OUTPUT}")


def plot_umap_by_split(df):
    plt.figure(figsize=(10, 8))

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        plt.scatter(sub["u1"], sub["u2"], s=4, alpha=0.5, label=split)

    plt.title("UMAP of Hydrological Embeddings by Split")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "umap_by_split.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_umap_by_year(df):
    plot_df = df.copy()
    plot_df["start_time"] = pd.to_datetime(plot_df["start_time"])
    plot_df["start_year"] = plot_df["start_time"].dt.year

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        plot_df["u1"],
        plot_df["u2"],
        c=plot_df["start_year"],
        s=4,
        alpha=0.5,
        cmap="viridis",
    )
    plt.colorbar(sc, label="Window start year")
    plt.title("UMAP Colored by Window Start Year")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "umap_by_start_year.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def add_window_mean_variable(umap_df, wavelet_df, variable):
    """
    Compute mean value of a basin-month variable over each sample window.

    Example variable:
    - lwe_thickness_anomaly
    - lwe_thickness_anomaly_normalized
    - tp_anomaly_normalized
    """
    if variable not in wavelet_df.columns:
        print(f"⚠️ Variable not found in wavelet data: {variable}")
        return umap_df

    wavelet_df = wavelet_df.copy()
    wavelet_df["time"] = pd.to_datetime(
        dict(year=wavelet_df["year"], month=wavelet_df["month"], day=1)
    )

    values = []

    # This is okay for diagnostics. If slow later, vectorize.
    for _, row in umap_df.iterrows():
        basin = row["basin"]
        start = pd.to_datetime(row["start_time"])
        end = pd.to_datetime(row["end_time"])

        sub = wavelet_df[
            (wavelet_df["basin"] == basin)
            & (wavelet_df["time"] >= start)
            & (wavelet_df["time"] <= end)
        ]

        values.append(sub[variable].mean())

    out = umap_df.copy()
    out[f"window_mean_{variable}"] = values
    return out


def plot_umap_by_continuous(df, column, title, filename, cmap="coolwarm", robust=True):
    if column not in df.columns:
        print(f"⚠️ Missing column for plot: {column}")
        return

    values = df[column]
    valid = values.notna()

    if robust:
        vmin = values.quantile(0.02)
        vmax = values.quantile(0.98)
    else:
        vmin = values.min()
        vmax = values.max()

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        df.loc[valid, "u1"],
        df.loc[valid, "u2"],
        c=df.loc[valid, column],
        s=4,
        alpha=0.55,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(sc, label=column)
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_umap_by_tws_mean(umap_df, wavelet_df):
    variable = "lwe_thickness_anomaly_normalized"

    enriched = add_window_mean_variable(
        umap_df=umap_df,
        wavelet_df=wavelet_df,
        variable=variable,
    )

    plot_umap_by_continuous(
        enriched,
        column=f"window_mean_{variable}",
        title="UMAP Colored by Mean GRACE TWSA Normalized Anomaly",
        filename="umap_by_mean_lwe_thickness_anomaly_normalized.png",
        cmap="coolwarm",
        robust=True,
    )

    return enriched


def add_drought_year_indicator(df):
    """
    Simple first drought/recent-extreme diagnostic.
    Adjust this later based on regional drought event definitions.
    """
    out = df.copy()
    out["start_time"] = pd.to_datetime(out["start_time"])
    out["start_year"] = out["start_time"].dt.year

    drought_years = {2015, 2016, 2019, 2021, 2022}
    out["drought_year_indicator"] = out["start_year"].isin(drought_years).astype(int)

    return out


def plot_umap_by_drought_indicator(df):
    df = add_drought_year_indicator(df)

    plt.figure(figsize=(10, 8))

    normal = df[df["drought_year_indicator"] == 0]
    drought = df[df["drought_year_indicator"] == 1]

    plt.scatter(normal["u1"], normal["u2"], s=3, alpha=0.25, label="other years")
    plt.scatter(drought["u1"], drought["u2"], s=5, alpha=0.7, label="drought/extreme start years")

    plt.title("UMAP Highlighting Selected Drought/Extreme Start Years")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "umap_by_drought_year_indicator.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_basin_trajectory(df, basin_id, label):
    sub = df[df["basin"] == basin_id].copy()

    if sub.empty:
        print(f"⚠️ No UMAP rows found for basin {basin_id}")
        return

    sub["start_time"] = pd.to_datetime(sub["start_time"])
    sub = sub.sort_values("start_time")

    plt.figure(figsize=(10, 8))

    sc = plt.scatter(
        sub["u1"],
        sub["u2"],
        c=sub["start_time"].dt.year,
        s=30,
        cmap="viridis",
        alpha=0.9,
    )

    plt.plot(sub["u1"], sub["u2"], linewidth=1, alpha=0.5)

    plt.colorbar(sc, label="Window start year")
    plt.title(f"Basin trajectory in UMAP space: {label} ({basin_id})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, f"trajectory_{label}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_selected_basin_trajectories(df):
    for label, basin_id in TRAJECTORY_BASINS.items():
        plot_basin_trajectory(df, basin_id, label)


def main():
    print("--- Step 9.1: Latent space visualization ---")

    os.makedirs(FIG_DIR, exist_ok=True)

    emb = load_embeddings()
    latent_cols = get_latent_columns(emb)

    if os.path.exists(UMAP_OUTPUT):
        print(f"ℹ️ Existing UMAP file found. Loading: {UMAP_OUTPUT}")
        umap_df = pd.read_parquet(UMAP_OUTPUT)
    else:
        umap_df = compute_umap(emb, latent_cols)
        save_umap(umap_df)

    wavelet_df = load_wavelet_data()

    plot_umap_by_split(umap_df)
    plot_umap_by_year(umap_df)

    enriched = plot_umap_by_tws_mean(umap_df, wavelet_df)
    plot_umap_by_drought_indicator(enriched)

    plot_selected_basin_trajectories(enriched)

    print("--- Done ---")


if __name__ == "__main__":
    main()
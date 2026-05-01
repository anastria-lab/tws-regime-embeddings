# interpretEmbeddingRegimes.py
# Step 9.4: Scientific interpretation of learned hydrological regimes

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WAVELET_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"
CLUSTERS_FILE = "data/processed/window_clusters.parquet"

SUMMARY_OUTPUT = "results/tables/cluster_physical_summary.csv"
PROFILE_OUTPUT = "results/tables/cluster_variable_profiles.csv"
FIG_DIR = "results/figures/regime_interpretation"

KEY_VARIABLES = [
    "lwe_thickness",
    "tp",
    "e",
    "pev",
    "sro",
    "ssro",
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
    "lai_hv",
    "lai_lv",
]


def load_inputs():
    wavelet = pd.read_parquet(WAVELET_FILE)
    clusters = pd.read_parquet(CLUSTERS_FILE)

    print(f"✅ Loaded wavelet data: {WAVELET_FILE}")
    print(f"   Rows: {len(wavelet):,}")

    print(f"✅ Loaded window clusters: {CLUSTERS_FILE}")
    print(f"   Rows: {len(clusters):,}")

    return wavelet, clusters


def prepare_wavelet_data(df):
    df = df.copy()
    df["time"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df


def compute_window_means(wavelet_df, clusters):
    """
    For each clustered window, compute mean physical summaries over the window.

    Output is one row per sample/window with cluster label and window-level summaries.
    """
    rows = []

    summary_cols = []

    for v in KEY_VARIABLES:
        for suffix in [
            "",
            "_anomaly",
            "_anomaly_normalized",
            "_short",
            "_seasonal",
            "_long",
        ]:
            col = f"{v}{suffix}"
            if col in wavelet_df.columns:
                summary_cols.append(col)

    summary_cols = sorted(set(summary_cols))
    print(f"✅ Summary columns used: {len(summary_cols)}")

    for i, row in clusters.iterrows():
        basin = row["basin_id"]
        start = pd.to_datetime(row["start_time"])
        end = pd.to_datetime(row["end_time"])

        sub = wavelet_df[
            (wavelet_df["basin"] == basin)
            & (wavelet_df["time"] >= start)
            & (wavelet_df["time"] <= end)
        ]

        if sub.empty:
            continue

        out = {
            "sample_id": row["sample_id"],
            "basin_id": basin,
            "split": row["split"],
            "cluster": row["cluster"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
        }

        means = sub[summary_cols].mean(numeric_only=True)
        for col, val in means.items():
            out[f"mean_{col}"] = val

        # Useful derived hydrological summaries
        if "tp_anomaly_normalized" in sub.columns and "lwe_thickness_long" in sub.columns:
            out["mean_precip_anom_norm"] = sub["tp_anomaly_normalized"].mean()
            out["mean_tws_long"] = sub["lwe_thickness_long"].mean()

        if "sro_anomaly_normalized" in sub.columns and "tp_anomaly_normalized" in sub.columns:
            denom = sub["tp_anomaly_normalized"].abs().mean()
            out["runoff_response_ratio"] = (
                sub["sro_anomaly_normalized"].abs().mean() / denom
                if denom and np.isfinite(denom)
                else np.nan
            )

        if "swvl4_long" in sub.columns and "swvl1_short" in sub.columns:
            out["soil_memory_index"] = (
                sub["swvl4_long"].abs().mean()
                / sub["swvl1_short"].abs().mean()
                if sub["swvl1_short"].abs().mean() != 0
                else np.nan
            )

        if "lwe_thickness_anomaly_normalized" in sub.columns:
            tws = sub.sort_values("time")["lwe_thickness_anomaly_normalized"].to_numpy()
            if len(tws) > 1 and np.isfinite(tws).sum() > 1:
                x = np.arange(len(tws))
                mask = np.isfinite(tws)
                out["tws_trend_norm_per_month"] = np.polyfit(x[mask], tws[mask], 1)[0]
                out["drought_frequency_tws_lt_minus1"] = np.mean(tws < -1)
            else:
                out["tws_trend_norm_per_month"] = np.nan
                out["drought_frequency_tws_lt_minus1"] = np.nan

        rows.append(out)

    window_summary = pd.DataFrame(rows)
    print(f"✅ Built window-level physical summaries: {len(window_summary):,} rows")
    return window_summary


def summarize_by_cluster(window_summary):
    """
    Aggregate window-level physical summaries to cluster-level summaries.
    """
    numeric_cols = [
        c for c in window_summary.columns
        if c not in ["sample_id", "basin_id", "split", "cluster", "start_time", "end_time"]
        and pd.api.types.is_numeric_dtype(window_summary[c])
    ]

    cluster_summary = (
        window_summary.groupby("cluster")[numeric_cols]
        .mean()
        .reset_index()
    )

    counts = (
        window_summary.groupby("cluster")
        .size()
        .rename("n_windows")
        .reset_index()
    )

    basin_counts = (
        window_summary.groupby("cluster")["basin_id"]
        .nunique()
        .rename("n_basins")
        .reset_index()
    )

    cluster_summary = cluster_summary.merge(counts, on="cluster", how="left")
    cluster_summary = cluster_summary.merge(basin_counts, on="cluster", how="left")

    return cluster_summary


def build_variable_profile_table(cluster_summary):
    """
    Tidy table of key physical metrics per cluster.
    """
    rows = []

    metrics = [
        "mean_lwe_thickness_anomaly_normalized",
        "mean_lwe_thickness_long",
        "mean_tp_anomaly_normalized",
        "mean_tp_short",
        "mean_sro_short",
        "mean_ssro_short",
        "mean_swvl1_short",
        "mean_swvl4_long",
        "mean_e_anomaly_normalized",
        "mean_pev_anomaly_normalized",
        "runoff_response_ratio",
        "soil_memory_index",
        "tws_trend_norm_per_month",
        "drought_frequency_tws_lt_minus1",
    ]

    for _, row in cluster_summary.iterrows():
        cluster = row["cluster"]
        for metric in metrics:
            if metric in cluster_summary.columns:
                rows.append({
                    "cluster": cluster,
                    "metric": metric,
                    "value": row[metric],
                })

    return pd.DataFrame(rows)


def save_tables(cluster_summary, variable_profiles):
    os.makedirs(os.path.dirname(SUMMARY_OUTPUT), exist_ok=True)

    cluster_summary.to_csv(SUMMARY_OUTPUT, index=False)
    variable_profiles.to_csv(PROFILE_OUTPUT, index=False)

    print(f"✅ Saved cluster physical summary: {SUMMARY_OUTPUT}")
    print(f"✅ Saved cluster variable profiles: {PROFILE_OUTPUT}")


def plot_metric_bar(cluster_summary, metric, filename, title=None):
    if metric not in cluster_summary.columns:
        print(f"⚠️ Missing metric for plot: {metric}")
        return

    os.makedirs(FIG_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(cluster_summary["cluster"].astype(str), cluster_summary[metric])
    plt.xlabel("Cluster")
    plt.ylabel(metric)
    plt.title(title if title else metric)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved: {out}")


def plot_key_metrics(cluster_summary):
    plot_metric_bar(
        cluster_summary,
        "mean_lwe_thickness_long",
        "cluster_mean_tws_long.png",
        "Mean TWS Long Component by Cluster",
    )

    plot_metric_bar(
        cluster_summary,
        "mean_tp_short",
        "cluster_mean_precip_short.png",
        "Mean Precipitation Short Component by Cluster",
    )

    plot_metric_bar(
        cluster_summary,
        "soil_memory_index",
        "cluster_soil_memory_index.png",
        "Soil Memory Index by Cluster",
    )

    plot_metric_bar(
        cluster_summary,
        "runoff_response_ratio",
        "cluster_runoff_response_ratio.png",
        "Runoff Response Ratio by Cluster",
    )

    plot_metric_bar(
        cluster_summary,
        "drought_frequency_tws_lt_minus1",
        "cluster_drought_frequency.png",
        "Drought Frequency: TWSA < -1 by Cluster",
    )


def print_cluster_interpretation_hint(cluster_summary):
    print("\n" + "=" * 60)
    print("Cluster interpretation hints")
    print("=" * 60)

    for _, row in cluster_summary.iterrows():
        c = int(row["cluster"])

        tws_long = row.get("mean_lwe_thickness_long", np.nan)
        precip_short = row.get("mean_tp_short", np.nan)
        memory = row.get("soil_memory_index", np.nan)
        drought_freq = row.get("drought_frequency_tws_lt_minus1", np.nan)
        runoff_ratio = row.get("runoff_response_ratio", np.nan)

        print(f"\nCluster {c}:")
        print(f"  n_windows: {row.get('n_windows', np.nan)}")
        print(f"  n_basins: {row.get('n_basins', np.nan)}")
        print(f"  mean TWS long: {tws_long:.3f}" if np.isfinite(tws_long) else "  mean TWS long: NaN")
        print(f"  mean precip short: {precip_short:.3f}" if np.isfinite(precip_short) else "  mean precip short: NaN")
        print(f"  soil memory index: {memory:.3f}" if np.isfinite(memory) else "  soil memory index: NaN")
        print(f"  runoff response ratio: {runoff_ratio:.3f}" if np.isfinite(runoff_ratio) else "  runoff response ratio: NaN")
        print(f"  drought frequency: {drought_freq:.3f}" if np.isfinite(drought_freq) else "  drought frequency: NaN")


def main():
    print("--- Step 9.4: Scientific interpretation of regimes ---")

    wavelet_df, clusters = load_inputs()
    wavelet_df = prepare_wavelet_data(wavelet_df)

    window_summary = compute_window_means(wavelet_df, clusters)
    cluster_summary = summarize_by_cluster(window_summary)
    variable_profiles = build_variable_profile_table(cluster_summary)

    save_tables(cluster_summary, variable_profiles)
    plot_key_metrics(cluster_summary)
    print_cluster_interpretation_hint(cluster_summary)

    print("--- Done ---")


if __name__ == "__main__":
    main()
# comparePhysicalRegimeSolutions.py
# Step 9C: compare physical fingerprints of candidate KMeans solutions k=2,3,4.
# Does NOT overwrite data/processed/window_clusters.parquet.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_continuous_v3.parquet"
WAVELET_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"
OUT_DIR = "results/tables/regime_physical_comparison"
FIG_DIR = "results/figures/regime_physical_comparison"
K_VALUES = [2, 3, 4]
WINDOW_LENGTH = 24
RANDOM_STATE = 42
N_INIT = 20

MEAN_VARS = [
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
RMS_VARS = [
    "lwe_thickness_short",
    "lwe_thickness_seasonal",
    "tp_short",
    "tp_seasonal",
    "swvl1_short",
    "swvl1_seasonal",
]
KEY_METRICS = [
    "mean_lwe_thickness_anomaly_normalized",
    "mean_lwe_thickness_long",
    "rms_lwe_thickness_short",
    "rms_lwe_thickness_seasonal",
    "tws_trend_norm_per_month",
    "tws_drought_fraction_lt_minus1",
    "tws_wet_fraction_gt_plus1",
    "mean_tp_anomaly_normalized",
    "rms_tp_short",
    "mean_pev_anomaly_normalized",
    "mean_sro_anomaly_normalized",
    "mean_swvl1_anomaly_normalized",
    "mean_swvl4_anomaly_normalized",
    "mean_swvl4_long",
]


def latent_cols(df):
    cols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def fit_candidates(emb, zcols):
    X = emb[zcols].to_numpy(dtype=np.float32)
    Xs = StandardScaler().fit_transform(X)
    out = emb[["sample_id", "basin", "start_time", "end_time", "split"]].copy()
    for k in K_VALUES:
        print(f"🚀 Fitting KMeans k={k} on continuous 16-D embeddings...")
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        out[f"cluster_k{k}"] = km.fit_predict(Xs).astype(np.int16)
    return out


def rolling_slope_24(values):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[0], np.nan)
    x = np.arange(WINDOW_LENGTH, dtype=float)
    xc = x - x.mean()
    denom = np.sum(xc ** 2)
    for end in range(WINDOW_LENGTH - 1, len(values)):
        w = values[end - WINDOW_LENGTH + 1:end + 1]
        if np.isfinite(w).all():
            out[end] = np.sum(xc * (w - w.mean())) / denom
    return out


def build_physical_end_metrics(wav):
    wav = wav.copy()
    if "time" not in wav.columns:
        wav["time"] = pd.to_datetime(dict(year=wav["year"], month=wav["month"], day=1))
    else:
        wav["time"] = pd.to_datetime(wav["time"])
    wav = wav.sort_values(["basin", "time"]).reset_index(drop=True)

    mean_vars = [c for c in MEAN_VARS if c in wav.columns]
    rms_vars = [c for c in RMS_VARS if c in wav.columns]
    print(f"✅ Mean variables available: {len(mean_vars)}")
    print(f"✅ RMS variables available: {len(rms_vars)}")

    pieces = []
    for basin, g in wav.groupby("basin", sort=False):
        g = g.sort_values("time").copy()
        p = g[["basin", "time"]].copy()
        for c in mean_vars:
            p[f"mean_{c}"] = g[c].rolling(WINDOW_LENGTH, min_periods=WINDOW_LENGTH).mean().to_numpy()
        for c in rms_vars:
            p[f"rms_{c}"] = np.sqrt(
                g[c].pow(2).rolling(WINDOW_LENGTH, min_periods=WINDOW_LENGTH).mean()
            ).to_numpy()
        tws = "lwe_thickness_anomaly_normalized"
        if tws in g.columns:
            p["tws_trend_norm_per_month"] = rolling_slope_24(g[tws].to_numpy())
            valid = g[tws].notna()
            drought = (g[tws] < -1.0).astype(float).where(valid, np.nan)
            wet = (g[tws] > 1.0).astype(float).where(valid, np.nan)
            p["tws_drought_fraction_lt_minus1"] = drought.rolling(WINDOW_LENGTH, min_periods=WINDOW_LENGTH).mean().to_numpy()
            p["tws_wet_fraction_gt_plus1"] = wet.rolling(WINDOW_LENGTH, min_periods=WINDOW_LENGTH).mean().to_numpy()
        pieces.append(p)

    phys = pd.concat(pieces, ignore_index=True).rename(columns={"time": "end_time"})
    return phys


def build_profiles(df, metrics):
    window_rows = []
    basin_rows = []
    for k in K_VALUES:
        ccol = f"cluster_k{k}"
        for cluster, g in df.groupby(ccol):
            row = {"k": k, "cluster": int(cluster), "n_windows": len(g), "n_basins": g["basin"].nunique()}
            row.update(g[metrics].mean().to_dict())
            window_rows.append(row)

        # Basin-balanced profile: each basin contributes at most once to each regime.
        bc = df.groupby(["basin", ccol])[metrics].mean().reset_index()
        for cluster, g in bc.groupby(ccol):
            row = {"k": k, "cluster": int(cluster), "n_basin_regime_members": len(g)}
            row.update(g[metrics].mean().to_dict())
            basin_rows.append(row)

    return pd.DataFrame(window_rows), pd.DataFrame(basin_rows)


def build_separation(df, basin_profiles, metrics):
    # Descriptive effect size: range of basin-balanced cluster means / global SD.
    global_sd = df[metrics].std(ddof=0)
    rows = []
    for k in K_VALUES:
        p = basin_profiles[basin_profiles["k"] == k]
        effects = []
        for m in metrics:
            sd = global_sd[m]
            effect = np.nan if (not np.isfinite(sd) or sd == 0) else (p[m].max() - p[m].min()) / sd
            rows.append({"k": k, "metric": m, "standardized_cluster_mean_range": effect})
            if np.isfinite(effect):
                effects.append(effect)
        rows.append({"k": k, "metric": "__OVERALL_MEDIAN__", "standardized_cluster_mean_range": float(np.median(effects))})
    return pd.DataFrame(rows)


def crosswalk(assignments):
    rows = []
    for ka, kb in [(2, 3), (3, 4), (2, 4)]:
        a, b = f"cluster_k{ka}", f"cluster_k{kb}"
        t = assignments.groupby([a, b]).size().reset_index(name="n_windows")
        t["fraction_within_parent"] = t["n_windows"] / t.groupby(a)["n_windows"].transform("sum")
        t.insert(0, "parent_k", ka)
        t.insert(1, "child_k", kb)
        t = t.rename(columns={a: "parent_cluster", b: "child_cluster"})
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def plot_heatmap(profiles, metrics, k):
    p = profiles[profiles["k"] == k].sort_values("cluster").reset_index(drop=True)
    allvals = profiles[metrics]
    mu = allvals.mean()
    sd = allvals.std(ddof=0).replace(0, np.nan)
    Z = ((p[metrics] - mu) / sd).to_numpy()

    plt.figure(figsize=(max(12, 0.72 * len(metrics)), 2.5 + 0.65 * len(p)))
    im = plt.imshow(Z, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Profile z-score across candidate regime profiles")
    plt.yticks(np.arange(len(p)), [f"cluster {int(c)}" for c in p["cluster"]])
    plt.xticks(np.arange(len(metrics)), metrics, rotation=60, ha="right")
    plt.title(f"k={k}: basin-balanced physical fingerprints")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"physical_fingerprints_k{k}.png")
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


def plot_separation(sep):
    p = sep[sep["metric"] == "__OVERALL_MEDIAN__"].sort_values("k")
    plt.figure(figsize=(7, 5))
    plt.plot(p["k"], p["standardized_cluster_mean_range"], marker="o")
    plt.xticks(K_VALUES)
    plt.xlabel("Candidate number of regimes (k)")
    plt.ylabel("Median standardized physical separation")
    plt.title("Physical differentiation of candidate regime solutions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "physical_separation_by_k.png")
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")


def main():
    print("--- Step 9C: Physical comparison of k=2,3,4 regime solutions ---")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    emb = pd.read_parquet(EMBEDDINGS_FILE)
    wav = pd.read_parquet(WAVELET_FILE)
    emb["start_time"] = pd.to_datetime(emb["start_time"])
    emb["end_time"] = pd.to_datetime(emb["end_time"])
    zcols = latent_cols(emb)
    if len(zcols) != 16:
        raise ValueError(f"Expected 16 latent dimensions; found {len(zcols)}")
    print(f"✅ Loaded continuous embeddings: {len(emb):,} windows")
    print(f"✅ Loaded atlas wavelet data: {len(wav):,} rows")

    assignments = fit_candidates(emb, zcols)
    phys = build_physical_end_metrics(wav)
    merged = assignments.merge(phys, on=["basin", "end_time"], how="left", validate="many_to_one")
    metrics = [m for m in KEY_METRICS if m in merged.columns]
    print(f"✅ Physical fingerprint metrics available: {len(metrics)}")
    if not metrics:
        raise ValueError("No physical fingerprint metrics were available.")

    window_profiles, basin_profiles = build_profiles(merged, metrics)
    sep = build_separation(merged, basin_profiles, metrics)
    xwalk = crosswalk(assignments)

    assignments.to_parquet(os.path.join(OUT_DIR, "candidate_assignments_k234.parquet"), index=False)
    merged[["sample_id", "basin", "start_time", "end_time", "split"] + [f"cluster_k{k}" for k in K_VALUES] + metrics].to_parquet(
        os.path.join(OUT_DIR, "window_physical_metrics.parquet"), index=False
    )
    window_profiles.to_csv(os.path.join(OUT_DIR, "physical_profiles_window_weighted.csv"), index=False)
    basin_profiles.to_csv(os.path.join(OUT_DIR, "physical_profiles_basin_balanced.csv"), index=False)
    sep.to_csv(os.path.join(OUT_DIR, "physical_separation_summary.csv"), index=False)
    xwalk.to_csv(os.path.join(OUT_DIR, "candidate_solution_crosswalk.csv"), index=False)

    for k in K_VALUES:
        plot_heatmap(basin_profiles, metrics, k)
    plot_separation(sep)

    print("\n" + "=" * 72)
    print("STEP 9C — PHYSICAL REGIME COMPARISON SUMMARY")
    print("=" * 72)
    print("\nMedian standardized physical separation:")
    print(sep[sep["metric"] == "__OVERALL_MEDIAN__"][["k", "standardized_cluster_mean_range"]].to_string(index=False))

    focus = [m for m in [
        "mean_lwe_thickness_anomaly_normalized",
        "mean_lwe_thickness_long",
        "tws_trend_norm_per_month",
        "tws_drought_fraction_lt_minus1",
        "tws_wet_fraction_gt_plus1",
        "mean_tp_anomaly_normalized",
        "mean_swvl4_long",
    ] if m in metrics]

    for k in K_VALUES:
        print(f"\n--- k={k}: basin-balanced fingerprints ---")
        cols = ["cluster", "n_basin_regime_members"] + focus
        print(basin_profiles[basin_profiles["k"] == k][cols].sort_values("cluster").to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nDecision rule:")
    print("- Prefer the smallest k that retains materially distinct, stable physical states.")
    print("- k=4 is justified only if its extra split adds a distinct hydrological fingerprint.")
    print("- Do NOT name clusters 'drought'/'wet' yet; SPEI validation is the next step.")
    print("--- Done ---")


if __name__ == "__main__":
    main()

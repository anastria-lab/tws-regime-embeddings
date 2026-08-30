# evaluateKMeansRegimeSolutions.py
# Step 9.2B: Evaluate KMeans regime granularity on the continuous 16-D atlas
#
# IMPORTANT:
# - This script does NOT overwrite data/processed/window_clusters.parquet.
# - It evaluates k=2..8 on exactly the continuous embeddings used for the
#   scientific atlas and downstream trajectory analysis.
# - Final k should be chosen from separation + stability + physical meaning,
#   not silhouette alone.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_continuous_v3.parquet"
OUTPUT_TABLE = "results/tables/kmeans_regime_solution_metrics_continuous.csv"
FIG_DIR = "results/figures/validation"

K_VALUES = list(range(2, 9))
RANDOM_STATE = 42
N_INIT = 20
METRIC_SAMPLE_SIZE = 50000


def get_latent_columns(df):
    cols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    cols = sorted(cols, key=lambda c: int(c[1:]))
    if not cols:
        raise ValueError("No latent columns z1...zN found.")
    return cols


def normalized_entropy(labels, k):
    counts = np.bincount(labels, minlength=k).astype(float)
    p = counts / counts.sum()
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    return float(h / np.log(k)) if k > 1 else np.nan


def main():
    print("--- Step 9.2B: Continuous-atlas KMeans solution evaluation ---")
    df = pd.read_parquet(EMBEDDINGS_FILE)
    print(f"✅ Loaded continuous embeddings: {EMBEDDINGS_FILE}")
    print(f"   Rows: {len(df):,}")

    latent_cols = get_latent_columns(df)
    print(f"✅ Latent dimensions: {len(latent_cols)}")

    X = df[latent_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("NaN/inf found in latent embeddings.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rng = np.random.default_rng(RANDOM_STATE)
    if len(Xs) > METRIC_SAMPLE_SIZE:
        metric_idx = rng.choice(len(Xs), METRIC_SAMPLE_SIZE, replace=False)
    else:
        metric_idx = np.arange(len(Xs))
    Xm = Xs[metric_idx]
    print(f"✅ Fixed metric sample: {len(metric_idx):,} windows")

    rows = []
    for k in K_VALUES:
        print(f"🚀 Fitting full continuous KMeans k={k}...")
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(Xs)
        lm = labels[metric_idx]

        counts = np.bincount(labels, minlength=k)
        fractions = counts / counts.sum()

        sil = silhouette_score(Xm, lm)
        ch = calinski_harabasz_score(Xm, lm)
        db = davies_bouldin_score(Xm, lm)

        row = {
            "k": k,
            "silhouette_score": float(sil),
            "calinski_harabasz": float(ch),
            "davies_bouldin": float(db),
            "inertia": float(model.inertia_),
            "smallest_cluster_fraction": float(fractions.min()),
            "largest_cluster_fraction": float(fractions.max()),
            "normalized_cluster_entropy": normalized_entropy(labels, k),
        }
        rows.append(row)

        print(
            f"   k={k} | silhouette={sil:.4f} | CH={ch:.1f} | DB={db:.4f} | "
            f"min_frac={fractions.min():.3f} | max_frac={fractions.max():.3f}"
        )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)
    out.to_csv(OUTPUT_TABLE, index=False)
    print(f"✅ Saved: {OUTPUT_TABLE}")

    os.makedirs(FIG_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(out["k"], out["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette score")
    plt.title("Continuous-atlas KMeans silhouette by k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(FIG_DIR, "kmeans_continuous_silhouette_by_k.png")
    plt.savefig(p, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(out["k"], out["inertia"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("KMeans inertia")
    plt.title("Continuous-atlas KMeans inertia by k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "kmeans_continuous_inertia_by_k.png")
    plt.savefig(p2, dpi=220, bbox_inches="tight")
    plt.close()

    print("\n============================================================")
    print("Regime-solution metrics")
    print("============================================================")
    print(out.to_string(index=False))
    print("\nInterpretation:")
    print("- higher silhouette is better")
    print("- higher Calinski-Harabasz is better")
    print("- lower Davies-Bouldin is better")
    print("- inertia must decrease with k; inspect the elbow, not its minimum")
    print("- cluster balance is descriptive, not a selection criterion")
    print("- DO NOT choose final k from these metrics alone")
    print("- combine these results with basin-resampling stability and physical validation")
    print("--- Done ---")


if __name__ == "__main__":
    main()

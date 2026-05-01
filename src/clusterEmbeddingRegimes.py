# clusterEmbeddingRegimes.py
# Step 9.2: KMeans regime discovery in 16D hydrological embedding space

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_v3.parquet"
UMAP_FILE = "data/processed/embeddings_umap.parquet"

WINDOW_CLUSTERS_OUTPUT = "data/processed/window_clusters.parquet"
SILHOUETTE_OUTPUT = "results/tables/kmeans_silhouette_scores.csv"
CLUSTER_FIGURE_OUTPUT = "results/figures/latent_space_NEW/umap_by_kmeans_cluster.png"

K_VALUES = [4, 5, 6, 7, 8]
RANDOM_STATE = 42

# Use a sample for silhouette if dataset is large
SILHOUETTE_SAMPLE_SIZE = 50000


def load_embeddings():
    df = pd.read_parquet(EMBEDDINGS_FILE)
    print(f"✅ Loaded embeddings: {EMBEDDINGS_FILE}")
    print(f"   Rows: {len(df):,}")
    return df


def get_latent_columns(df):
    latent_cols = [c for c in df.columns if c.startswith("z")]
    latent_cols = sorted(latent_cols, key=lambda x: int(x.replace("z", "")))

    if not latent_cols:
        raise ValueError("No latent columns found.")

    print(f"✅ Latent dimensions: {len(latent_cols)}")
    return latent_cols


def fit_kmeans_models(df, latent_cols):
    """
    Fit KMeans for several k values and compute silhouette scores.
    """
    X = df[latent_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    models = {}

    if len(X_scaled) > SILHOUETTE_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(len(X_scaled), size=SILHOUETTE_SAMPLE_SIZE, replace=False)
        X_sil = X_scaled[sample_idx]
    else:
        sample_idx = np.arange(len(X_scaled))
        X_sil = X_scaled

    print(f"✅ Silhouette sample size: {len(sample_idx):,}")

    for k in K_VALUES:
        print(f"🚀 Fitting KMeans k={k}...")

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )

        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_sil, labels[sample_idx])

        results.append({
            "k": k,
            "silhouette_score": sil,
            "inertia": model.inertia_,
        })

        models[k] = {
            "model": model,
            "labels": labels,
            "scaler": scaler,
        }

        print(f"   k={k} | silhouette={sil:.4f} | inertia={model.inertia_:.2f}")

    scores = pd.DataFrame(results)

    best_k = int(scores.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    print(f"\n✅ Best k by silhouette: {best_k}")

    return scores, models, best_k


def save_silhouette_scores(scores):
    os.makedirs(os.path.dirname(SILHOUETTE_OUTPUT), exist_ok=True)
    scores.to_csv(SILHOUETTE_OUTPUT, index=False)
    print(f"✅ Saved silhouette scores: {SILHOUETTE_OUTPUT}")


def build_window_clusters(df, labels, best_k):
    """
    Build cluster assignment table.
    """
    out = df[["sample_id", "basin", "start_time", "end_time", "split"]].copy()
    out = out.rename(columns={"basin": "basin_id"})
    out["cluster"] = labels.astype(int)
    out["k"] = best_k

    return out


def save_window_clusters(clusters):
    os.makedirs(os.path.dirname(WINDOW_CLUSTERS_OUTPUT), exist_ok=True)
    clusters.to_parquet(WINDOW_CLUSTERS_OUTPUT, index=False)
    print(f"✅ Saved window clusters: {WINDOW_CLUSTERS_OUTPUT}")


def plot_umap_by_cluster(clusters):
    """
    Plot KMeans clusters on existing UMAP coordinates.
    """
    if not os.path.exists(UMAP_FILE):
        print(f"⚠️ UMAP file not found, skipping cluster UMAP plot: {UMAP_FILE}")
        return

    umap_df = pd.read_parquet(UMAP_FILE)

    plot_df = umap_df.merge(
        clusters[["sample_id", "cluster"]],
        on="sample_id",
        how="inner"
    )

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        plot_df["u1"],
        plot_df["u2"],
        c=plot_df["cluster"],
        s=4,
        alpha=0.6,
        cmap="tab10",
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("KMeans cluster")

    plt.title("UMAP of Hydrological Embeddings Colored by KMeans Regime")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    os.makedirs(os.path.dirname(CLUSTER_FIGURE_OUTPUT), exist_ok=True)
    plt.savefig(CLUSTER_FIGURE_OUTPUT, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved cluster UMAP figure: {CLUSTER_FIGURE_OUTPUT}")


def print_cluster_summary(clusters):
    print("\n" + "=" * 60)
    print("Cluster Summary")
    print("=" * 60)

    counts = clusters.groupby(["split", "cluster"]).size().reset_index(name="n")
    print("\nCounts by split and cluster:")
    print(counts.to_string(index=False))

    overall = clusters["cluster"].value_counts(normalize=True).sort_index()
    print("\nOverall cluster fractions:")
    print(overall.to_string())


def main():
    print("--- Step 9.2: KMeans regime discovery ---")

    embeddings = load_embeddings()
    latent_cols = get_latent_columns(embeddings)

    scores, models, best_k = fit_kmeans_models(embeddings, latent_cols)
    save_silhouette_scores(scores)

    best_labels = models[best_k]["labels"]
    clusters = build_window_clusters(embeddings, best_labels, best_k)

    save_window_clusters(clusters)
    plot_umap_by_cluster(clusters)
    print_cluster_summary(clusters)

    print("\nInterpretation reminder:")
    print("Clusters are 24-month hydrological behavior regimes, not static basin classes.")
    print("The same basin can move between clusters over time.")

    print("--- Done ---")


if __name__ == "__main__":
    main()
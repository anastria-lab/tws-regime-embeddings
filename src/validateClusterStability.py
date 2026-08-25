# validateClusterStability.py
# Step 9.5.1: Validate KMeans cluster stability using Adjusted Rand Index

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_v3.parquet"

OUTPUT_TABLE = "results/tables/kmeans_cluster_stability_ari.csv"
OUTPUT_FIGURE = "results/figures/validation/kmeans_cluster_stability_ari.png"

K_VALUES = [4, 5, 6, 7, 8]
SEEDS = [0, 1, 2, 3, 4, 5, 10, 20, 42, 100]

RANDOM_STATE = 42

# Use sample for speed. Set to None for all samples.
SAMPLE_SIZE = 50000


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


def prepare_embedding_matrix(df, latent_cols):
    X = df[latent_cols].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if SAMPLE_SIZE is not None and len(X) > SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), size=SAMPLE_SIZE, replace=False)
        X = X[idx]
        print(f"✅ Using stability sample: {SAMPLE_SIZE:,} windows")
    else:
        print(f"✅ Using all windows: {len(X):,}")

    return X


def run_kmeans_repeats(X):
    rows = []

    for k in K_VALUES:
        print(f"\n🚀 Running KMeans stability test for k={k}")

        labels_by_seed = {}

        for seed in SEEDS:
            model = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=20,
            )

            labels = model.fit_predict(X)
            labels_by_seed[seed] = labels

            print(f"   Seed {seed} complete")

        seed_list = list(labels_by_seed.keys())

        for i in range(len(seed_list)):
            for j in range(i + 1, len(seed_list)):
                s1 = seed_list[i]
                s2 = seed_list[j]

                ari = adjusted_rand_score(
                    labels_by_seed[s1],
                    labels_by_seed[s2],
                )

                rows.append({
                    "k": k,
                    "seed_1": s1,
                    "seed_2": s2,
                    "adjusted_rand_index": ari,
                })

        k_scores = [r["adjusted_rand_index"] for r in rows if r["k"] == k]
        print(
            f"✅ k={k} ARI mean={np.mean(k_scores):.3f}, "
            f"median={np.median(k_scores):.3f}, "
            f"min={np.min(k_scores):.3f}"
        )

    return pd.DataFrame(rows)


def summarize_stability(ari_df):
    summary = (
        ari_df.groupby("k")["adjusted_rand_index"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )

    print("\n" + "=" * 60)
    print("Cluster Stability Summary")
    print("=" * 60)
    print(summary.to_string(index=False))

    return summary


def save_outputs(ari_df, summary):
    os.makedirs(os.path.dirname(OUTPUT_TABLE), exist_ok=True)

    ari_df.to_csv(OUTPUT_TABLE, index=False)
    print(f"✅ Saved ARI pairwise table: {OUTPUT_TABLE}")

    summary_file = OUTPUT_TABLE.replace(".csv", "_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"✅ Saved ARI summary table: {summary_file}")


def plot_stability(summary):
    os.makedirs(os.path.dirname(OUTPUT_FIGURE), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["k"],
        summary["mean"],
        yerr=summary["std"],
        marker="o",
        capsize=4,
    )

    plt.xlabel("Number of clusters k")
    plt.ylabel("Adjusted Rand Index")
    plt.title("KMeans Cluster Stability Across Random Seeds")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_FIGURE, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved stability figure: {OUTPUT_FIGURE}")


def main():
    print("--- Step 9.5.1: KMeans cluster stability validation ---")

    df = load_embeddings()
    latent_cols = get_latent_columns(df)

    X = prepare_embedding_matrix(df, latent_cols)

    ari_df = run_kmeans_repeats(X)
    summary = summarize_stability(ari_df)

    save_outputs(ari_df, summary)
    plot_stability(summary)

    print("\nInterpretation guide:")
    print("ARI > 0.7  : very stable clustering")
    print("ARI 0.4–0.7: moderately stable")
    print("ARI < 0.3  : unstable clustering")

    print("--- Done ---")


if __name__ == "__main__":
    main()
# validateClusterStability.py
# Step 9.5B: Robust KMeans stability on the CONTINUOUS atlas
#
# Two different questions are tested:
# 1) seed stability: does KMeans converge to the same partition when only
#    initialization changes?
# 2) basin-resampling stability: does the partition remain similar when the
#    set of basins used to fit KMeans changes?
#
# The second test is important because 24-month rolling windows overlap heavily;
# treating windows as independent resampling units would overstate stability.

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_continuous_v3.parquet"
OUTPUT_PAIRWISE = "results/tables/kmeans_cluster_stability_continuous_pairwise.csv"
OUTPUT_SUMMARY = "results/tables/kmeans_cluster_stability_continuous_summary.csv"
OUTPUT_FIGURE = "results/figures/validation/kmeans_cluster_stability_continuous.png"

K_VALUES = list(range(2, 9))
SEEDS = [0, 1, 2, 3, 4, 5, 10, 42]
RANDOM_STATE = 42
N_INIT = 10

FIT_SAMPLE_SIZE = 50000
EVAL_SAMPLE_SIZE = 30000
BASIN_FRACTION = 0.80
BASIN_RESAMPLE_REPEATS = 8


def latent_columns(df):
    cols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    cols = sorted(cols, key=lambda c: int(c[1:]))
    if not cols:
        raise ValueError("No latent columns z1...zN found.")
    return cols


def basin_column(df):
    if "basin" in df.columns:
        return "basin"
    if "basin_id" in df.columns:
        return "basin_id"
    raise ValueError("Embeddings need a basin or basin_id column.")


def pairwise_ari_rows(predictions, k, stability_type):
    rows = []
    keys = list(predictions)
    for a, b in itertools.combinations(keys, 2):
        rows.append({
            "k": k,
            "stability_type": stability_type,
            "run_1": str(a),
            "run_2": str(b),
            "adjusted_rand_index": adjusted_rand_score(predictions[a], predictions[b]),
        })
    return rows


def sample_rows(rng, eligible_idx, n):
    eligible_idx = np.asarray(eligible_idx)
    if len(eligible_idx) <= n:
        return eligible_idx
    return rng.choice(eligible_idx, size=n, replace=False)


def main():
    print("--- Step 9.5B: Robust KMeans stability on continuous embeddings ---")
    df = pd.read_parquet(EMBEDDINGS_FILE).reset_index(drop=True)
    print(f"✅ Loaded continuous embeddings: {EMBEDDINGS_FILE}")
    print(f"   Rows: {len(df):,}")

    zcols = latent_columns(df)
    bcol = basin_column(df)
    print(f"✅ Latent dimensions: {len(zcols)}")
    print(f"✅ Basins: {df[bcol].nunique():,}")

    X = df[zcols].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("NaN/inf found in embeddings.")

    # Match production clustering: one StandardScaler fitted to the entire
    # continuous atlas before KMeans.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rng = np.random.default_rng(RANDOM_STATE)
    all_idx = np.arange(len(df))
    eval_idx = sample_rows(rng, all_idx, EVAL_SAMPLE_SIZE)
    fit_idx_fixed = sample_rows(rng, all_idx, FIT_SAMPLE_SIZE)
    X_eval = Xs[eval_idx]

    print(f"✅ Fixed evaluation sample: {len(eval_idx):,} windows")
    print(f"✅ Fixed seed-stability fit sample: {len(fit_idx_fixed):,} windows")

    basins = df[bcol].drop_duplicates().to_numpy()
    n_basin_fit = max(2, int(np.floor(BASIN_FRACTION * len(basins))))
    print(
        f"✅ Basin-resampling test: {BASIN_RESAMPLE_REPEATS} repeats, "
        f"{BASIN_FRACTION:.0%} of basins per repeat ({n_basin_fit:,}/{len(basins):,})"
    )

    rows = []

    for k in K_VALUES:
        print(f"\n🚀 k={k}: initialization stability")
        seed_predictions = {}
        for seed in SEEDS:
            model = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT)
            model.fit(Xs[fit_idx_fixed])
            seed_predictions[f"seed_{seed}"] = model.predict(X_eval)
            print(f"   seed {seed} complete")
        rows.extend(pairwise_ari_rows(seed_predictions, k, "seed_fixed_sample"))

        print(f"🚀 k={k}: basin-resampling stability")
        basin_predictions = {}
        for r in range(BASIN_RESAMPLE_REPEATS):
            rrng = np.random.default_rng(RANDOM_STATE + 1000 + 97 * r + k)
            chosen_basins = rrng.choice(basins, size=n_basin_fit, replace=False)
            eligible = np.flatnonzero(df[bcol].isin(chosen_basins).to_numpy())
            fit_idx = sample_rows(rrng, eligible, FIT_SAMPLE_SIZE)

            model = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE + r,
                n_init=N_INIT,
            )
            model.fit(Xs[fit_idx])
            basin_predictions[f"basin_repeat_{r}"] = model.predict(X_eval)
            print(
                f"   repeat {r} complete | basins={len(chosen_basins):,} | "
                f"fit_windows={len(fit_idx):,}"
            )
        rows.extend(pairwise_ari_rows(basin_predictions, k, "basin_resample"))

    ari = pd.DataFrame(rows)
    summary = (
        ari.groupby(["stability_type", "k"])["adjusted_rand_index"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )

    os.makedirs(os.path.dirname(OUTPUT_PAIRWISE), exist_ok=True)
    ari.to_csv(OUTPUT_PAIRWISE, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n============================================================")
    print("Continuous-atlas stability summary")
    print("============================================================")
    print(summary.to_string(index=False))
    print(f"\n✅ Saved: {OUTPUT_PAIRWISE}")
    print(f"✅ Saved: {OUTPUT_SUMMARY}")

    os.makedirs(os.path.dirname(OUTPUT_FIGURE), exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    for stability_type, g in summary.groupby("stability_type"):
        g = g.sort_values("k")
        plt.errorbar(
            g["k"], g["mean"], yerr=g["std"], marker="o", capsize=4,
            label=stability_type,
        )
    plt.xlabel("Number of clusters k")
    plt.ylabel("Adjusted Rand Index on fixed evaluation windows")
    plt.title("KMeans stability: initialization vs basin resampling")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {OUTPUT_FIGURE}")

    print("\nInterpretation notes:")
    print("- seed_fixed_sample tests optimization/init stability only")
    print("- basin_resample is the stronger robustness test for this study")
    print("- ARI is label-permutation invariant")
    print("- rolling windows overlap by 23/24 months, so window-level bootstrap is intentionally avoided")
    print("- high stability does not imply strong cluster separation; read together with silhouette and physical validation")
    print("--- Done ---")


if __name__ == "__main__":
    main()

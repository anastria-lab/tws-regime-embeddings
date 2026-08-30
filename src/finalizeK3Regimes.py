# finalizeK3Regimes.py
# Step 9D: Freeze the production regime solution at k=3.

import os
import json
import shutil
from itertools import permutations

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_continuous_v3.parquet"
CANDIDATE_FILE = "results/tables/regime_physical_comparison/candidate_assignments_k234.parquet"
UMAP_FILE = "data/processed/embeddings_umap_continuous.parquet"

PRODUCTION_OUTPUT = "data/processed/window_clusters.parquet"
BACKUP_OUTPUT = "data/processed/window_clusters_pre_final_k3_backup.parquet"

MODEL_OUTPUT = "models/kmeans_regime_k3.joblib"
CONFIG_OUTPUT = "configs/kmeans_regime_k3.json"

SUMMARY_OUTPUT = "results/tables/final_k3_regime_summary.csv"
SPLIT_SUMMARY_OUTPUT = "results/tables/final_k3_regime_summary_by_period.csv"

FIG_OUTPUT = "results/figures/latent_space_NEW/umap_by_final_k3_regime.png"
GENERIC_FIG_OUTPUT = "results/figures/latent_space_NEW/umap_by_kmeans_cluster.png"

FINAL_K = 3
RANDOM_STATE = 42
N_INIT = 20


def latent_columns(df):
    cols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def load_embeddings():
    df = pd.read_parquet(EMBEDDINGS_FILE)
    zcols = latent_columns(df)

    required = {"sample_id", "basin", "start_time", "end_time", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Continuous embeddings missing required columns: {missing}")

    if len(zcols) != 16:
        raise ValueError(f"Expected 16 latent dimensions, found {len(zcols)}")

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values found in continuous embeddings.")

    print(f"✅ Loaded continuous embeddings: {EMBEDDINGS_FILE}")
    print(f"   Rows: {len(df):,}")
    print(f"   Basins: {df['basin'].nunique():,}")
    print(f"   Latent dimensions: {len(zcols)}")
    return df, zcols


def fit_final_k3(df, zcols):
    X = df[zcols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print("🚀 Fitting FINAL KMeans k=3 in standardized 16-D latent space...")
    model = KMeans(
        n_clusters=FINAL_K,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    raw_labels = model.fit_predict(Xs).astype(int)
    return scaler, model, raw_labels


def best_label_mapping(raw_labels, target_labels, k):
    best_map = None
    best_match = -1

    for perm in permutations(range(k)):
        mapping = {raw: perm[raw] for raw in range(k)}
        mapped = np.array([mapping[x] for x in raw_labels], dtype=int)
        matches = int(np.sum(mapped == target_labels))
        if matches > best_match:
            best_match = matches
            best_map = mapping

    return best_map, best_match


def align_to_step9c_if_available(df, raw_labels):
    identity = {i: i for i in range(FINAL_K)}

    if not os.path.exists(CANDIDATE_FILE):
        print("ℹ️ Step-9C candidate assignment file not found.")
        print("   Using deterministic fitted k=3 labels directly.")
        return raw_labels, identity, None

    cand = pd.read_parquet(CANDIDATE_FILE)
    if "cluster_k3" not in cand.columns or "sample_id" not in cand.columns:
        print("⚠️ Candidate file exists but lacks sample_id/cluster_k3.")
        print("   Using deterministic fitted labels directly.")
        return raw_labels, identity, None

    target = df[["sample_id"]].merge(
        cand[["sample_id", "cluster_k3"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if target["cluster_k3"].isna().any():
        n = int(target["cluster_k3"].isna().sum())
        raise ValueError(f"Step-9C candidate file is missing {n:,} continuous sample_ids.")

    target_labels = target["cluster_k3"].to_numpy(dtype=int)
    ari = adjusted_rand_score(target_labels, raw_labels)

    mapping, matches = best_label_mapping(raw_labels, target_labels, FINAL_K)
    mapped = np.array([mapping[x] for x in raw_labels], dtype=int)
    agreement = matches / len(mapped)

    print("✅ Compared final fit with Step-9C k=3 candidate solution")
    print(f"   ARI: {ari:.6f}")
    print(f"   Best label-aligned exact agreement: {agreement:.6%}")
    print(f"   Label map raw→Step9C: {mapping}")

    if ari < 0.99:
        raise ValueError(
            "Final k=3 fit does not reproduce the Step-9C candidate solution "
            f"closely enough (ARI={ari:.4f}). Refusing to finalize."
        )

    if agreement < 0.99:
        raise ValueError(
            "After label alignment, exact agreement with Step-9C is below 99%. "
            "Refusing to finalize."
        )

    return mapped, mapping, ari


def build_production_table(df, labels):
    out = df[["sample_id", "basin", "start_time", "end_time", "split"]].copy()
    out = out.rename(columns={"basin": "basin_id"})
    out["cluster"] = labels.astype(np.int16)
    out["k"] = FINAL_K

    if out["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in final production table.")

    if out["cluster"].nunique() != FINAL_K:
        raise ValueError(
            f"Expected {FINAL_K} final clusters, found {out['cluster'].nunique()}."
        )

    return out


def backup_previous():
    if not os.path.exists(PRODUCTION_OUTPUT):
        return

    if os.path.exists(BACKUP_OUTPUT):
        print(f"ℹ️ Existing backup retained: {BACKUP_OUTPUT}")
        return

    shutil.copy2(PRODUCTION_OUTPUT, BACKUP_OUTPUT)
    print(f"✅ Backed up previous production clusters: {BACKUP_OUTPUT}")


def save_outputs(production, scaler, model, label_map, candidate_ari):
    os.makedirs(os.path.dirname(PRODUCTION_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_OUTPUT), exist_ok=True)

    production.to_parquet(PRODUCTION_OUTPUT, index=False)

    bundle = {
        "scaler": scaler,
        "kmeans": model,
        "label_map_raw_to_final": label_map,
        "latent_dim": 16,
        "k": FINAL_K,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "embedding_file": EMBEDDINGS_FILE,
    }
    joblib.dump(bundle, MODEL_OUTPUT)

    config = {
        "final_k": FINAL_K,
        "selection_status": "final production regime solution",
        "selection_basis": (
            "parsimony + continuous-atlas cluster metrics + basin-resampling "
            "stability + Step-9C physical fingerprint comparison"
        ),
        "clustering_space": "standardized 16-D continuous latent embeddings",
        "umap_role": "visualization only; not used for clustering",
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "candidate_step9c_ari": None if candidate_ari is None else float(candidate_ari),
        "label_map_raw_to_final": {str(k): int(v) for k, v in label_map.items()},
        "important_interpretation": (
            "Three recurrent regions within a continuous hydrological manifold; "
            "cluster names remain provisional until independent SPEI validation."
        ),
    }
    with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    summary = (
        production.groupby("cluster")
        .agg(n_windows=("sample_id", "size"), n_basins=("basin_id", "nunique"))
        .reset_index()
    )
    summary["window_fraction"] = summary["n_windows"] / len(production)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    split_summary = (
        production.groupby(["split", "cluster"])
        .size()
        .reset_index(name="n_windows")
    )
    split_totals = split_summary.groupby("split")["n_windows"].transform("sum")
    split_summary["fraction_within_period"] = split_summary["n_windows"] / split_totals
    split_summary.to_csv(SPLIT_SUMMARY_OUTPUT, index=False)

    print(f"✅ Saved FINAL production clusters: {PRODUCTION_OUTPUT}")
    print(f"✅ Saved reusable KMeans/scaler bundle: {MODEL_OUTPUT}")
    print(f"✅ Saved config: {CONFIG_OUTPUT}")
    print(f"✅ Saved summary: {SUMMARY_OUTPUT}")
    print(f"✅ Saved period summary: {SPLIT_SUMMARY_OUTPUT}")

    return summary, split_summary


def plot_final_umap(production):
    if not os.path.exists(UMAP_FILE):
        print(f"⚠️ UMAP file not found; skipping final cluster UMAP: {UMAP_FILE}")
        return

    umap = pd.read_parquet(UMAP_FILE)
    plot_df = umap.merge(
        production[["sample_id", "cluster"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        plot_df["u1"],
        plot_df["u2"],
        c=plot_df["cluster"],
        s=4,
        alpha=0.6,
        cmap="tab10",
        vmin=-0.5,
        vmax=FINAL_K - 0.5,
    )
    cbar = plt.colorbar(sc, ticks=np.arange(FINAL_K))
    cbar.set_label("Final KMeans regime (k=3)")
    plt.title("Continuous Hydrological State Space — Final k=3 Regimes")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()

    os.makedirs(os.path.dirname(FIG_OUTPUT), exist_ok=True)
    plt.savefig(FIG_OUTPUT, dpi=220, bbox_inches="tight")
    plt.savefig(GENERIC_FIG_OUTPUT, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved final k=3 UMAP: {FIG_OUTPUT}")
    print(f"✅ Updated generic cluster UMAP: {GENERIC_FIG_OUTPUT}")


def main():
    print("--- Step 9D: FINALIZE production regime solution at k=3 ---")

    df, zcols = load_embeddings()
    scaler, model, raw_labels = fit_final_k3(df, zcols)
    final_labels, label_map, candidate_ari = align_to_step9c_if_available(df, raw_labels)
    production = build_production_table(df, final_labels)

    backup_previous()
    summary, split_summary = save_outputs(
        production, scaler, model, label_map, candidate_ari
    )
    plot_final_umap(production)

    print("\n" + "=" * 72)
    print("FINAL k=3 REGIME SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))

    print("\nCounts by period:")
    print(split_summary.to_string(index=False))

    print("\n✅ k=3 is now the production regime solution.")
    print("⚠️ Cluster IDs remain numeric until independent SPEI validation.")
    print("⚠️ Do not rerun the older automatic clusterEmbeddingRegimes.py,")
    print("   because it can overwrite this finalized k=3 production file.")
    print("--- Done ---")


if __name__ == "__main__":
    main()

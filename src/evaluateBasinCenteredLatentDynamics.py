# evaluateBasinCenteredLatentDynamics.py
# Controlled experiment for the poster's core question: BASIN DYNAMICS.
#
# We do NOT retrain the autoencoder.
#
# Instead, for each basin we define its reference latent state from TRAIN-period
# windows only, then express every continuous 24-month state as a departure
# from that basin-specific reference:
#
#       z_relative(basin, time) = z(basin, time) - mean_train[z(basin)]
#
# This removes persistent basin-specific offset while retaining temporal
# changes, direction, amplitude, timing, persistence, and trajectories.
#
# We compare:
#   RAW latent space       vs
#   BASIN-CENTERED latent space
#
# for k=2 and k=3 only.
#
# This script DOES NOT overwrite the finalized production cluster file.
# It is a diagnostic experiment. We decide what to do only after inspecting it.

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

EMBEDDINGS_FILE = "data/processed/embeddings_window_level_continuous_v3.parquet"

# Produced by the regime-semantic audit that was just run.
PHYSICAL_FILE = (
    "results/tables/regime_semantic_audit/"
    "window_physical_metrics_final_k3.parquet"
)

OUT_DIR = "results/tables/basin_centered_dynamics"
FIG_DIR = "results/figures/basin_centered_dynamics"

CENTERED_EMBEDDINGS_OUTPUT = (
    "data/processed/embeddings_window_level_continuous_basin_centered_v3.parquet"
)
CENTROIDS_OUTPUT = os.path.join(OUT_DIR, "basin_train_latent_centroids.csv")
ASSIGNMENTS_OUTPUT = os.path.join(OUT_DIR, "raw_vs_centered_candidate_assignments.parquet")
METRICS_OUTPUT = os.path.join(OUT_DIR, "raw_vs_centered_cluster_metrics.csv")
STABILITY_OUTPUT = os.path.join(OUT_DIR, "raw_vs_centered_basin_resampling_stability.csv")
PROFILES_OUTPUT = os.path.join(OUT_DIR, "raw_vs_centered_basin_balanced_profiles.csv")
PAIRWISE_OUTPUT = os.path.join(OUT_DIR, "raw_vs_centered_pairwise_semantic_consistency.csv")
ORDER_OUTPUT = os.path.join(OUT_DIR, "raw_vs_centered_full_ordering_consistency.csv")
VARIANCE_OUTPUT = os.path.join(OUT_DIR, "raw_vs_centered_latent_variance_decomposition.csv")

K_VALUES = [2, 3]
RANDOM_STATE = 42
N_INIT = 20

METRIC_SAMPLE = 50000
STABILITY_EVAL_SAMPLE = 30000
STABILITY_FIT_SAMPLE = 50000
BASIN_RESAMPLE_FRACTION = 0.80
BASIN_RESAMPLE_REPEATS = 6

KEY_PHYSICAL_METRICS = [
    "mean_lwe_thickness_anomaly_normalized",
    "mean_lwe_thickness_long",
    "mean_tp_anomaly_normalized",
    "mean_swvl4_anomaly_normalized",
    "mean_swvl4_long",
    "spei12_window_mean",
    "spei12_drought_fraction_le_m1",
    "spei12_wet_fraction_ge_p1",
]


def latent_columns(df):
    cols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    return sorted(cols, key=lambda c: int(c[1:]))


def load_embeddings():
    df = pd.read_parquet(EMBEDDINGS_FILE).copy()
    zcols = latent_columns(df)

    required = {"sample_id", "basin", "start_time", "end_time", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Continuous embeddings missing required columns: {missing}")

    if len(zcols) != 16:
        raise ValueError(f"Expected 16 latent dimensions, found {len(zcols)}")

    df["basin_id"] = df["basin"].astype("int64")
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in continuous embeddings.")

    print(f"✅ Loaded continuous embeddings: {len(df):,} windows")
    print(f"✅ Basins: {df['basin_id'].nunique():,}")
    print(f"✅ Latent dimensions: {len(zcols)}")

    return df, zcols


def build_train_centroids(df, zcols):
    """
    Basin reference state = mean latent vector of fully train-contained windows.

    We intentionally do NOT use future/validation/test windows to define a
    basin's reference state.
    """
    train = df[df["split"] == "train"].copy()

    counts = train.groupby("basin_id").size()
    all_basins = pd.Index(sorted(df["basin_id"].unique()))
    missing = all_basins.difference(counts.index)

    if len(missing):
        raise ValueError(
            f"{len(missing)} basins have no train-contained windows. "
            f"Cannot define leakage-safe basin reference centroids. "
            f"First examples: {missing[:10].tolist()}"
        )

    centroids = (
        train.groupby("basin_id")[zcols]
        .mean()
        .reset_index()
    )
    centroids["n_train_reference_windows"] = (
        centroids["basin_id"].map(counts).astype(int)
    )

    print(
        "✅ Defined basin-specific reference states from train windows only "
        f"({train['end_time'].min().date()} → {train['end_time'].max().date()})"
    )
    print(
        f"   Train reference windows per basin: "
        f"min={counts.min()}, median={int(counts.median())}, max={counts.max()}"
    )

    return centroids


def center_embeddings(df, centroids, zcols):
    crename = {z: f"{z}_ref" for z in zcols}
    c = centroids[["basin_id"] + zcols].rename(columns=crename)

    out = df.merge(
        c,
        on="basin_id",
        how="left",
        validate="many_to_one",
    )

    centered_cols = []
    for z in zcols:
        cz = f"c_{z}"
        out[cz] = out[z] - out[f"{z}_ref"]
        centered_cols.append(cz)

    keep = [
        "sample_id", "basin", "basin_id", "split",
        "start_time", "end_time",
    ] + centered_cols

    centered = out[keep].copy()

    # QC: the mean centered TRAIN state for each basin must be ~zero.
    qc = (
        centered[centered["split"] == "train"]
        .groupby("basin_id")[centered_cols]
        .mean()
        .abs()
    )
    max_abs = float(qc.to_numpy().max())

    print(f"✅ Basin-centered latent embeddings created")
    print(f"   Max |train-basin centered mean| across all dimensions: {max_abs:.3e}")

    if max_abs > 1e-5:
        print("⚠️ Centering QC is larger than expected; inspect before interpretation.")

    return centered, centered_cols


def variance_decomposition(df, cols, representation_name):
    """
    Descriptive total = between + within variance decomposition after global
    standardization so dimensions are comparable.
    """
    X = df[cols].to_numpy(dtype=np.float64)
    Xs = StandardScaler().fit_transform(X)

    basin = df["basin_id"].to_numpy()
    grand = Xs.mean(axis=0)

    total = float(np.sum((Xs - grand) ** 2))
    between = 0.0
    within = 0.0

    for b in np.unique(basin):
        G = Xs[basin == b]
        mu = G.mean(axis=0)
        between += float(len(G) * np.sum((mu - grand) ** 2))
        within += float(np.sum((G - mu) ** 2))

    return {
        "representation": representation_name,
        "total_ss": total,
        "between_basin_ss": between,
        "within_basin_ss": within,
        "between_basin_fraction": between / total,
        "within_basin_fraction": within / total,
        "decomposition_error": total - between - within,
    }


def fixed_sample_indices(n, size, seed):
    rng = np.random.default_rng(seed)
    size = min(size, n)
    return np.sort(rng.choice(n, size=size, replace=False))


def fit_solution(df, cols, representation, k, metric_idx):
    X = df[cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )
    labels = model.fit_predict(Xs).astype(np.int16)

    Xm = Xs[metric_idx]
    ym = labels[metric_idx]

    row = {
        "representation": representation,
        "k": k,
        "n_windows": len(df),
        "silhouette_score": float(silhouette_score(Xm, ym)),
        "calinski_harabasz": float(calinski_harabasz_score(Xm, ym)),
        "davies_bouldin": float(davies_bouldin_score(Xm, ym)),
        "inertia": float(model.inertia_),
        "smallest_cluster_fraction": float(
            pd.Series(labels).value_counts(normalize=True).min()
        ),
        "largest_cluster_fraction": float(
            pd.Series(labels).value_counts(normalize=True).max()
        ),
    }

    return scaler, model, labels, row


def basin_resampling_stability(df, cols, k, representation):
    """
    Stronger robustness diagnostic:
    repeatedly fit on windows from 80% of basins, then predict on one fixed
    global evaluation sample. Pairwise ARI is label-permutation invariant.

    Because rolling windows overlap heavily, we resample basins, not windows.
    """
    X = df[cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    basins = df["basin_id"].to_numpy()
    unique_basins = np.unique(basins)
    n_pick = int(round(BASIN_RESAMPLE_FRACTION * len(unique_basins)))

    eval_idx = fixed_sample_indices(
        len(df), STABILITY_EVAL_SAMPLE, RANDOM_STATE + 100 + k
    )
    Xe = Xs[eval_idx]

    rng = np.random.default_rng(RANDOM_STATE + 1000 + k)
    predictions = []

    print(f"   basin-resampling stability: {representation}, k={k}")
    for rep in range(BASIN_RESAMPLE_REPEATS):
        chosen = rng.choice(unique_basins, size=n_pick, replace=False)
        eligible = np.flatnonzero(np.isin(basins, chosen))

        fit_n = min(STABILITY_FIT_SAMPLE, len(eligible))
        fit_idx = rng.choice(eligible, size=fit_n, replace=False)

        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE + rep,
            n_init=10,
        )
        km.fit(Xs[fit_idx])
        predictions.append(km.predict(Xe))

        print(
            f"      repeat {rep}: basins={n_pick}, fit_windows={fit_n:,}"
        )

    aris = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            aris.append(adjusted_rand_score(predictions[i], predictions[j]))

    aris = np.asarray(aris, dtype=float)

    return {
        "representation": representation,
        "k": k,
        "n_repeats": BASIN_RESAMPLE_REPEATS,
        "basin_fraction_per_repeat": BASIN_RESAMPLE_FRACTION,
        "mean_ari": float(np.mean(aris)),
        "median_ari": float(np.median(aris)),
        "std_ari": float(np.std(aris)),
        "min_ari": float(np.min(aris)),
        "max_ari": float(np.max(aris)),
    }


def load_physical():
    if not os.path.exists(PHYSICAL_FILE):
        raise FileNotFoundError(
            f"Required audit output not found:\n  {PHYSICAL_FILE}\n"
            "Run auditRegimeSemanticsAndBasinIdentity.py first."
        )

    p = pd.read_parquet(PHYSICAL_FILE).copy()

    if "sample_id" not in p.columns or "basin_id" not in p.columns:
        raise ValueError("Physical audit file requires sample_id and basin_id.")

    usable = [m for m in KEY_PHYSICAL_METRICS if m in p.columns]

    if len(usable) < 5:
        raise ValueError(
            f"Too few physical metrics available for comparison: {usable}"
        )

    print(f"✅ Loaded physical audit metrics: {len(usable)} key metrics")
    return p[["sample_id", "basin_id"] + usable].copy(), usable


def basin_balanced_profiles(assignments, physical, representation, k, metrics):
    label_col = f"{representation}_k{k}"

    d = assignments[
        ["sample_id", "basin_id", label_col]
    ].merge(
        physical,
        on=["sample_id", "basin_id"],
        how="left",
        validate="one_to_one",
    )

    # First aggregate overlapping windows within basin × state.
    br = (
        d.groupby(["basin_id", label_col], observed=True)[metrics]
        .mean()
        .reset_index()
    )

    rows = []
    for cluster, g in br.groupby(label_col):
        row = {
            "representation": representation,
            "k": k,
            "cluster": int(cluster),
            "n_basin_state_members": len(g),
            "n_unique_basins": g["basin_id"].nunique(),
        }
        for m in metrics:
            row[m] = float(g[m].mean())
        rows.append(row)

    profile = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    return d, br, profile


def pairwise_semantic_consistency(br, profile, representation, k, metrics):
    """
    For every regime pair and metric:
    how often does the paired within-basin difference have the same direction
    as the basin-balanced global difference?
    """
    rows = []
    gp = profile.set_index("cluster")
    label_col = f"{representation}_k{k}"
    clusters = sorted(profile["cluster"].tolist())

    for metric in metrics:
        pivot = br.pivot(
            index="basin_id",
            columns=label_col,
            values=metric,
        )

        for i, a in enumerate(clusters):
            for b in clusters[i + 1:]:
                if a not in pivot.columns or b not in pivot.columns:
                    continue

                pair = pivot[[a, b]].dropna()
                if pair.empty:
                    continue

                global_diff = float(gp.loc[a, metric] - gp.loc[b, metric])
                diff = (pair[a] - pair[b]).to_numpy(dtype=float)

                if global_diff > 0:
                    same = diff > 0
                    opposite = diff < 0
                elif global_diff < 0:
                    same = diff < 0
                    opposite = diff > 0
                else:
                    same = np.isclose(diff, 0)
                    opposite = ~same

                rows.append({
                    "representation": representation,
                    "k": k,
                    "metric": metric,
                    "cluster_a": int(a),
                    "cluster_b": int(b),
                    "n_paired_basins": len(diff),
                    "global_a_minus_b": global_diff,
                    "same_global_direction_fraction": float(np.mean(same)),
                    "opposite_global_direction_fraction": float(np.mean(opposite)),
                })

    return pd.DataFrame(rows)


def rank_tuple(values):
    return tuple(values.sort_values(ascending=False).index.astype(int).tolist())


def spearman_small(a, b):
    ar = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b).rank(method="average").to_numpy(dtype=float)

    if np.std(ar) == 0 or np.std(br) == 0:
        return np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def ordering_consistency(br, profile, representation, k, metrics):
    """
    Complete state ranking consistency.

    k=2:
        exact ordering == pairwise direction consistency.
    k=3:
        asks whether all three states have the same high→mid→low ordering
        within a basin as in the basin-balanced global profile.
    """
    rows = []
    label_col = f"{representation}_k{k}"
    gp = profile.set_index("cluster")
    clusters = sorted(profile["cluster"].tolist())

    for metric in metrics:
        global_values = gp[metric].dropna()
        if len(global_values) != k:
            continue

        global_order = rank_tuple(global_values)

        pivot = br.pivot(
            index="basin_id",
            columns=label_col,
            values=metric,
        )

        if not set(clusters).issubset(pivot.columns):
            continue

        complete = pivot[clusters].dropna()
        if complete.empty:
            continue

        exact = []
        rhos = []

        gv = global_values.reindex(clusters).to_numpy(dtype=float)

        for _, row in complete.iterrows():
            lv = row[clusters].to_numpy(dtype=float)
            local_values = pd.Series(lv, index=clusters)
            local_order = rank_tuple(local_values)

            exact.append(local_order == global_order)
            rhos.append(spearman_small(lv, gv))

        rhos = np.asarray(rhos, dtype=float)

        rows.append({
            "representation": representation,
            "k": k,
            "metric": metric,
            "global_order_high_to_low": ">".join(map(str, global_order)),
            "n_basins_with_all_states": len(complete),
            "exact_global_order_fraction": float(np.mean(exact)),
            "mean_regime_rank_spearman": float(np.nanmean(rhos)),
            "fraction_negative_rank_agreement": float(np.nanmean(rhos < 0)),
        })

    return pd.DataFrame(rows)


def choose_extreme_pair(profile, metric):
    """
    For concise comparison, identify globally highest and lowest cluster for a
    metric; this avoids assuming cluster numeric labels have physical meaning.
    """
    p = profile.set_index("cluster")[metric].dropna()
    return int(p.idxmax()), int(p.idxmin())


def plot_cluster_metrics(metrics):
    m = metrics.copy()

    for metric, ylabel, filename in [
        ("silhouette_score", "Silhouette score", "raw_vs_centered_silhouette.png"),
        ("davies_bouldin", "Davies-Bouldin score (lower better)", "raw_vs_centered_db.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for rep in ["raw", "centered"]:
            sub = m[m["representation"] == rep].sort_values("k")
            plt.plot(sub["k"], sub[metric], marker="o", label=rep)

        plt.xticks(K_VALUES)
        plt.xlabel("k")
        plt.ylabel(ylabel)
        plt.title("Raw vs Basin-Centered Latent State Clustering")
        plt.legend()
        plt.tight_layout()

        out = os.path.join(FIG_DIR, filename)
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {out}")


def plot_semantic_consistency(order):
    key = order[order["metric"].isin(KEY_PHYSICAL_METRICS)].copy()
    if key.empty:
        return

    key["solution"] = (
        key["representation"]
        + "_k"
        + key["k"].astype(str)
    )

    # Median exact ordering across the key metrics.
    med = (
        key.groupby("solution")["exact_global_order_fraction"]
        .median()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(med.index, med.values)
    plt.ylim(0, 1)
    plt.ylabel("Median exact cross-basin physical ordering consistency")
    plt.xlabel("Latent-state solution")
    plt.title("Does Basin Centering Improve Dynamic-State Semantics?")
    plt.xticks(rotation=20)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "semantic_consistency_solution_comparison.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_variance_decomposition(var):
    v = var.set_index("representation").reindex(["raw", "centered"])

    x = np.arange(len(v))
    plt.figure(figsize=(7, 5))
    plt.bar(x, v["between_basin_fraction"].to_numpy())
    plt.xticks(x, ["Raw latent", "Basin-centered latent"])
    plt.ylim(0, max(0.2, float(v["between_basin_fraction"].max()) * 1.25))
    plt.ylabel("Between-basin fraction of standardized 16-D variance")
    plt.title("Persistent Basin Offset Before and After Centering")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "raw_vs_centered_between_basin_variance.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def print_solution_summary(metrics, stability, profiles, pairwise, order, variance):
    print("\n" + "=" * 90)
    print("BASIN-CENTERED DYNAMIC-STATE EXPERIMENT")
    print("=" * 90)

    print("\nA) Raw vs basin-centered latent variance")
    print(
        variance[
            [
                "representation",
                "between_basin_fraction",
                "within_basin_fraction",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x: .3f}")
    )

    print("\nB) Cluster geometry")
    print(
        metrics[
            [
                "representation", "k",
                "silhouette_score",
                "calinski_harabasz",
                "davies_bouldin",
                "smallest_cluster_fraction",
                "largest_cluster_fraction",
            ]
        ].sort_values(["representation", "k"])
        .to_string(index=False, float_format=lambda x: f"{x: .3f}")
    )

    print("\nC) Basin-resampling stability")
    print(
        stability[
            [
                "representation", "k",
                "mean_ari", "median_ari", "min_ari", "max_ari",
            ]
        ].sort_values(["representation", "k"])
        .to_string(index=False, float_format=lambda x: f"{x: .3f}")
    )

    print("\nD) Physical semantic consistency — complete state ordering")
    show_metrics = [m for m in KEY_PHYSICAL_METRICS if m in order["metric"].unique()]
    show = order[order["metric"].isin(show_metrics)].copy()
    print(
        show[
            [
                "representation", "k", "metric",
                "global_order_high_to_low",
                "n_basins_with_all_states",
                "exact_global_order_fraction",
                "mean_regime_rank_spearman",
                "fraction_negative_rank_agreement",
            ]
        ].sort_values(["metric", "representation", "k"])
        .to_string(index=False, float_format=lambda x: f"{x: .3f}")
    )

    print("\nE) Extreme-state contrast consistency")
    rows = []
    for (rep, k), prof in profiles.groupby(["representation", "k"]):
        for metric in [
            "mean_lwe_thickness_anomaly_normalized",
            "mean_lwe_thickness_long",
            "spei12_window_mean",
            "mean_swvl4_long",
        ]:
            if metric not in prof.columns:
                continue

            hi, lo = choose_extreme_pair(prof, metric)

            p = pairwise[
                (pairwise["representation"] == rep)
                & (pairwise["k"] == k)
                & (pairwise["metric"] == metric)
            ].copy()

            # pairwise table stores numerical a<b. Same-direction fraction is
            # invariant to reversing which member we call high/low.
            a, b = sorted([hi, lo])
            q = p[
                (p["cluster_a"] == a)
                & (p["cluster_b"] == b)
            ]

            if q.empty:
                continue

            r = q.iloc[0]
            rows.append({
                "representation": rep,
                "k": k,
                "metric": metric,
                "global_high_cluster": hi,
                "global_low_cluster": lo,
                "n_paired_basins": int(r["n_paired_basins"]),
                "same_global_direction_fraction": r["same_global_direction_fraction"],
                "opposite_global_direction_fraction": r["opposite_global_direction_fraction"],
            })

    extreme = pd.DataFrame(rows)
    if len(extreme):
        print(
            extreme.to_string(
                index=False,
                float_format=lambda x: f"{x: .3f}"
            )
        )

    print("\nDecision rule:")
    print("- We WANT dynamic states, so basin centering is useful only if it preserves/improves")
    print("  cluster stability while making physical state meaning more consistent across basins.")
    print("- Prefer the SMALLEST k that gives a stable and physically portable dynamic-state contrast.")
    print("- If centered k=2 is clearly cleaner than centered k=3, that is a strong result:")
    print("  the continuous trajectory remains rich, while two recurrent relative-state poles")
    print("  summarize departures from each basin's own reference condition.")
    print("- If centering does not improve semantic portability, do NOT adopt it automatically.")
    print("- This experiment does not alter the current production atlas.")


def main():
    print("--- Basin-centered latent dynamics experiment ---")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CENTERED_EMBEDDINGS_OUTPUT), exist_ok=True)

    emb, zcols = load_embeddings()
    centroids = build_train_centroids(emb, zcols)
    centered, centered_cols = center_embeddings(emb, centroids, zcols)

    centroids.to_csv(CENTROIDS_OUTPUT, index=False)
    centered.to_parquet(CENTERED_EMBEDDINGS_OUTPUT, index=False)
    print(f"✅ Saved: {CENTROIDS_OUTPUT}")
    print(f"✅ Saved: {CENTERED_EMBEDDINGS_OUTPUT}")

    physical, physical_metrics = load_physical()

    assignments = emb[
        ["sample_id", "basin_id", "split", "start_time", "end_time"]
    ].copy()

    variance_rows = [
        variance_decomposition(emb, zcols, "raw"),
        variance_decomposition(centered, centered_cols, "centered"),
    ]
    variance = pd.DataFrame(variance_rows)

    metric_idx = fixed_sample_indices(
        len(emb), METRIC_SAMPLE, RANDOM_STATE
    )

    metric_rows = []
    stability_rows = []
    profile_frames = []
    pairwise_frames = []
    order_frames = []

    representations = {
        "raw": (emb, zcols),
        "centered": (centered, centered_cols),
    }

    for representation, (rdf, cols) in representations.items():
        for k in K_VALUES:
            print(f"\n🚀 {representation}: fitting full KMeans k={k}...")
            scaler, model, labels, metric_row = fit_solution(
                rdf, cols, representation, k, metric_idx
            )
            assignments[f"{representation}_k{k}"] = labels
            metric_rows.append(metric_row)

            stab = basin_resampling_stability(
                rdf, cols, k, representation
            )
            stability_rows.append(stab)

            d, br, profile = basin_balanced_profiles(
                assignments,
                physical,
                representation,
                k,
                physical_metrics,
            )
            profile_frames.append(profile)

            pairwise_frames.append(
                pairwise_semantic_consistency(
                    br,
                    profile,
                    representation,
                    k,
                    physical_metrics,
                )
            )
            order_frames.append(
                ordering_consistency(
                    br,
                    profile,
                    representation,
                    k,
                    physical_metrics,
                )
            )

    metrics = pd.DataFrame(metric_rows)
    stability = pd.DataFrame(stability_rows)
    profiles = pd.concat(profile_frames, ignore_index=True)
    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    order = pd.concat(order_frames, ignore_index=True)

    assignments.to_parquet(ASSIGNMENTS_OUTPUT, index=False)
    metrics.to_csv(METRICS_OUTPUT, index=False)
    stability.to_csv(STABILITY_OUTPUT, index=False)
    profiles.to_csv(PROFILES_OUTPUT, index=False)
    pairwise.to_csv(PAIRWISE_OUTPUT, index=False)
    order.to_csv(ORDER_OUTPUT, index=False)
    variance.to_csv(VARIANCE_OUTPUT, index=False)

    print(f"\n✅ Saved: {ASSIGNMENTS_OUTPUT}")
    print(f"✅ Saved: {METRICS_OUTPUT}")
    print(f"✅ Saved: {STABILITY_OUTPUT}")
    print(f"✅ Saved: {PROFILES_OUTPUT}")
    print(f"✅ Saved: {PAIRWISE_OUTPUT}")
    print(f"✅ Saved: {ORDER_OUTPUT}")
    print(f"✅ Saved: {VARIANCE_OUTPUT}")

    plot_cluster_metrics(metrics)
    plot_semantic_consistency(order)
    plot_variance_decomposition(variance)

    print_solution_summary(
        metrics, stability, profiles, pairwise, order, variance
    )

    print("\n--- Done ---")
    print("STOP here. Do not replace the production regimes until this experiment is interpreted.")


if __name__ == "__main__":
    main()

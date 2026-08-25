# analyzeRegimeTransitions.py
# Step 9.6: Regime-transition and continuous latent-trajectory analysis
#
# Scientific purpose
# ------------------
# Quantify how basin hydrological state assignments evolve through successive
# 24-month rolling windows. All transition metrics are computed from the
# continuous full-record window sequence, and latent displacement is measured
# in standardized 16-D embedding space (never in 2-D UMAP space).
#
# IMPORTANT: windows start one month apart and overlap by 23 months. Therefore
# persistence metrics describe persistence/smoothness of the learned monthly
# state sequence; they are not independent-event transition probabilities.

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG
# =========================
EMBEDDINGS_FILE = "data/processed/embeddings_window_level_continuous_v3.parquet"
CLUSTERS_FILE = "data/processed/window_clusters.parquet"

STATE_OUTPUT = "data/processed/window_regime_states_with_segments.parquet"
TRANSITIONS_OUTPUT = "data/processed/window_regime_transitions.parquet"
RUNS_OUTPUT = "data/processed/regime_residence_runs.parquet"

TABLE_DIR = Path("results/tables/regime_transitions")
FIG_DIR = Path("results/figures/regime_transitions")

TRANSITION_COUNTS_OUTPUT = TABLE_DIR / "regime_transition_counts.csv"
TRANSITION_PROB_OUTPUT = TABLE_DIR / "regime_transition_probabilities.csv"
CHANGE_ONLY_PROB_OUTPUT = TABLE_DIR / "regime_transition_probabilities_change_only.csv"
PERSISTENCE_OUTPUT = TABLE_DIR / "regime_persistence_summary.csv"
BASIN_METRICS_OUTPUT = TABLE_DIR / "basin_transition_metrics.csv"
ARCHETYPE_OUTPUT = TABLE_DIR / "basin_trajectory_archetype_candidates.csv"
SUMMARY_OUTPUT = TABLE_DIR / "transition_analysis_summary.json"

TRANSITION_HEATMAP = FIG_DIR / "regime_transition_probability_matrix.png"
CHANGE_ONLY_HEATMAP = FIG_DIR / "regime_transition_change_only_matrix.png"
PERSISTENCE_FIGURE = FIG_DIR / "regime_persistence_probability.png"
STEP_DISTANCE_FIGURE = FIG_DIR / "latent_step_distance_distribution.png"
ARCHETYPE_FIGURE = FIG_DIR / "trajectory_archetype_diagnostic.png"

EXPECTED_START_STEP_MONTHS = 1

# Data-driven screening thresholds for trajectory archetype *candidates*.
# These are not physical ground-truth labels and should later be checked against
# SPEI / documented drought-wet events / ENSO.
ABRUPT_STEP_QUANTILE = 0.99
GRADUAL_MAX_STEP_QUANTILE = 0.95
STABLE_TRANSITION_RATE_QUANTILE = 0.25
STABLE_NET_DISPLACEMENT_QUANTILE = 0.25
GRADUAL_NET_DISPLACEMENT_QUANTILE = 0.75
GRADUAL_PATH_EFFICIENCY_QUANTILE = 0.75
MIN_ADJACENT_PAIRS_FOR_ARCHETYPE = 12


# =========================
# HELPERS
# =========================
def month_diff(t1, t2):
    """Whole-month difference between two timestamps."""
    t1 = pd.Timestamp(t1)
    t2 = pd.Timestamp(t2)
    return (t2.year - t1.year) * 12 + (t2.month - t1.month)


def get_latent_columns(df):
    latent_cols = [c for c in df.columns if c.startswith("z") and c[1:].isdigit()]
    latent_cols = sorted(latent_cols, key=lambda x: int(x[1:]))
    if not latent_cols:
        raise ValueError("No latent columns z1, z2, ... found in embeddings.")
    return latent_cols


def safe_quantile(series, q):
    values = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(q))


def euclidean(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(b - a))


# =========================
# LOAD + ALIGN STATE TABLE
# =========================
def load_inputs():
    emb = pd.read_parquet(EMBEDDINGS_FILE)
    clusters = pd.read_parquet(CLUSTERS_FILE)

    print(f"✅ Loaded continuous embeddings: {EMBEDDINGS_FILE}")
    print(f"   Rows: {len(emb):,}")
    print(f"✅ Loaded continuous regime assignments: {CLUSTERS_FILE}")
    print(f"   Rows: {len(clusters):,}")

    return emb, clusters


def build_state_table(emb, clusters):
    """Merge cluster labels onto continuous 16-D embeddings and validate alignment."""
    latent_cols = get_latent_columns(emb)

    required_emb = {"sample_id", "basin", "start_time", "end_time"} | set(latent_cols)
    required_cluster = {"sample_id", "basin_id", "start_time", "end_time", "cluster"}

    missing_emb = required_emb - set(emb.columns)
    missing_cl = required_cluster - set(clusters.columns)
    if missing_emb:
        raise ValueError(f"Embeddings missing required columns: {sorted(missing_emb)}")
    if missing_cl:
        raise ValueError(f"Clusters missing required columns: {sorted(missing_cl)}")

    emb_keep = ["sample_id", "basin", "start_time", "end_time"] + latent_cols
    emb_small = emb[emb_keep].copy().rename(columns={"basin": "basin_id"})
    emb_small["start_time"] = pd.to_datetime(emb_small["start_time"])
    emb_small["end_time"] = pd.to_datetime(emb_small["end_time"])

    cluster_meta_candidates = [
        "sample_id", "basin_id", "start_time", "end_time", "cluster", "k",
        "split", "start_split", "end_split", "crosses_split_boundary",
        "window_index_within_basin", "window_set",
    ]
    cluster_keep = [c for c in cluster_meta_candidates if c in clusters.columns]
    cl = clusters[cluster_keep].copy()
    cl["start_time"] = pd.to_datetime(cl["start_time"])
    cl["end_time"] = pd.to_datetime(cl["end_time"])

    # Merge by sample_id but retain duplicate identity fields for explicit checks.
    state = cl.merge(
        emb_small,
        on="sample_id",
        how="inner",
        suffixes=("_cluster", "_emb"),
        validate="one_to_one",
    )

    if len(state) != len(cl) or len(state) != len(emb_small):
        raise ValueError(
            "Cluster/embedding row mismatch after sample_id merge: "
            f"clusters={len(cl):,}, embeddings={len(emb_small):,}, merged={len(state):,}"
        )

    basin_match = state["basin_id_cluster"].astype(str) == state["basin_id_emb"].astype(str)
    start_match = state["start_time_cluster"] == state["start_time_emb"]
    end_match = state["end_time_cluster"] == state["end_time_emb"]

    if not basin_match.all():
        raise ValueError(f"Basin identity mismatch in {(~basin_match).sum()} merged rows.")
    if not start_match.all() or not end_match.all():
        raise ValueError(
            "Window timestamp mismatch between cluster and embedding tables: "
            f"start={(~start_match).sum()}, end={(~end_match).sum()}"
        )

    state["basin_id"] = state["basin_id_cluster"]
    state["start_time"] = state["start_time_cluster"]
    state["end_time"] = state["end_time_cluster"]

    drop_cols = [
        "basin_id_cluster", "basin_id_emb",
        "start_time_cluster", "start_time_emb",
        "end_time_cluster", "end_time_emb",
    ]
    state = state.drop(columns=[c for c in drop_cols if c in state.columns])

    # Reorder metadata first.
    meta_order = [
        "sample_id", "basin_id", "start_time", "end_time", "cluster", "k",
        "split", "start_split", "end_split", "crosses_split_boundary",
        "window_index_within_basin", "window_set",
    ]
    first = [c for c in meta_order if c in state.columns]
    rest = [c for c in state.columns if c not in first]
    state = state[first + rest]

    print(f"✅ State table aligned one-to-one: {len(state):,} windows")
    print(f"   Basins: {state['basin_id'].nunique():,}")
    print(f"   Latent dimensions: {len(latent_cols)}")
    return state, latent_cols


def add_standardized_latents_and_segments(state, latent_cols):
    """
    Standardize 16-D latents exactly as KMeans does, then identify independent
    continuous segments. A new segment starts whenever adjacent window starts
    are not exactly one month apart.
    """
    out = state.copy()
    X = out[latent_cols].to_numpy(dtype=np.float64)

    if not np.isfinite(X).all():
        raise ValueError("NaN/inf values found in latent embedding matrix.")

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    std_cols = []
    for j in range(X_std.shape[1]):
        col = f"zs{j + 1}"
        out[col] = X_std[:, j]
        std_cols.append(col)

    out = out.sort_values(["basin_id", "start_time"]).reset_index(drop=True)

    segment_ids = np.zeros(len(out), dtype=np.int64)
    global_segment_counter = 0

    for _, idx in out.groupby("basin_id", sort=False).groups.items():
        idx = list(idx)
        local_segment = 0
        previous_time = None

        for pos in idx:
            current_time = out.at[pos, "start_time"]
            if previous_time is not None:
                if month_diff(previous_time, current_time) != EXPECTED_START_STEP_MONTHS:
                    local_segment += 1
            segment_ids[pos] = global_segment_counter + local_segment
            previous_time = current_time

        global_segment_counter += local_segment + 1

    out["continuous_segment_id"] = segment_ids
    out["segment_index_within_basin"] = (
        out.groupby("basin_id")["continuous_segment_id"]
        .transform(lambda s: pd.factorize(s)[0])
        .astype(int)
    )

    n_segments = out[["basin_id", "continuous_segment_id"]].drop_duplicates().shape[0]
    print(f"✅ Identified {n_segments:,} continuous trajectory segments")
    print("   Non-consecutive windows are never connected across segment boundaries.")

    return out, std_cols, scaler


# =========================
# ADJACENT TRANSITIONS
# =========================
def build_adjacent_transitions(state, latent_cols, std_cols):
    """One row per valid one-month transition between successive window states."""
    rows = []

    for basin_id, basin_df in state.groupby("basin_id", sort=False):
        basin_df = basin_df.sort_values("start_time").reset_index(drop=True)

        for i in range(len(basin_df) - 1):
            a = basin_df.iloc[i]
            b = basin_df.iloc[i + 1]

            delta = month_diff(a["start_time"], b["start_time"])
            if delta != EXPECTED_START_STEP_MONTHS:
                continue
            if a["continuous_segment_id"] != b["continuous_segment_id"]:
                continue

            from_cluster = int(a["cluster"])
            to_cluster = int(b["cluster"])

            raw_dist = euclidean(a[latent_cols].to_numpy(), b[latent_cols].to_numpy())
            std_dist = euclidean(a[std_cols].to_numpy(), b[std_cols].to_numpy())

            rows.append({
                "basin_id": basin_id,
                "continuous_segment_id": int(a["continuous_segment_id"]),
                "from_sample_id": int(a["sample_id"]),
                "to_sample_id": int(b["sample_id"]),
                "from_start_time": a["start_time"],
                "to_start_time": b["start_time"],
                "from_end_time": a["end_time"],
                "to_end_time": b["end_time"],
                "delta_months": int(delta),
                "from_cluster": from_cluster,
                "to_cluster": to_cluster,
                "regime_changed": bool(from_cluster != to_cluster),
                "transition_label": f"R{from_cluster}→R{to_cluster}",
                "latent_step_distance_raw": raw_dist,
                "latent_step_distance_std": std_dist,
            })

    transitions = pd.DataFrame(rows)
    if transitions.empty:
        raise ValueError("No valid one-month adjacent transitions were found.")

    transitions["step_distance_percentile_global"] = (
        transitions["latent_step_distance_std"].rank(method="average", pct=True)
    )

    print(f"✅ Built valid adjacent transitions: {len(transitions):,}")
    print(f"   Regime changes: {int(transitions['regime_changed'].sum()):,}")
    print(
        f"   Global persistence: "
        f"{1.0 - transitions['regime_changed'].mean():.3f}"
    )
    return transitions


# =========================
# TRANSITION MATRICES
# =========================
def build_transition_matrices(transitions, clusters_present):
    clusters_present = sorted(int(c) for c in clusters_present)

    counts = pd.crosstab(
        transitions["from_cluster"],
        transitions["to_cluster"],
    ).reindex(index=clusters_present, columns=clusters_present, fill_value=0)

    row_totals = counts.sum(axis=1).replace(0, np.nan)
    probabilities = counts.div(row_totals, axis=0).fillna(0.0)

    change_counts = counts.copy()
    for c in clusters_present:
        change_counts.loc[c, c] = 0
    change_totals = change_counts.sum(axis=1).replace(0, np.nan)
    change_only_prob = change_counts.div(change_totals, axis=0).fillna(0.0)

    counts.index.name = "from_cluster"
    probabilities.index.name = "from_cluster"
    change_only_prob.index.name = "from_cluster"

    return counts, probabilities, change_only_prob


# =========================
# RESIDENCE / PERSISTENCE
# =========================
def build_residence_runs(state):
    """
    Build consecutive monthly runs of the same regime within each continuous
    trajectory segment.

    Because window starts advance monthly, n_windows is also the number of
    consecutive monthly state assignments in the run. It is not the number of
    independent 24-month observations.
    """
    rows = []

    for (basin_id, segment_id), seg in state.groupby(
        ["basin_id", "continuous_segment_id"], sort=False
    ):
        seg = seg.sort_values("start_time").reset_index(drop=True)
        if seg.empty:
            continue

        run_start = 0
        run_number = 0

        for i in range(1, len(seg) + 1):
            end_run = (i == len(seg)) or (seg.loc[i, "cluster"] != seg.loc[i - 1, "cluster"])
            if not end_run:
                continue

            run = seg.iloc[run_start:i]
            start_time = run["start_time"].iloc[0]
            end_time = run["start_time"].iloc[-1]
            n_windows = len(run)

            rows.append({
                "basin_id": basin_id,
                "continuous_segment_id": int(segment_id),
                "run_index_within_segment": int(run_number),
                "cluster": int(run["cluster"].iloc[0]),
                "run_start_time": start_time,
                "run_end_time": end_time,
                "n_consecutive_window_starts": int(n_windows),
                "run_calendar_span_months": int(month_diff(start_time, end_time) + 1),
            })

            run_start = i
            run_number += 1

    runs = pd.DataFrame(rows)
    print(f"✅ Built regime residence runs: {len(runs):,}")
    return runs


def summarize_regime_persistence(transitions, runs, clusters_present):
    rows = []

    for c in sorted(int(x) for x in clusters_present):
        outgoing = transitions[transitions["from_cluster"] == c]
        c_runs = runs[runs["cluster"] == c]

        n_pairs = len(outgoing)
        n_self = int((outgoing["to_cluster"] == c).sum())
        persistence = n_self / n_pairs if n_pairs > 0 else np.nan

        rows.append({
            "cluster": c,
            "n_outgoing_adjacent_pairs": int(n_pairs),
            "n_self_transitions": n_self,
            "persistence_probability": persistence,
            "n_residence_runs": int(len(c_runs)),
            "mean_run_monthly_assignments": (
                float(c_runs["n_consecutive_window_starts"].mean()) if not c_runs.empty else np.nan
            ),
            "median_run_monthly_assignments": (
                float(c_runs["n_consecutive_window_starts"].median()) if not c_runs.empty else np.nan
            ),
            "max_run_monthly_assignments": (
                int(c_runs["n_consecutive_window_starts"].max()) if not c_runs.empty else np.nan
            ),
        })

    return pd.DataFrame(rows)


# =========================
# BASIN-LEVEL METRICS
# =========================
def compute_segment_net_displacements(state_basin, std_cols):
    values = []
    for _, seg in state_basin.groupby("continuous_segment_id", sort=False):
        seg = seg.sort_values("start_time")
        if len(seg) < 2:
            continue
        values.append(
            euclidean(
                seg.iloc[0][std_cols].to_numpy(),
                seg.iloc[-1][std_cols].to_numpy(),
            )
        )
    return values


def compute_basin_transition_metrics(state, transitions, runs, std_cols):
    rows = []

    for basin_id, basin_state in state.groupby("basin_id", sort=False):
        basin_state = basin_state.sort_values("start_time")
        tr = transitions[transitions["basin_id"] == basin_id]
        br = runs[runs["basin_id"] == basin_id]

        n_windows = len(basin_state)
        n_pairs = len(tr)
        n_changes = int(tr["regime_changed"].sum()) if n_pairs else 0
        transition_rate = n_changes / n_pairs if n_pairs else np.nan
        persistence = 1.0 - transition_rate if n_pairs else np.nan

        occupancy = basin_state["cluster"].value_counts(normalize=True)
        dominant_cluster = int(occupancy.index[0]) if not occupancy.empty else np.nan
        dominant_fraction = float(occupancy.iloc[0]) if not occupancy.empty else np.nan
        n_distinct_clusters = int(basin_state["cluster"].nunique())

        step = tr["latent_step_distance_std"] if n_pairs else pd.Series(dtype=float)
        path_length = float(step.sum()) if n_pairs else 0.0
        segment_net = compute_segment_net_displacements(basin_state, std_cols)
        net_sum = float(np.sum(segment_net)) if segment_net else 0.0
        max_segment_net = float(np.max(segment_net)) if segment_net else 0.0
        path_efficiency = net_sum / path_length if path_length > 0 else np.nan

        # Normalized regime occupancy entropy: 0 = one regime only, 1 = evenly spread.
        if len(occupancy) <= 1:
            regime_entropy_norm = 0.0
        else:
            p = occupancy.to_numpy(dtype=float)
            entropy = -np.sum(p * np.log(p))
            regime_entropy_norm = float(entropy / np.log(len(p)))

        rows.append({
            "basin_id": basin_id,
            "first_window_start": basin_state["start_time"].min(),
            "last_window_start": basin_state["start_time"].max(),
            "n_windows": int(n_windows),
            "n_continuous_segments": int(basin_state["continuous_segment_id"].nunique()),
            "n_valid_adjacent_pairs": int(n_pairs),
            "n_regime_changes": n_changes,
            "transition_rate": transition_rate,
            "persistence_probability": persistence,
            "dominant_cluster": dominant_cluster,
            "dominant_cluster_fraction": dominant_fraction,
            "n_distinct_clusters": n_distinct_clusters,
            "regime_occupancy_entropy_norm": regime_entropy_norm,
            "mean_step_distance_std": float(step.mean()) if n_pairs else np.nan,
            "median_step_distance_std": float(step.median()) if n_pairs else np.nan,
            "p95_step_distance_std": float(step.quantile(0.95)) if n_pairs else np.nan,
            "max_step_distance_std": float(step.max()) if n_pairs else np.nan,
            "total_path_length_std": path_length,
            "sum_segment_net_displacement_std": net_sum,
            "max_segment_net_displacement_std": max_segment_net,
            "path_efficiency": path_efficiency,
            "n_residence_runs": int(len(br)),
            "mean_run_monthly_assignments": (
                float(br["n_consecutive_window_starts"].mean()) if not br.empty else np.nan
            ),
            "median_run_monthly_assignments": (
                float(br["n_consecutive_window_starts"].median()) if not br.empty else np.nan
            ),
            "max_run_monthly_assignments": (
                int(br["n_consecutive_window_starts"].max()) if not br.empty else np.nan
            ),
        })

    metrics = pd.DataFrame(rows)
    print(f"✅ Built basin transition metrics: {len(metrics):,} basins")
    return metrics


# =========================
# ARCHETYPE CANDIDATE SCREENING
# =========================
def classify_trajectory_archetype_candidates(metrics, transitions):
    """
    Data-driven screening for poster/case-study candidates.

    The labels are intentionally named '* candidate'. They should only become
    scientific case-study labels after Step 6 climatological validation.
    """
    out = metrics.copy()

    abrupt_step = safe_quantile(transitions["latent_step_distance_std"], ABRUPT_STEP_QUANTILE)
    gradual_step_limit = safe_quantile(
        transitions["latent_step_distance_std"], GRADUAL_MAX_STEP_QUANTILE
    )

    eligible = out[out["n_valid_adjacent_pairs"] >= MIN_ADJACENT_PAIRS_FOR_ARCHETYPE].copy()

    thresholds = {
        "abrupt_step_distance_std_q99": abrupt_step,
        "gradual_max_step_distance_std_q95": gradual_step_limit,
        "stable_transition_rate_q25": safe_quantile(
            eligible["transition_rate"], STABLE_TRANSITION_RATE_QUANTILE
        ),
        "stable_max_segment_net_displacement_std_q25": safe_quantile(
            eligible["max_segment_net_displacement_std"], STABLE_NET_DISPLACEMENT_QUANTILE
        ),
        "gradual_max_segment_net_displacement_std_q75": safe_quantile(
            eligible["max_segment_net_displacement_std"], GRADUAL_NET_DISPLACEMENT_QUANTILE
        ),
        "gradual_path_efficiency_q75": safe_quantile(
            eligible["path_efficiency"], GRADUAL_PATH_EFFICIENCY_QUANTILE
        ),
        "minimum_adjacent_pairs": int(MIN_ADJACENT_PAIRS_FOR_ARCHETYPE),
    }

    labels = []
    scores = []

    for _, row in out.iterrows():
        if row["n_valid_adjacent_pairs"] < MIN_ADJACENT_PAIRS_FOR_ARCHETYPE:
            labels.append("insufficient-record")
            scores.append(np.nan)
            continue

        max_step = row["max_step_distance_std"]
        rate = row["transition_rate"]
        net = row["max_segment_net_displacement_std"]
        efficiency = row["path_efficiency"]

        if np.isfinite(abrupt_step) and max_step >= abrupt_step:
            labels.append("abrupt-shift candidate")
            scores.append(float(max_step))
            continue

        stable = (
            np.isfinite(thresholds["stable_transition_rate_q25"])
            and np.isfinite(thresholds["stable_max_segment_net_displacement_std_q25"])
            and rate <= thresholds["stable_transition_rate_q25"]
            and net <= thresholds["stable_max_segment_net_displacement_std_q25"]
        )
        if stable:
            labels.append("stable candidate")
            # Larger score = more archetypally stable for convenient ranking.
            score = 1.0 / (1e-12 + max(rate, 0) + max(net, 0))
            scores.append(float(score))
            continue

        gradual = (
            np.isfinite(gradual_step_limit)
            and np.isfinite(thresholds["gradual_max_segment_net_displacement_std_q75"])
            and np.isfinite(thresholds["gradual_path_efficiency_q75"])
            and max_step <= gradual_step_limit
            and net >= thresholds["gradual_max_segment_net_displacement_std_q75"]
            and efficiency >= thresholds["gradual_path_efficiency_q75"]
        )
        if gradual:
            labels.append("gradual-migration candidate")
            scores.append(float(net * efficiency))
            continue

        labels.append("mixed/dynamic")
        scores.append(float(row["total_path_length_std"]))

    out["trajectory_archetype_candidate"] = labels
    out["archetype_screening_score"] = scores

    out["rank_within_archetype"] = (
        out.groupby("trajectory_archetype_candidate")["archetype_screening_score"]
        .rank(method="first", ascending=False)
    )

    # Expose thresholds as columns too, so the CSV is self-describing.
    for key, value in thresholds.items():
        out[f"threshold__{key}"] = value

    print("✅ Screened trajectory archetype candidates")
    print(out["trajectory_archetype_candidate"].value_counts().to_string())
    return out, thresholds


# =========================
# SAVE TABLES
# =========================
def save_matrix(matrix, path):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path)
    print(f"✅ Saved: {path}")


def save_outputs(state, transitions, runs, counts, probs, change_probs,
                 persistence, metrics, archetypes, thresholds):
    Path(os.path.dirname(STATE_OUTPUT)).mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    state.to_parquet(STATE_OUTPUT, index=False)
    transitions.to_parquet(TRANSITIONS_OUTPUT, index=False)
    runs.to_parquet(RUNS_OUTPUT, index=False)

    print(f"✅ Saved: {STATE_OUTPUT}")
    print(f"✅ Saved: {TRANSITIONS_OUTPUT}")
    print(f"✅ Saved: {RUNS_OUTPUT}")

    save_matrix(counts, TRANSITION_COUNTS_OUTPUT)
    save_matrix(probs, TRANSITION_PROB_OUTPUT)
    save_matrix(change_probs, CHANGE_ONLY_PROB_OUTPUT)

    persistence.to_csv(PERSISTENCE_OUTPUT, index=False)
    metrics.to_csv(BASIN_METRICS_OUTPUT, index=False)
    archetypes.to_csv(ARCHETYPE_OUTPUT, index=False)

    print(f"✅ Saved: {PERSISTENCE_OUTPUT}")
    print(f"✅ Saved: {BASIN_METRICS_OUTPUT}")
    print(f"✅ Saved: {ARCHETYPE_OUTPUT}")

    summary = {
        "input_embeddings": EMBEDDINGS_FILE,
        "input_clusters": CLUSTERS_FILE,
        "n_windows": int(len(state)),
        "n_basins": int(state["basin_id"].nunique()),
        "n_continuous_segments": int(
            state[["basin_id", "continuous_segment_id"]].drop_duplicates().shape[0]
        ),
        "n_adjacent_transitions": int(len(transitions)),
        "n_regime_changes": int(transitions["regime_changed"].sum()),
        "global_transition_rate": float(transitions["regime_changed"].mean()),
        "global_persistence_probability": float(1 - transitions["regime_changed"].mean()),
        "median_latent_step_distance_std": float(
            transitions["latent_step_distance_std"].median()
        ),
        "p95_latent_step_distance_std": float(
            transitions["latent_step_distance_std"].quantile(0.95)
        ),
        "archetype_screening_thresholds": thresholds,
        "archetype_candidate_counts": {
            str(k): int(v)
            for k, v in archetypes["trajectory_archetype_candidate"].value_counts().items()
        },
        "important_interpretation_note": (
            "Successive 24-month windows overlap by 23 months. Transition probabilities "
            "therefore quantify persistence/change of monthly window-state assignments, "
            "not independent 24-month hydrological events."
        ),
    }

    with open(SUMMARY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✅ Saved: {SUMMARY_OUTPUT}")


# =========================
# FIGURES
# =========================
def plot_matrix(matrix, title, path, cbar_label="Probability"):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    values = matrix.to_numpy(dtype=float)
    labels = list(matrix.index)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(values, vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([f"R{c}" for c in matrix.columns])
    ax.set_yticklabels([f"R{c}" for c in labels])
    ax.set_xlabel("Regime at t + 1 month")
    ax.set_ylabel("Regime at t")
    ax.set_title(title)

    if values.shape[0] <= 10 and values.shape[1] <= 10:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved figure: {path}")


def plot_persistence(persistence):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.bar(persistence["cluster"].astype(str), persistence["persistence_probability"])
    plt.ylim(0, 1)
    plt.xlabel("Regime")
    plt.ylabel("P(R$_{t+1}$ = R$_t$)")
    plt.title("Regime Persistence Across Consecutive Monthly Window Starts")
    plt.tight_layout()
    plt.savefig(PERSISTENCE_FIGURE, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure: {PERSISTENCE_FIGURE}")


def plot_step_distance_distribution(transitions, abrupt_threshold):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = transitions["latent_step_distance_std"].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=60)
    if np.isfinite(abrupt_threshold):
        plt.axvline(
            abrupt_threshold,
            linestyle="--",
            linewidth=1.5,
            label=f"q{int(ABRUPT_STEP_QUANTILE * 100)} abrupt-candidate threshold",
        )
        plt.legend()
    plt.xlabel("Adjacent-window displacement in standardized 16-D latent space")
    plt.ylabel("Count")
    plt.title("Distribution of Monthly Latent-State Displacements")
    plt.tight_layout()
    plt.savefig(STEP_DISTANCE_FIGURE, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure: {STEP_DISTANCE_FIGURE}")


def plot_archetype_diagnostic(archetypes):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = archetypes[
        archetypes["trajectory_archetype_candidate"] != "insufficient-record"
    ].copy()

    plt.figure(figsize=(8, 6))
    for label, sub in plot_df.groupby("trajectory_archetype_candidate"):
        plt.scatter(
            sub["transition_rate"],
            sub["max_segment_net_displacement_std"],
            s=18,
            alpha=0.65,
            label=label,
        )

    plt.xlabel("Regime-change rate across valid adjacent windows")
    plt.ylabel("Maximum segment net displacement (standardized 16-D)")
    plt.title("Basin Trajectory Archetype Screening")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ARCHETYPE_FIGURE, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved figure: {ARCHETYPE_FIGURE}")


# =========================
# MAIN
# =========================
def main():
    print("--- Step 9.6: Continuous regime-transition analysis ---")

    emb, clusters = load_inputs()
    state, latent_cols = build_state_table(emb, clusters)
    state, std_cols, _ = add_standardized_latents_and_segments(state, latent_cols)

    transitions = build_adjacent_transitions(state, latent_cols, std_cols)
    clusters_present = sorted(state["cluster"].unique())

    counts, probs, change_probs = build_transition_matrices(
        transitions, clusters_present
    )

    runs = build_residence_runs(state)
    persistence = summarize_regime_persistence(
        transitions, runs, clusters_present
    )

    metrics = compute_basin_transition_metrics(
        state, transitions, runs, std_cols
    )

    archetypes, thresholds = classify_trajectory_archetype_candidates(
        metrics, transitions
    )

    save_outputs(
        state=state,
        transitions=transitions,
        runs=runs,
        counts=counts,
        probs=probs,
        change_probs=change_probs,
        persistence=persistence,
        metrics=metrics,
        archetypes=archetypes,
        thresholds=thresholds,
    )

    plot_matrix(
        probs,
        "Regime Transition Probability Matrix",
        TRANSITION_HEATMAP,
    )
    plot_matrix(
        change_probs,
        "Destination Regime Conditional on a Regime Change",
        CHANGE_ONLY_HEATMAP,
    )
    plot_persistence(persistence)
    plot_step_distance_distribution(
        transitions,
        thresholds["abrupt_step_distance_std_q99"],
    )
    plot_archetype_diagnostic(archetypes)

    print("\nKey interpretation rules:")
    print("1. Only window starts exactly 1 month apart are treated as transitions.")
    print("2. No transition is constructed across observational / SWT gaps.")
    print("3. Latent displacement is measured in standardized 16-D space, not UMAP.")
    print("4. Stable/gradual/abrupt labels are screening candidates until physically validated.")
    print("5. 24-month windows overlap by 23 months, so adjacent states are not independent.")
    print("--- Done ---")


if __name__ == "__main__":
    main()

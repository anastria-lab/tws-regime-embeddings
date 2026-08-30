# visualizeFinalLatentTrajectories.py
#
# VISUALIZATION-ONLY companion to the completed validation.
#
# Purpose
# -------
# Show the selected basin trajectories in the learned 16-D latent atlas using
# a UMAP fitted ONLY to historical window embeddings.
#
# This script does NOT change any quantitative result.
# All migration/support conclusions remain those computed in 16-D.
#
# Each candidate figure shows:
#   - historical global atlas background
#   - historical windows from strong source-class anchor basins
#   - historical windows from strong target-class anchor basins
#   - the candidate basin's full rolling trajectory
#   - recent (start >= 2021-01) trajectory emphasized
#   - historical basin mean and recent basin mean
#
# It also makes a global panel containing the selected trajectories.
#
# IMPORTANT:
# UMAP is used for visualization only, never for distances, support tests,
# classification, or migration decisions.

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as e:
    raise ImportError(
        "This script requires umap-learn. Install with: uv add umap-learn"
    ) from e

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# PATHS
# =============================================================================

EMBEDDINGS_FILE = Path(
    "data/processed/embeddings_window_level_continuous_v3.parquet"
)
STATIC_FILE = Path(
    "data/processed/koppen_basin_static_composition_summary_labeled.csv"
)

# Preferred six candidates already selected for physical validation.
SELECTED_FILE = Path(
    "results/tables/strict_migration_physical_validation/"
    "selected_strict_migration_candidates.csv"
)

# Final validation result supplies the final interpretation label.
FINAL_VALIDATION_FILE = Path(
    "results/tables/final_physical_convergence/"
    "final_source_recent_target_physical_validation.csv"
)

# Fallback if selected file is unavailable.
STRICT_FILE = Path(
    "results/tables/koppen_duration_matched_support/"
    "duration_matched_strict_migrations_q95.csv"
)

OUT_DIR = Path("results/tables/latent_trajectory_visualization")
FIG_DIR = Path("results/figures/latent_trajectory_visualization")

COORDS_OUT = OUT_DIR / "selected_candidate_umap_trajectory_coordinates.csv"
ANCHOR_COORDS_OUT = OUT_DIR / "historical_source_target_anchor_umap_coordinates.csv"
CONFIG_OUT = OUT_DIR / "latent_trajectory_visualization_config.json"

HIST_END = pd.Timestamp("2020-12-01")
RECENT_START = pd.Timestamp("2021-01-01")

KG_ANCHOR_THRESHOLD = 0.80
N_SELECTED = 6

# UMAP is fitted to a reproducible sample of HISTORICAL windows only.
MAX_BACKGROUND_WINDOWS = 50000
MAX_CLASS_WINDOWS_PER_CLASS = 12000

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.12
UMAP_METRIC = "euclidean"
RANDOM_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def detect_basin_column(df):
    for c in ["basin", "basin_id", "HYBAS_ID"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect basin identifier column.")


def latent_columns(df):
    cols = [c for c in df.columns if c.startswith("z")]
    try:
        cols = sorted(cols, key=lambda x: int(x[1:]))
    except Exception:
        cols = sorted(cols)
    if len(cols) != 16:
        raise ValueError(
            f"Expected 16 latent dimensions, found {len(cols)}: {cols}"
        )
    return cols


def month_label(ts):
    return pd.Timestamp(ts).strftime("%Y-%m")


# =============================================================================
# LOAD
# =============================================================================

def load_inputs():
    emb = pd.read_parquet(EMBEDDINGS_FILE).copy()
    static = pd.read_csv(STATIC_FILE)

    bcol = detect_basin_column(emb)
    emb["basin_id"] = pd.to_numeric(
        emb[bcol], errors="raise"
    ).astype("int64")
    emb["start_time"] = pd.to_datetime(emb["start_time"])
    emb["end_time"] = pd.to_datetime(emb["end_time"])

    static["HYBAS_ID"] = pd.to_numeric(
        static["HYBAS_ID"], errors="raise"
    ).astype("int64")

    zcols = latent_columns(emb)

    if SELECTED_FILE.exists():
        selected = pd.read_csv(SELECTED_FILE)
    else:
        strict = pd.read_csv(STRICT_FILE)
        strict["pathway"] = (
            strict["source_KG"].astype(str)
            + "→"
            + strict["best_supported_validated_target"].astype(str)
        )
        selected = (
            strict.sort_values(
                [
                    "source_target_pairwise_auc",
                    "pair_axis_delta_toward_target",
                ],
                ascending=[False, False],
            )
            .groupby("pathway", as_index=False, sort=False)
            .head(1)
            .head(N_SELECTED)
            .copy()
        )

    selected["basin_id"] = pd.to_numeric(
        selected["basin_id"], errors="raise"
    ).astype("int64")

    # Harmonize target column.
    if "target_KG" not in selected.columns:
        selected["target_KG"] = selected[
            "best_supported_validated_target"
        ].astype(str)

    selected["source_KG"] = selected["source_KG"].astype(str)
    selected["target_KG"] = selected["target_KG"].astype(str)

    # Attach final physical interpretation when available.
    if FINAL_VALIDATION_FILE.exists():
        final = pd.read_csv(FINAL_VALIDATION_FILE)
        final["basin_id"] = pd.to_numeric(
            final["basin_id"], errors="raise"
        ).astype("int64")
        final = final[
            [
                "basin_id",
                "final_physical_validation_outcome",
                "physical_pair_cv_auc",
                "physical_source_support_ratio",
                "physical_target_support_ratio",
            ]
        ].copy()
        selected = selected.merge(
            final,
            on="basin_id",
            how="left",
            validate="one_to_one",
        )

    print(f"✅ Continuous windows: {len(emb):,}")
    print(f"✅ Basins:             {emb['basin_id'].nunique():,}")
    print(f"✅ Selected candidates:{len(selected):,}")
    print(f"✅ Latent dimensions:  {len(zcols)}")

    return emb, static, selected.head(N_SELECTED).copy(), zcols


# =============================================================================
# HISTORICAL STANDARDIZATION AND UMAP
# =============================================================================

def fit_historical_umap(emb, zcols):
    hist = emb[emb["end_time"] <= HIST_END].copy()
    if hist.empty:
        raise ValueError("No historical windows.")

    scaler = StandardScaler()
    scaler.fit(hist[zcols].to_numpy(dtype=float))

    rng = np.random.default_rng(RANDOM_SEED)

    if len(hist) > MAX_BACKGROUND_WINDOWS:
        idx = rng.choice(
            len(hist),
            size=MAX_BACKGROUND_WINDOWS,
            replace=False,
        )
        fit_df = hist.iloc[np.sort(idx)].copy()
    else:
        fit_df = hist.copy()

    Xfit = scaler.transform(
        fit_df[zcols].to_numpy(dtype=float)
    )

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        n_components=2,
        random_state=RANDOM_SEED,
        transform_seed=RANDOM_SEED,
    )

    Yfit = reducer.fit_transform(Xfit)

    fit_df["umap1"] = Yfit[:, 0]
    fit_df["umap2"] = Yfit[:, 1]

    print(f"✅ UMAP fitted on {len(fit_df):,} historical windows")

    return scaler, reducer, fit_df


# =============================================================================
# TRANSFORM SOURCE/TARGET HISTORICAL ANCHORS
# =============================================================================

def build_anchor_windows(
    emb,
    static,
    selected,
    scaler,
    reducer,
    zcols,
):
    classes = sorted(
        set(selected["source_KG"])
        | set(selected["target_KG"])
    )

    strong = static[
        static["KG_qc_pass_90pct_valid"].fillna(False).astype(bool)
        & (static["KG_dom_frac"] >= KG_ANCHOR_THRESHOLD)
        & static["KG_dom_abbr"].astype(str).isin(classes)
    ][
        ["HYBAS_ID", "KG_dom_abbr", "KG_dom_frac"]
    ].copy()

    d = emb[
        emb["end_time"] <= HIST_END
    ].merge(
        strong,
        left_on="basin_id",
        right_on="HYBAS_ID",
        how="inner",
        validate="many_to_one",
    )

    rng = np.random.default_rng(RANDOM_SEED)
    pieces = []

    for cls, g in d.groupby("KG_dom_abbr"):
        if len(g) > MAX_CLASS_WINDOWS_PER_CLASS:
            idx = rng.choice(
                len(g),
                size=MAX_CLASS_WINDOWS_PER_CLASS,
                replace=False,
            )
            g = g.iloc[np.sort(idx)].copy()
        pieces.append(g)

    if not pieces:
        raise ValueError("No strong historical anchor windows found.")

    anchors = pd.concat(pieces, ignore_index=True)

    X = scaler.transform(
        anchors[zcols].to_numpy(dtype=float)
    )
    Y = reducer.transform(X)

    anchors["umap1"] = Y[:, 0]
    anchors["umap2"] = Y[:, 1]

    keep = [
        "basin_id",
        "start_time",
        "end_time",
        "KG_dom_abbr",
        "KG_dom_frac",
        "umap1",
        "umap2",
    ]
    return anchors[keep].copy()


# =============================================================================
# TRANSFORM FULL CANDIDATE TRAJECTORIES
# =============================================================================

def transform_candidate_trajectories(
    emb,
    selected,
    scaler,
    reducer,
    zcols,
):
    d = emb[
        emb["basin_id"].isin(
            set(selected["basin_id"].astype("int64"))
        )
    ].copy()

    X = scaler.transform(d[zcols].to_numpy(dtype=float))
    Y = reducer.transform(X)

    d["umap1"] = Y[:, 0]
    d["umap2"] = Y[:, 1]
    d["period"] = np.where(
        d["start_time"] >= RECENT_START,
        "fully_post2020",
        np.where(
            d["end_time"] <= HIST_END,
            "historical",
            "boundary_spanning",
        ),
    )

    meta_cols = [
        "basin_id",
        "source_KG",
        "target_KG",
    ]
    extra = [
        c for c in [
            "final_physical_validation_outcome",
            "physical_pair_cv_auc",
            "physical_source_support_ratio",
            "physical_target_support_ratio",
        ]
        if c in selected.columns
    ]

    d = d.merge(
        selected[meta_cols + extra],
        on="basin_id",
        how="left",
        validate="many_to_one",
    )

    return d


# =============================================================================
# PLOTS
# =============================================================================

def plot_single_candidate(
    candidate,
    traj,
    anchors,
    fit_df,
):
    basin_id = int(candidate["basin_id"])
    source = str(candidate["source_KG"])
    target = str(candidate["target_KG"])

    t = traj[traj["basin_id"] == basin_id].copy()
    t = t.sort_values("start_time")

    src = anchors[
        anchors["KG_dom_abbr"].astype(str) == source
    ]
    tgt = anchors[
        anchors["KG_dom_abbr"].astype(str) == target
    ]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Global historical atlas context.
    ax.scatter(
        fit_df["umap1"],
        fit_df["umap2"],
        s=4,
        alpha=0.06,
        label="Historical atlas",
    )

    # Historical source/target climate-associated anchor windows.
    if len(src):
        ax.scatter(
            src["umap1"],
            src["umap2"],
            s=7,
            alpha=0.12,
            label=f"Historical {source} anchors",
        )
    if len(tgt):
        ax.scatter(
            tgt["umap1"],
            tgt["umap2"],
            s=7,
            alpha=0.12,
            label=f"Historical {target} anchors",
        )

    # Full candidate trajectory.
    ax.plot(
        t["umap1"],
        t["umap2"],
        linewidth=1.0,
        alpha=0.65,
        label="Full basin trajectory",
    )

    historical = t[t["end_time"] <= HIST_END]
    boundary = t[
        (t["end_time"] > HIST_END)
        & (t["start_time"] < RECENT_START)
    ]
    recent = t[t["start_time"] >= RECENT_START]

    if len(historical):
        ax.scatter(
            historical["umap1"],
            historical["umap2"],
            s=15,
            alpha=0.5,
            label="Historical basin windows",
        )

    if len(boundary):
        ax.scatter(
            boundary["umap1"],
            boundary["umap2"],
            s=17,
            marker="x",
            alpha=0.7,
            label="Boundary-spanning windows",
        )

    if len(recent):
        ax.plot(
            recent["umap1"],
            recent["umap2"],
            linewidth=2.2,
            label="Fully post-2020 trajectory",
        )
        ax.scatter(
            recent["umap1"],
            recent["umap2"],
            s=24,
            alpha=0.8,
        )

    # Historical/recent mean positions.
    if len(historical):
        ax.scatter(
            historical["umap1"].mean(),
            historical["umap2"].mean(),
            s=120,
            marker="o",
            edgecolors="black",
            linewidths=1.2,
            label="Historical basin mean",
        )

    if len(recent):
        ax.scatter(
            recent["umap1"].mean(),
            recent["umap2"].mean(),
            s=140,
            marker="*",
            edgecolors="black",
            linewidths=1.2,
            label="Recent basin mean",
        )

        # Mark recent start and latest point.
        first = recent.iloc[0]
        last = recent.iloc[-1]

        ax.scatter(
            [first["umap1"]],
            [first["umap2"]],
            s=70,
            marker="s",
            edgecolors="black",
            linewidths=1.0,
        )
        ax.annotate(
            f"Recent start\n{month_label(first['start_time'])}",
            (first["umap1"], first["umap2"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
        )

        ax.scatter(
            [last["umap1"]],
            [last["umap2"]],
            s=80,
            marker="D",
            edgecolors="black",
            linewidths=1.0,
        )
        ax.annotate(
            f"Latest\n{month_label(last['end_time'])}",
            (last["umap1"], last["umap2"]),
            xytext=(7, -16),
            textcoords="offset points",
            fontsize=8,
        )

    outcome = candidate.get(
        "final_physical_validation_outcome",
        "not available",
    )

    ax.set_title(
        f"Basin {basin_id}: latent trajectory in historical atlas\n"
        f"Static source {source} → proposed latent target {target} | "
        f"final physical result: {outcome}"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        loc="best",
        fontsize=8,
        frameon=True,
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()

    out = (
        FIG_DIR
        / f"basin_{basin_id}_{source}_to_{target}_latent_trajectory.png"
    )
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")


def plot_global_selected(traj, fit_df, selected):
    fig, ax = plt.subplots(figsize=(11, 8))

    ax.scatter(
        fit_df["umap1"],
        fit_df["umap2"],
        s=4,
        alpha=0.05,
        label="Historical atlas",
    )

    for _, c in selected.iterrows():
        basin_id = int(c["basin_id"])
        source = str(c["source_KG"])
        target = str(c["target_KG"])

        t = traj[
            traj["basin_id"] == basin_id
        ].sort_values("start_time")

        if t.empty:
            continue

        ax.plot(
            t["umap1"],
            t["umap2"],
            linewidth=1.0,
            alpha=0.55,
        )

        recent = t[t["start_time"] >= RECENT_START]
        if len(recent):
            ax.plot(
                recent["umap1"],
                recent["umap2"],
                linewidth=2.0,
                label=f"{basin_id} {source}→{target}",
            )
            last = recent.iloc[-1]
            ax.scatter(
                [last["umap1"]],
                [last["umap2"]],
                s=55,
                marker="D",
                edgecolors="black",
                linewidths=0.8,
            )

    ax.set_title(
        "Selected basin trajectories in the historical learned hydrological atlas"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        loc="best",
        fontsize=8,
        frameon=True,
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()

    out = FIG_DIR / "selected_basin_latent_trajectories_global.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Latent UMAP trajectory visualization ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    emb, static, selected, zcols = load_inputs()

    scaler, reducer, fit_df = fit_historical_umap(
        emb,
        zcols,
    )

    anchors = build_anchor_windows(
        emb,
        static,
        selected,
        scaler,
        reducer,
        zcols,
    )

    traj = transform_candidate_trajectories(
        emb,
        selected,
        scaler,
        reducer,
        zcols,
    )

    anchors.to_csv(ANCHOR_COORDS_OUT, index=False)
    traj.to_csv(COORDS_OUT, index=False)

    print(f"✅ Saved: {ANCHOR_COORDS_OUT}")
    print(f"✅ Saved: {COORDS_OUT}")

    for _, candidate in selected.iterrows():
        plot_single_candidate(
            candidate,
            traj,
            anchors,
            fit_df,
        )

    plot_global_selected(
        traj,
        fit_df,
        selected,
    )

    config = {
        "visualization_only": True,
        "quantitative_migration_space": "16-D latent space",
        "umap_fit_period": "historical windows ending <= 2020-12",
        "umap_fit_max_background_windows": MAX_BACKGROUND_WINDOWS,
        "umap_n_neighbors": UMAP_N_NEIGHBORS,
        "umap_min_dist": UMAP_MIN_DIST,
        "umap_metric": UMAP_METRIC,
        "strong_koppen_anchor_fraction": KG_ANCHOR_THRESHOLD,
        "recent_definition": "window start >= 2021-01",
        "selected_basins": selected["basin_id"].astype(int).tolist(),
    }
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 92)
    print("LATENT UMAP TRAJECTORY VISUALIZATION COMPLETE")
    print("=" * 92)
    print(
        "The UMAP is visualization only. All final physical-support and "
        "migration conclusions remain those computed in the original spaces."
    )
    print(
        "For the poster, prefer one or two individual trajectory panels plus "
        "the global historical atlas context rather than the previous distance bars."
    )
    print("\nSelected basins:")
    print(
        selected[
            [
                "basin_id",
                "source_KG",
                "target_KG",
            ]
            + (
                ["final_physical_validation_outcome"]
                if "final_physical_validation_outcome"
                in selected.columns
                else []
            )
        ].to_string(index=False)
    )

    print("\n--- Done ---")


if __name__ == "__main__":
    main()

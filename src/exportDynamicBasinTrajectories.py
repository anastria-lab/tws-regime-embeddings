# exportDynamicBasinTrajectories.py
#
# FINAL trajectory exporter after the temporal-regime fingerprint audit.
#
# Scientific interpretation of FINAL raw k=3 modes:
#
#   M0 = Balanced / high-variability mode
#   M1 = Globally drying/depletion-associated mode
#   M2 = Globally wetting/recharge-associated mode
#
# IMPORTANT:
# These are recurrent MULTIVARIATE 24-month temporal modes. They are NOT
# universal absolute wet / intermediate / dry classes.
#
# This script:
#   - uses the CONTINUOUS learned atlas;
#   - uses FINAL production k=3 labels;
#   - never reconnects missing months;
#   - exports publication-oriented trajectories and timelines;
#   - adds local 24-month dynamic indicators from the multiscale inputs;
#   - does NOT retrain, recluster, or overwrite production results.

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

GPKG_FILE = "data/interim/hydrobasins_l04_global.gpkg"
UMAP_FILE = "data/processed/embeddings_umap_continuous.parquet"
CLUSTERS_FILE = "data/processed/window_clusters.parquet"
WAVELET_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"

# Optional external validation enrichments generated previously.
SPEI_FILE = "data/processed/window_clusters_spei12_validation.parquet"
ENSO_FILE = "data/processed/window_clusters_enso_validation.parquet"

OUT_DIR = "results/dynamic_trajectories_final_k3_v3"

WINDOW_LENGTH = 24
AMAZON_MAIN_BAS = 6040007000


# =============================================================================
# FINAL DYNAMIC INTERPRETATION
# =============================================================================

# Cluster IDs remain 0/1/2 internally.  For figures we call them M0/M1/M2
# ("modes") so they cannot be mistaken for deterministic wet/dry classes.
#
# The descriptive associations below come from the GLOBAL basin-balanced
# temporal fingerprint audit.  They are interpretation aids, not definitions
# that every individual window must satisfy.
MODE_NAMES = {
    0: "Balanced / high-variability mode",
    1: "Globally drying/depletion-associated mode",
    2: "Globally wetting/recharge-associated mode",
}

MODE_SHORT = {
    0: "Balanced / variable",
    1: "Drying/depletion assoc.",
    2: "Wetting/recharge assoc.",
}

MODE_LABEL = {
    0: "M0",
    1: "M1",
    2: "M2",
}


# =============================================================================
# BASINS TO EXPORT
# =============================================================================

# Edit this dictionary whenever you want different examples.
SELECTED_BASINS = {
    "Amazon_PFAF6223_lower_mainstem": 6040239720,
    "Amazon_PFAF6229_upper_western": 6040280410,
    "Nile_Aswan_reference": 1040034260,
    "Congo_Kinshasa_reference": 1040020040,
    "Ganges_Patna_reference": 4040960480,
    "Colorado_GrandCanyon_reference": 7040707710,
    "Murray_Mildura_reference": 5040591050,
}

# Also export every Amazon L4 unit that is represented in the atlas.
EXPORT_ALL_AMAZON_L4 = True


# =============================================================================
# HELPERS
# =============================================================================

def sanitize(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def month_serial(s):
    s = pd.to_datetime(s)
    return s.dt.year * 12 + s.dt.month


def detect_basin_column(df):
    for c in ["basin_id", "basin"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find basin_id/basin column.")


def exact_month_diff(a, b):
    return (b.year - a.year) * 12 + (b.month - a.month)


def consecutive_pairs(sub):
    """
    Return row-index pairs only when rolling-window START months are exactly
    one month apart. This prevents fake connections across real gaps.
    """
    times = pd.to_datetime(sub["start_time"]).tolist()
    idx = sub.index.to_numpy()

    pairs = []
    for i in range(1, len(idx)):
        if exact_month_diff(times[i - 1], times[i]) == 1:
            pairs.append((idx[i - 1], idx[i]))
    return pairs


# =============================================================================
# LOAD FINAL ATLAS
# =============================================================================

def load_final_atlas():
    umap = pd.read_parquet(UMAP_FILE).copy()
    clu = pd.read_parquet(CLUSTERS_FILE).copy()

    ub = detect_basin_column(umap)
    umap = umap.rename(columns={ub: "basin_id"})

    required_umap = {"sample_id", "basin_id", "start_time", "end_time", "u1", "u2"}
    miss = required_umap - set(umap.columns)
    if miss:
        raise ValueError(f"Continuous UMAP file missing: {miss}")

    required_clu = {"sample_id", "cluster"}
    miss = required_clu - set(clu.columns)
    if miss:
        raise ValueError(f"Final cluster file missing: {miss}")

    if "k" in clu.columns:
        kvals = sorted(clu["k"].dropna().unique().tolist())
        if kvals != [3]:
            raise ValueError(f"Expected FINAL k=3 file; found k={kvals}")

    atlas = umap.merge(
        clu[["sample_id", "cluster"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if atlas["cluster"].isna().any():
        raise ValueError("Some continuous UMAP windows have no final state label.")

    atlas["basin_id"] = atlas["basin_id"].astype("int64")
    atlas["cluster"] = atlas["cluster"].astype(int)
    atlas["start_time"] = pd.to_datetime(atlas["start_time"])
    atlas["end_time"] = pd.to_datetime(atlas["end_time"])

    atlas["dynamic_mode"] = atlas["cluster"].map(MODE_NAMES)
    atlas["mode_short"] = atlas["cluster"].map(MODE_SHORT)

    print(f"✅ Final continuous atlas: {len(atlas):,} windows")
    print(f"✅ Basins: {atlas['basin_id'].nunique():,}")
    print("✅ Dynamic mode labels:")
    for k in sorted(MODE_NAMES):
        print(f"   M{k}: {MODE_NAMES[k]}")

    return atlas


# =============================================================================
# LOCAL DYNAMIC INDICATORS
# =============================================================================

def load_wavelet_panel():
    w = pd.read_parquet(WAVELET_FILE).copy()

    if "basin" not in w.columns:
        raise ValueError("Wavelet file requires 'basin'.")

    w["basin_id"] = w["basin"].astype("int64")

    if "time" in w.columns:
        w["time"] = pd.to_datetime(w["time"])
    elif {"year", "month"}.issubset(w.columns):
        w["time"] = pd.to_datetime(
            dict(year=w["year"], month=w["month"], day=1)
        )
    else:
        raise ValueError("Wavelet file requires time or year/month.")

    needed = [
        "lwe_thickness_long",
        "lwe_thickness_seasonal",
        "lwe_thickness_short",
        "tp_long",
        "tp_short",
        "swvl4_long",
    ]
    available = [c for c in needed if c in w.columns]

    print(f"✅ Local dynamic-indicator channels available: {len(available)}")
    for c in available:
        print(f"   {c}")

    w = (
        w[["basin_id", "time"] + available]
        .drop_duplicates(["basin_id", "time"])
        .sort_values(["basin_id", "time"])
        .reset_index(drop=True)
    )

    return w, available


def compute_window_dynamic_metrics(atlas, wavelet, available):
    """
    Calculate interpretable temporal summaries for each exact 24-month window.

    Signs are invariant to positive linear scaling, so these direction
    diagnostics are consistent with the standardized-input audit even if
    stored wavelet values are in their pre-model-scaler units.
    """
    lookup = {
        int(b): g.set_index("time").sort_index()
        for b, g in wavelet.groupby("basin_id", sort=False)
    }

    rows = []

    print("🚀 Computing local 24-month dynamic indicators...")

    for n, row in enumerate(atlas.itertuples(index=False), start=1):
        basin_id = int(row.basin_id)
        start = pd.Timestamp(row.start_time)
        end = pd.Timestamp(row.end_time)

        basin = lookup.get(basin_id)
        result = {"sample_id": int(row.sample_id)}

        if basin is None:
            rows.append(result)
            continue

        expected = pd.date_range(start, end, freq="MS")
        sub = basin.reindex(expected)

        if len(expected) != WINDOW_LENGTH or len(sub) != WINDOW_LENGTH:
            rows.append(result)
            continue

        for c in available:
            x = sub[c].to_numpy(dtype=float)

            if not np.isfinite(x).all():
                continue

            delta = float(x[-1] - x[0])
            result[f"{c}__delta"] = delta
            result[f"{c}__rms"] = float(np.sqrt(np.mean(x * x)))
            result[f"{c}__range"] = float(np.max(x) - np.min(x))

            tt = np.arange(WINDOW_LENGTH, dtype=float)
            result[f"{c}__slope"] = float(
                np.polyfit(tt, x, 1)[0]
            )

            result[f"{c}__late6_minus_early6"] = float(
                np.mean(x[-6:]) - np.mean(x[:6])
            )

        rows.append(result)

        if n % 50000 == 0:
            print(f"   {n:,}/{len(atlas):,} windows")

    metrics = pd.DataFrame(rows)

    out = atlas.merge(
        metrics,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    # Window-local direction flags used only as supporting diagnostics.
    if "lwe_thickness_long__delta" in out.columns:
        d = out["lwe_thickness_long__delta"]
        out["local_tws_long_direction"] = np.select(
            [d < 0, d > 0],
            ["decreasing", "increasing"],
            default="near-zero",
        )

    if "tp_long__delta" in out.columns:
        d = out["tp_long__delta"]
        out["local_precip_long_direction"] = np.select(
            [d < 0, d > 0],
            ["decreasing", "increasing"],
            default="near-zero",
        )

    return out


# =============================================================================
# OPTIONAL EXTERNAL VALIDATION COLUMNS
# =============================================================================

def add_optional_validation(df):
    out = df.copy()

    if os.path.exists(SPEI_FILE):
        s = pd.read_parquet(SPEI_FILE)
        candidates = [
            "sample_id",
            "spei12_end",
            "spei12_window_mean",
            "spei12_drought_fraction_le_m1",
            "spei12_wet_fraction_ge_p1",
        ]
        cols = [c for c in candidates if c in s.columns]
        if len(cols) > 1:
            out = out.merge(
                s[cols].drop_duplicates("sample_id"),
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            print("✅ Added SPEI-12 diagnostics")

    if os.path.exists(ENSO_FILE):
        e = pd.read_parquet(ENSO_FILE)
        candidates = [
            "sample_id",
            "oni_end",
            "enso_phase_end",
            "oni_window_mean",
            "el_nino_episode_fraction_24m",
            "la_nina_episode_fraction_24m",
        ]
        cols = [c for c in candidates if c in e.columns]
        if len(cols) > 1:
            out = out.merge(
                e[cols].drop_duplicates("sample_id"),
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            print("✅ Added ENSO diagnostics")

    return out


# =============================================================================
# GEO / SELECTION
# =============================================================================

def load_basins():
    gdf = gpd.read_file(GPKG_FILE)
    needed = {"HYBAS_ID", "MAIN_BAS", "PFAF_ID", "SUB_AREA", "UP_AREA"}
    miss = needed - set(gdf.columns)
    if miss:
        raise ValueError(f"GeoPackage missing fields: {miss}")
    return gdf


def build_selection(gdf, atlas):
    selection = dict(SELECTED_BASINS)

    if EXPORT_ALL_AMAZON_L4:
        amazon = gdf[gdf["MAIN_BAS"] == AMAZON_MAIN_BAS].copy()
        for _, r in amazon.iterrows():
            basin_id = int(r["HYBAS_ID"])
            label = f"Amazon_L4_PFAF{int(r['PFAF_ID'])}"
            selection.setdefault(label, basin_id)

    gpkg_ids = set(gdf["HYBAS_ID"].astype("int64"))
    atlas_ids = set(atlas["basin_id"].astype("int64").unique())

    rows = []
    valid = {}

    for label, basin_id in selection.items():
        basin_id = int(basin_id)
        in_gpkg = basin_id in gpkg_ids
        in_atlas = basin_id in atlas_ids

        meta = gdf[gdf["HYBAS_ID"].astype("int64") == basin_id]

        row = {
            "label": label,
            "HYBAS_ID": basin_id,
            "in_gpkg": in_gpkg,
            "in_atlas": in_atlas,
        }

        if not meta.empty:
            m = meta.iloc[0]
            row.update({
                "MAIN_BAS": int(m["MAIN_BAS"]),
                "PFAF_ID": int(m["PFAF_ID"]),
                "SUB_AREA_km2": float(m["SUB_AREA"]),
                "UP_AREA_km2": float(m["UP_AREA"]),
            })

        rows.append(row)

        if in_gpkg and in_atlas:
            valid[label] = basin_id

    check = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    check.to_csv(
        os.path.join(OUT_DIR, "selected_basin_id_check.csv"),
        index=False,
    )

    return valid, check


# =============================================================================
# PLOTS
# =============================================================================

def plot_state_space_trajectory(atlas, sub, label, basin_id, folder):
    plt.figure(figsize=(10, 8))

    plt.scatter(
        atlas["u1"],
        atlas["u2"],
        s=1,
        alpha=0.04,
        c="0.45",
        rasterized=True,
    )

    for i0, i1 in consecutive_pairs(sub):
        a = sub.loc[i0]
        b = sub.loc[i1]
        plt.plot(
            [a["u1"], b["u1"]],
            [a["u2"], b["u2"]],
            linewidth=0.8,
            alpha=0.35,
            c="0.25",
        )

    sc = plt.scatter(
        sub["u1"],
        sub["u2"],
        c=sub["cluster"],
        cmap="tab10",
        vmin=-0.5,
        vmax=2.5,
        s=28,
        alpha=0.95,
        zorder=3,
    )

    cbar = plt.colorbar(sc, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["M0", "M1", "M2"])
    cbar.set_label("Recurrent 24-month response mode")

    plt.scatter(
        [sub.iloc[0]["u1"]],
        [sub.iloc[0]["u2"]],
        marker="o",
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        zorder=4,
    )
    plt.scatter(
        [sub.iloc[-1]["u1"]],
        [sub.iloc[-1]["u2"]],
        marker="*",
        s=160,
        c="black",
        zorder=4,
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(
        "Dynamic hydrological-state trajectory\n"
        f"{label} | HYBAS_ID {basin_id}"
    )

    # Crucial interpretation note.
    plt.figtext(
        0.5,
        0.01,
        "M0/M1/M2 are recurrent multivariate 24-month modes. "
        "Global physical associations are descriptive, not deterministic for every window. "
        "UMAP line length is not a physical change magnitude.",
        ha="center",
        fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.035, 1, 1])

    out = os.path.join(folder, f"dynamic_trajectory_{sanitize(label)}.png")
    plt.savefig(out, dpi=260, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_dynamic_state_timeline(sub, label, basin_id, folder):
    plt.figure(figsize=(13, 4.5))

    for i0, i1 in consecutive_pairs(sub):
        a = sub.loc[i0]
        b = sub.loc[i1]
        plt.plot(
            [a["end_time"], b["end_time"]],
            [a["cluster"], b["cluster"]],
            linewidth=0.9,
            alpha=0.45,
            c="0.25",
        )

    plt.scatter(
        sub["end_time"],
        sub["cluster"],
        c=sub["cluster"],
        cmap="tab10",
        vmin=-0.5,
        vmax=2.5,
        s=26,
    )

    plt.yticks(
        [0, 1, 2],
        [
            "M0",
            "M1",
            "M2",
        ],
    )
    plt.ylim(-0.4, 2.4)
    plt.xlabel("24-month window end")
    plt.ylabel("Recurrent 24-month response mode")
    plt.title(
        "Evolution of learned hydrological response mode\n"
        f"{label} | HYBAS_ID {basin_id}"
    )
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    out = os.path.join(folder, f"dynamic_state_timeline_{sanitize(label)}.png")
    plt.savefig(out, dpi=260, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_tws_dynamic_indicator(sub, label, basin_id, folder):
    metric = "lwe_thickness_long__delta"
    if metric not in sub.columns or sub[metric].notna().sum() == 0:
        return

    plt.figure(figsize=(13, 4.8))

    # Gap-safe line: never connect across missing GRACE/GRACE-FO intervals.
    for i0, i1 in consecutive_pairs(sub):
        if pd.notna(sub.loc[i0, metric]) and pd.notna(sub.loc[i1, metric]):
            plt.plot(
                [sub.loc[i0, "end_time"], sub.loc[i1, "end_time"]],
                [sub.loc[i0, metric], sub.loc[i1, metric]],
                linewidth=1.0,
                alpha=0.8,
            )

    plt.scatter(
        sub["end_time"],
        sub[metric],
        c=sub["cluster"],
        cmap="tab10",
        vmin=-0.5,
        vmax=2.5,
        s=24,
    )
    plt.axhline(0, linewidth=1.0, linestyle="--")

    plt.xlabel("24-month window end")
    plt.ylabel("Long-component TWS change\n(end minus start)")
    plt.title(
        "Local 24-month storage-evolution diagnostic\n"
        f"{label} | HYBAS_ID {basin_id}"
    )
    plt.tight_layout()

    out = os.path.join(folder, f"tws_long_change_{sanitize(label)}.png")
    plt.savefig(out, dpi=260, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_precip_dynamic_indicator(sub, label, basin_id, folder):
    metric = "tp_long__delta"
    if metric not in sub.columns or sub[metric].notna().sum() == 0:
        return

    plt.figure(figsize=(13, 4.8))

    # Gap-safe line: never connect across missing GRACE/GRACE-FO intervals.
    for i0, i1 in consecutive_pairs(sub):
        if pd.notna(sub.loc[i0, metric]) and pd.notna(sub.loc[i1, metric]):
            plt.plot(
                [sub.loc[i0, "end_time"], sub.loc[i1, "end_time"]],
                [sub.loc[i0, metric], sub.loc[i1, metric]],
                linewidth=1.0,
                alpha=0.8,
            )

    plt.scatter(
        sub["end_time"],
        sub[metric],
        c=sub["cluster"],
        cmap="tab10",
        vmin=-0.5,
        vmax=2.5,
        s=24,
    )
    plt.axhline(0, linewidth=1.0, linestyle="--")

    plt.xlabel("24-month window end")
    plt.ylabel("Long-component precipitation change\n(end minus start)")
    plt.title(
        "Local 24-month precipitation-evolution diagnostic\n"
        f"{label} | HYBAS_ID {basin_id}"
    )
    plt.tight_layout()

    out = os.path.join(folder, f"precip_long_change_{sanitize(label)}.png")
    plt.savefig(out, dpi=260, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# =============================================================================
# SUMMARIES
# =============================================================================

def summarize_basin(sub, label, basin_id):
    sub = sub.sort_values("start_time").copy()

    row = {
        "label": label,
        "HYBAS_ID": int(basin_id),
        "n_windows": len(sub),
        "first_window_start": sub["start_time"].min(),
        "last_window_end": sub["end_time"].max(),
        "latest_mode": int(sub.iloc[-1]["cluster"]),
        "latest_mode_name": MODE_NAMES[int(sub.iloc[-1]["cluster"])],
    }

    for k in [0, 1, 2]:
        frac = float(np.mean(sub["cluster"].to_numpy() == k))
        row[f"M{k}_occupancy_fraction"] = frac

    # Count only actual consecutive-month state changes.
    n_changes = 0
    for i0, i1 in consecutive_pairs(sub):
        if int(sub.loc[i0, "cluster"]) != int(sub.loc[i1, "cluster"]):
            n_changes += 1
    row["adjacent_month_state_changes"] = n_changes

    # Basin-local mean dynamic indicators by state, useful for checking whether
    # the global interpretation is expressed locally.
    for metric in [
        "lwe_thickness_long__delta",
        "lwe_thickness_long__slope",
        "tp_long__delta",
        "swvl4_long__delta",
    ]:
        if metric not in sub.columns:
            continue
        for k in [0, 1, 2]:
            vals = sub.loc[sub["cluster"] == k, metric].dropna()
            row[f"M{k}__{metric}__mean"] = (
                float(vals.mean()) if len(vals) else np.nan
            )

    return row


def export_one(atlas, label, basin_id):
    sub = atlas[
        atlas["basin_id"].astype("int64") == int(basin_id)
    ].copy()

    if sub.empty:
        print(f"⚠️ No continuous atlas windows for {label} ({basin_id})")
        return None

    sub = sub.sort_values("start_time").reset_index(drop=True)

    folder = os.path.join(OUT_DIR, sanitize(label))
    os.makedirs(folder, exist_ok=True)

    csv_out = os.path.join(
        folder,
        f"dynamic_trajectory_points_{sanitize(label)}.csv",
    )
    sub.to_csv(csv_out, index=False)
    print(f"✅ Saved trajectory data: {csv_out}")

    plot_state_space_trajectory(atlas, sub, label, basin_id, folder)
    plot_dynamic_state_timeline(sub, label, basin_id, folder)
    plot_tws_dynamic_indicator(sub, label, basin_id, folder)
    plot_precip_dynamic_indicator(sub, label, basin_id, folder)

    return summarize_basin(sub, label, basin_id)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- FINAL dynamic basin trajectory exporter — v3 ---")
    print("Interpretation:")
    print("  M0 = Balanced / high-variability mode")
    print("  M1 = Globally drying/depletion-associated mode")
    print("  M2 = Globally wetting/recharge-associated mode")
    print("These are multivariate temporal modes; the associations are not deterministic per window.\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    atlas = load_final_atlas()

    wavelet, available = load_wavelet_panel()
    atlas = compute_window_dynamic_metrics(atlas, wavelet, available)
    atlas = add_optional_validation(atlas)

    gdf = load_basins()
    selected, check = build_selection(gdf, atlas)

    print("\nSelected basin availability:")
    print(check.to_string(index=False))

    summaries = []

    print("\nExporting final dynamic trajectories...")
    for label, basin_id in selected.items():
        summary = export_one(atlas, label, basin_id)
        if summary is not None:
            summaries.append(summary)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_out = os.path.join(
            OUT_DIR,
            "selected_basin_dynamic_summary.csv",
        )
        summary_df.to_csv(summary_out, index=False)
        print(f"\n✅ Saved basin summary: {summary_out}")

        print("\nSelected-basin dynamic summary:")
        console_cols = [
            "label",
            "HYBAS_ID",
            "n_windows",
            "M0_occupancy_fraction",
            "M1_occupancy_fraction",
            "M2_occupancy_fraction",
            "latest_mode_name",
            "adjacent_month_state_changes",
        ]
        print(
            summary_df[console_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    print("\nScientific interpretation rule:")
    print("- Use M0/M1/M2 as recurrent 24-month response modes.")
    print("- Never translate a single mode directly into an absolute basin climate class or")
    print("  assume every individual variable must have the globally associated sign.")
    print("- Use the local TWS/precipitation dynamic-indicator figures to describe")
    print("  how the global recurrent mode is expressed in the selected basin.")
    print("- Compare these time-evolving states with static/classic basin classification later.")
    print("- UMAP is a visualization; use latent/dynamic metrics for change magnitude.")
    print("\n--- Done ---")


if __name__ == "__main__":
    main()

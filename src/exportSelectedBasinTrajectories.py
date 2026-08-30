# exportSelectedBasinTrajectories.py
# Publication-oriented trajectory export for selected HydroBASINS L4 units.
#
# The core scientific idea:
#   static/classic basin identity  <->  dynamic 24-month learned hydrological states
#
# This script uses the CONTINUOUS atlas and FINAL k=3 regime assignments.
# It does NOT recompute the autoencoder, UMAP, or clustering.
#
# Outputs per selected basin:
#   - CSV of every valid rolling-window state
#   - year-colored state-space trajectory
#   - final-regime-colored state-space trajectory
#   - final-regime timeline
#
# It only draws trajectory segments between windows whose start months are
# exactly one month apart, so gaps are never connected artificially.

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

GPKG_FILE = "data/interim/hydrobasins_l04_global.gpkg"
UMAP_FILE = "data/processed/embeddings_umap_continuous.parquet"
CLUSTERS_FILE = "data/processed/window_clusters.parquet"

# Optional validation enrichments. If present, their variables are added to
# the exported trajectory CSV automatically.
SPEI_FILE = "data/processed/window_clusters_spei12_validation.parquet"
ENSO_FILE = "data/processed/window_clusters_enso_validation.parquet"

OUT_DIR = "results/trajectories"

# -------------------------------------------------------------------------
# EDIT THIS DICTIONARY whenever you want a different set of poster basins.
# IDs below were checked against the uploaded HydroBASINS L4 GeoPackage.
# -------------------------------------------------------------------------
SELECTED_BASINS = {
    "Amazon_upper_western": 6040280410,
    "Amazon_lower_mainstem": 6040239720,
    "Nile_Aswan_reference": 1040034260,
    "Congo_Kinshasa_reference": 1040020040,
    "Ganges_Patna_reference": 4040960480,
    "Colorado_GrandCanyon_reference": 7040707710,
    "Murray_Mildura_reference": 5040591050,
}

# If True, export every L4 unit belonging to the Amazon main basin system
# that is actually present in the learned atlas.
EXPORT_ALL_AMAZON_L4 = True
AMAZON_MAIN_BAS = 6040007000

# Optional human-readable conventional/static classifications.
# These are placeholders until a formal classification dataset is attached.
# Leave as None rather than treating broad descriptions as formal data.
CLASSIC_CLASSIFICATION = {
    "Amazon_upper_western": None,
    "Amazon_lower_mainstem": None,
    "Nile_Aswan_reference": None,
    "Congo_Kinshasa_reference": None,
    "Ganges_Patna_reference": None,
    "Colorado_GrandCanyon_reference": None,
    "Murray_Mildura_reference": None,
}

REGIME_DISPLAY = {
    0: "Regime 0 (wetter / storage-surplus associated)",
    1: "Regime 1 (intermediate)",
    2: "Regime 2 (drier / storage-deficit associated)",
}


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def month_serial(ts):
    return ts.dt.year * 12 + ts.dt.month


def detect_basin_column(df):
    if "basin" in df.columns:
        return "basin"
    if "basin_id" in df.columns:
        return "basin_id"
    raise ValueError("Could not find basin/basin_id column.")


def load_atlas():
    umap = pd.read_parquet(UMAP_FILE)
    clusters = pd.read_parquet(CLUSTERS_FILE)

    if "sample_id" not in umap.columns or "sample_id" not in clusters.columns:
        raise ValueError("UMAP and final clusters must both contain sample_id.")

    if "k" in clusters.columns:
        kvals = sorted(clusters["k"].dropna().unique().tolist())
        if kvals != [3]:
            raise ValueError(
                f"Expected finalized k=3 production clusters; found k={kvals}"
            )

    ubasin = detect_basin_column(umap)
    umap = umap.rename(columns={ubasin: "basin_id"}).copy()

    # Avoid duplicate metadata columns from merge.
    ckeep = ["sample_id", "cluster"]
    merged = umap.merge(
        clusters[ckeep],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    if merged["cluster"].isna().any():
        n = int(merged["cluster"].isna().sum())
        raise ValueError(f"{n:,} UMAP windows have no final k=3 cluster label.")

    merged["start_time"] = pd.to_datetime(merged["start_time"])
    merged["end_time"] = pd.to_datetime(merged["end_time"])

    print(f"✅ Loaded continuous atlas: {len(merged):,} windows")
    print(f"✅ Atlas basins: {merged['basin_id'].nunique():,}")
    return merged


def load_gpkg():
    gdf = gpd.read_file(GPKG_FILE)
    required = {"HYBAS_ID", "MAIN_BAS", "PFAF_ID", "SUB_AREA", "UP_AREA"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"GeoPackage missing columns: {missing}")
    return gdf


def add_optional_validation(df):
    out = df.copy()

    if os.path.exists(SPEI_FILE):
        spei = pd.read_parquet(SPEI_FILE)
        cols = [
            c for c in [
                "sample_id",
                "spei12_end",
                "spei12_window_mean",
                "spei12_drought_fraction_le_m1",
                "spei12_wet_fraction_ge_p1",
            ]
            if c in spei.columns
        ]
        if len(cols) > 1:
            out = out.merge(
                spei[cols].drop_duplicates("sample_id"),
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            print("✅ Added available SPEI-12 trajectory diagnostics")

    if os.path.exists(ENSO_FILE):
        enso = pd.read_parquet(ENSO_FILE)
        cols = [
            c for c in [
                "sample_id",
                "oni_end",
                "enso_phase_end",
                "oni_window_mean",
                "el_nino_episode_fraction_24m",
                "la_nina_episode_fraction_24m",
            ]
            if c in enso.columns
        ]
        if len(cols) > 1:
            out = out.merge(
                enso[cols].drop_duplicates("sample_id"),
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            print("✅ Added available ENSO trajectory diagnostics")

    return out


def amazon_units_in_atlas(gdf, atlas):
    group = gdf[gdf["MAIN_BAS"] == AMAZON_MAIN_BAS].copy()
    present = set(atlas["basin_id"].astype("int64").unique())
    group["present_in_atlas"] = group["HYBAS_ID"].astype("int64").isin(present)
    return group.sort_values("PFAF_ID")


def build_selection(gdf, atlas):
    selection = dict(SELECTED_BASINS)

    if EXPORT_ALL_AMAZON_L4:
        amazon = amazon_units_in_atlas(gdf, atlas)
        for _, row in amazon.iterrows():
            if bool(row["present_in_atlas"]):
                label = f"Amazon_L4_PFAF{int(row['PFAF_ID'])}"
                selection.setdefault(label, int(row["HYBAS_ID"]))

    # Validate against GeoPackage and atlas.
    gpkg_ids = set(gdf["HYBAS_ID"].astype("int64"))
    atlas_ids = set(atlas["basin_id"].astype("int64").unique())

    rows = []
    valid = {}

    for label, basin_id in selection.items():
        basin_id = int(basin_id)
        in_gpkg = basin_id in gpkg_ids
        in_atlas = basin_id in atlas_ids

        meta = gdf[gdf["HYBAS_ID"].astype("int64") == basin_id]
        if not meta.empty:
            r = meta.iloc[0]
            main_bas = int(r["MAIN_BAS"])
            pfaf = int(r["PFAF_ID"])
            sub_area = float(r["SUB_AREA"])
            up_area = float(r["UP_AREA"])
        else:
            main_bas = np.nan
            pfaf = np.nan
            sub_area = np.nan
            up_area = np.nan

        rows.append({
            "label": label,
            "HYBAS_ID": basin_id,
            "in_gpkg": in_gpkg,
            "in_atlas": in_atlas,
            "MAIN_BAS": main_bas,
            "PFAF_ID": pfaf,
            "SUB_AREA_km2": sub_area,
            "UP_AREA_km2": up_area,
        })

        if in_gpkg and in_atlas:
            valid[label] = basin_id

    check = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    check.to_csv(os.path.join(OUT_DIR, "selected_basin_id_check.csv"), index=False)

    print("\nSelected basin availability:")
    print(check.to_string(index=False))

    missing = check[~check["in_atlas"]]
    if len(missing):
        print(
            "\n⚠️ Some selected L4 IDs are valid GeoPackage basins but are absent "
            "from the learned atlas (usually because they lacked enough complete "
            "24-month input windows). They will be skipped."
        )

    return valid, check


def consecutive_segments(sub):
    """Return index-pair segments only for exactly one-month-adjacent starts."""
    serial = month_serial(sub["start_time"]).to_numpy()
    idx = sub.index.to_numpy()
    return [
        (idx[i - 1], idx[i])
        for i in range(1, len(idx))
        if serial[i] - serial[i - 1] == 1
    ]


def title_suffix(label, basin_id):
    classic = CLASSIC_CLASSIFICATION.get(label)
    if classic:
        return f"{label} | HYBAS_ID {basin_id} | classic: {classic}"
    return f"{label} | HYBAS_ID {basin_id}"


def plot_year_trajectory(atlas, sub, label, basin_id, folder):
    plt.figure(figsize=(10, 8))

    # Global learned manifold as context.
    plt.scatter(
        atlas["u1"],
        atlas["u2"],
        s=1.0,
        alpha=0.05,
        c="0.45",
        rasterized=True,
    )

    # Gap-safe temporal segments.
    for i0, i1 in consecutive_segments(sub):
        a = sub.loc[i0]
        b = sub.loc[i1]
        plt.plot(
            [a["u1"], b["u1"]],
            [a["u2"], b["u2"]],
            linewidth=0.8,
            alpha=0.45,
            c="0.2",
        )

    years = sub["start_time"].dt.year
    sc = plt.scatter(
        sub["u1"],
        sub["u2"],
        c=years,
        s=24,
        alpha=0.95,
        cmap="viridis",
        zorder=3,
    )

    # Mark first and last state.
    plt.scatter(
        [sub.iloc[0]["u1"]],
        [sub.iloc[0]["u2"]],
        marker="o",
        s=95,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        zorder=4,
    )
    plt.scatter(
        [sub.iloc[-1]["u1"]],
        [sub.iloc[-1]["u2"]],
        marker="*",
        s=150,
        c="black",
        zorder=4,
    )

    plt.colorbar(sc, label="Window start year")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Dynamic hydrological-state trajectory\n" + title_suffix(label, basin_id))
    plt.tight_layout()

    out = os.path.join(folder, f"trajectory_year_{sanitize(label)}.png")
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_regime_trajectory(atlas, sub, label, basin_id, folder):
    plt.figure(figsize=(10, 8))
    plt.scatter(
        atlas["u1"],
        atlas["u2"],
        s=1.0,
        alpha=0.05,
        c="0.45",
        rasterized=True,
    )

    for i0, i1 in consecutive_segments(sub):
        a = sub.loc[i0]
        b = sub.loc[i1]
        plt.plot(
            [a["u1"], b["u1"]],
            [a["u2"], b["u2"]],
            linewidth=0.8,
            alpha=0.45,
            c="0.2",
        )

    sc = plt.scatter(
        sub["u1"],
        sub["u2"],
        c=sub["cluster"],
        s=25,
        alpha=0.95,
        cmap="tab10",
        vmin=-0.5,
        vmax=2.5,
        zorder=3,
    )
    cbar = plt.colorbar(sc, ticks=[0, 1, 2])
    cbar.set_label("Final k=3 regime")

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Trajectory through final recurrent regimes\n" + title_suffix(label, basin_id))
    plt.tight_layout()

    out = os.path.join(folder, f"trajectory_regime_{sanitize(label)}.png")
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_regime_timeline(sub, label, basin_id, folder):
    plt.figure(figsize=(12, 4.2))
    plt.scatter(
        sub["end_time"],
        sub["cluster"],
        c=sub["cluster"],
        cmap="tab10",
        vmin=-0.5,
        vmax=2.5,
        s=22,
    )

    # Only connect one-month-adjacent windows.
    for i0, i1 in consecutive_segments(sub):
        a = sub.loc[i0]
        b = sub.loc[i1]
        plt.plot(
            [a["end_time"], b["end_time"]],
            [a["cluster"], b["cluster"]],
            c="0.25",
            linewidth=0.8,
            alpha=0.5,
        )

    plt.yticks(
        [0, 1, 2],
        [
            "R0\nwetter/storage+",
            "R1\nintermediate",
            "R2\ndrier/storage−",
        ],
    )
    plt.ylim(-0.45, 2.45)
    plt.xlabel("24-month window end")
    plt.ylabel("Learned regime")
    plt.title("Dynamic regime timeline\n" + title_suffix(label, basin_id))
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    out = os.path.join(folder, f"regime_timeline_{sanitize(label)}.png")
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def export_one(atlas, label, basin_id):
    sub = atlas[atlas["basin_id"].astype("int64") == int(basin_id)].copy()
    if sub.empty:
        print(f"⚠️ No atlas windows for {label} ({basin_id}); skipped.")
        return

    sub = sub.sort_values("start_time").reset_index(drop=True)

    # Reset index intentionally so segment helper can use .loc safely.
    folder = os.path.join(OUT_DIR, sanitize(label))
    os.makedirs(folder, exist_ok=True)

    csv_out = os.path.join(folder, f"trajectory_points_{sanitize(label)}.csv")
    sub.to_csv(csv_out, index=False)
    print(f"✅ Saved trajectory data: {csv_out}")

    plot_year_trajectory(atlas, sub, label, basin_id, folder)
    plot_regime_trajectory(atlas, sub, label, basin_id, folder)
    plot_regime_timeline(sub, label, basin_id, folder)

    changes = int((sub["cluster"] != sub["cluster"].shift(1)).sum() - 1)
    print(
        f"   {label}: {len(sub):,} windows | "
        f"{sub['start_time'].min().date()} → {sub['end_time'].max().date()} | "
        f"raw sequential regime-label changes={max(changes, 0)}"
    )


def main():
    print("--- Export selected continuous basin trajectories ---")
    atlas = load_atlas()
    atlas = add_optional_validation(atlas)
    gdf = load_gpkg()

    selected, check = build_selection(gdf, atlas)

    amazon = amazon_units_in_atlas(gdf, atlas)[
        [
            "HYBAS_ID", "MAIN_BAS", "PFAF_ID", "SUB_AREA", "UP_AREA",
            "NEXT_DOWN", "present_in_atlas"
        ]
    ].copy()
    amazon_out = os.path.join(OUT_DIR, "amazon_level4_units_in_atlas.csv")
    amazon.to_csv(amazon_out, index=False)
    print(f"\n✅ Saved Amazon L4 membership table: {amazon_out}")

    print("\nExporting trajectories...")
    for label, basin_id in selected.items():
        export_one(atlas, label, basin_id)

    print("\nImportant interpretation:")
    print("- Each trajectory belongs to one HydroBASINS L4 polygon.")
    print("- HYBAS_ID 6040007000 is the terminal local Amazon L4 unit; it is NOT")
    print("  a whole-Amazon spatial mean.")
    print("- The entire Amazon system in this Level-4 file is identified by")
    print("  MAIN_BAS == 6040007000, comprising 9 Level-4 units.")
    print("- For a poster-level 'Amazon Basin' story, inspect multiple Amazon L4")
    print("  trajectories or build a separate whole-Amazon aggregate later.")
    print("- Classic/static classifications should be attached from a formal")
    print("  classification source; the HydroBASINS file itself contains topology,")
    print("  area and IDs, not climate-regime classes.")
    print("--- Done ---")


if __name__ == "__main__":
    main()

# validateStrictMigrationPhysicalSignals.py
#
# Physical validation of a SMALL, diverse subset of the duration-matched
# strict Köppen-associated hydrological migration candidates.
#
# This is intentionally NOT another latent-space experiment.
# It asks whether the selected candidate trajectories are accompanied by
# physically interpretable changes in the basin time series already used by
# the model: GRACE TWSA, precipitation, deep soil moisture, runoff, etc.
#
# IMPORTANT:
# - The autoencoder stays frozen.
# - Candidate status comes from the completed duration-matched support audit.
# - We select at most one basin per source->target pathway first, to avoid a
#   poster panel dominated by repeated variants of the same transition.
# - Temperature is used only if it already exists in the processed dataset.
#   The original ERA5 input list did not include 2-m temperature, so this script
#   does NOT invent or download it. If absent, it reports that explicitly.
#
# Interpretation remains:
# "hydrological dynamics became more similar to a historical climate-associated
# signature", not "the formal Köppen class changed".

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

STRICT_FILE = Path(
    "results/tables/koppen_duration_matched_support/"
    "duration_matched_strict_migrations_q95.csv"
)

# Main physical monthly table used before SWT. This is preferred because it
# retains anomaly-normalized monthly variables directly.
PHYSICAL_FILE = Path(
    "data/processed/basin_dataset_anomaly_normalized.parquet"
)

# Optional multiscale table, used for long-component diagnostics if available.
WAVELET_FILE = Path(
    "data/processed/basin_dataset_wavelet_multiscale.parquet"
)

OUT_DIR = Path("results/tables/strict_migration_physical_validation")
FIG_DIR = Path("results/figures/strict_migration_physical_validation")

SELECTED_OUT = OUT_DIR / "selected_strict_migration_candidates.csv"
PHYSICAL_SUMMARY_OUT = OUT_DIR / "candidate_historical_vs_recent_physical_summary.csv"
AVAILABILITY_OUT = OUT_DIR / "physical_variable_availability.csv"
QC_OUT = OUT_DIR / "strict_migration_physical_validation_qc.csv"
CONFIG_OUT = OUT_DIR / "strict_migration_physical_validation_config.json"

HIST_END = pd.Timestamp("2020-12-01")
RECENT_START = pd.Timestamp("2021-01-01")

N_CANDIDATES = 6

# Prefer pathways with strong historical pair separability, then candidates
# well inside target support and clearly outside source support.
MIN_PAIR_AUC = 0.80


# =============================================================================
# HELPERS
# =============================================================================

def detect_basin_column(df):
    for c in ["basin", "basin_id", "HYBAS_ID"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect basin identifier column.")


def canonical_time(df):
    d = df.copy()
    if "time" in d.columns:
        d["time"] = pd.to_datetime(d["time"])
    elif "time_era5" in d.columns:
        d["time"] = pd.to_datetime(d["time_era5"])
    elif {"year", "month"}.issubset(d.columns):
        d["time"] = pd.to_datetime(
            dict(year=d["year"], month=d["month"], day=1)
        )
    elif "time_grace" in d.columns:
        d["time"] = pd.to_datetime(d["time_grace"])
    else:
        raise ValueError("Could not construct monthly time column.")
    return d


def linear_slope(values):
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def period_stats(g, col, prefix):
    x = pd.to_numeric(g[col], errors="coerce")
    return {
        f"{prefix}_n": int(x.notna().sum()),
        f"{prefix}_mean": float(x.mean()) if x.notna().any() else np.nan,
        f"{prefix}_median": float(x.median()) if x.notna().any() else np.nan,
        f"{prefix}_std": float(x.std(ddof=0)) if x.notna().any() else np.nan,
        f"{prefix}_slope_per_month": linear_slope(x.to_numpy(dtype=float)),
        f"{prefix}_frac_below_minus1": (
            float((x < -1.0).mean()) if x.notna().any() else np.nan
        ),
        f"{prefix}_frac_above_plus1": (
            float((x > 1.0).mean()) if x.notna().any() else np.nan
        ),
    }


# =============================================================================
# LOAD + SELECT DIVERSE CANDIDATES
# =============================================================================

def load_strict_candidates():
    c = pd.read_csv(STRICT_FILE)

    required = {
        "basin_id",
        "source_KG",
        "best_supported_validated_target",
        "KG_dom_frac",
        "source_support_ratio",
        "target_support_ratio",
        "source_target_pairwise_auc",
        "pair_axis_delta_toward_target",
    }
    missing = required - set(c.columns)
    if missing:
        raise ValueError(f"Strict candidate table missing: {sorted(missing)}")

    c["basin_id"] = pd.to_numeric(
        c["basin_id"], errors="raise"
    ).astype("int64")

    c = c[
        c["source_target_pairwise_auc"] >= MIN_PAIR_AUC
    ].copy()

    # Ranking aid only, not a statistical probability:
    # prefer high AUC, low target support ratio, clear source departure,
    # and large directional change.
    c["physical_validation_priority"] = (
        (2.0 * c["source_target_pairwise_auc"] - 1.0)
        * c["pair_axis_delta_toward_target"].clip(lower=0)
        * c["source_support_ratio"].clip(lower=1.0)
        / c["target_support_ratio"].clip(lower=0.10)
    )

    c["pathway"] = (
        c["source_KG"].astype(str)
        + "→"
        + c["best_supported_validated_target"].astype(str)
    )

    return c


def select_diverse_candidates(c):
    # First pass: best candidate from each unique pathway.
    pathway_best = (
        c.sort_values(
            [
                "source_target_pairwise_auc",
                "physical_validation_priority",
            ],
            ascending=[False, False],
        )
        .groupby("pathway", as_index=False, sort=False)
        .head(1)
        .copy()
    )

    # Encourage source-class diversity too.
    selected = []
    used_sources = set()

    for _, r in pathway_best.sort_values(
        [
            "source_target_pairwise_auc",
            "physical_validation_priority",
        ],
        ascending=[False, False],
    ).iterrows():
        if len(selected) >= N_CANDIDATES:
            break
        source = str(r["source_KG"])
        if source not in used_sources:
            selected.append(r)
            used_sources.add(source)

    # Fill remaining slots with strongest unused pathways.
    selected_paths = {
        str(r["pathway"]) for r in selected
    }
    if len(selected) < N_CANDIDATES:
        for _, r in pathway_best.sort_values(
            [
                "source_target_pairwise_auc",
                "physical_validation_priority",
            ],
            ascending=[False, False],
        ).iterrows():
            if len(selected) >= N_CANDIDATES:
                break
            if str(r["pathway"]) in selected_paths:
                continue
            selected.append(r)
            selected_paths.add(str(r["pathway"]))

    if not selected:
        raise ValueError("No strict candidates available for selection.")

    out = pd.DataFrame(selected).reset_index(drop=True)
    out.insert(0, "selection_rank", np.arange(1, len(out) + 1))

    return out


# =============================================================================
# PHYSICAL VARIABLE DISCOVERY
# =============================================================================

def discover_variables(df):
    cols = set(df.columns)

    # Keep exactly one preferred column per physical concept.
    candidates = {
        "GRACE_TWSA": [
            "lwe_thickness_anomaly_normalized",
            "lwe_thickness_anomaly",
            "lwe_thickness",
        ],
        "precipitation": [
            "tp_anomaly_normalized",
            "tp_anomaly",
            "tp",
        ],
        "deep_soil_moisture": [
            "swvl4_anomaly_normalized",
            "swvl4_anomaly",
            "swvl4",
        ],
        "surface_runoff": [
            "sro_anomaly_normalized",
            "sro_anomaly",
            "sro",
        ],
        "subsurface_runoff": [
            "ssro_anomaly_normalized",
            "ssro_anomaly",
            "ssro",
        ],
        "evaporation": [
            "e_anomaly_normalized",
            "e_anomaly",
            "e",
        ],
        "potential_evaporation": [
            "pev_anomaly_normalized",
            "pev_anomaly",
            "pev",
        ],
        "temperature_2m": [
            "t2m_anomaly_normalized",
            "t2m_anomaly",
            "t2m",
            "temperature_2m_anomaly_normalized",
            "temperature_2m",
            "2m_temperature",
        ],
    }

    found = {}
    rows = []

    for concept, opts in candidates.items():
        chosen = next((c for c in opts if c in cols), None)
        found[concept] = chosen
        rows.append({
            "physical_concept": concept,
            "selected_column": chosen,
            "available": chosen is not None,
        })

    return found, pd.DataFrame(rows)


# =============================================================================
# SUMMARIZE CANDIDATE PHYSICAL CHANGE
# =============================================================================

def summarize_candidate(df, basin_id, variable_map):
    g = df[df["basin_id"] == basin_id].copy()
    g = g.sort_values("time")

    hist = g[g["time"] <= HIST_END].copy()
    recent = g[g["time"] >= RECENT_START].copy()

    rows = []

    for concept, col in variable_map.items():
        if col is None:
            continue

        h = period_stats(hist, col, "historical")
        r = period_stats(recent, col, "recent")

        row = {
            "basin_id": int(basin_id),
            "physical_concept": concept,
            "column": col,
            **h,
            **r,
        }

        row["recent_minus_historical_mean"] = (
            row["recent_mean"] - row["historical_mean"]
            if np.isfinite(row["recent_mean"])
            and np.isfinite(row["historical_mean"])
            else np.nan
        )
        row["recent_minus_historical_slope"] = (
            row["recent_slope_per_month"]
            - row["historical_slope_per_month"]
            if np.isfinite(row["recent_slope_per_month"])
            and np.isfinite(row["historical_slope_per_month"])
            else np.nan
        )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# OPTIONAL WAVELET LONG-COMPONENT SUMMARY
# =============================================================================

def summarize_wavelet_long(wavelet, basin_ids):
    if wavelet is None:
        return pd.DataFrame()

    d = wavelet[wavelet["basin_id"].isin(basin_ids)].copy()
    long_cols = [
        c for c in [
            "lwe_thickness_long",
            "tp_long",
            "swvl4_long",
            "sro_long",
            "ssro_long",
        ]
        if c in d.columns
    ]

    rows = []
    for basin_id, g in d.groupby("basin_id"):
        g = g.sort_values("time")
        hist = g[g["time"] <= HIST_END]
        recent = g[g["time"] >= RECENT_START]

        for col in long_cols:
            rows.append({
                "basin_id": int(basin_id),
                "physical_concept": f"wavelet_long::{col}",
                "column": col,
                "historical_mean": float(hist[col].mean()),
                "recent_mean": float(recent[col].mean()),
                "recent_minus_historical_mean": float(
                    recent[col].mean() - hist[col].mean()
                ),
                "historical_slope_per_month": linear_slope(
                    hist[col].to_numpy(dtype=float)
                ),
                "recent_slope_per_month": linear_slope(
                    recent[col].to_numpy(dtype=float)
                ),
            })

    return pd.DataFrame(rows)


# =============================================================================
# FIGURES
# =============================================================================

def plot_candidate(df, candidate, variable_map):
    basin_id = int(candidate["basin_id"])
    source = str(candidate["source_KG"])
    target = str(candidate["best_supported_validated_target"])

    g = df[df["basin_id"] == basin_id].copy().sort_values("time")
    if g.empty:
        return

    preferred = [
        ("GRACE_TWSA", variable_map.get("GRACE_TWSA")),
        ("precipitation", variable_map.get("precipitation")),
        ("deep_soil_moisture", variable_map.get("deep_soil_moisture")),
        ("temperature_2m", variable_map.get("temperature_2m")),
    ]
    preferred = [(a, b) for a, b in preferred if b is not None]

    # If no temperature exists, use runoff as the fourth panel if available.
    if len(preferred) < 4 and variable_map.get("surface_runoff") is not None:
        if all(a != "surface_runoff" for a, _ in preferred):
            preferred.append(
                ("surface_runoff", variable_map["surface_runoff"])
            )

    preferred = preferred[:4]
    if not preferred:
        return

    fig, axes = plt.subplots(
        len(preferred),
        1,
        figsize=(12, 2.7 * len(preferred)),
        sharex=True,
    )
    if len(preferred) == 1:
        axes = [axes]

    for ax, (concept, col) in zip(axes, preferred):
        ax.plot(g["time"], g[col], linewidth=1.0)
        ax.axvline(RECENT_START, linestyle="--", linewidth=1)
        ax.axhline(0, linewidth=0.7, linestyle=":")
        ax.set_ylabel(concept)
        ax.set_title(col, fontsize=9)

    axes[-1].set_xlabel("Month")
    fig.suptitle(
        f"Basin {basin_id}: {source} → {target} hydrological-signature candidate\n"
        f"KG source fraction={candidate['KG_dom_frac']:.3f}, "
        f"pair AUC={candidate['source_target_pairwise_auc']:.3f}",
        y=0.995,
    )
    plt.tight_layout()

    out = FIG_DIR / f"basin_{basin_id}_{source}_to_{target}_physical_timeseries.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# =============================================================================
# CONSOLE
# =============================================================================

def print_console(selected, availability, summary, wavelet_summary):
    print("\n" + "=" * 100)
    print("STRICT MIGRATION CANDIDATE PHYSICAL VALIDATION")
    print("=" * 100)

    print("\nA) Selected diverse candidates")
    cols = [
        "selection_rank",
        "basin_id",
        "source_KG",
        "best_supported_validated_target",
        "KG_dom_frac",
        "source_support_ratio",
        "target_support_ratio",
        "source_target_pairwise_auc",
        "pair_axis_delta_toward_target",
    ]
    print(
        selected[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nB) Physical variable availability")
    print(availability.to_string(index=False))

    print("\nC) Historical vs recent monthly physical changes")
    show = summary[
        [
            "basin_id",
            "physical_concept",
            "historical_mean",
            "recent_mean",
            "recent_minus_historical_mean",
            "historical_slope_per_month",
            "recent_slope_per_month",
            "historical_frac_below_minus1",
            "recent_frac_below_minus1",
            "historical_frac_above_plus1",
            "recent_frac_above_plus1",
        ]
    ]
    print(
        show.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    if not wavelet_summary.empty:
        print("\nD) Long-component change summary")
        print(
            wavelet_summary[
                [
                    "basin_id",
                    "physical_concept",
                    "historical_mean",
                    "recent_mean",
                    "recent_minus_historical_mean",
                    "historical_slope_per_month",
                    "recent_slope_per_month",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )
    else:
        print("\nD) Long-component change summary")
        print("Wavelet long-component table unavailable or no target columns found.")

    print("\nE) Interpretation guardrails")
    temp_available = bool(
        availability.loc[
            availability["physical_concept"] == "temperature_2m",
            "available",
        ].iloc[0]
    )
    if temp_available:
        print("- 2-m temperature is available and included as an independent physical diagnostic.")
    else:
        print(
            "- 2-m temperature is NOT present in the existing processed ERA5 dataset. "
            "Do not make formal Dfb/Dfc or other temperature-defined Köppen-change claims yet."
        )
    print(
        "- The selected basins are hypotheses for physical validation; they are "
        "not confirmed formal climate reclassifications."
    )
    print(
        "- Look for coherent multi-variable changes in GRACE TWSA, precipitation, "
        "soil moisture and runoff, not a single-variable excursion."
    )
    print(
        "- This step is descriptive candidate validation. Do not add new model "
        "training or clustering."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Strict migration candidate physical validation ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    candidates = load_strict_candidates()
    selected = select_diverse_candidates(candidates)

    selected.to_csv(SELECTED_OUT, index=False)
    print(f"✅ Saved: {SELECTED_OUT}")

    phys = pd.read_parquet(PHYSICAL_FILE).copy()
    bcol = detect_basin_column(phys)
    phys["basin_id"] = pd.to_numeric(
        phys[bcol], errors="raise"
    ).astype("int64")
    phys = canonical_time(phys)

    variable_map, availability = discover_variables(phys)
    availability.to_csv(AVAILABILITY_OUT, index=False)
    print(f"✅ Saved: {AVAILABILITY_OUT}")

    rows = []
    for basin_id in selected["basin_id"].astype("int64"):
        s = summarize_candidate(
            phys,
            basin_id,
            variable_map,
        )
        if not s.empty:
            rows.append(s)

    if not rows:
        raise ValueError(
            "No candidate physical summaries could be built from "
            f"{PHYSICAL_FILE}"
        )

    summary = pd.concat(rows, ignore_index=True)

    # Add candidate metadata to every physical row.
    meta = selected[
        [
            "basin_id",
            "selection_rank",
            "source_KG",
            "best_supported_validated_target",
            "KG_dom_frac",
            "source_target_pairwise_auc",
            "source_support_ratio",
            "target_support_ratio",
            "pair_axis_delta_toward_target",
        ]
    ]
    summary = summary.merge(
        meta,
        on="basin_id",
        how="left",
        validate="many_to_one",
    )

    # Optional long components.
    wavelet_summary = pd.DataFrame()
    if WAVELET_FILE.exists():
        wave = pd.read_parquet(WAVELET_FILE).copy()
        wb = detect_basin_column(wave)
        wave["basin_id"] = pd.to_numeric(
            wave[wb], errors="raise"
        ).astype("int64")
        wave = canonical_time(wave)

        wavelet_summary = summarize_wavelet_long(
            wave,
            set(selected["basin_id"].astype("int64")),
        )
        if not wavelet_summary.empty:
            wavelet_summary = wavelet_summary.merge(
                meta,
                on="basin_id",
                how="left",
                validate="many_to_one",
            )

    # Save combined monthly/long table in one CSV using union of columns.
    if not wavelet_summary.empty:
        combined = pd.concat(
            [summary, wavelet_summary],
            ignore_index=True,
            sort=False,
        )
    else:
        combined = summary.copy()

    combined.to_csv(PHYSICAL_SUMMARY_OUT, index=False)
    print(f"✅ Saved: {PHYSICAL_SUMMARY_OUT}")

    for _, candidate in selected.iterrows():
        plot_candidate(
            phys,
            candidate,
            variable_map,
        )

    qc = pd.DataFrame([
        {
            "check": "strict_candidates_available",
            "value": len(candidates),
        },
        {
            "check": "selected_for_physical_validation",
            "value": len(selected),
        },
        {
            "check": "unique_selected_pathways",
            "value": selected["pathway"].nunique(),
        },
        {
            "check": "temperature_available",
            "value": bool(variable_map.get("temperature_2m")),
        },
        {
            "check": "monthly_physical_summary_rows",
            "value": len(summary),
        },
        {
            "check": "wavelet_long_summary_rows",
            "value": len(wavelet_summary),
        },
    ])
    qc.to_csv(QC_OUT, index=False)
    print(f"✅ Saved: {QC_OUT}")

    config = {
        "n_candidates_requested": N_CANDIDATES,
        "candidate_selection": (
            "one best candidate per source-target pathway first, "
            "with source-class diversity, then strongest remaining pathways"
        ),
        "historical_end": str(HIST_END.date()),
        "recent_start": str(RECENT_START.date()),
        "physical_file": str(PHYSICAL_FILE),
        "wavelet_file": str(WAVELET_FILE),
        "temperature_required_for_formal_koppen_change": True,
        "autoencoder_retrained": False,
    }
    with open(CONFIG_OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved: {CONFIG_OUT}")

    print_console(
        selected,
        availability,
        summary,
        wavelet_summary,
    )

    print("\n--- Done ---")
    print(
        "STOP here. Interpret A-E and the six physical time-series figures "
        "before adding temperature data or selecting final poster examples."
    )


if __name__ == "__main__":
    main()

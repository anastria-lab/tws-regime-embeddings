# validateFinalRegimesWithENSO.py
# Validation B2: ENSO association of the FINAL k=3 hydrological regimes.
#
# Inputs:
#   data/processed/window_clusters.parquet
#   data/processed/embeddings_window_level_continuous_v3.parquet
#   data/processed/enso_oni_monthly_v6.csv
#   results/tables/enso_oni_episode_summary.csv
#
# Main questions:
# 1) Does basin-balanced regime occupancy shift during El Niño / La Niña
#    relative to ENSO-neutral months?
# 2) Does rolling-window hydrological state motion / regime switching differ
#    by ENSO phase?
# 3) Are associations globally coherent or spatially heterogeneous?
#
# IMPORTANT:
# ENSO teleconnections are geographically heterogeneous. We do NOT expect a
# universal "El Niño = dry regime" mapping. Main analyses first aggregate
# overlapping monthly windows within basin × ENSO phase, then compare basins.
#
# Adjacent 24-month windows overlap by 23/24 months. Absolute persistence and
# transition rates are therefore not treated as independent Markov events.

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

CLUSTERS_FILE = "data/processed/window_clusters.parquet"
EMBEDDINGS_FILE = "data/processed/embeddings_window_level_continuous_v3.parquet"
ONI_FILE = "data/processed/enso_oni_monthly_v6.csv"
EPISODES_FILE = "results/tables/enso_oni_episode_summary.csv"

ENRICHED_OUTPUT = "data/processed/window_clusters_enso_validation.parquet"

TABLE_DIR = "results/tables/enso_validation"
FIG_DIR = "results/figures/enso_validation"

OCCUPANCY_BASIN_OUTPUT = os.path.join(TABLE_DIR, "enso_basin_phase_regime_occupancy.csv")
OCCUPANCY_SUMMARY_OUTPUT = os.path.join(TABLE_DIR, "enso_regime_occupancy_summary.csv")
OCCUPANCY_DELTA_OUTPUT = os.path.join(TABLE_DIR, "enso_regime_occupancy_shift_vs_neutral.csv")
TRANSITION_BASIN_OUTPUT = os.path.join(TABLE_DIR, "enso_basin_phase_transition_metrics.csv")
TRANSITION_SUMMARY_OUTPUT = os.path.join(TABLE_DIR, "enso_transition_summary.csv")
EVENT_OUTPUT = os.path.join(TABLE_DIR, "enso_event_regime_summary.csv")
COVERAGE_OUTPUT = os.path.join(TABLE_DIR, "enso_validation_coverage.csv")

N_BOOT = 1000
RANDOM_STATE = 42
MIN_WINDOWS_PER_BASIN_PHASE = 5
PHASE_ORDER = ["La_Nina", "Neutral", "El_Nino"]
FINAL_K = 3


def bootstrap_mean_ci(values, rng):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan

    out = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(x), size=len(x))
        out[i] = x[idx].mean()

    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def load_inputs():
    clusters = pd.read_parquet(CLUSTERS_FILE)
    emb = pd.read_parquet(EMBEDDINGS_FILE)
    oni = pd.read_csv(ONI_FILE, parse_dates=["time"])

    required_c = {
        "sample_id", "basin_id", "start_time", "end_time", "split", "cluster", "k"
    }
    missing = required_c - set(clusters.columns)
    if missing:
        raise ValueError(f"Final cluster file missing columns: {missing}")

    if sorted(clusters["k"].dropna().unique().tolist()) != [FINAL_K]:
        raise ValueError("Production clusters are not finalized at k=3.")

    clusters = clusters.copy()
    clusters["start_time"] = pd.to_datetime(clusters["start_time"])
    clusters["end_time"] = pd.to_datetime(clusters["end_time"])

    zcols = [c for c in emb.columns if c.startswith("z") and c[1:].isdigit()]
    zcols = sorted(zcols, key=lambda c: int(c[1:]))
    if len(zcols) != 16:
        raise ValueError(f"Expected 16 latent dimensions, found {len(zcols)}")

    if "episode_phase" not in oni.columns or "oni" not in oni.columns:
        raise ValueError(
            "Prepared ONI file lacks episode_phase/oni. Run prepareNOAAONIv6.py first."
        )

    print(f"✅ Loaded final k=3 clusters: {len(clusters):,} windows")
    print(f"✅ Loaded continuous embeddings: {len(emb):,} windows")
    print(
        f"✅ Loaded ONI: {oni['time'].min().date()} to {oni['time'].max().date()}"
    )

    return clusters, emb, oni, zcols


def build_oni_window_metrics(oni):
    g = oni.sort_values("time").copy()
    s = g["oni"].astype(float)

    out = g[["time", "oni", "episode_phase", "threshold_phase"]].copy()
    out = out.rename(
        columns={
            "time": "end_time",
            "oni": "oni_end",
            "episode_phase": "enso_phase_end",
            "threshold_phase": "enso_threshold_phase_end",
        }
    )

    # Trailing 24-month summaries, aligned with the hydrological window end.
    out["oni_window_mean"] = s.rolling(24, min_periods=24).mean().to_numpy()
    out["oni_window_min"] = s.rolling(24, min_periods=24).min().to_numpy()
    out["oni_window_max"] = s.rolling(24, min_periods=24).max().to_numpy()
    out["oni_window_max_abs"] = (
        s.abs().rolling(24, min_periods=24).max().to_numpy()
    )

    for phase, col in [
        ("El_Nino", "el_nino_episode_fraction_24m"),
        ("La_Nina", "la_nina_episode_fraction_24m"),
        ("Neutral", "neutral_fraction_24m"),
    ]:
        flag = (g["episode_phase"] == phase).astype(float)
        out[col] = flag.rolling(24, min_periods=24).mean().to_numpy()

    return out


def attach_oni(clusters, oni_metrics):
    out = clusters.merge(
        oni_metrics,
        on="end_time",
        how="left",
        validate="many_to_one",
    )
    out["has_enso_validation"] = out["oni_end"].notna()
    return out


def coverage_table(df):
    rows = []
    for label, g in [("ALL", df)] + list(df.groupby("split")):
        valid = g["has_enso_validation"]
        rows.append({
            "period": label,
            "n_windows": len(g),
            "n_validated": int(valid.sum()),
            "validation_fraction": float(valid.mean()),
            "first_window_end": g["end_time"].min(),
            "last_window_end": g["end_time"].max(),
            "last_validated_window_end": (
                g.loc[valid, "end_time"].max() if valid.any() else pd.NaT
            ),
        })
    return pd.DataFrame(rows)


def build_basin_phase_occupancy(valid):
    # Require an official episode phase at the endpoint.
    v = valid[valid["enso_phase_end"].isin(PHASE_ORDER)].copy()

    # Number of windows available in each basin × phase.
    group_n = (
        v.groupby(["basin_id", "enso_phase_end"])
        .size()
        .rename("n_phase_windows")
        .reset_index()
    )
    group_n = group_n[group_n["n_phase_windows"] >= MIN_WINDOWS_PER_BASIN_PHASE]

    v = v.merge(
        group_n,
        on=["basin_id", "enso_phase_end"],
        how="inner",
        validate="many_to_one",
    )

    counts = (
        v.groupby(["basin_id", "enso_phase_end", "cluster"])
        .size()
        .rename("n_regime_windows")
        .reset_index()
    )

    # Complete missing cluster combinations with occupancy=0.
    bases = group_n[["basin_id", "enso_phase_end", "n_phase_windows"]].copy()
    frames = []
    for cluster in range(FINAL_K):
        tmp = bases.copy()
        tmp["cluster"] = cluster
        frames.append(tmp)
    full = pd.concat(frames, ignore_index=True)

    full = full.merge(
        counts,
        on=["basin_id", "enso_phase_end", "cluster"],
        how="left",
        validate="one_to_one",
    )
    full["n_regime_windows"] = full["n_regime_windows"].fillna(0).astype(int)
    full["occupancy_fraction"] = full["n_regime_windows"] / full["n_phase_windows"]

    return full


def summarize_occupancy(occ):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for (phase, cluster), g in occ.groupby(["enso_phase_end", "cluster"]):
        x = g["occupancy_fraction"].to_numpy(dtype=float)
        lo, hi = bootstrap_mean_ci(x, rng)
        rows.append({
            "enso_phase": phase,
            "cluster": int(cluster),
            "n_basins": len(x),
            "basin_balanced_mean_occupancy": float(np.mean(x)),
            "median_occupancy": float(np.median(x)),
            "bootstrap_ci95_low": lo,
            "bootstrap_ci95_high": hi,
        })

    return pd.DataFrame(rows)


def occupancy_shifts_vs_neutral(occ):
    rng = np.random.default_rng(RANDOM_STATE + 1)
    rows = []

    for cluster in range(FINAL_K):
        sub = occ[occ["cluster"] == cluster].pivot(
            index="basin_id",
            columns="enso_phase_end",
            values="occupancy_fraction",
        )

        if "Neutral" not in sub.columns:
            continue

        for phase in ["El_Nino", "La_Nina"]:
            if phase not in sub.columns:
                continue

            pair = sub[[phase, "Neutral"]].dropna()
            if pair.empty:
                continue

            delta = (pair[phase] - pair["Neutral"]).to_numpy(dtype=float)
            lo, hi = bootstrap_mean_ci(delta, rng)

            rows.append({
                "enso_phase": phase,
                "cluster": cluster,
                "n_paired_basins": len(delta),
                "mean_occupancy_shift_vs_neutral": float(np.mean(delta)),
                "median_occupancy_shift_vs_neutral": float(np.median(delta)),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "fraction_basins_positive_shift": float(np.mean(delta > 0)),
                "fraction_basins_negative_shift": float(np.mean(delta < 0)),
            })

    return pd.DataFrame(rows)


def prepare_scaled_embeddings(emb, zcols):
    X = emb[zcols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    out = emb[["sample_id"]].copy()
    for j in range(Xs.shape[1]):
        out[f"zs{j+1}"] = Xs[:, j]
    return out


def build_adjacent_transition_table(valid, emb_scaled):
    zcols = [c for c in emb_scaled.columns if c.startswith("zs")]
    d = valid.merge(
        emb_scaled,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    d = d.sort_values(["basin_id", "start_time"]).copy()

    month_serial = d["start_time"].dt.year * 12 + d["start_time"].dt.month
    d["_month_serial"] = month_serial
    d["_prev_month_serial"] = d.groupby("basin_id")["_month_serial"].shift(1)
    d["_prev_cluster"] = d.groupby("basin_id")["cluster"].shift(1)

    for z in zcols:
        d[f"_prev_{z}"] = d.groupby("basin_id")[z].shift(1)

    d["delta_start_months"] = d["_month_serial"] - d["_prev_month_serial"]
    t = d[d["delta_start_months"] == 1].copy()

    t["regime_changed"] = (t["cluster"] != t["_prev_cluster"]).astype(float)

    step2 = np.zeros(len(t), dtype=float)
    for z in zcols:
        diff = t[z].to_numpy(dtype=float) - t[f"_prev_{z}"].to_numpy(dtype=float)
        step2 += diff * diff
    t["latent_step_distance_standardized"] = np.sqrt(step2)

    keep = [
        "sample_id", "basin_id", "start_time", "end_time",
        "cluster", "enso_phase_end", "oni_end",
        "regime_changed", "latent_step_distance_standardized"
    ]
    return t[keep]


def basin_phase_transition_metrics(t):
    t = t[t["enso_phase_end"].isin(PHASE_ORDER)].copy()

    agg = (
        t.groupby(["basin_id", "enso_phase_end"])
        .agg(
            n_adjacent_pairs=("regime_changed", "size"),
            transition_rate=("regime_changed", "mean"),
            mean_latent_step=("latent_step_distance_standardized", "mean"),
            p95_latent_step=(
                "latent_step_distance_standardized",
                lambda x: float(np.quantile(x, 0.95))
            ),
        )
        .reset_index()
    )

    return agg[agg["n_adjacent_pairs"] >= MIN_WINDOWS_PER_BASIN_PHASE].copy()


def summarize_transition_metrics(bpt):
    rng = np.random.default_rng(RANDOM_STATE + 2)
    rows = []

    for phase, g in bpt.groupby("enso_phase_end"):
        for metric in ["transition_rate", "mean_latent_step", "p95_latent_step"]:
            x = g[metric].dropna().to_numpy(dtype=float)
            lo, hi = bootstrap_mean_ci(x, rng)
            rows.append({
                "enso_phase": phase,
                "metric": metric,
                "n_basins": len(x),
                "basin_balanced_mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
            })

    return pd.DataFrame(rows)


def event_summary(valid):
    if not os.path.exists(EPISODES_FILE):
        return pd.DataFrame()

    ep = pd.read_csv(
        EPISODES_FILE,
        parse_dates=["start_center_month", "end_center_month"],
    )
    ep = ep[ep["end_center_month"] >= pd.Timestamp("2002-01-01")].copy()

    rows = []
    for _, e in ep.iterrows():
        sub = valid[
            (valid["end_time"] >= e["start_center_month"])
            & (valid["end_time"] <= e["end_center_month"])
        ].copy()

        if sub.empty:
            continue

        frac = sub["cluster"].value_counts(normalize=True)
        row = {
            "episode_id": int(e["episode_id"]),
            "phase": e["phase"],
            "start_center_month": e["start_center_month"],
            "end_center_month": e["end_center_month"],
            "n_windows": len(sub),
            "n_basins": sub["basin_id"].nunique(),
            "mean_oni_end": sub["oni_end"].mean(),
        }
        for c in range(FINAL_K):
            row[f"regime_{c}_window_fraction"] = float(frac.get(c, 0.0))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_occupancy(summary):
    s = summary.copy()
    x = np.arange(len(PHASE_ORDER))
    width = 0.24

    plt.figure(figsize=(9, 5))
    for c in range(FINAL_K):
        sub = (
            s[s["cluster"] == c]
            .set_index("enso_phase")
            .reindex(PHASE_ORDER)
        )
        y = sub["basin_balanced_mean_occupancy"].to_numpy()
        plt.bar(x + (c - 1) * width, y, width, label=f"Regime {c}")

    plt.xticks(x, ["La Niña", "Neutral", "El Niño"])
    plt.ylabel("Basin-balanced regime occupancy")
    plt.xlabel("ENSO phase at 24-month window endpoint")
    plt.title("Final k=3 Regime Occupancy by ENSO Phase")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "regime_occupancy_by_enso_phase.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_occupancy_shift_heatmap(delta):
    phases = ["La_Nina", "El_Nino"]
    mat = np.full((FINAL_K, len(phases)), np.nan)

    for i, c in enumerate(range(FINAL_K)):
        for j, phase in enumerate(phases):
            q = delta[
                (delta["cluster"] == c) & (delta["enso_phase"] == phase)
            ]
            if not q.empty:
                mat[i, j] = q["mean_occupancy_shift_vs_neutral"].iloc[0]

    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        mat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(im, label="Mean occupancy shift vs Neutral")
    plt.xticks([0, 1], ["La Niña", "El Niño"])
    plt.yticks(np.arange(FINAL_K), [f"Regime {c}" for c in range(FINAL_K)])
    plt.title("Basin-Paired ENSO Regime Occupancy Shifts")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "enso_occupancy_shifts_vs_neutral.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_transition_summary(summary):
    s = summary[summary["metric"] == "mean_latent_step"].copy()
    s = s.set_index("enso_phase").reindex(PHASE_ORDER)

    x = np.arange(len(PHASE_ORDER))
    mean = s["basin_balanced_mean"].to_numpy()
    low = mean - s["bootstrap_ci95_low"].to_numpy()
    high = s["bootstrap_ci95_high"].to_numpy() - mean

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x,
        mean,
        yerr=np.vstack([low, high]),
        fmt="o",
        capsize=4,
    )
    plt.xticks(x, ["La Niña", "Neutral", "El Niño"])
    plt.ylabel("Mean standardized 16-D latent step")
    plt.xlabel("ENSO phase at destination-window endpoint")
    plt.title("Hydrological State-Space Motion by ENSO Phase")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "latent_state_motion_by_enso_phase.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def print_summary(occ_summary, delta, trans_summary, coverage, events):
    print("\n" + "=" * 78)
    print("ENSO VALIDATION SUMMARY — FINAL k=3 REGIMES")
    print("=" * 78)

    print("\nBasin-balanced regime occupancy by ENSO endpoint phase:")
    show = occ_summary[
        [
            "enso_phase", "cluster", "n_basins",
            "basin_balanced_mean_occupancy",
            "bootstrap_ci95_low", "bootstrap_ci95_high",
        ]
    ].copy()
    show["enso_phase"] = pd.Categorical(
        show["enso_phase"], categories=PHASE_ORDER, ordered=True
    )
    show = show.sort_values(["enso_phase", "cluster"])
    print(show.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nPaired basin-level occupancy shifts relative to ENSO-neutral:")
    show2 = delta[
        [
            "enso_phase", "cluster", "n_paired_basins",
            "mean_occupancy_shift_vs_neutral",
            "bootstrap_ci95_low", "bootstrap_ci95_high",
            "fraction_basins_positive_shift",
            "fraction_basins_negative_shift",
        ]
    ].sort_values(["enso_phase", "cluster"])
    print(show2.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nBasin-balanced state-motion / transition metrics:")
    show3 = trans_summary[
        trans_summary["metric"].isin(["transition_rate", "mean_latent_step"])
    ][
        [
            "enso_phase", "metric", "n_basins",
            "basin_balanced_mean",
            "bootstrap_ci95_low", "bootstrap_ci95_high",
        ]
    ].copy()
    show3["enso_phase"] = pd.Categorical(
        show3["enso_phase"], categories=PHASE_ORDER, ordered=True
    )
    show3 = show3.sort_values(["metric", "enso_phase"])
    print(show3.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nCoverage:")
    print(
        coverage[
            ["period", "n_windows", "n_validated", "validation_fraction",
             "last_validated_window_end"]
        ].to_string(index=False)
    )

    if not events.empty:
        print("\nENSO episode regime fractions (descriptive, window-weighted):")
        cols = [
            "episode_id", "phase", "start_center_month", "end_center_month",
            "regime_0_window_fraction", "regime_1_window_fraction",
            "regime_2_window_fraction",
        ]
        print(events[cols].to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\nInterpretation guardrails:")
    print("- ENSO is an external validation index, not an autoencoder input.")
    print("- Global ENSO teleconnections are spatially heterogeneous; cancellation is expected.")
    print("- Main occupancy comparisons are paired within basin relative to Neutral.")
    print("- Adjacent 24-month windows overlap by 23 months; transition rates are comparative diagnostics,")
    print("  not independent Markov transition probabilities.")
    print("- Use the continuous latent-step result alongside discrete regime-switching results.")


def main():
    print("--- Validation B2: ENSO association of FINAL k=3 regimes ---")
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ENRICHED_OUTPUT), exist_ok=True)

    clusters, emb, oni, zcols = load_inputs()
    oni_metrics = build_oni_window_metrics(oni)
    enriched = attach_oni(clusters, oni_metrics)
    coverage = coverage_table(enriched)
    valid = enriched[enriched["has_enso_validation"]].copy()

    occ = build_basin_phase_occupancy(valid)
    occ_summary = summarize_occupancy(occ)
    occ_delta = occupancy_shifts_vs_neutral(occ)

    emb_scaled = prepare_scaled_embeddings(emb, zcols)
    transitions = build_adjacent_transition_table(valid, emb_scaled)
    bpt = basin_phase_transition_metrics(transitions)
    trans_summary = summarize_transition_metrics(bpt)

    events = event_summary(valid)

    enriched.to_parquet(ENRICHED_OUTPUT, index=False)
    occ.to_csv(OCCUPANCY_BASIN_OUTPUT, index=False)
    occ_summary.to_csv(OCCUPANCY_SUMMARY_OUTPUT, index=False)
    occ_delta.to_csv(OCCUPANCY_DELTA_OUTPUT, index=False)
    bpt.to_csv(TRANSITION_BASIN_OUTPUT, index=False)
    trans_summary.to_csv(TRANSITION_SUMMARY_OUTPUT, index=False)
    events.to_csv(EVENT_OUTPUT, index=False)
    coverage.to_csv(COVERAGE_OUTPUT, index=False)

    print(f"✅ Saved: {ENRICHED_OUTPUT}")
    print(f"✅ Saved: {OCCUPANCY_BASIN_OUTPUT}")
    print(f"✅ Saved: {OCCUPANCY_SUMMARY_OUTPUT}")
    print(f"✅ Saved: {OCCUPANCY_DELTA_OUTPUT}")
    print(f"✅ Saved: {TRANSITION_BASIN_OUTPUT}")
    print(f"✅ Saved: {TRANSITION_SUMMARY_OUTPUT}")
    print(f"✅ Saved: {EVENT_OUTPUT}")
    print(f"✅ Saved: {COVERAGE_OUTPUT}")

    plot_occupancy(occ_summary)
    plot_occupancy_shift_heatmap(occ_delta)
    plot_transition_summary(trans_summary)

    print_summary(occ_summary, occ_delta, trans_summary, coverage, events)
    print("--- Done ---")


if __name__ == "__main__":
    main()

# compareStaticKoppenDynamicModes.py
#
# Formal STATIC-vs-DYNAMIC basin comparison.
#
# STATIC:
#   Köppen–Geiger 1991–2020 basin composition, preserving full class fractions.
#
# DYNAMIC:
#   FINAL production raw k=3 rolling-window modes (M0/M1/M2).
#
# Scientific principle:
#   Do NOT assume that a Köppen climate class maps one-to-one onto a dynamic
#   mode.  Köppen describes background climate; M0/M1/M2 describe recurrent
#   24-month coupled climate–storage evolution.
#
# The analysis therefore asks:
#   1) Are statically homogeneous basins also dynamically persistent?
#   2) Does static climatic heterogeneity co-vary with dynamic mode diversity?
#   3) Within each Köppen class, what dynamic modes are most/least prevalent?
#   4) Which basins are strong examples of static simplicity but dynamic
#      complexity (and vice versa)?
#
# No arbitrary "mixed/homogeneous" threshold is introduced.
# The existing >=90% Köppen valid-coverage QC is used for the main analysis.

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

STATIC_SUMMARY = Path(
    "data/processed/koppen_basin_static_composition_summary_labeled.csv"
)
STATIC_LONG = Path(
    "data/processed/koppen_basin_class_fractions_long_labeled.csv"
)
FINAL_CLUSTERS = Path("data/processed/window_clusters.parquet")

OUT_DIR = Path("results/tables/static_dynamic_koppen")
FIG_DIR = Path("results/figures/static_dynamic_koppen")

BASIN_JOIN_OUT = OUT_DIR / "static_dynamic_basin_join.csv"
DYNAMIC_SUMMARY_OUT = OUT_DIR / "final_k3_dynamic_basin_summary.csv"
CORR_OUT = OUT_DIR / "static_dynamic_spearman_correlations.csv"
KG_WEIGHTED_OUT = OUT_DIR / "kg_fraction_weighted_dynamic_profiles.csv"
KG_DOMINANT_OUT = OUT_DIR / "kg_dominant_class_dynamic_profiles.csv"
EXEMPLARS_OUT = OUT_DIR / "static_dynamic_exemplar_rankings.csv"
QC_OUT = OUT_DIR / "static_dynamic_join_qc.csv"

BOOTSTRAP_REPEATS = 1000
RANDOM_SEED = 42

MODE_LABELS = {
    0: "M0",
    1: "M1",
    2: "M2",
}


# =============================================================================
# HELPERS
# =============================================================================

def month_diff(a, b):
    a = pd.Timestamp(a)
    b = pd.Timestamp(b)
    return (b.year - a.year) * 12 + (b.month - a.month)


def normalized_entropy(probs, k):
    p = np.asarray(probs, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return np.nan, np.nan
    h = float(-np.sum(p * np.log(p)))
    hnorm = h / math.log(k) if k > 1 else 0.0
    return h, hnorm


def detect_basin_column(df):
    for c in ["basin_id", "basin", "HYBAS_ID"]:
        if c in df.columns:
            return c
    raise ValueError("Could not detect basin identifier column.")


def spearman_corr(x, y):
    """
    Spearman rho without requiring scipy:
    Pearson correlation of average ranks.
    """
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    ok = x.notna() & y.notna()

    if ok.sum() < 3:
        return np.nan

    xr = x[ok].rank(method="average")
    yr = y[ok].rank(method="average")

    if xr.std(ddof=0) == 0 or yr.std(ddof=0) == 0:
        return np.nan

    return float(np.corrcoef(xr.to_numpy(), yr.to_numpy())[0, 1])


def bootstrap_spearman_ci(df, xcol, ycol, n_boot=BOOTSTRAP_REPEATS):
    d = df[[xcol, ycol]].dropna().reset_index(drop=True)
    n = len(d)

    if n < 10:
        return np.nan, np.nan, np.nan, n

    rho = spearman_corr(d[xcol], d[ycol])

    rng = np.random.default_rng(RANDOM_SEED)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = d.iloc[idx]
        r = spearman_corr(sample[xcol], sample[ycol])
        if np.isfinite(r):
            vals.append(r)

    if not vals:
        return rho, np.nan, np.nan, n

    low, high = np.quantile(vals, [0.025, 0.975])
    return rho, float(low), float(high), n


# =============================================================================
# DYNAMIC SUMMARY FROM FINAL PRODUCTION k=3
# =============================================================================

def load_final_clusters():
    d = pd.read_parquet(FINAL_CLUSTERS).copy()

    required = {"cluster", "start_time", "end_time"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"Final cluster file missing required columns: {sorted(missing)}"
        )

    if "k" in d.columns:
        kvals = sorted(d["k"].dropna().unique().tolist())
        if kvals != [3]:
            raise ValueError(
                f"Expected FINAL production k=3 only; found k={kvals}"
            )

    bcol = detect_basin_column(d)
    d["basin_id"] = pd.to_numeric(d[bcol], errors="raise").astype("int64")
    d["cluster"] = pd.to_numeric(d["cluster"], errors="raise").astype(int)
    d["start_time"] = pd.to_datetime(d["start_time"])
    d["end_time"] = pd.to_datetime(d["end_time"])

    bad_modes = sorted(set(d["cluster"]) - {0, 1, 2})
    if bad_modes:
        raise ValueError(f"Unexpected production cluster IDs: {bad_modes}")

    if "sample_id" in d.columns and d["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in final production file.")

    print(f"✅ Final k=3 windows: {len(d):,}")
    print(f"✅ Dynamic basins: {d['basin_id'].nunique():,}")
    print(
        f"✅ Window period: {d['start_time'].min().date()} "
        f"→ {d['end_time'].max().date()}"
    )

    return d


def summarize_one_basin(g):
    g = g.sort_values("start_time").reset_index(drop=True)
    n = len(g)

    counts = g["cluster"].value_counts().reindex([0, 1, 2], fill_value=0)
    p = counts.to_numpy(dtype=float) / n

    h, hnorm = normalized_entropy(p, 3)
    effective = float(np.exp(h)) if np.isfinite(h) else np.nan

    order = np.argsort(-p)
    dominant = int(order[0])
    second = int(order[1])
    dom_frac = float(p[dominant])
    second_frac = float(p[second])
    margin = dom_frac - second_frac

    # Consecutive monthly rolling-window pairs only.
    adjacent_pairs = 0
    transitions = 0

    # Longest run measured in consecutive rolling-window starts.
    longest_run = 1 if n else 0
    current_run = 1 if n else 0
    current_mode = int(g.loc[0, "cluster"]) if n else None

    # Segment count records breaks in continuous monthly trajectory.
    continuous_segments = 1 if n else 0

    for i in range(1, n):
        adjacent = month_diff(g.loc[i - 1, "start_time"], g.loc[i, "start_time"]) == 1

        if not adjacent:
            continuous_segments += 1
            current_run = 1
            current_mode = int(g.loc[i, "cluster"])
            longest_run = max(longest_run, current_run)
            continue

        adjacent_pairs += 1

        mode = int(g.loc[i, "cluster"])
        prev = int(g.loc[i - 1, "cluster"])

        if mode != prev:
            transitions += 1

        if mode == current_mode:
            current_run += 1
        else:
            current_mode = mode
            current_run = 1

        longest_run = max(longest_run, current_run)

    transition_rate = (
        transitions / adjacent_pairs if adjacent_pairs > 0 else np.nan
    )

    latest_mode = int(g.iloc[-1]["cluster"])

    return pd.Series({
        "n_windows": n,
        "M0_occupancy": float(p[0]),
        "M1_occupancy": float(p[1]),
        "M2_occupancy": float(p[2]),
        "dynamic_dominant_mode": dominant,
        "dynamic_dominant_mode_label": MODE_LABELS[dominant],
        "dynamic_dominant_occupancy": dom_frac,
        "dynamic_second_mode": second,
        "dynamic_second_occupancy": second_frac,
        "dynamic_dominance_margin": margin,
        "dynamic_entropy": h,
        "dynamic_entropy_norm3": hnorm,
        "dynamic_effective_modes": effective,
        "adjacent_month_pairs": adjacent_pairs,
        "adjacent_month_transitions": transitions,
        "transition_rate": transition_rate,
        "longest_consecutive_mode_run_windows": longest_run,
        "continuous_trajectory_segments": continuous_segments,
        "latest_mode": latest_mode,
        "latest_mode_label": MODE_LABELS[latest_mode],
        "first_window_start": g["start_time"].min(),
        "last_window_end": g["end_time"].max(),
    })


def build_dynamic_summary(clusters):
    out = (
        clusters.groupby("basin_id", group_keys=False)
        .apply(summarize_one_basin, include_groups=False)
        .reset_index()
    )
    out["basin_id"] = out["basin_id"].astype("int64")
    return out


# =============================================================================
# STATIC JOIN
# =============================================================================

def load_static():
    s = pd.read_csv(STATIC_SUMMARY)
    l = pd.read_csv(STATIC_LONG)

    if "HYBAS_ID" not in s.columns or "HYBAS_ID" not in l.columns:
        raise ValueError("Both Köppen files must contain HYBAS_ID.")

    s["HYBAS_ID"] = pd.to_numeric(s["HYBAS_ID"], errors="raise").astype("int64")
    l["HYBAS_ID"] = pd.to_numeric(l["HYBAS_ID"], errors="raise").astype("int64")

    if s["HYBAS_ID"].duplicated().any():
        raise ValueError("Static summary contains duplicate HYBAS_ID rows.")

    required_summary = {
        "KG_dom_code",
        "KG_dom_frac",
        "KG_dom_abbr",
        "KG_second_code",
        "KG_second_frac",
        "KG_second_abbr",
        "KG_entropy_norm30",
        "KG_effective_classes",
        "KG_valid_coverage",
        "KG_qc_pass_90pct_valid",
    }
    missing = required_summary - set(s.columns)
    if missing:
        raise ValueError(
            f"Labeled static summary missing: {sorted(missing)}"
        )

    required_long = {"KG_code", "KG_abbr", "KG_name", "KG_fraction"}
    missing = required_long - set(l.columns)
    if missing:
        raise ValueError(
            f"Labeled static long table missing: {sorted(missing)}"
        )

    print(f"✅ Köppen summary basins: {len(s):,}")
    print(f"✅ Köppen long rows: {len(l):,}")
    print(
        f"✅ Static QC-pass basins (>=90% valid): "
        f"{int(s['KG_qc_pass_90pct_valid'].astype(bool).sum()):,}"
    )

    return s, l


def join_static_dynamic(static, dynamic):
    out = static.merge(
        dynamic,
        left_on="HYBAS_ID",
        right_on="basin_id",
        how="left",
        validate="one_to_one",
    )

    out["has_dynamic_profile"] = out["basin_id"].notna()

    # Continuous static/dynamic descriptors useful for interpretation.
    out["static_non_dominant_fraction"] = 1.0 - out["KG_dom_frac"]
    out["dynamic_non_dominant_fraction"] = (
        1.0 - out["dynamic_dominant_occupancy"]
    )

    # Heuristic exemplar scores — NOT scientific classes.
    out["score_static_simple_dynamic_diverse"] = (
        out["KG_dom_frac"] * out["dynamic_entropy_norm3"]
    )
    out["score_static_mixed_dynamic_persistent"] = (
        out["KG_entropy_norm30"]
        * (1.0 - out["dynamic_entropy_norm3"])
    )

    return out


# =============================================================================
# MAIN STATIC-vs-DYNAMIC STATISTICS
# =============================================================================

def correlation_table(main):
    pairs = [
        (
            "KG_dom_frac",
            "dynamic_dominant_occupancy",
            "static dominance vs dynamic dominance",
        ),
        (
            "KG_dom_frac",
            "dynamic_entropy_norm3",
            "static dominance vs dynamic diversity",
        ),
        (
            "KG_dom_frac",
            "transition_rate",
            "static dominance vs transition rate",
        ),
        (
            "KG_entropy_norm30",
            "dynamic_entropy_norm3",
            "static climate entropy vs dynamic mode entropy",
        ),
        (
            "KG_entropy_norm30",
            "dynamic_effective_modes",
            "static climate entropy vs effective dynamic modes",
        ),
        (
            "KG_entropy_norm30",
            "transition_rate",
            "static climate entropy vs transition rate",
        ),
        (
            "KG_effective_classes",
            "dynamic_effective_modes",
            "effective static classes vs effective dynamic modes",
        ),
    ]

    rows = []
    for x, y, label in pairs:
        rho, low, high, n = bootstrap_spearman_ci(main, x, y)
        rows.append({
            "comparison": label,
            "x": x,
            "y": y,
            "n_basins": n,
            "spearman_rho": rho,
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
        })

    return pd.DataFrame(rows)


def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.average(v[ok], weights=w[ok]))


def kg_fraction_weighted_profiles(long_df, basin_join):
    """
    Uses the FULL fractional Köppen composition.

    A basin that is 52% Dfb and 48% Dfc contributes 0.52 basin-weight to the
    Dfb profile and 0.48 basin-weight to Dfc, rather than being forced wholly
    into one class.
    """
    dyncols = [
        "HYBAS_ID",
        "KG_qc_pass_90pct_valid",
        "M0_occupancy",
        "M1_occupancy",
        "M2_occupancy",
        "dynamic_dominant_occupancy",
        "dynamic_entropy_norm3",
        "dynamic_effective_modes",
        "transition_rate",
    ]

    d = long_df.merge(
        basin_join[dyncols],
        on="HYBAS_ID",
        how="inner",
        validate="many_to_one",
    )

    d = d[
        d["KG_qc_pass_90pct_valid"].astype(bool)
        & d["M0_occupancy"].notna()
    ].copy()

    rows = []
    for (code, abbr, name), g in d.groupby(
        ["KG_code", "KG_abbr", "KG_name"],
        observed=True,
    ):
        w = g["KG_fraction"].to_numpy(dtype=float)

        rows.append({
            "KG_code": int(code),
            "KG_abbr": abbr,
            "KG_name": name,
            "n_contributing_basins": int(g["HYBAS_ID"].nunique()),
            # Because each basin's KG fractions sum to one, this is an
            # "equivalent basin count" under fractional membership.
            "fractional_basin_weight_sum": float(np.sum(w)),
            "weighted_M0_occupancy": weighted_mean(g["M0_occupancy"], w),
            "weighted_M1_occupancy": weighted_mean(g["M1_occupancy"], w),
            "weighted_M2_occupancy": weighted_mean(g["M2_occupancy"], w),
            "weighted_dynamic_dominant_occupancy": weighted_mean(
                g["dynamic_dominant_occupancy"], w
            ),
            "weighted_dynamic_entropy_norm3": weighted_mean(
                g["dynamic_entropy_norm3"], w
            ),
            "weighted_transition_rate": weighted_mean(
                g["transition_rate"], w
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("KG_code")
        .reset_index(drop=True)
    )


def dominant_kg_profiles(main):
    """
    Descriptive grouping by dominant KG class only.
    Unlike the fractional profile, this is NOT used to erase mixed composition.
    KG_dom_frac remains available alongside the group summaries.
    """
    metrics = [
        "M0_occupancy",
        "M1_occupancy",
        "M2_occupancy",
        "dynamic_dominant_occupancy",
        "dynamic_entropy_norm3",
        "dynamic_effective_modes",
        "transition_rate",
    ]

    rows = []
    for (code, abbr, name), g in main.groupby(
        ["KG_dom_code", "KG_dom_abbr", "KG_dom_name"],
        observed=True,
    ):
        row = {
            "KG_dom_code": int(code),
            "KG_dom_abbr": abbr,
            "KG_dom_name": name,
            "n_basins": len(g),
            "mean_KG_dom_frac": float(g["KG_dom_frac"].mean()),
            "median_KG_dom_frac": float(g["KG_dom_frac"].median()),
        }
        for m in metrics:
            row[f"mean_{m}"] = float(g[m].mean())
            row[f"median_{m}"] = float(g[m].median())
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("KG_dom_code")
        .reset_index(drop=True)
    )


def exemplar_rankings(main):
    cols = [
        "HYBAS_ID",
        "KG_dom_code",
        "KG_dom_abbr",
        "KG_dom_name",
        "KG_dom_frac",
        "KG_second_abbr",
        "KG_second_frac",
        "KG_entropy_norm30",
        "KG_effective_classes",
        "M0_occupancy",
        "M1_occupancy",
        "M2_occupancy",
        "dynamic_dominant_mode_label",
        "dynamic_dominant_occupancy",
        "dynamic_entropy_norm3",
        "transition_rate",
        "longest_consecutive_mode_run_windows",
        "score_static_simple_dynamic_diverse",
        "score_static_mixed_dynamic_persistent",
    ]

    frames = []

    a = main.sort_values(
        "score_static_simple_dynamic_diverse",
        ascending=False,
    ).head(50)[cols].copy()
    a.insert(0, "ranking_type", "static_simple_dynamic_diverse")
    a.insert(1, "rank", np.arange(1, len(a) + 1))
    frames.append(a)

    b = main.sort_values(
        "score_static_mixed_dynamic_persistent",
        ascending=False,
    ).head(50)[cols].copy()
    b.insert(0, "ranking_type", "static_mixed_dynamic_persistent")
    b.insert(1, "rank", np.arange(1, len(b) + 1))
    frames.append(b)

    c = main.sort_values(
        "transition_rate",
        ascending=False,
    ).head(50)[cols].copy()
    c.insert(0, "ranking_type", "highest_transition_rate")
    c.insert(1, "rank", np.arange(1, len(c) + 1))
    frames.append(c)

    return pd.concat(frames, ignore_index=True)


# =============================================================================
# FIGURES
# =============================================================================

def plot_static_dynamic_entropy(main):
    plt.figure(figsize=(8.5, 6))
    plt.scatter(
        main["KG_entropy_norm30"],
        main["dynamic_entropy_norm3"],
        s=18,
        alpha=0.55,
    )
    plt.xlabel("Static Köppen composition entropy (normalized)")
    plt.ylabel("Dynamic M0/M1/M2 occupancy entropy (normalized)")
    plt.title("Static climate heterogeneity vs dynamic mode diversity")
    plt.tight_layout()

    out = FIG_DIR / "static_entropy_vs_dynamic_entropy.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_static_dominance_dynamic_dominance(main):
    plt.figure(figsize=(8.5, 6))
    plt.scatter(
        main["KG_dom_frac"],
        main["dynamic_dominant_occupancy"],
        s=18,
        alpha=0.55,
    )
    plt.xlabel("Dominant Köppen class fraction")
    plt.ylabel("Dominant dynamic-mode occupancy")
    plt.title("Static climate dominance vs dynamic-state persistence")
    plt.tight_layout()

    out = FIG_DIR / "static_dominance_vs_dynamic_dominance.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


def plot_fraction_weighted_mode_heatmap(kg):
    """
    Matplotlib-only heatmap.  We intentionally show only classes with enough
    fractional basin support for a readable descriptive figure.  This display
    filter does NOT alter the saved full table.
    """
    show = kg[kg["fractional_basin_weight_sum"] >= 5].copy()
    if show.empty:
        return

    arr = show[
        [
            "weighted_M0_occupancy",
            "weighted_M1_occupancy",
            "weighted_M2_occupancy",
        ]
    ].to_numpy(dtype=float)

    fig_h = max(6, 0.35 * len(show))
    plt.figure(figsize=(7.5, fig_h))
    im = plt.imshow(arr, aspect="auto", vmin=0, vmax=1)

    plt.colorbar(im, label="Fractionally weighted mean occupancy")
    plt.xticks([0, 1, 2], ["M0", "M1", "M2"])
    plt.yticks(
        np.arange(len(show)),
        [
            f"{a} ({int(n)} basins)"
            for a, n in zip(
                show["KG_abbr"],
                show["n_contributing_basins"],
            )
        ],
    )
    plt.xlabel("Dynamic mode")
    plt.ylabel("Köppen–Geiger class")
    plt.title(
        "Dynamic-mode occupancy by fractional Köppen climate membership"
    )
    plt.tight_layout()

    out = FIG_DIR / "kg_fraction_weighted_dynamic_mode_heatmap.png"
    plt.savefig(out, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {out}")


# =============================================================================
# QC + CONSOLE
# =============================================================================

def build_qc(static, dynamic, joined, main):
    rows = [
        {
            "check": "static_summary_rows",
            "value": len(static),
        },
        {
            "check": "dynamic_profile_basins",
            "value": len(dynamic),
        },
        {
            "check": "static_rows_with_dynamic_match",
            "value": int(joined["has_dynamic_profile"].sum()),
        },
        {
            "check": "main_qc_pass_joined_basins",
            "value": len(main),
        },
        {
            "check": "main_missing_dynamic_entropy",
            "value": int(main["dynamic_entropy_norm3"].isna().sum()),
        },
        {
            "check": "main_missing_transition_rate",
            "value": int(main["transition_rate"].isna().sum()),
        },
    ]
    return pd.DataFrame(rows)


def print_summary(joined, main, corr, kg_weighted, exemplars):
    print("\n" + "=" * 92)
    print("STATIC-vs-DYNAMIC KÖPPEN–GEIGER COMPARISON")
    print("=" * 92)

    print("\nA) Coverage")
    print(f"Static basin rows:                {len(joined):,}")
    print(
        f"Static basins matched dynamically: "
        f"{int(joined['has_dynamic_profile'].sum()):,}"
    )
    print(
        f"Main analysis basins "
        f"(>=90% valid KG + dynamic):       {len(main):,}"
    )

    print("\nB) Dynamic summary — main analysis basins")
    cols = [
        "dynamic_dominant_occupancy",
        "dynamic_entropy_norm3",
        "transition_rate",
        "longest_consecutive_mode_run_windows",
    ]
    print(
        main[cols]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .to_string(float_format=lambda x: f"{x: .3f}")
    )

    print("\nC) Static-vs-dynamic Spearman correlations")
    print(
        corr.to_string(
            index=False,
            float_format=lambda x: f"{x: .3f}",
        )
    )

    print("\nD) Fractionally weighted Köppen → dynamic-mode profiles")
    show = kg_weighted[
        kg_weighted["fractional_basin_weight_sum"] >= 5
    ][
        [
            "KG_code",
            "KG_abbr",
            "n_contributing_basins",
            "fractional_basin_weight_sum",
            "weighted_M0_occupancy",
            "weighted_M1_occupancy",
            "weighted_M2_occupancy",
            "weighted_dynamic_entropy_norm3",
            "weighted_transition_rate",
        ]
    ]
    print(
        show.to_string(
            index=False,
            float_format=lambda x: f"{x: .3f}",
        )
    )

    print("\nE) Top 10 'static-simple but dynamically diverse' exemplar candidates")
    top = exemplars[
        exemplars["ranking_type"] == "static_simple_dynamic_diverse"
    ].head(10)
    print(
        top[
            [
                "rank",
                "HYBAS_ID",
                "KG_dom_abbr",
                "KG_dom_frac",
                "dynamic_dominant_mode_label",
                "dynamic_dominant_occupancy",
                "dynamic_entropy_norm3",
                "transition_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x: .3f}",
        )
    )

    print("\nInterpretation guardrails:")
    print("- Köppen classes are static climate-background descriptors.")
    print("- M0/M1/M2 are recurrent 24-month coupled climate-storage modes.")
    print("- There is no assumed one-to-one KG-class ↔ dynamic-mode correspondence.")
    print("- The fractional Köppen analysis preserves mixed basins rather than")
    print("  forcing them wholly into their dominant climate class.")
    print("- Transition rates are descriptive rolling-window diagnostics, not")
    print("  independent Markov transition probabilities.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("--- Formal static-vs-dynamic Köppen comparison ---")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    clusters = load_final_clusters()
    dynamic = build_dynamic_summary(clusters)
    dynamic.to_csv(DYNAMIC_SUMMARY_OUT, index=False)
    print(f"✅ Saved: {DYNAMIC_SUMMARY_OUT}")

    static, long_df = load_static()
    joined = join_static_dynamic(static, dynamic)
    joined.to_csv(BASIN_JOIN_OUT, index=False)
    print(f"✅ Saved: {BASIN_JOIN_OUT}")

    main_df = joined[
        joined["KG_qc_pass_90pct_valid"].astype(bool)
        & joined["has_dynamic_profile"]
    ].copy()

    corr = correlation_table(main_df)
    corr.to_csv(CORR_OUT, index=False)
    print(f"✅ Saved: {CORR_OUT}")

    kg_weighted = kg_fraction_weighted_profiles(long_df, joined)
    kg_weighted.to_csv(KG_WEIGHTED_OUT, index=False)
    print(f"✅ Saved: {KG_WEIGHTED_OUT}")

    kg_dom = dominant_kg_profiles(main_df)
    kg_dom.to_csv(KG_DOMINANT_OUT, index=False)
    print(f"✅ Saved: {KG_DOMINANT_OUT}")

    exemplars = exemplar_rankings(main_df)
    exemplars.to_csv(EXEMPLARS_OUT, index=False)
    print(f"✅ Saved: {EXEMPLARS_OUT}")

    qc = build_qc(static, dynamic, joined, main_df)
    qc.to_csv(QC_OUT, index=False)
    print(f"✅ Saved: {QC_OUT}")

    plot_static_dynamic_entropy(main_df)
    plot_static_dominance_dynamic_dominance(main_df)
    plot_fraction_weighted_mode_heatmap(kg_weighted)

    print_summary(
        joined,
        main_df,
        corr,
        kg_weighted,
        exemplars,
    )

    print("\n--- Done ---")
    print(
        "STOP here and inspect this comparison before defining any "
        "'aligned/contrasting' poster categories."
    )


if __name__ == "__main__":
    main()

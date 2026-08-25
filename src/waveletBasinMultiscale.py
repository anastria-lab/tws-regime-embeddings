# waveletBasinMultiscale.py
# Step 6: Gap-aware multi-scale SWT decomposition of basin-level normalized anomalies

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning)

INPUT_FILE = "data/processed/basin_dataset_anomaly_normalized.parquet"
OUTPUT_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"

FIGURE_DIR = "results/figures/wavelet_examples"
VARIANCE_OUTPUT = "results/tables/wavelet_variance_partition.csv"
RECON_ERROR_OUTPUT = "results/tables/wavelet_reconstruction_error.csv"
CONFIG_OUTPUT = "configs/wavelet_config.yaml"
SUMMARY_OUTPUT = "results/tables/wavelet_basin_variable_summary.csv"
SUMMARY_TIDY_OUTPUT = "results/tables/wavelet_basin_variable_summary_tidy.csv"

WAVELET = "db4"
LEVEL = 5

# Short isolated gaps may be interpolated. Longer gaps (including the GRACE /
# GRACE-FO mission gap) remain missing and split the series into independent SWT
# segments. This prevents the wavelet transform from treating distant months as
# adjacent observations or inventing a long bridge across the mission gap.
MAX_INTERPOLATION_GAP_MONTHS = 2
MIN_SEGMENT_MONTHS = 24

TARGET_COLUMNS = [
    "lwe_thickness_anomaly_normalized",
    "tp_anomaly_normalized",
    "swvl1_anomaly_normalized",
]

ID_COLUMNS = ["basin", "year", "month", "time", "time_era5", "time_grace"]

EXAMPLE_BASINS = {
    "amazon": 6050298170,
    "iceland": 2050058330,
    "thailand": 4050018280,
    "egypt": 1050000010,
}


def save_wavelet_config(variables):
    config = {
        "wavelet_transform": {
            "transform_type": "swt",
            "wavelet_family": WAVELET,
            "levels": LEVEL,
            "padding_method": "reflect",
        },
        "scale_grouping": {
            "short": "D1 + D2",
            "seasonal": "D3",
            "long": "D4 + D5 + A5",
        },
        "gap_policy": {
            "canonical_frequency": "monthly",
            "max_interpolation_gap_months": MAX_INTERPOLATION_GAP_MONTHS,
            "minimum_segment_months": MIN_SEGMENT_MONTHS,
            "long_gaps": "left missing; SWT performed independently on each valid segment",
        },
        "input": {
            "file": INPUT_FILE,
            "value_type": "anomaly_normalized",
            "variables": variables,
        },
        "output": {
            "dataset": OUTPUT_FILE,
            "variance_table": VARIANCE_OUTPUT,
            "reconstruction_table": RECON_ERROR_OUTPUT,
            "summary_table": SUMMARY_OUTPUT,
            "figures": FIGURE_DIR,
        },
    }

    os.makedirs(os.path.dirname(CONFIG_OUTPUT), exist_ok=True)
    with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"✅ Saved wavelet config: {CONFIG_OUTPUT}")


def load_dataset(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded dataset: {file_path}")
        print(f"   Rows: {len(df):,}")
        return df
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None


def add_canonical_time(df):
    """Build/validate the month-start timestamp used for all SWT operations."""
    df = df.copy()
    canonical = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))

    if "time" in df.columns:
        existing = pd.to_datetime(df["time"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        mismatch = existing.notna() & (existing != canonical)
        if mismatch.any():
            raise ValueError(f"Canonical time disagrees with year/month in {int(mismatch.sum())} rows.")

    df["time"] = canonical
    return df


def validate_monthly_grid(df):
    """Require one explicit row per basin-month before wavelet decomposition."""
    dup = df.duplicated(["basin", "time"]).sum()
    if dup:
        raise ValueError(f"Found {int(dup)} duplicate basin-month rows before SWT.")

    bad = []
    for basin, group in df.groupby("basin"):
        times = group["time"].sort_values().reset_index(drop=True)
        if len(times) < 2:
            continue
        month_steps = (times.dt.year.diff() * 12 + times.dt.month.diff()).iloc[1:]
        if not (month_steps == 1).all():
            bad.append(basin)

    if bad:
        raise ValueError(
            f"Monthly calendar has missing timestamps for {len(bad)} basins before SWT. "
            "Run the fixed mergeBasinsTimeseries.py first."
        )

    print("✅ SWT input has an explicit consecutive monthly calendar for every basin")


def detect_available_target_columns(df):
    target_cols = sorted([c for c in df.columns if c.endswith("_anomaly_normalized")])
    if not target_cols:
        raise ValueError("No normalized anomaly columns found for wavelet decomposition.")

    print("✅ Wavelet target columns:")
    for c in target_cols:
        print(f"   - {c}")
    return target_cols


def pad_series_for_swt(x, level):
    """Reflect-pad one finite segment to a length divisible by 2**level."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    block = 2 ** level
    target_len = int(np.ceil(n / block) * block)
    pad_len = target_len - n
    if pad_len == 0:
        return x, 0
    return np.pad(x, (0, pad_len), mode="reflect"), pad_len


def swt_reconstruct_band(x, wavelet=WAVELET, level=LEVEL):
    """Decompose one finite monthly segment and reconstruct three scale bands."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("swt_reconstruct_band received non-finite values.")

    x_pad, pad_len = pad_series_for_swt(x, level=level)
    coeffs = pywt.swt(x_pad, wavelet=wavelet, level=level, norm=True)

    def reconstruct_component(detail_levels_to_keep=None, keep_final_approx=False):
        detail_levels_to_keep = detail_levels_to_keep or []
        coeffs_band = []

        for lev in range(level, 0, -1):
            cA, cD = coeffs[level - lev]
            keep_a = cA.copy() if (keep_final_approx and lev == level) else np.zeros_like(cA)
            keep_d = cD.copy() if lev in detail_levels_to_keep else np.zeros_like(cD)
            coeffs_band.append((keep_a, keep_d))

        return np.asarray(pywt.iswt(coeffs_band, wavelet=wavelet, norm=True))

    short_rec = reconstruct_component([1, 2], keep_final_approx=False)
    seasonal_rec = reconstruct_component([3], keep_final_approx=False)
    long_rec = reconstruct_component([4, 5], keep_final_approx=True)
    reconstructed = short_rec + seasonal_rec + long_rec

    if pad_len:
        short_rec = short_rec[:-pad_len]
        seasonal_rec = seasonal_rec[:-pad_len]
        long_rec = long_rec[:-pad_len]
        reconstructed = reconstructed[:-pad_len]

    return {
        "short": short_rec,
        "seasonal": seasonal_rec,
        "long": long_rec,
        "reconstructed": reconstructed,
    }


def missing_run_lengths(x):
    """Return (start, end, length) for consecutive NaN/non-finite runs."""
    x = np.asarray(x, dtype=float)
    missing = ~np.isfinite(x)
    runs = []
    i = 0
    while i < len(x):
        if not missing[i]:
            i += 1
            continue
        start = i
        while i < len(x) and missing[i]:
            i += 1
        end = i - 1
        runs.append((start, end, end - start + 1))
    return runs


def interpolate_only_short_internal_gaps(x, max_gap=MAX_INTERPOLATION_GAP_MONTHS):
    """Linearly fill only bounded missing runs whose full length <= max_gap.

    Long gaps and leading/trailing missing runs are deliberately left missing.
    """
    x = np.asarray(x, dtype=float).copy()
    imputed = np.zeros(len(x), dtype=bool)

    for start, end, length in missing_run_lengths(x):
        bounded = start > 0 and end < len(x) - 1
        if not bounded or length > max_gap:
            continue
        if not (np.isfinite(x[start - 1]) and np.isfinite(x[end + 1])):
            continue

        left = x[start - 1]
        right = x[end + 1]
        fill = np.linspace(left, right, length + 2)[1:-1]
        x[start:end + 1] = fill
        imputed[start:end + 1] = True

    return x, imputed


def finite_segments(x):
    """Return inclusive index ranges of consecutive finite values."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    segments = []
    i = 0
    while i < len(x):
        if not valid[i]:
            i += 1
            continue
        start = i
        while i < len(x) and valid[i]:
            i += 1
        segments.append((start, i - 1))
    return segments


def safe_nanvar(x):
    x = np.asarray(x, dtype=float)
    return np.nan if np.isfinite(x).sum() < 2 else np.nanvar(x)


def compute_variance_fractions(raw, short, seasonal, long):
    mask = np.isfinite(raw) & np.isfinite(short) & np.isfinite(seasonal) & np.isfinite(long)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan

    raw_var = np.var(raw[mask])
    if not np.isfinite(raw_var) or raw_var == 0:
        return np.nan, np.nan, np.nan

    return (
        np.var(short[mask]) / raw_var,
        np.var(seasonal[mask]) / raw_var,
        np.var(long[mask]) / raw_var,
    )


def compute_rmse(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[mask] - y[mask]) ** 2)))


def decompose_one_group(group_df, variable, wavelet=WAVELET, level=LEVEL):
    """Gap-aware SWT for one basin-variable monthly series.

    1. Keep the full calendar.
    2. Interpolate only short bounded gaps (<= configured threshold).
    3. Leave long gaps missing.
    4. Apply SWT independently to each sufficiently long finite segment.
    """
    group_df = group_df.sort_values("time").reset_index(drop=True).copy()
    x_original = group_df[variable].to_numpy(dtype=float)
    x_filled, imputed_mask = interpolate_only_short_internal_gaps(x_original)

    short = np.full(len(x_filled), np.nan, dtype=float)
    seasonal = np.full(len(x_filled), np.nan, dtype=float)
    long = np.full(len(x_filled), np.nan, dtype=float)
    reconstructed = np.full(len(x_filled), np.nan, dtype=float)

    all_segments = finite_segments(x_filled)
    used_segments = []

    for start, end in all_segments:
        segment_len = end - start + 1
        if segment_len < MIN_SEGMENT_MONTHS:
            continue

        bands = swt_reconstruct_band(x_filled[start:end + 1], wavelet=wavelet, level=level)
        short[start:end + 1] = bands["short"]
        seasonal[start:end + 1] = bands["seasonal"]
        long[start:end + 1] = bands["long"]
        reconstructed[start:end + 1] = bands["reconstructed"]
        used_segments.append((start, end))

    base = variable.replace("_anomaly_normalized", "")
    out = group_df[["basin", "year", "month"]].copy()
    out[f"{base}_short"] = short
    out[f"{base}_seasonal"] = seasonal
    out[f"{base}_long"] = long
    out[f"{base}_reconstructed"] = reconstructed

    # Compare reconstruction only with genuinely observed months, not imputed ones.
    observed_for_eval = x_original.copy()
    observed_for_eval[imputed_mask] = np.nan

    short_frac, seasonal_frac, long_frac = compute_variance_fractions(
        x_filled, short, seasonal, long
    )
    rmse = compute_rmse(observed_for_eval, reconstructed)

    original_runs = missing_run_lengths(x_original)
    unfilled_missing = ~np.isfinite(x_filled)
    used_mask = np.isfinite(reconstructed)

    diagnostics = {
        "basin": group_df["basin"].iloc[0],
        "variable": base,
        "n_calendar_months": int(len(x_original)),
        "n_observed_months": int(np.isfinite(x_original).sum()),
        "n_missing_original": int((~np.isfinite(x_original)).sum()),
        "n_short_gap_imputed": int(imputed_mask.sum()),
        "n_missing_left_unfilled": int(unfilled_missing.sum()),
        "max_original_missing_run": int(max([r[2] for r in original_runs], default=0)),
        "n_finite_segments_after_gap_policy": int(len(all_segments)),
        "n_swt_segments_used": int(len(used_segments)),
        "n_months_with_swt_output": int(used_mask.sum()),
        "min_used_segment_months": int(min([e - s + 1 for s, e in used_segments], default=0)),
        "max_used_segment_months": int(max([e - s + 1 for s, e in used_segments], default=0)),
        "short_frac": short_frac,
        "seasonal_frac": seasonal_frac,
        "long_frac": long_frac,
        "rmse": rmse,
    }

    return out, diagnostics


def run_wavelet_decomposition(df, variables):
    diagnostics = []
    per_variable_outputs = []

    for variable in variables:
        selected_cols = ["basin", "year", "month", "time", variable]
        sub = df[selected_cols].copy()

        variable_groups = []
        for basin_id, group in sub.groupby("basin"):
            group = group.sort_values("time").reset_index(drop=True)
            out_group, diag = decompose_one_group(group, variable)
            diagnostics.append(diag)
            variable_groups.append(out_group)

        if variable_groups:
            variable_df = pd.concat(variable_groups, ignore_index=True)
            if variable_df.duplicated(["basin", "year", "month"]).any():
                raise ValueError(f"Duplicate basin-month rows produced for {variable}.")
            per_variable_outputs.append(variable_df)

    if not per_variable_outputs:
        return pd.DataFrame(), pd.DataFrame(diagnostics)

    wavelet_df = per_variable_outputs[0]
    for other_df in per_variable_outputs[1:]:
        wavelet_df = wavelet_df.merge(
            other_df, on=["basin", "year", "month"], how="outer", validate="one_to_one"
        )

    return wavelet_df, pd.DataFrame(diagnostics)


def merge_wavelet_outputs_back(df, wavelet_df):
    return df.merge(
        wavelet_df,
        on=["basin", "year", "month"],
        how="left",
        validate="one_to_one",
    )


def save_tidy_summary_table(diagnostics):
    cols = [
        "basin", "variable", "n_calendar_months", "n_observed_months",
        "n_missing_original", "n_short_gap_imputed", "n_missing_left_unfilled",
        "max_original_missing_run", "n_swt_segments_used", "n_months_with_swt_output",
        "short_frac", "seasonal_frac", "long_frac", "rmse",
    ]
    tidy = diagnostics[[c for c in cols if c in diagnostics.columns]].copy()
    tidy = tidy.rename(columns={
        "short_frac": "var_short_frac",
        "seasonal_frac": "var_seasonal_frac",
        "long_frac": "var_long_frac",
        "rmse": "reconstruction_rmse_observed_months",
    })
    tidy = tidy.sort_values(["variable", "basin"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(SUMMARY_TIDY_OUTPUT), exist_ok=True)
    tidy.to_csv(SUMMARY_TIDY_OUTPUT, index=False)
    print(f"✅ Saved tidy summary table: {SUMMARY_TIDY_OUTPUT}")


def save_outputs(df, diagnostics):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(VARIANCE_OUTPUT), exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Saved wavelet dataset: {OUTPUT_FILE}")

    variance_cols = ["basin", "variable", "short_frac", "seasonal_frac", "long_frac"]
    diagnostics[variance_cols].to_csv(VARIANCE_OUTPUT, index=False)
    diagnostics[["basin", "variable", "rmse"]].to_csv(RECON_ERROR_OUTPUT, index=False)
    diagnostics.to_csv(SUMMARY_OUTPUT, index=False)

    print(f"✅ Saved variance partition: {VARIANCE_OUTPUT}")
    print(f"✅ Saved reconstruction error: {RECON_ERROR_OUTPUT}")
    print(f"✅ Saved basin-variable summary: {SUMMARY_OUTPUT}")
    save_tidy_summary_table(diagnostics)


def plot_example_decomposition(raw_df, wavelet_df, basin_id, variable, out_path):
    if basin_id is None:
        return

    norm_col = variable if variable.endswith("_anomaly_normalized") else f"{variable}_anomaly_normalized"
    base = norm_col.replace("_anomaly_normalized", "")
    short_col = f"{base}_short"
    seasonal_col = f"{base}_seasonal"
    long_col = f"{base}_long"

    required_raw = ["basin", "year", "month", "time", norm_col]
    required_wavelet = ["basin", "year", "month", short_col, seasonal_col, long_col]
    raw_missing = [c for c in required_raw if c not in raw_df.columns]
    wavelet_missing = [c for c in required_wavelet if c not in wavelet_df.columns]
    if raw_missing or wavelet_missing:
        print(f"⚠️ Skipping plot for basin {basin_id}, {base}; missing {raw_missing + wavelet_missing}")
        return

    raw_plot = raw_df.loc[raw_df["basin"] == basin_id, required_raw].copy()
    wav_plot = wavelet_df.loc[wavelet_df["basin"] == basin_id, required_wavelet].copy()
    if raw_plot.empty or wav_plot.empty:
        return

    plot_df = raw_plot.merge(wav_plot, on=["basin", "year", "month"], how="inner")
    plot_df = plot_df.sort_values("time")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(plot_df["time"], plot_df[norm_col])
    axes[0].set_title(f"Basin {basin_id} | {base} | raw normalized anomaly")
    axes[1].plot(plot_df["time"], plot_df[short_col])
    axes[1].set_title("short-term (D1 + D2)")
    axes[2].plot(plot_df["time"], plot_df[seasonal_col])
    axes[2].set_title("seasonal/annual (D3)")
    axes[3].plot(plot_df["time"], plot_df[long_col])
    axes[3].set_title("long-term (D4 + D5 + A5)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved figure: {out_path}")


def plot_required_examples(raw_df, wavelet_df):
    variable_map = {
        "GRACE TWSA": "lwe_thickness",
        "precipitation": "tp",
        "swvl1": "swvl1",
    }
    for basin_label, basin_id in EXAMPLE_BASINS.items():
        for _, variable in variable_map.items():
            out_path = os.path.join(FIGURE_DIR, f"{basin_label}_{variable}_wavelet_example.png")
            plot_example_decomposition(raw_df, wavelet_df, basin_id, variable, out_path)


def main():
    print("--- Gap-aware wavelet basin multi-scale decomposition ---")
    print(f"Wavelet: {WAVELET} | Levels: {LEVEL}")
    print("Bands: short = D1 + D2, seasonal = D3, long = D4 + D5 + A5")
    print(
        f"Gap policy: interpolate only bounded gaps <= {MAX_INTERPOLATION_GAP_MONTHS} months; "
        "leave longer gaps missing and decompose independent segments."
    )

    df = load_dataset(INPUT_FILE)
    if df is None:
        return

    df = add_canonical_time(df)
    validate_monthly_grid(df)

    variables = detect_available_target_columns(df)
    save_wavelet_config(variables)

    wavelet_df, diagnostics = run_wavelet_decomposition(df, variables)
    merged_df = merge_wavelet_outputs_back(df, wavelet_df)

    if merged_df.duplicated(["basin", "year", "month"]).any():
        raise ValueError("Duplicate basin-month rows found after wavelet merge.")

    save_outputs(merged_df, diagnostics)
    plot_required_examples(df, merged_df)

    print("\nGap / SWT diagnostics by variable:")
    summary_cols = [
        "n_missing_original", "n_short_gap_imputed", "n_missing_left_unfilled",
        "n_swt_segments_used", "n_months_with_swt_output", "rmse",
    ]
    print(diagnostics.groupby("variable")[summary_cols].mean().round(3).to_string())
    print("--- Done ---")


if __name__ == "__main__":
    main()

# waveletBasinMultiscale.py
# Step 6: Multi-scale wavelet decomposition of basin-level anomaly-normalized signals

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import yaml
import warnings
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

# Expected normalized anomaly inputs
TARGET_COLUMNS = [
    "lwe_thickness_anomaly_normalized",
    "tp_anomaly_normalized",
    "swvl1_anomaly_normalized",
]

ID_COLUMNS = ["basin", "year", "month", "time_era5", "time_grace"]

# Example basins to plot. Replace the placeholder IDs below if you know them.
EXAMPLE_BASINS = {
    "amazon": 6050298170,
    "iceland": 2050058330,
    "thailand": 4050018280,
    "egypt": 1050000010,
}

def save_wavelet_config(variables):
    """
    Save wavelet configuration for reproducibility.
    """
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

    try:
        os.makedirs(os.path.dirname(CONFIG_OUTPUT), exist_ok=True)
        with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        print(f"✅ Saved wavelet config: {CONFIG_OUTPUT}")
    except Exception as e:
        print(f"❌ Failed to save wavelet config: {e}")

def load_dataset(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded dataset: {file_path}")
        print(f"   Rows: {len(df):,}")
        return df
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None


def detect_available_target_columns(df):
    """
    Detect all normalized anomaly columns for wavelet decomposition.
    """
    target_cols = [
        c for c in df.columns
        if c.endswith("_anomaly_normalized")
    ]

    if not target_cols:
        raise ValueError("No normalized anomaly columns found for wavelet decomposition.")

    print("✅ Wavelet target columns:")
    for c in target_cols:
        print(f"   - {c}")

    return target_cols


def choose_time_column(df, variable):
    """
    Prefer GRACE time for GRACE-derived variable, otherwise ERA5 time.
    """
    if variable.startswith("lwe_thickness"):
        if "time_grace" in df.columns:
            return "time_grace"
    if "time_era5" in df.columns:
        return "time_era5"
    if "time_grace" in df.columns:
        return "time_grace"
    raise ValueError("No usable time column found.")


def pad_series_for_swt(x, level):
    """
    SWT needs length divisible by 2**level.
    Pad at the end using reflection.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    block = 2 ** level
    target_len = int(np.ceil(n / block) * block)
    pad_len = target_len - n

    if pad_len == 0:
        return x, 0

    x_pad = np.pad(x, (0, pad_len), mode="reflect")
    return x_pad, pad_len


def swt_reconstruct_band(x, wavelet="db4", level=5):
    """
    Decompose one 1D series using SWT (MODWT-style practical choice),
    then reconstruct interpretable bands:

    short    = D1 + D2
    seasonal = D3
    long     = D4 + D5 + A5

    Returns arrays aligned to original length.
    """
    x = np.asarray(x, dtype=float)
    x_pad, pad_len = pad_series_for_swt(x, level=level)

    coeffs = pywt.swt(x_pad, wavelet=wavelet, level=level, norm=True)

    # coeffs order:
    # [(cA_n, cD_n), ..., (cA_1, cD_1)]
    # so coeffs[0] is level n, coeffs[-1] is level 1

    def reconstruct_component(detail_levels_to_keep=None, keep_final_approx=False):
        if detail_levels_to_keep is None:
            detail_levels_to_keep = []

        coeffs_band = []

        for lev in range(level, 0, -1):
            cA, cD = coeffs[level - lev]

            # Keep A5 only at the highest level if requested
            if keep_final_approx and lev == level:
                keep_a = cA.copy()
            else:
                keep_a = np.zeros_like(cA)

            keep_d = cD.copy() if lev in detail_levels_to_keep else np.zeros_like(cD)

            coeffs_band.append((keep_a, keep_d))

        rec = pywt.iswt(coeffs_band, wavelet=wavelet, norm=True)
        return np.asarray(rec)

    short_rec = reconstruct_component(detail_levels_to_keep=[1, 2], keep_final_approx=False)
    seasonal_rec = reconstruct_component(detail_levels_to_keep=[3], keep_final_approx=False)
    long_rec = reconstruct_component(detail_levels_to_keep=[4, 5], keep_final_approx=True)

    reconstructed = short_rec + seasonal_rec + long_rec

    if pad_len > 0:
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


def safe_nanvar(x):
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)

    if valid.sum() < 2:
        return np.nan

    return np.nanvar(x)


def compute_variance_fractions(raw, short, seasonal, long):
    raw_var = safe_nanvar(raw)
    short_var = safe_nanvar(short)
    seasonal_var = safe_nanvar(seasonal)
    long_var = safe_nanvar(long)

    if not np.isfinite(raw_var) or raw_var == 0:
        return np.nan, np.nan, np.nan

    return (
        short_var / raw_var if np.isfinite(short_var) else np.nan,
        seasonal_var / raw_var if np.isfinite(seasonal_var) else np.nan,
        long_var / raw_var if np.isfinite(long_var) else np.nan,
    )


def compute_rmse(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((x[mask] - y[mask]) ** 2))


def decompose_one_group(group_df, variable, wavelet=WAVELET, level=LEVEL):
    """
    Decompose one basin-variable time series.
    Assumes rows already sorted by time.
    """
    x = group_df[variable].to_numpy(dtype=float)

    valid_count = np.isfinite(x).sum()

    # Skip all-NaN or too-sparse series
    if valid_count < 24:
        return None, {
            "basin": group_df["basin"].iloc[0],
            "variable": variable.replace("_anomaly_normalized", ""),
            "short_frac": np.nan,
            "seasonal_frac": np.nan,
            "long_frac": np.nan,
            "rmse": np.nan,
        }

    if np.isnan(x).any():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)

    # If interpolation still leaves bad values, skip
    if np.isfinite(x).sum() < 24:
        return None, {
            "basin": group_df["basin"].iloc[0],
            "variable": variable.replace("_anomaly_normalized", ""),
            "short_frac": np.nan,
            "seasonal_frac": np.nan,
            "long_frac": np.nan,
            "rmse": np.nan,
        }


    bands = swt_reconstruct_band(x, wavelet=wavelet, level=level)

    out = group_df.copy()
    base = variable.replace("_anomaly_normalized", "")

    out[f"{base}_short"] = bands["short"]
    out[f"{base}_seasonal"] = bands["seasonal"]
    out[f"{base}_long"] = bands["long"]
    out[f"{base}_reconstructed"] = bands["reconstructed"]

    short_frac, seasonal_frac, long_frac = compute_variance_fractions(
        x, bands["short"], bands["seasonal"], bands["long"]
    )
    rmse = compute_rmse(x, bands["reconstructed"])

    diagnostics = {
        "basin": group_df["basin"].iloc[0],
        "variable": base,
        "n_obs": int(np.isfinite(x).sum()),
        "signal_mean": float(np.nanmean(x)) if np.isfinite(x).any() else np.nan,
        "signal_std": float(np.nanstd(x)) if np.isfinite(x).any() else np.nan,
        "signal_min": float(np.nanmin(x)) if np.isfinite(x).any() else np.nan,
        "signal_max": float(np.nanmax(x)) if np.isfinite(x).any() else np.nan,
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
        time_col = choose_time_column(df, variable)

        selected_cols = ["basin", "year", "month", time_col, variable]
        selected_cols = list(dict.fromkeys(selected_cols))

        sub = df[selected_cols].copy()
        sub[time_col] = pd.to_datetime(sub[time_col])

        variable_groups = []

        for basin_id, group in sub.groupby("basin"):
            group = group.sort_values(time_col).reset_index(drop=True)

            if len(group) < 16:
                continue

            out_group, diag = decompose_one_group(group, variable)
            diagnostics.append(diag)

            if out_group is not None:
                # Keep only the keys and newly created wavelet cols
                base = variable.replace("_anomaly_normalized", "")
                keep_cols = [
                    "basin", "year", "month",
                    f"{base}_short",
                    f"{base}_seasonal",
                    f"{base}_long",
                    f"{base}_reconstructed",
                ]
                keep_cols = [c for c in keep_cols if c in out_group.columns]
                variable_groups.append(out_group[keep_cols].copy())

        if variable_groups:
            variable_df = pd.concat(variable_groups, ignore_index=True)
            variable_df = variable_df.drop_duplicates(subset=["basin", "year", "month"])
            per_variable_outputs.append(variable_df)

    if not per_variable_outputs:
        return pd.DataFrame(), pd.DataFrame(diagnostics)

    # Merge all variable outputs horizontally
    wavelet_df = per_variable_outputs[0]
    for other_df in per_variable_outputs[1:]:
        wavelet_df = wavelet_df.merge(
            other_df,
            on=["basin", "year", "month"],
            how="outer"
        )

    diag_df = pd.DataFrame(diagnostics)

    return wavelet_df, diag_df

def merge_wavelet_outputs_back(df, wavelet_df):
    """
    Merge multiscale outputs back to the original dataset
    using basin-year-month keys.
    """
    merge_cols = ["basin", "year", "month"]
    merged = df.merge(wavelet_df, on=merge_cols, how="left")
    return merged


def save_tidy_summary_table(diagnostics):
    """
    Save clean per-basin / per-variable summary table.
    """
    tidy = diagnostics[
        [
            "basin",
            "variable",
            "short_frac",
            "seasonal_frac",
            "long_frac",
            "rmse",
        ]
    ].copy()

    tidy = tidy.rename(
        columns={
            "short_frac": "var_short_frac",
            "seasonal_frac": "var_seasonal_frac",
            "long_frac": "var_long_frac",
            "rmse": "reconstruction_rmse",
        }
    )

    tidy = tidy.sort_values(["variable", "basin"]).reset_index(drop=True)

    try:
        os.makedirs(os.path.dirname(SUMMARY_TIDY_OUTPUT), exist_ok=True)
        tidy.to_csv(SUMMARY_TIDY_OUTPUT, index=False)
        print(f"✅ Saved tidy summary table: {SUMMARY_TIDY_OUTPUT}")
    except Exception as e:
        print(f"❌ Failed to save tidy summary table: {e}")

def save_outputs(df, diagnostics):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(VARIANCE_OUTPUT), exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Saved wavelet dataset: {OUTPUT_FILE}")

    variance_df = diagnostics[["basin", "variable", "short_frac", "seasonal_frac", "long_frac"]].copy()
    variance_df.to_csv(VARIANCE_OUTPUT, index=False)
    print(f"✅ Saved variance partition: {VARIANCE_OUTPUT}")

    rmse_df = diagnostics[["basin", "variable", "rmse"]].copy()
    rmse_df.to_csv(RECON_ERROR_OUTPUT, index=False)
    print(f"✅ Saved reconstruction error: {RECON_ERROR_OUTPUT}")

    diagnostics.to_csv(SUMMARY_OUTPUT, index=False)
    print(f"✅ Saved basin-variable summary: {SUMMARY_OUTPUT}")

    save_tidy_summary_table(diagnostics)


def plot_example_decomposition(raw_df, wavelet_df, basin_id, variable, out_path):
    """
    Plot raw normalized anomaly series + reconstructed bands.
    """
    if basin_id is None:
        return

    norm_col = variable if variable.endswith("_anomaly_normalized") else f"{variable}_anomaly_normalized"
    base = norm_col.replace("_anomaly_normalized", "")

    short_col = f"{base}_short"
    seasonal_col = f"{base}_seasonal"
    long_col = f"{base}_long"

    time_col = choose_time_column(raw_df, norm_col)

    raw_req = ["basin", "year", "month", time_col, norm_col]
    wavelet_req = ["basin", "year", "month", short_col, seasonal_col, long_col]

    raw_missing = [c for c in raw_req if c not in raw_df.columns]
    wavelet_missing = [c for c in wavelet_req if c not in wavelet_df.columns]

    if raw_missing or wavelet_missing:
        print(
            f"⚠️ Skipping plot for basin {basin_id}, variable {base}. "
            f"Missing raw: {raw_missing} | missing wavelet: {wavelet_missing}"
        )
        return

    raw_plot = raw_df[raw_req].copy()
    wavelet_plot = wavelet_df[wavelet_req].copy()

    raw_plot = raw_plot[raw_plot["basin"] == basin_id].copy()
    wavelet_plot = wavelet_plot[wavelet_plot["basin"] == basin_id].copy()

    if raw_plot.empty or wavelet_plot.empty:
        print(f"⚠️ No rows for basin {basin_id}, variable {base}")
        return

    plot_df = raw_plot.merge(
        wavelet_plot,
        on=["basin", "year", "month"],
        how="inner"
    )

    if plot_df.empty:
        print(f"⚠️ No merged rows for basin {basin_id}, variable {base}")
        return

    plot_df[time_col] = pd.to_datetime(plot_df[time_col])
    plot_df = plot_df.sort_values(time_col)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(plot_df[time_col], plot_df[norm_col])
    axes[0].set_title(f"Basin {basin_id} | {base} | raw normalized anomaly")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(plot_df[time_col], plot_df[short_col])
    axes[1].set_title("short-term (D1 + D2)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(plot_df[time_col], plot_df[seasonal_col])
    axes[2].set_title("seasonal/annual (D3)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(plot_df[time_col], plot_df[long_col])
    axes[3].set_title("long-term (D4 + D5 + A5)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved figure: {out_path}")



def plot_required_examples(raw_df, wavelet_df):
    """
    Create requested example plots for selected basins and variables.
    """
    variable_map = {
        "GRACE TWSA": "lwe_thickness",
        "precipitation": "tp",
        "swvl1": "swvl1",
    }

    for basin_label, basin_id in EXAMPLE_BASINS.items():
        for human_name, variable in variable_map.items():
            out_name = f"{basin_label}_{variable}_wavelet_example.png"
            out_path = os.path.join(FIGURE_DIR, out_name)
            plot_example_decomposition(raw_df, wavelet_df, basin_id, variable, out_path)


def main():
    print("--- Wavelet basin multi-scale decomposition ---")
    print(f"Wavelet: {WAVELET} | Levels: {LEVEL}")
    print("Bands: short = D1 + D2, seasonal = D3, long = D4 + D5 + A5")

    df = load_dataset(INPUT_FILE)
    if df is None:
        return

    variables = detect_available_target_columns(df)
    save_wavelet_config(variables)

    wavelet_df, diagnostics = run_wavelet_decomposition(df, variables)
    merged_df = merge_wavelet_outputs_back(df, wavelet_df)
    before = len(merged_df)

    merged_df = merged_df.drop_duplicates(subset=["basin", "year", "month"])

    after = len(merged_df)
    print(f"✅ Dropped {before - after} duplicated basin-month rows")

    dup_max = merged_df.groupby(["basin", "year", "month"]).size().max()
    print(f"✅ Max rows per basin-year-month in wavelet output: {dup_max}")

    dup_counts = merged_df.groupby(["basin", "year", "month"]).size()
    bad = dup_counts[dup_counts > 1]

    print(f"⚠️ Number of duplicated basin-month keys: {len(bad)}")

    if len(bad) > 0:
        print("Example duplicated keys:")
        print(bad.head(10))

    if len(bad) > 0:
        bad_keys = bad.index.to_frame(index=False).head(10)
        debug = merged_df.merge(bad_keys, on=["basin", "year", "month"], how="inner")
        print(debug.sort_values(["basin", "year", "month"]).head(20))

    save_outputs(merged_df, diagnostics)
    plot_required_examples(df, merged_df)

    print("\nDiagnostics summary:")
    print(diagnostics.groupby("variable")[["short_frac", "seasonal_frac", "long_frac", "rmse"]].mean())

    print("--- Done ---")


if __name__ == "__main__":
    main()
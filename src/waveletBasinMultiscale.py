# waveletBasinMultiscale.py
# Step 6: Dual gap-aware SWT products for retrospective atlas and held-out evaluation

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning)

INPUT_FILE = "data/processed/basin_dataset_anomaly_normalized.parquet"

# Retrospective full-record SWT used for the scientific atlas and continuous trajectories.
ATLAS_OUTPUT_FILE = "data/processed/basin_dataset_wavelet_multiscale.parquet"
# Split-local SWT used only for train/validation/test model evaluation.
EVAL_OUTPUT_FILE = "data/processed/basin_dataset_wavelet_multiscale_eval.parquet"

FIGURE_DIR = "results/figures/wavelet_examples"
ATLAS_VARIANCE_OUTPUT = "results/tables/wavelet_variance_partition.csv"
ATLAS_RECON_ERROR_OUTPUT = "results/tables/wavelet_reconstruction_error.csv"
ATLAS_SUMMARY_OUTPUT = "results/tables/wavelet_basin_variable_summary.csv"
EVAL_SUMMARY_OUTPUT = "results/tables/wavelet_basin_variable_summary_eval.csv"
CONFIG_OUTPUT = "configs/wavelet_config.yaml"

WAVELET = "db4"
LEVEL = 5
MAX_INTERPOLATION_GAP_MONTHS = 2
MIN_SEGMENT_MONTHS = 24

TRAIN_END_YEAR = 2020
VAL_START_YEAR = 2021
VAL_END_YEAR = 2023
TEST_START_YEAR = 2024

EXAMPLE_BASINS = {
    "amazon": 6050298170,
    "iceland": 2050058330,
    "thailand": 4050018280,
    "egypt": 1050000010,
}


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
    df = df.copy()
    canonical = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    if "time" in df.columns:
        existing = pd.to_datetime(df["time"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        mismatch = existing.notna() & (existing != canonical)
        if mismatch.any():
            raise ValueError(f"Canonical time disagrees with year/month in {int(mismatch.sum())} rows.")
    df["time"] = canonical
    return df


def assign_split(df):
    df = df.copy()
    conditions = [
        df["year"] <= TRAIN_END_YEAR,
        (df["year"] >= VAL_START_YEAR) & (df["year"] <= VAL_END_YEAR),
        df["year"] >= TEST_START_YEAR,
    ]
    df["split"] = np.select(conditions, ["train", "val", "test"], default="unknown")
    return df


def validate_monthly_grid(df):
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
    print(f"✅ Wavelet variables: {len(target_cols)}")
    for c in target_cols:
        print(f"   - {c}")
    return target_cols


def pad_series_for_swt(x, level):
    x = np.asarray(x, dtype=float)
    block = 2 ** level
    target_len = int(np.ceil(len(x) / block) * block)
    pad_len = target_len - len(x)
    return (x, 0) if pad_len == 0 else (np.pad(x, (0, pad_len), mode="reflect"), pad_len)


def swt_reconstruct_band(x, wavelet=WAVELET, level=LEVEL):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("SWT received non-finite values.")

    x_pad, pad_len = pad_series_for_swt(x, level)
    coeffs = pywt.swt(x_pad, wavelet=wavelet, level=level, norm=True)

    def reconstruct_component(detail_levels_to_keep, keep_final_approx=False):
        coeffs_band = []
        for lev in range(level, 0, -1):
            cA, cD = coeffs[level - lev]
            keep_a = cA.copy() if (keep_final_approx and lev == level) else np.zeros_like(cA)
            keep_d = cD.copy() if lev in detail_levels_to_keep else np.zeros_like(cD)
            coeffs_band.append((keep_a, keep_d))
        return np.asarray(pywt.iswt(coeffs_band, wavelet=wavelet, norm=True))

    short = reconstruct_component([1, 2])
    seasonal = reconstruct_component([3])
    long = reconstruct_component([4, 5], keep_final_approx=True)
    reconstructed = short + seasonal + long

    if pad_len:
        short = short[:-pad_len]
        seasonal = seasonal[:-pad_len]
        long = long[:-pad_len]
        reconstructed = reconstructed[:-pad_len]

    return {"short": short, "seasonal": seasonal, "long": long, "reconstructed": reconstructed}


def missing_run_lengths(x):
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
    x = np.asarray(x, dtype=float).copy()
    imputed = np.zeros(len(x), dtype=bool)
    for start, end, length in missing_run_lengths(x):
        bounded = start > 0 and end < len(x) - 1
        if not bounded or length > max_gap:
            continue
        if not (np.isfinite(x[start - 1]) and np.isfinite(x[end + 1])):
            continue
        x[start:end + 1] = np.linspace(x[start - 1], x[end + 1], length + 2)[1:-1]
        imputed[start:end + 1] = True
    return x, imputed


def finite_segments(x):
    valid = np.isfinite(np.asarray(x, dtype=float))
    segments = []
    i = 0
    while i < len(valid):
        if not valid[i]:
            i += 1
            continue
        start = i
        while i < len(valid) and valid[i]:
            i += 1
        segments.append((start, i - 1))
    return segments


def safe_nanvar(x):
    x = np.asarray(x, dtype=float)
    return np.nan if np.isfinite(x).sum() < 2 else np.nanvar(x)


def compute_variance_fractions(raw, short, seasonal, long):
    raw_var = safe_nanvar(raw)
    if not np.isfinite(raw_var) or raw_var == 0:
        return np.nan, np.nan, np.nan
    vals = [safe_nanvar(short), safe_nanvar(seasonal), safe_nanvar(long)]
    return tuple(v / raw_var if np.isfinite(v) else np.nan for v in vals)


def compute_rmse(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return np.nan if mask.sum() == 0 else np.sqrt(np.mean((x[mask] - y[mask]) ** 2))


def decompose_one_group(group_df, variable, product, split_label=None):
    group_df = group_df.sort_values("time").reset_index(drop=True).copy()
    x_original = group_df[variable].to_numpy(dtype=float)
    x_filled, imputed_mask = interpolate_only_short_internal_gaps(x_original)

    short = np.full(len(x_filled), np.nan)
    seasonal = np.full(len(x_filled), np.nan)
    long = np.full(len(x_filled), np.nan)
    reconstructed = np.full(len(x_filled), np.nan)

    all_segments = finite_segments(x_filled)
    used_segments = []
    for start, end in all_segments:
        if end - start + 1 < MIN_SEGMENT_MONTHS:
            continue
        bands = swt_reconstruct_band(x_filled[start:end + 1])
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

    observed_for_eval = x_original.copy()
    observed_for_eval[imputed_mask] = np.nan
    short_frac, seasonal_frac, long_frac = compute_variance_fractions(x_filled, short, seasonal, long)
    original_runs = missing_run_lengths(x_original)

    diag = {
        "product": product,
        "split": split_label if split_label is not None else "full_record",
        "basin": group_df["basin"].iloc[0],
        "variable": base,
        "n_calendar_months": int(len(x_original)),
        "n_observed_months": int(np.isfinite(x_original).sum()),
        "n_missing_original": int((~np.isfinite(x_original)).sum()),
        "n_short_gap_imputed": int(imputed_mask.sum()),
        "n_missing_left_unfilled": int((~np.isfinite(x_filled)).sum()),
        "max_original_missing_run": int(max([r[2] for r in original_runs], default=0)),
        "n_swt_segments_used": int(len(used_segments)),
        "n_months_with_swt_output": int(np.isfinite(reconstructed).sum()),
        "short_frac": short_frac,
        "seasonal_frac": seasonal_frac,
        "long_frac": long_frac,
        "rmse": compute_rmse(observed_for_eval, reconstructed),
    }
    return out, diag


def _merge_variable_outputs(per_variable_outputs):
    if not per_variable_outputs:
        return pd.DataFrame()
    wavelet_df = per_variable_outputs[0]
    for other in per_variable_outputs[1:]:
        wavelet_df = wavelet_df.merge(
            other, on=["basin", "year", "month"], how="outer", validate="one_to_one"
        )
    return wavelet_df


def run_atlas_decomposition(df, variables):
    diagnostics = []
    per_variable_outputs = []
    for variable in variables:
        variable_groups = []
        sub = df[["basin", "year", "month", "time", variable]].copy()
        for _, group in sub.groupby("basin"):
            out, diag = decompose_one_group(group, variable, product="atlas")
            variable_groups.append(out)
            diagnostics.append(diag)
        per_variable_outputs.append(pd.concat(variable_groups, ignore_index=True))
    return _merge_variable_outputs(per_variable_outputs), pd.DataFrame(diagnostics)


def run_evaluation_decomposition(df, variables):
    """Perform SWT independently within train/val/test blocks.

    This prevents wavelet support near 2020/2021 or 2023/2024 from mixing samples
    across held-out split boundaries.
    """
    diagnostics = []
    per_variable_outputs = []
    valid = df[df["split"].isin(["train", "val", "test"])].copy()

    for variable in variables:
        variable_groups = []
        sub = valid[["basin", "year", "month", "time", "split", variable]].copy()
        for (basin, split_name), group in sub.groupby(["basin", "split"]):
            out, diag = decompose_one_group(
                group, variable, product="evaluation", split_label=split_name
            )
            variable_groups.append(out)
            diagnostics.append(diag)
        if variable_groups:
            combined = pd.concat(variable_groups, ignore_index=True)
            if combined.duplicated(["basin", "year", "month"]).any():
                raise ValueError(f"Duplicate evaluation basin-month rows for {variable}.")
            per_variable_outputs.append(combined)

    return _merge_variable_outputs(per_variable_outputs), pd.DataFrame(diagnostics)


def merge_wavelet_outputs_back(df, wavelet_df):
    return df.merge(wavelet_df, on=["basin", "year", "month"], how="left", validate="one_to_one")


def save_config(variables):
    config = {
        "wavelet_transform": {
            "transform_type": "swt",
            "wavelet_family": WAVELET,
            "levels": LEVEL,
            "padding_method": "reflect",
            "short": "D1 + D2",
            "seasonal": "D3",
            "long": "D4 + D5 + A5",
        },
        "gap_policy": {
            "max_interpolation_gap_months": MAX_INTERPOLATION_GAP_MONTHS,
            "minimum_segment_months": MIN_SEGMENT_MONTHS,
        },
        "products": {
            "atlas": {
                "file": ATLAS_OUTPUT_FILE,
                "definition": "full-record retrospective SWT; no train/val/test boundaries",
                "use": "continuous atlas, clustering, physical interpretation, trajectories, transitions",
            },
            "evaluation": {
                "file": EVAL_OUTPUT_FILE,
                "definition": "SWT performed independently inside train/val/test blocks",
                "use": "autoencoder train/validation/test reconstruction evaluation",
            },
        },
        "variables": variables,
    }
    os.makedirs(os.path.dirname(CONFIG_OUTPUT), exist_ok=True)
    with open(CONFIG_OUTPUT, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"✅ Saved dual-product wavelet config: {CONFIG_OUTPUT}")


def save_outputs(atlas_df, atlas_diag, eval_df, eval_diag):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    atlas_df.to_parquet(ATLAS_OUTPUT_FILE, index=False)
    eval_df.to_parquet(EVAL_OUTPUT_FILE, index=False)
    atlas_diag.to_csv(ATLAS_SUMMARY_OUTPUT, index=False)
    eval_diag.to_csv(EVAL_SUMMARY_OUTPUT, index=False)

    cols = ["basin", "variable", "short_frac", "seasonal_frac", "long_frac"]
    atlas_diag[cols].to_csv(ATLAS_VARIANCE_OUTPUT, index=False)
    atlas_diag[["basin", "variable", "rmse"]].to_csv(ATLAS_RECON_ERROR_OUTPUT, index=False)

    print(f"✅ Saved atlas SWT dataset: {ATLAS_OUTPUT_FILE}")
    print(f"✅ Saved evaluation SWT dataset: {EVAL_OUTPUT_FILE}")
    print(f"✅ Saved atlas diagnostics: {ATLAS_SUMMARY_OUTPUT}")
    print(f"✅ Saved evaluation diagnostics: {EVAL_SUMMARY_OUTPUT}")


def plot_example_decomposition(raw_df, wavelet_df, basin_id, variable, out_path):
    norm_col = f"{variable}_anomaly_normalized"
    base = variable
    needed = [f"{base}_short", f"{base}_seasonal", f"{base}_long"]
    if norm_col not in raw_df.columns or any(c not in wavelet_df.columns for c in needed):
        return
    raw = raw_df[raw_df["basin"] == basin_id][["basin", "year", "month", "time", norm_col]]
    wav = wavelet_df[wavelet_df["basin"] == basin_id][["basin", "year", "month"] + needed]
    plot_df = raw.merge(wav, on=["basin", "year", "month"], how="inner").sort_values("time")
    if plot_df.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(plot_df["time"], plot_df[norm_col]); axes[0].set_title("normalized anomaly")
    axes[1].plot(plot_df["time"], plot_df[f"{base}_short"]); axes[1].set_title("short-term (D1 + D2)")
    axes[2].plot(plot_df["time"], plot_df[f"{base}_seasonal"]); axes[2].set_title("seasonal (D3)")
    axes[3].plot(plot_df["time"], plot_df[f"{base}_long"]); axes[3].set_title("long-term (D4 + D5 + A5)")
    for ax in axes: ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_required_examples(raw_df, atlas_df):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for label, basin_id in EXAMPLE_BASINS.items():
        for variable in ["lwe_thickness", "tp", "swvl1"]:
            plot_example_decomposition(
                raw_df, atlas_df, basin_id, variable,
                os.path.join(FIGURE_DIR, f"{label}_{variable}_wavelet_example.png")
            )


def main():
    print("--- Step 6: Dual gap-aware SWT decomposition ---")
    print("Atlas product: full-record retrospective decomposition")
    print("Evaluation product: decomposition independently within train/val/test blocks")

    df = load_dataset(INPUT_FILE)
    if df is None:
        return
    df = add_canonical_time(df)
    df = assign_split(df)
    validate_monthly_grid(df)
    variables = detect_available_target_columns(df)

    atlas_wav, atlas_diag = run_atlas_decomposition(df, variables)
    eval_wav, eval_diag = run_evaluation_decomposition(df, variables)

    atlas_df = merge_wavelet_outputs_back(df, atlas_wav)
    eval_df = merge_wavelet_outputs_back(df, eval_wav)

    for label, out in [("atlas", atlas_df), ("evaluation", eval_df)]:
        if out.duplicated(["basin", "year", "month"]).any():
            raise ValueError(f"Duplicate basin-month rows in {label} SWT product.")

    save_config(variables)
    save_outputs(atlas_df, atlas_diag, eval_df, eval_diag)
    plot_required_examples(df, atlas_df)

    print("\n✅ Role separation complete:")
    print("   basin_dataset_wavelet_multiscale.parquet -> retrospective scientific atlas")
    print("   basin_dataset_wavelet_multiscale_eval.parquet -> held-out ML evaluation")
    print("--- Done ---")


if __name__ == "__main__":
    main()

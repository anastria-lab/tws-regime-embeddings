# normalizeBasinTimeseries.py
# Step 5: Train-reference climatology/anomaly normalization for basin-level time series

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "data/interim/basin_era5_grace_merged.parquet"
OUTPUT_FILE = "data/processed/basin_dataset_anomaly_normalized.parquet"
CLIMATOLOGY_OUTPUT_FILE = "data/processed/basin_monthly_climatology.parquet"
STD_OUTPUT_FILE = "data/processed/basin_anomaly_std.parquet"
QC_OUTPUT_FILE = "results/tables/normalization_reference_qc.csv"

# Keep the anomaly definition independent of validation/test years.
# This matches the train block used later by the autoencoder pipeline.
REFERENCE_END_YEAR = 2020

ID_COLUMNS = [
    "basin",
    "year",
    "month",
    "time",
    "time_era5",
    "time_grace",
]

NON_GEOPHYSICAL_COLUMNS = ["number", "expver"]
NON_FEATURE_COLUMNS = set(ID_COLUMNS + NON_GEOPHYSICAL_COLUMNS)


def drop_non_geophysical_columns(df):
    cols_to_drop = [c for c in NON_GEOPHYSICAL_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"✅ Dropped non-geophysical columns: {cols_to_drop}")
    return df


def load_dataset(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded dataset: {file_path}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None


def detect_feature_columns(df):
    feature_cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    print("\n✅ Feature columns selected for reference climatology/anomalies:")
    for col in feature_cols:
        print(f"   - {col}")
    return feature_cols


def get_reference_rows(df):
    ref = df[df["year"] <= REFERENCE_END_YEAR].copy()
    if ref.empty:
        raise ValueError(f"No rows found in reference period <= {REFERENCE_END_YEAR}.")

    print(
        f"✅ Normalization reference period: {int(ref['year'].min())}–"
        f"{int(ref['year'].max())} (validation/test years excluded)"
    )
    return ref


def compute_monthly_climatology(reference_df, feature_cols):
    """Fit basin × calendar-month climatology using reference years only."""
    climatology = (
        reference_df.groupby(["basin", "month"])[feature_cols]
        .mean()
        .reset_index()
    )
    climatology = climatology.rename(
        columns={col: f"{col}_climatology" for col in feature_cols}
    )
    climatology["reference_end_year"] = REFERENCE_END_YEAR

    print("✅ Reference-period monthly climatology computed")
    print(f"   Rows: {len(climatology):,}")
    return climatology


def add_anomalies(df, climatology, feature_cols):
    """Apply the fixed reference climatology to the complete record."""
    df_out = df.merge(
        climatology.drop(columns=["reference_end_year"], errors="ignore"),
        on=["basin", "month"],
        how="left",
        validate="many_to_one",
    )

    for col in feature_cols:
        clim_col = f"{col}_climatology"
        anom_col = f"{col}_anomaly"
        df_out[anom_col] = df_out[col] - df_out[clim_col]

    print("✅ Anomalies computed relative to the fixed train-era climatology")
    return df_out


def compute_reference_stds(df_with_anomalies, feature_cols):
    """Fit basin-specific anomaly std using reference years only."""
    ref = df_with_anomalies[df_with_anomalies["year"] <= REFERENCE_END_YEAR].copy()
    anomaly_cols = [f"{col}_anomaly" for col in feature_cols]

    stds = ref.groupby("basin")[anomaly_cols].std(ddof=0).reset_index()
    stds = stds.rename(columns={col: f"{col}_std" for col in anomaly_cols})
    stds["reference_end_year"] = REFERENCE_END_YEAR
    return stds


def add_normalized_anomalies(df, stds, feature_cols):
    """Apply fixed reference-period basin std values to all years."""
    df_out = df.merge(
        stds.drop(columns=["reference_end_year"], errors="ignore"),
        on="basin",
        how="left",
        validate="many_to_one",
    )

    for col in feature_cols:
        anom_col = f"{col}_anomaly"
        std_col = f"{anom_col}_std"
        norm_col = f"{col}_anomaly_normalized"
        df_out[norm_col] = np.where(
            df_out[std_col].notna() & (df_out[std_col] > 0),
            df_out[anom_col] / df_out[std_col],
            np.nan,
        )

    print("✅ Normalized anomalies computed with fixed reference-period std")
    return df_out


def build_reference_qc(df, feature_cols):
    rows = []
    ref = df[df["year"] <= REFERENCE_END_YEAR]
    later = df[df["year"] > REFERENCE_END_YEAR]

    for col in feature_cols:
        anom = f"{col}_anomaly"
        norm = f"{col}_anomaly_normalized"
        rows.append({
            "variable": col,
            "reference_end_year": REFERENCE_END_YEAR,
            "n_reference_values": int(ref[col].notna().sum()),
            "n_post_reference_values": int(later[col].notna().sum()),
            "reference_anomaly_mean": float(ref[anom].mean(skipna=True)),
            "reference_normalized_std": float(ref[norm].std(ddof=0, skipna=True)),
            "post_reference_anomaly_mean": float(later[anom].mean(skipna=True)) if len(later) else np.nan,
            "post_reference_normalized_mean": float(later[norm].mean(skipna=True)) if len(later) else np.nan,
            "normalized_missing_fraction_all": float(df[norm].isna().mean()),
        })

    qc = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(QC_OUTPUT_FILE), exist_ok=True)
    qc.to_csv(QC_OUTPUT_FILE, index=False)
    print(f"✅ Saved normalization reference QC: {QC_OUTPUT_FILE}")
    return qc


def quality_check_reference_monthly_means(df, feature_cols):
    print("\n" + "=" * 60)
    print("QUALITY CHECK — Reference-period monthly anomaly means near zero")
    print("=" * 60)
    ref = df[df["year"] <= REFERENCE_END_YEAR]
    rows = []
    for col in feature_cols:
        anom_col = f"{col}_anomaly"
        grouped = ref.groupby(["basin", "month"])[anom_col].mean().abs()
        rows.append({
            "variable": col,
            "mean_abs_reference_monthly_anomaly_mean": grouped.mean(),
            "median_abs_reference_monthly_anomaly_mean": grouped.median(),
            "max_abs_reference_monthly_anomaly_mean": grouped.max(),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def quality_check_reference_normalized_std(df, feature_cols):
    print("\n" + "=" * 60)
    print("QUALITY CHECK — Reference-period normalized basin std near one")
    print("=" * 60)
    ref = df[df["year"] <= REFERENCE_END_YEAR]
    rows = []
    for col in feature_cols:
        norm_col = f"{col}_anomaly_normalized"
        grouped_std = ref.groupby("basin")[norm_col].std(ddof=0)
        rows.append({
            "variable": col,
            "mean_basin_std": grouped_std.mean(),
            "median_basin_std": grouped_std.median(),
            "min_basin_std": grouped_std.min(),
            "max_basin_std": grouped_std.max(),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def quality_check_plot_raw_vs_anomaly(df, basin_id, variable):
    raw_col = variable
    anom_col = f"{variable}_anomaly"
    if raw_col not in df.columns or anom_col not in df.columns:
        return

    plot_df = df[df["basin"] == basin_id].copy().sort_values("time")
    if plot_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(plot_df["time"], plot_df[raw_col])
    axes[0].set_title(f"Raw time series — basin {basin_id}, variable {variable}")
    axes[1].plot(plot_df["time"], plot_df[anom_col])
    axes[1].axvline(pd.Timestamp(f"{REFERENCE_END_YEAR}-12-31"), linestyle="--", alpha=0.5)
    axes[1].set_title("Anomaly relative to fixed train-era monthly climatology")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_parquet(df, output_file, label):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"✅ Saved {label}: {output_file}")


def main():
    print("--- Step 5: Train-reference anomaly normalization ---")

    df = load_dataset(INPUT_FILE)
    if df is None:
        return

    df = drop_non_geophysical_columns(df)
    df["time"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))

    feature_cols = detect_feature_columns(df)
    if not feature_cols:
        raise ValueError("No geophysical feature columns found.")

    reference = get_reference_rows(df)
    climatology = compute_monthly_climatology(reference, feature_cols)
    df = add_anomalies(df, climatology, feature_cols)
    stds = compute_reference_stds(df, feature_cols)
    df = add_normalized_anomalies(df, stds, feature_cols)

    quality_check_reference_monthly_means(df, feature_cols)
    quality_check_reference_normalized_std(df, feature_cols)
    build_reference_qc(df, feature_cols)

    example_basin = df["basin"].iloc[0]
    example_variable = feature_cols[0]
    quality_check_plot_raw_vs_anomaly(df, example_basin, example_variable)

    save_parquet(df, OUTPUT_FILE, "normalized basin dataset")
    save_parquet(climatology, CLIMATOLOGY_OUTPUT_FILE, "reference monthly climatology")
    save_parquet(stds, STD_OUTPUT_FILE, "reference anomaly standard deviations")

    print("\nScientific interpretation:")
    print(
        f"   All years are expressed relative to the <= {REFERENCE_END_YEAR} basin/month baseline."
    )
    print("   Validation/test years do not influence climatology or normalization parameters.")
    print("--- Done ---")


if __name__ == "__main__":
    main()

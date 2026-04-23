import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "data/interim/basin_era5_grace_merged.parquet"
OUTPUT_FILE = "data/processed/basin_dataset_anomaly_normalized.parquet"
CLIMATOLOGY_OUTPUT_FILE = "data/processed/basin_monthly_climatology.parquet"
STD_OUTPUT_FILE = "data/processed/basin_anomaly_std.parquet"

ID_COLUMNS = [
    "basin",
    "year",
    "month",
    "time_era5",
    "time_grace",
]


NON_GEOPHYSICAL_COLUMNS = [
    "number",
    "expver",
]

NON_FEATURE_COLUMNS = set(ID_COLUMNS + NON_GEOPHYSICAL_COLUMNS)

def drop_non_geophysical_columns(df):
    cols_to_drop = [c for c in NON_GEOPHYSICAL_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"✅ Dropped non-geophysical columns: {cols_to_drop}")
    return df

def load_dataset(file_path):
    """
    Load merged basin-month dataset from parquet.
    """
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
    """
    Detect numeric feature columns to normalize.
    Excludes id/time columns.
    """
    feature_cols = []

    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    print("\n✅ Feature columns selected for climatology/anomaly/normalization:")
    for col in feature_cols:
        print(f"   - {col}")

    return feature_cols

def compute_monthly_climatology(df, feature_cols):
    """
    Compute monthly climatology per basin and variable.

    Returns a wide table with one row per basin-month.
    """
    climatology = (
        df.groupby(["basin", "month"])[feature_cols]
        .mean()
        .reset_index()
    )

    climatology = climatology.rename(
        columns={col: f"{col}_climatology" for col in feature_cols}
    )

    print("\n✅ Monthly climatology computed")
    print(f"   Rows: {len(climatology):,}")

    return climatology


def add_anomalies(df, climatology, feature_cols):
    """
    Merge climatology back and compute anomalies:
    anomaly = raw - monthly_climatology
    """
    df_out = df.merge(climatology, on=["basin", "month"], how="left")

    for col in feature_cols:
        clim_col = f"{col}_climatology"
        anom_col = f"{col}_anomaly"
        df_out[anom_col] = df_out[col] - df_out[clim_col]

    print("✅ Anomalies computed")
    return df_out

def add_normalized_anomalies(df, feature_cols):
    """
    Normalize anomalies by basin-variable std across time.

    Returns:
    - normalized dataframe
    - std parameter table
    """
    df_out = df.copy()

    anomaly_cols = [f"{col}_anomaly" for col in feature_cols]

    stds = (
        df_out.groupby("basin")[anomaly_cols]
        .std(ddof=0)
        .reset_index()
    )

    stds = stds.rename(
        columns={col: f"{col}_std" for col in anomaly_cols}
    )

    df_out = df_out.merge(stds, on="basin", how="left")

    for col in feature_cols:
        anom_col = f"{col}_anomaly"
        std_col = f"{anom_col}_std"
        norm_col = f"{col}_anomaly_normalized"

        df_out[norm_col] = np.where(
            (df_out[std_col].notnull()) & (df_out[std_col] != 0),
            df_out[anom_col] / df_out[std_col],
            np.nan
        )

    print("✅ Normalized anomaly signals computed")
    return df_out, stds


def quality_check_monthly_anomaly_means(df, feature_cols):
    """
    Check 1:
    For each basin-variable-month, anomaly means should be near zero.
    """
    print("\n" + "=" * 60)
    print("QUALITY CHECK 1 — Monthly anomaly means near zero")
    print("=" * 60)

    summary = []

    for col in feature_cols:
        anom_col = f"{col}_anomaly"

        grouped = (
            df.groupby(["basin", "month"])[anom_col]
            .mean()
            .abs()
        )

        summary.append({
            "variable": col,
            "mean_abs_monthly_anomaly_mean": grouped.mean(),
            "median_abs_monthly_anomaly_mean": grouped.median(),
            "max_abs_monthly_anomaly_mean": grouped.max(),
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    return summary_df


def quality_check_normalized_std(df, feature_cols):
    """
    Check 2:
    For each basin-variable, normalized anomaly std should be near one.
    """
    print("\n" + "=" * 60)
    print("QUALITY CHECK 2 — Normalized std near one")
    print("=" * 60)

    summary = []

    for col in feature_cols:
        norm_col = f"{col}_anomaly_normalized"

        grouped_std = df.groupby("basin")[norm_col].std(ddof=0)

        summary.append({
            "variable": col,
            "mean_basin_std": grouped_std.mean(),
            "median_basin_std": grouped_std.median(),
            "min_basin_std": grouped_std.min(),
            "max_basin_std": grouped_std.max(),
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    return summary_df


def quality_check_plot_raw_vs_anomaly(df, basin_id, variable, time_col="time_era5"):
    """
    Check 3:
    Plot raw vs anomaly time series for one basin-variable example.
    """
    raw_col = variable
    anom_col = f"{variable}_anomaly"

    if raw_col not in df.columns or anom_col not in df.columns:
        print(f"❌ Columns for plotting not found: {raw_col}, {anom_col}")
        return

    plot_df = df[df["basin"] == basin_id].copy()

    if plot_df.empty:
        print(f"❌ No data found for basin {basin_id}")
        return

    # Prefer ERA5 time if present, otherwise GRACE time
    if time_col not in plot_df.columns:
        if "time_grace" in plot_df.columns:
            time_col = "time_grace"
        else:
            print("❌ No usable time column found for plotting.")
            return

    plot_df[time_col] = pd.to_datetime(plot_df[time_col])
    plot_df = plot_df.sort_values(time_col)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(plot_df[time_col], plot_df[raw_col])
    axes[0].set_title(f"Raw time series — basin {basin_id}, variable {variable}")
    axes[0].set_ylabel(raw_col)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(plot_df[time_col], plot_df[anom_col])
    axes[1].set_title(f"Anomaly time series — basin {basin_id}, variable {variable}")
    axes[1].set_ylabel(anom_col)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_dataset(df, output_file):
    """
    Save processed dataset to parquet.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_parquet(output_file, index=False)
        print(f"\n✅ Saved processed dataset to: {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Failed to save processed dataset: {e}")
        return None

def save_parquet(df, output_file, label):
    """
    Save dataframe to parquet.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_parquet(output_file, index=False)
        print(f"✅ Saved {label} to: {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Failed to save {label}: {e}")
        return None

def main():
    print("--- Normalize basin time series ---")

    df = load_dataset(INPUT_FILE)
    if df is None:
        return

    df = drop_non_geophysical_columns(df)

    feature_cols = detect_feature_columns(df)
    if not feature_cols:
        print("❌ No feature columns found.")
        return

    climatology = compute_monthly_climatology(df, feature_cols)
    df = add_anomalies(df, climatology, feature_cols)
    df, stds = add_normalized_anomalies(df, feature_cols)

    # Quality checks
    quality_check_monthly_anomaly_means(df, feature_cols)
    quality_check_normalized_std(df, feature_cols)

    # Example plot
    example_basin = df["basin"].iloc[0]
    example_variable = feature_cols[0]
    print("\n✅ Plotting example raw vs anomaly time series")
    print(f"   Basin: {example_basin}")
    print(f"   Variable: {example_variable}")
    quality_check_plot_raw_vs_anomaly(df, example_basin, example_variable)

    # Save outputs
    save_parquet(df, OUTPUT_FILE, "normalized basin dataset")
    save_parquet(climatology, CLIMATOLOGY_OUTPUT_FILE, "monthly climatology")
    save_parquet(stds, STD_OUTPUT_FILE, "anomaly standard deviations")

    print("--- Done ---")

if __name__ == "__main__":
    main()
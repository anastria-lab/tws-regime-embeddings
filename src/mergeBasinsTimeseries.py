# mergeBasinsTimeseries.py
# Step 4.2: Merge basin-level ERA5 and GRACE on a complete monthly calendar

import os
import xarray as xr
import pandas as pd
import numpy as np

ERA5_FILE = "data/interim/era5_basin_means_level04.nc"
GRACE_FILE = "data/interim/grace_basin_means_level04.nc"
OUTPUT_FILE = "data/interim/basin_era5_grace_merged.parquet"
QC_OUTPUT = "results/tables/merged_monthly_continuity_qc.csv"


def load_dataset(file_path, label):
    try:
        ds = xr.open_dataset(file_path)
        print(f"✅ Loaded {label}: {file_path}")
        return ds
    except Exception as e:
        print(f"❌ Failed to load {label}: {e}")
        return None


def prepare_dataset_dataframe(ds, dataset_name):
    """Convert a basin-level xarray dataset to one row per basin-month.

    The original source timestamp is preserved as ``time_<dataset_name>`` and a
    canonical month-start timestamp is stored in ``time`` for alignment.
    """
    if ds is None:
        raise ValueError(f"{dataset_name} dataset is None")
    if "time" not in ds.coords:
        raise ValueError(f"{dataset_name} dataset has no 'time' coordinate")
    if "basin" not in ds.coords:
        raise ValueError(f"{dataset_name} dataset has no 'basin' coordinate")

    var_names = list(ds.data_vars)
    df = ds[var_names].to_dataframe().reset_index()

    source_time_col = f"time_{dataset_name}"
    df[source_time_col] = pd.to_datetime(df["time"])
    df["time"] = df[source_time_col].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["time"].dt.year.astype(int)
    df["month"] = df["time"].dt.month.astype(int)

    key_cols = ["basin", "time"]
    dup = df.duplicated(key_cols, keep=False)
    if dup.any():
        example = df.loc[dup, key_cols].head(10)
        raise ValueError(
            f"{dataset_name} contains duplicate basin-month rows. Examples:\n{example}"
        )

    keep_cols = ["basin", "time", "year", "month", source_time_col] + var_names
    df = df[keep_cols].sort_values(["basin", "time"]).reset_index(drop=True)

    print(f"✅ Prepared {dataset_name} DataFrame")
    print(f"   Rows: {len(df):,}")
    print(f"   Basins: {df['basin'].nunique():,}")
    print(f"   Period: {df['time'].min().date()} to {df['time'].max().date()}")
    return df, var_names


def determine_common_time_envelope(era5_df, grace_df):
    """Return the shared monthly time envelope of the two source products."""
    start = max(era5_df["time"].min(), grace_df["time"].min())
    end = min(era5_df["time"].max(), grace_df["time"].max())
    if start > end:
        raise ValueError("ERA5 and GRACE have no overlapping monthly period.")
    return pd.Timestamp(start), pd.Timestamp(end)


def build_complete_monthly_calendar(era5_df, start, end):
    """Create a complete basin x month calendar using ERA5 basin coverage.

    ERA5 is used only as the basin-support reference. Missing source months are
    retained as explicit rows instead of being dropped by an inner merge.
    """
    basins = np.sort(era5_df["basin"].dropna().unique())
    months = pd.date_range(start=start, end=end, freq="MS")

    full_index = pd.MultiIndex.from_product(
        [basins, months], names=["basin", "time"]
    )
    calendar = full_index.to_frame(index=False)
    calendar["year"] = calendar["time"].dt.year.astype(int)
    calendar["month"] = calendar["time"].dt.month.astype(int)

    print("✅ Complete monthly calendar built")
    print(f"   Basins: {len(basins):,}")
    print(f"   Months: {len(months):,}")
    print(f"   Expected basin-month rows: {len(calendar):,}")
    return calendar


def merge_on_complete_calendar(era5_df, grace_df, era5_vars, grace_vars):
    """Align ERA5 and GRACE without deleting months that are missing in GRACE."""
    start, end = determine_common_time_envelope(era5_df, grace_df)
    print(f"✅ Common time envelope: {start.date()} to {end.date()}")

    era5_sub = era5_df[(era5_df["time"] >= start) & (era5_df["time"] <= end)].copy()
    grace_sub = grace_df[(grace_df["time"] >= start) & (grace_df["time"] <= end)].copy()

    calendar = build_complete_monthly_calendar(era5_sub, start, end)

    era5_keep = ["basin", "time", "time_era5"] + era5_vars
    grace_keep = ["basin", "time", "time_grace"] + grace_vars

    merged = calendar.merge(
        era5_sub[era5_keep], on=["basin", "time"], how="left", validate="one_to_one"
    )
    merged = merged.merge(
        grace_sub[grace_keep], on=["basin", "time"], how="left", validate="one_to_one"
    )

    merged = merged.sort_values(["basin", "time"]).reset_index(drop=True)

    print("✅ ERA5 and GRACE aligned on complete monthly calendar")
    print(f"   Rows: {len(merged):,}")
    return merged


def monthly_grid_is_complete(df):
    """Verify exactly one row per basin for every calendar month in the envelope."""
    duplicate_keys = int(df.duplicated(["basin", "time"]).sum())
    if duplicate_keys:
        raise ValueError(f"Found {duplicate_keys} duplicate basin-month rows.")

    expected_months = pd.date_range(df["time"].min(), df["time"].max(), freq="MS")
    counts = df.groupby("basin")["time"].nunique()
    bad_basins = counts[counts != len(expected_months)]

    if len(bad_basins):
        raise ValueError(
            f"Monthly calendar is incomplete for {len(bad_basins)} basins."
        )

    print("✅ Monthly continuity check passed for every basin")
    return True


def build_qc_table(df, era5_vars, grace_vars):
    """Summarize calendar continuity and source-variable missingness."""
    rows = []

    base = {
        "start_month": str(df["time"].min().date()),
        "end_month": str(df["time"].max().date()),
        "n_basins": int(df["basin"].nunique()),
        "n_months": int(df["time"].nunique()),
        "n_rows": int(len(df)),
    }

    for source, variables in [("ERA5", era5_vars), ("GRACE", grace_vars)]:
        for var in variables:
            row = dict(base)
            row.update({
                "source": source,
                "variable": var,
                "n_missing": int(df[var].isna().sum()),
                "missing_fraction": float(df[var].isna().mean()),
            })
            rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(df, qc):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(QC_OUTPUT), exist_ok=True)

    df.to_parquet(OUTPUT_FILE, index=False)
    qc.to_csv(QC_OUTPUT, index=False)

    print(f"✅ Saved merged monthly dataset: {OUTPUT_FILE}")
    print(f"✅ Saved monthly continuity QC: {QC_OUTPUT}")


def main():
    print("--- Merge basin ERA5 + GRACE on complete monthly calendar ---")

    era5_ds = load_dataset(ERA5_FILE, "ERA5")
    grace_ds = load_dataset(GRACE_FILE, "GRACE")
    if era5_ds is None or grace_ds is None:
        return

    era5_df, era5_vars = prepare_dataset_dataframe(era5_ds, "era5")
    grace_df, grace_vars = prepare_dataset_dataframe(grace_ds, "grace")

    merged = merge_on_complete_calendar(
        era5_df=era5_df,
        grace_df=grace_df,
        era5_vars=era5_vars,
        grace_vars=grace_vars,
    )
    monthly_grid_is_complete(merged)

    qc = build_qc_table(merged, era5_vars, grace_vars)
    print("\nMissingness summary:")
    print(qc[["source", "variable", "n_missing", "missing_fraction"]].to_string(index=False))

    save_outputs(merged, qc)
    print("--- Done ---")


if __name__ == "__main__":
    main()

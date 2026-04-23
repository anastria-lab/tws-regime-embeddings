import os
import xarray as xr
import pandas as pd

ERA5_FILE = "data/interim/era5_basin_means_level05.nc"
GRACE_FILE = "data/interim/grace_basin_means_level05.nc"
OUTPUT_FILE = "data/interim/basin_era5_grace_merged.parquet"


def load_dataset(file_path, label):
    """
    Load a NetCDF dataset.
    """
    try:
        ds = xr.open_dataset(file_path)
        print(f"✅ Loaded {label}: {file_path}")
        return ds
    except Exception as e:
        print(f"❌ Failed to load {label}: {e}")
        return None


def prepare_dataset_dataframe(ds, dataset_name):
    """
    Convert basin-level xarray dataset to a long pandas DataFrame
    and extract year/month from time.
    """
    try:
        if ds is None:
            raise ValueError(f"{dataset_name} dataset is None")

        if "time" not in ds.coords:
            raise ValueError(f"{dataset_name} dataset has no 'time' coordinate")

        if "basin" not in ds.coords:
            raise ValueError(f"{dataset_name} dataset has no 'basin' coordinate")

        # Keep only actual data variables
        var_names = list(ds.data_vars)

        df = ds[var_names].to_dataframe().reset_index()

        # Make sure time is datetime
        df["time"] = pd.to_datetime(df["time"])

        # Extract merge keys
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

        # Rename time to preserve source provenance if useful later
        df = df.rename(columns={"time": f"time_{dataset_name}"})

        print(f"✅ Prepared {dataset_name} DataFrame")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"❌ Failed to prepare {dataset_name} DataFrame: {e}")
        return None


def merge_era5_and_grace(era5_df, grace_df):
    """
    Merge ERA5 and GRACE basin-level time series by basin, year, month.
    """
    try:
        if era5_df is None or grace_df is None:
            raise ValueError("One or both input DataFrames are None")

        merge_keys = ["basin", "year", "month"]

        merged = pd.merge(
            era5_df,
            grace_df,
            on=merge_keys,
            how="inner",
            suffixes=("_era5", "_grace")
        )

        print("✅ Merged ERA5 and GRACE")
        print(f"   Rows: {len(merged):,}")
        print(f"   Columns: {list(merged.columns)}")

        if "time_era5" in merged.columns and "time_grace" in merged.columns:
            print("\nSample merged time columns:")
            print(merged[["time_era5", "time_grace", "year", "month"]].head())

        return merged

    except Exception as e:
        print(f"❌ Failed to merge ERA5 and GRACE: {e}")
        return None


def save_merged_dataframe(df, output_file):
    """
    Save merged DataFrame to parquet.
    """
    try:
        if df is None:
            raise ValueError("Merged DataFrame is None")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_parquet(output_file, index=False)

        print(f"✅ Saved merged dataset to: {output_file}")
        return output_file

    except Exception as e:
        print(f"❌ Failed to save merged dataset: {e}")
        return None


def main():
    print("--- Merge basin ERA5 + GRACE time series ---")

    era5_ds = load_dataset(ERA5_FILE, "ERA5")
    grace_ds = load_dataset(GRACE_FILE, "GRACE")

    if era5_ds is None or grace_ds is None:
        return

    era5_df = prepare_dataset_dataframe(era5_ds, "era5")
    grace_df = prepare_dataset_dataframe(grace_ds, "grace")

    if era5_df is None or grace_df is None:
        return

    merged_df = merge_era5_and_grace(era5_df, grace_df)

    if merged_df is None:
        return

    save_merged_dataframe(merged_df, OUTPUT_FILE)

    print("--- Done ---")


if __name__ == "__main__":
    main()
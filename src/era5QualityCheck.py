# era5QualityCheck.py
# Step 2.2: Load and perform quality checks on ERA5 data. Export years and months for later merging with GRACE data.

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys

FILE_PATH = "data/raw/era5/data_stream-moda.nc"


def load_era5_data(file_path, chunks=None):
    """
    Load ERA5 data from NetCDF.
    """
    try:
        if chunks:
            ds = xr.open_dataset(file_path, chunks=chunks)
        else:
            ds = xr.open_dataset(file_path)
        return ds
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def get_data_info(ds):
    """
    Print dataset summary and variable metadata.
    """
    if ds is None:
        print("Dataset is None")
        return

    try:
        print("=" * 60)
        print("ERA5 Dataset Information")
        print("=" * 60)
        print(ds)

        print("\n" + "=" * 60)
        print("Variables:")
        print("=" * 60)
        for var in ds.data_vars:
            print(f"\n{var}:")
            print(f"  Dims: {ds[var].dims}")
            print(f"  Shape: {ds[var].shape}")
            print(f"  Dtype: {ds[var].dtype}")
            print(f"  Attributes: {ds[var].attrs}")
    except Exception as e:
        print(f"Error getting data info: {e}")


def list_variables(ds):
    """
    Print available variable names.
    """
    if ds is None:
        print("Dataset is None")
        return []

    vars_list = list(ds.data_vars)
    print("\nAvailable variables:")
    for var in vars_list:
        print(f"  - {var}")
    return vars_list


def _get_time_name(ds):
    if "valid_time" in ds.coords:
        return "valid_time"
    if "time" in ds.coords:
        return "time"
    return None


def get_available_year_months(ds):
    """
    Return sorted available (year, month) pairs from the dataset time coordinate.
    """
    if ds is None:
        print("Dataset is None")
        return []

    try:
        time_name = _get_time_name(ds)
        if time_name is None:
            print("No time coordinate found.")
            return []

        times = ds[time_name].values
        year_months = sorted(set((t.astype("datetime64[M]").astype(object).year,
                                  t.astype("datetime64[M]").astype(object).month)
                                 for t in times))
        return year_months
    except Exception as e:
        print(f"Error getting available year/month pairs: {e}")
        return []


def find_time_index(ds, year, month):
    """
    Find the index of the timestep matching the given year and month.
    """
    if ds is None:
        print("Dataset is None")
        return None

    try:
        time_name = _get_time_name(ds)
        if time_name is None:
            print("No time coordinate found.")
            return None

        times = ds[time_name].values

        for i, t in enumerate(times):
            dt = t.astype("datetime64[M]").astype(object)
            if dt.year == year and dt.month == month:
                return i

        print(f"No timestep found for {year}-{month:02d}.")
        return None

    except Exception as e:
        print(f"Error finding time index: {e}")
        return None


def check_data_quality(ds, sample_time_index=0):
    """
    Compute lightweight quality statistics on a single timestep per variable.
    """
    if ds is None:
        print("Dataset is None")
        return None

    report = {}

    try:
        time_name = _get_time_name(ds)

        print("\n" + "=" * 60)
        print("Lightweight Data Quality Report (single timestep)")
        print("=" * 60)

        for var in ds.data_vars:
            data = ds[var]

            if time_name is not None and time_name in data.dims:
                if sample_time_index >= data.sizes[time_name]:
                    print(f"Skipping {var}: sample_time_index out of range")
                    continue
                sample = data.isel({time_name: sample_time_index})
            else:
                sample = data

            sample = sample.load()

            total_values = int(sample.size)
            missing_values = int(sample.isnull().sum().values)
            valid_values = total_values - missing_values
            missing_pct = (missing_values / total_values * 100.0) if total_values > 0 else 0.0

            if valid_values > 0:
                min_val = float(sample.min(skipna=True).values)
                max_val = float(sample.max(skipna=True).values)
                mean_val = float(sample.mean(skipna=True).values)
                std_val = float(sample.std(skipna=True).values)
            else:
                min_val = np.nan
                max_val = np.nan
                mean_val = np.nan
                std_val = np.nan

            report[var] = {
                "total_values": total_values,
                "valid_values": valid_values,
                "missing_values": missing_values,
                "missing_percentage": missing_pct,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
            }

            print(f"\n{var}:")
            print(f"  Total values: {total_values}")
            print(f"  Valid values: {valid_values}")
            print(f"  Missing values: {missing_values}")
            print(f"  Missing percentage: {missing_pct:.2f}%")
            print(f"  Min: {min_val:.4f}" if not np.isnan(min_val) else "  Min: NaN")
            print(f"  Max: {max_val:.4f}" if not np.isnan(max_val) else "  Max: NaN")
            print(f"  Mean: {mean_val:.4f}" if not np.isnan(mean_val) else "  Mean: NaN")
            print(f"  Std Dev: {std_val:.4f}" if not np.isnan(std_val) else "  Std Dev: NaN")

        return report

    except Exception as e:
        print(f"Error performing quality checks: {e}")
        return None


def plot_variable_map(ds, variable, year, month):
    """
    Plot one variable for a given year and month directly from xarray.
    """
    if ds is None:
        print("Dataset is None")
        return None

    try:
        if variable not in ds.data_vars:
            raise ValueError(f"Variable '{variable}' not found in dataset.")

        time_name = _get_time_name(ds)
        data = ds[variable]

        if time_name is not None and time_name in data.dims:
            time_index = find_time_index(ds, year, month)
            if time_index is None:
                return None

            plot_data = data.isel({time_name: time_index}).load()
            plot_time = ds[time_name].values[time_index]
            title = f"{variable} at {str(plot_time)}"
        else:
            plot_data = data.load()
            title = variable

        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        plot_data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            cbar_kwargs={"label": variable}
        )

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.set_title(title)
        plt.show()

        return plot_data

    except Exception as e:
        print(f"Error plotting variable map: {e}")
        return None


def ask_user(question):
    """Asks a yes/no question via input and returns a boolean."""
    check = input(f"{question} (y/n): ").lower().strip()
    try:
        if check[0] == "y":
            return True
        elif check[0] == "n":
            return False
        else:
            print("Invalid input, please enter 'y' or 'n'.")
            return ask_user(question)
    except Exception:
        return False


def ask_for_variable(ds):
    variables = list_variables(ds)

    while True:
        var = input("Enter variable name to plot: ").strip()
        if var in variables:
            return var
        print("Invalid variable name. Please choose one from the list above.")


def ask_for_year():
    while True:
        try:
            year = int(input("Enter year (e.g. 2023): ").strip())
            return year
        except ValueError:
            print("Invalid year. Please enter an integer.")


def ask_for_month():
    while True:
        try:
            month = int(input("Enter month (1-12): ").strip())
            if 1 <= month <= 12:
                return month
            print("Month must be between 1 and 12.")
        except ValueError:
            print("Invalid month. Please enter an integer.")


def main():
    print("--- 🌍 TWS Regime Embeddings Pipeline ---")

    if ask_user("Do you need to update your ERA5 data?"):
        print("🚀 Starting downloader...")
        try:
            era5Downloader.download_era5_data("era5_2002_2026.zip")
            print("✅ Data update complete.")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            sys.exit(1)
    else:
        print("⏭️ Skipping download. Using existing data.")

    print("📂 Loading ERA5 data...")
    era5_data = load_era5_data(FILE_PATH)

    if era5_data is None:
        print("❌ Failed to load ERA5 data.")
        sys.exit(1)

    print("✅ ERA5 data loaded.")

    if ask_user("Do you want to inspect dataset info?"):
        get_data_info(era5_data)

    if ask_user("Do you want a lightweight quality check on one timestep?"):
        print("🔍 Running lightweight quality check...")
        try:
            check_data_quality(era5_data, sample_time_index=0)
            print("✅ Quality check complete.")
        except Exception as e:
            print(f"❌ Quality check failed: {e}")

    while ask_user("Do you want to plot a variable for a chosen year and month?"):
        try:
            variable = ask_for_variable(era5_data)
            year = ask_for_year()
            month = ask_for_month()

            print(f"🗺️ Plotting {variable} for {year}-{month:02d}...")
            result = plot_variable_map(
                era5_data,
                variable=variable,
                year=year,
                month=month
            )

            if result is not None:
                print("✅ Plot complete.")
            else:
                print("❌ Plot failed or no matching timestep found.")

        except Exception as e:
            print(f"❌ Plot failed: {e}")

    print("--- Proceeding to Analysis ---")


if __name__ == "__main__":
    main()
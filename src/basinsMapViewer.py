# basin_map_viewer.py
# Geographic map viewer for basin polygons or basin-aggregated weighted means from ERA5 / GRACE

import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import TwoSlopeNorm, Normalize

ERA5_NETCDF_FILE = "data/interim/era5_basin_means_level05.nc"
GRACE_NETCDF_FILE = "data/interim/grace_basin_means_level05.nc"
BASINS_FILE = "data/interim/hydrobasins_l05_global.gpkg"
BASIN_ID_COLUMN = "HYBAS_ID"


def ask_user(question):
    """Ask a yes/no question and return True/False."""
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


def choose_mode():
    """Ask the user to choose ERA5, GRACE, or basins-only mode."""
    while True:
        choice = input("Choose what to plot ('era5', 'grace', or 'basins'): ").strip().lower()
        if choice in ["era5", "grace", "basins"]:
            return choice
        print("Invalid choice. Please type 'era5', 'grace', or 'basins'.")


def get_netcdf_path(dataset_choice):
    """Return the NetCDF path for the selected dataset."""
    if dataset_choice == "era5":
        return ERA5_NETCDF_FILE
    elif dataset_choice == "grace":
        return GRACE_NETCDF_FILE
    else:
        raise ValueError(f"Unknown dataset choice: {dataset_choice}")


def load_basin_means(nc_path):
    try:
        ds = xr.open_dataset(nc_path)
        print(f"✅ Loaded NetCDF: {nc_path}")
        return ds
    except Exception as e:
        print(f"❌ Failed to load NetCDF: {e}")
        return None


def load_basins(gpkg_path):
    try:
        gdf = gpd.read_file(gpkg_path)
        print(f"✅ Loaded GeoPackage: {gpkg_path}")
        return gdf
    except Exception as e:
        print(f"❌ Failed to load GeoPackage: {e}")
        return None


def detect_time_name(ds):
    if "time" in ds.coords:
        return "time"
    if "valid_time" in ds.coords:
        return "valid_time"
    return None


def find_time_index(ds, year, month):
    """
    Find the timestep matching the requested year and month.
    Works with normal datetime64 and cftime-like objects.
    """
    time_name = detect_time_name(ds)
    if time_name is None:
        print("❌ No time coordinate found in NetCDF.")
        return None

    times = ds[time_name].values

    matches = []
    for i, t in enumerate(times):
        try:
            t_year = t.year
            t_month = t.month
        except AttributeError:
            ts = pd.Timestamp(t)
            t_year = ts.year
            t_month = ts.month

        if t_year == year and t_month == month:
            matches.append(i)

    if not matches:
        print(f"❌ No timestep found for {year}-{month:02d}")
        return None

    return matches[0]


def inspect_ids(ds, gdf):
    print("\n" + "=" * 60)
    print("ID CHECK")
    print("=" * 60)

    if "basin" not in ds.coords:
        print("❌ NetCDF has no 'basin' coordinate.")
        return

    print("First 10 NetCDF basin ids:")
    print(ds["basin"].values[:10])

    if BASIN_ID_COLUMN not in gdf.columns:
        print(f"❌ GeoPackage has no '{BASIN_ID_COLUMN}' column.")
        print("Available columns:")
        print(list(gdf.columns))
        return

    print(f"\nFirst 10 GeoPackage {BASIN_ID_COLUMN} values:")
    print(gdf[BASIN_ID_COLUMN].values[:10])


def print_available_variables(ds):
    print("\nAvailable variables:")
    for var in ds.data_vars:
        print(f"  - {var}")


def ask_for_variable(ds):
    variables = list(ds.data_vars)

    while True:
        variable = input("\nEnter variable to plot: ").strip()
        if variable in variables:
            return variable
        print("Invalid variable. Please choose one from the list above.")


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

def print_available_times(ds, max_show=24):
    """
    Print available time values and year-month pairs.
    """
    time_name = detect_time_name(ds)
    if time_name is None:
        print("❌ No time coordinate found in NetCDF.")
        return

    times = ds[time_name].values

    print("\nAvailable raw time values:")
    for t in times[:max_show]:
        print(f"  - {t}")

    year_months = []
    for t in times:
        try:
            y, m = t.year, t.month
        except AttributeError:
            ts = pd.Timestamp(t)
            y, m = ts.year, ts.month
        year_months.append((y, m))

    year_months = sorted(set(year_months))

    print("\nAvailable year-month pairs:")
    for ym in year_months[:max_show]:
        print(f"  - {ym[0]}-{ym[1]:02d}")

    if len(year_months) > max_show:
        print(f"  ... and {len(year_months) - max_show} more")

def build_plot_gdf(ds, gdf, variable, year, month):
    if variable not in ds.data_vars:
        raise ValueError(f"Variable '{variable}' not found in NetCDF.")

    if BASIN_ID_COLUMN not in gdf.columns:
        raise ValueError(
            f"Column '{BASIN_ID_COLUMN}' not found in GeoPackage. "
            f"Available columns: {list(gdf.columns)}"
        )

    time_name = detect_time_name(ds)
    da = ds[variable]

    if time_name and time_name in da.dims:
        time_index = find_time_index(ds, year, month)
        if time_index is None:
            return None, None

        sliced = da.isel({time_name: time_index})
        selected_time = pd.to_datetime(ds[time_name].values[time_index])
    else:
        sliced = da
        selected_time = None

    values_df = pd.DataFrame({
        BASIN_ID_COLUMN: ds["basin"].values,
        variable: sliced.values
    })

    values_df[BASIN_ID_COLUMN] = values_df[BASIN_ID_COLUMN].astype(str)

    plot_gdf = gdf.copy()
    plot_gdf[BASIN_ID_COLUMN] = plot_gdf[BASIN_ID_COLUMN].astype(str)

    plot_gdf = plot_gdf.merge(values_df, on=BASIN_ID_COLUMN, how="left")

    return plot_gdf, selected_time


def plot_basin_map(plot_gdf, dataset_name, variable, selected_time=None, robust=True):
    if plot_gdf is None:
        return

    values = plot_gdf[variable].dropna()

    if len(values) == 0:
        print("❌ No valid values to plot.")
        return

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    # Choose robust limits
    if robust:
        vmin = values.quantile(0.02)
        vmax = values.quantile(0.98)
    else:
        vmin = values.min()
        vmax = values.max()

    # Use diverging colors if data crosses zero
    if vmin < 0 < vmax:
        cmap = "RdBu_r"
        max_abs = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    else:
        cmap = "viridis"
        norm = Normalize(vmin=vmin, vmax=vmax)

    plot_gdf.plot(
        column=variable,
        ax=ax,
        legend=True,
        cmap=cmap,
        norm=norm,
        missing_kwds={
            "color": "lightgrey",
            "label": "No data"
        },
        linewidth=0.1,
        edgecolor="black"
    )

    # plot_gdf.plot(n
    #     column=variable,
    #     ax=ax,
    #     legend=True,
    #     cmap="RdBu_r" if values.min() < 0 < values.max() else "viridis",
    #     scheme="quantiles",
    #     k=7,
    #     missing_kwds={
    #         "color": "lightgrey",
    #         "label": "No data"
    #     },
    #     linewidth=0.1,
    #     edgecolor="black"
    # )

    if selected_time is not None:
        title = f"{dataset_name.upper()} - {variable} by basin - {selected_time.year}-{selected_time.month:02d}"
    else:
        title = f"{dataset_name.upper()} - {variable} by basin"

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_basins_only(gdf):
    """Plot just the basin polygons."""
    if gdf is None:
        return

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    gdf.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=0.2
    )

    ax.set_title("HydroBASINS Level 05")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def main():
    print("--- Basin Map Viewer ---")

    gdf = load_basins(BASINS_FILE)
    if gdf is None:
        return
    

    inspected_ids = False

    while True:
        mode = choose_mode()

        if mode == "basins":
            try:
                print("\n🗺️ Plotting basin polygons...")
                plot_basins_only(gdf)
            except Exception as e:
                print(f"❌ Failed to plot basins: {e}")

            if not ask_user("Do you want to plot another map"):
                break
            continue

        nc_path = get_netcdf_path(mode)
        ds = load_basin_means(nc_path)

        print_available_times(ds)

        if ds is None:
            if not ask_user("Do you want to try another dataset"):
                break
            continue

        if not inspected_ids:
            inspect_ids(ds, gdf)
            inspected_ids = True

        print_available_variables(ds)

        try:
            variable = ask_for_variable(ds)
            year = ask_for_year()
            month = ask_for_month()

            print(f"\n🗺️ Building map for {mode.upper()} | {variable} | {year}-{month:02d} ...")
            plot_gdf, selected_time = build_plot_gdf(ds, gdf, variable, year, month)

            if plot_gdf is None:
                print("❌ Failed to create plot GeoDataFrame.")
            else:
                matched = plot_gdf[variable].notna().sum()
                total = len(plot_gdf)
                print(f"✅ Basins with matched values: {matched}/{total}")

                plot_basin_map(plot_gdf, mode, variable, selected_time)

        except Exception as e:
            print(f"❌ Failed to build or plot map: {e}")

        if not ask_user("Do you want to plot another map"):
            break

    print("--- Exiting Basin Map Viewer ---")


if __name__ == "__main__":
    main()
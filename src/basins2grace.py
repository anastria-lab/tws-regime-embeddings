# basins2grace.py
# Step 3.1: Aggregate GRACE CSR Mascon LWE data to HydroBASINS level 04 basin means
#
# IMPORTANT: CSR GRACE/GRACE-FO contains documented "special months" where
# the solution center date lies in the previous calendar month.  We therefore
# preserve the raw solution-center timestamp for provenance but assign the
# canonical `time` coordinate to the nominal GRACE month before basin
# aggregation.

import os
import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask
import pandas as pd

# =========================
# CONFIG
# =========================
GRACE_FILE = "data/raw/grace/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
BASINS_FILE = "data/interim/hydrobasins_l04_global.gpkg"
OUTPUT_FILE = "data/interim/grace_basin_means_level04.nc"

VARIABLES = [
    "lwe_thickness"
]

BASIN_ID_COLUMN = "HYBAS_ID"

# Documented special cases for CSR/GFZ GRACE month assignment.
# The keys are the calendar month containing the solution CENTER date.
# The values are the corresponding nominal GRACE months, in chronological
# solution order.  These are the two duplicate-center-month cases relevant
# to CSR RL06-style monthly solutions.
CSR_SPECIAL_CENTER_MONTHS = {
    "2011-10": ["2011-10", "2011-11"],  # nominal Nov 2011 is centered in Oct
    "2015-04": ["2015-04", "2015-05"],  # nominal May 2015 is centered in Apr
}


def _decode_grace_center_dates(time_values):
    """Decode CSR time values expressed as days since 2002-01-01."""
    return pd.DatetimeIndex(
        pd.to_datetime(time_values, origin="2002-01-01", unit="D")
    )


def assign_nominal_grace_months(center_dates):
    """
    Convert CSR solution-center dates to canonical nominal GRACE months.

    Most solutions map directly to the calendar month containing their center
    date.  CSR has documented special cases where a nominal monthly solution
    is centered in the previous calendar month.  We resolve those cases before
    enforcing uniqueness.

    Returns
    -------
    pd.DatetimeIndex
        First day of each nominal GRACE calendar month.
    """
    center_dates = pd.DatetimeIndex(center_dates)

    # Mutable month labels, one per GRACE solution.
    nominal = pd.Series(
        center_dates.to_period("M").astype(str),
        index=np.arange(len(center_dates)),
        dtype="object",
    )

    for center_month, target_months in CSR_SPECIAL_CENTER_MONTHS.items():
        idx = np.flatnonzero(nominal.to_numpy() == center_month)

        if len(idx) == 0:
            # Supports truncated GRACE files that may not contain the case.
            continue

        if len(idx) != len(target_months):
            raise ValueError(
                f"CSR special-month correction expected {len(target_months)} "
                f"solutions centered in {center_month}, found {len(idx)}. "
                "Inspect the GRACE time coordinate before continuing."
            )

        # Ensure chronological solution order before applying nominal labels.
        idx = idx[np.argsort(center_dates[idx].values)]
        for i, target_month in zip(idx, target_months):
            nominal.iloc[i] = target_month

        centers = ", ".join(str(center_dates[i].date()) for i in idx)
        print(
            f"✅ Corrected CSR special center month {center_month}: "
            f"centers [{centers}] -> nominal months {target_months}"
        )

    # No nominal GRACE month may remain duplicated.
    duplicate_mask = nominal.duplicated(keep=False)
    if duplicate_mask.any():
        bad = pd.DataFrame({
            "solution_center_time": center_dates[duplicate_mask.to_numpy()],
            "nominal_month": nominal[duplicate_mask].to_numpy(),
        })
        raise ValueError(
            "Unresolved duplicate nominal GRACE months after applying known "
            f"CSR special-month corrections. Examples:\n{bad.head(20).to_string(index=False)}"
        )

    nominal_time = pd.DatetimeIndex(
        pd.PeriodIndex(nominal.to_numpy(), freq="M").to_timestamp(how="start")
    )

    if not nominal_time.is_monotonic_increasing:
        raise ValueError("Canonical nominal GRACE months are not monotonically increasing.")

    return nominal_time


def load_grace_data(file_path):
    """
    Load GRACE NetCDF, decode solution-center time, and assign nominal months.
    """
    try:
        ds = xr.open_dataset(file_path)
        print(f"✅ GRACE loaded: {file_path}")

        if "time" not in ds.coords:
            raise ValueError("GRACE dataset has no 'time' coordinate.")

        # Manual decoding because this CSR file uses a non-standard time-unit
        # attribute capitalization in some releases.
        center_dates = _decode_grace_center_dates(ds["time"].values)
        nominal_time = assign_nominal_grace_months(center_dates)

        # Preserve the original physical solution-center epoch for provenance.
        # Use canonical first-of-month timestamps for all downstream monthly work.
        ds = ds.assign_coords(
            time=("time", nominal_time.values),
            solution_center_time=("time", center_dates.values),
        )

        print("✅ GRACE solution-center dates decoded")
        print("✅ Canonical nominal GRACE months assigned")
        print(f"   Solutions: {len(nominal_time):,}")
        print(f"   Nominal period: {nominal_time.min().date()} to {nominal_time.max().date()}")
        print(f"   Duplicate nominal months: {nominal_time.duplicated().sum()}")

        return ds

    except Exception as e:
        print(f"❌ Failed to load GRACE data: {e}")
        return None


def load_basins(gpkg_path, basin_id_column=BASIN_ID_COLUMN):
    """Load basin polygons from GeoPackage."""
    try:
        gdf = gpd.read_file(gpkg_path)

        if basin_id_column not in gdf.columns:
            raise ValueError(
                f"Basin ID column '{basin_id_column}' not found. "
                f"Available columns: {list(gdf.columns)}"
            )

        gdf = gdf[[basin_id_column, "geometry"]].copy()
        gdf = gdf.dropna(subset=["geometry"]).reset_index(drop=True)

        print(f"✅ Basins loaded: {gpkg_path}")
        print(f"   Number of basins: {len(gdf)}")
        return gdf

    except Exception as e:
        print(f"❌ Failed to load basins: {e}")
        return None


def detect_coord_names(ds):
    """Detect longitude / latitude / time names."""
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    lat_name = "lat" if "lat" in ds.coords else "latitude"

    if "time" in ds.coords:
        time_name = "time"
    elif "valid_time" in ds.coords:
        time_name = "valid_time"
    else:
        raise ValueError("No time coordinate found.")

    if lon_name not in ds.coords or lat_name not in ds.coords:
        raise ValueError("Longitude / latitude coordinates not found.")

    return lon_name, lat_name, time_name


def normalize_longitudes(ds, lon_name):
    """Convert longitudes to [-180, 180)."""
    lon = ds[lon_name].values
    lon_fixed = ((lon + 180) % 360) - 180
    ds = ds.assign_coords({lon_name: lon_fixed})
    ds = ds.sortby(lon_name)

    lon_min = float(ds[lon_name].min().values)
    lon_max = float(ds[lon_name].max().values)
    print(f"ℹ️ GRACE longitude range after normalization: {lon_min:.3f} .. {lon_max:.3f}")
    return ds


def ensure_basins_crs(gdf):
    """Ensure basin CRS is EPSG:4326."""
    if gdf.crs is None:
        print("⚠️ Basin CRS missing. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        print("ℹ️ Reprojected basins to EPSG:4326")
    return gdf


def subset_variables(ds, variables):
    """Keep only requested data variables that exist."""
    existing = [v for v in variables if v in ds.data_vars]
    missing = [v for v in variables if v not in ds.data_vars]

    if missing:
        print("⚠️ Variables not found and skipped:")
        for v in missing:
            print(f"   - {v}")

    if not existing:
        raise ValueError(
            f"None of requested variables found. Available: {list(ds.data_vars)}"
        )

    print("✅ Variables to process:")
    for v in existing:
        print(f"   - {v}")

    # Coordinates such as solution_center_time are retained by xarray.
    return ds[existing], existing


def build_basin_mask(ds, basins_gdf, lon_name, lat_name):
    """Build 2D basin mask."""
    basins_gdf = basins_gdf.reset_index(drop=True).copy()

    lon_min = float(ds[lon_name].min().values)
    lon_max = float(ds[lon_name].max().values)
    print(f"ℹ️ Mask input longitude range: {lon_min:.3f} .. {lon_max:.3f}")

    mask = regionmask.mask_geopandas(
        basins_gdf,
        ds[lon_name],
        ds[lat_name],
        numbers=None,
        wrap_lon=False,
    )

    print("✅ Basin mask created")
    return mask


def build_latitude_weights(ds, lat_name, lon_name):
    """Build 2D cosine-latitude weights for a regular lat/lon grid."""
    lat_weights_1d = xr.DataArray(
        np.cos(np.deg2rad(ds[lat_name].values)),
        coords={lat_name: ds[lat_name]},
        dims=(lat_name,),
        name="lat_weights",
    )

    ones_lon = xr.DataArray(
        np.ones(ds[lon_name].size),
        coords={lon_name: ds[lon_name]},
        dims=(lon_name,),
    )

    weights_2d = lat_weights_1d * ones_lon
    weights_2d.name = "weights"
    return weights_2d


def compute_basin_means(ds, mask, basins_gdf, lon_name, lat_name, time_name):
    """Compute cosine-latitude weighted basin means for every timestamp."""
    basin_ids = basins_gdf[BASIN_ID_COLUMN].values
    results = {}

    print("🚀 Computing weighted basin means...")

    weights_2d = build_latitude_weights(ds, lat_name, lon_name)

    spatial_dim = "cell"
    mask_1d = mask.stack({spatial_dim: (lat_name, lon_name)})
    weights_1d = weights_2d.stack({spatial_dim: (lat_name, lon_name)})

    valid_cells = mask_1d.notnull()
    mask_1d = mask_1d.where(valid_cells, drop=True)
    weights_1d = weights_1d.where(valid_cells, drop=True)

    basin_index = np.arange(len(basin_ids))

    for var in ds.data_vars:
        print(f"   Processing {var}...")
        da = ds[var]

        da_1d = da.stack({spatial_dim: (lat_name, lon_name)})
        da_1d = da_1d.where(valid_cells, drop=True)

        weighted_data = da_1d * weights_1d
        numerator = weighted_data.groupby(mask_1d).sum(skipna=True)

        valid_weights = weights_1d.where(da_1d.notnull())
        denominator = valid_weights.groupby(mask_1d).sum(skipna=True)

        grouped = numerator / denominator

        non_time_dims = [d for d in grouped.dims if d != time_name]
        if len(non_time_dims) != 1:
            raise ValueError(f"Unexpected grouped dims for {var}: {grouped.dims}")

        basin_dim = non_time_dims[0]
        if basin_dim != "basin":
            grouped = grouped.rename({basin_dim: "basin"})

        grouped = grouped.reindex(basin=basin_index)
        grouped = grouped.assign_coords(basin=("basin", basin_ids))
        results[var] = grouped

    out = xr.Dataset(results)
    if time_name in ds.coords:
        out = out.assign_coords({time_name: ds[time_name]})

    # Preserve source solution-center timestamps in the basin product.
    if "solution_center_time" in ds.coords:
        out = out.assign_coords(solution_center_time=ds["solution_center_time"])

    if time_name in out.dims and time_name != "time":
        out = out.rename({time_name: "time"})

    # Final monthly uniqueness assertion before writing anything downstream.
    time_index = pd.DatetimeIndex(out["time"].values)
    if time_index.duplicated().any():
        duplicated = time_index[time_index.duplicated(keep=False)]
        raise ValueError(
            f"Duplicate nominal GRACE months remain after aggregation: {duplicated}"
        )

    out["basin"].attrs["long_name"] = "HydroBASINS level 04 basin id"
    out["time"].attrs["long_name"] = "Nominal GRACE/GRACE-FO calendar month"
    out["solution_center_time"].attrs["long_name"] = (
        "Original CSR GRACE/GRACE-FO solution center epoch"
    )
    out.attrs["source_basins_file"] = BASINS_FILE
    out.attrs["description"] = (
        "Weighted mean values per HydroBASINS level 04 basin and nominal GRACE month "
        "using cosine(latitude) area weights. Original solution-center epochs are "
        "preserved in solution_center_time."
    )

    print("✅ Weighted basin means computed")
    print("✅ Nominal GRACE month uniqueness check passed")
    return out


def save_to_netcdf(ds, output_file):
    """Save output dataset."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
        ds.to_netcdf(output_file, encoding=encoding)

        print(f"✅ Output saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"❌ Failed to save output: {e}")
        return None


def main():
    print("--- Basin to GRACE aggregation (nominal-month safe) ---")

    ds = load_grace_data(GRACE_FILE)
    if ds is None:
        return

    basins = load_basins(BASINS_FILE)
    if basins is None:
        return

    lon_name, lat_name, time_name = detect_coord_names(ds)

    ds = normalize_longitudes(ds, lon_name)
    basins = ensure_basins_crs(basins)

    ds, _ = subset_variables(ds, VARIABLES)
    mask = build_basin_mask(ds, basins, lon_name, lat_name)

    basin_means = compute_basin_means(
        ds=ds,
        mask=mask,
        basins_gdf=basins,
        lon_name=lon_name,
        lat_name=lat_name,
        time_name=time_name,
    )

    save_to_netcdf(basin_means, OUTPUT_FILE)
    print("--- Done ---")


if __name__ == "__main__":
    main()

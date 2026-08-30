# aggregateSPEI12ToBasins.py
# Validation A1: Aggregate independent SPEIbase SPEI-12 to HydroBASINS Level 4.
#
# Expected local input: a global SPEI-12 NetCDF downloaded from SPEIbase.
# The script searches several common filenames under data/raw/spei/.
#
# Output:
#   data/interim/spei12_basin_means_level04.nc
#   results/tables/spei12_basin_aggregation_qc.csv
#
# Notes:
# - Basin aggregation is cosine(latitude)-weighted across SPEI grid-cell centers
#   whose centers fall inside each HydroBASINS L4 polygon.
# - Small basins with no 0.5-degree cell center are left missing and reported
#   rather than silently assigned a nearest grid cell.
# - Only the GRACE-era validation period is loaded to keep memory moderate.

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask

BASINS_FILE = "data/interim/hydrobasins_l04_global.gpkg"
BASIN_ID_COLUMN = "HYBAS_ID"

SPEI_CANDIDATES = [
    "data/raw/spei/spei12.nc",
    "data/raw/spei/spei_12.nc",
    "data/raw/spei/spei12_v2.11.nc",
    "data/raw/spei/spei_12_v2.11.nc",
    "data/raw/spei/SPEI12.nc",
]

OUTPUT_FILE = "data/interim/spei12_basin_means_level04.nc"
QC_OUTPUT = "results/tables/spei12_basin_aggregation_qc.csv"

START_MONTH = "2002-01-01"
END_MONTH = "2024-12-01"


def find_spei_file():
    for f in SPEI_CANDIDATES:
        if os.path.exists(f):
            print(f"✅ Found SPEI-12 file: {f}")
            return f

    folder = Path("data/raw/spei")
    if folder.exists():
        nc_files = sorted(folder.glob("*.nc"))
        if len(nc_files) == 1:
            f = str(nc_files[0])
            print(f"✅ Using the only NetCDF found in data/raw/spei/: {f}")
            return f

    searched = "\n".join(f"  - {x}" for x in SPEI_CANDIDATES)
    raise FileNotFoundError(
        "No SPEI-12 NetCDF found.\n"
        "Download SPEIbase SPEI-12 and place it under data/raw/spei/.\n"
        f"Searched:\n{searched}"
    )


def detect_coords(ds):
    lon_candidates = ["lon", "longitude", "x"]
    lat_candidates = ["lat", "latitude", "y"]
    time_candidates = ["time", "valid_time"]

    lon = next((c for c in lon_candidates if c in ds.coords), None)
    lat = next((c for c in lat_candidates if c in ds.coords), None)
    time = next((c for c in time_candidates if c in ds.coords), None)

    if lon is None or lat is None or time is None:
        raise ValueError(
            f"Could not detect lon/lat/time coordinates. Coordinates: {list(ds.coords)}"
        )
    return lon, lat, time


def detect_spei_variable(ds, lon, lat, time):
    preferred = [
        "spei",
        "SPEI",
        "spei12",
        "spei_12",
        "SPEI_12_month",
        "spei_12_month",
    ]

    for v in preferred:
        if v in ds.data_vars:
            dims = set(ds[v].dims)
            if {lon, lat, time}.issubset(dims):
                return v

    candidates = []
    for v in ds.data_vars:
        dims = set(ds[v].dims)
        if {lon, lat, time}.issubset(dims):
            candidates.append(v)

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Could not uniquely identify the SPEI variable. "
        f"Candidate 3-D variables: {candidates}; all data variables: {list(ds.data_vars)}"
    )


def month_start_index(values):
    out = []
    for x in values:
        try:
            ts = pd.Timestamp(x)
        except Exception:
            ts = pd.Timestamp(str(x)[:10])
        out.append(pd.Timestamp(ts.year, ts.month, 1))
    return pd.DatetimeIndex(out)


def normalize_longitudes(ds, lon_name):
    lon = ds[lon_name].values.astype(float)
    fixed = ((lon + 180.0) % 360.0) - 180.0
    ds = ds.assign_coords({lon_name: fixed}).sortby(lon_name)
    return ds


def load_spei(path):
    ds = xr.open_dataset(path, decode_times=True)
    lon, lat, time = detect_coords(ds)
    var = detect_spei_variable(ds, lon, lat, time)

    print(f"✅ Coordinates: lon={lon}, lat={lat}, time={time}")
    print(f"✅ SPEI variable: {var}")

    # Canonical month-start time coordinate.
    canonical_time = month_start_index(ds[time].values)
    if canonical_time.duplicated().any():
        dup = canonical_time[canonical_time.duplicated()].unique()
        raise ValueError(f"Duplicate SPEI months detected: {dup[:10].tolist()}")

    ds = ds.assign_coords({time: canonical_time})
    ds = normalize_longitudes(ds, lon)

    da = ds[var].sel({time: slice(START_MONTH, END_MONTH)})
    # Standardize dimension order before loading.
    da = da.transpose(time, lat, lon)
    da = da.astype("float32").load()

    print(
        f"✅ SPEI validation period loaded: "
        f"{pd.Timestamp(da[time].values[0]).date()} to "
        f"{pd.Timestamp(da[time].values[-1]).date()}"
    )
    print(f"   Shape: {tuple(da.shape)}")
    print(
        f"   Grid: {da.sizes[lat]} lat × {da.sizes[lon]} lon "
        f"({da.sizes[lat] * da.sizes[lon]:,} cells)"
    )

    return da, lon, lat, time, var


def load_basins():
    gdf = gpd.read_file(BASINS_FILE)
    if BASIN_ID_COLUMN not in gdf.columns:
        raise ValueError(
            f"{BASIN_ID_COLUMN} not found in {BASINS_FILE}. "
            f"Available: {list(gdf.columns)}"
        )

    gdf = gdf[[BASIN_ID_COLUMN, "geometry"]].dropna(subset=["geometry"]).copy()
    gdf = gdf.reset_index(drop=True)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    if gdf[BASIN_ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate HydroBASINS Level-4 IDs found.")

    print(f"✅ Loaded HydroBASINS L4 polygons: {len(gdf):,}")
    return gdf


def aggregate(da, gdf, lon_name, lat_name, time_name):
    mask = regionmask.mask_geopandas(
        gdf,
        da[lon_name],
        da[lat_name],
        numbers=None,
        wrap_lon=False,
    )

    mask_flat = mask.values.reshape(-1)
    lat_vals = da[lat_name].values.astype(float)
    lon_vals = da[lon_name].values.astype(float)

    # Cosine-latitude weights repeated across longitude.
    lat_weights = np.cos(np.deg2rad(lat_vals))
    weights_flat = np.repeat(lat_weights, len(lon_vals)).astype(np.float64)

    values = da.values.reshape(da.sizes[time_name], -1).astype(np.float64)
    n_time = values.shape[0]

    basin_ids = gdf[BASIN_ID_COLUMN].to_numpy()
    result = np.full((n_time, len(gdf)), np.nan, dtype=np.float32)
    qc_rows = []

    print("🚀 Aggregating SPEI-12 to basin means...")
    for i, basin_id in enumerate(basin_ids):
        idx = np.flatnonzero(mask_flat == i)
        n_cells = len(idx)

        if n_cells == 0:
            qc_rows.append({
                "basin": basin_id,
                "n_spei_grid_cells": 0,
                "finite_months": 0,
                "total_months": n_time,
                "finite_month_fraction": 0.0,
                "first_finite_month": None,
                "last_finite_month": None,
            })
            continue

        sub = values[:, idx]
        w = weights_flat[idx][None, :]
        finite = np.isfinite(sub)

        numerator = np.nansum(np.where(finite, sub * w, 0.0), axis=1)
        denominator = np.sum(np.where(finite, w, 0.0), axis=1)
        mean = np.divide(
            numerator,
            denominator,
            out=np.full(n_time, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
        result[:, i] = mean.astype(np.float32)

        finite_months = int(np.isfinite(mean).sum())
        finite_idx = np.flatnonzero(np.isfinite(mean))
        times = pd.DatetimeIndex(da[time_name].values)

        qc_rows.append({
            "basin": basin_id,
            "n_spei_grid_cells": n_cells,
            "finite_months": finite_months,
            "total_months": n_time,
            "finite_month_fraction": finite_months / n_time if n_time else np.nan,
            "first_finite_month": (
                str(times[finite_idx[0]].date()) if len(finite_idx) else None
            ),
            "last_finite_month": (
                str(times[finite_idx[-1]].date()) if len(finite_idx) else None
            ),
        })

        if (i + 1) % 200 == 0 or i + 1 == len(gdf):
            print(f"   processed {i + 1:,}/{len(gdf):,} basins")

    out = xr.Dataset(
        {
            "spei12": (
                ("time", "basin"),
                result,
                {
                    "long_name": "Basin-mean 12-month Standardized Precipitation-Evapotranspiration Index",
                    "aggregation": "cosine-latitude weighted mean of SPEI grid-cell centers inside basin",
                    "timescale_months": 12,
                },
            )
        },
        coords={
            "time": pd.DatetimeIndex(da[time_name].values),
            "basin": basin_ids,
        },
        attrs={
            "source_product": "SPEIbase SPEI-12",
            "basin_level": "HydroBASINS Level 4",
            "spatial_aggregation": (
                "cell-center zonal mean with cosine(latitude) weighting; "
                "basins with zero SPEI cell centers remain missing"
            ),
        },
    )

    qc = pd.DataFrame(qc_rows)
    return out, qc


def main():
    print("--- Validation A1: SPEI-12 → HydroBASINS L4 ---")
    spei_file = find_spei_file()
    da, lon, lat, time, var = load_spei(spei_file)
    basins = load_basins()
    out, qc = aggregate(da, basins, lon, lat, time)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(QC_OUTPUT), exist_ok=True)

    out.to_netcdf(OUTPUT_FILE)
    qc.to_csv(QC_OUTPUT, index=False)

    covered = int((qc["finite_months"] > 0).sum())
    zero_cells = int((qc["n_spei_grid_cells"] == 0).sum())
    complete = int((qc["finite_month_fraction"] == 1.0).sum())

    print(f"✅ Saved basin SPEI-12: {OUTPUT_FILE}")
    print(f"✅ Saved aggregation QC: {QC_OUTPUT}")
    print("\nCoverage summary:")
    print(f"   Basins total: {len(qc):,}")
    print(f"   Basins with any SPEI-12: {covered:,}")
    print(f"   Basins with complete monthly coverage: {complete:,}")
    print(f"   Basins with zero 0.5° SPEI cell centers: {zero_cells:,}")
    print("--- Done ---")


if __name__ == "__main__":
    main()

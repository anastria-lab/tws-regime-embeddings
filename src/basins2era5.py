import os
import numpy as np
import xarray as xr
import geopandas as gpd
import regionmask

# =========================
# CONFIG
# =========================
ERA5_FILE = "data/raw/era5/data_stream-moda.nc"
BASINS_FILE = "data/interim/hydrobasins_l05_global.gpkg"
OUTPUT_FILE = "data/interim/era5_basin_means_level05.nc"

VARIABLES = [
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
    "ssro",
    "sro",
    "e",
    "tp",
    "pev",
    "lai_hv",
    "lai_lv",
]

BASIN_ID_COLUMN = "HYBAS_ID"


def load_era5_data(file_path):
    """
    Load ERA5 data from NetCDF.

    Returns
    -------
    xarray.Dataset
    """
    try:
        ds = xr.open_dataset(file_path)
        print(f"✅ ERA5 loaded: {file_path}")
        return ds
    except Exception as e:
        print(f"❌ Failed to load ERA5 data: {e}")
        return None


def load_basins(gpkg_path, basin_id_column=BASIN_ID_COLUMN):
    """
    Load basin polygons from a GeoPackage.

    Returns
    -------
    geopandas.GeoDataFrame
    """
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
    """
    Detect ERA5 longitude/latitude/time coordinate names.
    """
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    lat_name = "latitude" if "latitude" in ds.coords else "lat"

    if "valid_time" in ds.coords:
        time_name = "valid_time"
    elif "time" in ds.coords:
        time_name = "time"
    else:
        raise ValueError("No time coordinate found. Expected 'valid_time' or 'time'.")

    if lon_name not in ds.coords or lat_name not in ds.coords:
        raise ValueError("Expected longitude/latitude coordinates.")

    return lon_name, lat_name, time_name


def normalize_longitudes(ds, lon_name):
    """
    Force longitude coordinates to the [-180, 180) convention.
    """
    lon = ds[lon_name].values

    lon_fixed = ((lon + 180) % 360) - 180

    ds = ds.assign_coords({lon_name: lon_fixed})
    ds = ds.sortby(lon_name)

    lon_min = float(ds[lon_name].min().values)
    lon_max = float(ds[lon_name].max().values)

    print(f"ℹ️ ERA5 longitude range after normalization: {lon_min:.3f} .. {lon_max:.3f}")
    return ds


def ensure_basins_crs(gdf):
    """
    Ensure basins are in EPSG:4326.
    """
    if gdf.crs is None:
        print("⚠️ Basin CRS missing. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        print("ℹ️ Reprojected basins to EPSG:4326")

    return gdf


def subset_variables(ds, variables):
    """
    Keep only variables that exist in the dataset.
    """
    existing = [v for v in variables if v in ds.data_vars]
    missing = [v for v in variables if v not in ds.data_vars]

    if missing:
        print("⚠️ These variables were not found and will be skipped:")
        for v in missing:
            print(f"   - {v}")

    if not existing:
        raise ValueError("None of the requested variables were found in the ERA5 dataset.")

    print("✅ Variables to process:")
    for v in existing:
        print(f"   - {v}")

    return ds[existing], existing

def build_basin_mask(ds, basins_gdf, lon_name, lat_name):
    """
    Build a 2D basin mask directly from the GeoDataFrame.

    Returns
    -------
    xarray.DataArray
        Dimensions: (lat, lon)
        Values: basin index, NaN outside basins
    """
    basins_gdf = basins_gdf.reset_index(drop=True).copy()

    lon_min = float(ds[lon_name].min().values)
    lon_max = float(ds[lon_name].max().values)
    print(f"ℹ️ Mask input longitude range: {lon_min:.3f} .. {lon_max:.3f}")

    mask = regionmask.mask_geopandas(
        basins_gdf,
        ds[lon_name],
        ds[lat_name],
        numbers=None,     # use GeoDataFrame index
        wrap_lon=False    # do not let regionmask try to wrap again
    )

    print("✅ Basin mask created")
    return mask

def build_latitude_weights(ds, lat_name, lon_name):
    """
    Build 2D cosine-latitude weights for a regular lat/lon grid.
    """
    lat_weights_1d = xr.DataArray(
        np.cos(np.deg2rad(ds[lat_name].values)),
        coords={lat_name: ds[lat_name]},
        dims=(lat_name,),
        name="lat_weights"
    )

    ones_lon = xr.DataArray(
        np.ones(ds[lon_name].size),
        coords={lon_name: ds[lon_name]},
        dims=(lon_name,)
    )

    weights_2d = lat_weights_1d * ones_lon
    weights_2d.name = "weights"
    return weights_2d

def compute_basin_means(ds, mask, basins_gdf, lon_name, lat_name, time_name):
    """
    Compute weighted basin means for each variable and each timestamp.

    Uses cosine(latitude) weights and a vectorized grouped reduction.
    """
    basin_ids = basins_gdf[BASIN_ID_COLUMN].values
    results = {}

    print("🚀 Computing weighted basin means...")

    weights_2d = build_latitude_weights(ds, lat_name, lon_name)

    # stack spatial dimensions once
    spatial_dim = "cell"
    mask_1d = mask.stack({spatial_dim: (lat_name, lon_name)})
    weights_1d = weights_2d.stack({spatial_dim: (lat_name, lon_name)})

    # keep only cells that belong to a basin
    valid_cells = mask_1d.notnull()
    mask_1d = mask_1d.where(valid_cells, drop=True)
    weights_1d = weights_1d.where(valid_cells, drop=True)

    basin_index = np.arange(len(basin_ids))

    for var in ds.data_vars:
        print(f"   Processing {var}...")

        da = ds[var]

        # stack spatial dims
        da_1d = da.stack({spatial_dim: (lat_name, lon_name)})
        da_1d = da_1d.where(valid_cells, drop=True)

        # weighted numerator
        weighted_data = da_1d * weights_1d

        numerator = weighted_data.groupby(mask_1d).sum(skipna=True)

        # denominator should include only cells where data is valid
        valid_weights = weights_1d.where(da_1d.notnull())
        denominator = valid_weights.groupby(mask_1d).sum(skipna=True)

        grouped = numerator / denominator

        # xarray version differences: grouped dim may not be literally called "group"
        non_time_dims = [d for d in grouped.dims if d != time_name]
        if len(non_time_dims) != 1:
            raise ValueError(f"Unexpected grouped dims for {var}: {grouped.dims}")

        basin_dim = non_time_dims[0]
        if basin_dim != "basin":
            grouped = grouped.rename({basin_dim: "basin"})

        # align to full basin list
        grouped = grouped.reindex(basin=basin_index)
        grouped = grouped.assign_coords(basin=("basin", basin_ids))

        results[var] = grouped

    out = xr.Dataset(results)

    if time_name in out.dims and time_name != "time":
        out = out.rename({time_name: "time"})

    out["basin"].attrs["long_name"] = "HydroBASINS level 05 basin id"
    out.attrs["source_basins_file"] = BASINS_FILE
    out.attrs["description"] = (
        "Weighted mean values per HydroBASINS level 05 basin and timestep "
        "using cosine(latitude) area weights"
    )

    print("✅ Weighted basin means computed")
    return out

def save_to_netcdf(ds, output_file):
    """
    Save dataset to NetCDF.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {"zlib": True, "complevel": 4}

        ds.to_netcdf(output_file, encoding=encoding)
        print(f"✅ Output saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"❌ Failed to save output: {e}")
        return None


def main():
    print("--- Basin to ERA5 aggregation ---")

    ds = load_era5_data(ERA5_FILE)
    if ds is None:
        return

    basins = load_basins(BASINS_FILE, basin_id_column=BASIN_ID_COLUMN)
    if basins is None:
        return

    lon_name, lat_name, time_name = detect_coord_names(ds)
    ds = normalize_longitudes(ds, lon_name)
    lon_vals = ds[lon_name].values
    if (lon_vals < 0).any() and (lon_vals > 180).any():
        raise ValueError(
            "Longitude normalization failed: dataset still contains both negative and >180 values."
        )
    basins = ensure_basins_crs(basins)

    ds, used_variables = subset_variables(ds, VARIABLES)

    mask = build_basin_mask(ds, basins, lon_name, lat_name)

    # print("🧪 Test mode: using first 3 timesteps only")
    # ds = ds.isel({time_name: slice(0, 3)})

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
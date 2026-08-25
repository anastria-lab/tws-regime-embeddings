# graceQualityCheck.py
# Step 2.1: Load and perform quality checks on GRACE data. Export years and months for later merging with ERA5 data.

import xarray as xr
import pandas as pd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

FILE_PATH = "data/raw/grace/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"

def load_grace_data(file_path):
    """
    Load GRACE data from a NetCDF file.

    Parameters:
    file_path (str): Path to the NetCDF file containing GRACE data.

    Returns:
    xarray.DataArray: GRACE data as an xarray DataArray.
    """
    try:
        ds = xr.open_dataset(file_path)
        tws = ds['tws']  # TWS and LWE are the same variable in this dataset
        return tws
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def convert_to_dataframe(ds):
    """
    Convert xarray DataArray to a pandas DataFrame and 
    extract year and month.

    Parameters:
    ds (xarray.DataArray): GRACE data as an xarray DataArray.

    Returns:
    pandas.DataFrame: GRACE data as a pandas DataFrame.
    """
    try:
        ds["time"] = pd.to_datetime(ds.time.values, origin="2002-01-01", unit="D")
        ds=ds.drop_vars("time_bounds", errors="ignore")
        df = ds.to_dataframe().reset_index()
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        return df
    except Exception as e:
        print(f"Error converting to DataFrame: {e}")
        return None


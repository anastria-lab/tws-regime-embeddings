# import xarray as xr

# ds = xr.open_dataset("data/interim/era5_basin_means_level05.nc")

# print(ds)
# print(ds["time"])
# print(ds["time"].attrs)

# if "time_bounds" in ds:
#     print(ds["time_bounds"])
#     print(ds["time_bounds"].attrs)

# import pyarrow.parquet as pq
# import numpy as np
# import pandas as pd
# import pyarrow as pa

# test=pq.read_table("data/processed/basin_dataset_anomaly_normalized.parquet").to_pandas()
# print(test.columns)
# print(test.shape)
# print(test["basin"].nunique())
# print(test["lwe_thickness_anomaly_normalized"].isna().mean())
# print(test["tp_anomaly_normalized"].isna().mean())
# print(test["swvl1_anomaly_normalized"].isna().mean())

import pandas as pd

df = pd.read_parquet("data/processed/basin_dataset_wavelet_multiscale.parquet")

print(df.shape)

dup = df.groupby(["basin", "year", "month"]).size()
print("max rows per basin-year-month:", dup.max())
print("value counts of duplicate multiplicity:")
print(dup.value_counts().sort_index().head(20))
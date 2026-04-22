import xarray as xr

ds = xr.open_dataset("data/raw/grace/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc")

print(ds)
print(ds["time"])
print(ds["time"].attrs)

if "time_bounds" in ds:
    print(ds["time_bounds"])
    print(ds["time_bounds"].attrs)
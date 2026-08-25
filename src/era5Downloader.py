# era5Downloader.py
# Step 1: Download ERA5 Land monthly means data using CDS API

from pathlib import Path
import cdsapi

DATA_DIR = Path(r"H:\temp\InputData\era5")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_era5_data(target_filename="era5_land_monthly.zip"):
    """Downloads ERA5 Land monthly means data."""
    
    dataset = "reanalysis-era5-land-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4",
            "sub_surface_runoff", "surface_runoff", "total_evaporation",
            "total_precipitation", "potential_evaporation",
            "leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation"
        ],
        "year": [str(year) for year in range(2002, 2027)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "zip"
    }

    client = cdsapi.Client()
    target_path = DATA_DIR / target_filename
    
    print(f"📡 Requesting data from CDS... Target: {target_path}")
    client.retrieve(dataset, request).download(target_path)
    print("✅ Download finished.")

if __name__ == "__main__":
    download_era5_data()
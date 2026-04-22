# Project <span style="color:deeppink">**tws-regime-embeddings**</span>
<span style="color:deeppink">A Multi-scale Embedding Atlas of Terrestrial Water Storage Dynamics from ERA5 and GRACE</span>

### UV Environment
Install UV https://docs.astral.sh/uv/getting-started/installation/
Create a venv:

    uv venv

### Requirements
Make sure the file .cdsapirc with the lines url: https://cds.climate.copernicus.eu/api

There is a file named <span style="color:cyan">***requirements.in***</span> that lists the projects dependencies. 

Run the following command. uv will resolve all sub-dependencies and create a locked requirements.txt file:

    uv pip compile requirements.in -o requirements.txt

This happens almost instantly, even for large dependency trees.

### Suggested folder structure
    tws-regime-embeddings/
    ├── README.md
    ├── environment.yml
    ├── .gitignore
    ├── data/
    │   ├── raw/
    │   │   ├── grace/
    │   │   ├── era5/
    │   │   └── hybas_lev05_v1c/
    │   ├── interim/
    │   └── processed/
    ├── notebooks/
    ├── src/
    ├── tests/
    ├── results/
    │   ├── figures/
    │   ├── tables/
    │   └── logs/
    └──

### Input Data
#### Grace Data
https://www2.csr.utexas.edu/grace/RL06_mascons.html

Run the <span style="color:cyan">graceQualityCheck.py</span> file to do a quality check of your dataset

#### ERA5 Data
ERA5 data are downloaded from the CDM API.
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview

Run the <span style="color:cyan">era5Downloader.py</span>  or the <span style="color:cyan">main.py</span>  to download them.

#### HydroBASINS
HydroBASINS represents a series of vectorized polygon layers that depict sub-basin boundaries at a global scale.
https://www.hydrosheds.org/products/hydrobasins
For this study we downloaded level 5 basins globally and merge them into a geopackage (CRF/EPSG 4326) (located in the data/interim folder)
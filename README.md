# tws-regime-embeddings
A Multi-scale Embedding Atlas of Terrestrial Water Storage Dynamics from ERA5 and GRACE

# UV Environment
Install UV https://docs.astral.sh/uv/getting-started/installation/
Create a venv:

    uv venv

# Requirements
Make sure the file .cdsapirc with the lines url: https://cds.climate.copernicus.eu/api

There is a file named <span style="color:cyan">***requirements.in***</span> that lists the projects dependencies. 

Run the following command. uv will resolve all sub-dependencies and create a locked requirements.txt file:

    uv pip compile requirements.in -o requirements.txt

This happens almost instantly, even for large dependency trees.
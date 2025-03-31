import io
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from owslib.wcs import WebCoverageService

# Mapping type to the correct WCS URL
WCS_URLS = {
    "ph": "https://maps.isric.org/mapserv?map=/map/phh2o.map",
    "clay": "https://maps.isric.org/mapserv?map=/map/clay.map",
    "sand": "https://maps.isric.org/mapserv?map=/map/sand.map"
}

# Mapping type to available layer names
LAYER_NAMES = {
    "ph": ["phh2o_0-5cm_mean", "phh2o_5-15cm_mean", "phh2o_15-30cm_mean"],
    "clay": ["clay_0-5cm_mean", "clay_5-15cm_mean", "clay_15-30cm_mean"],
    "sand": ["sand_0-5cm_mean", "sand_5-15cm_mean", "sand_15-30cm_mean"]
}

def fetch_soil_data(soil_type, bbox):
    """ Fetches soil data for a given type using WCS and aggregates the data. """
    wcs_url = WCS_URLS[soil_type]
    layers = LAYER_NAMES[soil_type]
    wcs = WebCoverageService(wcs_url, version='1.0.0')

    soil_data = {}

    for layer in layers:
        try:
            response = wcs.getCoverage(
                identifier=layer,
                crs='urn:ogc:def:crs:EPSG::32628',
                bbox=bbox,
                resx=30,
                resy=30,
                format='GEOTIFF_INT16'
            )
            temp_file = io.BytesIO(response.read())

            with MemoryFile(temp_file) as memfile:
                with memfile.open() as dataset:
                    layer_data = dataset.read(1)
                    soil_data[layer] = np.nanmean(layer_data)  # Aggregate data

        except Exception as e:
            print(f"Error fetching {layer}: {e}")
            soil_data[layer] = np.nan  # Handle missing data

    return soil_data

def create_soil_dataset(bbox):
    """ Fetches soil data for all types and aggregates them into a DataFrame. """
    dataset = {}

    for soil_type in WCS_URLS.keys():
        data = fetch_soil_data(soil_type, bbox)
        dataset.update(data)  # Merge data

    df = pd.DataFrame([dataset])  # Convert to single-row DataFrame
    return df

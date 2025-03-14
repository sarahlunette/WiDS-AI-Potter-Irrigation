import io
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from rasterio import plot
from owslib.wcs import WebCoverageService

# Define the SoilGrids WCS URL
SOILGRIDS_URL = "https://maps.isric.org/mapserv?map=/map/phh2o.map"

def fetch_soil_data(lat, lon, bbox=None):
    """
    Fetches soil pH data for a given coordinate using WCS (Web Coverage Service).

    Args:
        lat (float): Latitude.
        lon (float): Longitude.
        bbox (tuple, optional): Bounding box (minx, miny, maxx, maxy). Defaults to a fixed bbox.

    Returns:
        numpy.ndarray: Array of soil pH values.
    """
    # Connect to WebCoverageService
    wcs = WebCoverageService(SOILGRIDS_URL, version='1.0.0')

    # Default bounding box (needs proper calculation for real use case)
    if bbox is None:
        bbox = (-1784000, 1356000, -1140000, 1863000)

    try:
        # Fetch soil data as a GeoTIFF
        response = wcs.getCoverage(
            identifier='phh2o_0-5cm_mean',
            crs='urn:ogc:def:crs:EPSG::152160',
            bbox=bbox,
            resx=250,
            resy=250,
            format='GEOTIFF_INT16'
        )

        # Read response into memory
        temp_file = io.BytesIO(response.read())

        # Open raster in memory
        with MemoryFile(temp_file) as memfile:
            with memfile.open() as dataset:
                ph_data = dataset.read(1)  # Read first band as NumPy array

        return ph_data, ph_data.mean()  # Return the extracted array

    except Exception as e:
        print(f"Error fetching soil data: {e}")
        return None

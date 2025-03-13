# TODO other data: for now we have only ph data
import io
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.io import MemoryFile
from rasterio import plot
from owslib.wcs import WebCoverageService

# Define the SoilGrids WCS URL
SOILGRIDS_URL = "https://maps.isric.org/mapserv?map=/map/phh2o.map"

def fetch_soil_data(lat, lon, bbox):
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

        return ph_data  # Return the extracted array

    except Exception as e:
        print(f"Error fetching soil data: {e}")
        return None

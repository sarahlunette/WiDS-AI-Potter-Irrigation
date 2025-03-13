
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from sentinelsat import SentinelAPI
from baitsss import evapotranspiration
from owslib.wcs import WebCoverageService
import rasterio

SOILGRIDS_URL = "http://maps.isric.org/mapserv?map=/map/phh2o.map"
def fetch_soil_data(lat, lon, bbox): # TODO: create the bbox for the patch (another function)
    wcs = WebCoverageService(SOILGRIDS_URL, version='1.0.0')
    bbox = (-1784000, 1356000, -1140000, 1863000) 
    params = {"lon": lon, "lat": lat}
    response = wcs.getCoverage(
        identifier='phh2o_0-5cm_mean', 
        crs='urn:ogc:def:crs:EPSG::152160',
        bbox=bbox, 
        resx=250, resy=250, 
        format='GEOTIFF_INT16')
    with open('./data/Senegal_pH_0-5_mean.tif', 'wb') as file:
        file.write(response.read())
    ph = rasterio.open("./data/Senegal_pH_0-5_mean.tif", driver="GTiff")
    return response.json() if response.status_code == 200 else None
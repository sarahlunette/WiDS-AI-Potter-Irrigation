## TODO: replace csv and the right urls as well as API keys, make sure these are the right way to query the API and sites
## TODO: Add parameters in the different requests if necessary (polygoon, date, cloud_coverage for instance)
## TODO: Simple data, do we have to add dependencies with weather data in the functions ?

### ETL Pipeline for Smart Irrigation AI

#### **Step 1: Data Acquisition**
''''**APIs Used:**
- OpenWeatherMap API (real-time & historical weather data)
- Weatherstack API (reliable weather information)
- SoilGrids (global soil properties data)
- FAO Data (crop production, soil, climate datasets)
- Copernicus Open Access Hub (satellite imagery for soil moisture & crop health)
'''

import requests
import pandas as pd
import os
from dotenv import load_dotenv
from sentinelsat import SentinelAPI
from owslib.wcs import WebCoverageService
import rasterio
from soilgrid.bbox_computation import compute_bbox
from soilgrid.get_soil_data import fetch_soil_data
from weather.weather import get_weather
from evapotranspiration.get_evapotranspiration import fetch_evapotranspiration_data

load_dotenv()
# Constants ## TODO: put them all in .env 
OWM_API_KEY = os.getenv("OWM_API_KEY")
SOILGRIDS_URL = "http://maps.isric.org/mapserv?map=/map/phh2o.map"
SENTINEL_USERNAME = ''
SENTINEL_PASSWORD = ''

# OpenWeatherMap API
# Function to fetch weather data
def get_weather(lat, lon, API_KEY): # TODO: strange way of inputing API_KEY not sure we need a microservice for this
    results = get_weather(lat, lon, API_KEY)
    return results

# SoilGrids API
def fetch_soil_data(lat, lon, extension=5):
    bbox = compute_bbox(lat, lon, extension)
    results = fetch_soil_data(lat, lon, bbox)
    return results

'''# Fetching FAO Data (Placeholder for FAO dataset retrieval)
def fetch_fao_data():
    fao_data = pd.read_csv("FAO_dataset.csv")  # Assuming local storage or API access
    return fao_data

# Fetching Copernicus Data (Placeholder for Copernicus API call)
def fetch_copernicus_data():
    api = SentinelAPI("your_username", "your_password", "https://scihub.copernicus.eu/dhus")
    products = api.query(area="POLYGON ((-1.5 43.5, -1.5 44.5, 0 44.5, 0 43.5, -1.5 43.5))", 
                        platformname="Sentinel-2",
                        date=("NOW-7DAYS", "NOW"),
                        cloudcoverpercentage=(0, 30))
    return products'''

def fetch_evapo_data(solar_radiation, temperature, humidity):
    evapo_rate = fetch_evapotranspiration_data(solar_radiation, temperature, humidity)
    return evapo_rate
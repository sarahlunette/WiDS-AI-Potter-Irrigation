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
import dotenv # TODO: get environment variables API keys
from sentinelsat import SentinelAPI
from baitsss import evapotranspiration

# Constants ## TODO: put them all in .env 
OWM_API_KEY = "your_api_key"
OWM_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHERSTACK_API_KEY = "your_api_key"
WEATHERSTACK_URL = "http://api.weatherstack.com/current"
SOILGRIDS_URL = "https://rest.soilgrids.org/query"
SENTINEL_USERNAME = ''
SENTINEL_PASSWORD = ''

# OpenWeatherMap API
def fetch_weather_data(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"}
    response = requests.get(OWM_URL, params=params)
    return response.json() if response.status_code == 200 else None

# Weatherstack API
def fetch_weatherstack_data(location):
    params = {"access_key": WEATHERSTACK_API_KEY, "query": location}
    response = requests.get(WEATHERSTACK_URL, params=params)
    return response.json() if response.status_code == 200 else None

# SoilGrids API
def fetch_soil_data(lat, lon):
    params = {"lon": lon, "lat": lat}
    response = requests.get(SOILGRIDS_URL, params=params)
    return response.json() if response.status_code == 200 else None

# Fetching FAO Data (Placeholder for FAO dataset retrieval)
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
    return products

def fetch_evapotranspiration_data(solar_radiation, temperature, humidity):
    evapo_rate = evapotranspiration(solar_radiation, temperature, humidity)
    return evapo_rate
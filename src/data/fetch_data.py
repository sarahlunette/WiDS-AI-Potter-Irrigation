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
from baitsss import evapotranspiration
from owslib.wcs import WebCoverageService
import rasterio

load_dotenv()
# Constants ## TODO: put them all in .env 
OWM_API_KEY = os.getenv("OWM_API_KEY")
SOILGRIDS_URL = "http://maps.isric.org/mapserv?map=/map/phh2o.map"
SENTINEL_USERNAME = ''
SENTINEL_PASSWORD = ''

# OpenWeatherMap API
# Function to fetch weather data
def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract current weather
        current_temp = data["current"]["temp"]
        current_conditions = data["current"]["weather"][0]["description"]

        # Extract daily forecast (8 days)
        forecast = []
        for day in data["daily"]:
            forecast.append({
                "date": day["dt"],  # Timestamp (convert to date if needed)
                "temp_min": day["temp"]["min"],
                "temp_max": day["temp"]["max"],
                "weather": day["weather"][0]["description"]
            })
        
        return {
            "current": {
                "temperature": current_temp,
                "conditions": current_conditions
            },
            "forecast": forecast
        }
    
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# SoilGrids API
def fetch_soil_data(lat, lon):
    wcs = WebCoverageService(SOILGRIDS_URL, version='1.0.0')
    params = {"lon": lon, "lat": lat}
    bbox = (-1784000, 1356000, -1140000, 1863000) # TODO: create the bbox for the patch (another function)
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
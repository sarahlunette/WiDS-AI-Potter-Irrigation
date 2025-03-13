## TODO do we need weatherstack ?
## Change main to get all the data
## Remove Locations, and send locations to process api to get the processed data ? How does it integrate into the pipeline ?
import requests
import pandas as pd
import time
from weather.weather import get_weather
from evapotranspiration.get_evapotranspiration import fetch_evapotranspiration_data
from soilgrid.get_soil_data import fetch_soil_data
import os
from dotenv import load_dotenv

'''# Locations for data collection
LOCATIONS = [
    {"lat": 48.8566, "lon": 2.3522, "name": "Paris", "extension": 5}, # TODO: modify extension
    {"lat": 37.7749, "lon": -122.4194, "name": "San Francisco", "extension": 5},
    {"lat": 51.5074, "lon": -0.1278, "name": "London", "extension": 5},
    {"lat": 40.7128, "lon": -74.0060, "name": "New York", "extension": 5},
    {"lat": 35.6895, "lon": 139.6917, "name": "Tokyo", "extension": 5},
    {"lat": 55.7558, "lon": 37.6176, "name": "Moscow", "extension": 5},
    {"lat": -33.8688, "lon": 151.2093, "name": "Sydney", "extension": 5},
    {"lat": -23.5505, "lon": -46.6333, "name": "Sao Paulo", "extension": 5},
    {"lat": 19.0760, "lon": 72.8777, "name": "Mumbai", "extension": 5},
    {"lat": 40.4168, "lon": -3.7038, "name": "Madrid", "extension": 5}
]'''

# Process and Save Data
# TODO: Has to return main result in API

load_dotenv()

API_KEY = os.getenv('API_KEY')

def get_data(LOCATIONS):
    all_data = []
    for location in LOCATIONS:
        lat, lon, name, extension = location["lat"], location["lon"], location["name"], location["extension"]
        print(f"Fetching data for {name}...")

        weather_owm = get_weather(lat, lon, API_KEY)
        soil_data = fetch_soil_data(lat, lon)
        evapotranspiration_data = fetch_evapotranspiration_data(weather_own['current']['temperature'], humidity)

        data_entry = {
            "location": name,
            "lat": lat,
            "lon": lon,
            "temperature": weather_owm["main"]["temp"] if weather_owm else None,
            "humidity": weather_owm["main"]["humidity"] if weather_owm else None,
            "soil_ph": soil_data["properties"]["phh2o"] if soil_data else None
        }
        all_data.append(data_entry)
        time.sleep(1)  # Avoid hitting API limits

    df = pd.DataFrame(all_data)
    # TODO: to DB
    return df


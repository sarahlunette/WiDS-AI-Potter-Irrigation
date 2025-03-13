import pandas as pd
from src.fetch_data import fetch_weather_data, fetch_soil_data, fetch_fao_data, fetch_copernicus_data, fetch_evapotranspiration_data, fetch_weatherstack_data

# Example Usage
lat, lon = 48.8566, 2.3522  # Paris coordinates for testing
## TODO: use etl main function to then save data
weather_data = fetch_weather_data(lat, lon)
weatherstack_data = fetch_weatherstack_data(lat, lon)
soil_data = fetch_soil_data(lat, lon)
fao_data = fetch_fao_data()
copernicus_data = fetch_copernicus_data()


data = {
    "temperature": weather_data["main"]["temp"] if weather_data else None,
    "humidity": weather_data["main"]["humidity"] if weather_data else None,
    "soil_pH": soil_data["properties"]["phh2o"]["M"] if soil_data else None,
}

df = pd.DataFrame([data])
# TODO: TO DB
df.to_csv("irrigation_data.csv", index=False)


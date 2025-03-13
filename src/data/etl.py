## TODO do we need weatherstack ?
## Change main to get all the data
## Remove Locations, and send locations to process api to get the processed data ? How does it integrate into the pipeline ?
import requests
import pandas as pd
import time
from src.fetch_data import fetch_weather_data, fetch_weatherstack_data, fetch_soil_data, fetch_fao_data, fetch_copernicus_data, fetch_evapotranspiration_data

# Locations for data collection
LOCATIONS = [
    {"lat": 48.8566, "lon": 2.3522, "name": "Paris"},
    {"lat": 37.7749, "lon": -122.4194, "name": "San Francisco"}
]

# Process and Save Data
# TODO: Has to return main result in API
def main():
    all_data = []
    for location in LOCATIONS:
        lat, lon, name = location["lat"], location["lon"], location["name"]
        print(f"Fetching data for {name}...")

        weather_owm = fetch_weather_openweathermap(lat, lon)
        weather_ws = fetch_weather_weatherstack(lat, lon)
        soil_data = fetch_soil_data(lat, lon)

        data_entry = {
            "location": name,
            "lat": lat,
            "lon": lon,
            "temperature": weather_owm["main"]["temp"] if weather_owm else None,
            "humidity": weather_owm["main"]["humidity"] if weather_owm else None,
            "wind_speed": weather_ws["current"]["wind_speed"] if weather_ws else None,
            "soil_ph": soil_data["properties"]["phh2o"] if soil_data else None,
            "soil_sand": soil_data["properties"]["sand"] if soil_data else None,
        }
        all_data.append(data_entry)
        time.sleep(1)  # Avoid hitting API limits

    df = pd.DataFrame(all_data)
    df.to_csv("irrigation_data.csv", index=False)
    print("Data collection complete. Saved as 'irrigation_data.csv'.")

if __name__ == "__main__":
    main()

import requests
from datetime import datetime, timedelta
# TODO: add other data (such as wind speed, pressure) # Look at paper
# {'coord': {'lon': -122.4194, 'lat': 37.7749}, 'weather': [{'id': 800, 'main': 'Clear', 'description': 'clear sky', 'icon': '01d'}], 'base': 'stations', 'main': {'temp': 11.95, 'feels_like': 10.85, 'temp_min': 10.34, 'temp_max': 13.58, 'pressure': 1012, 'humidity': 63, 'sea_level': 1012, 'grnd_level': 1008}, 'visibility': 10000, 'wind': {'speed': 7.6, 'deg': 295, 'gust': 12.07}, 'clouds': {'all': 8}, 'dt': 1741901588, 'sys': {'type': 2, 'id': 2017837, 'country': 'US', 'sunrise': 1741875801, 'sunset': 1741918493}, 'timezone': -25200, 'id': 5391959, 'name': 'San Francisco', 'cod': 200}

def get_weather(lat, lon, API_KEY): # TODO: don't know if API_KEY should be a variable
    # Using the Current Weather Data API which is available for free accounts
    current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    # Using the 5 Day / 3 Hour Forecast API which is available for free accounts
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    current_response = requests.get(current_url)
    print(current_response)

    forecast_response = requests.get(forecast_url)
    
    if current_response.status_code == 200 and forecast_response.status_code == 200:
        current_data = current_response.json()
        forecast_data = forecast_response.json()
        
        # Extract current weather
        current_temp = current_data["main"]["temp"]
        current_conditions = current_data["weather"][0]["description"]
        current_humidity = current_data["main"]["humidity"]
        
        # Extract next day's forecast
        next_day = datetime.utcnow() + timedelta(days=1)
        next_day_forecast = [
            forecast for forecast in forecast_data["list"]
            if datetime.utcfromtimestamp(forecast["dt"]).date() == next_day.date()
        ]
        
        if next_day_forecast:
            next_day_temp_min = min(forecast["main"]["temp_min"] for forecast in next_day_forecast)
            next_day_temp_max = max(forecast["main"]["temp_max"] for forecast in next_day_forecast)
            next_day_conditions = next_day_forecast[0]["weather"][0]["description"]
        else:
            next_day_temp_min = None
            next_day_temp_max = None
            next_day_conditions = None
        
        return {
            "current": {
                "temperature": current_temp,
                "conditions": current_conditions,
                "humidity": current_humidity,

            },
            "next_day": {
                "temperature_min": next_day_temp_min,
                "temperature_max": next_day_temp_max,
                "conditions": next_day_conditions
            }
        }
    
    else:
        print(f"Error: {current_response.status_code}, {current_response.text}")
        print(f"Error: {forecast_response.status_code}, {forecast_response.text}")
        return None

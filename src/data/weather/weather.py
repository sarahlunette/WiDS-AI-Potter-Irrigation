import requests
from datetime import datetime, timedelta

def get_weather(lat, lon, API_KEY):
    # Using the Current Weather Data API which is available for free accounts
    current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    # Using the 5 Day / 3 Hour Forecast API which is available for free accounts
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    current_response = requests.get(current_url)
    forecast_response = requests.get(forecast_url)
    
    if current_response.status_code == 200 and forecast_response.status_code == 200:
        current_data = current_response.json()
        forecast_data = forecast_response.json()
        
        # Extract current weather
        current_temp = current_data["main"]["temp"]
        current_conditions = current_data["weather"][0]["description"]
        
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
                "conditions": current_conditions
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
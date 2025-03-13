import requests
from weather import get_weather

# Replace with your actual latitude, longitude, and API key
lat = 37.7749  # Example: San Francisco latitude
lon = -122.4194  # Example: San Francisco longitude
API_KEY = "6a210d4c91e9f455ade17f75196c1a17"  # Replace with your actual API key

weather_data = get_weather(lat, lon, API_KEY)

if weather_data:
    print("Current Weather:")
    print(f"Temperature: {weather_data['current']['temperature']}°C")
    print(f"Conditions: {weather_data['current']['conditions']}")
    print(f"Humidity: {weather_data['current']['humidity']}%")
    print("\nNext Day Forecast:")
    print(f"Min Temperature: {weather_data['next_day']['temperature_min']}°C")
    print(f"Max Temperature: {weather_data['next_day']['temperature_max']}°C")
    print(f"Conditions: {weather_data['next_day']['conditions']}")
else:
    print("Failed to retrieve weather data.")

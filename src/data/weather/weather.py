def get_weather(lat, lon, API_KEY):
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
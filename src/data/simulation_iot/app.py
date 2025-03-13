# Second way of generating sensor data
from fastapi import FastAPI
from pydantic import BaseModel
import random
import time
import threading

app = FastAPI()

# Simulated IoT Data (Global)
iot_data = {
    "sector": "Vineyard A",
    "moisture": 30.0,  # Soil moisture (%)
    "temperature": 25.0,  # °C
    "humidity": 60.0,  # %
    "valve_status": "closed"  # open/closed
}

# Function to Simulate Sensor Updates
def update_sensor_data():
    while True:
        iot_data["moisture"] = round(random.uniform(20, 40), 1)  # Soil moisture (20-40%)
        iot_data["temperature"] = round(random.uniform(15, 35), 1)  # Temperature (15-35°C)
        iot_data["humidity"] = round(random.uniform(40, 80), 1)  # Humidity (40-80%)
        iot_data["valve_status"] = "open" if iot_data["moisture"] < 25 else "closed"  # Auto valve control
        time.sleep(10)  # Update every 10 sec

# Background thread to update sensor data
threading.Thread(target=update_sensor_data, daemon=True).start()

# API Route for IoT Data
@app.get("/sensor_data")
def get_sensor_data():
    return iot_data

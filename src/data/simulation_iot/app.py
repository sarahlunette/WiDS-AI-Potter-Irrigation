from flask import Flask, jsonify
import random
import time
import threading

app = Flask(__name__)

# Simulated IoT Data (Global)
iot_data = {
    "sector": "Vineyard A",
    "moisture": 30.0,  # Soil moisture (%)
    "temperature": 25.0,  # °C
    "humidity": 60.0,  # %
    "valve_status": "closed"  # open/closed
}

# ✅ Function to Simulate Sensor Updates
def update_sensor_data():
    while True:
        iot_data["moisture"] = round(random.uniform(20, 40), 1)  # Soil moisture (20-40%)
        iot_data["temperature"] = round(random.uniform(15, 35), 1)  # Temperature (15-35°C)
        iot_data["humidity"] = round(random.uniform(40, 80), 1)  # Humidity (40-80%)
        iot_data["valve_status"] = "open" if iot_data["moisture"] < 25 else "closed"  # Auto valve control
        time.sleep(10)  # Update every 10 sec

# ✅ API Route for IoT Data
@app.route('/sensor_data', methods=['GET'])
def get_sensor_data():
    return jsonify(iot_data)

# ✅ Start Background Sensor Updates
threading.Thread(target=update_sensor_data, daemon=True).start()

# ✅ Run Flask Server
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

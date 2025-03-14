import json
import time
import threading
from kafka import KafkaConsumer, KafkaProducer
from etl import get_data  # Importing the existing get_data function

# Kafka Configuration
KAFKA_BROKER = 'localhost:9092'
SENSOR_TOPIC = 'sensor_data'
ENRICHED_TOPIC = 'enriched_sensor_data'

# List of locations for `get_data()`
LOCATIONS = [
    {"lat": 48.8566, "lon": 2.3522, "name": "Paris", "extension": 5},
    {"lat": 37.7749, "lon": -122.4194, "name": "San Francisco", "extension": 5}
]

# Kafka Consumer to receive sensor data
consumer = KafkaConsumer(
    SENSOR_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Kafka Producer to send enriched data
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Temporary storage for external data (weather & soil), updated every 15 min
external_data = {"timestamp": 0, "data": None}

def update_external_data():
    """Fetch external data every 15 minutes"""
    global external_data
    while True:
        print("Updating external data...")
        try:
            external_data["data"] = get_data(LOCATIONS).to_dict(orient="records")
            external_data["timestamp"] = time.time()
            print("External data updated:", external_data["data"])
        except Exception as e:
            print(f"Error fetching external data: {e}")
        
        time.sleep(900)  # Sleep for 15 minutes (900 seconds)

# Start the external data update thread
threading.Thread(target=update_external_data, daemon=True).start()

def wait_for_external_data():
    """Wait until external data is available before processing sensor data"""
    while external_data["data"] is None:
        print("Waiting for external data to load...")
        time.sleep(5)

def process_sensor_data():
    """Consumes sensor data and enriches it with external data"""
    wait_for_external_data()  # Ensure external data is loaded
    print("Listening for sensor data...")

    for message in consumer:
        sensor_data = message.value
        print("Received sensor data:", sensor_data)

        # Find matching external data # When several locations
        matched_data = external_data["data"][0]  # For now, use the first location
        print(matched_data)

        if matched_data:
            enriched_data = {**sensor_data, **matched_data}
            print(enriched_data)
        else:
            enriched_data = sensor_data  # Send raw sensor data if no match

        #print(f"Sending enriched data: {enriched_data}")
        producer.send(ENRICHED_TOPIC, enriched_data)

if __name__ == "__main__":
    process_sensor_data()



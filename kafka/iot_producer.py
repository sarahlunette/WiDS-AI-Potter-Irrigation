from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def generate_sensor_data():
    return {
        "sector": random.choice(["A", "B"]),
        "temperature": round(random.uniform(15, 35), 2),
        "soil_moisture": round(random.uniform(20, 60), 2),
        "timestamp": time.time()
    }

while True:
    data = generate_sensor_data()
    producer.send("sensor_data", value=data)
    print(f"Sent: {data}")
    time.sleep(5)  # Send data every 5 seconds

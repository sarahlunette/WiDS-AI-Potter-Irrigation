from kafka import KafkaProducer
import random
import time
import json

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_sensor_data():
    while True:
        data = {
            "sector": "Vineyard A",
            "soil_moisture": round(random.uniform(20, 40), 1),  # Soil moisture (20-40%)
            "temperature": round(random.uniform(15, 35), 1),    # Temperature (15-35°C)
            "humidity": round(random.uniform(40, 80), 1)        # Humidity (40-80%)
        }
        producer.send('sensor_data', value=data)
        print(f"Produced: {data}")
        time.sleep(900)  # Simulate data generation every 15 minutes

if __name__ == "__main__":
    generate_sensor_data()

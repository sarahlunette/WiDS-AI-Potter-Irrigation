from fastapi import FastAPI, WebSocket
from kafka import KafkaProducer
import random
import asyncio
import json

app = FastAPI()

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # Change to your Kafka server
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Simulated IoT Data Generator
async def generate_sensor_data():
    while True:
        data = {
            "sector": "Vineyard A",
            "moisture": round(random.uniform(20, 40), 1),  # Soil moisture (20-40%)
            "temperature": round(random.uniform(15, 35), 1),  # Temperature (15-35°C)
            "humidity": round(random.uniform(40, 80), 1),  # Humidity (40-80%)
            "valve_status": "open" if random.uniform(20, 40) < 25 else "closed"
        }
        producer.send('iot_data', value=data)  # Send data to Kafka topic
        await asyncio.sleep(2)  # Simulate data every 2 seconds

# WebSocket endpoint to stream data
@app.websocket("/ws/sensor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async for sensor_data in generate_sensor_data():
        await websocket.send_json(sensor_data)

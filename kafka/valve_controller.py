from kafka import KafkaConsumer, KafkaProducer
import json

# Consumer to receive sensor data
consumer = KafkaConsumer(
    "sensor_data",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

# Producer to send valve commands
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def process_data(data):
    sector = data["sector"]
    moisture = data["soil_moisture"]
    
    # Simple rule: if soil moisture < 30, open valve
    valve_action = "OPEN" if moisture < 30 else "CLOSE"
    
    command = {"sector": sector, "command": valve_action}
    producer.send("valve_commands", value=command)
    print(f"Sent command: {command}")

print("Listening for sensor data...")
for message in consumer:
    process_data(message.value)

# TODO: introduce result for sensor data passed through algorithm (with other data) instead of sensor data only
# TODO: have continuous response instead of discrete response (say open on a continuous scale of 1 to 10) instead of discrete)
from kafka import KafkaConsumer, KafkaProducer
import json

# Kafka Configuration
KAFKA_BROKER = 'localhost:9092'
SENSOR_TOPIC = 'sensor_data'
COMMAND_TOPIC = 'valve_commands'

# Kafka Consumer for receiving sensor data
consumer = KafkaConsumer(
    SENSOR_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Kafka Producer for sending valve commands
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def process_data():
    print("Listening for sensor data...")
    for message in consumer:
        data = message.value
        moisture = data.get("soil_moisture")
        
        # Simple rule: if soil moisture < 30, open valve
        valve_action = "OPEN" if moisture < 30 else "CLOSE"
        
        command = {"sector": data.get("sector"), "command": valve_action}
        producer.send(COMMAND_TOPIC, value=command)
        print(f"Sent command: {command}")

if __name__ == "__main__":
    process_data()

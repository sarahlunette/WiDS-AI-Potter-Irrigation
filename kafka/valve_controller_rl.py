from kafka import KafkaConsumer, KafkaProducer
import json
import pickle
import numpy as np

# Kafka Configuration
KAFKA_BROKER = 'localhost:9092'
ENRICHED_SENSOR_TOPIC = 'enriched_sensor_data'
COMMAND_TOPIC = 'valve_commands'

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    ENRICHED_SENSOR_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Load trained RL model (Q-table)
def load_q_table(filename="/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/models/rl/q_table.pkl"):
    with open(filename, "rb") as file:
        return pickle.load(file)

q_table = load_q_table()
actions = np.linspace(0, 5, num=11)  # Possible irrigation amounts (0 to 5 mm)

def predict_irrigation(sensor_data):
    """Predict optimal irrigation based on real-time sensor data using Q-learning model."""
    state = (round(sensor_data["soil_moisture"], 1), round(sensor_data["evapotranspiration"], 1))

    # Choose the action with the highest Q-value
    best_action = max(actions, key=lambda a: q_table.get((state, a), 0))
    return best_action

# Process incoming sensor data
def process_sensor_data():
    print("📡 Listening for sensor data...")
    for message in consumer:
        sensor_data = message.value
        print(f"📥 Received sensor data: {sensor_data}")

        # Get irrigation decision from RL model
        valve_action = predict_irrigation(sensor_data)

        if valve_action is not None:
            command = {
                "sector": sensor_data.get("sector"),
                "command": valve_action
            }
            producer.send(COMMAND_TOPIC, value=command)
            print(f"✅ Sent valve command: {command}")
        else:
            print("⚠️ No valid valve action received.")

if __name__ == "__main__":
    process_sensor_data()

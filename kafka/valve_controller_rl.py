from kafka import KafkaConsumer, KafkaProducer
import json
import requests

# Kafka Configuration
KAFKA_BROKER = 'localhost:9092'
SENSOR_TOPIC = 'sensor_data'
COMMAND_TOPIC = 'valve_commands'

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    SENSOR_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Function to call the RL model prediction API
def get_rl_prediction(sensor_data):
    api_url = 'http://localhost:8000/predict'  # Replace with your actual API endpoint
    response = requests.post(api_url, json=sensor_data)
    if response.status_code == 200:
        return response.json().get('valve_action')
    else:
        print(f"Error: Received status code {response.status_code} from prediction API.")
        return None

# Process incoming sensor data
def process_sensor_data():
    print("Listening for sensor data...")
    for message in consumer:
        sensor_data = message.value
        print(f"Received sensor data: {sensor_data}")

        # Get prediction from RL model
        valve_action = get_rl_prediction(sensor_data)

        if valve_action:
            command = {
                "sector": sensor_data.get("sector"),
                "command": valve_action
            }
            producer.send(COMMAND_TOPIC, value=command)
            print(f"Sent valve command: {command}")
        else:
            print("No valid valve action received.")

if __name__ == "__main__":
    process_sensor_data()

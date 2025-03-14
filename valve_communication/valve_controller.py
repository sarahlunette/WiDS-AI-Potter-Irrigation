from kafka import KafkaConsumer
import json
import requests

# Kafka Configuration
KAFKA_BROKER = 'localhost:9092'
COMMAND_TOPIC = 'valve_commands'

# Valve Controller API URL
VALVE_API_URL = 'http://192.168.1.100:5000/control'  # Replace with actual valve API endpoint (if connected device)

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    COMMAND_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

def send_to_valve(command):
    """Send irrigation command to the actual valve via API."""
    try:
        response = requests.post(VALVE_API_URL, json=command, timeout=5)
        if response.status_code == 200:
            print(f"✅ Successfully sent command to valve: {command}")
        else:
            print(f"⚠️ Valve API error: {response.status_code}, {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Connection error: {e}")

# Process incoming irrigation commands
def process_commands():
    print("📡 Listening for irrigation commands...")
    for message in consumer:
        command = message.value
        print(f"📥 Received irrigation command: {command}")
        send_to_valve(command)

if __name__ == "__main__":
    process_commands()

from kafka import KafkaConsumer
import json
import requests
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Kafka Configuration
KAFKA_BROKER = 'localhost:9092'
COMMAND_TOPIC = 'valve_commands'

# Valve Controller API URL
VALVE_API_URL = 'http://192.168.1.100:5000/control'  # Replace with actual valve API endpoint

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    COMMAND_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

def send_to_valve(command):
    """Send irrigation command to the valve via API."""
    try:
        valve_value = command.get("command")
        sector = command.get("sector")

        if valve_value is None or not (1 <= valve_value <= 10):
            logging.warning(f"⚠️ Invalid valve value received: {valve_value}. Must be between 1 and 10.")
            return

        payload = {"sector": sector, "valve_value": valve_value}
        response = requests.post(VALVE_API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            logging.info(f"✅ Successfully sent command to valve: {payload}")
        else:
            logging.error(f"⚠️ Valve API error: {response.status_code}, {response.text}")

    except requests.exceptions.RequestException as e:
        logging.error(f"🚨 Connection error: {e}")

# Process incoming irrigation commands
def process_commands():
    logging.info("📡 Listening for irrigation commands...")
    for message in consumer:
        command = message.value
        logging.info(f"📥 Received irrigation command: {command}")
        send_to_valve(command)

if __name__ == "__main__":
    process_commands()

import json
import threading
from kafka import KafkaConsumer, KafkaProducer
from llm import query_chatbot

KAFKA_BROKER = "localhost:9092"
SENSOR_TOPIC = "sensor_data"
COMMAND_TOPIC = "COMMAND_TOPIC"


def fetch_latest_data(topic):
    consumer = KafkaConsumer(topic, bootstrap_servers=KAFKA_BROKER, value_deserializer=lambda m: json.loads(m.decode("utf-8")))
    latest_message = None
    for message in consumer:
        latest_message = message.value
    consumer.close()
    return latest_message


def automated_decision_making():
    consumer = KafkaConsumer(SENSOR_TOPIC, bootstrap_servers=KAFKA_BROKER, value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER, value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    
    for message in consumer:
        sensor_data = message.value
        rl_data = fetch_latest_data(COMMAND_TOPIC)
        
        context = {
            "sensor_data": sensor_data,
            "rl_data": rl_data
        }
        
        decision = query_chatbot(f"Decision-making context: {json.dumps(context)}. Output a number between 1 and 10.")
        
        try:
            decision_value = int(decision.strip())
            producer.send(COMMAND_TOPIC, {"sector": sensor_data['sector'], "command_value": decision_value})
        except ValueError:
            print("Invalid LLM output format, ignoring command.")


threading.Thread(target=automated_decision_making, daemon=True).start()

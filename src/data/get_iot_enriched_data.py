from kafka import KafkaConsumer, KafkaProducer
import json
import time
import threading
from etl import get_data  # Import de la fonction get_data existante

# Configuration Kafka
KAFKA_BROKER = 'localhost:9092'
SENSOR_TOPIC = 'sensor_data'
ENRICHED_TOPIC = 'enriched_sensor_data'

# Liste des emplacements pour `get_data()`
LOCATIONS = [
    {"lat": 48.8566, "lon": 2.3522, "name": "Paris", "extension": 5},
    {"lat": 37.7749, "lon": -122.4194, "name": "San Francisco", "extension": 5}
]

# Kafka Consumer pour recevoir les données des capteurs
consumer = KafkaConsumer(
    SENSOR_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Kafka Producer pour envoyer les données enrichies
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Stockage temporaire des données externes mises à jour toutes les 15 minutes
external_data = {"timestamp": 0, "data": None}

def update_external_data():
    """Met à jour les données météorologiques et de sol toutes les 15 minutes"""
    global external_data
    while True:
        print("Mise à jour des données externes...")
        external_data["data"] = get_data(LOCATIONS).to_dict(orient="records")
        external_data["timestamp"] = time.time()
        time.sleep(900)  # 15 minutes

# Lancer le thread pour récupérer les données toutes les 15 minutes
threading.Thread(target=update_external_data, daemon=True).start()

def process_sensor_data():
    """Consomme les données des capteurs et les enrichit avec les données externes"""
    print("Écoute des données des capteurs...")
    for message in consumer:
        sensor_data = message.value
        sensor_lat = sensor_data.get("lat")
        sensor_lon = sensor_data.get("lon")

        # Trouver les données météorologiques correspondantes
        matched_data = next(
            (entry for entry in external_data["data"] if entry["lat"] == sensor_lat and entry["lon"] == sensor_lon),
            None
        )

        if matched_data:
            enriched_data = {**sensor_data, **matched_data}
        else:
            enriched_data = sensor_data  # Si aucune donnée correspondante, envoyer brut

        print(f"Envoi des données enrichies : {enriched_data}")
        producer.send(ENRICHED_TOPIC, enriched_data)

if __name__ == "__main__":
    process_sensor_data()

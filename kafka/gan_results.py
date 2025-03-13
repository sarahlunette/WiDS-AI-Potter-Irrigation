from fastapi import FastAPI
from kafka import KafkaConsumer, KafkaProducer
import json
import torch  # Ou tensorflow selon ton modèle
import numpy as np

# Initialisation de FastAPI
app = FastAPI()

# Configuration Kafka
KAFKA_BROKER = "localhost:9092"
SENSOR_TOPIC = "sensor_data"
FORECAST_TOPIC = "gan_forecast_results"

consumer = KafkaConsumer(
    SENSOR_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Chargement du modèle GAN
MODEL_PATH = "models/gan_model.pth"  # Adapte le chemin à ton setup
gan_model = torch.load(MODEL_PATH)  # Ou autre méthode selon ton framework
gan_model.eval()

def generate_forecast(sensor_data):
    """
    Utilise le modèle GAN pour générer une prévision d'irrigation.
    """
    # Convertir les données en tenseur (adapte selon ton modèle)
    input_data = np.array([
        sensor_data["soil_moisture"],
        sensor_data["temperature"],
        sensor_data["humidity"]
    ])
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    # Prédiction avec le modèle GAN
    with torch.no_grad():
        prediction = gan_model(input_tensor).item()

    return {"forecast_irrigation": f"{prediction:.1f} mm", "confidence": 0.95}  # Score fictif

@app.post("/forecast-irrigation") # TODO: not sure it is useful to use kafka for this forecast
def forecast_irrigation(data: dict):
    """
    Reçoit les données du fermier, météo, sol et capteurs et retourne une prévision GAN.
    """
    sensor_data = data.get("sensor_data", {})
    
    if not sensor_data:
        return {"error": "Missing sensor data"}
    
    result = generate_forecast(sensor_data)

    # Publier la prévision sur Kafka
    producer.send(FORECAST_TOPIC, value=result)
    return result

# Lancer avec: uvicorn gan_forecast_api:app --host 0.0.0.0 --port 5002 --reload

from fastapi import FastAPI
import random

app = FastAPI()

@app.post("/predict")
def predict_irrigation(data: dict):
    """
    Simule une décision d'irrigation basée sur un modèle RL.
    """
    sector = data.get("sector", "Unknown")
    moisture = data.get("soil_moisture", 0)

    # Simulation d'un modèle RL (remplacez ceci par un appel réel à un modèle RL)
    valve_action = "OPEN" if moisture < 30 else "CLOSE"

    response = {
        "sector": sector,
        "command": valve_action,
        "confidence": round(random.uniform(0.7, 1.0), 2)  # Score fictif pour illustrer
    }
    return response

# Lancer avec: uvicorn rl_model_api:app --host 0.0.0.0 --port 5001 --reload

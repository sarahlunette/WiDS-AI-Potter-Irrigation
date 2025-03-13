from fastapi import FastAPI

app = FastAPI()

@app.get("/recommend-irrigation")
def get_recommendation():
    moisture_level = 0.4  # Example value from sensor
    if moisture_level < 0.3:
        return {"action": "Water needed", "amount": "High"}
    elif moisture_level < 0.6:
        return {"action": "Optimal", "amount": "None"}
    else:
        return {"action": "Reduce water", "amount": "Low"}

# Run with: uvicorn filename:app --reload

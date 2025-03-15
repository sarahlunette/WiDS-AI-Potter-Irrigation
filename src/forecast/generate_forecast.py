import numpy as np
import json
import torch
import torch.nn as nn

# Define Generator Model (Same as used in training)
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensures non-negative irrigation output
        )

    def forward(self, x):
        return self.model(x)

# Load the trained generator model
MODEL_PATH = "/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/models/gan/gan_generator.pth"

generator = Generator(input_dim=4)  # Ensure input_dim matches training setup
generator.load_state_dict(torch.load(MODEL_PATH))
generator.eval()

def preprocess_sensor_data(sensor_data):
    """Convert sensor data into a format suitable for GAN input."""
    features = np.array([
        sensor_data.get("temperature", 0),
        sensor_data.get("humidity", 0),
        sensor_data.get("soil_moisture", 0),
        sensor_data.get("solar_radiation", 0)
    ], dtype=np.float32)
    
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

def generate_forecast(sensor_data):
    """
    Generate a forecast using the GAN model based on real-time sensor data.
    """
    input_tensor = preprocess_sensor_data(sensor_data)

    with torch.no_grad():
        forecast = generator(input_tensor)  # Model output

    # Convert forecast to a dictionary (assuming a single output)
    forecast_value = forecast.item()
    forecast_result = {
        "predicted_irrigation": round(forecast_value, 2),
        "sensor_data_used": sensor_data
    }

    return forecast_result

if __name__ == "__main__":
    sample_sensor_data = {
        "temperature": 28,
        "humidity": 65,
        "soil_moisture": 40,
        "solar_radiation": 300
    }
    prediction = generate_forecast(sample_sensor_data)
    print(json.dumps(prediction, indent=2))

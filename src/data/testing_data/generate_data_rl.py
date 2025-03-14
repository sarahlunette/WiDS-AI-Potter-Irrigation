import numpy as np
import pandas as pd
import json

# ET Calculation Functions (from your script)
def extraterrestrial_radiation(lat, doy):
    """ Compute Ra (extraterrestrial radiation) using FAO-56 formula """
    Gsc = 0.0820  # Solar constant (MJ/m²/min)
    phi = np.radians(lat)
    delta = 0.409 * np.sin((2 * np.pi * doy / 365) - 1.39)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    omega_s = np.arccos(-np.tan(phi) * np.tan(delta))

    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        omega_s * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    return Ra

def hargreaves_et(tmin, tmax, Ra):
    """ Compute evapotranspiration using the Hargreaves equation """
    if Ra is None or Ra <= 0:
        return None  # Invalid values

    tmean = (tmin + tmax) / 2
    temp_diff = tmax - tmin
    if temp_diff <= 0:
        return None  # Ensure valid temperature range

    ET0 = 0.0023 * Ra * np.sqrt(temp_diff) * (tmean + 17.8)
    return max(0, ET0)  # Ensure ET is not negative

def fetch_evapotranspiration_data(tmean, lat):
    """ Compute Evapotranspiration using the manual Hargreaves method """
    tmin = tmean - 2
    tmax = tmean + 2
    doy = pd.Timestamp.today().day_of_year  # Get the current day of the year
    Ra = extraterrestrial_radiation(lat, doy)  # Compute Ra
    return hargreaves_et(tmin, tmax, Ra)

# Function to Generate Samples
def generate_training_samples(n_samples=100000, lat=48.8566, lon=2.3522, location="Paris"):
    np.random.seed(42)  # Ensure reproducibility
    samples = []

    for _ in range(n_samples):
        # Simulate sensor values
        tmean = np.random.uniform(5, 35)  # Temperature (°C)
        humidity = np.random.uniform(30, 90)  # Humidity (%)
        soil_moisture = np.random.uniform(5, 45)  # Soil Moisture (%)

        # Compute ET₀
        et0 = fetch_evapotranspiration_data(tmean, lat)
        if et0 is None:
            continue  # Skip invalid samples

        # Define irrigation action based on ET₀ & soil moisture
        if soil_moisture < 20:
            irrigation = min(10, et0 * 1.5)  # Open valve more
        elif soil_moisture < 30:
            irrigation = min(5, et0)  # Moderate irrigation
        else:
            irrigation = max(0, et0 - 2)  # Reduce irrigation when soil is wet

        # Format the sample as JSON
        sample = {
            "sector": "Vineyard A",
            "soil_moisture": round(soil_moisture, 1),
            "temperature": round(tmean, 2),
            "humidity": round(humidity, 1),
            "location": location,
            "lat": lat,
            "lon": lon,
            "evapotranspiration": round(et0, 6),
            "irrigation": round(irrigation, 2)
        }

        samples.append(sample)

    return samples

# Generate data
training_data = generate_training_samples(1000)

# Save to JSON file
with open("training_data.json", "w") as f:
    json.dump(training_data, f, indent=4)

print("Training data saved to training_data.json!")

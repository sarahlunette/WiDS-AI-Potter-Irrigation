import gymnasium as gym
import numpy as np
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from gymnasium.spaces import Box

# TODO: remove the functions and import them from the right scripts
# TODO: get iot_sensor_data from kafka (create pipeline)
# API for weather data
def get_weather_forecast(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,soil_moisture"
    response = requests.get(url).json()
    return response["current"]

# Simulating IoT sensor data
def get_iot_sensor_data(sector):
    # Example: Each sector has a different soil moisture sensor and valve range
    sensor_data = {
        "soil_moisture": np.random.uniform(10, 50),  # Simulated moisture data
        "temperature": np.random.uniform(20, 40),  # Simulated temperature data
        "valve_range": (0, 10) if sector == 0 else (0, 5)  # Different valve ranges per sector
    }
    return sensor_data

# Custom Environment for Multi-Sector Smart Irrigation
class IrrigationEnv(gym.Env):
    def __init__(self, num_sectors=2):
        super(IrrigationEnv, self).__init__()
        self.num_sectors = num_sectors
        
        # Continuous action space: Valve opening between min-max per sector
        self.action_space = Box(low=np.array([0, 0]), high=np.array([10, 5]), dtype=np.float32)
        
        # Observation space: Soil moisture, temperature for each sector
        self.observation_space = Box(low=0, high=100, shape=(num_sectors * 2,), dtype=np.float32)

    def reset(self):
        self.state = []
        for sector in range(self.num_sectors):
            sensor_data = get_iot_sensor_data(sector)
            self.state.extend([sensor_data["soil_moisture"], sensor_data["temperature"]])
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        reward = 0
        new_state = []
        done = False

        for sector in range(self.num_sectors):
            sensor_data = get_iot_sensor_data(sector)
            
            # Update soil moisture based on irrigation action
            applied_water = action[sector]
            sensor_data["soil_moisture"] += applied_water - np.random.uniform(1, 3)  # Simulating evaporation
            
            # Keep track of new state
            new_state.extend([sensor_data["soil_moisture"], sensor_data["temperature"]])
            
            # Reward function: Closer to 30% soil moisture, the better
            reward -= abs(sensor_data["soil_moisture"] - 30)
            
            # Check if sector reaches critical moisture levels
            if sensor_data["soil_moisture"] < 5 or sensor_data["soil_moisture"] > 80:
                done = True

        return np.array(new_state, dtype=np.float32), reward, done, {}

# Train RL Model
env = DummyVecEnv([lambda: IrrigationEnv(num_sectors=2)])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# TODO: save model

# Predict best irrigation action
obs = env.reset()
action, _ = model.predict(obs)
print(f"Optimal irrigation actions per sector: {action}")

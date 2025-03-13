import gymnasium as gym
import numpy as np
import json
import time
from kafka import KafkaConsumer
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from gymnasium.spaces import Box

# Kafka Configuration
KAFKA_BROKER = 'localhost:9092'
SENSOR_TOPIC = 'enriched_sensor_data'

# Kafka Consumer for enriched sensor data
consumer = KafkaConsumer(
    SENSOR_TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Custom RL Environment with dynamic feature size
class DynamicIrrigationEnv(gym.Env):
    def __init__(self):
        super(DynamicIrrigationEnv, self).__init__()
        self.state = None
        self.num_features = None  # Will be initialized with first data point
        
    def initialize_env(self, first_data):
        self.num_features = len(first_data)
        self.action_space = Box(low=0, high=10, shape=(1,), dtype=np.float32)  # Valve opening (continuous)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32)
        print(f"Initialized environment with {self.num_features} features.")

    def reset(self):
        return self.state if self.state is not None else np.zeros(self.num_features, dtype=np.float32)

    def step(self, action):
        reward = -abs(self.state[0] - 30)  # Reward based on soil moisture closeness to 30%
        done = False  # Continuous learning, so no terminal state
        return self.state, reward, done, {}

# Initialize RL environment dynamically
env = DynamicIrrigationEnv()
env = DummyVecEnv([lambda: env])
model = PPO("MlpPolicy", env, verbose=1)

# Training loop with Kafka
def train_incrementally():
    print("Listening for enriched sensor data...")
    for message in consumer:
        data = message.value
        feature_vector = np.array(list(data.values()), dtype=np.float32)  # Convert to numpy array
        
        # Initialize env if not already done
        if env.envs[0].num_features is None:
            env.envs[0].initialize_env(feature_vector)
        
        env.envs[0].state = feature_vector  # Update state
        obs = env.reset()
        
        # Train incrementally with new data
        model.learn(total_timesteps=1)
        action, _ = model.predict(obs)
        print(f"Optimal irrigation action: {action}")
        
        time.sleep(2)  # Wait for next sensor data

if __name__ == "__main__":
    train_incrementally()

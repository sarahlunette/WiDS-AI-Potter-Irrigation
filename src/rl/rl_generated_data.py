import json
import numpy as np
import random
import pickle
import os
import sys

# Load historical irrigation data from JSON file
def load_training_data(filename):
    with open(filename, "r") as file:
        return json.load(file)

# Define action space (irrigation levels)
actions = np.linspace(0, 5, num=11)  # Irrigation amounts (0 to 5 mm)

class RL_IrrigationAgent:
    def __init__(self):
        self.q_table = {}  # Store Q-values
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def get_state(self, data):
        """Extracts the state from sensor data (soil_moisture, evapotranspiration)."""
        return (round(data["soil_moisture"], 1), round(data["evapotranspiration"], 1))

    def choose_action(self, state):
        """Selects an action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:  # Exploration
            return random.choice(actions)
        return max(actions, key=lambda a: self.q_table.get((state, a), 0))  # Exploitation

    def update_q_table(self, state, action, reward):
        """Updates the Q-table based on the reward received."""
        old_value = self.q_table.get((state, action), 0)
        best_next_q = max([self.q_table.get((state, a), 0) for a in actions])
        self.q_table[(state, action)] = old_value + self.alpha * (reward + self.gamma * best_next_q - old_value)

    def train(self, data):
        """Trains the RL agent using historical irrigation data."""
        for sample in data:
            state = self.get_state(sample)
            action = sample["irrigation"]

            # Reward function: encourage maintaining moisture at 30-35% while minimizing irrigation
            reward = -abs(sample["soil_moisture"] - 32) - action

            self.update_q_table(state, action, reward)

    # TODO: change path abs to relative ?
    def save_model(self, path='/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/models/rl/', filename="q_table.pkl"):
        """Saves the trained Q-table."""
        with open(path + filename, "wb") as file:
            pickle.dump(self.q_table, file)

# Load training data
# TODO: change path
path = '/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/src/data/testing_data/training_data.json'
data = load_training_data(path)

# Train RL agent
agent = RL_IrrigationAgent()
agent.train(data)

# Save trained Q-table
agent.save_model()
print("✅ RL model trained and saved as 'q_table.pkl'.")

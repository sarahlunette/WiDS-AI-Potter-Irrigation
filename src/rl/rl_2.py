import gym
import numpy as np

class IrrigationEnv(gym.Env):
    def __init__(self):
        self.state = np.array([0.5, 0.3])  # Soil moisture, temperature
        self.action_space = gym.spaces.Discrete(3)  # No water, low water, high water
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def step(self, action):
        if action == 1:
            self.state[0] += 0.1  # Increase soil moisture
        elif action == 2:
            self.state[0] += 0.2
        self.state[0] = max(0, min(1, self.state[0]))  # Keep in range
        reward = self.state[0] - abs(self.state[1] - 0.5)  # Balance soil moisture & temp
        return self.state, reward, False, {}

env = IrrigationEnv()

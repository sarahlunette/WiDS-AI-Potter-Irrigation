class DynamicIrrigationEnv(gym.Env):
    def __init__(self):
        super(DynamicIrrigationEnv, self).__init__()
        self.state = None
        self.num_features = None  # Initialize dynamically
    
    def initialize_env(self, first_data):
        self.num_features = len(first_data) + 1  # +1 for ET₀
        self.action_space = Box(low=0, high=10, shape=(1,), dtype=np.float32)  # Valve opening
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32)
        print(f"Initialized environment with {self.num_features} features.")

    def reset(self):
        return self.state if self.state is not None else np.zeros(self.num_features, dtype=np.float32)

    def step(self, action):
        soil_moisture = self.state[0]  
        et0 = self.state[-1]  # Extract ET₀ from last feature
        
        # Reward based on keeping soil moisture around 30%
        reward = -abs(soil_moisture - 30) - et0  # Penalize high ET₀ to save water

        done = False  # Continuous learning, no terminal state
        return self.state, reward, done, {}

def train_incrementally():
    print("Listening for enriched sensor data...")
    for message in consumer:
        data = message.value
        doy = pd.Timestamp.today().day_of_year  # Get day of year
        
        # Compute ET₀ using scientific formulas
        et0 = compute_evapotranspiration(
            tmean=data["temperature"],
            tmax=data["temperature"] + 2,
            tmin=data["temperature"] - 2,
            humidity=data["humidity"],
            wind_speed=2,  # Assume 2m/s wind speed
            lat=data["lat"],
            doy=doy
        )

        # Convert data to numpy array & append ET₀
        feature_vector = np.array(list(data.values()) + [et0], dtype=np.float32)

        # Initialize env if not already done
        if env.envs[0].num_features is None:
            env.envs[0].initialize_env(feature_vector)

        env.envs[0].state = feature_vector  # Update state
        obs = env.reset()
        
        # Train incrementally with new data
        model.learn(total_timesteps=1)
        action, _ = model.predict(obs)
        print(f"Optimal irrigation action: {action}, ET₀: {et0}")

        time.sleep(2)  # Wait for next sensor data

#### **Step 2: Data Processing**
'''- **Cleaning & structuring** data from multiple sources.
- **Handling missing values**, standardizing formats.
- **Merging datasets** (weather, soil, satellite, crop info).
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Data Cleaning and Processing
irrigation_data = pd.read_csv("irrigation_data.csv")

# Handling Missing Values
irrigation_data.fillna(irrigation_data.mean(), inplace=True)

# Normalizing Data
scaler = MinMaxScaler()
irrigation_data_scaled = pd.DataFrame(scaler.fit_transform(irrigation_data), columns=irrigation_data.columns)

# Saving Processed Data
irrigation_data_scaled.to_csv("processed_irrigation_data.csv", index=False)
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load training data
with open("/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/src/data/testing_data/training_data.json", "r") as file:
    data = json.load(file)

# Extract features and labels
features = np.array([[entry["soil_moisture"], entry["temperature"], entry["humidity"], entry["evapotranspiration"]] for entry in data], dtype=np.float32)
labels = np.array([entry["irrigation"] for entry in data], dtype=np.float32)

# Convert to PyTorch tensors
X_train = torch.tensor(features)
y_train = torch.tensor(labels).view(-1, 1)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define GAN Model
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator(input_dim=4)
discriminator = Discriminator(input_dim=4)

# Define optimizers and loss function
optimizer_G = optim.Adam(generator.parameters(), lr=0.002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.002)
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(1000):
    for real_x, real_y in dataloader:
        batch_size = real_x.shape[0]
        
        # Generate fake irrigation values
        fake_y = generator(real_x)
        
        # Train discriminator
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        real_data = torch.cat((real_x, real_y), dim=1)
        fake_data = torch.cat((real_x, fake_y.detach()), dim=1)
        
        loss_real = loss_fn(discriminator(real_data), real_labels)
        loss_fake = loss_fn(discriminator(fake_data), fake_labels)
        
        optimizer_D.zero_grad()
        (loss_real + loss_fake).backward()
        optimizer_D.step()
        
        # Train generator
        generated_data = torch.cat((real_x, fake_y), dim=1)
        loss_G = loss_fn(discriminator(generated_data), real_labels)
        
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss D: {loss_real + loss_fake}, Loss G: {loss_G}")

# Save model
torch.save(generator.state_dict(), "/Users/sarahlenet/Desktop/WiDS-AI-Potter-Irrigation/models/gan/gan_generator.pth")
print("GAN model saved.")

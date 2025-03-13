import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU

# TODO: complexify models

# Generator Model
def build_generator():
    model = Sequential([
        Dense(16, activation="relu", input_dim=10),
        Dense(32, activation=LeakyReLU(0.2)),
        Dense(1, activation="tanh")
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = Sequential([
        Dense(32, activation=LeakyReLU(0.2), input_dim=1),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    return model

# GAN Training
def train_gan(generator, discriminator, epochs=1000, batch_size=32):
    gan = tf.keras.models.Sequential([generator, discriminator])
    discriminator.compile(loss="binary_crossentropy", optimizer="adam")
    gan.compile(loss="binary_crossentropy", optimizer="adam")

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, 10))
        generated_data = generator.predict(noise)
        
        real_data = np.random.uniform(10, 100, (batch_size, 1))  # Real water demand
        labels_real = np.ones((batch_size, 1))
        labels_fake = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_data, labels_real)
        d_loss_fake = discriminator.train_on_batch(generated_data, labels_fake)
        
        noise = np.random.normal(0, 1, (batch_size, 10))
        g_loss = gan.train_on_batch(noise, labels_real)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Generator Loss = {g_loss}, Discriminator Loss = {d_loss_real + d_loss_fake}")

# Initialize & Train
generator = build_generator()
discriminator = build_discriminator()
train_gan(generator, discriminator)

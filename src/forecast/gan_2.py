import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.models import Model

def build_generator():
    input_noise = Input(shape=(100,))
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(input_noise)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(1, activation="linear")(x)  # Water demand prediction
    return Model(input_noise, x)

generator = build_generator()
generator.summary()

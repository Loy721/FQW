import keras_tuner as kt

from keras.layers import Dense
from keras.src.applications.densenet import layers
from keras_tuner.src.backend import keras
from tensorboard.plugins.hparams import api as hp

import numpy as np
import datetime


def build_model(hp):
    model = keras.Sequential()

    model.add(layers.Flatten())

    model.add(
        Dense(
            # Define the hyperparameter.
            units=hp.Int("units", min_value=32,
                         max_value=512,
                         step=32),
            activation="relu",
        )
    )
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],

    )
    return model


if __name__ == "__main__":
    print(build_model(kt.HyperParameters()))

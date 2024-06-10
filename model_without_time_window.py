import numpy as np
import pandas as pd
import tensorflow as tf
from keras import activations
from keras import callbacks
from keras.layers import Dense
from tensorflow import keras
from sklearn.model_selection import TimeSeriesSplit

if __name__ == "__main__":
    ds = pd.read_excel('SE.xls', skiprows=6)
    data = ds['T'].ffill()
    tscv = TimeSeriesSplit(n_splits=4)

    initializer = tf.initializers.GlorotNormal()
    regularizer = tf.keras.regularizers.l1(1e-5)
    layers = [
        tf.keras.layers.Dense(80, activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_dim=(None,), kernel_initializer=initializer, kernel_regularizer=regularizer),
        #tf.keras.layers.AlphaDropout(0.0012),
        tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer, kernel_regularizer=regularizer),
        #tf.keras.layers.AlphaDropout(0.0012),
        tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer, kernel_regularizer=regularizer),
        #tf.keras.layers.AlphaDropout(0.0012),
        tf.keras.layers.Dense(1)
    ]
    model = tf.keras.Sequential(layers)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mae')


    es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100)
    model.fit(
        x=x_train,
        y=y_train,
        epochs=200,
        callbacks=[es_callback],
        validation_data=(x_valid, y_valid)
    )
    print("end fit")

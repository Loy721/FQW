import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from keras import callbacks

def get_train_data():
    ds = pd.read_excel('SE.xls', skiprows=6)
    data = ds['T'].ffill()
    train_data = data[:]
    dataset_train = train_data.values
    dataset_train = np.reshape(dataset_train, (-1, 1))

    ln = 40
    X_train, y_train = [], []
    for i in range(ln, len(dataset_train)):
        X_train.append(dataset_train[i - ln:i, 0])
        y_train.append(dataset_train[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train

def build_rnn_model(hp, X_train):
    rnn_model = tf.keras.models.Sequential()

    # Добавление указанного числа слоев
    num_layers = hp.Int('num_layers', min_value=1, max_value=5)
    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=16, max_value=128, step=16)
        activation = hp.Choice(f'activation_{i}', ["relu", "tanh"])

        # Добавление слоя RNN
        rnn_model.add(tf.keras.layers.SimpleRNN(units=units, activation=activation, return_sequences=True if i < num_layers - 1 else False, input_shape=(X_train.shape[1], 1)))
        rnn_model.add(keras.layers.Dropout(0.2))
    # Выходной слой
    rnn_model.add(tf.keras.layers.Dense(1))

    # Компиляция модели
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2)
    rnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=["mean_absolute_error"]
    )

    return rnn_model

if __name__ == "__main__":
    X_train, y_train = get_train_data()
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_rnn_model(hp, X_train),
        objective="val_mean_absolute_error",
        max_trials=10,
        executions_per_trial=2,
        overwrite=True,
        directory="hyperparams_rnn",
        project_name="для_отчета_40_rnn"
    )

    tuner.search_space_summary()
    es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=2)
    tuner.search(
        X_train,
        y_train,
        batch_size=32,
        epochs=200,
        validation_split=0.2,
        callbacks=[es_callback]
    )

    best_params = tuner.get_best_hyperparameters()[0].values
    print(best_params)

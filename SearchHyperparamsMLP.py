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

    # Normalization
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_train = dataset_train#scaler.fit_transform(dataset_train)
    ln = 15
    X_train, y_train = [], []
    for i in range(ln, len(scaled_train)):
        X_train.append(scaled_train[i - ln:i, 0])
        y_train.append(scaled_train[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train

def build_model(hp):
    mlp_model = tf.keras.models.Sequential()
    mlp_model.add(tf.keras.layers.Dense(
        units=hp.Int("units_l1", min_value=16, max_value=1024, step=16),
        activation=hp.Choice("activation_l1", ["relu", "tanh", "sigmoid"])))
    mlp_model.add(tf.keras.layers.Dense(1))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2)
    mlp_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=["mean_absolute_error"]
    )
    return mlp_model

if __name__ == "__main__":
    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective="val_mean_absolute_error",
        max_trials=10,
        executions_per_trial=2,
        overwrite=True,
        directory="hyperparams_MLP",
        project_name="для_отчета_1"
    )

    tuner.search_space_summary()
    x_train, y_train = get_train_data()
    es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=6)
    tuner.search(
        x_train,
        y_train,
        batch_size=16,
        epochs=200,
        validation_split=0.2,
        callbacks=[es_callback]
    )

    best_params = tuner.get_best_hyperparameters()[0].values
    print(best_params)
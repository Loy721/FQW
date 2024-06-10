import pandas as pd
import os
import math
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    layers
    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)

    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

def to_sequences(seq_size, obs):
    x = []
    y = []

    for i in range(len(obs)-SEQUENCE_SIZE):
        #print(i)
        window = obs[i:(i+SEQUENCE_SIZE)]
        after_window = obs[i+SEQUENCE_SIZE]
        window = [[x] for x in window]
        #print("{} - {}".format(window,after_window))
        x.append(window)
        y.append(after_window)

    return np.array(x),np.array(y)

def normalize_data(train_data, test_data):
    min_val = np.min(train_data)
    max_val = np.max(train_data)
    normalized_train_data = (train_data - min_val) / (max_val - min_val)
    normalized_test_data = (test_data - min_val) / (max_val - min_val)
    return normalized_train_data, normalized_test_data, min_val, max_val

def denormalize_data(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val) + min_val
from sklearn.metrics import mean_absolute_error

if __name__ == '__main__': #лучшая 1.25
    df = pd.read_excel('SE.xls', skiprows=6)
    df['T'] = df['T'].ffill()
    df['P'] = df['P'].ffill()
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', dayfirst=True)
    df['Day_of_year'] = (df['Date'].dt.dayofyear.astype(str) + df['Date'].dt.hour.astype(str)).astype(int)
    train_prosents = 0.8
    data = df.loc[:, ['T']]
    train_size = math.floor(data.size * train_prosents)
    df_train = data[:train_size]
    df_test = data[train_size:]
    spots_train = df_train.to_numpy().reshape(-1, 1)
    spots_test = df_test.to_numpy().reshape(-1, 1)

    spots_train = spots_train.flatten().tolist()
    spots_test = spots_test.flatten().tolist()

    SEQUENCE_SIZE = 15
    x_train,y_train = to_sequences(SEQUENCE_SIZE,spots_train)
    x_test,y_test = to_sequences(SEQUENCE_SIZE,spots_test)

    print("Shape of training set: {}".format(x_train.shape))
    print("Shape of test set: {}".format(x_test.shape))

    input_shape = x_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="mae",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4)
    )
    #model.summary()

    callbacks = [keras.callbacks.EarlyStopping(patience=10, \
                                               restore_best_weights=True)]

# Normalize training and testing data
    x_train_normalized, x_test_normalized, x_min, x_max = normalize_data(x_train, x_test)
    y_train_normalized, y_test_normalized, y_min, y_max = normalize_data(y_train, y_test)
    print(x_train_normalized)
    print(x_train)
    model.fit(
        x_train_normalized,
        y_train_normalized,
        validation_split=0.2,
        epochs=4,
        batch_size=64,
        callbacks=callbacks,
    )

    y_pred_normalized = model.predict(x_test_normalized)
    y_pred = denormalize_data(y_pred_normalized, y_min, y_max)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE:", mae)
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from keras import callbacks

if __name__ == "__main__":
    ds = pd.read_excel('SE.xls', skiprows=6)
    ds = ds.dropna(subset=['T'])
    data = ds['T']
    training_data_len = math.ceil(len(data) * .8)
    train_data = data[:training_data_len]
    test_data = data[training_data_len:]
    dataset_train = train_data.values
    dataset_train = np.reshape(dataset_train, (-1, 1))
    # normalization
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_train = scaler.fit_transform(dataset_train)
    print(scaled_train[:5])

    dataset_test = test_data.values
    # Reshaping 1D to 2D array
    dataset_test = np.reshape(dataset_test, (-1, 1))
    # Normalizing values between 0 and 1
    scaled_test = scaler.fit_transform(dataset_test)

    X_train = []
    y_train = []
    for i in range(50, len(scaled_train)):
        X_train.append(scaled_train[i - 50:i, 0])
        y_train.append(scaled_train[i, 0])

    X_test = []
    y_test = []
    for i in range(50, len(scaled_test)):
        X_test.append(scaled_test[i - 50:i, 0])
        y_test.append(scaled_test[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    print("X_train :", X_train.shape, "y_train :", y_train.shape)

    X_test, y_test = np.array(X_test), np.array(y_test)

    # Reshaping
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    regressor = Sequential()

    # adding RNN layers and dropout regularization
    regressor.add(SimpleRNN(units=50,
                            activation="tanh",
                            return_sequences=True,
                            input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(SimpleRNN(units=50,
                            activation="tanh",
                            return_sequences=True))

    regressor.add(SimpleRNN(units=50,
                            activation="tanh",
                            return_sequences=True))

    regressor.add(SimpleRNN(units=50))

    # adding the output layer
    regressor.add(Dense(units=1,  activation="tanh"))

    # compiling RNN
    regressor.compile(optimizer="adam",
                      loss="mean_absolute_error", metrics=["mean_absolute_error"])

    #fitting the model
    es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    regressor.fit(X_train, y_train, epochs=10, validation_split=0.2, callbacks=[es_callback])
    regressor.summary()

    y_RNN = regressor.predict(X_test)
    y_RNN_O = scaler.inverse_transform(y_RNN)

    fig, axs = plt.subplots(3, figsize=(18, 12), sharex=True, sharey=True)
    fig.suptitle('Model Predictions')
    axs[0].plot(range(0, 406), np.array(train_data[32000:]), label="train_data", color="b")
    axs[0].plot(range(406, 406 + 200),  np.array(test_data[:200]), label="test_data", color="g")#8100
    axs[0].plot(range(406 + 50, 406 + 200),  np.array(y_RNN_O[:150]), label="y_RNN", color="brown")
    axs[0].legend()
    axs[0].title.set_text("Basic RNN")
    plt.show()

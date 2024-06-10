# -*- coding: utf-8 -*-

import math

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks
from keras.layers import Dense
from tensorflow import keras


class NeuralNetwork:
    def normalize_data(self):
        to = math.floor(self.data.size * (self.train_prosents + self.validate_prosents))
        train_data = self.data[:to]
        min_val = np.min(train_data)
        max_val = np.max(train_data)
        self.data = ((self.data - min_val) / (max_val - min_val)) * 2 - 1
        return min_val, max_val

    def denormalize_data(self, normalized_data):
        return ((normalized_data + 1) / 2) * (self.max_normalize - self.min_normalize) + self.min_normalize

    def __init__(self, n_steps, n_futere, data):
        self.n_steps = n_steps  # представляет собой количество временных шагов, которые используются для формирования последовательности
        self.n_futere = n_futere
        self.data = data
        self.train_prosents = 0.64
        self.validate_prosents = 0.16
        self.test_prosents = 0.2
        self.batch_size = 128
        self.model = None
        self.min_normalize, self.max_normalize = self.normalize_data()

    def split_sequence(self, data):
        x, y = list(), list()
        for i in range(len(data)):
            # поиск конца
            end_ix = i + self.n_steps
            # проверяем выход за границы
            if end_ix > len(data) - self.n_futere:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix:end_ix + 1]
            x.append(seq_x)
            y.append(seq_y)
        x_arr, y_arr = np.array(x), np.array(y)
        indices = np.random.permutation(len(x_arr))
        x_arr = x_arr[indices]
        y_arr = y_arr[indices]
        return x_arr, y_arr

    def get_train_dataset(self, batch_size=16):
        train_size = math.floor(self.data.size * self.train_prosents)
        x, y = self.split_sequence(self.data[:train_size])
        return x, y

    def get_validate_dataset(self, batch_size=16):
        size_from = math.floor(self.data.size * self.train_prosents)
        validate_size = math.floor(self.data.size * self.validate_prosents)
        size_to = size_from + validate_size
        return self.split_sequence(self.data[size_from:size_to])

    def get_test_data(self):
        size_from = math.floor(self.data.size * (self.train_prosents + self.validate_prosents))
        x, y = self.split_sequence(self.data[size_from:])
        return x, y


class MLP(
    NeuralNetwork):  # почему то из-за батч нормализации очень плохо обучаетясь, добавил LeakyRelu, регулязатор, инишалайхер
    def __init__(self, n_steps, n_futere, data):
        NeuralNetwork.__init__(self, n_steps, n_futere, data)
        initializer = tf.initializers.GlorotNormal()
        regularizer = tf.keras.regularizers.l1(1e-5)
        layers = [
            tf.keras.layers.Dense(80, activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_dim=self.n_steps,
                                  kernel_initializer=initializer, kernel_regularizer=regularizer),
            # tf.keras.layers.AlphaDropout(0.0012),
            tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer,
                                  kernel_regularizer=regularizer),
            # tf.keras.layers.AlphaDropout(0.0012),
            tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer,
                                  kernel_regularizer=regularizer),
            # tf.keras.layers.AlphaDropout(0.0012),
            tf.keras.layers.Dense(1)
        ]
        model = tf.keras.Sequential(layers)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')  # TODO  weight_decay=0.1
        self.model = model

    def fitModel(self, EPOCHS=1000):
        x_train, y_train = self.get_train_dataset()
        x_valid, y_valid = self.get_validate_dataset()
        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=24)
        self.model.fit(
            x=x_train,
            y=y_train,
            epochs=EPOCHS,
            callbacks=[es_callback],
            validation_data=(x_valid, y_valid)
        )
        print("end fit")
        return self.model

    def predict(self, dat):
        ls = []
        for i in range(0, self.n_futere):
            ls.append(self.model.predict(dat.reshape(1, self.n_steps), verbose=0)[0])
            dat[:self.n_steps - 1] = dat[1:self.n_steps]
            dat[self.n_steps - 1] = ls[i]
        return ls

    def get_predict(self):
        x, y = self.get_test_data()
        for i in range(10, 100, 5):
            print("Predict: ", self.predict(x[i])[0])
            print("Real value: ", y[i])
            print("-----------------------------------")


class LSTM(NeuralNetwork):

    def __init__(self, n_steps, n_futere, data):
        NeuralNetwork.__init__(self, n_steps, n_futere, data)

    def __initModel(self):
        hyperparameters = {'num_layers': 5, 'units_0': 32, 'activation_0': 'tanh', 'lr': 0.0025200304939859757,
                           'units_1': 16, 'activation_1': 'tanh', 'units_2': 48, 'activation_2': 'relu',
                           'units_3': 64, 'activation_3': 'tanh', 'units_4': 64, 'activation_4': 'relu'}

        simple_lstm_model = keras.models.Sequential()

        for i in range(hyperparameters['num_layers']):
            units = hyperparameters[f'units_{i}']
            activation = hyperparameters[f'activation_{i}']

            # Добавление слоя LSTM
            simple_lstm_model.add(tf.keras.layers.LSTM(units=units, activation=activation,
                                                       return_sequences=True if i < hyperparameters[
                                                           'num_layers'] - 1 else False, input_shape=(self.n_steps, 1)))
        simple_lstm_model.add(tf.keras.layers.Dense(1))
        simple_lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0025200304939859757),
                                  loss='mse')  # оптимизируем
        # среднюю абсолютную ошибку с помощью adam
        return simple_lstm_model

    def fitModel(self, EPOCHS=200):
        x_train, y_train = self.get_train_dataset()
        lstm_model = self.__initModel()
        x_valid, y_valid = self.get_validate_dataset()
        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0,
                                              patience=5)
        lstm_model.fit(x=x_train, y=y_train,
                       epochs=EPOCHS,
                       callbacks=[es_callback],
                       validation_data=(x_valid, y_valid)
                       )
        return lstm_model

    def predict(self, dat):
        ls = []
        dat = dat.reshape((1, self.n_steps, 1))
        for i in range(0, self.n_futere):
            ls.append(self.model.predict(dat, verbose=0))
            dat[:, :self.n_steps - 1:, :] = dat[:, 1:self.n_steps:, :]
            dat[0, self.n_steps - 1, 0] = ls[i]
        return np.array(ls).reshape(self.n_futere)

    def get_predict(self):
        x, y = self.get_test_data()
        for i in range(10, 100, 5):
            print("Predict: ", self.predict(x[i]))
            print("Real value: ", y[i])
            print("-----------------------------------")


class RNN(NeuralNetwork):
    def __init__(self, n_steps, n_futere, data):
        NeuralNetwork.__init__(self, n_steps, n_futere, data)

    def __initModel(self):

        hyperparameters = {'num_layers': 1, 'units_0': 80, 'activation_0': 'tanh', 'lr': 0.000555783015174179,
                           'units_1': 16, 'activation_1': 'tanh', 'units_2': 128, 'activation_2': 'tanh',
                           'units_3': 32, 'activation_3': 'tanh', 'units_4': 112, 'activation_4': 'relu'}

        rnn_model = keras.models.Sequential()

        for i in range(hyperparameters['num_layers']):
            units = hyperparameters[f'units_{i}']
            activation = hyperparameters[f'activation_{i}']

            # Добавление слоя SimpleRNN
            rnn_model.add(tf.keras.layers.SimpleRNN(units=units, activation=activation,
                                                    return_sequences=True if i < hyperparameters[
                                                        'num_layers'] - 1 else False, input_shape=(self.n_steps, 1)))
            rnn_model.add(keras.layers.Dropout(0.2))

        # adding the output layer
        rnn_model.add(Dense(units=1))

        # compiling RNN
        rnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000555783015174179), loss='mae')
        return rnn_model

    def fitModel(self, EPOCHS=200):
        x_train, y_train = self.get_train_dataset()
        x_valid, y_valid = self.get_validate_dataset()

        rnn_model = self.__initModel()

        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0,
                                              patience=5)
        rnn_model.fit(x=x_train,
                      y=y_train,
                      callbacks=[es_callback],
                      epochs=EPOCHS,
                      validation_data=(x_valid, y_valid)
                      )
        return rnn_model

    def predict(self, dat):
        ls = []
        dat = dat.reshape((1, self.n_steps, 1))
        for i in range(0, self.n_futere):
            ls.append(self.model.predict(dat, verbose=0))
            dat[:, :self.n_steps - 1:, :] = dat[:, 1:self.n_steps:, :]
            dat[0, self.n_steps - 1, 0] = ls[i]
        return np.array(ls).reshape(self.n_futere)

    def get_predict(self):
        x, y = self.get_test_data()
        for i in range(10, 100, 5):
            print("Predict: ", self.predict(x[i]))
            print("Real value: ", y[i])
            print("-----------------------------------")


from sklearn.metrics import mean_absolute_error, mean_squared_error


def mlp(data):
    mlp = MLP(15, 1, data)
    model = mlp.fitModel()
    # model.save('mlp_model')
    x, y = mlp.get_test_data()
    print("evaluate: ", model.evaluate(x, y))
    y_pred = model.predict(x)
    mae = np.mean(np.abs(mlp.denormalize_data(y) - mlp.denormalize_data(y_pred)))
    rmse = np.sqrt(mean_squared_error(mlp.denormalize_data(y), mlp.denormalize_data(y_pred)))
    print("RMSE: ", rmse)
    print("MAE ", mae)


def lstm(data):
    lstm = LSTM(15, 1, data)
    model = lstm.fitModel()
    x, y = lstm.get_test_data()
    print(model.evaluate(x, y))


def rnn(data):
    rnn = RNN(15, 1, data)
    model = rnn.fitModel()
    x, y = rnn.get_test_data()
    print(model.evaluate(x, y))


def predict(model, dat):
    ls = []
    for i in range(0, 7):
        ls.append(model.predict(dat.reshape(1, 15), verbose=0)[0])
        dat[:15 - 1] = dat[1:15]
        dat[15 - 1] = ls[i]
    return ls


def get_predict(x, y, model):
    for i in range(10, 100, 5):
        print(predict(model, x[i])[0])
        print("Real value: ", y[i])
        print("-----------------------------------")


if __name__ == "__main__":
    ds = pd.read_excel('SE.xls', skiprows=6)
    data = ds['T'].ffill()
    # ds = pd.read_csv('weatherHistory.csv')
    # ds = ds.dropna()
    # data = ds['Temperature (C)']
    # print(x[0])
    # loaded_model = tf.keras.models.load_model('mlp_model')
    # get_predict(x, y, loaded_model)
    mlp(data)
    # lstm(data)
    # rnn(data)

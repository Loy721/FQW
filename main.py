# -*- coding: utf-8 -*-

import math

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import activations
from keras import callbacks
from keras.layers import Dense
from tensorflow import keras


class NeuralNetwork:
    def __init__(self, n_steps, n_futere, data):
        self.n_steps = n_steps  # представляет собой количество временных шагов, которые используются для формирования последовательности
        self.n_futere = n_futere
        self.data = data
        self.train_prosents = 0.75
        self.validate_prosents = 0.2
        self.test_prosents = 0.05
        self.batch_size = 16

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
        return np.array(x), np.array(y)

    def get_train_dataset(self, batch_size=16):
        train_size = math.floor(self.data.size * self.train_prosents)
        return self.split_sequence(self.data[:train_size])

    def get_validate_dataset(self, batch_size=16):
        size_from = math.floor(self.data.size * self.train_prosents)
        validate_size = math.floor(self.data.size * self.validate_prosents)
        size_to = size_from + validate_size
        return self.split_sequence(self.data[size_from:size_to])

    def get_test_data(self):
        size_from = math.floor(self.data.size * (self.train_prosents + self.validate_prosents))
        x, y = self.split_sequence(self.data[size_from:])
        return x, y


class MLP(NeuralNetwork):
    def __init__(self, n_steps, n_futere, data):
        NeuralNetwork.__init__(self, n_steps, n_futere, data)

    def initModel(self):
        mlp_model = tf.keras.models.Sequential()
        mlp_model.add(tf.keras.layers.Dense(480, activation=activations.relu, input_dim=self.n_steps))
        mlp_model.add(tf.keras.layers.Dense(1))

        mlp_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005997502769579017), loss='mae')
        return mlp_model

    def fitModel(self, EPOCHS=200):
        x_train, y_train = self.get_train_dataset()
        mlp_model = self.initModel()
        x_valid, y_valid = self.get_validate_dataset()
        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
        mlp_model.fit(
            x=x_train,
            y=y_train,
            epochs=EPOCHS,
            callbacks=[es_callback],
            validation_data=(x_valid, y_valid)
        )

        return mlp_model

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
                                  loss='mae')  # оптимизируем
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

        cnn_model = self.__initModel()

        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0,
                                              patience=5)
        cnn_model.fit(x=x_train,
                      y=y_train,
                      callbacks=[es_callback],
                      epochs=EPOCHS,
                      validation_data=(x_valid, y_valid)
                      )
        return cnn_model

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


def mlp(data):
    mlp = MLP(15, 1, data)
    model = mlp.fitModel()
    x, y = mlp.get_test_data()
    print(model.evaluate(x, y))


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


if __name__ == "__main__":
    ds = pd.read_excel('SE.xls', skiprows=6)
    data = ds['T'].ffill()
    mlp(data)
    lstm(data)
    rnn(data)

from sklearn.preprocessing import LabelEncoder

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import activations
from keras import callbacks
from keras.layers import Dense
from tensorflow import keras


class NeuralNetwork:
    def __init__(self, n_steps, data):
        self.n_steps = n_steps
        self.data = data
        self.train_prosents = 0.75
        self.validate_prosents = 0.16
        self.test_prosents = 0.09
        self.batch_size = 16

    def split_sequence(self, data):
        x, y = list(), list()
        for i in range(len(data)):
            end_ix = i + self.n_steps
            if end_ix > len(data) - 1:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix:end_ix + 1, 0]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)

    def get_train_dataset(self):
        train_size = math.floor(self.data.shape[0] * self.train_prosents)
        return self.split_sequence(self.data[:train_size])

    def get_validate_dataset(self):
        size_from = math.floor(self.data.shape[0] * self.train_prosents)
        validate_size = math.floor(self.data.shape[0] * self.validate_prosents)
        size_to = size_from + validate_size
        return self.split_sequence(self.data[size_from:size_to])

    def get_test_data(self):
        size_from = math.floor(self.data.shape[0] * (self.train_prosents + self.validate_prosents))
        x, y = self.split_sequence(self.data[size_from:, :])
        return x, y


class MLP(NeuralNetwork):
    def __init__(self, n_steps, data):
        NeuralNetwork.__init__(self, n_steps, data)

    def initModel(self, x_size, y_size):
        print(x_size)
        print(y_size)
        initializer = tf.initializers.GlorotNormal()
        regularizer = tf.keras.regularizers.l1(1e-5)
        layers = [
            tf.keras.layers.Dense(80, activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_shape=(self.n_steps, x_size[2]), kernel_initializer=initializer, kernel_regularizer=regularizer),
            #tf.keras.layers.AlphaDropout(0.0012),
            tf.keras.layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer, kernel_regularizer=regularizer),
            #tf.keras.layers.AlphaDropout(0.0012),
            tf.keras.layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer, kernel_regularizer=regularizer),
            #tf.keras.layers.AlphaDropout(0.0012),
            tf.keras.layers.Dense(1)
        ]
        model = tf.keras.Sequential(layers)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mae')

        return model

    def fitModel(self, EPOCHS=200):
        x_train, y_train = self.get_train_dataset()
        mlp_model = self.initModel(x_train.shape, y_train.shape)
        x_valid, y_valid = self.get_validate_dataset()
        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
        mlp_model.fit(
            x=x_train,
            y=y_train,
            epochs=EPOCHS,
            callbacks=[es_callback],
            validation_data=(x_valid, y_valid),
            batch_size=self.batch_size  # Добавлен параметр batch_size
        )

        return mlp_model


class LSTM(NeuralNetwork):

    def __init__(self, n_steps, data):
        NeuralNetwork.__init__(self, n_steps, data)

    def __initModel(self, x_size):
        hyperparameters = {'num_layers': 5, 'units_0': 32, 'activation_0': 'tanh', 'lr': 0.0025200304939859757,
                           'units_1': 16, 'activation_1': 'tanh', 'units_2': 48, 'activation_2': 'relu',
                           'units_3': 64, 'activation_3': 'tanh', 'units_4': 64, 'activation_4': 'relu'}

        simple_lstm_model = keras.models.Sequential()
        simple_lstm_model.add(tf.keras.layers.LSTM(units=32, activation=activations.tanh, return_sequences=True,
                                                   input_shape=(self.n_steps, x_size)))
        simple_lstm_model.add(tf.keras.layers.LSTM(units=16, activation=activations.tanh, return_sequences=True,
                                                   input_shape=(self.n_steps, x_size)))
        simple_lstm_model.add(tf.keras.layers.LSTM(units=48, activation=activations.tanh, return_sequences=True,
                                                   input_shape=(self.n_steps, x_size)))
        simple_lstm_model.add(tf.keras.layers.LSTM(units=64, activation=activations.tanh, return_sequences=True,
                                                   input_shape=(self.n_steps, x_size)))
        simple_lstm_model.add(tf.keras.layers.LSTM(units=64, activation=activations.tanh, return_sequences=False,
                                                   input_shape=(self.n_steps, x_size)))
        simple_lstm_model.add(tf.keras.layers.Dense(1))
        simple_lstm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0025200304939859757),
                                  loss='mae')  # оптимизируем
        # среднюю абсолютную ошибку с помощью adam
        return simple_lstm_model

    def fitModel(self, EPOCHS=200):
        x_train, y_train = self.get_train_dataset()
        lstm_model = self.__initModel(x_train.shape[2])
        x_valid, y_valid = self.get_validate_dataset()
        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0,
                                              patience=5)
        lstm_model.fit(x=x_train, y=y_train,
                       epochs=EPOCHS,
                       callbacks=[es_callback],
                       validation_data=(x_valid, y_valid)
                       )
        return lstm_model


class RNN(NeuralNetwork):
    def __init__(self, n_steps, data):
        NeuralNetwork.__init__(self, n_steps, data)

    def __initModel(self, x_size):

        hyperparameters = {'num_layers': 5, 'units_0': 80, 'activation_0': 'tanh', 'lr': 0.000555783015174179,
                           'units_1': 16, 'activation_1': 'tanh', 'units_2': 128, 'activation_2': 'tanh',
                           'units_3': 32, 'activation_3': 'tanh', 'units_4': 112, 'activation_4': 'relu'}

        rnn_model = keras.models.Sequential()

        for i in range(hyperparameters['num_layers']):
            units = hyperparameters[f'units_{i}']
            activation = hyperparameters[f'activation_{i}']

            # Добавление слоя SimpleRNN
            rnn_model.add(tf.keras.layers.SimpleRNN(units=units, activation=activation,
                                                    return_sequences=True if i < hyperparameters[
                                                        'num_layers'] - 1 else False,
                                                    input_shape=(self.n_steps, x_size[2])))
            rnn_model.add(keras.layers.Dropout(0.2))

        # adding the output layer
        rnn_model.add(Dense(units=1))

        # compiling RNN
        rnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000555783015174179), loss='mae')
        return rnn_model

    def fitModel(self, EPOCHS=200):
        x_train, y_train = self.get_train_dataset()
        x_valid, y_valid = self.get_validate_dataset()

        cnn_model = self.__initModel(x_train.shape)

        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0,
                                              patience=5)
        cnn_model.fit(x=x_train,
                      y=y_train,
                      callbacks=[es_callback],
                      epochs=EPOCHS,
                      validation_data=(x_valid, y_valid)
                      )
        return cnn_model


def mlp(data):
    mlp = MLP(15, data.values)
    model = mlp.fitModel()
    x, y = mlp.get_test_data()

    # Оценка модели
    loss = model.evaluate(x, y)
    print("Test Loss:", loss)

    predictions = model.predict(x)

    plt.plot(y[:100], label='Actual')
    plt.plot(predictions[:100], label='Predicted')
    plt.legend()
    plt.show()

def lstm(data):
    lstm = LSTM(15, data.values)
    model = lstm.fitModel()
    x, y = lstm.get_test_data()

    loss = model.evaluate(x, y)
    print("Test Loss:", loss)

    predictions = model.predict(x)

    plt.plot(y[:10], label='Actual')
    plt.plot(predictions[:10], label='Predicted')
    plt.legend()
    plt.show()

def rnn(data):
    rnn = RNN(15, data.values)
    model = rnn.fitModel()
    x, y = rnn.get_test_data()

    loss = model.evaluate(x, y)
    print("Test Loss:", loss)

    predictions = model.predict(x)

    plt.plot(y[:10], label='Actual')
    plt.plot(predictions[:10], label='Predicted')
    plt.legend()
    plt.show()

if __name__ == "__main__":  # TODO рузультаты показали, что даже при включении признаков которые сильно коррелируют с температурой точность модели в лучем случае осталось той же
    # TODO также попробовал вообще без признака T результаты получились немного хуже(1.34)
    ds = pd.read_excel('SE.xls', skiprows=6)
    ds['Date'] = pd.to_datetime(ds['Date'], format='%d.%m.%Y %H:%M', dayfirst=True)
    ds['Day_of_year'] = ds['Date'].dt.dayofyear.astype(str).astype(int)
    ds['Hour'] = ds['Date'].dt.hour.astype(str).astype(int)
    ds['T'] = ds['T'].ffill()
    ds = ds.loc[:, ['T', 'Day_of_year', 'Hour']]
    le = LabelEncoder()
    #ds['DD'] = le.fit_transform(ds['DD'])
    #ds['W1'] = le.fit_transform(ds['W1'])
    #ds['Td'] = ds['Td'].ffill()
    mlp(ds)

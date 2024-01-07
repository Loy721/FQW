import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras import callbacks
from keras import layers
from keras import activations
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from keras.layers import Dropout
from keras.optimizers import SGD

class NeuralNetwork:
    def __init__(self, n_steps, n_futere, data):
        self.n_steps = n_steps
        self.n_futere = n_futere
        self.data = data

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

    def get_train_data_validate_data(self):
        TRAIN_SPLIT = self.data.size
        train_size = (TRAIN_SPLIT // 6) * 5
        x_train, y_train = self.split_sequence(self.data[:train_size])
        BATCH_SIZE = 8  # размер пакета
        tsize = (train_size // 6) * 5
        x_hlp = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        y_hlp = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
        train_data = tf.data.Dataset.from_tensor_slices((x_hlp[:tsize], y_hlp[:tsize]))  # упаковываем данные в объекты
        train_data = train_data.cache().shuffle(TRAIN_SPLIT).batch(
            BATCH_SIZE).repeat()  # кэширование данных,премешивание,пакетирование
        validate_val = tf.data.Dataset.from_tensor_slices((np.array(x_hlp[tsize:]), np.array(y_hlp[tsize:])))
        validate_val = validate_val.batch(BATCH_SIZE).repeat()
        return train_data, validate_val

    def get_test_data(self):
        train_size = (self.data.size // 6) * 5
        return self.split_sequence(self.data[train_size:])


class MLP(NeuralNetwork):
    def __init__(self, n_steps, n_futere, data, epohs, STEPS_INTERVAL):
        NeuralNetwork.__init__(self, n_steps, n_futere, data)
        self.model = self.fitModel(epohs, STEPS_INTERVAL)

    def initModel(self):
        mlp_model = tf.keras.models.Sequential()
        mlp_model.add(tf.keras.layers.Dense(208, activation=activations.relu, input_dim=self.n_steps))
        # mlp_model.add(tf.keras.layers.Dense(32, activation=activations.tanh))
        # mlp_model.add(tf.keras.layers.Dense(64, activation=activations.tanh))
        mlp_model.add(tf.keras.layers.Dense(1))

        mlp_model.compile( optimizer=keras.optimizers.Adam(learning_rate=0.007037202528080141), loss='mae')
        return mlp_model

    def fitModel(self, EPOCHS, STEPS_INTERVAL):
        train_data, validate_val = self.get_train_data_validate_data()

        mlp_model = self.initModel()

        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0,
                                              patience=5)
        mlp_model.fit(train_data, epochs=EPOCHS,
                      steps_per_epoch=STEPS_INTERVAL,
                      callbacks=[es_callback],
                      validation_data=validate_val,
                      validation_steps=50)
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

    def get_mae(self):
        mae_mlp = 0
        x_val, y_val = self.get_test_data()
        for i in range(0, x_val.shape[0] // 100):
            x, y = x_val[i * 100], y_val[i * 100]
            mae_mlp = mae_mlp + abs(self.model.predict(x.reshape(1, self.n_steps), verbose=0)[0, 0] - y[0])
        mae_mlp = mae_mlp / (x_val.shape[0] // 100)
        print("RES: ", mae_mlp)


class LSTM(NeuralNetwork):

    def __init__(self, n_steps, n_futere, data, epohs, STEPS_INTERVAL):
        NeuralNetwork.__init__(self, n_steps, n_futere, data)
        self.model = self.fitModel(epohs, STEPS_INTERVAL)

    def __initModel(self):
        simple_lstm_model = tf.keras.models.Sequential()
        simple_lstm_model.add(tf.keras.layers.LSTM(32, input_shape=(self.n_steps, 1)))
        simple_lstm_model.add(tf.keras.layers.Dense(1))
        simple_lstm_model.compile(optimizer='adam', loss='mae')  # оптимизируем
        # среднюю абсолютную ошибку с помощью adam
        return simple_lstm_model

    def fitModel(self, EPOCHS, STEPS_INTERVAL):
        train_data, validate_val = self.get_train_data_validate_data()

        cnn_model = self.__initModel()

        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0,
                                              patience=5)
        cnn_model.fit(train_data, epochs=EPOCHS,
                      steps_per_epoch=STEPS_INTERVAL,
                      callbacks=[es_callback],
                      validation_data=validate_val,
                      validation_steps=50)
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

    def get_mae(self):
        mae_rnn = 0
        x_val, y_val = self.get_test_data()
        for i in range(0, x_val.shape[0] // 100):
            x, y = x_val[i], y_val[i]
            mae_rnn = mae_rnn + abs(self.model.predict(x.reshape(1, self.n_steps, 1), verbose=0) - y[0])
        mae_rnn = mae_rnn / (x_val.shape[0] // 100)
        return mae_rnn


class RNN(NeuralNetwork):
    def __init__(self, n_steps, n_futere, data, epohs, STEPS_INTERVAL):
        NeuralNetwork.__init__(self, n_steps, n_futere, data)
        self.model = self.fitModel(epohs, STEPS_INTERVAL)

    def __initModel(self):
        train_data, validate_val = self.get_train_data_validate_data()

        regressor = Sequential()

        # adding RNN layers and dropout regularization
        regressor.add(SimpleRNN(units=50,
                                activation="tanh",
                                return_sequences=True,
                                input_shape=(self.n_steps, 1)))
        regressor.add(Dropout(0.2))

        regressor.add(SimpleRNN(units=50,
                                activation="tanh",
                                return_sequences=True))

        regressor.add(SimpleRNN(units=50,
                                activation="tanh",
                                return_sequences=True))

        regressor.add(SimpleRNN(units=50))

        # adding the output layer
        regressor.add(Dense(units=1))

        # compiling RNN
        regressor.compile(optimizer='adam',
                          loss="mae")
        return regressor

    def fitModel(self, EPOCHS, STEPS_INTERVAL):
        train_data, validate_val = self.get_train_data_validate_data()

        cnn_model = self.__initModel()

        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0,
                                              patience=5)
        cnn_model.fit(train_data, epochs=EPOCHS,
                      steps_per_epoch=STEPS_INTERVAL,
                      callbacks=[es_callback],
                      validation_data=validate_val,
                      validation_steps=50)
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

    def get_mae(self):
        mae_cnn = 0
        x_val, y_val = self.get_test_data()
        for i in range(0, x_val.shape[0] // 100):
            x, y = x_val[i], y_val[i]
            mae_cnn = mae_cnn + abs(self.model.predict(x.reshape(1, self.n_steps, 1), verbose=0) - y[0])
        mae_cnn = mae_cnn / (x_val.shape[0] // 100)
        return mae_cnn


if __name__ == "__main__":
    ds = pd.read_excel('SE.xls', skiprows=6)
    ds = ds.dropna(subset=['T'])
    print(ds.shape)
    # z = np.abs(stats.zscore(ds['Temperature (C)']))
    # emissions = np.where(z > 3)[0]
    # ds = ds.drop(emissions, axis=0) ри нормолизации все хуже)

    mlp = MLP(1000, 1, ds['T'].values, 100, 220)
    print(mlp.get_predict())
    lstm = LSTM(1000, 1, ds['T'].values, 30, 500)
    lstm.get_predict()

    # rnn = RNN(100, 1, ds['T'].values, 30, 220)
    # rnn.get_predict()

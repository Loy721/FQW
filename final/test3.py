import tensorflow as tf
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks, Sequential, Input
from keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import layers

import math

class DataUtil:
    def __init__(self, data, train_prosent=0.8, n_input=15, n_future=1):
        self.data = data
        self.train_prosent = train_prosent
        self.test_prosent = 1 - train_prosent
        self.n_input = n_input
        self.n_future = n_future
        self.min_normalize = None
        self.max_normalize = None

    def split_to_time_window(self, data):
        x, y = list(), list()
        for i in range(len(data)):
            end_ix = i + self.n_input
            if end_ix > len(data) - self.n_future:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix:end_ix + self.n_future]
            x.append(seq_x)
            y.append(seq_y)
        x_arr, y_arr = np.array(x), np.array(y)
        indices = np.random.permutation(len(x_arr))
        x_arr = x_arr[indices]
        y_arr = y_arr[indices]
        return np.array(x_arr), np.array(y_arr)

    def get_train_data(self):
        to_idx = math.floor(self.data.shape[0] * self.train_prosent)
        return self.data[:to_idx]

    def get_test_data(self):
        from_idx = math.floor(self.data.shape[0] * (self.test_prosent))
        return self.data[from_idx:]

    def get_splitted_train(self):
        return self.split_to_time_window(self.get_train_data())

    def get_splitted_test(self):
        return self.split_to_time_window(self.get_test_data())

    def get_splitted_train_normalized(self):
        return self.split_to_time_window(self.__normalize(self.get_train_data()))

    def get_splitted_test_normalized(self):
        return self.split_to_time_window(self.__normalize(self.get_test_data()))
    def __normalize(self, data):
        self.min_val = np.min(self.get_train_data())
        self.max_val = np.max(self.get_train_data())
        return ((data - self.min_val) / (self.max_val - self.min_val)) * 2 - 1

    def denormalize(self, normalized_data):
        return ((normalized_data + 1) / 2) * (self.max_val - self.min_val) + self.min_val


def build_model_conv_lstm():
    initializer = tf.initializers.GlorotNormal()
    # define model
    model = Sequential()
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 4), activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                input_shape=(1, 1, window, 1), return_sequences=True))
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(8))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01))
    # fit network
    return model
def __rmse_for_each_step(pred, test, dataUtil):
    n_future = pred.shape[1]
    ls_rmse = []
    ls_maes = []
    for i in range(n_future):
        rmse = np.sqrt(mean_squared_error(dataUtil.denormalize(pred[:, i]), dataUtil.denormalize(test[:, i])))
        mae = np.mean(np.abs(dataUtil.denormalize(pred[:, i]) - dataUtil.denormalize(test[:, i])))
        ls_rmse.append(rmse)
        ls_maes.append(mae)
    return ls_rmse, ls_maes

def test_rmse_best_model(X_test, y_test, dataUtil,model):
    y_pred = model.predict(X_test)
    if len(y_pred.shape) == 3:
        y_pred = y_pred.squeeze()
    print("MAE ", np.mean(np.abs(dataUtil.denormalize(y_pred) - dataUtil.denormalize(y_test))))
    rmses, maes = __rmse_for_each_step(y_pred, y_test, dataUtil)
    print("MAE for each day: ", maes)
    print("Mean MAE : ", np.mean(maes))
    print("RMSE for each day: ", rmses)
    print("Mean RMSE : ", np.mean(rmses))
    return rmses, maes

if __name__ == "__main__":
    time_windows = [80]
    windows_maes = []
    for window in time_windows:
        ds = pd.read_excel('SE_final.xls', skiprows=6)
        data = ds['T'].ffill()
        n_future = 8
        n_step = window
        dataUtil = DataUtil(data, n_input=n_step, n_future=n_future)
        np.random.seed(0)
        model = build_model_conv_lstm()
        x_train, y_train = dataUtil.get_splitted_train_normalized()
        x_train = x_train.reshape((x_train.shape[0], 1, 1, window, 1))
        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=6)
        model.fit(x_train, y_train, epochs=300, validation_split=0.2, callbacks=[es_callback], batch_size=32)
        X_test, y_test = dataUtil.get_splitted_test_normalized()
        X_test = X_test.reshape((X_test.shape[0], 1, 1, window, 1))
        rmses, maes = test_rmse_best_model(X_test, y_test, dataUtil, model)
        windows_maes.append(maes)

    for i in range(len(windows_maes)):
        plt.plot(range(1, 9), windows_maes[i], label=str(time_windows[i]))

    plt.legend(title="Количество временных\n     шагов в окне")
    plt.xlabel('Временной шаг')
    plt.ylabel('MAE')

    plt.show()
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks, Sequential, Input
from keras.losses import mean_squared_error, Huber
from keras_tuner import GridSearch
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K

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
def build_model_conv_lstm(hp):
    initializer = tf.initializers.GlorotNormal()
    # define model
    model = Sequential()
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 4), activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                input_shape=(1, 1, 40, 1), return_sequences=True))
    model.add(layers.ConvLSTM2D(filters=128, kernel_size=(1, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(8))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3])
    model.compile(loss=Huber(delta=1.0), optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate))
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


def test_rmse_best_model(X_test, y_test, dataUtil, model):
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

if __name__ == "__main__":  # rmse: mlp = 2.8 [1.6 ...],
    ds = pd.read_excel('SE_final.xls', skiprows=6)
    data = ds['T'].ffill()
    n_future = 8
    n_step = 40
    dataUtil = DataUtil(data, n_input=n_step, n_future=n_future)

    np.random.seed(0)

    tuner = GridSearch(
        build_model_conv_lstm,
        objective='val_loss',
        directory='lstm',  # Директория для сохранения результатов
        project_name='lstm_2',  # Имя проекта
        overwrite=True,
        max_consecutive_failed_trials=1,
        max_trials=1
    )
    print("Число комбинаций: ", tuner.search_space_summary())

    x_train, y_train = dataUtil.get_splitted_train_normalized()
    # CONVLSTM
    x_train = x_train.reshape((x_train.shape[0], 1, 1, 40, 1))
    # CONVLSTM
    es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    tuner.search(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=[es_callback], batch_size=128)
    best_model = tuner.get_best_models()[0]
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:", best_hyperparameters)
    X_test, y_test = dataUtil.get_splitted_test_normalized()
    # CONVLSTM
    X_test = X_test.reshape((X_test.shape[0], 1, 1, 40, 1))
    # CONVLSTM
    test_rmse_best_model(X_test, y_test, dataUtil, best_model)

#ADAM
# MAE  2.0813144795000635
# MAE for each day:  [1.093059380095657, 1.6205075324041043, 1.9278739052923826, 2.136617768262568, 2.296366535387488, 2.4272194958316056, 2.5183029625259303, 2.6305682562007697]
# Mean MAE :  2.0813144795000635
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks, Sequential, Input
from keras.losses import mean_squared_error
from keras_tuner import GridSearch
from tensorflow import keras
from tensorflow.keras import layers


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


def build_model_mlp(hp):
    model = keras.Sequential()
    initializer = tf.initializers.GlorotNormal()
    regularizer = tf.keras.regularizers.l2(1e-5)
    # Подбор количества скрытых слоев и нейронов в каждом слое
    # for i in range(hp.Int('num_layers', min_value=1, max_value=5)):
    #    model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
    #                           activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer, kernel_regularizer=regularizer))
    model.add(layers.Dense(80, activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_dim=n_step,
                           kernel_initializer=initializer, kernel_regularizer=regularizer))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer,
                           kernel_regularizer=regularizer))
    model.add(layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer,
                           kernel_regularizer=regularizer))
    model.add(layers.Dense(8))

    # Подбор скорости обучения
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')

    return model


def build_model_lstm(hp):
    initializer = tf.initializers.GlorotNormal()
    regularizer = tf.keras.regularizers.l2(1e-4)

    model = keras.Sequential()
    model.add(layers.LSTM(units=40, input_shape=(n_step, 1),
                          activation=keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer,
                          kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(8))
    # Подбор скорости обучения
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

    return model


def build_model_conv_lstm(hp):
    initializer = tf.initializers.GlorotNormal()
    # define model
    model = Sequential()
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 4), activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                input_shape=(5, 1, 8, 1), return_sequences=True))
    model.add(layers.ConvLSTM2D(filters=128, kernel_size=(1, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                                return_sequences=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(8))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate))
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


def build_model_cnn(ch):
    model = keras.Sequential()
    model.add(Input(shape=(n_step, 1)))

    # Сверточные слои
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Выравнивание перед полносвязными слоями
    model.add(layers.Flatten())

    model.add(layers.Dense(n_future))

    model.compile(optimizer='adam', loss='mse')

    return model


if __name__ == "__main__":  # rmse: mlp = 2.8 [1.6 ...],
    ds = pd.read_excel('SE_final.xls', skiprows=6)
    data = ds['T'].ffill()
    n_future = 8
    n_step = 300
    dataUtil = DataUtil(data, n_input=n_step, n_future=n_future)

    np.random.seed(0)

    tuner = GridSearch(
        build_model_lstm,
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
    #x_train = x_train.reshape((x_train.shape[0], 5, 1, 8, 1))
    # CONVLSTM
    es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
    tuner.search(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=[es_callback], batch_size=128)
    best_model = tuner.get_best_models()[0]
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters:", best_hyperparameters)
    X_test, y_test = dataUtil.get_splitted_test_normalized()
    # CONVLSTM
    #X_test = X_test.reshape((X_test.shape[0], 5, 1, 8, 1))
    # CONVLSTM
    test_rmse_best_model(X_test, y_test, dataUtil, best_model)

# MLP MAE  2.093004950876411
# MAE for each day:  [1.11688152458197, 1.6145568766455813, 1.9469714028639518, 2.148913865921949, 2.30983892378451, 2.4263817054693795, 2.538149379448406, 2.6423459282955397]
# Mean MAE :  2.093004950876411
# RMSE for each day:  [1.5385443590115244, 2.1628668200102856, 2.5759281962607075, 2.831956565871507, 3.043831172977354, 3.2016199363917273, 3.3527951729147505, 3.5012395806070598]
# Mean RMSE :  2.7760977255056147

# LSTM MAE  2.067252773336993
# MAE for each day:  [1.1587499177959601, 1.6069303912797128, 1.883389826847333, 2.0914710072333147, 2.257823155596001, 2.3885995946778222, 2.5143640411593893, 2.6366942521064103]
# Mean MAE :  2.067252773336993
# RMSE for each day:  [1.5730482999653055, 2.1435670814261965, 2.5026983598598003, 2.772490216892738, 2.9859746526786948, 3.1589780444620468, 3.3255109511983822, 3.4960440144084353]
# Mean RMSE :  2.74478895261145

# CNN MAE  2.119964324046791
# MAE for each day:  [1.1989569776830538, 1.6473671320160748, 1.9754815255762441, 2.1836474056735558, 2.324816652759732, 2.4584648043317996, 2.5427762546787678, 2.6282038396550975]
# Mean MAE :  2.1199643240467907
# RMSE for each day:  [1.6178296490534905, 2.1906211299944323, 2.597165246926802, 2.858321437537694, 3.041658012713311, 3.2131933482097104, 3.3422017165174434, 3.479771664220338]
# Mean RMSE :  2.7925952756466526

# MAE  1.7809725014510194
# MAE for each day:  [1.0583035627755908, 1.439971914709549, 1.6707140825582318, 1.806017896868655, 1.8972062653554995, 1.987486072898133, 2.1051729706656004, 2.282907245776896]
# Mean MAE :  1.7809725014510196
# RMSE for each day:  [1.4464579900536563, 1.9299113348583472, 2.238135673447324, 2.4290697898557836, 2.5642662608131035, 2.686273579114225, 2.837111378717418, 3.043108896823663]
# Mean RMSE :  2.39679186296044

#ConvLSTM MAE  1.8969381967802275
# MAE for each day:  [1.3765595094185155, 1.5714909472100802, 1.7662581518850324, 1.885344238651266, 1.9833374159542012, 2.084376911637987, 2.1768695757943797, 2.3312688236903543]
# Mean MAE :  1.896938196780227


# initializer = tf.initializers.GlorotNormal()
# hp_reg = hp.Choice('reg_rate', values=[1e-5, 1e-4, 1e-3])
# regulaizer = tf.keras.regularizers.l2(hp_reg)
#
# model = keras.Sequential()
# activation_choice = hp.Choice('activation_choice', ['sigmoid', 'tanh', 'leaky_relu'])
# if activation_choice == 'sigmoid':
#     activation = keras.activations.sigmoid
# elif activation_choice == 'tanh':
#     activation = keras.activations.tanh
# else:
#     activation = keras.layers.LeakyReLU(alpha=0.2)
# model.add(layers.LSTM(units=hp.Int('units_0', min_value=16, max_value=96, step=32), input_shape=(15, 1),
#                       activation=activation_choice, kernel_initializer=initializer, kernel_regularizer=regulaizer))
# # Подбор количества скрытых слоев и нейронов в каждом слое
# for i in range(1, hp.Int('num_layers', min_value=1, max_value=3)):
#     model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=16, max_value=96, step=32),
#                            activation=activation, kernel_initializer=initializer, kernel_regularizer=regulaizer))
# model.add(tf.keras.layers.Dense(8))
# # Подбор скорости обучения
# hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
# opts_choice = hp.Choice("Optimizer", ['adam', 'rmsprop'])
# if opts_choice == 'adam':
#     optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
# else:
#     optimizer = keras.optimizers.RMSprop(learning_rate=hp_learning_rate)
# model.compile(optimizer=optimizer, loss='mse')
#
# return model

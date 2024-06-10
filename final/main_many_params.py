import math
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks, Input, Model, Sequential
from keras.losses import mean_squared_error
from keras_tuner import GridSearch
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

from final.main import test_rmse_best_model


class DataUtil:
    def __init__(self, data, train_prosent=0.8, n_input=15, n_future=1):
        self.data = data
        self.train_prosent = train_prosent
        self.test_prosent = 1 - train_prosent
        self.n_input = n_input
        self.n_future = n_future
        self.min_val = None
        self.max_val = None

    def split_to_time_window(self, data):
        x, y = list(), list()
        for i in range(len(data)):
            end_ix = i + self.n_input
            if end_ix > len(data) - self.n_future:
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix:end_ix + self.n_future, 0]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)

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
        if self.min_val is None:
            self.min_val = np.min(self.get_train_data(), axis=0)
            self.max_val = np.max(self.get_train_data(), axis=0)
        return ((data - self.min_val) / (self.max_val - self.min_val)) * 2 - 1

    def denormalize(self, normalized_data):
        return ((normalized_data + 1) / 2) * (self.max_val[0] - self.min_val[0]) + self.min_val[0]

def build_model_mlp():
    print("MLP")
    model = keras.Sequential()
    model.add(Input(shape=(n_step, x_train.shape[2])))
    model.add(layers.Flatten())
    initializer = tf.initializers.GlorotNormal()
    regularizer = tf.keras.regularizers.l2(1e-5)
    # Подбор количества скрытых слоев и нейронов в каждом слое
    # for i in range(hp.Int('num_layers', min_value=1, max_value=5)):
    #    model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
    #                           activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer, kernel_regularizer=regularizer))
    model.add(layers.Dense(80, activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                           kernel_initializer=initializer, kernel_regularizer=regularizer))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer,
                           kernel_regularizer=regularizer))
    model.add(layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer,
                           kernel_regularizer=regularizer))
    model.add(layers.Dense(n_future))

    # Подбор скорости обучения
    #hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse')

    return model


def build_model_lstm():
    model = keras.Sequential()
    initializer = tf.initializers.GlorotNormal()
    regularizer = tf.keras.regularizers.l2(1e-5)
    model.add(layers.LSTM(units=200, input_shape=(n_step, x_train.shape[2]),
                          activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer, kernel_regularizer=regularizer))
    model.add(layers.Dropout(0.2))
    # Подбор количества скрытых слоев и нейронов в каждом слое
    model.add(layers.Dense(40, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer,
                           kernel_regularizer=regularizer))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(40, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer,
                           kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(8))
    # Подбор скорости обучения

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss='mse')

    return model

def build_model_cnn():
    model = keras.Sequential()
    model.add(Input(shape=(n_step, x_train.shape[2])))

    # Сверточные слои
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Выравнивание перед полносвязными слоями
    model.add(layers.Flatten())

    model.add(layers.Dense(n_future))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    return model

def build_model_conv_lstm():

    model = Sequential()
    model.add(layers.ConvLSTM2D(filters=128, kernel_size=(8, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_shape=(5, data.shape[1], 8, 1), return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 1, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.ConvLSTM2D(filters=128, kernel_size=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.2), return_sequences=False))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(8))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
    # fit network
    return model


def calculate_average_from_range(value):
    # Извлечение чисел из строки
    if isinstance(value, str):
        numbers = re.findall(r'\d+', value)
        # Преобразование чисел к целочисленному типу
        numbers = [int(num) for num in numbers]
        # Вычисление среднего значения
        average = sum(numbers) / len(numbers)
        return average
    return value


def prepare_data():
    df = pd.read_excel('SE.xls', skiprows=6)
    df["WW"] = df["WW"].fillna("someValue")
    df = df.fillna(method='ffill')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', dayfirst=True)
    df['Day_of_year'] = df['Date'].dt.dayofyear
    df['Hour'] = df['Date'].dt.hour
    df = df.loc[:, ['T', 'Day_of_year', 'Hour', 'P', 'U', 'DD', 'Ff', 'WW', 'VV', 'N']]
    # df['W1'] = df['W1'].fillna("someValue")

    df.loc[df["N"] == "Облаков нет.", "N"] = 0
    df.loc[df["N"] == "Небо не видно из-за тумана и/или других метеорологических явлений.", "N"] = 0
    df["N"] = df["N"].apply(calculate_average_from_range)
    df.loc[df["VV"] == "менее 0.05", "VV"] = 0
    df.loc[df["VV"] == "менее 0.1", "VV"] = 0.05
    df["VV"] = df["VV"].astype(float)

    le = LabelEncoder()
    df['DD'] = le.fit_transform(df['DD'])
    df['WW'] = le.fit_transform(df['WW'])
    return df


#Мониторинг и отладка: Внимательное отслеживание процесса обучения, визуализация метрик производительности модели и анализ ошибок помогают выявить проблемы и улучшить точность прогноза
#Ансамблирование
#Оптимизатор
#Использование предобученных моделей: Использование предварительно обученных моделей, особенно в области компьютерного зрения и обработки естественного языка (NLP), может значительно улучшить точность прогноза.
if __name__ == "__main__":
    seed_value = 44
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    ds = pd.read_excel('SE_final.xls', skiprows=6)
    ds_beg = ds.copy()
    threshold = len(ds) * 0.9
    ds.dropna(axis=1, thresh=threshold, inplace=True)

    ds['T'] = ds['T'].ffill()
    ds['P'] = ds['P'].ffill()
    ds['Date'] = pd.to_datetime(ds['Date'], format='%d.%m.%Y %H:%M', dayfirst=True)
    ds['Day_of_year'] = ds['Date'].dt.dayofyear
    ds['Hour'] = ds['Date'].dt.hour


    data = prepare_data()  # TODO: первый параметр всегда температура!
    data = np.array(data)
    n_future = 8
    n_step = 40
    dataUtil = DataUtil(data, n_input=n_step, n_future=n_future)

    #
    # tuner = GridSearch(
    #     build_model_lstm,
    #     objective='val_loss',
    #     directory='mlp',  # Директория для сохранения результатов
    #     project_name='mlp_1',  # Имя проекта
    #     overwrite=True,
    #     max_trials=1,
    #     max_consecutive_failed_trials=1
    # )
    # print("Число комбинаций: ", tuner.search_space_summary())

    x_train, y_train = dataUtil.get_splitted_train_normalized()

    x_train = x_train.reshape((x_train.shape[0], 5, data.shape[1], 8, 1))
    print("dsfsdfsdf", x_train.shape)
    print("dsfsdfsdf", y_train.shape)
    es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
    # tuner.search(x_train, y_train, epochs=1000, validation_split=0.2, callbacks=[es_callback], batch_size=32)


    model = build_model_conv_lstm()


    model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es_callback], batch_size=32)
    best_model = model #tuner.get_best_models()[0]

    X_test, y_test = dataUtil.get_splitted_test_normalized()
    X_test = X_test.reshape((X_test.shape[0], 5, data.shape[1], 8, 1))
    res = best_model.evaluate(X_test, y_test)
    print("eval", res)
    test_rmse_best_model(X_test, y_test, dataUtil, best_model)

#MLP MAE  1.9457410919655977
# MAE for each day:  [1.3124431408248518, 1.5489869407857435, 1.749412760180124, 1.9058007834500619, 2.0615869486212626, 2.1996709491890467, 2.3285528996075304, 2.4594743130661634]
# Mean MAE :  1.9457410919655982
# RMSE for each day:  [1.705727109668997, 2.027560527283914, 2.292803728951273, 2.4999414951166803, 2.710178672473312, 2.8997713702683185, 3.0773718883890138, 3.250796626833971]
# Mean RMSE :  2.558018927373185

#LSTM MAE  1.9392107612790925
# MAE for each day:  [1.3166363448865714, 1.5035534104884931, 1.7152796943682933, 1.9087814610923277, 2.0614252798551096, 2.190562096800451, 2.330615841165456, 2.486831961576037]
# Mean MAE :  1.9392107612790925
# RMSE for each day:  [1.7377156434101109, 1.9864924782405338, 2.2643933033966044, 2.5216455035047707, 2.7257119057679717, 2.8929372248235277, 3.072946021179525, 3.2595868597421]
# Mean RMSE :  2.5576786175081434

# CNN MAE  1.731788521194935
# MAE for each day:  [1.1788543387188777, 1.4647820391702695, 1.597148162273104, 1.7426590080713735, 1.846475991898733, 1.9277698634508351, 1.9896923335626984, 2.1069264324135912]
# Mean MAE :  1.7317885211949353
# RMSE for each day:  [1.5839786117903023, 1.9825508240690903, 2.205953104890688, 2.4303461119591008, 2.5963457234589993, 2.7312750932907144, 2.8462731227732956, 3.003849425868416]
# Mean RMSE :  2.422571502262576

#ConvLSTM  MAE  1.9291610104073176
# MAE for each day:  [1.3343990495793239, 1.5429132033849726, 1.789252146581011, 1.9145232940654033, 2.0510541222231886, 2.176906721249921, 2.2628541964558404, 2.3613853497188786]
# Mean MAE :  1.9291610104073174
# RMSE for each day:  [1.778411078362683, 2.0429276213865433, 2.3598398634508375, 2.5285676879477683, 2.7100842433379846, 2.87559385288262, 2.9928822198552028, 3.1215738180200248]
# Mean RMSE :  2.551235048155458
import math

import tensorflow as tf
from keras.losses import mean_squared_error
from keras_tuner import HyperParameters, GridSearch
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch, BayesianOptimization
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks, Sequential, Input
from keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import layers

from final.main import DataUtil


def build_model_conv_lstm():
    initializer = tf.initializers.GlorotNormal()
    # define model
    model = Sequential()
    model.add(layers.ConvLSTM2D(filters=32, kernel_size=(1, 4), activation=tf.keras.layers.LeakyReLU(alpha=0.2), input_shape=(1, 1, n_step, 1), return_sequences=True))
    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(1, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), return_sequences=True))
    model.add(layers.ConvLSTM2D(filters=128, kernel_size=(1, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.2), return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.2), kernel_initializer=initializer))
    model.add(layers.Dense(8))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=1e-3))
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

if __name__ == "__main__": # rmse: mlp = 2.8 [1.6 ...],
    time_windows = [8, 20, 40, 80, 240, 732, 1460, 2920]
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
        x_train = x_train.reshape((x_train.shape[0], 1, 1, n_step, 1))
        es_callback = callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=12)
        model.fit(x_train, y_train, epochs=1, validation_split=0.2, callbacks=[es_callback], batch_size=32)
        X_test, y_test = dataUtil.get_splitted_test_normalized()
        X_test = X_test.reshape((X_test.shape[0], 1, 1, n_step, 1))
        rmses, maes = test_rmse_best_model(X_test, y_test, dataUtil, model)
        windows_maes.append(maes)

    for i in range(len(windows_maes)):
        plt.plot(range(1, 9), windows_maes[i], label=str(time_windows[i]))
    plt.legend(title="Количество временных\n     шагов в окне")
    plt.xlabel('Временной шаг')
    plt.ylabel('MAE')
    plt.show()

    x = np.arange(len(windows_maes))
    width = 0.2
    plt.bar(x, [np.mean(t) for t in windows_maes], width, color='skyblue', edgecolor='black')
    plt.xticks(x, [str(t) for t in time_windows])
    plt.xlabel('Количество временных шагов в окне')
    plt.ylabel('MAE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

#     MAE  2.2076626952199283
# MAE for each day:  [1.2481674933381621, 1.8025651294106, 2.090973431948462, 2.285202572910539, 2.4285506714346883, 2.528963077189241, 2.6028117307543748, 2.6740674547733576]
# Mean MAE :  2.2076626952199283
#
# MAE  1.870851780310166
# MAE for each day:  [1.0879454439484058, 1.4767784652588314, 1.723247156936439, 1.8793581337362346, 2.013168160111354, 2.120024957061289, 2.2545633686155897, 2.4117285568131877]
# Mean MAE :  1.8708517803101663
#
# MAE  2.016754790296831
# MAE for each day:  [1.0913999108438575, 1.537722778178238, 1.8441943685943125, 2.0561059900889154, 2.220468496534221, 2.3460680374740193, 2.4545915220696486, 2.5834872185914373]
# Mean MAE :  2.0167547902968312
#
# MAE  2.1059485642985805
# MAE for each day:  [1.2564839727802772, 1.6155476272128466, 1.920473584927501, 2.153120670252358, 2.33736626245171, 2.4207723294889227, 2.5214780684483102, 2.622345998826716]
# Mean MAE :  2.10594856429858
#
# MAE  2.1032719177669663
# MAE for each day:  [1.36033446105434, 1.677554687426255, 1.9547112307499723, 2.1239311865182073, 2.2551932016836322, 2.3824006314979207, 2.4900509513437297, 2.5819989918616724]
# Mean MAE :  2.103271917766966
#
# MAE  2.1687486828957203
# MAE for each day:  [1.583354182080828, 1.9544617508443254, 2.0714701178666144, 2.2246182046397274, 2.333917173638381, 2.3367691992975854, 2.374149085510925, 2.471249749287383]
# Mean MAE :  2.168748682895721
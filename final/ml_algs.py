import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from main import DataUtil
def maes_for_each_step(pred, test):
    n_future = pred.shape[1]
    ls = []
    for i in range(n_future):
        mae = np.mean(np.abs(dataUtil.denormalize(pred[:, i]) - dataUtil.denormalize(test[:, i])))
        ls.append(mae)
    return ls

def get_the_best_model(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    maes_for_days = maes_for_each_step(y_test, y_pred)
    print("MAE:", np.mean(maes_for_days))
    print("MAE for days:", maes_for_days)
    return best_model, maes_for_days

def predict_on_n_step_SVR(best_model, n_step, n_future, X_test):
    forecasts = []
    for i in range(len(X_test)):
        last_sequence = X_test[i, -n_step:]  # Начинаем с последних n_step значений истории
        forecast = []
        for _ in range(n_future):
            y_hat = best_model.predict([last_sequence])[0]
            forecast.append(y_hat)
            last_sequence = np.append(last_sequence[1:], y_hat)  # Обновляем последовательность для прогноза следующего значения
        forecasts.append(forecast)

    return np.array(forecasts)

if __name__ == "__main__":
    rmses_for_steps_dict = {}
    ds = pd.read_excel('SE_final.xls', skiprows=6)
    data = ds['T'].ffill()
    n_future = 8
    n_step = 40
    dataUtil = DataUtil(data, n_input=n_step, n_future=n_future)

    np.random.seed(0)

    X_train, y_train = dataUtil.get_splitted_train_normalized()
    X_test, y_test = dataUtil.get_splitted_test_normalized()

    print("Start Ridge")
    param_grid_ridge = {
        'alpha': [0.01, 0.05, 0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'tol': [1e-3, 1e-4, 1e-5]
    }
    best_ridge_model, rmse_for_days_ridge = get_the_best_model(Ridge(), param_grid_ridge)
    rmses_for_steps_dict["Ridge"] = rmse_for_days_ridge
    print("End Ridge")

    # Отображение графиков
    plt.show()
    print("Start KNN")
    param_grid_KNN = {
        'n_neighbors': [3, 5, 7, 10, 12, 15, 18, 20, 25, 50],  # количество соседей
        'weights': ['uniform', 'distance'],  # веса для соседей (равномерные или взвешенные)
        'metric': ['euclidean', 'manhattan']  # метрика расстояния
    }
    best_KNN_model, rmse_for_days_KNN = get_the_best_model(KNeighborsRegressor(), param_grid_KNN)
    rmses_for_steps_dict["KNN"] = rmse_for_days_KNN
    print("End KNN")

    print("Start Decision Tree Regressor")
    param_grid_DTR = {
        'max_depth': [None, 5, 10, 20],  # максимальная глубина дерева
        'min_samples_split': [2, 5, 10],  # минимальное количество выборок для разделения узла
        'min_samples_leaf': [1, 2, 4],  # минимальное количество выборок в листовом узле
        'max_features': ['auto', 'sqrt']  # количество признаков для поиска лучшего разделения
    }
    best_DTR_model, rmse_for_days_DTR = get_the_best_model(DecisionTreeRegressor(), param_grid_DTR)
    rmses_for_steps_dict["DTR"] = rmse_for_days_DTR
    print("End Decision Tree Regressor")

    print("Start SVR")
    dataUtilSVR = DataUtil(data, n_input=n_step, n_future=1)
    X_train_SVR, y_train_SVR = dataUtilSVR.get_splitted_train_normalized()

    param_grid_SVR = {
        'kernel': ['linear', 'rbf', 'sigmoid'],
        'C': [0.1, 1],
        'epsilon': [0.01, 0.1]
    }
    grid_search = GridSearchCV(SVR(), param_grid_SVR, scoring='neg_mean_squared_error')

    grid_search.fit(X_train_SVR, y_train_SVR)

    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    best_model = grid_search.best_estimator_
    y_pred_SVR = predict_on_n_step_SVR(best_model, n_step, n_future, X_test)
    mae_for_days_SVR = maes_for_each_step(y_test, y_pred_SVR)
    print("MAE:", np.mean(mae_for_days_SVR))
    print("MAE for days:", mae_for_days_SVR)
    rmses_for_steps_dict["SVR"] = mae_for_days_SVR
    print("End SVR")

    print(rmses_for_steps_dict)

    for label, values in rmses_for_steps_dict.items():
        plt.plot(range(1, len(values) + 1), values, label=label)

    plt.legend()
    plt.xlabel('Временной шаг')
    plt.ylabel('MAE')

    plt.show()

# MAE: 2.1526604169594012
# MAE for days: [1.117443238863702, 1.686345672677133, 2.0393733548933115, 2.2600678620320305, 2.3914503761429002, 2.487635326030302, 2.5679033603272154, 2.6710641447086143]
# End Ridge

# Best hyperparameters: {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}
# MAE: 0.7274684172651729
# MAE for days: [0.5155532745921989, 0.6129942685429055, 0.6810137347170433, 0.7291782941593997, 0.7671117226264814, 0.8014325059900168, 0.8349233031174906, 0.8775402343758468]
# End KNN

# Best hyperparameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5}
# MAE: 2.491440372977437
# MAE for days: [1.879734109063366, 2.21549275864517, 2.401474357509436, 2.512809346162748, 2.6146993843329325, 2.676648522729482, 2.7435154547908387, 2.8871490505855197]
# End Decision Tree Regressor


#     MAE: 2.026706634804467
# MAE for days: [1.0176235430204228, 1.5323836070551204, 1.8660719555047107, 2.086614423356428, 2.2456482683584267, 2.3739770706682624, 2.4843541905432462, 2.6069800199291193]
# End SVR
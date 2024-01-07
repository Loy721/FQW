import matplotlib.pyplot as plt
import pandas as pd

# Функция для замены выбросов предельными значениями
def replace_outliers_with_threshold(dataframe, threshold):
    q1 = dataframe.quantile(0.25)
    q3 = dataframe.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    dataframe = dataframe.apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return dataframe

if __name__ == '__main__':
    ds = pd.read_excel('SE.xls', skiprows=6)
    data = ds['T'].ffill()

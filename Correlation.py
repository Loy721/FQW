import re

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
def main():
    df = pd.read_excel('SE.xls', skiprows=6)
    df["WW"] = df["WW"].fillna("someValue")
    df = df.fillna(method='ffill')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', dayfirst=True)
    df['Day_of_year'] = df['Date'].dt.dayofyear
    df['Hour'] = df['Date'].dt.hour
    df = df.loc[:, ['Day_of_year', 'Hour', 'T', 'P', 'U', 'DD', 'Ff', 'WW', 'Td', 'VV', 'N']]
    #df['W1'] = df['W1'].fillna("someValue")

    df.loc[df["N"] == "Облаков нет.", "N"] = 0
    df.loc[df["N"] == "Небо не видно из-за тумана и/или других метеорологических явлений.", "N"] = 0
    df["N"] = df["N"].apply(calculate_average_from_range)
    df.loc[df["VV"] == "менее 0.05", "VV"] = 0
    df.loc[df["VV"] == "менее 0.1", "VV"] = 0.05
    df["VV"] = df["VV"].astype(float)

    le = LabelEncoder()
    df['DD'] = le.fit_transform(df['DD'])
    df['WW'] = le.fit_transform(df['WW'])
    correlation_matrix = df.corr()

    # Фильтрация корреляций для признака 'T'
    correlation_matrix = df.corr()

    # Фильтрация корреляций для признака 'T'
    t_correlation = correlation_matrix['T']

    # Вывод корреляций
    print("Correlation with feature 'T':\n", t_correlation)

    # Создание тепловой карты для корреляций с признаком 'T'
    plt.figure(figsize=(10, 8))
    sns.heatmap(t_correlation.drop('T').to_frame().T, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation with Feature T Heatmap')
    plt.show()

if __name__ == '__main__':
    main()

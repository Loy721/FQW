from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_excel('SE.xls', skiprows=6)
    df = df.loc[:, ['Date', 'T', 'P', 'U', 'DD', 'Ff', 'W1', 'Td']]
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M', dayfirst=True)
    df['W1'] = df['W1'].fillna("someValue")
    le = LabelEncoder()
    df['DD'] = le.fit_transform(df['DD'])
    df['W1'] = le.fit_transform(df['W1'])
    numeric_columns = df
    # Рассчитайте матрицу корреляции
    correlation_matrix = numeric_columns.corr()

    # Создайте тепловую карту с использованием seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Pirson Heatmap')
    plt.show()
    seasonal_corr = df.set_index('Date').resample('M').mean().corr()

    # Создать тепловую карту
    plt.figure(figsize=(10, 8))
    sns.heatmap(seasonal_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Seasonal Correlation Heatmap')
    plt.show()

if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    categories = ['1', '5', '10', '31', '92', '183', '365']
    #categories = ['MLP_1D', 'LSTM_1D', 'CNN_1D', 'ConvLSTM_1D', 'MLP_nD', 'LSTM_nD', 'CNN_nD', "ConvLSTM_nD"]
    #categories = ['KNN', 'SVR', 'DTR', 'Ridge', 'MLP', 'LSTM', 'CNN', "ConvLSTM"]
    #categories = ['8', '20', '40', '80', '240', '732', '1460']
    #values = [2.1900018425433716, 2.09877357595375, 2.0605814882709788, 2.14741178422635,  1.7550961838875612, 1.4882652731197115, 1.5125001580127198 ]
    #MLNNvalues = [0.81016, 2.03431, 2.49144, 2.152660,  2.093004950876411, 2.067252773336993, 2.119964324046791, 1.7209725014510194 ]
    #values = [2.093004950876411, 2.067252773336993, 2.119964324046791, 1.7209725014510194, 1.9457410919655977, 1.9392107612790925, 1.731788521194935, 1.9291610104073176]
    values = [2.2076626952199283, 1.713472, 2.016754790296831, 2.1059485642985805, 2.1032719177669663, 2.1687486828957203, 1.54431]
    # Задание положения столбцов
    x = np.arange(len(categories))

    # Ширина столбцов
    width = 0.2
    # Построение столбчатой диаграммы
    plt.bar(x, values, width, color='skyblue', edgecolor='black')

    # Добавление подписей категорий на оси X
    plt.xticks(x, categories)

    # Добавление заголовка и подписей осей
    plt.xlabel('Количество дней')
    plt.ylabel('MAE')

    # Добавление сетки
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Отображение графика
    plt.show()
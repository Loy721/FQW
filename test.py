import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Данные
    categories = ['MLP', 'LSTM', 'RNN']
    values1 = [1.170430]
    values2 = [1.149173]
    values3 = [1.201438]


    # Определение цветов
    colors = ['r', 'g', 'b']

    # Построение столбчатой диаграммы
    plt.bar( "MLP", values1, color=colors[0], label='MLP')
    plt.bar( "LSTM",values2, color=colors[1], label='LSTM')
    plt.bar("RNN",values3, color=colors[2], label='RNN')

    # Добавление названий осей и заголовка
    plt.xlabel('Модели')
    plt.ylabel('MAE')
    plt.title('Сравнение архитектур')

    # Добавление легенды
    plt.legend()

    # Отображение диаграммы
    plt.show()

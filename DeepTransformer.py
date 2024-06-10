import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class DeepTransformer(keras.Model):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, output_dim, dropout=0.1):
        super(DeepTransformer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=input_dim)
        self.positional_encoding = layers.Embedding(input_dim=input_dim, output_dim=input_dim)

        self.transformer_blocks = [
            TransformerBlock(num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ]

        self.flatten = layers.Flatten()
        self.final_dense = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.embedding(inputs)
        seq_length = x.shape[1]
        x += self.positional_encoding(tf.range(seq_length))

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.flatten(x)
        x = self.final_dense(x)
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(num_heads),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

if __name__ == '__main__':
    # Гиперпараметры модели
    input_dim = 15  # Размерность входных данных (например, количество признаков)
    num_heads = 15   # Количество голов в многоголовом внимании
    ff_dim = 32     # Размерность скрытых слоев прямого прохода
    num_layers = 4  # Количество трансформерных блоков
    output_dim = 1  # Размерность выходного вектора (например, 1 для одномерного временного ряда)

    # Создание экземпляра модели DeepTransformer
    model = DeepTransformer(input_dim=input_dim, num_heads=num_heads, ff_dim=ff_dim,
                            num_layers=num_layers, output_dim=output_dim)

    # Компиляция модели
    model.compile(optimizer='adam', loss='mae')

    # Подготовка данных
    # Предположим, у вас есть временной ряд в формате DataFrame с именем 'data'
    # Вам нужно выполнить предварительную обработку данных и разделить их на обучающий и тестовый наборы
    # В этом примере данные будут сгенерированы случайным образом
    ds = pd.read_excel('SE.xls', skiprows=6)
    data_raw = ds['T'].ffill()[:40512]
    data = pd.DataFrame(data_raw)
    print(data_raw.shape)

    X = [data.iloc[i:i+15, :].values for i in range(len(data) - 14)]
    y = [data.iloc[i+14, :] for i in range(len(data) - 14)]

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = np.squeeze(np.array(X_train))
    y_train = np.squeeze(np.array(y_train))
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(np.squeeze(X_train).shape)

    # Обучение модели
    model.fit(X_train, y_train, epochs=10, batch_size=1, validation_split=0.2)

    # Оценка модели на тестовых данных
    loss = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
import torch.nn as nn
import torch
import numpy as np
def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        sum_j = 0.0
        for j in range(num_classes):
            sum_j += np.exp(scores[j]) #1

        for j in range(num_classes):
            dW[:, j] += (np.exp(scores[j]) * X[i]) / sum_j
            if (j == y[i]):
                dW[:, y[i]] -= X[i]

    dW /= num_train #3
    dW += W * reg

    return loss, dW

if __name__ == '__main__':
    m = nn.Softmax(dim=0)
    matrix = np.array([[2.6, 0.3, 1.1],
                           [1.7, 4, -1.7],
                           [0.5, 2.3, 2.7]])
    softmax_loss_naive(matrix, [0,1,2,3],)
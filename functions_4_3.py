import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss_log_reg(train_X, train_y, max_val=100, n_w=101):
    n_samples = train_X.shape[0]
    n_features = train_X.shape[1]
    w_list = [np.linspace(-max_val, max_val, n_w) for i in range(n_features+1)] # +1 for bias
    w = np.meshgrid(w_list, sparse=True)
    loss = np.zeros([n_w for i in range(n_features+1)])
    for i in range(n_samples):
        z = w[0]
        for x_f, w_f in zip(train_X[i], w[1:]): # iterate over features
            z += w_f * x_f
        h_w = sigmoid(z)
        loss += -1 / n_samples * (train_y[i] * np.log(h_w) + (1-train_y[i]) * np.log(1-h_w))
    return loss

def grad_log_reg(train_X, train_y, max_val=100, n_w=101):
    n_samples = train_X.shape[0]
    n_features = train_X.shape[1]
    w_list = [np.linspace(-max_val, max_val, n_w) for i in range(n_features+1)] # +1 for bias
    w = np.meshgrid(w_list, sparse=True)
    grad = np.zeros([n_features+1 if not i else n_w for i in range(n_features+2)], dtype='float32')
    for i in range(n_samples):
        z = w[0]
        for x_f, w_f in zip(train_X[i], w[1:]): # iterate over features
            z += w_f * x_f
        h_w = sigmoid(z)
        for w_i in range(n_features+1):
            grad[w_i] += -1 / n_samples * (train_y[i] - h_w) * train_X[i, w_i]
    return grad
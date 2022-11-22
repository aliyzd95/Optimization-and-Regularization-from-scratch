import numpy as np


def backward_prop_with_l2(X, Y, store, m, L, af, af_choices, parameters, landa=0.01):
    derivatives = dict()
    store['A0'] = X.T
    A = store[f'A{str(L)}']
    dZ = A - Y.T
    dW = (1. / m) * dZ.dot(store[f'A{str(L - 1)}'].T) + landa * parameters[f'W{str(L)}']
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = store[f'W{str(L)}'].T.dot(dZ)
    derivatives[f'dW{str(L)}'] = dW
    derivatives[f'db{str(L)}'] = db
    for l in range(L - 1, 0, -1):
        dZ = dA_prev * af_choices[af[l - 1]][1](store[f'Z{str(l)}'])
        dW = (1. / m) * dZ.dot(store[f'A{str(l - 1)}'].T) + landa * parameters[f'W{str(l)}']
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dA_prev = store[f'W{str(l)}'].T.dot(dZ)
        derivatives[f'dW{str(l)}'] = dW
        derivatives[f'db{str(l)}'] = db
    return derivatives


def backward_prop_with_l1(X, Y, store, m, L, af, af_choices, parameters, landa=0.0001):
    derivatives = dict()
    store['A0'] = X.T
    A = store[f'A{str(L)}']
    dZ = A - Y.T
    dW = (1. / m) * dZ.dot(store[f'A{str(L - 1)}'].T) + (landa / 2) * np.sign(parameters[f'W{str(L)}'])
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = store[f'W{str(L)}'].T.dot(dZ)
    derivatives[f'dW{str(L)}'] = dW
    derivatives[f'db{str(L)}'] = db
    for l in range(L - 1, 0, -1):
        dZ = dA_prev * af_choices[af[l - 1]][1](store[f'Z{str(l)}'])
        dW = (1. / m) * dZ.dot(store[f'A{str(l - 1)}'].T) + (landa / 2) * np.sign(parameters[f'W{str(l)}'])
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dA_prev = store[f'W{str(l)}'].T.dot(dZ)
        derivatives[f'dW{str(l)}'] = dW
        derivatives[f'db{str(l)}'] = db
    return derivatives


def crossentropy_with_l2(A, Y, parameters, L, landa=0.01):
    cost = -np.mean(Y * np.log(A.T))
    W_sum = 0
    for l in range(1, L + 1):
        W_sum += np.sum(parameters[f'W{str(l)}'] ** 2)
    return np.squeeze(cost + (landa / 2) * W_sum)


def crossentropy_with_l1(A, Y, parameters, L, landa=0.0001):
    cost = -np.mean(Y * np.log(A.T))
    W_sum = 0
    for l in range(1, L + 1):
        W_sum += np.sum(np.abs(parameters[f'W{str(l)}']))
    return np.squeeze(cost + (landa / 2) * W_sum)

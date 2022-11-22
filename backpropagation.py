import numpy as np
import matplotlib.pylab as plt


def xavier_initializer(ni, nh):  # Xavier normal
    np.random.seed(1)
    nin = ni
    nout = nh
    ih_weights = np.zeros((ni, nh))
    sd = np.sqrt(2.0 / (nin + nout))
    for i in range(ni):
        for j in range(nh):
            x = np.float64(np.random.normal(0.0, sd))
            ih_weights[i, j] = x
    return ih_weights


def initialize_parameters(layers, L):
    np.random.seed(1)
    parameters = dict()
    for l in range(1, L + 1):
        # parameters[f'W{str(l)}'] = np.random.randn(layers[l], layers[l - 1]) / np.sqrt(layers[l - 1])
        parameters[f'W{str(l)}'] = xavier_initializer(layers[l], layers[l - 1])
        parameters[f'b{str(l)}'] = np.zeros((layers[l], 1))
    return parameters


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def d_sigmoid(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)


def relu(Z):
    A = np.maximum(0, Z)
    return A


def d_relu(Z):
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z


def tanh(Z):
    return np.tanh(Z)


def d_tanh(Z):
    return 1.0 - np.tanh(Z) ** 2


def linear(Z):
    return Z


def d_linear(Z):
    return np.ones(Z.shape)


# def softmax(Z):
#     # exps = np.exp(Z - np.max(Z))
#     # return exps / np.sum(exps)
#     Z -= np.max(Z)
#     sm = (np.exp(Z) / np.sum(np.exp(Z), axis=0))
#     return sm

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)


def categorical_crossentropy(A, Y):
    return -np.mean(Y * np.log(A.T))


def plot_cost(costs):
    plt.figure()
    plt.plot(np.arange(len(costs)), costs)
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.show()


def forward_prop(X, parameters, L, af, af_choices):
    store = dict()
    A = X.T
    for l in range(L - 1):
        Z = parameters[f'W{str(l + 1)}'].dot(A) + parameters[f'b{str(l + 1)}']
        A = af_choices[af[l]][0](Z)
        store[f'A{str(l + 1)}'] = A
        store[f'W{str(l + 1)}'] = parameters[f'W{str(l + 1)}']
        store[f'Z{str(l + 1)}'] = Z
    Z = parameters[f'W{str(L)}'].dot(A) + parameters[f'b{str(L)}']
    A = af_choices[af[-1]][0](Z)
    store[f"A{str(L)}"] = A
    store[f"W{str(L)}"] = parameters[f"W{str(L)}"]
    store[f"Z{str(L)}"] = Z
    return A, store


def backward_prop(X, Y, store, m, L, af, af_choices):
    derivatives = dict()
    store['A0'] = X.T
    A = store[f'A{str(L)}']
    dZ = A - Y.T
    dW = (1. / m) * dZ.dot(store[f'A{str(L - 1)}'].T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = store[f'W{str(L)}'].T.dot(dZ)
    derivatives[f'dW{str(L)}'] = dW
    derivatives[f'db{str(L)}'] = db
    for l in range(L - 1, 0, -1):
        dZ = dA_prev * af_choices[af[l - 1]][1](store[f'Z{str(l)}'])
        dW = (1. / m) * dZ.dot(store[f'A{str(l - 1)}'].T)
        db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dA_prev = store[f'W{str(l)}'].T.dot(dZ)
        derivatives[f'dW{str(l)}'] = dW
        derivatives[f'db{str(l)}'] = db
    return derivatives


def pred(X, Y, parameters, L, af, af_choices):
    A, store = forward_prop(X, parameters, L, af, af_choices)
    Y_hat = np.argmax(A, axis=0)
    Y = np.argmax(Y, axis=1)
    accuracy = (Y_hat == Y).mean()
    return accuracy * 100


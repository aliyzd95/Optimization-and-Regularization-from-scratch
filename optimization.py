from backpropagation import *


def gradient_descent(train_X, train_Y, epochs, l_rate, layers, L, af, af_choices, batch_size):
    s = batch_size
    m = s
    parameters = initialize_parameters(layers, L)
    costs = []
    accs = []
    for e in range(epochs):
        for i in range(0, train_X.shape[0], s):
            X = train_X[i:s + i, :]
            Y = train_Y[i:s + i, :]
            A, store = forward_prop(X, parameters, L, af, af_choices)
            derivatives = backward_prop(X, Y, store, m, L, af, af_choices)
            for l in range(1, L + 1):
                parameters[f'W{str(l)}'] -= l_rate * derivatives[f'dW{str(l)}']
                parameters[f'b{str(l)}'] -= l_rate * derivatives[f'db{str(l)}']
        if e % 1 == 0:
            AA, _ = forward_prop(train_X, parameters, L, af, af_choices)
            cost = categorical_crossentropy(AA, train_Y)
            acc = pred(train_X, train_Y, parameters, L, af, af_choices)
            costs.append(cost)
            accs.append(acc)
            print(f'epoch: {e} - cost= {cost} - Train Acc= {acc}')
    return parameters, costs, accs


def adagrad(train_X, train_Y, epochs, l_rate, layers, L, af, af_choices, batch_size):
    s = batch_size
    m = s
    parameters = initialize_parameters(layers, L)
    costs = []
    accs = []
    eps = 1e-7
    r_dW = dict()
    r_db = dict()
    X = train_X[0:s, :]
    Y = train_Y[0:s, :]
    A, store = forward_prop(X, parameters, L, af, af_choices)
    derivatives = backward_prop(X, Y, store, m, L, af, af_choices)
    for l in range(1, L + 1):
        r_dW[f'dW{str(l)}'] = np.zeros(derivatives[f'dW{str(l)}'].shape)
        r_db[f'db{str(l)}'] = np.zeros(derivatives[f'db{str(l)}'].shape)
    for e in range(epochs):
        for i in range(0, train_X.shape[0], s):
            X = train_X[i:s + i, :]
            Y = train_Y[i:s + i, :]
            A, store = forward_prop(X, parameters, L, af, af_choices)
            derivatives = backward_prop(X, Y, store, m, L, af, af_choices)
            for l in range(1, L + 1):
                r_dW[f'dW{str(l)}'] += derivatives[f'dW{str(l)}'] ** 2
                r_db[f'db{str(l)}'] += derivatives[f'db{str(l)}'] ** 2
                parameters[f'W{str(l)}'] -= (l_rate / (eps + np.sqrt(r_dW[f'dW{str(l)}']))) * derivatives[f'dW{str(l)}']
                parameters[f'b{str(l)}'] -= (l_rate / (eps + np.sqrt(r_db[f'db{str(l)}']))) * derivatives[f'db{str(l)}']
        if e % 1 == 0:
            AA, _ = forward_prop(train_X, parameters, L, af, af_choices)
            cost = categorical_crossentropy(AA, train_Y)
            acc = pred(train_X, train_Y, parameters, L, af, af_choices)
            accs.append(acc)
            costs.append(cost)
            print(f'epoch: {e} - cost= {cost} - Train Acc= {acc}')
    return parameters, costs, accs


def rmsprop(train_X, train_Y, epochs, l_rate, layers, L, af, af_choices, batch_size):
    s = batch_size
    m = s
    parameters = initialize_parameters(layers, L)
    costs = []
    accs = []
    eps = 1e-8
    rho = 0.9
    r_dW = dict()
    r_db = dict()
    X = train_X[0:s, :]
    Y = train_Y[0:s, :]
    A, store = forward_prop(X, parameters, L, af, af_choices)
    derivatives = backward_prop(X, Y, store, m, L, af, af_choices)
    for l in range(1, L + 1):
        r_dW[f'dW{str(l)}'] = np.zeros(derivatives[f'dW{str(l)}'].shape)
        r_db[f'db{str(l)}'] = np.zeros(derivatives[f'db{str(l)}'].shape)
    for e in range(epochs):
        for i in range(0, train_X.shape[0], s):
            X = train_X[i:s + i, :]
            Y = train_Y[i:s + i, :]
            A, store = forward_prop(X, parameters, L, af, af_choices)
            derivatives = backward_prop(X, Y, store, m, L, af, af_choices)
            for l in range(1, L + 1):
                r_dW[f'dW{str(l)}'] = rho * r_dW[f'dW{str(l)}'] + (1 - rho) * derivatives[f'dW{str(l)}'] ** 2
                r_db[f'db{str(l)}'] = rho * r_db[f'db{str(l)}'] + (1 - rho) * derivatives[f'db{str(l)}'] ** 2
                parameters[f'W{str(l)}'] -= (l_rate / np.sqrt(eps + r_dW[f'dW{str(l)}'])) * derivatives[f'dW{str(l)}']
                parameters[f'b{str(l)}'] -= (l_rate / np.sqrt(eps + r_db[f'db{str(l)}'])) * derivatives[f'db{str(l)}']
        if e % 1 == 0:
            AA, _ = forward_prop(train_X, parameters, L, af, af_choices)
            cost = categorical_crossentropy(AA, train_Y)
            acc = pred(train_X, train_Y, parameters, L, af, af_choices)
            accs.append(acc)
            costs.append(cost)
            print(f'epoch: {e} - cost= {cost} - Train Acc= {acc}')
    return parameters, costs, accs


def adam(train_X, train_Y, epochs, l_rate, layers, L, af, af_choices, batch_size):
    s = batch_size
    m = s
    parameters = initialize_parameters(layers, L)
    costs = []
    accs = []
    eps = 1e-8
    rho1 = 0.9
    rho2 = 0.999
    s_dW = dict()
    s_db = dict()
    r_dW = dict()
    r_db = dict()
    X = train_X[0:s, :]
    Y = train_Y[0:s, :]
    A, store = forward_prop(X, parameters, L, af, af_choices)
    derivatives = backward_prop(X, Y, store, m, L, af, af_choices)
    for l in range(1, L + 1):
        s_dW[f'dW{str(l)}'] = np.zeros(derivatives[f'dW{str(l)}'].shape)
        s_db[f'db{str(l)}'] = np.zeros(derivatives[f'db{str(l)}'].shape)
        r_dW[f'dW{str(l)}'] = np.zeros(derivatives[f'dW{str(l)}'].shape)
        r_db[f'db{str(l)}'] = np.zeros(derivatives[f'db{str(l)}'].shape)
    t = 0
    for e in range(epochs):
        for i in range(0, train_X.shape[0], s):
            X = train_X[i:s + i, :]
            Y = train_Y[i:s + i, :]
            A, store = forward_prop(X, parameters, L, af, af_choices)
            derivatives = backward_prop(X, Y, store, m, L, af, af_choices)
            t += 1
            for l in range(1, L + 1):
                s_dW[f'dW{str(l)}'] = rho1 * s_dW[f'dW{str(l)}'] + (1 - rho1) * derivatives[f'dW{str(l)}']
                s_db[f'db{str(l)}'] = rho1 * s_db[f'db{str(l)}'] + (1 - rho1) * derivatives[f'db{str(l)}']
                r_dW[f'dW{str(l)}'] = rho2 * r_dW[f'dW{str(l)}'] + (1 - rho2) * (derivatives[f'dW{str(l)}'] ** 2)
                r_db[f'db{str(l)}'] = rho2 * r_db[f'db{str(l)}'] + (1 - rho2) * (derivatives[f'db{str(l)}'] ** 2)
                s_dW_hat = s_dW[f'dW{str(l)}'] / (1 - (rho1 ** t))
                s_db_hat = s_db[f'db{str(l)}'] / (1 - (rho1 ** t))
                r_dW_hat = r_dW[f'dW{str(l)}'] / (1 - (rho2 ** t))
                r_db_hat = r_db[f'db{str(l)}'] / (1 - (rho2 ** t))
                parameters[f'W{str(l)}'] -= (l_rate * s_dW_hat) / (eps + np.sqrt(r_dW_hat))
                parameters[f'b{str(l)}'] -= (l_rate * s_db_hat) / (eps + np.sqrt(r_db_hat))
        if e % 1 == 0:
            AA, _ = forward_prop(train_X, parameters, L, af, af_choices)
            cost = categorical_crossentropy(AA, train_Y)
            acc = pred(train_X, train_Y, parameters, L, af, af_choices)
            accs.append(acc)
            costs.append(cost)
            print(f'epoch: {e} - cost= {cost} - Train Acc= {acc}')
    return parameters, costs, accs

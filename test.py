from optimization_with_regularization import *
from save_data import load_cifar10

# available activation functions

activation_function_choices = {'relu': [relu, d_relu], 'tanh': [tanh, d_tanh], 'sigmoid': [sigmoid, d_sigmoid],
                               'linear': [linear, d_linear], 'softmax': [softmax]}

# load data
train_X, train_Y, test_X, test_Y = load_cifar10()
# 'relu' | 'tanh' | 'sigmoid' | 'linear' | 'softmax'
layers = [
    {'units': 400, 'activation': 'relu'},
    {'units': 20, 'activation': 'relu'},
    {'units': train_Y.shape[1], 'activation': 'softmax'}
]

# number_of_epochs = 29
learning_rate = 0.0001
batch_size = 1000
# gradient_descent: 22 / 0.1 --> 51.95 | L1: 16 / 0.1 --> 51.45 | L2: 30 / 0.1 --> 52.66
# adagrad: 100 / 0.001 --> 52.40
# adam: 40 / 0.0001 --> 52.76 | L2: 175 / 0.0001 --> 55.55
# rmsprop: 35 / 0.0001 --> 52.55


layers_dims = [i['units'] for i in layers]
activations = [i['activation'] for i in layers]
layers_dims.insert(0, train_X.shape[1])
n_layers = len(layers)

params0, costs0, accs0 = adam(train_X, train_Y, 10, learning_rate, layers_dims, n_layers,
                              activations, activation_function_choices, batch_size)

params1, costs1, accs1 = adam_with_l1(train_X, train_Y, 10, learning_rate, layers_dims, n_layers,
                                      activations, activation_function_choices, batch_size)

params2, costs2, accs2 = adam_with_l2(train_X, train_Y, 10, learning_rate, layers_dims, n_layers,
                                      activations, activation_function_choices, batch_size)

print('Gradient Descent:')
print("Test Accuracy:", pred(test_X, test_Y, params0, n_layers, activations, activation_function_choices))
print("Test Accuracy with L1:", pred(test_X, test_Y, params1, n_layers, activations, activation_function_choices))
print("Test Accuracy with L2:", pred(test_X, test_Y, params2, n_layers, activations, activation_function_choices))

plt.subplot(2, 1, 1)
plt.plot(np.arange(len(costs0)), costs0, '.-', c='red', label='-')
plt.plot(np.arange(len(costs1)), costs1, '.-', c='green', label='L1')
plt.plot(np.arange(len(costs2)), costs2, '.-', c='blue', label='L2')
plt.title('Adam')
plt.ylabel('Cost')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(accs0)), accs0, '.-', c='red', label='-')
plt.plot(np.arange(len(accs1)), accs1, '.-', c='green', label='L1')
plt.plot(np.arange(len(accs2)), accs2, '.-', c='blue', label='L2')
plt.xlabel('Epochs')
plt.ylabel('Train Accuracy')

plt.show()

import numpy as np
import pickle
import os
import cv2
from tqdm import tqdm


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def extract_cfar10():
    try:
        os.mkdir('cifar10_images')
        os.mkdir(os.path.join('cifar10_images', 'test'))
        os.mkdir(os.path.join('cifar10_images', 'train'))
        for i in range(10):
            os.mkdir(os.path.join('cifar10_images', 'test', str(i + 1)))
            os.mkdir(os.path.join('cifar10_images', 'train', str(i + 1)))
    except:
        pass

    for batch in range(5):
        dic = unpickle('cifar-10-batches-py/data_batch_' + str(batch + 1))
        for i in tqdm(range(10000)):
            image = dic[b'data'][i].reshape(3, 32, 32)
            image = np.transpose(image, [1, 2, 0])
            cv2.imwrite(os.path.join('cifar10_images', 'train',
                                     str(dic[b'labels'][i]),
                                     str(i + batch * 10000) + '.jpg'), image)

    dic = unpickle('cifar-10-batches-py/test_batch')
    for i in tqdm(range(10000)):
        image = dic[b'data'][i].reshape(3, 32, 32)
        image = np.transpose(image, [1, 2, 0])
        cv2.imwrite(os.path.join('cifar10_images', 'test',
                                 str(dic[b'labels'][i]), str(i) + '.jpg'), image)


def load_cifar10():
    X_train = np.zeros((10000 * 5, 3072))
    y_train = np.zeros(10000 * 5, dtype=np.int32)
    for batch in range(5):
        dic = unpickle('cifar-10-batches-py/data_batch_' + str(batch + 1))
        X_train[batch * 10000: (batch + 1) * 10000] = dic[b'data']
        y_train[batch * 10000: (batch + 1) * 10000] = dic[b'labels']

    dic = unpickle('cifar-10-batches-py/test_batch')
    X_test = np.array(dic[b'data'])
    y_test = np.array(dic[b'labels'])

    ####################################################################################### normalization
    m = X_train.mean()
    v = X_train.std()
    X_train = (X_train - m) / v
    X_test = (X_test - m) / v

    ####################################################################################### one-hot encoding
    y_train_categorical = np.zeros((y_train.shape[0], 10))
    y_train_categorical[np.arange(y_train.shape[0]), y_train] = 1
    y_test_categorical = np.zeros((y_test.shape[0], 10))
    y_test_categorical[np.arange(y_test.shape[0]), y_test] = 1

    return X_train, y_train_categorical, X_test, y_test_categorical

# X_train, y_train, X_test, y_test = load_cifar10()
# print(X_train.shape)
# print(y_train.shape)

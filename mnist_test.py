#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2018-04-23 23:08
import numpy as np
import sys
from sklearn import preprocessing
from tensor_lite.util import *
from tensor_lite.tensor import *
from tensor_lite.model import *

from mnist import MNIST


def get_training_set():
    global X_train, Y_train, X_test, Y_test

    mndata = MNIST('./mnist_data')
    data = mndata.load_training()
    X_train = np.asarray(data[0], dtype=np.float64)
    Y_train = np.asarray(data[1])
    Y_train = Y_train.reshape(-1, 1)
    Y_train = one_hot(Y_train, 10)

    data = mndata.load_testing()
    X_test = np.asarray(data[0], dtype=np.float64)
    Y_test = np.asarray(data[1])
    Y_test = Y_test.reshape(-1, 1)
    Y_test = one_hot(Y_test, 10)


get_training_set()

# ----------------------------- CNN ---------------------------------
X_train = X_train.reshape(len(X_train), 1, 28, 28)
X_test = X_test.reshape(len(X_test), 1, 28, 28)

model = CNN(feature_shape=(1, 28, 28))
model.add_conv_layer(kernel_shape=(3, 3, 3))
model.add_maxpool_layer(size=2, stride=2)
model.add_fully_connected_layer(shape=(20, 10), dropout=0.8)
model.fit(
    X_train,
    Y_train,
    learning_rate=0.001,
    epoch=10,
    batch_size=16,
    regularizer=L2Regularizer(0.001))

# ----------------------------- ANN ---------------------------------
# model = ANN(shape=(784, 30, 10), dropout=1)
# # ann.check_gradient(X_train, Y_train)
# model.fit(
#     X_train,
#     Y_train,
#     batch_size=8,
#     epoch=50,
#     learning_rate=0.001,
#     regularizer=L2Regularizer(0.01))

# ----------------------------- LR ---------------------------------

# model = LogisticRegression(shape=(784, 10))
# model.fit(
#     X_train,
#     Y_train,
#     batch_size=512,
#     epoch=20,
#     learning_rate=0.01,
#     regularizer=L2Regularizer(0.001))

y_hat = model.predict(X_train)
y_hat = np.argmax(y_hat, axis=1)
y = np.argmax(Y_train, axis=1)
print("accu on training set:", np.sum(y == y_hat) / len(y))

y_hat = model.predict(X_test)
y_hat = np.argmax(y_hat, axis=1)
y = np.argmax(Y_test, axis=1)
print("accu on test set:", np.sum(y == y_hat) / len(y))

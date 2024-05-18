#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2018-04-23 23:08
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.datasets import load_digits
from tensor_lite.util import *
from tensor_lite.tensor import *
from tensor_lite.model import *


def get_training_set():
    X, Y = load_digits(10, True)
    Y = Y.reshape(-1, 1)
    Y = one_hot(Y, 10)
    [m, features] = X.shape
    Z = np.concatenate((X, Y), axis=1)
    np.random.shuffle(Z)
    X = Z[:, :features]
    Y = Z[:, features:]
    # X = preprocessing.normalize(X)
    offset = int(0.8 * m)
    global X_train, Y_train, X_test, Y_test
    X_train, Y_train = X[:offset], Y[:offset]
    X_test, Y_test = X[offset:], Y[offset:]


get_training_set()

# # ----------------------------- ANN ---------------------------------
model = ANN(shape=(64, 40, 10), dropout=0.9)
# model.check_gradient(X_train, Y_train)
model.fit(
    X_train,
    Y_train,
    batch_size=4,
    epoch=50,
    learning_rate=0.01,
    regularizer=L2Regularizer(0.01))

# # ----------------------------- CNN ---------------------------------
# X_train = X_train.reshape(len(X_train), 1, 8, 8)
# X_test = X_test.reshape(len(X_test), 1, 8, 8)

# model = CNN(feature_shape=(1, 8, 8))
# model.add_conv_layer(kernel_shape=(3, 3, 3))
# model.add_maxpool_layer(size=2, stride=2)
# model.add_fully_connected_layer(shape=(20, 10))
# # model.check_gradient(X_train, Y_train)
# model.fit(
#     X_train,
#     Y_train,
#     learning_rate=0.01,
#     epoch=20,
#     batch_size=4,
#     regularizer=L2Regularizer(0.001))
# ----------------------------- LR ---------------------------------
# model = LogisticRegression(shape=(64, 10))
# model.check_gradient(X_train, Y_train)
# model.fit(
#     X_train,
#     Y_train,
#     batch_size=4,
#     epoch=50,
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

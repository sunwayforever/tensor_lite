#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2018-04-23 23:08
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from tensor_lite.tensor import *
from tensor_lite.model import *

EPOCH = 1000
LEARNING_RATE = 0.01


def get_training_set():
    data = np.loadtxt("../simple/data.txt", delimiter=",")
    X = data[:, 0].reshape(-1, 1)
    X = preprocessing.scale(X)
    Y = data[:, 1].reshape(-1, 1)
    Y = preprocessing.scale(Y)
    return X, Y


X, Y = get_training_set()
model = LinearRegression(feature_size=1)
model.check_gradient(X, Y)
model.fit(X, Y, LEARNING_RATE, EPOCH)

plt.scatter(x=X[:, 0], y=Y[:, 0])
plt.plot(X, model.predict(X), "r")
plt.show()

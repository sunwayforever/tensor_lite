#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2018-04-23 23:08
from sklearn import preprocessing
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from tensor_lite.tensor import *
from tensor_lite.model import *

EPOCH = 100
LEARNING_RATE = 0.1
POLY_FEATURES = 5
BATCH_SIZE = 4


def get_training_set():
    X, Y = make_moons(n_samples=500, noise=0.1)
    Y = Y.reshape(500, 1)

    FEATURE_SIZE = X.shape[1]

    Z = np.concatenate((X, Y), axis=1)
    np.random.shuffle(Z)
    X = Z[:, :FEATURE_SIZE]
    Y = Z[:, FEATURE_SIZE:]

    poly = preprocessing.PolynomialFeatures(POLY_FEATURES, include_bias=False)
    X = poly.fit_transform(X)

    return X, Y


X, Y = get_training_set()
model = LogisticRegression(shape=(X.shape[1], 1))
model.check_gradient(X, Y)
model.fit(
    X, Y, learning_rate=LEARNING_RATE, epoch=EPOCH, batch_size=BATCH_SIZE)

cm = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(x=X[:, 0], y=X[:, 1], c=Y[:, 0], cmap=cm)
xx, yy = np.meshgrid(np.arange(-1.5, 2.5, 0.02), np.arange(-1.5, 2.5, 0.02))
X = np.c_[xx.ravel(), yy.ravel()]
poly = preprocessing.PolynomialFeatures(POLY_FEATURES, include_bias=False)
X = poly.fit_transform(X)

Y = (model.predict(X) > 0.5)[:, 0]
Y = Y.reshape(xx.shape)
cm = ListedColormap(['#FF0000', '#0000FF'])
plt.contour(xx, yy, Y, cmap=cm)

plt.show()

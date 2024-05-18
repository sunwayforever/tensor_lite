#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2018-05-10 22:04
import numpy as np


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def cross_entropy(predictions, labels):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    N = predictions.shape[0]
    zzz = labels * np.log(predictions)
    ce = 0 - (np.sum(zzz, axis=1))
    ce = ce.reshape(-1, 1)
    return ce


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def relu(data, epsilon=0.1):
    return np.maximum(epsilon * data, data)


def relu_prime(data, epsilon=0.1):
    gradients = 1. * (data > 0)
    gradients[gradients == 0] = epsilon
    return gradients

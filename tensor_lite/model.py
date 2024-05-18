#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2018-05-12 21:38
import numpy as np
from tensor_lite.tensor import *
from tensor_lite.regularizer import *
from tensor_lite.util import *


class Model:
    def __init__(self, X, Y, H, loss):
        self.X = X
        self.Y = Y
        self.H = H
        self.loss = loss

    def fit(self,
            X,
            Y,
            learning_rate=0.01,
            epoch=200,
            batch_size=0,
            regularizer=None):
        if regularizer is not None:
            regularizer.initialize(self.loss)
        if batch_size == 0:
            batch_size = len(X)
        batch = len(X) // batch_size
        for i in range(epoch):
            for X_batch, Y_batch in zip(
                    np.split(X[:batch * batch_size], batch),
                    np.split(Y[:batch * batch_size], batch)):
                self.X.feed(X_batch)
                self.Y.feed(Y_batch)
                self.loss.eval()
                if regularizer is not None:
                    regularizer.regularize()
                self.loss.update(learning_rate)

            print("training: #%d, %f" % (i, self.loss.cost()))

    def predict(self, X):
        self.X.feed(X)
        self.H.forward()
        return self.H.val

    def check_gradient(self, X, Y):
        variables = []

        def collect_variable(tensor):
            if isinstance(tensor, Variable):
                variables.append(tensor)

        self.loss.traverse(collect_variable)

        self.X.feed(X)
        self.Y.feed(Y)
        self.loss.eval()

        J = self.loss.cost()

        for variable in variables:
            num_gradients = []
            for i in np.nditer(variable.val, op_flags=['readwrite']):
                i[...] += 1e-4
                self.loss.forward()
                J2 = self.loss.cost()
                i[...] -= 1e-4
                d = (J2 - J) / 1e-4
                num_gradients.append(d)
            d2 = np.asarray(num_gradients).reshape(variable.d.shape)
            if not np.allclose(d2, variable.d, atol=1e-2):
                print("check_gradient failed for ", variable.name, variable,
                      d2, variable.d)
            else:
                print("check_gradient passed for ", variable.name, variable)


class LinearRegression(Model):
    def __init__(self, feature_size):
        W = Weight((feature_size, 1))
        B = Bias((1, 1))
        X = PlaceHolder()
        Y = PlaceHolder()
        H = LinearTensor(W, X, B)
        loss = MSETensor(H, Y)
        loss.initialize()
        super().__init__(X, Y, H, loss)


class LogisticRegression(Model):
    def __init__(self, shape):
        W = Weight(shape)
        B = Bias((1, shape[1]))
        X = PlaceHolder()
        Y = PlaceHolder()
        L = LinearTensor(W, X, B)
        class_count = shape[1]
        if class_count == 1:
            H = SigmoidTensor(L)
            loss = SigmoidCrossEntropyTensor(L, Y)
        else:
            H = SoftmaxTensor(L)
            loss = SoftmaxCrossEntropyTensor(L, Y)
        loss.initialize()
        super().__init__(X, Y, H, loss)


activation_funcs = {
    "relu": ReluTensor,
    "sigmoid": SigmoidTensor,
    "tanh": TanhTensor
}


class ANN(Model):
    def __init__(self, shape, activation_func="relu", dropout=1):
        # shape: (feature_size, HIDDEN_ONE, HIDDEN_TWO,...,output_size)
        X, Y = PlaceHolder(), PlaceHolder()
        activation = X
        self.dropout_tensors = []
        for f, t in zip(shape, shape[1:]):
            W = Weight((f, t))
            B = Bias((1, t))
            activation = activation_funcs[activation_func](LinearTensor(
                W, activation, B))
            activation = DropoutTensor(activation, dropout)
            self.dropout_tensors.append(activation)
        H = SoftmaxTensor(activation)
        loss = SoftmaxCrossEntropyTensor(activation, Y)
        loss.initialize()
        super().__init__(X, Y, H, loss)

    def predict(self, X):
        for dropout in self.dropout_tensors:
            dropout.ratio = 1
        return super().predict(X)


class CNN(Model):
    def __init__(self, feature_shape):
        self.X = PlaceHolder()
        self.Y = PlaceHolder()
        self.prev_activation = self.X
        self.prev_activation_shape = feature_shape
        self.dropout_tensors = []

    def add_conv_layer(self,
                       kernel_shape,
                       stride=1,
                       padding=1,
                       activation_func="relu"):

        tensor = activation_funcs[activation_func](ConvTensor(
            self.prev_activation,
            x_shape=self.prev_activation_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding))

        n, kernel_size = kernel_shape[0], kernel_shape[1]
        _, h, w = self.prev_activation_shape
        h = w = (h - kernel_size + 2 * padding) // stride + 1

        self.prev_activation_shape = (n, h, w)
        self.prev_activation = tensor

    def add_maxpool_layer(self, size, stride):
        tensor = MaxPoolTensor(
            self.prev_activation,
            self.prev_activation_shape,
            size=size,
            stride=stride)
        n, h, w = self.prev_activation_shape

        h = w = (h - size) // stride + 1
        self.prev_activation_shape = (n, h, w)
        self.prev_activation = tensor

    def add_fully_connected_layer(self, shape, activation_func="relu", dropout=1):
        activation = FlatTensor(self.prev_activation)

        for f, t in zip((np.prod(self.prev_activation_shape), *shape), shape):
            W = Weight((f, t))
            B = Bias((1, t))
            activation = activation_funcs[activation_func](LinearTensor(
                W, activation, B))
            
            activation = DropoutTensor(activation, dropout)
            self.dropout_tensors.append(activation)
            
        H = SoftmaxTensor(activation)
        loss = SoftmaxCrossEntropyTensor(activation, self.Y)
        loss.initialize()
        self.H = H
        self.loss = loss

    def predict(self, X):
        for dropout in self.dropout_tensors:
            dropout.ratio = 1
        return super().predict(X)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2018-05-10 17:09
import numpy as np

from tensor_lite.util import *
from tensor_lite.im2col import *
from tensor_lite.im2col_cython import *

class Tensor:
    def __init__(self, *p):
        self.p = p
        self.val = None

    def eval(self):
        self.forward()
        self.backward()

    def backward(self, gradients=None):
        if len(self.p) == 0:
            return

        for leaf, d in zip(self.p, gradients):
            leaf.backward(d)

    def forward(self):
        for leaf in self.p:
            leaf.forward()

    def cost(self):
        return self.val.mean(axis=0)

    def update(self, alpha):
        for leaf in self.p:
            leaf.update(alpha)

    def initialize(self):
        for leaf in self.p:
            leaf.initialize()

    def traverse(self, func):
        func(self)
        for leaf in self.p:
            leaf.traverse(func)


class PlaceHolder(Tensor):
    def __init__(self, val=None):
        super().__init__()
        self.val = val

    def feed(self, val):
        self.val = val


class Variable(Tensor):
    def __init__(self, shape, name=None):
        super().__init__()
        self.shape = shape
        self.name = name

    def update(self, alpha):
        self.val -= alpha * self.d

    def backward(self, d):
        self.d = d


class Weight(Variable):
    def initialize(self):
        self.val = np.random.random_sample(self.shape) / 10.


class Bias(Variable):
    def initialize(self):
        self.val = np.zeros(self.shape)


class ReluTensor(Tensor):
    def forward(self):
        super().forward()
        self.val = relu(self.p[0].val)

    def backward(self, d):
        dx = d * (relu_prime(self.p[0].val))
        super().backward([dx])


class LinearTensor(Tensor):
    def forward(self):
        super().forward()
        self.val = np.dot(self.p[1].val, self.p[0].val) + self.p[2].val

    def backward(self, d=None):
        if d is None:
            d = np.ones_like(self.val)
        self.dw = np.dot(self.p[1].val.T, d)
        self.dx = np.dot(d, self.p[0].val.T)
        self.db = d
        m = self.val.shape[0]
        super().backward([self.dw / m, self.dx, self.db.mean(axis=0)])


class TanhTensor(Tensor):
    def forward(self):
        super().forward()
        self.val = np.tanh(self.p[0].val)

    def backward(self, d):
        self.dx = d * (1. - self.val * self.val)
        super().backward([self.dx])


class SigmoidTensor(Tensor):
    def forward(self):
        super().forward()
        self.val = sigmoid(self.p[0].val)

    def backward(self, d):
        self.dx = d * self.val * (1 - self.val)
        super().backward([self.dx])


class SoftmaxTensor(Tensor):
    def forward(self):
        super().forward()
        self.val = softmax(self.p[0].val)

    def backward(self, d):
        pass


class MSETensor(Tensor):
    def forward(self):
        super().forward()
        self.val = np.square(np.subtract(self.p[0].val, self.p[1].val)) / 2

    def backward(self, d=None):
        if d is None:
            d = np.ones_like(self.val)

        self.dx = (self.p[0].val - self.p[1].val) * d
        super().backward([self.dx, 1])


class SigmoidCrossEntropyTensor(Tensor):
    def forward(self):
        super().forward()
        self.y_hat = sigmoid(self.p[0].val)
        self.val = cross_entropy(np.c_[self.y_hat, 1 - self.y_hat],
                                 np.c_[self.p[1].val, 1 - self.p[1].val])

    def backward(self, d=None):
        if d is None:
            d = np.ones_like(self.val)
        self.dx = (self.y_hat - self.p[1].val) * d
        super().backward([self.dx, 1])


class SoftmaxCrossEntropyTensor(Tensor):
    def forward(self):
        super().forward()
        self.y_hat = softmax(self.p[0].val)
        self.val = cross_entropy(self.y_hat, self.p[1].val)

    def backward(self, d=None):
        if d is None:
            d = np.ones_like(self.p[0].val)
        self.dx = (self.y_hat - self.p[1].val) * d
        super().backward([self.dx, 1])


class FlatTensor(Tensor):
    def forward(self):
        super().forward()
        x = self.p[0].val
        n = x.shape[0]
        self.val = x.reshape(n, -1)

    def backward(self, d):
        super().backward([d.reshape(self.p[0].val.shape)])


class DropoutTensor(Tensor):
    def __init__(self, x, ratio):
        super().__init__(x)
        self.ratio = ratio

    def forward(self):
        super().forward()
        
        mask_shape = (1, *self.p[0].val.shape[1:])
        self.dropout_mask = (np.random.rand(*mask_shape) < self.ratio) / self.ratio
        self.val = self.p[0].val * self.dropout_mask
        
    def backward(self, d):
        d *= self.dropout_mask
        super().backward([d])


class ConvTensor(Tensor):
    def __init__(self, x, x_shape, kernel_shape, stride, padding):
        self.d_X, self.h_X, self.w_X = x_shape

        self.n_kernel, self.h_kernel, self.w_kernel = kernel_shape
        self.stride, self.padding = stride, padding
        self.W = Weight(
            shape=((self.n_kernel, self.d_X, self.h_kernel,
                    self.w_kernel) // np.sqrt(self.n_kernel // 2)).astype(int))
        self.B = Bias((self.n_kernel, 1))

        super().__init__(x, self.W, self.B)

        self.h_out = (self.h_X - self.h_kernel + 2 * padding) / stride + 1
        self.w_out = (self.w_X - self.w_kernel + 2 * padding) / stride + 1

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)

    def forward(self):
        super().forward()

        self.n_X = self.p[0].val.shape[0]
        self.X_col = im2col_cython(
            self.p[0].val,
            self.h_kernel,
            self.w_kernel,
            self.padding, 
            self.stride,
            )
        W_row = self.W.val.reshape(self.n_kernel, -1)

        out = W_row @ self.X_col + self.B.val
        out = out.reshape(self.n_kernel, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)

        self.val = out

    def backward(self, d):
        dout_flat = d.transpose(1, 2, 3, 0).reshape(self.n_kernel, -1)

        self.dw = dout_flat @ self.X_col.T
        self.dw = self.dw.reshape(self.W.shape)

        self.db = np.sum(d, axis=(0, 2, 3)).reshape(self.n_kernel, -1)

        W_flat = self.W.val.reshape(self.n_kernel, -1)

        dX_col = W_flat.T @ dout_flat
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        dx = col2im_cython(dX_col, shape[0], shape[1], shape[2], shape[3], self.h_kernel, self.w_kernel,
                            self.padding, self.stride)

        super().backward([dx, self.dw / self.n_X, self.db / self.n_X])


class MaxPoolTensor(Tensor):
    def __init__(self, x, x_shape, size, stride):
        super().__init__(x)

        self.d_X, self.h_X, self.w_X = x_shape

        self.params = []

        self.size = size
        self.stride = stride

        self.h_out = (self.h_X - size) / stride + 1
        self.w_out = (self.w_X - size) / stride + 1

        if not self.h_out.is_integer() or not self.w_out.is_integer():
            raise Exception("Invalid dimensions!")

        self.h_out, self.w_out = int(self.h_out), int(self.w_out)

    def forward(self):
        super().forward()
        X = self.p[0].val
        self.n_X = X.shape[0]
        X_reshaped = X.reshape(X.shape[0] * X.shape[1], 1, X.shape[2],
                               X.shape[3])

        self.X_col = im2col_cython(
            X_reshaped, self.size, self.size, padding=0, stride=self.stride)

        self.max_indexes = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_indexes, range(self.max_indexes.size)]

        out = out.reshape(self.h_out, self.w_out, self.n_X,
                          self.d_X).transpose(2, 3, 0, 1)
        self.val = out

    def backward(self, d):

        dX_col = np.zeros_like(self.X_col)
        dout_flat = d.transpose(2, 3, 0, 1).ravel()

        dX_col[self.max_indexes, range(self.max_indexes.size)] = dout_flat

        # get the original X_reshaped structure from col2im
        shape = (self.n_X * self.d_X, 1, self.h_X, self.w_X)
        dx = col2im_cython(
            dX_col, shape[0], shape[1], shape[2], shape[3], self.size, self.size, padding=0, stride=self.stride)
        dx = dx.reshape(self.n_X, self.d_X, self.h_X, self.w_X)

        super().backward([dx])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2018-05-14 18:48
import numpy as np
from tensor_lite.tensor import *


class Regularizer:
    def __init__(self):
        pass

    def initialize(self, loss):
        # traverse tensor to get all Ws
        self.loss = loss
        self.weights = []

        def collect_weights(tensor):
            if type(tensor) is Weight:
                self.weights.append(tensor)

        self.loss.traverse(collect_weights)

    def regularize(self):
        pass


class L2Regularizer(Regularizer):
    def __init__(self, factor):
        self.factor = factor

    def regularize(self):
        for w in self.weights:
            w.d += self.factor * w.val
            self.loss.val += self.factor * np.sum(np.square(w.val)) / 2.

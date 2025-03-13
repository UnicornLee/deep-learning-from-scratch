# coding: utf-8
import numpy as np

from common.layers import Relu

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# mask = (x <= 0)
# print(mask)

relu_layer = Relu()
y = relu_layer.forward(x)
print(y)
dout = np.array([[1, 1], [1, 1]])
dx = relu_layer.backward(dout)
print(dx)

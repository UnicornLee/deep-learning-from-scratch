# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

print(f'1e-50 = {np.float32(1e-50):.50f}')
print(f'1e-4 = {np.float32(1e-4):.10f}')

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

x = np.arange(0.0, 20.0, 0.1) # 以0.1为单位，从0到20的数组x
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2.0

print(numerical_diff(function_tmp1, 3.0))

def function_tmp2(x1):
    return 3.0 ** 2.0 + x1 * x1

print(numerical_diff(function_tmp2, 4.0))

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同且所有元素都为0的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值

    return grad

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0]) ))
print(numerical_gradient(function_2, np.array([3.0, 0.0]) ))
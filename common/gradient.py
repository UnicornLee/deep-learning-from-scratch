# coding: utf-8
import numpy as np

def _numerical_gradient_1d(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1- fxh2) / (2*h)

        x[idx] = tmp_val # 还原值

    return grad

def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)

        return grad

def numerical_gradient(f, x):
    """
    计算函数f在点x处的数值梯度。

    参数:
    f: 目标函数，通常是一个误差或损失函数。
    x: 自变量，可以是一维或多维数组，表示函数f的输入参数。

    返回:
    grad: 与x形状相同的数组，表示f在x处的梯度。
    """
    h = 1e-4 # 0.0001，选择一个小的h值用于计算梯度，平衡精度和数值稳定性
    grad = np.zeros_like(x) # 初始化梯度数组，形状与x相同

    # 使用numpy的nditer迭代x中的每个元素，包括多维情况
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index # 获取当前元素的索引
        tmp_val = x[idx] # 保存当前元素的值

        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)

        # 根据梯度公式计算梯度：(f(x+h) - f(x-h)) / (2h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值

        # 迭代到下一个元素
        it.iternext()

    return grad
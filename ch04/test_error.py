# coding: utf-8
import numpy as np

print("----------均方误差----------")

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 设“2”为正确解
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# 例1：“2”的概率最高的情况（0.6）
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
loss = mean_squared_error(np.array(y), np.array(t))
print(loss)

# 例1：“7”的概率最高的情况（0.6）
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
loss = mean_squared_error(np.array(y), np.array(t))
print(loss)

print("----------交叉熵误差----------")

def cross_entropy_error(y, t):
    print(f'cross_entropy_error: y = {str(y)}, t = {str(t)}')
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
loss = cross_entropy_error(np.array(y), np.array(t))
print(loss)

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
loss = cross_entropy_error(np.array(y), np.array(t))
print(loss)

print("----------mini-batch版交叉熵误差的实现----------")

def cross_entropy_error_batch(y, t, one_hot=True):
    print(f'cross_entropy_error(before reshape): y = {str(y)}, t = {str(t)}')
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    print(f'cross_entropy_error(after reshape): y = {str(y)}, t = {str(t)}')
    batch_size = y.shape[0]
    print(f'cross_entropy_error: batch_size = {str(batch_size)}')

    if one_hot:
        result = -np.sum(t * np.log(y + 1e-7)) / batch_size
    else:
        print(f'{str(y[np.arange(batch_size), t])}')
        result = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return result

print("###########one-hot表示###########")
t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
     [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0],
     [0.6, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
     [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.6],
     [0.1, 0.05, 0.05, 0.0, 0.6, 0.1, 0.0, 0.1, 0.0, 0.0]]
loss = cross_entropy_error_batch(np.array(y), np.array(t))
print(loss)

y = [[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0],
     [0.6, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
     [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.6],
     [0.1, 0.05, 0.05, 0.0, 0.6, 0.1, 0.0, 0.1, 0.0, 0.0],
     [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]]
loss = cross_entropy_error_batch(np.array(y), np.array(t))
print(loss)

print("###########非one-hot表示###########")
t = [2, 7, 0, 9, 4]
y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
     [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0],
     [0.6, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
     [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.6],
     [0.1, 0.05, 0.05, 0.0, 0.6, 0.1, 0.0, 0.1, 0.0, 0.0]]
loss = cross_entropy_error_batch(np.array(y), np.array(t), False)
print(loss)

y = [[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0],
     [0.6, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
     [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.6],
     [0.1, 0.05, 0.05, 0.0, 0.6, 0.1, 0.0, 0.1, 0.0, 0.0],
     [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]]
loss = cross_entropy_error_batch(np.array(y), np.array(t), False)
print(loss)
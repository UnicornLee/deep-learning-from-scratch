# ch08 note

## Data Augmentation（数据扩充）

基于算法“人为地”扩充输入图像（训练图像）

对于输入图像，通过施加旋转、垂直或水平方向上的移动等微小变化，增加图像的数量。这在数据集的图像数量有限时尤其有效。

通过Data Augmentation巧妙地增加训练图像，就可以提高深度学习的识别精度。

## 加深层的动机

### 可以减少网络的参数数量

叠加小型滤波器来加深网络的好处是可以减少参数的数量，扩大感受野（receptive field，给神经元施加变化的某个局部空间区域）。

通过叠加层，将ReLU等激活函数夹在卷积层的中间，进一步提高了网络表现力。这是因为向网络添加了基于激活函数的“非线性”表现力，通过非线性函数的叠加，可以表现更加复杂的东西。

### 使学习更加高效

与没有加深层的网络相比，通过加深层，可以减少学习数据，从而高效地进行学习。

通过加深网络，可以分层次地分解需要学习的问题。因此，各层需要学习的问题就变成了更简单的问题。

通过加深层，可以分层次地传递消息，这一点也很重要。

通过加深层，可以将各层要学习的问题分解成容易解决的简单问题，从而可以进行高效的学习。


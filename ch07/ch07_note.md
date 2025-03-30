# ch07 note

## 卷积层

假设输入大小为 $(H, W)$，滤波器大小为 $(FH, FW)$，输出大小为 $(OH, OW)$，填充为 $P$，步幅为 $S$。输出大小可通过以下公式进行计算：

$$
\begin{align}
OH = \left\lfloor\frac{H + 2P - FH}{S} + 1\right\rfloor, \\
OW = \left\lfloor\frac{W + 2P - FW}{S} + 1\right\rfloor
\end{align}
$$

- 当输出大小无法除尽时（结果是小数时），需要采取报错等对策。
- 根据深度学习的框架的不同，当值无法除尽时，有时会向最接近的整数四舍五入，不进行报错而继续运行。

<p style="color: red; font-weight: bold;">滤波器的通道数只能设定为和输入数据的通道数相同的值</p>

一般来说，池化的窗口大小会和步幅设定成相同的值。

## 池化层

池化层的实现按下面3个阶段进行：
1. 展开输入数据
2. 求各行的最大值
3. 转换为合适的输出大小

最大值的计算可以使用NumPy的`np.max`方法。`np.max`可以指定`axis`参数，并在这个参数指定的各个轴方向上求最大值。

## CNN的可视化

卷积层的滤波器会提取边缘或斑块等原始信息

根据深度学习的可视化相关的研究，随着层次加深，提取的信息（正确地讲，是反应强烈的神经元）也越来越抽象。

## 具有代表性的CNN

### LeNet

和“现在的CNN”相比，LeNet的不同点：
- 激活函数：LeNet中使用sigmoid函数，而现在的CNN中主要使用ReLU函数。
- 原始的LeNet中使用子采样（subsampling）缩小中间数据的大小，而现在的CNN中Max池化是主流。

### AlexNet

AlexNet与LeNet的差异：
- 激活函数使用ReLU
- 使用进行局部正规化的LRN（Local Response Normalization）层
- 使用Dropout

大数据和GPU已成为深度学习发展的巨大的原动力



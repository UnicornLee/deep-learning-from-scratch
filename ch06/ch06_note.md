# ch06 note

## SGD
随机梯度下降法（Stochastic Gradient Descent）
 
$$
\large
\begin{aligned}
\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial L}{\partial \mathbf{W}}
\end{aligned}
$$

这里把需要更新的权重参数记为$ \mathbf{W} $，把损失函数$ L $关于$ \mathbf{W} $的梯度记为$\Large \frac{\partial L}{\partial \mathbf{W}} $。

$ \eta $表示学习率，实际上会取0.01或0.001这些事先决定好的值。

式子中的$ \leftarrow $表示用右边的值更新左边的值。

### SGD的缺点

SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。

## Momentum

$$
\Large
\begin{aligned}
    v & \leftarrow \alpha v - \eta \frac{\partial L}{\partial \mathbf{W}} \\
    \mathbf{W} & \leftarrow \mathbf{W} + v
\end{aligned}
$$

## AdaGrad

**学习率衰减（learning rate decay）**

随着学习的进行，使学习率逐渐减小。

实际上，一开始“多”学，然后逐渐“少”学的方法，在神经网络的学习中经常被使用。

逐渐减小学习率的想法，相当于将“全体”参数的学习率值一起降低。

$$
\Large
\begin{aligned}
    h & \leftarrow h + \frac{\partial L}{\partial \mathbf{W}} \odot \frac{\partial L}{\partial \mathbf{W}} \\
    \mathbf{W} & \leftarrow \mathbf{W} - \eta \frac{1}{\sqrt{h}} \frac{\partial L}{\partial \mathbf{W}} \\
\end{aligned}
$$

## Adam

- 通过组合Momentum和AdaGrad两个方法的有点，有望实现参数空间的高效搜索。
- 进行超参数的“偏执校正”

## 使用哪种更新方法呢？
（目前）并不存在能在所有问题中都表现良好的方法。

这4种方法各有各的特点，都有各自擅长解决的问题和不擅长解决的问题。

一般而言，与SGD相比，其他3种方法可以学习得更快，有时最终的识别精度也更高。

## 权重的初始值

设定什么样的权重初始值，经常关系到神经网络的学习能否成功。

### 可以将权重初始值设为0吗？

**权重衰减（weight decay）**：一种以减小权重参数的值为目的进行学习的方法。

通过减小权重参数的值来抑制过拟合的发生

如果想减小权重的值，一开始就将初始值设为较小的值才是正途。

将权重初始值设为0的话，将无法正确进行学习。

<p style="color: red; font-weight: bold;">为什么不能将权重初始值设为一样的值呢？</p>
因为在误差反向传播时，所有的权重都会进行相同的更新。

权重被更新为相同的值，并拥有了对称的值（重复的值），这使得神经网络拥有许多不同的权重的意义丧失了。

为了防止“权重均一化”（严格地讲，是为了瓦解权重的对称性），必须随机生成初始值。

### 隐藏层的激活值的分布

**各层的激活值的分布都要求有适当的广度。**<span style="color: red; font-weight: bold;">为什么呢？</span>

因为通过在各层传递多样性的数据，神经网络可以进行高效的学习。
反过来，如果传递的是有所偏向的数据，就会出现梯度消失或者“表现力受限”的问题，导致学习可能无法顺利进行。

在一般的深度学习框架中，Xavier初始值已被作为标准使用。

Xavier的论文中，为了使各层的激活值呈现出具有相同广度的分布，推导了合适的权重尺度：
> 如果前一层的节点数为 $ n $，则初始值使用标准差为 $ \Large \sqrt{\frac{1}{n}} $ 的分布。

### ReLU的权重初始值

当激活函数使用ReLU时，一般推荐使用ReLU专用的初始值，也就是Kaiming He等人推荐的初始值，也称为“He初始值”。
> 当前一层的节点数为 $ n $ 时，He初始值使用标准差为 $ \Large \sqrt{\frac{2}{n}} $ 的高斯分布。

> 当激活函数使用ReLU时，权重初始值使用He初始值，当激活函数为sigmoid或tanh等S型曲线函数时，初始值使用Xavier初始值。

在神经网络的学习中，权重初始值非常重要。

很多时候权重初始值的设定关系到神经网络的学习能否成功

权重初始值的重要性容易被忽视，而任何事情的开始（初始值）总是关键的。

## Batch Normalization

优点：
- 可以使学习快速进行（可以增大学习率）
- 不那么依赖初始值（对于初始值不用那么神经质）
- 抑制过拟合（降低Dropout等的必要性）

Batch Norm的思路是调整各层的激活值分布使其拥有适当的广度。

要向神经网络中插入对数据分布进行正规化的层，即Batch Normalization（下文简称Batch Norm层）。

**Batch Norm**：以进行学习时的mini-batch为单位，按mini-batch进行正规化（进行使数据分布的均值为0、方差为1的正规化）。

$$
\Large
\begin{aligned}
\mu_B & \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i \\
\sigma_B^2 & \leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2 \\
\hat{x}_i & \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}
\end{aligned}
$$

$ \mu_B $ 和 $ \sigma_B^2 $ ：mini-batch的m个输入的集合 $ B = \{ x_1, x_2, ..., x_m \} $ 的均值和方差。

$ \varepsilon $：一个微小值，为了防止出现除以0的情况。

接着，Batch Norm层会对正规化后的数据进行缩放和平移的变换：
$$
\Large y_i = \gamma \hat{x}_i + \beta
$$
$ \gamma $ 和 $ \beta $：两个可训练的参数。
一开始 $ \gamma = 1 $，$ \beta = 0 $，然后再通过学习调整到合适的值。
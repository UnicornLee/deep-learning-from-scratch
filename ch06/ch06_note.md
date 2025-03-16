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

## SGD的缺点

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

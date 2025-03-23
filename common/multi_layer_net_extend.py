# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNetExtend:
    """基于扩展版本全耦合的多层神经网络

    具有 Weight Decay、Dropout、Batch Normalization 等功能

    Parameters
    ----------
    input_size : 输入大小（MNIST时为784）
    hidden_size_list : 隐藏层神经元数量列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST时为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 权重的标准偏差的设定（e.g. 0.01）
        如果值为“relu”或“he”时，则设置“He初始值”
        如果值为“sigmoid”或“xavier”时，则设置“Xavier初始值”
    weight_decay_lambda : Weight Decay（L2范式）的强度 -- 衰减强度
    use_dropout : 是否使用Dropout（True or False）
    dropout_ration : Dropout的分配比例（0.2, 0.5）
    use_batchNorm : 是否使用Batch Normalization（True or False）
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout=False, dropout_ration=0.5, use_batchnorm=False):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine%d' % idx] = Affine(self.params['W%d' % idx], self.params['b%d' % idx])
            if self.use_batchnorm:
                self.params['gamma%d' % idx] = np.ones(hidden_size_list[idx-1])
                self.params['beta%d' % idx] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm%d' % idx] = BatchNormalization(self.params['gamma%d' % idx], self.params['beta%d' % idx])

            self.layers['Activation_function%d' % idx] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout%d' % idx] = Dropout(dropout_ration)

        idx = self.hidden_layer_num + 1
        self.layers['Affine%d' % idx] = Affine(self.params['W%d' % idx], self.params['b%d' % idx])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """设置权重的初始值

        Parameters
        ----------
        weight_init_std : 权重的标准偏差的设定（e.g. 0.01）
            如果值为“relu”或“he”时，则设置“He初始值”
            如果值为“sigmoid”或“xavier”时，则设置“Xavier初始值”
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU时建议的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid时推荐的初始值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """求损失函数值

        Parameters
        ----------
        x : 输入数据
        t : 监督数据

        Returns
        ----------
        损失函数的值
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W%d' % idx]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """求梯度（数值微分）

        Parameters
        ----------
        x : 输入数据
        t : 监督数据

        Returns
        ----------
        具有每层梯度的字典变量
            grad['W1']、grad['W2']、... : 各层的权重
            grad['b']、grad['b']、... : 各层的偏置
        """
        loss_W = lambda W : self.loss(x, t, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W%d' % idx] = numerical_gradient(loss_W, self.params['W%d' % idx])
            grads['b%d' % idx] = numerical_gradient(loss_W, self.params['b%d' % idx])

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma%d' % idx] = numerical_gradient(loss_W, self.params['gamma%d' % idx])
                grads['beta%d' % idx] = numerical_gradient(loss_W, self.params['beta%d' % idx])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设置梯度
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W%d' % idx] = self.layers['Affine%d' % idx].dW + self.weight_decay_lambda * self.params['W%d' % idx]
            grads['b%d' % idx] = self.layers['Affine%d' % idx].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma%d' % idx] = self.layers['BatchNorm%d' % idx].dgamma
                grads['beta%d' % idx] = self.layers['BatchNorm%d' % idx].dbeta

        return grads

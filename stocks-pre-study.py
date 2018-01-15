# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:12:01 2017

@author:
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import TensorFlow
import tensorflow as tf

# Import data
data_csv = pd.read_csv('data/data_stocks.csv')

# 获取数据集的尺寸大小, 数据条数:行数, 数据字段数
line_nums = data_csv.shape[0]
columns = data_csv.shape[1]
print(line_nums)
print(columns)

data = data_csv.values

print(type(data))

# #分割数据 生成数据集
# #################################################
# #分割为训练数据Training and 测试数据test data
# #80/20

# #训练数据的起始为位置
train_start = 0
train_end = int(np.floor(0.8 * line_nums))

# #测试数据的起始为位置
test_start = train_end + 1
test_end = line_nums

# #分别申请训练数据 测试数据的向量
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]
##################################################

# #数据标准化处理 缩放输入数据的范围Scale data
# #################################################
# #使用sklearn 的 MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# #使用sklearn 的 MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_train)

# #数据转换
scaler.transform(data_train)
scaler.transform(data_test)

# Build X and y
X_train = data_train[:, 2:]
print('X_train:')
print(X_train)
print(type(X_train))
print(len(X_train))

y_train = data_train[:, 0]
X_test = data_test[:, 2:]
y_test = data_test[:, 0]
# pycharm
# ################################################

# #定义模型 各层 及参数
# #################################################
# #该模型由四个隐藏层组成 第一层包含1024个神经元 后面三层依次以 2的倍数减少即512、256和128个神经元
# Model architecture parameters

# #定义股票的个数
n_stocks = 500

# #定义每层的神经元个数
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1
# #################################################


# #定义占位符 Placeholder
# #################################################
# # X: 神经网络的输入 所有S&P500在时间 T=t的股票价格
# # Y: 神经网络的输出 S&P500在时间 T=t+1的指数值
# None指代我们暂时不知道每个批量传递到神经网络的数量. 表示的是批量大小batch_size
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])
##################################################

# Initializers
sigma = 10
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()

# #定义模型的各层 包括隐藏层 输出层
# #################################################
# #该模型由四个隐藏层组成
# #将每一层的输出作为输入传递给下一层 偏置项的维度等于该层中的神经元数量

# #定义各层的变量, 包括权重矩阵[股票个数,神经元个数] 偏置项向量[神经元个数],
# Layer 1 : Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2 : Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# Layer 3 : Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

# Layer 4 : Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer : Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))
##################################################

# #指定神经网络的网络架构 把占位符(输入数据)和变量(权重和偏置项)组合成一个连续的矩阵乘法系统
# #本次指定为前馈网络或全连接网络,前馈表示输入的批量数据只会从左向右流动
# #################################################
# #该模型由四个隐藏层组成, 每个隐藏层中的每一个神经元还需要有激活函数
# #最常见的就是线性修正单元ReLU激活函数
# #
# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer ( must be transposed )
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
##################################################


# #定义损失函数, 主要是用于生成网络预测与实际观察到的训练目标之间的偏差值
# #对回归问题而言,均方误差MSE函数最为常用
# #################################################
# #定义损失函数为MSE MSE就是计算预测值与目标值之间的平均平方误差
# loss function
loss_mse = tf.reduce_mean(tf.squared_difference(out, Y))
# #################################################

# #定义优化器, 主要是用于训练过程中用于适应网络权重和偏差变量的必要计算
# #深度学习中的默认优化器是Adam
# #一般用梯度下降计算
# #################################################
# Optimizer
opt = tf.train.AdamOptimizer().minimize(loss_mse)
# #################################################

# start session
net = tf.Session()

# Run initializer
net.run(tf.global_variables_initializer())

# Setup interactive plot
plt.ion()

fig = plt.figure()
ax1 = fig.add_subplot(111)

line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)

plt.show()

# Number of epochs and batch size
epochs = 10
batch_size = 256
print('epochs:')

for e in range(epochs):
    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    print(shuffle_indices)
    print(type(shuffle_indices))
    print(len(shuffle_indices))

    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    print('epochs-X_train:')
    print(X_train)
    print(type(X_train))
    print(len(X_train))

    # #使用小批量训练方法
    # #所有样本数据被小批量的送到网络中去,即完成了一个epoch
    # Minibatch training
    for i in range(0, len(y_train)//batch_size):
        start = i * batch_size

        print('batch i:',i)

        # #从训练数据提取数量为batch_size的数据样本
        batch_x = X_train[start: start + batch_size]
        batch_y = y_train[start: start + batch_size]

        print(batch_x)
        print(type(batch_x))
        print(len(batch_x))

        # Run optimizer with batch
        # #把上步提取的数据样本送到网络中 进行优化器的运行

        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        # #每训练 5个batch 就评估下网络预测能力
        if np.mod(i, 5) == 0:
            # Prediction
            # #在测试集(没有被网络学习过的数据)上评估网络的预测能力
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)

            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'

            plt.savefig(file_name)
            plt.pause(0.01)

net.close()

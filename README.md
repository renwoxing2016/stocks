# stocks
股票预测的练习

在开始运行之前:
1.安装pandas
2.创建目录stocks下img
3.download data(sp500.zip),放置到stocks下的data（如果没有则创建此目录）

之后就可以运行了。

有个问题注意；
新版本的tf，用如下方式。
weight_initializer = tf.contrib.layers.variance_scaling_initializer(factor=sigma,mode="FAN_AVG")
旧版本的tf，用下述写法。
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)

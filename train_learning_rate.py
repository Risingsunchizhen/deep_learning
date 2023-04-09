import numpy as np
np.random.seed(1)
from train_model import *
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
learning_rates=[0.002,0.004,0.006,0.008,0.01,0.05,0.1]
optimizers_learning_rate=[]
for learning_rate in learning_rates:
    optimizers_learning_rate.append(SGD(lr=learning_rate))
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,weight_decay_lambda=0.5):
        I, H, O = input_size, hidden_size, output_size

        # 系数权重初始化
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # 隐藏层生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss(weight_decay_lambda=weight_decay_lambda)

        # 梯度保存
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


max_epoch = 10
batch_size = 256
hidden_size = 20
learning_rate = 0.01

mnist = fetch_openml('mnist_784', version=1)
x = mnist['data'].astype('float32').to_numpy()
t= mnist['target'].astype('int32')
enc = OneHotEncoder()
t = enc.fit_transform(t.to_frame()).toarray()
model = TwoLayerNet(input_size=784, hidden_size=hidden_size, output_size=10)
optimizer = SGD(lr=learning_rate)

data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []
loss_list_learning_rate=[]
for optimizer in optimizers_learning_rate:
    loss_list = []
    for epoch in range(max_epoch):

        idx = np.random.permutation(data_size)
        x = x[idx]
        t = t[idx]

        for iters in range(max_iters):
            batch_x = x[iters*batch_size:(iters+1)*batch_size]
            batch_t = t[iters*batch_size:(iters+1)*batch_size]

            # 迭代次数
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss
            loss_count += 1

            # 画图
            if (iters+1) % 10 == 0:
                avg_loss = total_loss / loss_count
                print('| epoch %d |  iter %d / %d | loss %.2f'
                      % (epoch + 1, iters + 1, max_iters, avg_loss))
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0
    loss_list_learning_rate.append(loss_list)
# 训练结果
plt.plot(np.arange(len(loss_list_learning_rate[0])), loss_list_learning_rate[0], label='learning_rate=0.002')
plt.plot(np.arange(len(loss_list_learning_rate[1])), loss_list_learning_rate[1], label='learning_rate=0.004')
plt.plot(np.arange(len(loss_list_learning_rate[2])), loss_list_learning_rate[2], label='learning_rate=0.006')
plt.plot(np.arange(len(loss_list_learning_rate[3])), loss_list_learning_rate[3], label='learning_rate=0.008')
plt.plot(np.arange(len(loss_list_learning_rate[4])), loss_list_learning_rate[4], label='learning_rate=0.01')
plt.plot(np.arange(len(loss_list_learning_rate[5])), loss_list_learning_rate[5], label='learning_rate=0.05')
plt.plot(np.arange(len(loss_list_learning_rate[6])), loss_list_learning_rate[6], label='learning_rate=0.1')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.legend()
plt.show()


######正则化查找
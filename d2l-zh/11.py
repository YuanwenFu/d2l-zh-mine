import torch 
import os 
import sys 
import time 
import math 
import numpy as np 
from torch import nn 
from d2l import torch as d2l 
import pdb 

def func1():
    """
    模型选择
    """
    #猜测MLP中的超参数 
    #dataset:训练集train,测试集test 
    #一般来说，你是无法获取到测试集test的 
    #K则交叉验证 

    return 

def func2():
    """
    过拟合和欠拟合
    """
    #模型容量:拟合各种函数的能力
    #低容量的模型难以拟合训练数据，高容量的模型可以记住所有的训练数据。

    return 

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and len(y_hat.shape) > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_epoch_ch3(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期（定义见第3章）
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)  
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
        else:
            l.backward()
            updater(X.shape[0]) 
        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def func3():
    """
    代码实现 
    """
    #y = 5 + 1.2x - 3.4 / 2! * x^2 + 5.6 / 3! * x^3 + epsilon, epsilon ~ N(0, 0.1^2)
    max_degree = 20 
    n_train, n_test = 100, 100 
    true_w = np.zeros(max_degree) #true_w的shape是(20,)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) 

    features = np.random.normal(size=(n_train + n_test, 1)) #features的shape是(200,1)
    #pdb.set_trace() 
    np.random.shuffle(features) 
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) #poly_features的shape是(200,20)
    #pdb.set_trace()
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1) 
    labels = np.dot(poly_features, true_w) #labels的shape是(200,)
    labels += np.random.normal(0, 0.1, size=labels.shape) #加上噪声 
    #pdb.set_trace()

    true_w, features, poly_features, labels = [
        torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]
    ]
    #true_w的shape是(20)
    #features的shape是(200,1)
    #poly_features的shape是(200,20)
    #labels的shape是(200)
    print(f"a = {features[:2]}, b = {poly_features[:2, :]}, c = {labels[:2]}") 
    #pdb.set_trace() 

    def evaluate_loss(net, data_iter, loss):
        """
        评估给定数据集上模型的损失 
        """
        metric = d2l.Accumulator(2) 
        for X, y in data_iter: 
            out = net(X) 
            y = y.reshape(out.shape) 
            l = loss(out, y) 
            metric.add(l.sum(), l.numel()) 
        return metric[0] / metric[1] 
    
    def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
        loss = nn.MSELoss()
        input_shape = train_features.shape[-1]
        net = nn.Sequential(nn.Linear(input_shape, 1, bias=False)) 
        batch_size = min(10, train_labels.shape[0])
        train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size) 
        test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)), batch_size, is_train=False)
        trainer = torch.optim.SGD(net.parameters(), lr=0.01) 
        animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                                xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                                legend=['train', 'test'])
        for epoch in range(num_epochs):
            train_epoch_ch3(net, train_iter, loss, trainer)
            if epoch == 0 or (epoch + 1) % 20 == 0:
                animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                evaluate_loss(net, test_iter, loss)))
        print(f'weight: {net[0].weight.data.numpy()}')
        print(f"train loss: {evaluate_loss(net, train_iter, loss)}")
        print(f"test loss: {evaluate_loss(net, test_iter, loss)}")
    
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])
    
    return 

def func4():
    """
    QA 
    """
    return 

if __name__ == "__main__":
    print("start...")
    #func1()
    #func2()
    #func3()
    func4()
    print("end...")
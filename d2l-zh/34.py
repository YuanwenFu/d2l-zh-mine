import os 
import sys 
import time 
import torch 
from torch import nn 
from torch.nn import functional as F
import torchvision 
from d2l import torch as d2l 
import my_d2l 

def func1():
    """
    从零开始 
    """
    scale = 0.01 
    W1 = torch.randn(size=(20, 1, 3, 3)) * scale 
    b1 = torch.zeros(20) 
    W2 = torch.randn(size=(50, 20, 5, 5)) * scale 
    b2 = torch.zeros(50) 
    W3 = torch.randn(size=(800, 128)) * scale 
    b3 = torch.zeros(128) 
    W4 = torch.randn(size=(128, 10)) * scale 
    b4 = torch.zeros(10) 
    params = [W1, b1, W2, b2, W3, b3, W4, b4] 

    def lenet(X, params):
        h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1]) 
        h1_activation = F.relu(h1_conv) 
        h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2)) 
        h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3]) 
        h2_activation = F.relu(h2_conv) 
        h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2)) 
        h2 = h2.reshape(h2.shape[0], -1) 
        h3_linear = torch.matmul(h2, params[4]) + params[5] 
        h3 = F.relu(h3_linear) 
        y_hat = torch.matmul(h3, params[6]) + params[7] 
        return y_hat  

    loss = nn.CrossEntropyLoss(reduction='none') 

    def get_params(params, device):
        new_params = [p.clone().to(device) for p in params] 
        for p in new_params:
            p.requires_grad_() 
        return new_params 
    
    new_params = get_params(params, my_d2l.try_gpu()) 
    print(f"b1 weight: {new_params[1]}, shape: {new_params[1].shape}")  
    print(f"b1 grad: {new_params[1].grad}") 

    def allreduce(data):
        for i in range(1, len(data)):
            data[0][:] += data[i].to(data[0].device) 
        for i in range(1, len(data)):
            data[i] = data[0].to(data[i].device) 
    
    #data = [torch.ones((1, 2), device=d2l.try_gpu()) for _ in range(4)] 
    data = [torch.ones((1, 2), device=my_d2l.try_gpu(i)) * (i + 1) for i in range(3)]
    print(f"data befor allreduce: \n{data}") 
    allreduce(data) 
    print(f"data after allreduce: \n{data}") 

    data = torch.arange(20).reshape(4, 5) 
    devices = [torch.device("cuda:0"), torch.device("cuda:0")] 
    split = nn.parallel.scatter(data, devices) 
    print(f"input: \n{data}") 
    print(f"load into: \n{devices}")  
    print(f"output: \n{split}") 

    def split_batch(X, y, devices):
        """
        将X和y拆分到多个设备上 
        """
        assert X.shape[0] == y.shape[0] 
        return (nn.parallel.scatter(X, devices), nn.parallel.scatter(y, devices)) 
    
    def train_batch(X, y, device_params, devices, lr):
        X_shards, y_shards = split_batch(X, y, devices) 
        ls = [
            loss(lenet(X_shard, device_W), y_shard).sum() 
            for X_shard, y_shard, device_W in zip(X_shards, y_shards, device_params)
        ]
        for l in ls:
            l.backward() 
        with torch.no_grad():
            for i in range(len(device_params[0])):
                allreduce([device_params[c][i].grad for c in range(len(devices))])
        for param in device_params:
            d2l.sgd(param, lr, X.shape[0]) 

    def train(num_gpus, batch_size, lr):
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 
        devices = [my_d2l.try_gpu(i) for i in range(num_gpus)]
        device_params = [get_params(params, d) for d in devices]
        num_epochs = 10
        animator = my_d2l.Animator(
            "epoch",
            "test acc",
            xlim=[1, num_epochs]
        ) 
        timer = d2l.Timer() 
        for epoch in range(num_epochs):
            timer.start() 
            for X, y in train_iter:
                train_batch(X, y, device_params, devices, lr) 
                torch.cuda.synchronize() 
            timer.stop() 
            # 创建包装函数，将lenet和参数绑定
            def lenet_wrapper(X):
                return lenet(X, device_params[0])
            animator.add(
                epoch + 1, 
                (my_d2l.evaluate_accuracy_gpu(lenet_wrapper, test_iter, devices[0]), None)
            )
        print(f"test acc: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch on {str(devices)}")
    
    train(num_gpus=1, batch_size=256, lr=0.2) 
    return 

def func2():
    """
    简洁实现 
    """
    def resnet18(num_classes, in_channels=1):
        """
        稍加修改的ResNet-18模型 
        """
        def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
            blk = [] 
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    blk.append(my_d2l.Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
                else:
                    blk.append(my_d2l.Residual(out_channels, out_channels))
            return nn.Sequential(*blk)
        net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
        net.add_module("resnet_block2", resnet_block(64, 128, 2))
        net.add_module("resnet_block3", resnet_block(128, 256, 2))
        net.add_module("resnet_block4", resnet_block(256, 512, 2))
        net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
        net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
        return net 
    
    def train(net, num_gpus, batch_size, lr):
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 
        devices = [my_d2l.try_gpu(i) for i in range(num_gpus)] 

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                nn.init.normal_(m.weight, std=0.01) 
        
        net.apply(init_weights) 
        net = nn.DataParallel(net, device_ids=devices) #这行是关键。
        trainer = torch.optim.SGD(net.parameters(), lr=lr) 
        loss = nn.CrossEntropyLoss() 
        timer, num_epochs = d2l.Timer(), 10 
        animator = my_d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], 
            legend=['train loss', 'train acc', 'test acc']
        ) 
        for epoch in range(num_epochs):
            net.train() 
            timer.start() 
            metric = my_d2l.Accumulator(3)  # 用于累计 train_loss, train_acc, 样本数
            for X, y in train_iter:
                trainer.zero_grad() 
                X = X.to(devices[0]) 
                y = y.to(devices[0]) 
                y_hat = net(X) 
                l = loss(y_hat, y) 
                trainer.zero_grad() 
                l.backward() 
                trainer.step() 
                metric.add(float(l.sum()), my_d2l.accuracy(y_hat, y), y.numel())
            timer.stop() 
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            test_acc = my_d2l.evaluate_accuracy_gpu(net, test_iter, devices[0])
            animator.add(
                epoch + 1, 
                (train_loss, train_acc, test_acc)
            ) 
        print(f"train loss: {animator.Y[0][-1]:.3f}, train acc: {animator.Y[1][-1]:.3f}, test acc: {animator.Y[2][-1]:.3f}, {timer.avg():.3f} sec/epoch on {str(devices)}") 
        return net 
    
    net = resnet18(10, 1) 
    train(net, num_gpus=1, batch_size=256, lr=0.1) 
    return 

def func3():
    """
    QA 
    """
    #batch_size调小，lr也需要调小 
    
    return 

if __name__ == "__main__":
    print("start...")
    #func1() 
    #func2() 
    func3() 
    print("end...") 
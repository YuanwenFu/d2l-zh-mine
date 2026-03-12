import torch 
from torch import nn 
from torch.nn import functional as F 
import torchvision 
from torch.utils import data 
from torchvision import transforms 
import os 
import sys 
import time 
from d2l import torch as d2l 
from IPython import display 

def accuracy(y_hat, y):
    """
    计算预测正确的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y 
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """
    计算在指定数据集上模型的精度
    """
    if isinstance(net, torch.nn.Module):
        net.eval() #将模型设置为评估模式
    metric = Accumulator(2) #正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:
    """
    在n个变量上累加
    """
    def __init__(self, n):
        self.data = [0.0] * n 
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class Animator:
    """
    在动画中绘制数据
    """
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g--', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
        #增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        #使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts 

    def add(self, x, y):
        """
        向图表中添加多个数据点
        """ 
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n 
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes() 
        display.display(self.fig)
        display.clear_output(wait=True)

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3) 
    for X, y in train_iter:
        y_hat = net(X) 
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            # loss 可能是 per-sample（reduction='none'），需变为标量再 backward
            # 必须用 mean 而非 sum：sum 会使梯度放大 batch_size 倍，导致有效 lr 过大、无法收敛
            l_scalar = l.mean() if l.numel() > 1 else l
            l_scalar.backward()
            updater.step()
            total_loss = l.sum().detach() if l.numel() > 1 else l.detach() * len(y)
            metric.add(float(total_loss), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    训练模型
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics 
    print(f"train loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")

def try_gpu(i=0):
    """
    如果存在，则返回gpu(i)，否则返回cpu()
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}") 
    return torch.device("cpu") 

def try_all_gpus():
    """
    返回所有可用的gpu，如果没有gpu，则返回[cpu(),]
    """
    devices = [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count()) 
    ]
    return devices if devices else [torch.device("cpu")] 

def coor2d(X, K):
    """
    计算二维互相关运算 
    """
    h, w = K.shape 
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) 
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y 

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__() 
        self.weight = nn.Parameter(torch.rand(kernel_size)) 
        self.bias = nn.Parameter(torch.zeros(1)) 
    
    def forward(self, x):
        return coor2d(x, self.weight) + self.bias 
    
def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape) 
    Y = conv2d(X) 
    return Y.reshape(Y.shape[2:]) 

def corr2d_multi_in(X, K):
    return sum(coor2d(x, k) for x, k in zip(X, K)) 

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0) 

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape 
    c_o = K.shape[0] 
    X = X.reshape((c_i, h * w)) 
    K = K.reshape((c_o, c_i)) 
    #全连接层中的矩阵乘法 
    Y = torch.matmul(K, X) 
    return Y.reshape((c_o, h, w)) 

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size 
    Y = torch.zeros((X.shape[0]- p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == "max":
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == "avg":
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean() 
    return Y 

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    使用GPU计算模型在数据集上的精度
    """
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device 
    metric = Accumulator(2) 
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X] 
        else:
            X = X.to(device) 
        y = y.to(device) 
        metric.add(accuracy(net(X), y), y.numel()) 
    return metric[0] / metric[1] 

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """
    用GPU训练模型
    """
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight) 
    net.apply(init_weights) 
    print(f"training on {device}") 
    net.to(device) 
    optimizer = torch.optim.SGD(net.parameters(), lr=lr) 
    loss = nn.CrossEntropyLoss() 
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc']) 
    timer, num_batches = d2l.Timer(), len(train_iter) 
    for epoch in range(num_epochs):
        #训练损失之和，训练准确率之和，样本数 
        metric = Accumulator(3) 
        net.train() 
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad() 
            X, y = X.to(device), y.to(device) 
            y_hat = net(X) #正向传播 
            l = loss(y_hat, y) #计算Loss 
            l.backward() #反向传播 
            optimizer.step() #更新参数 
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0]) 
            timer.stop() 
            train_l = metric[0] / metric[2] 
            train_acc = metric[1] / metric[2] 
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None)) 
        test_acc = evaluate_accuracy_gpu(net, test_iter) 
        animator.add(epoch + 1, (None, None, test_acc)) 
    print(f"loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}") 
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}") 

def load_data_fashion_mnist(batch_size, resize=None):
    """
    下载Fashion-MNIST数据集，然后将其加载到内存中
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=get_dataloader_workers()))

def get_dataloader_workers():
    """
    使用4个进程来读取数据
    """
    return 4 

def wgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)) 
        layers.append(nn.ReLU()) 
        in_channels = out_channels 
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) 
    return nn.Sequential(*layers) 

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding), 
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), 
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), 
        nn.ReLU()
    )

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs) 
        #线路1，单1*1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1) 
        #线路2,1*1卷积层后接3*3卷积层 
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1) 
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1) 
        #线路3,1*1卷积层后接5*5卷积层 
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1) 
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2) 
        #线路4，3*3最大汇聚层后接1*1卷积层 
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) 
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1) 
    
    def forward(self, x):
        p1 = F.relu(self.p1_1(x)) 
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x)))) 
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x)))) 
        p4 = F.relu(self.p4_2(self.p4_1(x))) 
        #在通道维度上连结输出 
        return torch.cat((p1, p2, p3, p4), dim=1) 

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    #通过is_grad_enabled来判断当前模式是训练模式还是预测模式 
    if not torch.is_grad_enabled():
        #如果是在预测模式下，直接使用传入的移动平均所得的均值和方差 
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps) 
    else:
        assert len(X.shape) in (2, 4) 
        if len(X.shape) == 2:
            mean = X.mean(dim=0) 
            var = ((X - mean) ** 2).mean(dim=0) 
        else:
            #使用二维卷积层的情况，计算通道维上(axis=1)的均值和方差 
            #这里我们需要保持X的形状以便后面可以做广播运算。
            mean = X.mean(dim=(0, 2, 3), keepdim=True) 
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True) 
        #训练模式下，用当前的均值和方差做标准化 
        X_hat = (X - mean) / torch.sqrt(var + eps) 
        #更新移动平均的均值和方差 
        moving_mean = momentum * moving_mean + (1 - momentum) * mean 
        moving_var = momentum * moving_var + (1 - momentum) * var 
    Y = gamma * X_hat + beta #缩放和移位 
    return Y, moving_mean.data, moving_var.data 

class BatchNorm(nn.Module):
    #num_features: 完全连接层的输出数量或者卷积层的输出通道数 
    #num_dims: 2表示完全连接层，4表示卷积层 
    def __init__(self, num_features, num_dims):
        super().__init__() 
        if num_dims == 2:
            shape = (1, num_features) 
        else:
            shape = (1, num_features, 1, 1) 
        #参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0 
        self.gamma = nn.Parameter(torch.ones(shape)) #待学习的参数
        self.beta = nn.Parameter(torch.zeros(shape)) #待学习的参数 
        #非模型参数的变量初始化为0和1 
        self.moving_mean = torch.zeros(shape) 
        self.moving_var = torch.ones(shape) 
    
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在的显存上 
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device) 
            self.moving_var = self.moving_var.to(X.device) 
        #保存更新过的moving_mean和moving_var 
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, 
            self.gamma, self.beta, 
            self.moving_mean, self.moving_var, 
            eps=1e-5, momentum=0.9
        )
        return Y

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides) 
        else:
            self.conv3 = None 
        self.bn1 = nn.BatchNorm2d(num_channels) 
        self.bn2 = nn.BatchNorm2d(num_channels) 
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X))) 
        Y = self.bn2(self.conv2(Y)) 
        if self.conv3:
            X = self.conv3(X) 
        Y += X 
        return F.relu(Y) 

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2)) 
        else:
            blk.append(Residual(num_channels, num_channels)) 
    return blk 

def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X) 
    l = loss(pred, y) 
    l.sum().backward() 
    trainer.step() 
    train_loss_sum = l.sum() 
    train_acc_sum = accuracy(pred, y) 
    return train_loss_sum, train_acc_sum 

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                devices=try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter) 
    animator = Animator(
        xlabel='epoch', 
        xlim=[1, num_epochs],
        ylim=[0, 1],
        legend=['train loss', 'train acc', 'test acc']
    )
    net = nn.DataParallel(net, device_ids=devices).to(devices[0]) 
    for epoch in range(num_epochs):
        metric = Accumulator(4) 
        for i, (features, labels) in enumerate(train_iter):
            timer.start() 
            l, acc = train_batch_ch13(
                net, 
                features,
                labels,
                loss,
                trainer,
                devices 
            )
            metric.add(l, acc, labels.shape[0], labels.numel()) 
            timer.stop() 
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(
                    epoch + (i + 1) / num_batches, 
                    (metric[0] / metric[2], metric[1] / metric[3], None)
                )
        test_acc = evaluate_accuracy_gpu(net, test_iter) 
        animator.add(epoch + 1, (None, None, test_acc)) 
    print(f"loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}")
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}") 
    
def resnet18(num_classes, in_channels=1):
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2)) 
            else:
                blk.append(Residual(out_channels, out_channels)) 
        return nn.Sequential(*blk)
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), 
    )
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True)) 
    net.add_module("resnet_block2", resnet_block(64, 128, 2)) 
    net.add_module("resnet_block3", resnet_block(128, 256, 2)) 
    net.add_module("resnet_block4", resnet_block(256, 512, 2)) 
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1))) 
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))) 
    return net 

def box_corner_to_center(boxes):
    """
    从（左下，右下）转换到（中间，宽度，高度）
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] 
    cx = (x1 + x2) / 2 
    cy = (y1 + y2) / 2 
    w = x2 - x1 
    h = y2 - y1 
    boxes = torch.stack((cx, cy, w, h), axis=-1) 
    return boxes 

def box_center_to_corner(boxes):
    """
    从（中间，宽度，高度）转换到（左上，右下）
    """ 
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] 
    x1 = cx - 0.5 * w 
    y1 = cy - 0.5 * h 
    x2 = cx + 0.5 * w 
    y2 = cy + 0.5 * h 
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes 

def box_to_rect(bbox, color):
    """
    将边界框（左上x，左上y，右下x，右下y）格式转换成matplotlib格式 
    (x1, y1, x2, y2) —> (x1, y1, width, height)
    """
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )



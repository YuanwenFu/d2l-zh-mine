import os 
import time 
import sys 
import torch 
from torch import nn 
from torch.nn import functional as F 
import pdb 

def func1():
    """
    模型构造
    """
    net = nn.Sequential(
        nn.Linear(20, 256), 
        nn.ReLU(), 
        nn.Linear(256, 10)
        )
    X = torch.rand(2, 20) 
    print(net(X))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(20, 256) 
            self.out = nn.Linear(256, 10) 
        
        def forward(self, X):
            return self.out(F.relu(self.hidden(X))) 
    
    net = MLP()
    print(net(X)) 

    class MySequential(nn.Module):
        def __init__(self, *args):
            super().__init__()
            for block in args:
                self._modules[block] = block 
        
        def forward(self, X):
            for block in self._modules.values():
                X = block(X) 
            return X 
    
    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10)) 
    print(net(X)) 

    #自定义的module 
    class FixedHiddenMLP(nn.Module):
        def __init__(self):
            super().__init__() 
            self.rand_weight = torch.rand((20, 20), requires_grad=False) 
            self.linear = nn.Linear(20, 20) 
        
        def forward(self, X):
            X = self.linear(X)
            X = F.relu(torch.mm(X, self.rand_weight) + 1) 
            X = self.linear(X) 
            while X.abs().sum() > 1:
                X /= 2 
            return X.sum() 

    net = FixedHiddenMLP() 
    print(net(X)) 

    class NestMLP(nn.Module):
        def __init__(self):
            super().__init__() 
            self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), 
                                     nn.Linear(64, 32), nn.ReLU()) 
            self.linear = nn.Linear(32, 16) 
        
        def forward(self, X):
            return self.linear(self.net(X)) 
    
    chimera = nn.Sequential(
        NestMLP(),
        nn.Linear(16, 20),
        FixedHiddenMLP()
    )
    print(f"chimera(X) = {chimera(X)}") 

    return 

def func2():
    """
    参数管理 
    """
    net = nn.Sequential(
        nn.Linear(4, 8), 
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    X = torch.rand(size=(2, 4)) 
    print(f"net(X) = {net(X)}")
    print(f"net[2].state_dict() = {net[2].state_dict()}")
    print(f"net[2].bias = {net[2].bias}")
    print(f"net[2].bias.data = {net[2].bias.data}")
    print(f"net[2].bias.grad = {net[2].bias.grad}")

    #一次性访问所有参数 
    for name, param in net.named_parameters():
        #pdb.set_trace()
        name 
        param.data 
        param.grad 
        param.shape 
        print(f"name = {name}, param.data = {param.data}!") 
    print(net.state_dict()['2.bias'].data) 

    def block1():
        return nn.Sequential(
            nn.Linear(4, 8), 
            nn.ReLU(),
            nn.Linear(8, 4), 
            nn.ReLU()
        )
    
    def block2():
        net = nn.Sequential()
        for i in range(4):
            net.add_module(f"block {i}", block1()) 
        return net 
    
    rgnet = nn.Sequential(
        block2(),
        nn.Linear(4, 1)
    )
    print(f"rgnet(X) = {rgnet(X)}") 
    print(rgnet) 

    #修改默认的初始化函数 
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0, std=0.01) 
            nn.init.zeros_(m.bias) 
    
    net.apply(init_normal)
    print(f"net[0].weight.data[0] = {net[0].weight.data[0]}")
    print(f"net[0].bias.data[0] = {net[0].bias.data}")
    #pdb.set_trace()
    
    def init_constant(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 1)
            nn.init.zeros_(m.bias) 
    
    net.apply(init_constant) 
    print(f"constant net[0].weight.data = {net[0].weight.data[0]}")
    print(f"constant net[0].bias.data = {net[0].bias.data}")
    #pdb.set_trace()
    #不能把weight初始化成常数
    
    def xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight) 
    
    def init_42(m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 42) 
    
    net[0].apply(xavier) 
    net[2].apply(init_42) 
    print(f"net[0].weight.data[0] = {net[0].weight.data[0]}") 
    print(f"net[2].weight.data = {net[2].weight.data}") 

    def my_init(m):
        if type(m) == nn.Linear:
            for name, param in m.named_parameters():
                print(f"name = {name}, param.data = {param.data}") 
            nn.init.uniform_(m.weight, -10, 10) 
            m.weight.data *= m.weight.data.abs() >= 5 
    
    net.apply(my_init) 
    print(f"net[0].weight[:2] = {net[0].weight[:2]}") 

    net[0].weight.data[:] += 1 
    net[0].weight.data[0, 0] = 42 
    print(f"net[0].weight.data = {net[0].weight.data}") 

    shared = nn.Linear(8, 8) 
    net = nn.Sequential(
        nn.Linear(4, 8), 
        nn.ReLU(),
        shared,
        nn.ReLU(),
        shared,
        nn.ReLU(),
        nn.Linear(8, 1)
    ) 
    print(f"net(X) = {net(X)}") 
    print(net[2].weight.data[0] == net[4].weight.data[0]) 
    net[2].weight.data[0, 0] = 100
    print(net[2].weight.data[0] == net[4].weight.data[0]) 

    return 

def func3():
    """
    自定义层 
    """
    class CenteredLayer(nn.Module):
        def __init__(self):
            super().__init__() 
        
        def forward(self, X):
            return X - X.mean() 
    
    layer = CenteredLayer() 
    out = layer(torch.FloatTensor([1, 2, 3, 4, 5])) 
    print(f"out = {out}") 

    net = nn.Sequential(
        nn.Linear(8, 128),
        CenteredLayer()
    ) 
    Y = net(torch.rand(4, 8)) 
    print(Y.mean()) 

    class MyLinear(nn.Module):
        def __init__(self, in_units, units):
            super().__init__() 
            self.weight = nn.Parameter(torch.randn(in_units, units)) 
            self.bias = nn.Parameter(torch.randn(units)) 
        
        def forward(self, X):
            linear = torch.matmul(X, self.weight.data) + self.bias.data 
            return F.relu(linear) 
    dense = MyLinear(5, 3) 
    print("*"*50)
    print(dense.weight) 
    print(dense)

    print(dense(torch.rand(2, 5))) 
    net = nn.Sequential(
        MyLinear(64, 8),
        MyLinear(8, 1)
    )
    print(net) 
    print(net(torch.rand(2, 64))) 

    return 

def func4():
    """
    读写文件 
    """
    x = torch.arange(4) 
    torch.save(x, "x-file")

    x2 = torch.load("x-file")
    print(x2) 

    y = torch.zeros(4) 
    torch.save([x, y], "x-file") 
    x2, y2 = torch.load("x-file") 
    print(f"x2 = {x2}, y2 = {y2}") 

    mydict = {"x": x, "y": y} 
    torch.save(mydict, "mydict") 
    mydict2 = torch.load("mydict")
    print(mydict2) 

    #模型的存储
    class MLP(nn.Module):
        def __init__(self):
            super().__init__() 
            self.hidden1 = nn.Linear(20, 256) 
            self.output = nn.Linear(256, 10) 
        
        def forward(self, X):
            return self.output(F.relu(self.hidden1(X))) 
    
    net = MLP()
    X = torch.randn(2, 20) 
    Y = net(X) 
    torch.save(net.state_dict(), "mlp.params")
    for name, param in net.named_parameters():
        pass 
        #pdb.set_trace()  

    clone = MLP() 
    clone.load_state_dict(torch.load("mlp.params")) 
    print(clone.eval())

    Y_clone = clone(X) 
    print(Y_clone == Y) 

    return 

def func5():
    """
    QA 
    """
    #nn.Module类实现了__call__()方法，net(X)等价于调用net.forward(X) 
    return 

if __name__ == "__main__":
    print("start...")
    #func1() 
    #func2()
    #func3()
    #func4() 
    func5()  
    print("end...") 
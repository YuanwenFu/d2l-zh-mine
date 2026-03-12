import sys
import torch
import os
import time

def func1():
    """
    自动求导
    """

    return

def func2():
    """
    自动求导实现
    """
    x = torch.arange(4.0)
    print(f"x = {x}")

    x.requires_grad_(True) #等价于x = torch.arange(4.0, requires_grad=True)
    print(f"x.grad = {x.grad}")

    y = 2 * torch.dot(x, x)
    print(f"y = {y}")
    print(f"y.grad_fn = {y.grad_fn}")

    y.backward()
    #访问我的导数
    print(f"x.grad = {x.grad}")
    print(f"x.grad == 4 * x = {x.grad == 4 * x}")

    #在默认情况下,pytorch会累积梯度，我们需要清楚之前的值
    x.grad.zero_()
    y = x.sum()
    y.backward()
    print(f"x.grad = {x.grad}")

    #如果y不是标量，
    y = x * x 
    print(f"y = {y}")
    x.grad.zero_()
    z = y.sum()
    z.backward()
    #y.sum().backward()
    print(f"x.grad = {x.grad}")

    x.grad.zero_()
    y = x * x 
    u = y.detach() 
    z = u * x 
    z.sum().backward()
    print(f"x.grad = {x.grad}, \nu = {u}")

    x.grad.zero_()
    y.sum().backward()
    print(f"x.grad = {x.grad}, \n 2 * x = {2 * x}")

    def f(a):
        b = a * 2
        while b.norm() < 1000:
            b = b * 2
        if b.sum() > 0:
            c = b 
        else:
            c = 100 * b 
        return c 
    
    a = torch.randn(size=(), requires_grad=True)
    d = f(a)
    print(f"a = {a}, d = {d}")
    d.backward()
    print(f"a.grad = {a.grad}, \n d / a = {d / a}")

    return


if __name__ == "__main__":
    print("start...")

    #func1()
    func2()

    print("end...")
#线性代数 
import torch 
import os
import sys

def func1():
    """
    线性代数实现
    """
    x = torch.arange(4)
    print(f"x = {x}")
    print(f"x[3] = {x[3]}")
    print(len(x))
    print(x.shape)

    A = torch.arange(20).reshape(5,4)
    print(f"A = {A}")
    print(A.T)

    A_numpy = A.numpy()
    A_numpy_tensor = torch.tensor(A_numpy)
    print(f"{type(A_numpy)}, {type(A_numpy_tensor)}")

    X = torch.arange(24).reshape(2,3,4)
    print(f"X = {X}")

    A = torch.arange(20, dtype=torch.float32).reshape(5,4)
    B = A.clone() #通过分配新内存，将A的副本分配给B
    print(f"A = {A}\n, A + B = {A + B}")
    print(f"A * B = {A * B}") #按元素做乘法，并非矩阵乘法

    a = 2
    X = torch.arange(24).reshape(2,3,4)
    print(f"X = {X}")
    print(f"a + X = {a + X}")
    print(f"a * X = {a * X}, (a * X).shape = {(a * X).shape}")

    x = torch.arange(4, dtype=torch.float32)
    print(f"x = {x}, x.sum() = {x.sum()}")

    A = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    #print(f"A.shape = {A.shape}, A.sum() = {A.sum()}")
    print(f"A = \n{A}")
    #print(f"A.sum(axis=0) = {A.sum(axis=0)}, A.sum(axis=0).shape = {A.sum(axis=0).shape}")
    #print(f"A.sum(axis=1) = {A.sum(axis=1)}, A.sum(axis=1).shape = {A.sum(axis=1).shape}")
    #print(f"A.sum(axis=2) = {A.sum(axis=2)}, A.sum(axis=2).shape = {A.sum(axis=2).shape}")
    print(f"A.sum(axis=[0,1]) = {A.sum(axis=[0,1])}, A.sum(axis=[0,1]).shape = {A.sum(axis=[0,1]).shape}")

    #求均值
    print(f"A = \n{A}")
    #print(f"A.mean() = \n{A.mean()}")
    #print(f"A.sum() / A.numel() = {A.sum() / A.numel()}")
    #print(f"A.mean(axis=0) = \n{A.mean(axis=0)}")
    print(f"A.mean(axis=1) = \n{A.mean(axis=1)}")
    #print(f"A.mean(axis=2) = \n{A.mean(axis=2)}")

    A = torch.arange(12,dtype=torch.float32).reshape(3,4)
    sum_A = A.sum(axis=1, keepdim=True) #保持维数相等
    print(f"sum_A = \n{sum_A}")
    print(f"A / sum_A = \n{A / sum_A}")

    print(f"A = \n{A}")
    print(f"A.cumsum(axis=0) = \n{A.cumsum(axis=0)}")

    x = torch.arange(4, dtype=torch.float32)
    y = torch.ones(4, dtype=torch.float32)
    print(f"x = {x}, y = {y}, torch.dot(x,y) = {torch.dot(x,y)}")

    A = torch.arange(12,dtype=torch.float32).reshape(3, 4)
    x = torch.arange(4,dtype=torch.float32)
    y = torch.mv(A, x) #矩阵*向量
    print(f"A = \n{A}\n, x = \n{x}\n, y = \n{y}\n")

    B = torch.ones(4, 3)
    C = torch.mm(A, B) #矩阵乘以矩阵
    print(f"A = \n{A}\n, B = \n{B}\n, C = \n{C}\n")

    #向量的L2范数:所有元素的平方和再开根号
    u = torch.tensor([3.0, -4.0])
    print(torch.norm(u))

    #向量的L1范数:所有元素绝对值再求和
    print(torch.abs(u).sum())
    print(torch.norm(u, p=1))
    print(torch.norm(u, p=2))

    #矩阵的F范数:所有元素的平方和再开根号
    A = torch.ones(4, 9)
    print(torch.norm(A))

    return

def func2():
    """
    按照特定轴求和
    """
    a = torch.ones(2, 5, 4)
    print(a.shape)
    print(a.sum().shape) #shape为空,表示它是一个标量
    print(a.sum(axis=1, keepdim=True).shape)

    return

def func3():
    """
    线性代数答疑
    """
    a = torch.tensor([1, 3, 4], dtype=torch.float32)
    print(a)

    a_numpy = a.numpy()
    a_numpy_tensor = torch.tensor(a_numpy)
    print(f"a_numpy = {a_numpy}, type(a_numpy) = {type(a_numpy)}")
    print(f"a_numpy_tensor = {a_numpy_tensor}, type(a_numpy_tensor) = {type(a_numpy_tensor)}")
    
    return

if __name__ == "__main__":
    #func1()
    #func2()
    func3()
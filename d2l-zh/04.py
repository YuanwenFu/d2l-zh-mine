import torch 
import numpy 
import os
import pdb
import pandas as pd


def data_operate():
    x = torch.arange(12)
    print(x)
    print(x.shape)
    print(x.numel())

    X = x.reshape(3, 4)
    print(X)

    print(torch.zeros((2, 3, 4)))

    print(torch.ones((2, 3, 4)))

    a = torch.tensor([[[1,2,3],[4,5,6]]])
    print(a)
    print(a.shape)

    a = torch.tensor([1.0, 2, 4, 8])
    b = torch.tensor([2, 2, 2, 2])
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b) #对应元素相除
    print(a ** b)
    print(torch.exp(a))

    a = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    b = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(torch.cat((a, b), dim=0))
    print(torch.cat((a, b), dim=1))
    print(a.sum())

    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print(a)
    print(b)

    print(a + b)

    a = torch.arange(12).reshape(3, 4)
    print(a)
    print(a[-1])
    print(a[1:3])
    a[1, 2] = 9
    print(a)
    a[0:2,:] = 12
    print(a)

    a = torch.arange(12).reshape(3,4)
    print(a)
    id1 = id(a)
    print(id1)
    a[0:2,:] = 12
    id2 = id(a)
    print(id2)
    a = a + torch.ones(12).reshape(3,4)
    id3 = id(a)
    print(id3)
    print(a)

    c = torch.zeros_like(a)
    print(f"id(c) = {id(c)}")
    c[:] = c + torch.arange(12).reshape(3,4) #原地操作
    print(f"id(c) = {id(c)}")

    c = torch.zeros_like(a)
    print(f"id(c) = {id(c)}")
    c += torch.arange(12).reshape(3, 4) #原地操作
    print(f"id(c) = {id(c)}")

    X = torch.arange(12).reshape(3, 4)
    print(f"X = {X}, type(X) = {type(X)}")
    A = X.numpy()
    print(f"A = {A}, type(A) = {type(A)}")
    B = torch.tensor(A)
    print(f"B = {B}, type(B) = {type(B)}")

    a = torch.tensor([3.5])
    print(f"a = {a}, a.item() = {a.item()}, float(a) = {float(a)}, int(a) = {int(a)}")
    print(f"type(a.item()) = {type(a.item())}")
    return 

def data_preprocess():
    os.makedirs(os.path.join('.', 'data'), exist_ok=True)
    data_file = os.path.join('.', 'data', 'house_tiny.csv')
    print(data_file)
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')
        f.write('NA,Pave,127500\n')
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    
    #读取数据
    data = pd.read_csv(data_file)
    print(f"data = {data}, type(data) = {type(data)}")

    inputs, outputs = data.iloc[:, 0:2], data.iloc[:,2]
    #注意data.iloc[:,0:2]这里是指[0:2]列，总共3列，右边也是闭的。
    # 只对数值列用均值填充，避免对字符串列（如 Alley）求 mean 报错
    numeric_cols = inputs.select_dtypes(include=[numpy.number]).columns
    inputs[numeric_cols] = inputs[numeric_cols].fillna(inputs[numeric_cols].mean())
    print(f"inputs = {inputs}")

    inputs = pd.get_dummies(inputs, dummy_na=True,dtype=int)
    print(f"inputs = {inputs}")

    X, y = torch.tensor(inputs.values, dtype=torch.float32), torch.tensor(outputs.values)
    print(f"X = {X}, y = {y}")
    print(f"X.shape = {X.shape}, y.shape = {y.shape}")

    return

def data_qa():
    a = torch.arange(12)
    b = a.reshape((3,4)) #两者共用同一个地址，即同一块内容
    b[:] = 2
    print(a) 

    a = torch.arange(12).reshape(3,4)
    print(f"a.shape = {a.shape}, a.ndim = {a.ndim}")

    return

if __name__ == "__main__":
    #data_operate()
    #data_preprocess()
    data_qa()
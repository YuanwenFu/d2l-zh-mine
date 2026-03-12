import os 
import time 
import sys 
import torch 
import torchvision 
from torch import nn 
from PIL import Image
import matplotlib.pyplot as plt
import my_d2l 

def func1():
    """
    数据增广 
    """
    #色调、饱和度、明亮度 
    #数据增强通过数据变形来获取数据多样性。 
    return 

def func2():
    """
    代码
    """
    plt.rcParams['figure.figsize'] = (3.5, 2.5)  # 设置图形大小
    img = Image.open("./cat1.jpg") 
    img.save("cat2.jpg") 

    def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
        Y = [aug(img) for _ in range(num_rows * num_cols)]
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(scale * num_cols, scale * num_rows))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            ax.imshow(Y[i])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("cat2.jpg")
        plt.close()
        return 
    
    #左右翻转图像 
    apply(img, torchvision.transforms.RandomHorizontalFlip())  

    #上下翻转图像 
    apply(img, torchvision.transforms.RandomVerticalFlip())  

    #随机剪裁 
    shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    apply(img, shape_aug) 

    #随机改变图像的亮度 
    apply(img, torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0))

    #随机改变图像的色调 
    apply(img, torchvision.transforms.ColorJitter(hue=0.5)) 

    #随机改变图像的亮度(brightness)、对比度(contrast)、饱和度(saturation)、色调(hue)
    color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    apply(img, color_aug) 
    
    #结合多种图像增广方法 
    augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        color_aug, 
        shape_aug 
    ])
    apply(img, augs) 

    def show_images(imgs, num_rows, num_cols, scale=1.5):
        """
        显示图像网格
        Args:
            imgs: 图像列表
            num_rows: 行数
            num_cols: 列数
            scale: 图像缩放比例
        """
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.show()
        plt.savefig("cifar10.jpg")
        plt.close()
        return axes
    
    #下载cifar10数据集
    #all_images = torchvision.datasets.CIFAR10(root="./data_cifar10", train=True, download=True) 
    #show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8) 

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ])
    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    def load_cifar10(is_train, augs, batch_size):
        dataset = torchvision.datasets.CIFAR10(
            root="./data_cifar10",
            train=is_train,
            transform=augs,
            download=True 
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=4 
        )
        return dataloader 
    
    batch_size, devices, net = 256, my_d2l.try_all_gpus(), my_d2l.resnet18(10, 3)

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) 
    
    net.apply(init_weights) 


    def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
        train_iter = load_cifar10(True, train_augs, batch_size) 
        test_iter = load_cifar10(False, test_augs, batch_size) 
        loss = nn.CrossEntropyLoss(reduction='none') 
        trainer = torch.optim.Adam(net.parameters(), lr=lr) 
        my_d2l.train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices) 

    #train_with_data_aug(train_augs, test_augs, net) #loss 0.255, train acc 0.912, test acc 0.779
    train_with_data_aug(test_augs, test_augs, net) #loss 0.106, train acc 0.963, test acc 0.746. overfitting了

    return 

def func3():
    """
    QA 
    """
    #你的图片多，不代表你的多样性好 
    #增广一般不会改变数据分布 
    #增广的目的之一是使得你的训练集与测试集更加相似 

    return 

if __name__ == "__main__":
    print("start...") 
    #func1()
    #func2()
    func3()
    print("end...") 
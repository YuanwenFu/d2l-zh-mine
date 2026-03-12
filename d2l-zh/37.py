import os 
import sys 
import time 
import torch 
import torchvision 
from torch import nn 
from d2l import torch as d2l 
import my_d2l 

def func1():
    """
    微调
    """
    #微调也称迁移学习 
    #微调时使用更小的学习率和更少的数据迭代 
    #底层特征更加通用，高层特征更加业务。因此可以固定底层网络的参数，不参与更新。 
    return 

def func2():
    """
    代码 
    """
    d2l.DATA_HUB["hotdog"] = (d2l.DATA_URL + "hotdog.zip",
                              "fba480b33e4074c1db93a9132e94192b979aea05")
    data_dir = d2l.download_extract("hotdog")
    train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train")) 
    test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"))  

    hotdogs = [train_imgs[i][0] for i in range(8)] 
    not_hotdogs = [train_imgs[-i-1][0] for i in range(8)] 
    d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
    d2l.plt.savefig("hotdog_samples.png", dpi=150, bbox_inches='tight')
    d2l.plt.close()
    print("图像已保存为: hotdog_samples.png")

    #使用RGB通道的均值和标准差，以标准化每个通道 
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224), #224x224是imagenet的输入图像大小 
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    
    pretrained_net = torchvision.models.resnet18(pretrained=True) 
    print(pretrained_net.fc) 
    finetune_net = torchvision.models.resnet18(pretrained=True) 
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2) 
    nn.init.xavier_uniform_(finetune_net.fc.weight) 

    def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, 
                          param_group=True): 
        train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=train_augs),
            batch_size=batch_size, shuffle=True
        )
        test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "test"), transform=test_augs),
            batch_size=batch_size
        )
        devices = my_d2l.try_all_gpus() 
        loss = nn.CrossEntropyLoss(reduction="none") 
        if param_group:
            params_1x = [param for name, param in net.named_parameters()
                if name not in ["fc.weight", "fc.bias"]]
            trainer = torch.optim.SGD([{"params": params_1x},
                                       {"params": net.fc.parameters(), 
                                       "lr": learning_rate * 10}],
                                    lr=learning_rate, weight_decay=0.001)
        else:
            trainer = torch.optim.SGD(net.parameters(),
                                      lr=learning_rate, 
                                      weight_decay=0.001)
        my_d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices) 
    if False:
        train_fine_tuning(finetune_net, 5e-5) 
        #loss 0.157, train acc 0.943, test acc 0.929 
    else:
        scratch_net = torchvision.models.resnet18() 
        scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2) 
        train_fine_tuning(scratch_net, 5e-4, param_group=False) 
        #由于是从头开始训练，因此这个学习率还可以调大一些 
        #loss 0.375, train acc 0.837, test acc 0.761 
    
    return 

def func3():
    """
    QA 
    """
    #微调的话，源数据集需要和目标数据集比较类似。 
    #微调对学习率不敏感 
    return 

if __name__ == "__main__":
    print("start...")
    #func1() 
    #func2() 
    func3() 
    print("end...") 
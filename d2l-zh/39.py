import os 
import time 
import sys 
import torch 
import torchvision 
from torch import nn 
import collections 
import math 
import pandas as pd 
import shutil 
from d2l import torch as d2l 
import my_d2l 

def func1():
    """
    kaggle cifar10 
    """
    #比赛的网址是https://www.kaggle.com/c/cifar-10/data 
    d2l.DATA_HUB["cifar10_tiny"] = (d2l.DATA_URL + "kaggle_cifar10_tiny.zip",
                              "2068874e4b9a9f0fb07ebe0ad2b29754449ccacd")
    #如果使用完整的kaggle竞赛的数据集，设置demo为False 
    demo = True 
    if demo:
        data_dir = d2l.download_extract("cifar10_tiny")
    else:
        data_dir = "../data/cifar-10/" 
    
    def read_csv_labels(fname):
        with open(fname, "r") as f:
            lines = f.readlines()[1:] 
        tokens = [l.rstrip().split(",") for l in lines] 
        return dict(((name, label) for name, label in tokens)) 
    
    labels = read_csv_labels(os.path.join(data_dir, "trainLabels.csv")) 
    print(f"train sample {len(labels)}") 
    print(f"labels {len(set(labels.values()))}") 
    #print(f"labels {labels}") 

    def copyfile(filename, target_dir):
        os.makedirs(target_dir, exist_ok=True) 
        shutil.copy(filename, target_dir) 
    
    def reorg_train_valid(data_dir, labels, valid_ratio):
        n = collections.Counter(labels.values()).most_common()[-1][1]
        n_valid_per_label = max(1, math.floor(n * valid_ratio)) 
        label_count = {} 
        for train_file in os.listdir(os.path.join(data_dir, "train")): 
            label = labels[train_file.split(".")[0]] 
            fname = os.path.join(data_dir, "train", train_file) 
            copyfile(fname, os.path.join(data_dir, "train_valid_test", "train_valid", label)) 
            if label not in label_count or label_count[label] < n_valid_per_label:
                copyfile(fname, os.path.join(data_dir, "train_valid_test", "valid", label)) 
                label_count[label] = label_count.get(label, 0) + 1 
            else:
                copyfile(fname, os.path.join(data_dir, "train_valid_test", "train", label)) 
        return n_valid_per_label 
    
    def reorg_test(data_dir):
        for test_file in os.listdir(os.path.join(data_dir, "test")):
            copyfile(
                os.path.join(data_dir, "test", test_file), 
                os.path.join(data_dir, "train_valid_test", 
                "test", 
                "unknown")
            ) 

    def reorg_cifar10_data(data_dir, valid_ratio):
        labels = read_csv_labels(os.path.join(data_dir, "trainLabels.csv")) 
        reorg_train_valid(data_dir, labels, valid_ratio) 
        reorg_test(data_dir) 
    
    batch_size = 32 if demo else 128 
    valid_ratio = 0.1 
    reorg_cifar10_data(data_dir, valid_ratio) 

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize(40),
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "train_valid_test", folder),
        transform=transform_train
    ) for folder in ["train", "train_valid"]] 

    valid_ds, test_ds = [
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "train_valid_test", folder), 
            transform=transform_test 
        )
        for folder in ["valid", "test"] 
    ]
    
    train_iter, train_valid_iter = [torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, drop_last=True
    ) for dataset in (train_ds, train_valid_ds)] 

    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True) 
    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False) 

    def get_net():
        num_classes = 10 
        net = my_d2l.resnet18(num_classes, 3) 
        return net 
    
    loss = nn.CrossEntropyLoss(reduction="none") 
    def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd) 
        scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay) 
        num_batches, timer = len(train_iter), d2l.Timer() 
        legend = ["train loss", "train acc"] 
        if valid_iter is not None:
            legend.append("valid acc") 
        animator = my_d2l.Animator(xlabel="epoch", xlim=[1, num_epochs], legend=legend) 
        net = nn.DataParallel(net, device_ids=devices).to(devices[0]) 
        for epoch in range(num_epochs):
            net.train()
            metric = my_d2l.Accumulator(3) 
            for i, (features, labels) in enumerate(train_iter):
                timer.start() 
                l, acc = my_d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)
                metric.add(l, acc, labels.shape[0]) 
                timer.stop() 
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches, 
                    (metric[0] / metric[2], metric[1] / metric[2], None)) 
            if valid_iter is not None:
                valid_acc = my_d2l.evaluate_accuracy_gpu(net, valid_iter) 
                animator.add(epoch + 1, (None, None, valid_acc)) 
            scheduler.step() 
        measures = (f"train loss {metric[0] / metric[2]:.3f}"
                    f", train acc {metric[1] / metric[2]:.3f}") 
        if valid_iter is not None:
            measures += f", valid acc {valid_acc:.3f}" 
        print(f"{measures}\n{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}")

    devices, num_epochs, lr, wd = my_d2l.try_all_gpus(), 20, 2e-4, 5e-4 
    lr_period, lr_decay, net = 4, 0.9, get_net() 
    train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay) 
    #train loss 0.654, train acc 0.787, valid acc 0.359 

    #训练并提交结果
    net, preds = get_net(), [] 
    train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay) 
    for X, _ in test_iter:
        y_hat = net(X.to(devices[0])) 
        preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy()) 
    sorted_ids = list(range(1, 1 + len(test_ds))) 
    sorted_ids.sort(key=lambda x: str(x))  
    df = pd.DataFrame({"id": sorted_ids, "label": preds}) 
    df["label"] = df["label"].apply(lambda x : train_valid_ds.classes[x]) 
    df.to_csv("submission.csv", index=False) 

    return 

def func2():
    """
    QA 
    """
    #凸函数的表示能力有限。 
    #weight decay是一个正则项，使得权重不要过大，从而避免过拟合。
    #而lr decay是一个学习率衰减策略，使得在最后阶段，学习率不要过大，从而避免震荡。 

    return 

if __name__ == "__main__":
    print("start...")
    #func1() 
    func2() 
    print("end...") 
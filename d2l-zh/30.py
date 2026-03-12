import os 
import time 
import sys 
import torch 
from torch import nn 
import torchvision 
from torchvision import transforms, datasets 
from torch.utils.data import DataLoader, Dataset
import pandas as pd 
from PIL import Image
import pdb
import my_d2l 

class LeafDataset(Dataset):
    """
    自定义叶子分类数据集
    """
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe: 包含 'image' 列和可选 'label' 列的 DataFrame
            root_dir: 数据集根目录
            transform: 图像变换
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        # 检查是否有标签列（用于区分训练集和测试集）
        self.has_label = 'label' in dataframe.columns
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # 获取图片路径
        img_name = self.dataframe.iloc[idx]['image']
        
        # 构建完整路径
        img_path = os.path.join(self.root_dir, img_name)
        
        # 读取图片
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 如果有标签，返回图片和标签；否则只返回图片
        if self.has_label:
            label = self.dataframe.iloc[idx]['label']
            return image, label
        else:
            return image

def func1():
    """
    图片分类竞赛 
    """
    #叶子分类 
    # https://www.kaggle.com/c/classify-leaves

    #数据准备 
    dir_path = "/home/fyw/d2l-zh/leaf_dataset"
    train_df = pd.read_csv(os.path.join(dir_path, "train.csv")) #shape: (18353, 2)
    test_df = pd.read_csv(os.path.join(dir_path, "test.csv")) #shape: (8800, 1)
    print(train_df.head())
    print(train_df.shape)
    print(test_df.head())
    print(test_df.shape)  

    #构建树叶名字的对应编号
    leaf_names = train_df["label"].unique()
    leaf_name_to_idx = {name: idx for idx, name in enumerate(leaf_names)} #len: 176
    #pdb.set_trace() 
    idx_to_leaf_name = {v: k for k, v in leaf_name_to_idx.items()} 

    print(f"len(leaf_name_to_idx): {len(leaf_name_to_idx)}")
    train_df["label"] = train_df["label"].map(leaf_name_to_idx) 
    print(train_df.head()) 
    print(train_df.shape)

    #从train_df中抽出一部分数据做验证集
    val_df = train_df.sample(frac=0.2, random_state=42)
    train_df = train_df.drop(val_df.index)
    print(train_df.shape) #shape: (14682, 2)
    print(val_df.shape) #shape: (3671, 2)
    #pdb.set_trace()

    #构建验证集数据集
    # val_dataset = LeafDataset(dataframe=val_df, root_dir=dir_path, transform=trans)
    # val_loader = DataLoader(val_dataset, batch_size=36, shuffle=False, num_workers=4)

    if False:
    #读入图片大小
        img_path = os.path.join(dir_path, "images")
        trans = transforms.Compose([transforms.ToTensor()]) 
        # 使用自定义数据集类，而不是 ImageFolder
        train_dataset = LeafDataset(dataframe=train_df, root_dir=dir_path, transform=trans) 
        train_loader = DataLoader(train_dataset, batch_size=36, shuffle=True, num_workers=4)
        for X, y in train_loader:
            #pdb.set_trace() 
            #X。shape: [36, 3, 224, 224]
            #y.shape: [36]
            pass
    
    #构建训练集、验证集和测试集
    batch_size = 36
    # ImageNet的均值和标准差，用于归一化
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    
    # 训练集：添加数据增强
    train_trans = transforms.Compose([
        transforms.Resize(256),  # 先resize到256
        transforms.RandomResizedCrop(224),  # 随机裁剪到224
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        normalize
    ])
    
    # 验证集和测试集：只做resize和归一化，不做数据增强
    val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),
        normalize
    ])
    test_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    train_dataset = LeafDataset(dataframe=train_df, root_dir=dir_path, transform=train_trans)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = LeafDataset(dataframe=val_df, root_dir=dir_path, transform=val_trans)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataset = LeafDataset(dataframe=test_df, root_dir=dir_path, transform=test_trans)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    #模型架构-使用预训练的ResNet-18（迁移学习）
    # 使用预训练模型可以显著提升性能
    # 使用weights参数替代deprecated的pretrained参数
    try:
        pretrained_resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except AttributeError:
        # 兼容旧版本PyTorch
        pretrained_resnet = torchvision.models.resnet18(pretrained=True)
    
    # 冻结前面的层，只训练最后几层（可选，这里我们微调所有层）
    # 如果数据集较小，可以冻结更多层
    # for param in list(pretrained_resnet.parameters())[:-10]:
    #     param.requires_grad = False
    
    # 替换最后的全连接层以适应我们的分类任务
    num_features = pretrained_resnet.fc.in_features
    pretrained_resnet.fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout放在Linear之前
        nn.Linear(num_features, len(leaf_names))
    )
    net = pretrained_resnet
    
    # 如果不想使用预训练模型，可以使用原来的自定义ResNet（注释掉上面的代码，取消下面的注释）
    # b1 = nn.Sequential(
    #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    #     nn.BatchNorm2d(64),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    # )
    # b2 = nn.Sequential(*my_d2l.resnet_block(64, 64, 2, first_block=True)) 
    # b3 = nn.Sequential(*my_d2l.resnet_block(64, 128, 2)) #2表示残差块的个数 
    # b4 = nn.Sequential(*my_d2l.resnet_block(128, 256, 2)) 
    # b5 = nn.Sequential(*my_d2l.resnet_block(256, 512, 2)) 
    # net = nn.Sequential(b1, b2, b3, b4, b5, 
    #                     nn.AdaptiveAvgPool2d((1, 1)),
    #                     nn.Flatten(),
    #                     nn.Dropout(0.5), #增加dropout层，防止过拟合 
    #                     nn.Linear(512, len(leaf_names))) 
    
    # 测试模型输出形状（使用预训练模型时结构不同，这里简化测试）
    X = torch.randn(size=(36, 3, 224, 224))
    with torch.no_grad():
        y_test = net(X)
        print(f'Model output shape: {y_test.shape}')
    #pdb.set_trace()

    #训练模型
    lr, num_epochs = 0.001, 50  # 使用预训练模型时，学习率应该更小
    # 如果使用自定义ResNet，可以使用更大的学习率，如0.01
    
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight) 

    def try_get_gpu(i=0):
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f"cuda:{i}")
        return torch.device("cpu") 

    def accuracy(y_hat, y):
        """
        计算预测正确的数量
        """
        y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y 
        return float(cmp.sum()) / y.numel(), float(cmp.sum()), y.numel()

    def evaluate_accuracy_gpu(net, data_iter, device=None):
        """
        使用GPU计算模型在数据集上的精度
        """
        if isinstance(net, torch.nn.Module):
            net.eval() 
            if not device:
                device = next(iter(net.parameters())).device 
        tot_correct, tot_num = 0.0, 0.0
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X] 
            else:
                X = X.to(device) 
            y = y.to(device) 
            y_hat = net(X) 
            _, tmp_correct, tmp_num = accuracy(y_hat, y)
            tot_correct += float(tmp_correct)
            tot_num += tmp_num
        return tot_correct / tot_num 

    #模型训练
    device = try_get_gpu() 
    # 如果使用预训练模型，不需要重新初始化权重
    # net.apply(init_weights)  # 只对自定义模型使用
    print(f"training on {device}")
    net.to(device)
    
    # 使用Adam优化器，添加权重衰减（L2正则化）
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    # 或者使用SGD with momentum（对预训练模型通常效果更好）
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    # 添加学习率调度器：当验证准确率不再提升时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    loss = nn.CrossEntropyLoss()
    t1 = time.time()
    best_vali_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        net.train()
        t2 = time.time() 
        train_tot_correct, train_tot_num = 0.0, 0.0
        train_loss_sum = 0.0
        num_batches = 0
        
        for X, y in train_loader:
            optimizer.zero_grad() 
            X, y = X.to(device), y.to(device) 
            #pdb.set_trace()
            y_hat = net(X) 
            l = loss(y_hat, y) 
            l.backward() 
            optimizer.step()
            _, tmp_correct, tmp_num = accuracy(y_hat, y)
            train_tot_correct += float(tmp_correct)
            train_tot_num += tmp_num
            train_loss_sum += l.item()
            num_batches += 1
            
        vali_acc = evaluate_accuracy_gpu(net, val_loader)
        train_acc = train_tot_correct / train_tot_num
        avg_train_loss = train_loss_sum / num_batches
        
        # 更新学习率
        scheduler.step(vali_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        if vali_acc > best_vali_acc:
            best_vali_acc = vali_acc
            best_model_state = net.state_dict().copy()
        
        print(f"epoch {epoch + 1}, loss {avg_train_loss:.3f}, train_acc {train_acc:.3f}, "
              f"vali_acc {vali_acc:.3f}, lr {current_lr:.6f}")
        t3 = time.time()
        print(f"time cost: {t3 - t2:.3f}s")
    
    # 加载最佳模型
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        print(f"Loaded best model with vali_acc: {best_vali_acc:.3f}")
    
    t4 = time.time()
    print(f"total time cost: {t4 - t1:.3f}s")
    
    #推理模型
    #test_loader，并将结果保存到submission.csv
    net.eval()  # 设置为评估模式
    test_preds = []
    with torch.no_grad():  # 推理时不需要计算梯度
        for X in test_loader:
            X = X.to(device)
            y_hat = net(X)
            y_hat = y_hat.argmax(axis=1)
            test_preds.append(y_hat)
    test_preds = torch.cat(test_preds, dim=0)
    test_preds = test_preds.cpu().numpy()
    test_preds = [idx_to_leaf_name[pred] for pred in test_preds]
    test_df["label"] = test_preds
    test_df.to_csv("submission.csv", index=False)

    return 

if __name__ == "__main__":
    print("start...")
    func1()
    print("end...") 
import os 
import sys 
import time 
import torch 
import torch.nn as nn 
import torchvision 
import pandas as pd 
from d2l import torch as d2l 
import my_d2l 
import matplotlib.pyplot as plt


def func1():
    """
    物体检测 
    """
    #图片分类：图片中有一个主体 
    #目标检测：图片中有多个主体，找出每个物体的位置。

    #一个边缘框有两种表示方法
    # 左上角+右下角
    # 左上角+ 宽+ 高 

    #COCO数据集 

    #物体的类别和位置 

    return 

def func2():
    """
    边缘框实现  
    """
    d2l.set_figsize() 
    img = d2l.plt.imread("dog_cat_20260301.png") 
    d2l.plt.imshow(img) 

    #dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 650.0, 493.0] 
    dog_bbox, cat_bbox = [60, 40, 760, 980], [520, 420, 980, 1023]
    boxes = torch.tensor((dog_bbox, cat_bbox)) 
    print(my_d2l.box_center_to_corner(my_d2l.box_corner_to_center(boxes)) == boxes)

    fig = d2l.plt.imshow(img) 
    fig.axes.add_patch(my_d2l.box_to_rect(dog_bbox, "blue"))
    fig.axes.add_patch(my_d2l.box_to_rect(cat_bbox, "red"))
    d2l.plt.show()
    d2l.plt.savefig("dog_cat_20260301_xiu.png")

    return 

def func3():
    """
    数据集
    """
    #目标检测没有比较好的小的数据集 
    d2l.DATA_HUB['banana-detection'] = (
        d2l.DATA_URL + 'banana-detection.zip',
        '5de26c8fce5ccdea9f91267273464dc968d20d72')

    def read_data_bananas(is_train=True):
        """读取香蕉检测数据集中的图像和标签"""
        data_dir = d2l.download_extract('banana-detection')
        csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                                else 'bananas_val', 'label.csv')
        csv_data = pd.read_csv(csv_fname)
        csv_data = csv_data.set_index('img_name')
        images, targets = [], []
        for img_name, target in csv_data.iterrows():
            images.append(torchvision.io.read_image(
                os.path.join(data_dir, 'bananas_train' if is_train else
                            'bananas_val', 'images', f'{img_name}')))
            # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
            # 其中所有图像都具有相同的香蕉类（索引为0）
            targets.append(list(target))
        return images, torch.tensor(targets).unsqueeze(1) / 256
    
    class BananasDataset(torch.utils.data.Dataset):
        """一个用于加载香蕉检测数据集的自定义数据集"""
        def __init__(self, is_train):
            self.features, self.labels = read_data_bananas(is_train)
            print('read ' + str(len(self.features)) + (f' training examples' if
                is_train else f' validation examples'))

        def __getitem__(self, idx):
            return (self.features[idx].float(), self.labels[idx])

        def __len__(self):
            return len(self.features)

    def load_data_bananas(batch_size):
        """加载香蕉检测数据集"""
        train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                                batch_size, shuffle=True)
        val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                            batch_size)
        return train_iter, val_iter    

    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_bananas(batch_size)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape)
    
    imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
    axes = d2l.show_images(imgs, 2, 5, scale=2)
    # for ax, label in zip(axes, batch[1][0:10]):
    #     d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
        
    # 保存带bboxes的图像

    save_dir = "banana_results"
    os.makedirs(save_dir, exist_ok=True)
    for i, (img, label) in enumerate(zip(imgs, batch[1][0:10])):
        fig, ax = plt.subplots()
        ax.imshow(d2l.numpy(img))  # show_images 不支持 ax 参数，直接用 imshow
        d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"banana_{i}.png"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    return 

def func4():
    """
    QA
    """
    
    return 

if __name__ == "__main__":
    print("start...")
    #func1()
    #func2()
    #func3()
    func4()
    print("end...")
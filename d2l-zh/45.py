import os 
import sys 
import time 
import torch 
import torch.nn as nn 
import torchvision 
from d2l import torch as d2l 
from torch.nn import functional as F 

def func1():
    """
    多尺度锚框
    """
    img = d2l.plt.imread("catdog.jpg")
    h, w = img.shape[:2]
    print(h, w)

    def display_anchors(fmap_w, fmap_h, s):
        d2l.set_figsize()
        # 前两个维度上的值不影响输出
        fmap = torch.zeros((1, 10, fmap_h, fmap_w))
        anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
        bbox_scale = torch.tensor((w, h, w, h))
        d2l.show_bboxes(d2l.plt.imshow(img).axes,
                        anchors[0] * bbox_scale)
        d2l.plt.savefig(f"anchors_{fmap_w}_{fmap_h}.png")
        d2l.plt.show()
        d2l.plt.close()

    display_anchors(fmap_w=4, fmap_h=2, s=[0.15])
    display_anchors(fmap_w=2, fmap_h=1, s=[0.4])
    display_anchors(fmap_w=1, fmap_h=1, s=[0.8])

    return 

def func2():
    """
    SSD算法
    """
    def cls_predictor(num_inputs, num_anchors, num_classe):
        return nn.Conv2d(num_inputs, num_anchors * (num_classe + 1), kernel_size=3, padding=1)

    def bbox_predictor(num_inputs, num_anchors):
        return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

    def forward(x, block):
        return block(x)

    Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
    Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
    print(Y1.shape)
    print(Y2.shape)

    return 

if __name__ == "__main__":
    print("start...")
    #func1()
    func2()
    print("end...")
import os 
import sys 
import time 
import torch 
import torchvision 
import json
import pdb

def func1():
    """
    牛仔行头检测
    """
    #kaggle competition url: https://www.kaggle.com/competitions/cowboyoutfits/rules
    #codalab competition url: https://competitions.codalab.org/competitions/33573
    return 

def func2():
    """
    目标检测竞赛
    """
    #数据下载-已完成
    dir_path = "/home/fyw/d2l-zh-mine/data/cowboy_outfits"
    if False:
        json_file_path = os.path.join(dir_path, "train.json")
        with open(json_file_path, "r") as f:
            data = json.load(f)
        pdb.set_trace()
    #please take a look at the 50_1.py script.
    return 

if __name__ == "__main__":
    print("start...")
    t1 = time.time()
    #func1()
    func2()
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
    print("end...")
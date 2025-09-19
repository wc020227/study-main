from torch.utils.data import Dataset      
from PIL import Image
import os   
import torch    
import numpy as np  
import random

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir      #成员变量root_dir 接受来自外部的root路径
        self.label_dir = label_dir    # 成员变量label_dir 接受来自外部的数据集对应的标签
        self.path = os.path.join(self.root_dir, self.label_dir)  #将路径叠加形成完整路径
        self.img_path = os.listdir(self.path)   #将一个已有的列表赋值给 self.img_path 这个成员变量
    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    def __len__(self):   #重写len函数，方便后续对不同数据集进行相加
        return len(self.img_path)   

if __name__ == '__main__':
    dataset = MyDataset(root_dir='dataset/train', label_dir='ants')  
    img, label = dataset[3] #注意要用两个变量接受dataset的返回，因为有两个返回值
    img.show()  #同理dataset[3].show()无法运行，因为函数有两个返回值
       
        
       
        
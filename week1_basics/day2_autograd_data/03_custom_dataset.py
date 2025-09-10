from torch.utils.data import Dataset
from PIL import Image

import os

class MyData(Dataset):

    def __init__(self, dir_path:str, label_dir:str):
        self.root_dir = dir_path
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, index):
        image_name = self.img_path[index]
        image_item_path = os.path.join(self.root_dir, self.label_dir, image_name)
        image = Image.open(image_item_path)
        lable = self.label_dir
        return image, lable

    def __len__(self):
        return len(self.img_path)


root_dir = "datas/hymenoptera_data/train"

ants_dataset = MyData(root_dir, "ants")
bees_dataset = MyData(root_dir, "bees")
train_dataset = ants_dataset + bees_dataset
print(f"ants len: {len(ants_dataset)}")
print(f"bees len: {len(bees_dataset)}")
print(f"total len: {len(train_dataset)}")


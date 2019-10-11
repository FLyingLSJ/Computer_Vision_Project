# -*- coding: utf-8 -*-

import os
import glob
from random import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
__all__ = ['Classification_dataset']

def _init_classes(path):
    with open(os.path.join(path,"classes.txt"),encoding="utf-8") as file:
        content = file.read()
    class_list = content.split("\n")
    class_dict = {}
    for c in class_list:
        c_list = c.split(" ")
        print(c_list)
        class_dict[c_list[0]] = c_list[1]
    return class_dict
def gene_data_txt(dataset_path):
    print('start generate dataset files')
    # regenerate dataset file
    class_dict = _init_classes(dataset_path)
    train_data = []
    test_data = []
    val_data = []
    folder_list = os.listdir(dataset_path)
    for folder in folder_list:
        img_paths = glob.glob(os.path.join(dataset_path, folder, "*.png"))
        n = 0
        for id, img_path in enumerate(img_paths):
            if id < len(img_paths) * 0.1:
                val_data.append((img_path, class_dict[folder]))
                n += 1
            elif id < len(img_paths) * 0.3:
                test_data.append((img_path, class_dict[folder]))
                n += 1
            else:
                train_data.append((img_path, class_dict[folder]))
                n += 1
        print("the number of {} is {}".format(folder, n))
    # 2.shuffle
    shuffle(train_data)
    shuffle(test_data)
    shuffle(val_data)
    # 3.write data
    with open(os.path.join(dataset_path, "train.txt"), "w", encoding="utf-8") as file:
        for item in train_data:
            file.write(item[0] + " " + item[1] + "\n")
    with open(os.path.join(dataset_path, "test.txt"), "w", encoding="utf-8") as file:
        for item in test_data:
            file.write(item[0] + " " + item[1] + "\n")
    with open(os.path.join(dataset_path, "val.txt"), "w", encoding="utf-8") as file:
        for item in val_data:
            file.write(item[0] + " " + item[1] + "\n")
    print('end of build dataset files')


class MyDataset(Dataset):
    def __init__(
            self,
            root,
            transform=None,
            train=None,
            target_transform=None):
        super(MyDataset).__init__()
        # assert train == True or False  ,"train must be True or False"
        if train == 0:
            fh = open(os.path.join(root, "train.txt"), 'r', encoding="utf-8")
        elif train == 1:
            fh = open(os.path.join(root, "val.txt"), 'r', encoding="utf-8")
        elif train == 2:
            fh = open(os.path.join(root, "test.txt"), 'r', encoding="utf-8")
        imgs = []
        for line in fh:
            line = line.strip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert("RGB")
        # img = Image.open(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def Classification_dataset(
        root,
        train,
        transform=None,
        target_transform=None):
    txt_list = sorted([path.replace("\\", "/").split('/')[-1]
                       for path in glob.glob(os.path.join(root, '*.txt'))])
    print(txt_list)
    if len(txt_list) == 4 and set(txt_list) == {
            'classes.txt', 'test.txt', 'train.txt', 'val.txt'}:
        print("Dataset files already exist")
    else:
        gene_data_txt(root)

    return MyDataset(
        root=root,
        transform=transform,
        train=train,
        target_transform=target_transform)


if __name__ == "__main__":
    dataset = Classification_dataset(
        root="test",
        train=0,
        transform=transforms.ToTensor())
    data_load = DataLoader(dataset=dataset, batch_size=6,)
    for (input, target) in data_load:
        print(input.shape)
        print(target)
        break
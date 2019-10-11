# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 18:21
# @Author  : ljf
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from classification_dataset import Classification_dataset
from torch.utils.data import DataLoader
transform = transforms.Compose([
    # transforms.RandomCrop([480,480], padding=10),
    transforms.Resize((300, 300)),
    transforms.ToTensor()])

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

if __name__ == "__main__":

    dataloader = Classification_dataset
    dataset_path = "./data/enhance-magnetic-6-902"
    trainset = dataloader(
        root=dataset_path,
        train=0,
        transform=transform)
    valset = dataloader(
        root=dataset_path,
        train=1,
        transform=transform)
    testset = dataloader(
        root=dataset_path,
        train=2,
        transform=transform)
    mean0,std0 = get_mean_and_std(trainset)
    mean1, std1 = get_mean_and_std(valset)
    mean2, std2 = get_mean_and_std(testset)
    # print(mean0,std0)
    # print(mean1,std1)
    # print(mean2,std2)
    print(mean0.add_(mean1).add_(mean2).div_(3))
    print(std0.add_(std1).add_(std2).div_(3))

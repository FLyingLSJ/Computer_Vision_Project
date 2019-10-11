#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:Administrator
@file: train_demo_from_pytorch.py
@time: 2019/08/26
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tensorboardX import SummaryWriter

plt.ion()   # interactive mode

# writer就相当于一个日志，保存 tensorboard 做图的所有信息
# 会在当前文件夹创建一个 runs 文件夹，存放画图用的文件
writer = SummaryWriter()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    训练模型函数
    :param model: 定义的神经网络
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param scheduler: 学习率
    :param num_epochs: 迭代次数
    :return:
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # 用来保存最好的模型
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                scheduler.step()  #
                print(scheduler.get_lr())

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad() # 先将梯度置 0

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):  # 将梯度计算设置为打开在 train 阶段
                    outputs = model(inputs) # 将数据输入网络，得到第一轮网络前向传播的预测结果 outputs
                    _, preds = torch.max(outputs, 1)  # 预测结果
                    loss = criterion(outputs, labels) # 预测结果 outputs 和 labels 通过定义的交叉熵计算损失

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()  # 误差反向传播
                        optimizer.step() # 优化权重


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)  #

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            if phase == 'train':
                # 每次写入训练损失和精度以及当前迭代的轮次
                # 保存的结果可以用来做可视化
                writer.add_scalar('data/trainloss', epoch_loss, epoch)
                writer.add_scalar('data/trainacc', epoch_acc, epoch)
            else:
                writer.add_scalar('data/valloss', epoch_loss, epoch)
                writer.add_scalar('data/valacc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    return model



# 可视化
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':
    # Data augmentation and normalization for training
    # Just normalization for validation
    # 训练集数据增强和归一化
    # 测试集仅仅做归一化
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(48),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    data_dir = './train_val_Data/'
    # 数据集路径
    """
    - hymenoptera_data
        - train
            - ants
            - bees 
        - val
            - ants
            - bees
    """
    # 图片数据集信息
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                            for x in ['train', 'val']}
    # 数据加载器
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=16,  # 批次大小
            shuffle=True,  # 将数据打乱
            num_workers=4) for x in [
            'train',
            'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} # 数据集数量  {'train': 244, 'val': 153}
    class_names = image_datasets['train'].classes  # 数据集的每个类的名称  ['ants', 'bees']

    # 看设备是否支持 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(class_names)

    TRAIN_FLAG = False
    if TRAIN_FLAG:  # 是否可视化数据
        # ------------------------------------------------------------
        # 数据集可视化
        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])
        # ------------------------------------------------------------

    if os.path.exists("models") is False:
        os.mkdir('models')

    # 迁移学习的方式
    # pretrained_method = ["finetuning", "feature_extraction"]
    pretrained_method = "feature_extraction"
    print("pretrained_method:", pretrained_method)
    if pretrained_method == "finetuning":
        ###################   方法一:微调
        # 加载预训练模型并重置最终的全连接层的参数
        model_ft = models.resnet18(pretrained=True)  # 18 层的残差网络
        num_ftrs = model_ft.fc.in_features  # 取出全连接层上一层卷积的输出大小（或全连接层的输入大小）
        # print(model_ft,num_ftrs)
        model_ft.fc = nn.Linear(num_ftrs, 3)  # 重新设置全连接层的参数
        model_ft = model_ft.to(device)  #
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)  # 优化器设置
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 学习率设置
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)  # 模型训练
        # 保存模型
        torch.save(model_ft.state_dict(), 'models/model.ckpt')  # 保存训练好的模型

    if pretrained_method == "feature_extraction":
        #######################  方法二：特征提取
        model_conv = torchvision.models.resnet18(pretrained=True)
        for param in model_conv.parameters():  # 首先提取出模型中的参数， 然后将其设定为不求梯度
            param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features  # 取出全连接层上一层卷积的输出大小（或全连接层的输入大小）
        model_conv.fc = nn.Linear(num_ftrs, 3)  # 重新设置全连接层的参数，默认的是可以训练的参数
        model_conv = model_conv.to(device)
        criterion = nn.CrossEntropyLoss()
        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        optimizer_conv = optim.Adam(model_conv.parameters()) # 优化器  # 优化全连接层的参数
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=100, gamma=0.1)
        model_conv = train_model(model_conv, criterion, optimizer_conv,
                                 exp_lr_scheduler, num_epochs=500)
        # 保存模型
        torch.save(model_conv.state_dict(), 'models/model.ckpt')  # 保存训练好的模型
        ############################################################





"""
要对损失和精度做可视化需要安装 tensorboard

首先需要安装tensorboard
    pip install tensorboard
然后再安装tensorboardx
    pip install tensorboardx
然后在当前文件夹路径下在终端（xshell）运行 tensorboard --logdir runs   (runs 是保存的文件夹名)
然后浏览器运行 http://localhost:6006 即可看到效果
"""

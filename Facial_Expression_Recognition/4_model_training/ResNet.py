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

# writer���൱��һ����־������ tensorboard ��ͼ��������Ϣ
# ���ڵ�ǰ�ļ��д���һ�� runs �ļ��У���Ż�ͼ�õ��ļ�
writer = SummaryWriter()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    ѵ��ģ�ͺ���
    :param model: �����������
    :param criterion: ��ʧ����
    :param optimizer: �Ż���
    :param scheduler: ѧϰ��
    :param num_epochs: ��������
    :return:
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())  # ����������õ�ģ��
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
                optimizer.zero_grad() # �Ƚ��ݶ��� 0

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):  # ���ݶȼ�������Ϊ���� train �׶�
                    outputs = model(inputs) # �������������磬�õ���һ������ǰ�򴫲���Ԥ���� outputs
                    _, preds = torch.max(outputs, 1)  # Ԥ����
                    loss = criterion(outputs, labels) # Ԥ���� outputs �� labels ͨ������Ľ����ؼ�����ʧ

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()  # ���򴫲�
                        optimizer.step() # �Ż�Ȩ��


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)  #

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            if phase == 'train':
                # ÿ��д��ѵ����ʧ�;����Լ���ǰ�������ִ�
                # ����Ľ���������������ӻ�
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



# ���ӻ�
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
    # ѵ����������ǿ�͹�һ��
    # ���Լ���������һ��
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
    # ���ݼ�·��
    """
    - hymenoptera_data
        - train
            - ants
            - bees 
        - val
            - ants
            - bees
    """
    # ͼƬ���ݼ���Ϣ
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                            for x in ['train', 'val']}
    # ���ݼ�����
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=16,  # ���δ�С
            shuffle=True,  # �����ݴ���
            num_workers=4) for x in [
            'train',
            'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} # ���ݼ�����  {'train': 244, 'val': 153}
    class_names = image_datasets['train'].classes  # ���ݼ���ÿ���������  ['ants', 'bees']

    # ���豸�Ƿ�֧�� GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(class_names)

    TRAIN_FLAG = False
    if TRAIN_FLAG:  # �Ƿ���ӻ�����
        # ------------------------------------------------------------
        # ���ݼ����ӻ�
        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])
        # ------------------------------------------------------------

    if os.path.exists("models") is False:
        os.mkdir('models')

    # Ǩ��ѧϰ�ķ�ʽ
    # pretrained_method = ["finetuning", "feature_extraction"]
    pretrained_method = "feature_extraction"
    print("pretrained_method:", pretrained_method)
    if pretrained_method == "finetuning":
        ###################   ����һ:΢��
        # ����Ԥѵ��ģ�Ͳ��������յ�ȫ���Ӳ�Ĳ���
        model_ft = models.resnet18(pretrained=True)  # 18 ��Ĳв�����
        num_ftrs = model_ft.fc.in_features  # ȡ��ȫ���Ӳ���һ�����������С����ȫ���Ӳ�������С��
        # print(model_ft,num_ftrs)
        model_ft.fc = nn.Linear(num_ftrs, 3)  # ��������ȫ���Ӳ�Ĳ���
        model_ft = model_ft.to(device)  #
        criterion = nn.CrossEntropyLoss()  # ��������ʧ
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)  # �Ż�������
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # ѧϰ������
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)  # ģ��ѵ��
        # ����ģ��
        torch.save(model_ft.state_dict(), 'models/model.ckpt')  # ����ѵ���õ�ģ��

    if pretrained_method == "feature_extraction":
        #######################  ��������������ȡ
        model_conv = torchvision.models.resnet18(pretrained=True)
        for param in model_conv.parameters():  # ������ȡ��ģ���еĲ����� Ȼ�����趨Ϊ�����ݶ�
            param.requires_grad = False
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features  # ȡ��ȫ���Ӳ���һ�����������С����ȫ���Ӳ�������С��
        model_conv.fc = nn.Linear(num_ftrs, 3)  # ��������ȫ���Ӳ�Ĳ�����Ĭ�ϵ��ǿ���ѵ���Ĳ���
        model_conv = model_conv.to(device)
        criterion = nn.CrossEntropyLoss()
        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        optimizer_conv = optim.Adam(model_conv.parameters()) # �Ż���  # �Ż�ȫ���Ӳ�Ĳ���
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=100, gamma=0.1)
        model_conv = train_model(model_conv, criterion, optimizer_conv,
                                 exp_lr_scheduler, num_epochs=500)
        # ����ģ��
        torch.save(model_conv.state_dict(), 'models/model.ckpt')  # ����ѵ���õ�ģ��
        ############################################################





"""
Ҫ����ʧ�;��������ӻ���Ҫ��װ tensorboard

������Ҫ��װtensorboard
    pip install tensorboard
Ȼ���ٰ�װtensorboardx
    pip install tensorboardx
Ȼ���ڵ�ǰ�ļ���·�������նˣ�xshell������ tensorboard --logdir runs   (runs �Ǳ�����ļ�����)
Ȼ����������� http://localhost:6006 ���ɿ���Ч��
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import sys
import torch.nn.functional as F
from net import simpleconv3
from pathlib import Path

data_transforms =  transforms.Compose([
            transforms.Resize(48),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

net = simpleconv3()
# modelpath = sys.argv[1]
modelpath ="./models/model.ckpt"
net.load_state_dict(torch.load(modelpath,map_location=lambda storage,loc: storage))

img_list = Path(r"F:\jupyter\Pytorch\computer_vision\projects\classification\pytorch\motion_project\data\train_val_Data\validation/2/").glob("*.jpg")
# imagepath = sys.argv[2]
for imagepath in img_list:
    print(imagepath.name+"\n")
    image = Image.open(str(imagepath))
    imgblob = data_transforms(image).unsqueeze(0)
    # print(imgblob.shape) # [1, 3, 48, 48]
    imgblob = Variable(imgblob)

    torch.no_grad()

    predict = F.softmax(net(imgblob), dim=1)
    print(predict.argmax().item())

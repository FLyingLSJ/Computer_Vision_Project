#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:asus_pc
@file: face_detector.py
@time: 2019/08/17
"""

# 开发环境
# Python3.6

# Python 2/3 compatibility
from __future__ import print_function
import cv2  # 4.0.0
import dlib  # 19.8.1 到 https://pypi.org/simple/dlib/ 下载 whl 文件 pip install *.whl 安装
import numpy as np  # 1.16.2
from pathlib import Path
import sys
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
import torch.nn.functional as F
from net import simpleconv3

# 配置 Dlib 关键点检测路径
# 文件可以从 http://dlib.net/files/ 下载
PREDICTOR_PATH = "./static/face_detector_trained/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)  # 关键点检测
# 配置人脸检测器路径
cascade_path = "./static/face_detector_trained/haarcascade_frontalface_default.xml"  # 在 opencv github 可以找到
# 初始化分类器
cascade = cv2.CascadeClassifier(cascade_path)


data_transforms = transforms.Compose([
    transforms.Resize(48),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

net = simpleconv3()
modelpath = "./models/expression_models/model.ckpt"
net.load_state_dict(
    torch.load(modelpath, map_location=lambda storage, loc: storage))

result_dict = {0: "no-smile", 1: "pout", 2: "smile"}
    

    
    
# 调用 cascade.detectMultiScale 人脸检测器和 Dlib 的关键点检测算法 predictor 获得关键点结果
def get_landmarks(im):
    try:
        rects = cascade.detectMultiScale(im, 1.3, 5)  # 进行多尺度检测
        if len(rects) == 1:
            x, y, w, h = rects[0]
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))  # 获得检测框
            return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])  # 调用 dlib 关键点检测
    except:
        return None


#  打印关键点信息方便调试
def annotat_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 5, color=(0, 255, 255))
    return im


def get_mouth(im):
    # 得到 68 个关键点
    landmarks = get_landmarks(im)
    if landmarks is not None:
        # print(landmarks)
        xmin = 10000
        xmax = 0
        ymin = 10000
        ymax = 0
        # 根据最外围的关键点获取包围嘴唇的最小矩形框
        # 68 个关键点是从
        # 左耳朵0 -下巴-右耳朵16-左眉毛（17-21）-右眉毛（22-26）-左眼睛（36-41）
        # 右眼睛（42-47）-鼻子从上到下（27-30）-鼻孔（31-35）
        # 嘴巴外轮廓（48-59）嘴巴内轮廓（60-67）
        for i in range(48, 67):
            x = landmarks[i, 0]
            y = landmarks[i, 1]
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
        # print("xmin", xmin)
        # print("xmax", xmax)
        # print("ymin", ymin)
        # print("ymax", ymax)
        roiwidth = xmax - xmin  # 矩形框的宽和高
        roiheight = ymax - ymin
        roi = im[ymin:ymax, xmin:xmax, :]
        # cv2.imshow("roi_0", roi)
        # 将最小矩形扩大 1.5 倍，获得最终矩形框
        if roiwidth > roiheight:  # 宽和高哪个大哪个就 ×1.5 倍
            dstlen = 1.5 * roiwidth
        else:
            dstlen = 1.5 * roiheight

        diff_xlen = dstlen - roiwidth
        diff_ylen = dstlen - roiheight
        newx = xmin
        newy = ymin
        imagerows, imagecols, ch = im.shape
        # print("imagerows, imagecols", imagerows, imagecols)
        if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
            newx = newx - diff_xlen / 2
        elif newx < diff_xlen / 2:
            newx = 0
        else:
            newx = imagecols - dstlen
        if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
            newy = newy - diff_ylen / 2
        elif newy < diff_ylen / 2:
            newy = 0
        else:
            newy = imagecols - dstlen

        roi = im[int(newy):int(newy + dstlen), int(newx):int(newx + dstlen), :]
        return roi
    return None


def expression_predict(img_path):
    im = cv2.imread(img_path)
    roi = get_mouth(im)
    if roi is not None:
        roi_pil = Image.fromarray(roi)
        imgblob = data_transforms(roi_pil).unsqueeze(0)
        #print(imgblob.shape)  # [1, 3, 48, 48]
        imgblob = Variable(imgblob)
        torch.no_grad()
        predict = F.softmax(net(imgblob), dim=1)
        result = predict.argmax().item()
#         print(result_dict[result])
        return result_dict[result]
    return None    
# print(expression_predict("test_img/pout.jpg"))
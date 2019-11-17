import xml.etree.ElementTree as ET
import pickle
import string
import os
import shutil
from os import listdir, getcwd
from os.path import join
import cv2
 
sets=[('2012', 'train')]
 
classes = ["cat", "dog"]  # 根据自己的需要修改
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
def convert_annotation(image_id,flag,savepath):
 
    if flag == 0:
        in_file = open(savepath+'/trainImageXML/%s.xml' % (os.path.splitext(image_id)[0]))
        out_file = open(savepath+'/trainImage/%s.txt' % (os.path.splitext(image_id)[0]), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
 
        img = cv2.imread('./trainImage/'+str(image_id))
        h = img.shape[0]
        w = img.shape[1]
 
    elif flag == 1:
        in_file = open(savepath+'/validateImageXML/%s.xml' % (os.path.splitext(image_id)[0]))
        out_file = open(savepath+'/validateImage/%s.txt' % (os.path.splitext(image_id)[0]), 'w')
 
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
 
        img = cv2.imread('./validateImage/' + str(image_id))
        h = img.shape[0]
        w = img.shape[1]
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text  # 类的名称
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)  # 类别的索引
        xmlbox = obj.find('bndbox')  # 边框的区域
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))  # 得到的是两点坐标
        bb = convert((w,h), b)  # 将两点坐标转换成 x y w h
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')  # 写入到文件夹
 
wd = getcwd()
 
for year, image_set in sets:
    savepath = os.getcwd();
    idtxt = savepath + "/validateImageId.txt";
    pathtxt = savepath + "/validateImagePath.txt";
    image_ids = open(idtxt).read().strip().split()
    list_file = open(pathtxt, 'w')
    s = '\xef\xbb\xbf'
    for image_id in image_ids:
        nPos = image_id.find(s)
        if nPos >= 0:
            image_id = image_id[3:]
        list_file.write('%s/validateImage/%s\n' % (wd, image_id))
        print(image_id)
        convert_annotation(image_id, 1, savepath)
    list_file.close()
 
    idtxt = savepath + "/trainImageId.txt";
    pathtxt = savepath + "/trainImagePath.txt" ;
    image_ids = open(idtxt).read().strip().split()
    list_file = open(pathtxt, 'w')
    s = '\xef\xbb\xbf'
    for image_id in image_ids:
        nPos = image_id.find(s)
        if nPos >= 0:
           image_id = image_id[3:]
        list_file.write('%s/trainImage/%s\n'%(wd,image_id))
        print(image_id)
        convert_annotation(image_id,0,savepath)
    list_file.close()
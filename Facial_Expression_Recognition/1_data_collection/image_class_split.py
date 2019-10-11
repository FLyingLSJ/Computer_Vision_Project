#!usr/bin/env python
# -*- coding:utf-8 _*-
import shutil
import pandas as pd
"""
@author:asus_pc
@file: img_processing.py
@time: 2019/08/18
"""
"""
通过数据说明文件或者其他信息，将原始数据进行分类，分成不同类别
"""

describe_file = r"./list_attr_celeba.csv"  # 描述文件路径
describe_info = pd.read_csv(describe_file)
img_info = describe_info[["image_id", "Smiling"]]  # 只需要 图片 id 以及 Smiling 字段

with open("img_info.csv", 'r') as f:  # 将两个字段抽取后保存成一个中间文件，只有 图片 id 以及 Smiling 字段
    info = f.readlines()

src_dir = r"./img_align_celeba/"  # 图片路径
for i in info:
    image_id = i.split(",")[0]
    smiling_flag = i.split(",")[-1].split("\n")[0]
    if smiling_flag == "1":
        dst = r"./smiling/"  # 微笑表情保存路径
    else:
        dst = r"./no_smiling/" # 中性表情保存路径
    shutil.copyfile(src_dir + image_id, dst + image_id)


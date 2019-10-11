#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:Administrator
@file: split_train_val_test.py
@time: 2019/08/20
"""
import shutil
import os


def path_init(src_path, dst_path, rate=(0.6, 0.2, 0.2)):
    """
    将原始数据按比较分配成 train validation test
    :param src_path: 原始数据路径，要求格式如下
    - src_path
        - class_1
        - class_2
        ...
    :param dst_path: 目标路径
    :param rate: 分配比例，加起来一定要等于 1
    :return:
    # 以下几行是创建如下格式的文件夹
    - img_data
        - train
            - class_1
            - class_2
            ...
        - validation
            - class_1
            - class_2
            ...
        - test
            - class_1
            - class_2
            ...
    """
    try:
        class_names = os.listdir(src_path)  # 获取原始数据所有类别的纯文件名
        dst_path = dst_path + '/' + 'splited_data'
        os.mkdir(dst_path)  # 创建目标文件夹
        three_paths = [dst_path + '/' +
                       i for i in ['train', 'validation', 'test']]  # 三个文件夹的路径
        for three_path in three_paths:
            os.mkdir(three_path)
            for class_name in class_names:
                os.mkdir(three_path + '/' + class_name)
        # -----------------------------

        dst_train = dst_path + '/' + 'train'
        dst_validation = dst_path + '/' + 'validation'
        dst_test = dst_path + '/' + 'test'

        class_names_list = [
            src_path +
            '/' +
            class_name for class_name in class_names]  # 获取原始数据所有类别的路径

        for class_li in class_names_list:
            imgs = os.listdir(class_li)  # 当前类别所有图片的文件名，不包括路径
            # 得到当前类别的所有图片的路径，指定后缀
            imgs_list = [class_li + '/' +
                         img for img in imgs if img.endswith("jpg")]
            print(len(imgs_list))
            img_num = len(imgs_list)  # 当前类别的图片数量
            # 三个文件夹的数量
            train_num = int(rate[0] * img_num)
            validation_num = int(rate[1] * img_num)
            # test_num = int(rate[2]*img_num)

            for img in imgs_list[0:train_num]:
                # 训练集复制
                src = img
                dst = dst_train + '/' + \
                    img.split('/')[-2] + '/' + img.split('/')[-1]
                # print(src, " ", dst)
                shutil.copyfile(src=img, dst=dst)
            print("训练集数量：", len(imgs_list[0:train_num]))

            for img in imgs_list[train_num:train_num + validation_num]:
                # 验证集复制
                src = img
                dst = dst_validation + '/' + \
                    img.split('/')[-2] + '/' + img.split('/')[-1]
                # print(src, " ", dst)
                shutil.copyfile(src=img, dst=dst)
            print("验证集数量：",
                  len(imgs_list[train_num:train_num + validation_num]))

            for img in imgs_list[train_num + validation_num:]:
                # 测试集复制
                src = img
                dst = dst_test + '/' + \
                    img.split('/')[-2] + '/' + img.split('/')[-1]
                # print(src, " ", dst)
                shutil.copyfile(src=img, dst=dst)
            print("测试集数量：", len(imgs_list[train_num + validation_num:]))

    except BaseException:
        print("目标文件夹已经存在或原始文件夹不存在，请检查！")


# # 例程
src_path = r"F:\jupyter\Pytorch\computer_vision\projects\classification\pytorch\motion_project\data\train_val/"
dst_path = r'F:\jupyter\Pytorch\computer_vision\projects\classification\pytorch\motion_project\data\train_val/'    # 基本都在相对路径下创建
path_init(src_path, dst_path, rate=(0.9, 0.1, 0))

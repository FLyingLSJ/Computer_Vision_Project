import cv2
from pathlib import Path
import os
import sys

img_path = sys.argv[1]#r"F:/jupyter/motion/final_data/pout_mouth//"
img_list = [im for im in Path(img_path).glob("*.jpg")]
for im in img_list:
    image = cv2.imread(str(im))
    shape = image.shape[0]
    if shape > 60: # 缩小图像，一般使用 INTER_AREA 方式   
        dst = cv2.resize(image, (60, 60), interpolation=cv2.INTER_AREA)
    elif shape < 60:  # 放大图片 一般采用 INTER_LINEAR 效率更高、速度更快 （解析来源于：《OpenCV3 编程入门》：毛星云）
        dst = cv2.resize(image, (60, 60), interpolation=cv2.INTER_LINEAR)
    else: # 保持不动
        dst = image 
    os.remove(str(im))
    cv2.imwrite(str(im), dst)

import cv2
from pathlib import Path
import os
import sys

img_path = sys.argv[1]#r"F:/jupyter/motion/final_data/pout_mouth//"
img_list = [im for im in Path(img_path).glob("*.jpg")]
for im in img_list:
    image = cv2.imread(str(im))
    shape = image.shape[0]
    if shape > 60: # ��Сͼ��һ��ʹ�� INTER_AREA ��ʽ   
        dst = cv2.resize(image, (60, 60), interpolation=cv2.INTER_AREA)
    elif shape < 60:  # �Ŵ�ͼƬ һ����� INTER_LINEAR Ч�ʸ��ߡ��ٶȸ��� ��������Դ�ڣ���OpenCV3 ������š���ë���ƣ�
        dst = cv2.resize(image, (60, 60), interpolation=cv2.INTER_LINEAR)
    else: # ���ֲ���
        dst = image 
    os.remove(str(im))
    cv2.imwrite(str(im), dst)

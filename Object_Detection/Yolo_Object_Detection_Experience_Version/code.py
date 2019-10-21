# -*- coding: utf-8 -*-
# 载入所需库
import cv2
import numpy as np
import os
import time


def NMSBoxes_fix(boxes, confidences, confidence_thre, nms_thre, class_id):
    class_id_set = set(class_id)  #  总共有几类
    result = []  #  用来存储结果的
    for cls in class_id_set:  # 遍历每个类别
        cls_boxes = []  # 用来保存每个类别的  边框
        cls_confidences = []  # 用来保存每个类别边框的分数
        indices = [i for i, c in enumerate(class_id) if c == cls] # 某一类在原始输入的所有索引
        for i in indices:
            cls_boxes.append(boxes[i])  # 把属于该类的框框和分数找出来
            cls_confidences.append(confidences[i]) 
        idxs = cv2.dnn.NMSBoxes(cls_boxes, cls_confidences, confidence_thre, nms_thre)  # 对每类进行 NMS 操作
        for i in idxs:  #  找出原始输入的索引，并把经过 NMS 操作后保留下来的框框的索引保存下来到一个列表中
            result.append([indices[i[0]]])  #  
    return np.array(result)  #  opencv 原始的 NMS 输出是一个 np.array 的数据，所以我们也将其转化成指定格式


def yolo_detect(pathIn='',
                pathOut=None,
                label_path='./cfg/coco.names',
                config_path='./cfg/yolov3_coco.cfg',
                weights_path='./cfg/yolov3_coco.weights',
                confidence_thre=0.5,
                nms_thre=0.20,
                jpg_quality=80):

    '''
    pathIn：原始图片的路径
    pathOut：结果图片的路径
    label_path：类别标签文件的路径
    config_path：模型配置文件的路径
    weights_path：模型权重文件的路径
    confidence_thre：0-1，置信度（概率/打分）阈值，即保留概率大于这个值的边界框，默认为0.5
    nms_thre：非极大值抑制的阈值，默认为0.3
    jpg_quality：设定输出图片的质量，范围为0到100，默认为80，越大质量越好
    '''

    # 加载类别标签文件
    LABELS = open(label_path).read().strip().split("\n")
    nclass = len(LABELS)
    
    # 为每个类别的边界框随机匹配相应颜色
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(nclass, 3), dtype='uint8')
    
    # 载入图片并获取其维度
    base_path = os.path.basename(pathIn)
    print(base_path)
    img = cv2.imread(pathIn)
    img_copy = img.copy()
    (H, W) = img.shape[:2]
    
    # 加载模型配置和权重文件
    print('从硬盘加载YOLO......')
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # 获取YOLO输出层的名字
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # 将图片构建成一个blob，设置图片尺寸，然后执行一次
    # YOLO前馈网络计算，最终获取边界框和相应概率
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    
    # 显示预测所花费时间
    print('YOLO模型花费 {:.2f} 秒来预测一张图片'.format(end - start), pathIn.split("/")[-1])
    
    # 初始化边界框，置信度（概率）以及类别
    boxes = []
    confidences = []
    classIDs = []
    
    # 迭代每个输出层，总共三个
    for output in layerOutputs:
    	# 迭代每个检测
    	for detection in output:
    		# 提取类别ID和置信度
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]
    
    		# 只保留置信度大于某值的边界框
    		if confidence > confidence_thre:
    			# 将边界框的坐标还原至与原图片相匹配，记住YOLO返回的是
                # 边界框的中心坐标以及边界框的宽度和高度
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")
    
    			
    			# 计算边界框的左上角位置
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))
    
    			# 更新边界框，置信度（概率）以及类别
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)
    			print(LABELS[classID]+"area",width*height)
    			cv2.rectangle(img_copy, (x, y), (x+width, y+height), (0, 0, 255), 2, 8, 0)
    			cv2.putText(img_copy, LABELS[classID], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    print("boxes", boxes)
    print("confidences", confidences)
    print("classIDs", classIDs)
    
    # 自己修正过的 非极大值抑制方法
    #idxs = NMSBoxes_fix(boxes, confidences, confidence_thre, nms_thre, classIDs)
    # 使用非极大值抑制方法抑制弱、重叠边界框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thre, nms_thre)
    print(type(idxs))

    # 确保至少一个边界框
    if len(idxs) > 0:
    	# 迭代每个边界框
    	for i in idxs.flatten():
            # 提取边界框的坐标
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # 绘制边界框以及在左上角添加类别标签和置信度
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            print("LABELS:", LABELS[classIDs[i]])
            text = '{}: {:.3f}'.format(LABELS[classIDs[i]], confidences[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 绘制分数大于阈值的所有框
            cv2.rectangle(img, (x, y-text_h-baseline), (x + text_w, y), color, -1)
            cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # 输出结果图片
    if pathOut is None:
        cv2.imwrite('with_box_'+base_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    else:
        cv2.imwrite(pathOut, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        cv2.imwrite("NOT_NMS.jpg", img_copy)

## 测试
src = './test_imgs/'
dst = './result_imgs/'
img_list = os.listdir(src)
for img in img_list[-1:]:
    pathIn = src + img
    pathOut = dst + img
    yolo_detect(pathIn, pathOut)

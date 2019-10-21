import cv2 as cv
import numpy as np

model_bin = "./cfg/yolov3_coco.weights";
config_text = "./cfg/yolov3_coco.cfg";

# Load names of classes
classes = None
with open('./cfg/coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
print("classes names", classes)
print()
# load tensorflow model
net = cv.dnn.readNetFromDarknet(config_text, model_bin)
image = cv.imread(r"./test_imgs/cat.jpg")
image_copy = image.copy()

h = image.shape[0]
w = image.shape[1]

# 获得所有层名称与索引
layerNames = net.getLayerNames()
lastLayerId = net.getLayerId(layerNames[-1])
lastLayer = net.getLayer(lastLayerId)
print(lastLayer.type)
print()

# 基于多个Region层输出getUnconnectedOutLayersNames
blobImage = cv.dnn.blobFromImage(image, 1.0/255.0, (416, 416), None, True, False);
outNames = net.getUnconnectedOutLayersNames()
net.setInput(blobImage)
outs = net.forward(outNames)

# Put efficiency information.
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
cv.putText(image, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

print(outs)
print()
print(outs[0].shape)

# 绘制检测矩形
classIds = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]  #  总共有 80 个类，该值得到的是每个类的分数
        classId = np.argmax(scores)  #  分数最高的那个序号
        confidence = scores[classId]  #  预测的分数
        # numbers are [center_x, center_y, width, height]
        if confidence > 0.5:
            center_x = int(detection[0] * w)
            center_y = int(detection[1] * h)
            width = int(detection[2] * w)
            height = int(detection[3] * h)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            classIds.append(classId)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])
            #  绘制所有分数大于阈值的框
            cv.rectangle(image_copy, (left, top), (left+width, top+height), (0, 0, 255), 2, 8, 0)
            cv.putText(image_copy, classes[classId], (left, top), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

# cv.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]])
"""
"""
# 非极大值抑制
indices = cv.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
print(indices)
for i in indices:
    i = i[0]
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    cv.rectangle(image, (left, top), (left+width, top+height), (0, 0, 255), 2, 8, 0)
    cv.putText(image, classes[classIds[i]], (left, top), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

cv.imwrite('YOLOv3-Detection-Demo_NMS.jpg', image)
cv.imwrite('YOLOv3-Detection-Demo.jpg', image_copy)
cv.waitKey(0)
cv.destroyAllWindows()
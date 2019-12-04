### 前言

本项目参考 https://github.com/eriklindernoren/PyTorch-YOLOv3 ，感谢大佬开源，在此基础上增加了数据准备的说明，项目流程说明。

[数据准备说明文档]: data/custom/readme.md	"  "

更多项目实战可以关注『**机器视觉CV**』公众号

<img src="公众号.jpg" alt=" " style="zoom:50%;" />



在数据集整理完毕后训练的步骤如下：

### 1. 修改配置文件

```bash
$ cd config/   # Navigate to config dir
# Will create custom model 'yolov3-custom.cfg'
$ bash create_custom_model.sh <num-classes>   #  <num-classes> 类别数目参数，根据你的需要修改
```

### 2. 修改 config/custom.data 文件

```python
classes= 2  # 类别数，根据你的需要修改
train=data/custom/train.txt
valid=data/custom/valid.txt
names=data/custom/classes.names
```


### 3. 修改 data/custom/classes.names 文件

每个类别一行，顺序要和 data/custom/3_trans.py 中的 classes 变量的顺序一样


### 4. 数据集处理流程请见 data/custom/readme.txt

### 5. 上述准备准备完毕后开始训练

```bash
pip3 install -r requirements.txt
pip install terminaltables
```


```python
# 训练命令
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights weights/darknet53.conv.74
# 添加其他参数请见 train.py 文件
    
# 从中断的地方开始训练
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --pretrained_weights checkpoints/yolov3_ckpt_299.pth --epoch 

```

```python
# 测试：
python detect_2.py --image_folder data/samples/ --weights_path checkpoints/yolov3_ckpt_25.pth --model_def config/yolov3-custom.cfg --class_path data/custom/classes.names
# 若是在 GPU 的电脑上训练，在 CPU 的电脑上预测，则需要修改 model.load_state_dict(torch.load(opt.weights_path, map_location='cpu'))
```


​    

- 本项目参考：https://github.com/eriklindernoren/PyTorch-YOLOv3
- yolo 博客地址：https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
- 机器之心翻译：https://www.jiqizhixin.com/articles/2018-04-23-3
- yolo源码解析：https://zhuanlan.zhihu.com/p/49981816
- yolo 解读：https://zhuanlan.zhihu.com/p/76802514

### 6. 其他

出现警告解决方案
`UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead`. 
在 model.py  **计算损失的位置 大概在 196 行左右**添加以下两句

```python 
obj_mask=obj_mask.bool() # convert int8 to bool
noobj_mask=noobj_mask.bool() #convert int8 to bool
```



注意预测的时候需要检查数据的格式问题（单通道？三通道？）



预测的值分别是  x1, y1, x2, y2, conf, cls_conf, cls_pred

cfg 中的 路由层（Route）
它的参数 layers 有一个或两个值。当只有一个值时，它输出这一层通过该值索引的特征图。
在我们的实验中设置为了 - 4，所以层级将输出路由层之前第四个层的特征图。
当层级有两个值时，它将返回由这两个值索引的拼接特征图。在我们的实验中为 - 1 和 61，
因此该层级将输出从前一层级（-1）到第 61 层的特征图，并将它们按深度拼接。
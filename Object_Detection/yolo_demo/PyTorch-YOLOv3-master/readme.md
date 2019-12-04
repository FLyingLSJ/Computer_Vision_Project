### 前言

本项目参考 https://github.com/eriklindernoren/PyTorch-YOLOv3 ，感谢大佬开源，在此基础上增加了数据准备的说明，项目流程说明。

[数据准备说明文档]: data/custom/readme.md	"数据准备说明文档"

更多项目实战可以关注『**机器视觉CV**』公众号

<img src="公众号.jpg" alt=" " style="zoom:50%;" />



`注意：因为本人的电脑没有 GPU ，故跑不了代码，本项目所有代码都是在 Linux  下运行的，如果你的电脑（Win 操作系统想要运行本代码，可以安装 Git 软件来执行 Linux 命令）`

在数据集整理完毕后训练的步骤如下：

### 1. 修改配置文件和下载预训练权重

```bash
$ cd config/   # Navigate to config dir
# Will create custom model 'yolov3-custom.cfg'
$ bash create_custom_model.sh <num-classes>   #  <num-classes> 类别数目参数，根据你的需要修改

# 下载预训练权重
$ cd weights/
$ bash download_weights.sh
```

### 2. 修改 config/custom.data 文件

```python
classes= 2  # 类别数，根据你的需要修改
train=data/custom/train.txt
valid=data/custom/valid.txt
names=data/custom/classes.names
```


### 3. 修改 data/custom/classes.names 文件

这里需要分两种情况

- 有完整的数据，也就是有原始的图片和 txt 文件的，请执行下面的操作

  因为你已经有了 txt  格式的标注数据了，大致的内容如下，第一个数字就是类别对应的代码了，比如 cat 对应 0，dog 对应 1，pig 对应 2

  ```bash
  0 0.014656616415410386 0.41642011834319526 0.024288107202680067 0.051775147928994084
  1 0.22989949748743718 0.33357988165680474 0.01423785594639866 0.034023668639053255
  2 0.28936348408710216 0.30029585798816566 0.01256281407035176 0.03254437869822485
  ```

  那么 data/custom/classes.names 文件必须写成以下形式，必须要对应，否则会出错的

  ```bash
  cat
  dog
  pig
  
  ```

  

- 没有完整的数据的，即，只有原始图片和 xml 文件的，请执行下面的操作

  每个类别一行，顺序要和 data/custom/3_trans.py 中的 classes 变量的顺序一样

不管你有么有完整的数据，也请看看第 4 步，没有完整数据的，根据第 4 步进行数据格式转换，有完整数据的，请看看数据的存放格式是怎么样的

### 4. 数据集处理流程请见 data/custom/readme.md



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
在 model.py  **计算损失的位置 大概在 196 行左右**添加以下两句（我已经添加到源码中了）

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
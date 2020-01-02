# 卷积神经网络之-Alexnet

![](https://cdn.nlark.com/yuque/0/2019/png/653487/1576498235499-59fd5c49-05ce-4431-9d4f-57d92df4ab1a.png)

论文地址：[https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

2012年， AlexNet横空出世。这个模型的名字来源于论⽂第⼀作者的姓名Alex Krizhevsky [1]。AlexNet使⽤了8层卷积神经⽹络，并以很⼤的优势赢得了ImageNet 2012图像识别挑战赛冠军。

- top1: 37.5%
- top5: 17.0% 
- 6千万参数 650000 个神经元

![细化的结构图，来自互联网，侵删](https://cdn.nlark.com/yuque/0/2019/jpeg/653487/1577255770296-de14583d-c16e-4743-acd8-b29aee475def.jpeg)
### 与 LeNet 相比较

第⼀，与相对较⼩的 LeNet 相⽐， AlexNet 包含8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。AlexNet第⼀层中的卷积窗⼝形状是 11×11。因为ImageNet中绝⼤多数图像的⾼和宽均⽐MNIST图像 的⾼和宽⼤10倍以上， ImageNet图像的物体占⽤更多的像素，所以需要更⼤的卷积窗⼝来捕获物体。 第⼆层中的卷积窗⼝形状减⼩到 5×5 ，之后全采⽤  3×3。此外，第⼀、第⼆和第五个卷积层之后都使 ⽤了窗⼝形状为 3×3 、步幅为2的最⼤池化层。⽽且， AlexNet使⽤的卷积通道数也⼤于LeNet中的卷积通道数数⼗倍。

第⼆，** AlexNet将sigmoid激活函数改成了更加简单的ReLU激活函数**。⼀⽅⾯， ReLU激活函数的计算更简单，例如它并没有sigmoid激活函数中的求幂运算。另⼀⽅⾯，ReLU激活函数在不同的参数初始化⽅法下使模型更容易训练。这是由于当 sigmoid 激活函数输出极接近0或1时，这些区域的梯度⼏乎为0，从⽽造成反向传播⽆法继续更新部分模型参数；⽽ReLU激活函数在正区间的梯度恒为1。因此，若模型参数初始化不当， sigmoid函数可能在正区间得到⼏乎为0的梯度，从⽽令模型⽆法得到有效训练。

> **Relu 比 Sigmoid 的效果好在哪里？**
> Sigmoid 的导数只有在 0 的附近时有较好的激活性，而在正负饱和区域的梯度趋向于 0, 从而产生梯度弥散的现象，而 relu 在大于 0 的部分梯度为常数，所以不会有梯度弥散现象。Relu 的导数计算的更快。Relu 在负半区的导数为 0, 所以神经元激活值为负时，梯度为 0, 此神经元不参与训练，具有稀疏性。



![](https://cdn.nlark.com/yuque/0/2019/png/653487/1577255512819-3963a4d4-dbf7-4070-aff3-5f4d6b980200.png)

第三， **AlexNet通过丢弃法（dropout）来控制全连接层的模型复杂度**。⽽LeNet并没有使⽤丢弃法。

第四，**AlexNet引⼊了⼤量的图像增⼴**，如翻转、裁剪和颜⾊变化，从⽽进⼀步扩⼤数据集来缓解过拟合。

### 代码实现

```python
import time
import torch
from torch import nn, optim
import torchvision



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    
    
    
    
```



### 模型特性

- SGD, 128 batch-size, momentum=0.9, weight decay=0.0005, 学习率=0.01
- 所有卷积层都使用ReLU作为非线性映射函数，使模型收敛速度更快
- 在多个GPU上进行模型的训练，不但可以提高模型的训练速度，还能提升数据的使用规模
- 使用LRN对局部的特征进行归一化，结果作为ReLU激活函数的输入能有效降低错误率
- 重叠最大池化（overlapping max pooling），即池化范围z与步长s存在关系 z>s （如 ![](https://cdn.nlark.com/yuque/__latex/01c9349454e0729d5783060aa15733fd.svg#card=math&code=S_%7Bmax%7D&height=14&width=27)中核尺度为 3×3/2），避免平均池化（average pooling）的平均效应
- 使用随机丢弃技术（dropout）选择性地忽略训练中的单个神经元，避免模型的过拟合（也使用数据增强防止过拟合）


### 贡献/意义

- AlexNet跟LeNet结构类似，但使⽤了更多的卷积层和更⼤的参数空间来拟合⼤规模数据集ImageNet。它是浅层神经⽹络和深度神经⽹络的分界线。
- 虽然看上去AlexNet的实现⽐LeNet的实现也就多了⼏⾏代码⽽已，但这个观念上的转变和真正优秀实验结果的产⽣令学术界付出了很多年。





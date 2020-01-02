# 卷积神经网络之-Lenet



### 前言

Lenet 是一系列网络的合称，包括 Lenet1 - Lenet5，由 Yann LeCun 等人在1990 年《Handwritten Digit Recognition with a Back-Propagation Network》中提出，是卷积神经网络的 HelloWorld。


### Lenet5

Lenet 的最终版本是 Lenet5，是一个 7 层的神经网络，包含3个卷积层，2个池化层，1个全连接层。其中所有卷积层的所有卷积核都为5x5，步长 strid=1，池化方法都为全局pooling，激活函数为 Sigmoid，网络结构如下：

![](https://cdn.nlark.com/yuque/0/2019/png/653487/1576200015803-7d48ad06-b845-4b95-944d-05ef7e1fd00b.png#align=left&display=inline&height=212&originHeight=594&originWidth=2064&size=0&status=done&style=none&width=735#align=left&display=inline&height=594&originHeight=594&originWidth=2064&status=done&style=none&width=2064#align=left&display=inline&height=594&originHeight=594&originWidth=2064&status=done&style=none&width=2064)


### 代码复现

Lenet 网络的参数量，以及每层的输出特征图大小如下：

- 卷积的卷积核都为 5×5 步长 stride=1
- 输入是 32×32
- -> 6@28*28（卷积C1）   参数：5×5×6+6 =156
- -> 6@14*14（池化S2）   参数：偏移量参数  2×6
- -> 16@10*10（卷积C3）  参数：5×5×6×16+16 = 2416  # 这里与原始的 LeNet 网络有区别
- -> 16@5*5（池化S4）    参数：偏移量参数  2×16
- -> 120@1*1（卷积C5）当然，这里也可以认为是全连接层（因为上一层得到的特征图是5x5，卷积核也为5x5）   参数：5×5×16×120+120 = 48120
- -> 84（全连接F6） 这个 84 的选取有个背景：与 ASCII 码表示的 7×12 的位图大小相等  参数：120×84
- -> 10(输出类别数)  参数：84×10

下面我们用 Pytorh 框架实现一些 Lenet5，实际代码会与上面的说明有些差别，并模拟一个输入进行测试。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False): 
        """
        num_classes: 分类的数量
        grayscale：是否为灰度图
        """
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes
        
        if self.grayscale: # 可以适用单通道和三通道的图像
            in_channels = 1
        else:
            in_channels = 3
        
        # 卷积神经网络
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2)   # 原始的模型使用的是 平均池化
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),  # 这里把第三个卷积当作是全连接层了
            nn.Linear(120, 84), 
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x) # 输出 16*5*5 特征图
        x = torch.flatten(x, 1) # 展平 （1， 16*5*5）
        logits = self.classifier(x) # 输出 10
        probas = F.softmax(logits, dim=1)
        return logits, probas
        
        
    
num_classes = 10  # 分类数目
grayscale = True  # 是否为灰度图
data = torch.rand((1, 1, 32, 32))
print("input data:\n", data, "\n")
model = LeNet5(num_classes, grayscale)
logits, probas = model(data)
print("logits:\n",logits)
print("probas:\n",probas)
```

最后模拟了一个输入，输出一个分类器运算后的结果和 10 个 softmax 概率值

![image.png](https://cdn.nlark.com/yuque/0/2019/png/653487/1576201454491-8c74e1b4-2f6d-4994-a9a0-c7d4224a827a.png)



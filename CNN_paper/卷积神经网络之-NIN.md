# NiN网络（Network In Network）




### 简介

Network In Network 是发表于 2014 年 ICLR 的一篇 paper。当前被引了 3298 次。这篇文章采用较少参数就取得了 Alexnet 的效果，Alexnet 参数大小为 230M，而 Network In Network 仅为 29M，这篇 paper 主要两大亮点：mlpconv (multilayer perceptron，MLP，多层感知机)作为 "micro network"和 Global Average Pooling（全局平均池化）。论文地址：[https://arxiv.org/abs/1312.4400](https://arxiv.org/abs/1312.4400)


### 创新点

(1) mlpconv Layer

在介绍 mlpconv Layer 之前，我们先看看经典的 Linear Convolutional Layer（线性卷积层）是怎么进行操作的，

![Linear Convolutional Layer 结构](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1577971032705-5e6b0337-631f-4f41-b449-d9c0ddcef250.jpeg)
![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1577971181522-2562bd77-7d3d-454d-be9a-d0a15cc1ee2a.jpeg#align=left&display=inline&height=43&originHeight=43&originWidth=231&size=0&status=done&style=none&width=231)

(_i_, _j_) 是特征图中像素的位置索引，x_ij 表示像素值，而 k 用于特征图通道的索引，W 是参数，经过 WX 计算以后经过一个 relu 激活进行特征的抽象表示。

下面就介绍 mlpconv Layer 结构

<br />![mlpconv Layer 结构](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1577972363174-4a780595-bb17-4a20-9a57-1ceede574376.jpeg)<br />

![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1577972313033-0f253167-e9a2-4756-b8bd-ed5660f24c2f.jpeg)



i, j  表示像素下标，xi,j 表示像素值，wk,n 表示第 n 层卷积卷积参数。 以上结构可以进行跨通道的信息融合。MLP 的参数也可以使用 BP 算法训练，与 CNN 高度整合；同时，1×1 卷积可以实现通道数的降维或者升维，1*1*n，如果 n 小于之前通道数，则实现了降维，如果 n 大于之前通道数，则实现了升维。<br />


(2) Global Average Pooling Layer

之前的卷积神经网络的最后会加一层全连接来进行分类，但是，全连接层具有非常大的参数量，导致模型非常容易过拟合。因此，全局平均池化的出现替代了全连接层。我们在最后一层 mlpconv 得到的特征图使用全局平均池化（也就是取每个特征图的平均值）进而将结果输入到 softmax 层中。

全局平均池化的优点

- 全局平均池化没有参数，可以避免过拟合产生。
- 全局平均池可以对空间信息进行汇总，因此对输入的空间转换具有更强的鲁棒性。

下图是全连接层和全局平均池化的一个对比图

![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1577970284799-b42251c9-a0d8-41e3-a4f0-83fad80a06e5.jpeg#align=left&display=inline&height=294&originHeight=294&originWidth=754&size=0&status=done&style=none&width=754)


<a name="wrCvL"></a>
### 整体网络架构


![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577973809330-40d83a19-b8c6-4606-8fa4-bca769a46e57.png)

NiN 网络由三个 mlpconv 块组成，最后一个mlpconv 后使用全局平均池化。用 Pytorch 代码实现如下：假设输入是 32×32×3 (即 3 通道，32 高宽大小的图片)



```python
import torch
import torch.nn as nn

class NiN(nn.Module):
    def __init__(self, num_classes):
        super(NiN, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )

    def forward(self, x):
        x = self.classifier(x)
        logits = x.view(x.size(0), self.num_classes)
        probas = torch.softmax(logits, dim=1)
        return logits, probas
```





### 结果分析

- CIFAR-10 测试数据集

![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1578014952393-1e3bc4e5-0c84-4e25-9926-a690d7ec42ce.jpeg#align=left&display=inline&height=238&originHeight=238&originWidth=665&size=0&status=done&style=none&width=665)

NIN + Dropout  得到 10.41% 的错误率，加上数据增强（平移和水平翻转）错误率降到了 8.81%

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578015109190-02bd540d-2c20-4bbd-8dd8-eaa5460eb8ba.png)<br />以上曲线对比了有无 Dropout 机制前200 epoch 的错误率，实验结果表明，引入Dropout机制，可以降低测试集 20% 的错误率

在 CIFAR-10 测试数据集上对比了全连接层和全局平均池化的效果

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578016269516-e923e9ef-c2ab-4162-9392-693bbccbab09.png)

- CIFAR-100 

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578015924846-3cfdb419-d7b2-4fb9-a1e6-8c1b586fa12a.png)<br />
<br />上面对比方法时提到了 maxout，这是是蒙特利尔大学信息与信息技术学院的几位大牛 2013 年在 ICML 上发表的一篇论文，有兴趣的可以看：[https://arxiv.org/abs/1302.4389](https://arxiv.org/abs/1302.4389)<br />
<br />

- Street View House Numbers (SVHN) 街景门牌号码数据集

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578016060275-426668da-fe02-495f-b5ef-8733907ac2c8.png)

- MNIST

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578016118457-1123e769-127f-484a-a168-4ecef2ba9a2f.png)



NiN + Dropout 效果在 MNIST 上效果比 maxout +Dropout 逊色了点


### 总结
<br />Network In Network 通过创新的创建 MLP 卷积层，提高了网络的非线性表达同时降低了参数量，用全局均值池化代替全连接层，极大的降低了参数量。

参考：

- [有三AI【模型解读】network in network 中的 1*1 卷积，你懂了吗](https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649029550&idx=1&sn=13a3f1e12815694c595b9ee88708af1a&chksm=871345d3b064ccc547637ad3daa56565c25c234686452228b052e10589740d697f55e8945fe9&scene=21#wechat_redirect)
- [http://teleported.in/posts/network-in-network/](http://teleported.in/posts/network-in-network/)
- [https://openreview.net/forum?id=ylE6yojDR5yqX](https://openreview.net/forum?id=ylE6yojDR5yqX)
- [https://d2l.ai/chapter_convolutional-modern/nin.html](https://d2l.ai/chapter_convolutional-modern/nin.html) 动手学深度学习
- [https://github.com/jiecaoyu/pytorch-nin-cifar10](https://github.com/jiecaoyu/pytorch-nin-cifar10)
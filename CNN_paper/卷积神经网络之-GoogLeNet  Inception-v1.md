
# GoogLeNet / Inception-v1

![GoogLeNet大纲](https://cdn.nlark.com/yuque/0/2020/png/653487/1580179958744-20c58659-c08e-4fac-a4f9-d87df0c7216e.png)


### 简介

论文地址：[https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)

Inception 是一个代号，是 Google 提出的一种深度卷积网络架构（PS：有一部电影的英文名就是它，中文名叫做盗梦空间）。

![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1579680360631-e06bd414-dc76-4819-bacb-d30754b81801.jpeg)

Inception 的第一个版本也叫 GoogLeNet，在 2014 年 ILSVRC（[ImageNet 大规模视觉识别竞赛](http://www.image-net.org/challenges/LSVRC/)）的图像分类竞赛提出的，它对比 ZFNet（2013 年的获奖者）和 AlexNet （2012 年获胜者）有了显着改进，并且与 VGGNet（2014 年亚军）相比错误率相对较低。是 2014 年的 ILSVRC 冠军。

Goog**LeNet**一词包含 **LeNet** ，这是在向第一代卷积神经网络为代表的 LeNet 致敬。

<a name="Ze27C"></a>
### 1×1 卷积

1×1 卷积最初是在 NiN 网络中首次提出的，使用它的目的是为了增加网络非线性表达的能力。

GoogLeNet 也使用了 1×1 卷积，但是使用它的目的是为了设计瓶颈结构，通过降维来减少计算量，不仅可以增加网络的深度和宽度，也有一定的防止过拟合的作用。

以下对比使用 1×1 卷积前后计算量的对比，看看 1×1 卷积是如何有效减少参数的。

![未使用 1×1 卷积](https://cdn.nlark.com/yuque/0/2020/png/653487/1579681695739-00dd223c-9ae0-4a7a-973b-e47357602c38.png)


![使用1×1卷积](https://cdn.nlark.com/yuque/0/2020/png/653487/1579681696282-f659be5a-1c1d-4d51-b930-53463dacef10.png)


![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580188089446-86b0ddcf-359b-4697-9641-8e340959bff4.png)

从上表可以看出，使用 1×1 卷积，参数量只有未使用 1×1 卷积的 5.3/112.9=4.7%，大幅度减少了参数量。


### Inception 模块

![Inception模块.png](https://cdn.nlark.com/yuque/0/2020/png/653487/1579784299324-f39ff7c5-7b28-4a55-8098-a161da744929.png)


上图中，图(a) 是不带 1×1 卷积的版本，不具备降维的作用，图(b) 是带 1×1 卷积的版本，具有降维的作用，可以降低参数量。

Inception 模块有四条通路，包括三条卷积通路和一条池化通路，具有不同的卷积核大小，不同卷积核大小可以提取不同尺度的特征。最后将不同通路提取的特征 concate 起来，不同通路得到的特征图大小是一致的。

<a name="1ELTL"></a>
### 总体架构

GoogLenet 网络的结构如下，总共有 22 层，主干网络都是全部使用卷积神经网络，仅仅在最终的分类上使用全连接层。

![GoogLeNet](https://cdn.nlark.com/yuque/0/2020/png/653487/1579683940296-04f1bdc0-48ab-47df-9475-fb2a4ea95a79.png)


可以在 GoogLeNet 看到多个 softmax 分支，网络越深，越影响梯度的回传，作者希望通过不同深度的分支增加梯度的回传，用于解决梯度消失问题，并提供一定的正则化，所以在训练阶段使用多个分支结构来进行训练，它们产生的损失加到总损失中，并设置占比权重为 0.3，但是这些分支结构在推理阶段不使用。它们的详细参数可以看下图的注释

![GoogLeNet分支结构](https://cdn.nlark.com/yuque/0/2020/png/653487/1580128207610-e2155191-5d80-4d07-b042-fe3f53f2b96d.png)

各层网络的具体参数见下表
![GoogLeNet 网络中各层参数的详细信息](https://cdn.nlark.com/yuque/0/2020/png/653487/1580127773867-66108a0e-3fbb-4912-9c72-2f677872c0a5.png)

<a name="qvtlW"></a>
### 代码实现
<a name="Xyrnx"></a>
#### Inception 模块

```python
class Inception(nn.Module):
    
    def __init__(self,in_ch,out_ch1,mid_ch13,out_ch13,mid_ch15,out_ch15,out_ch_pool_conv,auxiliary=False):
        # auxiliary 用来标记是否要有一条 softmax 分支
        super(Inception,self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch1,kernel_size=1,stride=1),
            nn.ReLU())
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_ch,mid_ch13,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch13,out_ch13,kernel_size=3,stride=1,padding=1),
            nn.ReLU())
        
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_ch,mid_ch15,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_ch15,out_ch15,kernel_size=5,stride=1,padding=2),
            nn.ReLU())
        
        self.pool_conv1 = nn.Sequential(
            nn.MaxPool2d(3,stride=1,padding=1),
            nn.Conv2d(in_ch,out_ch_pool_conv,kernel_size=1,stride=1),
            nn.ReLU())
        
        self.auxiliary = auxiliary
        
        if auxiliary:
            self.auxiliary_layer = nn.Sequential(
                nn.AvgPool2d(5,3),
                nn.Conv2d(in_ch,128,1),
                nn.ReLU())
        
    def forward(self,inputs,train=False):
        conv1_out = self.conv1(inputs)
        conv13_out = self.conv13(inputs)
        conv15_out = self.conv15(inputs)
        pool_conv_out = self.pool_conv1(inputs)
        outputs = torch.cat([conv1_out,conv13_out,conv15_out,pool_conv_out],1) # depth-wise concat
        
        if self.auxiliary:
            if train:
                outputs2 = self.auxiliary_layer(inputs)
            else:
                outputs2 = None
            return outputs, outputs2
        else:
            return outputs
```


<a name="2f4hT"></a>
### 实验结果

GoogLeNet 使用了多种方法来进行测试，从而提升精度，如模型集成，最多同时使用 7 个模型进行融合；多尺度测试，使用256、288、320、352等尺度对测试集进行测试；crop 裁剪操作，最多达到 144 个不同方式比例的裁剪。

在集成7个模型，使用 144 种不同比例裁剪方式下，在 ILSVRC 竞赛中 Top-5 降到 6.67，GoogLeNet 的表现优于之前的其他深度学习网络，并在 ILSVRC 2014 上获奖。

![GoogLeNet分类性能](https://cdn.nlark.com/yuque/0/2020/png/653487/1580129374230-9e75d304-e35e-46b9-adb1-7018912d9476.png)

参考:

- [https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7](https://medium.com/coinmonks/paper-review-of-googlenet-inception-v1-winner-of-ilsvlc-2014-image-classification-c2b3565a64e7)
- [https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py)
# 卷积神经网络之-Inception-v3



### 简介

论文地址：[https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)

Inception-v3 架构的主要思想是 factorized convolutions(分解卷积) 和 aggressive regularization(激进的正则化)

注：一般认为 Inception-v2 (BN  技术的使用)和 Inception-v3(分解卷积技术) 网络是一样的，只是配置上不同，那么就暂且本文所述的是 Inception-v3 吧。

### 设计原则

作者在文章中提出了 4 个设计网络的原则，虽然不能证明这些原则是有用的，但是也能为网络的设计提供一定的指导作用。笔者对这些原则理解也并不透彻，并且没有做过相关实验，所以以下的表述可能表述有不周到，请读者指正：

- 网络从输入到输出的过程中，应该避免维度过度压缩，输入到输出特征的维度应该缓缓变化，例如像金字塔那般。
- 获得更加明晰的特征，得到的网络训练起来更快。
- 空间聚合可以通过低维度嵌套得到，而不损失表示能力。
- 网络的宽度和深度要均衡。

### 分解卷积

分解卷积的主要目的是为了减少参数量，分解卷积的方法有：大卷积分解成小卷积；分解为非对称卷积；

#### 大卷积分解成小卷积
使用 2 个 3×3 卷积代替一个 5×5 卷积，可以减少 28% 的参数量，另外分解后多使用了一个激活函数（卷积层后面跟着激活函数，以前只有一个5×5 卷积，也就只有一个激活函数，现在有 2 个 3×3 卷积，也就有了 2 个激活函数），增加了非线性表达的能力，（VGGNet 也使用了相似的技术）分解示意图如下所示：

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580264650721-7f48871a-13a1-47fc-a2e4-e52620cf660a.png)

网络具体结构如下，简称为 **Module A**

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580279643832-8c2423bd-aa6d-4755-9c27-26d5b1cc789f.png)

那么经过分解，是否会对性能造成影响呢，作者做了实验，得到结果如下图，其中蓝色曲线是对分解的卷积使用 2 个 ReLU 激活（得到验证集 77.2% 的精度），红色曲线是对分解的卷积使用 Linear+ReLU 激活（得到验证集 76.2% 的精度），实验结果表明，**经过分解不会降低模型的 representation（表征）能力。**

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580216878946-1a4d1415-b7e6-4983-a1f2-5b9e630878a1.png)

#### 分解为非对称卷积
用 1 个 1×3 卷积和 1 个 3×1 卷积替换 3×3 卷积，这样可以减少 33% 的参数量

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580264637245-629e04e3-30bd-49e9-8098-f0511295204c.png)



具体结构如下：简称为 **Module B**

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580279728871-93c7589b-1b0c-4a22-8ed4-4933e498ad74.png)

其他的非对称分解卷积如下，简称为 **Module C**

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580264613305-f5c3ada7-db26-48f3-91c7-fd8788c1a55e.png)


### 辅助分类器（Auxiliary Classifier）

在 Inception v1 中，使用了 2 个辅助分类器，用来帮助梯度回传，以加深网络的深度，在  Inception v3 中，也使用了辅助分类器，但其作用是用作正则化器，这是因为，如果辅助分类器经过批归一化，或有一个dropout层，那么网络的主分类器效果会更好一些。这也间接的说明，**批归一化可以作为正则化器使用。**

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580278799263-ef3c63f0-eb8a-4b7a-8010-85c9a3088d1b.png)

### 有效的特征网格大小缩减（Grid Size Reduction）

传统上，卷积网络使用一些池化操作来减小特征图的网格大小。为避免表示瓶颈，在进行最大池化或平均池化之前，增大网络滤波器激活的维数。作者设计了一个结构，可以有效减少计算量和参数

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580279325758-b1bb1cd8-647e-496a-add3-92fe7a5bba1d.png)

其中左边的图是详细的结构，右边的图是结构简图

### Inception-v3 架构

![Inception-v3 Architecture.png](https://cdn.nlark.com/yuque/0/2020/png/653487/1580279551723-040637b5-b9f3-485c-91f5-159f54d73507.png)

该网络有 42 层，计算量比 GoogLeNet 高 2.5 倍，但是比 VGGNet 更高效，具体网络参数如下：

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580280157858-15c48dc5-bb01-41f2-b3b5-4a4214716e96.png)

注：图中的 **Module A、B、C **模型结构在前文已经说明

### 通过标签平滑来进行模型正则化

为了保证模型能够有较好的泛化能力，作者提出通过对标签进行平滑，原因是如果模型学会了对每个训练样本的真值标签都赋予充分最高概率，那么泛化能力就不能保证了。主要的工作是让真实标签的 logit 不要那么大。

### 实验结果

以下实验的数据集都是 ILSVRC 竞赛数据集
#### Inception 系列模型性能对比
![不同Inception性能对比](https://cdn.nlark.com/yuque/0/2020/png/653487/1580281086201-a84d184b-2f6a-4bbd-bcfe-3589584df1ec.png)

#### 不同网络性能对比

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580281266107-064379cc-c876-4867-8d70-989231b46a3d.png)
#### 模型集成结果
![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580281390078-2bddecf8-c702-4ad5-bc5a-e4c477a8ad7e.png)

可以看出，Inception-v3 在性能上有好的表现。在 **ILSVRC 2015 **竞赛上获得亚军，当年的冠军是 ResNet 。

参考:

- [https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c](https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c)
- [https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py) 
- [https://cloud.google.com/tpu/docs/inception-v3-advanced](https://cloud.google.com/tpu/docs/inception-v3-advanced)
- [https://github.com/Mycenae/PaperWeekly/blob/master/Inception-V3.md](https://github.com/Mycenae/PaperWeekly/blob/master/Inception-V3.md)
- [https://blog.ddlee.cn/posts/5e3f4a2c/](https://blog.ddlee.cn/posts/5e3f4a2c/)
- [https://mp.weixin.qq.com/s/mXhVMHBsxrQQf_MV4_7iaw](https://mp.weixin.qq.com/s/mXhVMHBsxrQQf_MV4_7iaw)
- [https://blog.csdn.net/u014061630/article/details/80383285](https://blog.csdn.net/u014061630/article/details/80383285)
- [https://blog.csdn.net/weixin_43624538/article/details/84963116](https://blog.csdn.net/weixin_43624538/article/details/84963116)




# 残差⽹络（RESNET）


ResNet论文网址：[https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)


残差神经网络(ResNet)是由微软研究院的何恺明、张祥雨、任少卿、孙剑等人提出的。ResNet 在2015 年的ILSVRC（ImageNet
Large Scale Visual Recognition Challenge）中取得了图像分类、检测、定位三个冠军。2016 年 CVPR 论文：《Deep Residual Learning for Image Recognition》就介绍了 ResNet，该论文截至当前(2020.1.3)已被引用超过 36500 次。

残差神经网络的主要贡献是发现了“退化现象（Degradation）”，并针对退化现象发明了
“快捷连接（Shortcut connection）”（或者跳过连接），极大的消除了深度过大的神经网络训练困难问题。神经网络的“深度”首次突破了100层、最大的神经网络甚至超过了1000层。

![ILSVRC 2015 图像分类排名](https://cdn.nlark.com/yuque/0/2020/png/653487/1578051767875-c496038f-67dd-4aed-8735-9913c62dfe2f.png)



### 从一个信念说起

在 2012 年的 ILSVRC 挑战赛中，AlexNet 取得了冠军，并且大幅度领先于第二名。由此引发了对 AlexNet 广泛研究，并让大家树立了一个信念——“越深网络准确率越高”。这个信念随着 VGGNet、Inception v1、Inception v2、Inception v3 不断验证、不断强化，得到越来越多的认可，但是，始终有一个问题无法回避，这个信念正确吗？<br />它是正确的，至少在理论上是正确的。

假设一个层数较少的神经网络已经达到了较高准确率，我们可以在这个神经网络之后，拼接一段恒等变换的网络层，这些恒等变换的网络层对输入数据不做任何转换，直接返回（_y_=_x_），就能得到一个深度较大的神经网络，并且，这个深度较大的神经网络的准确率等于拼接之前的神经网络准确率，准确率没有理由降低。

层数较多的神经网络，可由较浅的神经网络和恒等变换网络拼接而成，下图所示。

![层数较多的神经网络](https://cdn.nlark.com/yuque/0/2020/png/653487/1578051404377-cdbac1c9-24b2-49af-84b0-1eccc79dba69.png)



### 退化现象与对策

在讲退化这个概念之前先说一下梯度消失(Gradients Vanishing)和梯度爆炸 (Gradients Exploding) 这个概念。 也就是在训练神经网络的时候，导数或坡度有时会变得非常大，或者非常小，多个层以后梯度将以指数方式变大或者变小，这加大了训练的难度。

当网络很深时，很小的数乘起来将会变成 0（梯度消失），很大的数乘起来会变得非常大（梯度爆炸）

通过实验，ResNet随着网络层不断的加深，模型的准确率先是不断的提高，达到最大值（准确率饱和），然后随着网络深度的继续增加，模型准确率毫无征兆的出现大幅度的降低。以下曲线显示 20 层普通网络的训练误差和测试误差低于 56 层普通网络，这个现象与“越深的网络准确率越高”的信念显然是矛盾的、冲突的。ResNet团队把这一现象称为“退化（Degradation）”。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578052620348-7e9dfe8f-61bb-4948-a82e-a8db1530b4b2.png)

ResNet团队把退化现象归因为深层神经网络难以实现“恒等变换（_y_=_x_）”。乍一看，让人难以置信，原来能够模拟任何函数的深层神经网络，竟然无法实现恒等变换这么简单的映射了？

让我们来回想深度学习的起源，与传统的机器学习相比，深度学习的关键特征在于网络层数更深、非线性转换（激活）、自动的特征提取和特征转换，其中，非线性转换是关键目标，它将数据映射到高纬空间以便于更好的完成“数据分类”。随着网络深度的不断增大，所引入的激活函数也越来越多，数据被映射到更加离散的空间，此时已经难以让数据回到原点（恒等变换）。或者说，神经网络将这些数据映射回原点所需要的计算量，已经远远超过我们所能承受的。

退化现象让我们对非线性转换进行反思，非线性转换极大的提高了数据分类能力，但是，随着网络的深度不断的加大，我们在非线性转换方面已经走的太远，竟然无法实现线性转换。显然，在神经网络中增加线性转换分枝成为很好的选择，于是，ResNet 团队在 ResNet 模块中增加了快捷连接分枝，在线性转换和非线性转换之间寻求一个平衡。


### 残差网络

为了解决梯度消失 / 爆炸的问题，添加了一个跳过 / 快捷方式连接，将输入 x 添加到经过几个权重层之后的输出中，如下图所示：

![残差网络构建块](https://cdn.nlark.com/yuque/0/2020/png/653487/1578056505424-daf7133e-92a1-49bd-9421-9b4c0d0f4274.png)



输出为 H(x) = F(x) + x，权重层实际上是学习一种残差映射：F(x) = H(x) - x，即使权重层的梯度消失了，我们仍然始终具有标识 x 可以转移回较早的层。


### ResNet网络架构

按照这个思路，ResNet团队分别构建了带有“快捷连接（Shortcut Connection）”的 ResNet 构建块、以及降采样的ResNet构建块， 区别是降采样构建块的主杆分枝上增加了一个1×1的卷积操作，见下图。<br />![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578051430079-57380f1b-de9f-41b8-9de9-c90a462a483c.png)

下图展示了 34 层 ResNet 模型的架构图，仿照 AlexNet 的 8 层网络结构，我们也将 ResNet 划分成 8个构建层（Building Layer）。一个构建层可以包含一个或多个网络层、以及一个或多个构建块（如 ResNet构建块）。

![34层ResNet模型架构图（此图来源于《**TensorFlow深度学习实战大全**》）](https://cdn.nlark.com/yuque/0/2020/png/653487/1578051440850-da3437c2-93ef-43a0-a84e-7c8a90dc8d7e.png)

第一个构建层，由1个普通卷积层和最大池化层构建。<br />第二个构建层，由3个残差模块构成。<br />第三、第四、第五构建层，都是由降采样残差模块开始，紧接着3个、5个、2个残差模块。

 <br />ResNet 各个版本的网络架构如下所示：

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578057041724-980284d2-3365-498f-acd0-95465c9bdfb6.png) 


### 实验结果
 <br />一个概念：**10 -crops**: 取图片（左上，左下，右上，右下，正中）以及它们的水平翻转。这 10 个 crops 在 CNN 下的预测输出取平均作为最终预测结果。

- 图像分类

1. ILSVRC 

![10-crop 下的检测结果
](https://cdn.nlark.com/yuque/0/2020/png/653487/1578057138766-02f58afd-a948-4b0e-b432-2ca96457d07b.png)

其中 plain-34 就是普通的卷积叠加起来的网络，把 ResNet 深度一直加深，错误率也一直降低

![10-Crop + 多尺度全卷积](https://cdn.nlark.com/yuque/0/2020/png/653487/1578057938536-a5f5c7f5-5f4c-49ba-bda5-f4409c61199a.png)


![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578058259921-ed23f316-80dd-4fa8-93de-014da64b52a2.png)

10-Crop + 多尺度全卷积 + 6 个模型融合，错误率降到了 3.57%

2. CIFAR-10 数据集

作者们干脆把网络深度加到了 1202 层，此时网络优化起来也没有那么困难，即仍可以收敛，但是，当层数从 110 增加到 1202 时，发现错误率从 6.43％增加到 7.93％，这个问题仍悬而未决！
![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578058345925-55c48b2a-9562-48b7-978d-8d375b73b4a8.png)

- 目标检测

PASCAL VOC 2007/2012 数据集 mAP (%) 测试结果如下：

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578058474823-95981081-7393-425c-a462-52fe5bb4e5ea.png)

MS COCO 数据集 mAP (%) 测试结果如下：
![](https://cdn.nlark.com/yuque/0/2020/png/653487/1578058522438-b46c27ec-8ce8-4d2b-9b7e-adfbce74dfd3.png)

通过将 ResNet-101 应用于 Faster R-CNN [3-4]，ResNet 可以获得比 VGG-16 更好的性能



参考资料：

- [https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- [https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8](https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8)
- 吴恩达视频_梯度消失/爆炸：https://www.bilibili.com/video/av48340026?p=10
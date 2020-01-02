# 卷积神经网络之 - ZFNet

更多内容请关注『机器视觉 CV』公众号

### 说在前面

（貌似江湖上有两篇 ZFNet 的论文，也即：Visualizing and Understanding Convolutional Networks ）最新的请见论文地址：[https://arxiv.org/pdf/1311.2901.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1311.2901.pdf)

这两篇还是有细微差别的，比如以下两张图，版本一的图 (e) 没有在论文展示出来，但是却在题注上出现了，我猜作者应该是发现了这个问题，进而提交了版本二的论文，大家还是以官方地址的论文为主吧！



版本一：

![图中没有 (e) 题注却出现了 (e)](https://pic3.zhimg.com/80/v2-fefb9099c9b0c04387a199060b10ae52_hd.png)



版本二：官方地址中的论文

![](https://pic3.zhimg.com/80/v2-61ab87fbe8e6fafa85b7d8f1ede946b2_hd.png)



### 背景知识

ImageNet 数据集是超过 1500 万张带标签的高分辨率图像的数据集，包含大约 22,000 个类别。ILSVRC 使用 ImageNet 的子集，其中包含 1000 个类别中的大约 1000 个图像。总共大约有 130 万张训练图像，5,000 张验证图像和 100,000 张测试图像。

![2013ILSVRC 竞赛错误率](https://pic1.zhimg.com/80/v2-7af32bbb3d05f4544b33c5f7bbe2aed8_hd.png)



ZFNet 是由 Matthew D.Zeiler 和 Rob Fergus 在 AlexNet 基础上提出的大型卷积网络，在 2013 年 ILSVRC 图像分类竞赛中以 11.19% 的错误率获得冠军。ZFNet 实际上并不是 ILSVLC 2013 的赢家。相反，当时刚刚成立的初创公司 Clarifai 是 ILSVLC 2013 图像分类的赢家。又刚刚好 Zeiler 是 Clarifai 的创始人兼首席执行官，而 Clarifai 对 ZFNet 的改动较小，故认为 ZFNet 是当年的冠军



### ZFNet

### 简介

ZFNet 对如何解释卷积神经网络的表现如此出色以及如何对卷积神经网络进行改进提出疑问。在论文中，介绍了一种可视化的技术，深入了解卷积神经网络中间层的功能和分类器的操作，使用可视化技术，可以帮助我们找到较好的模型。作者还进行消融研究（ablation study）即如果我们删除模型的某些组件，它会对模型有什么影响。

ZFNet 实际上是微调（fine-tuning）了的 AlexNet，并通过反卷积（Deconvolution）的方式可视化各层的输出特征图，进一步解释了卷积操作在大型网络中效果显著的原因。

### 反卷积

![](https://pic2.zhimg.com/v2-0160c88e28c864d683fb2970105384a5_b.png)



上图：反卷积层（左）与卷积层（右）相连。反卷积网络将从下面的层重建一个近似版本的卷积网络特征。下图：反卷积网络中使用 switch 反池化操作的示意图，switch 记录卷积网络池化时每个池化区域（彩色区域）中局部最大值的位置。黑 / 白条在特征图中表示负 / 正激活。

卷积操作的标准流程是：卷积层 + 激活函数 + 池化层，图像经过上述步骤以后，得到特征图，为了可视化深层特征，我们需要对卷积进行逆过程操作，以便可以进行可视化。

最大池化是不可逆的操作，但是我们通过记录最大值所在的位置来近似最大池化的逆操作。同时，在卷积的流程中使用了激活函数，所以进行反卷积时，也需要加上激活函数

![](https://pic2.zhimg.com/v2-d9a4632ddbc0884fde22f384ebf1f6f5_b.png)

### 卷积网络可视化

- Layer1 & Layer2

  

![](https://pic2.zhimg.com/v2-4a791620ad21632eda3882858d7e7b59_b.png)



第 1 层的滤波器混合了极高和极低的频率信息，几乎没有覆盖中频。如果没有中频，就会产生连锁效应，即深度特征只能从极高和极低的频率信息中学习。

第 2 层可视化呈现出由第 1 层卷积中使用的大步幅 4 引起的混叠伪影

为了解决这些问题，作者们进行了以下改进（i）将第一层滤波器尺寸从 11x11 缩小到 7x7，并且（ii）使卷积的步幅由 4 改为 2。

改进可视化图对比
(b) 改进前 (即 AlexNet) 第 1 层可视化
(c) 改进后 (即 ZFNet) 第 1 层可视化
(d) 改进前 (即 AlexNet) 第 2 层可视化。(e) 改进后 (即 ZFNet) 第 2 层可视化。

![](https://pic4.zhimg.com/v2-de9b9240a18e7566c3ed7d79dfd4c2b7_b.png)

这种新架构在第 1 层和第 2 层特征中保留了更多信息，并且提高了分类性能

- Layer3

第 3 层具有更复杂的不变性，捕获相似的纹理（例如网格图案（第 1 行，第 1 列）; 文本（第 2 行，第 4 列））『**以下使用 R 代表行，C 代表列**』

![](https://pic1.zhimg.com/v2-85f50ac7e57acf0fdd243be7a245dde0_b.png)

- Layer4 & Layer5

第 4 层显示出显着的变化，并且更具有特定类别：狗脸 (R1，C1) 鸟的腿 (R4，C2)。

第 5 层显示具有显着姿势变化的整个对象，例如， 键盘（R1，C11）和狗（R4）。

![](https://pic2.zhimg.com/v2-d245d6132a73f3bf2b9549a3c1db0f99_b.png)



### 基于可视化结果的 AlexNet 的微调



对 AlexNet 改进的地方就是上文提到的（i）将第一层滤波器尺寸从 11x11 缩小到 7x7，并且（ii）使第一层卷积的步幅由 4 改为 2。

![](https://pic4.zhimg.com/v2-2922515e65bb4c9589d39ccdb71f10eb_b.png)



### 实验结果

- 下表为 ImageNet 2012/2013 分类错误率，* 表示在 ImageNet 2011 和 2012 训练集上都经过训练的模型。

![](https://pic4.zhimg.com/v2-6632d855b4e1ea138feae0b01c16ab27_b.png)

使用 AlexNet，Top-5 验证错误率为 18.1％。通过使用 ZFNet，Top-5 验证错误率为 16.5％。我们可以得出结论，基于可视化的修改是必不可少的。通过使用（a）中的 5 个 ZFNet 和（b）中的 1 个修改后的 ZFNet，Top-5 验证错误率为 14.7％。

- Caltech-101 数据集测试结果

![](https://pic3.zhimg.com/v2-fb92a943309872dc3d1c184b87c81c06_b.png)

- Caltech 256 数据集测试结果

![](https://pic2.zhimg.com/v2-c192a84963d303d6ada234748059a35d_b.png)

![](https://pic4.zhimg.com/v2-1de6d43c0a8b286022d290ad8195ab1b_b.png)

- PASCAL 2012 数据集测试结果

![](https://pic3.zhimg.com/v2-c4660e512cf99fb202e13a918a240052_b.png)

从上表中可以看出，不进行预训练（即从头开始训练 ZFNet）的准确性较低。通过在预训练的 ZFNet 之上进行训练，准确性很高。这意味着训练过的滤波器可以推广到不同的图像，而不仅仅是 ImageNet 的图像。

对于 PASCAL 2012 来说，PASCAL 图像可以包含多个对象，与 ImageNet 中的对象相比与自然有很大的不同。因此，精确度稍低一些，但仍然可以与最先进的方法相媲美。



### 消融研究

![](https://pic3.zhimg.com/v2-a5edfc78687b4f276e75d771ab43677e_b.png)

对于去除或调整层的消融研究。修改后的 ZFNet 可以获得 Top-5 16.0％ 的验证错误率 。



### 结论

虽然之前的研究只能观察到浅层特征，但本文提供了一种有趣的方法来观察像素域中的深层特征。



通过逐层可视化卷积网络，ZFNet 可以调整各层的超参数，例如过滤器大小或 AlexNet 的步幅，并成功降低错误率。

- [https://mc.ai/paper-review-of-zfnet-the-winner-of-ilsvlc-2013-image-classification/](https://link.zhihu.com/?target=https%3A//mc.ai/paper-review-of-zfnet-the-winner-of-ilsvlc-2013-image-classification/)
- [https://pechyonkin.me/architectures/zfnet/](https://link.zhihu.com/?target=https%3A//pechyonkin.me/architectures/zfnet/)
- [https://blog.csdn.net/C_chuxin/article/details/82475913](https://link.zhihu.com/?target=https%3A//blog.csdn.net/C_chuxin/article/details/82475913)
- [https://github.com/bigcindy/ZFNet](https://link.zhihu.com/?target=https%3A//github.com/bigcindy/ZFNet) 翻译文

![](https://pic4.zhimg.com/v2-31667898f5a052ce544ecb27b94338b7_r.jpg)
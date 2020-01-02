# 卷积神经网络之-ZFNet

<a name="PZeRx"></a>

### 说在前面

（貌似江湖上有两篇 ZFNet 的论文，也即：Visualizing and Understanding Convolutional Networks ）最新的请见论文地址：[https://arxiv.org/pdf/1311.2901.pdf](https://arxiv.org/pdf/1311.2901.pdf)

这两篇还是有细微差别的，比如以下两张图，版本一的 (e) 没有在论文展示出来，但是却在题注上出现了，我猜作者应该是发现了这个问题，进而提交了版本二的论文，大家还是以官方地址的论文为主吧！

版本一：

![图中没有 (e) 题注却出现了 (e)](https://cdn.nlark.com/yuque/0/2020/png/653487/1577927486184-54b116f5-38d3-4303-9166-ff1eb19b194d.png#align=left&display=inline&height=405&name=image.png&originHeight=405&originWidth=1067&size=382203&status=done&style=none&width=1067)

版本二：官方地址中的论文

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577927466150-57960e2f-2792-403f-a28e-ca07ca809f88.png#align=left&display=inline&height=438&name=image.png&originHeight=438&originWidth=1123&size=557514&status=done&style=none&width=1123)

<a name="0nqDw"></a>
### 背景知识

ImageNet 数据集是超过 1500 万张带标签的高分辨率图像的数据集，包含大约 22,000 个类别。ILSVRC 使用 ImageNet 的子集，其中包含 1000 个类别中的大约 1000 个图像。总共大约有 130 万张训练图像，5,000 张验证图像和 100,000 张测试图像。

![2013ILSVRC 竞赛错误率](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1577881866080-0a685a70-15f7-4c0f-8f2f-27fa3a9c2aff.jpeg#align=left&display=inline&height=482&name=2013ILSVRC%20%E7%AB%9E%E8%B5%9B%E9%94%99%E8%AF%AF%E7%8E%87&originHeight=482&originWidth=773&size=0&status=done&style=none&width=773)
<br />ZFNet是由 Matthew D.Zeiler 和 Rob Fergus 在 AlexNet 基础上提出的大型卷积网络，在 2013年 ILSVRC 图像分类竞赛中以 11.19% 的错误率获得冠军。ZFNet 实际上并不是 ILSVLC 2013 的赢家。相反，当时刚刚成立的初创公司 Clarifai 是 ILSVLC 2013 图像分类的赢家。又刚刚好 Zeiler 是 Clarifai 的创始人兼首席执行官，而 Clarifai 对 ZFNet 的改动较小，故认为 ZFNet 是当年的冠军

<a name="lPQax"></a>
### ZFNet

<a name="xGXoK"></a>
#### 简介

ZFNet 对如何解释卷积神经网络的表现如此出色以及如何对卷积神经网络进行改进提出疑问。在论文中，介绍了一种可视化的技术，深入了解卷积神经网络中间层的功能和分类器的操作，使用可视化技术，可以帮助我们找到较好的模型。作者还进行消融研究（[ablation study](http://qingkaikong.blogspot.com/2017/12/what-is-ablation-study-in-machine.html)）即如果我们删除模型的某些组件，它会对模型有什么影响。

ZFNet实际上是微调（fine-tuning）了的AlexNet，并通过反卷积（Deconvolution）的方式可视化各层的输出特征图，进一步解释了卷积操作在大型网络中效果显著的原因。

<a name="5dwb3"></a>
#### 反卷积


![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577882421765-511414be-cf8b-4d7e-a4dc-33fe6c84c9ca.png#align=left&display=inline&height=493&name=image.png&originHeight=658&originWidth=719&size=141635&status=done&style=none&width=539)<br />上图：反卷积层（左）与卷积层（右）相连。反卷积网络将从下面的层重建一个近似版本的卷积网络特征。下图：反卷积网络中使用 switch 反池化操作的示意图，switch记录卷积网络池化时每个池化区域（彩色区域）中局部最大值的位置。黑/白条在特征图中表示负/正激活。

卷积操作的标准流程是：卷积层 + 激活函数 + 池化层，图像经过上述步骤以后，得到特征图，为了可视化深层特征，我们需要对卷积进行逆过程操作，以便可以进行可视化。

最大池化是不可逆的操作，但是我们通过记录最大值所在的位置来近视最大池化的逆操作。同时，在卷积的流程中使用了激活函数，所以进行反卷积时，也需要加上激活函数<br>
![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1577882547691-7530cd49-ab5e-4dd0-b382-825b2306c1a8.jpeg#align=left&display=inline&height=353&name=&originHeight=470&originWidth=836&size=0&status=done&style=none&width=627)


<a name="mx4IT"></a>
#### 卷积网络可视化

- Layer1 & Layer2
<br>![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577883061125-9986b44f-77ae-4dd7-87a9-7112704adefb.png#align=left&display=inline&height=483&name=image.png&originHeight=483&originWidth=1261&size=935353&status=done&style=none&width=1261)


第 1 层的滤波器混合了极高和极低的频率信息，几乎没有覆盖中频。如果没有中频，就会产生连锁效应，即深度特征只能从极高和极低的频率信息中学习。

第 2 层可视化呈现出由第 1 层卷积中使用的大步幅4引起的混叠伪影

为了解决这些问题，作者们进行了以下改进（i）将第一层滤波器尺寸从11x11缩小到7x7，并且（ii）使卷积的步幅由 4 改为 2。

改进可视化图对比 <br />(b) 改进前(即 AlexNet)第 1 层可视化<br />(c) 改进后(即 ZFNet)第 1 层可视化 <br />(d) 改进前(即 AlexNet)第 2 层可视化。<br />(e) 改进后(即 ZFNet)第 2 层可视化。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577925699700-53554241-420f-4059-b618-a03d24e3b855.png#align=left&display=inline&height=440&name=image.png&originHeight=440&originWidth=1154&size=794991&status=done&style=none&width=1154)

这种新架构在第 1 层和第 2 层特征中保留了更多信息，并且提高了分类性能

- Layer3 

第 3 层具有更复杂的不变性，捕获相似的纹理（例如网格图案（第 1 行，第 1 列）;文本（第 2 行，第 4 列））『**以下使用 R 代表行，C 代表列**』

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577883824401-f45be289-d13b-488f-b1a6-1c5e6a4cba19.png#align=left&display=inline&height=480&name=image.png&originHeight=480&originWidth=1294&size=1171519&status=done&style=none&width=1294)

- Layer4 & Layer5

第 4 层显示出显着的变化，并且更具有特定类别：狗脸 (R1，C1)  鸟的腿 (R4，C2)。

第5层显示具有显着姿势变化的整个对象，例如， 键盘（R1，C11）和狗（R4）。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577884070818-b8eb6eed-330b-43c5-92a1-2a2481d280d7.png#align=left&display=inline&height=612&name=image.png&originHeight=612&originWidth=990&size=1195292&status=done&style=none&width=990)

<a name="VbIta"></a>
#### 基于可视化结果的 AlexNet 的微调

对  AlexNet 改进的地方就是上文提到的（i）将第一层滤波器尺寸从11x11缩小到7x7，并且（ii）使第一层卷积的步幅由 4 改为 2。



![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1577884320819-f65523b9-45ca-4b63-8e49-9f54ad26899e.jpeg#align=left&display=inline&height=533&originHeight=533&originWidth=932&size=0&status=done&style=none&width=932)


<a name="DIQO0"></a>
### 实验结果

<br/>
- 下表为 ImageNet 2012/2013 分类错误率，* 表示在ImageNet 2011和2012训练集上都经过训练的模型。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577886723263-5c38cda5-9969-4557-ae7e-7c2117804244.png#align=left&display=inline&height=559&name=image.png&originHeight=559&originWidth=653&size=145946&status=done&style=none&width=653)

使用 AlexNet，Top-5 验证错误率为 18.1％。<br />通过使用 ZFNet，Top-5 验证错误率为 16.5％。我们可以得出结论，基于可视化的修改是必不可少的。<br />通过使用（a）中的 5 个 ZFNet 和（b）中的 1 个修改后的 ZFNet，Top-5 验证错误率为 14.7％。

- Caltech-101 数据集测试结果

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577887551822-d4a987cc-0178-467a-be24-f7c2ce8a8ba9.png#align=left&display=inline&height=297&name=image.png&originHeight=297&originWidth=820&size=66950&status=done&style=none&width=820)

- Caltech 256 数据集测试结果

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577887566881-c42953e6-f90f-4738-8fca-d96b1cb2c413.png#align=left&display=inline&height=312&name=image.png&originHeight=312&originWidth=1009&size=87560&status=done&style=none&width=1009)

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577887641663-d5418324-e3d3-4cc6-95d1-45854087fd9b.png#align=left&display=inline&height=444&name=image.png&originHeight=444&originWidth=564&size=57405&status=done&style=none&width=564)

- PASCAL 2012 数据集测试结果

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1577887662255-47257302-fc84-44d5-abce-64f16e9ff4f1.png#align=left&display=inline&height=546&name=image.png&originHeight=546&originWidth=1030&size=219577&status=done&style=none&width=1030)

从上表中可以看出，不进行预训练（即从头开始训练 ZFNet）的准确性较低。通过在预训练的 ZFNet 之上进行训练，准确性很高。这意味着训练过的滤波器可以推广到不同的图像，而不仅仅是 ImageNet 的图像。

对于PASCAL 2012 来说，PASCAL图像可以包含多个对象，与 ImageNet 中的对象相比与自然有很大的不同。因此，精确度稍低一些，但仍然可以与最先进的方法相媲美。


<a name="ZdpmL"></a>
### 消融研究


![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1577887016653-f6883954-6132-4169-bad7-3d823764d031.jpeg#align=left&display=inline&height=340&originHeight=340&originWidth=932&size=0&status=done&style=none&width=932)


对于去除或调整层的消融研究。修改后的 ZFNet 可以获得 Top-5 16.0％ 的验证错误率 。

<a name="Kya6z"></a>
### 结论

虽然之前的研究只能观察到浅层特征，但本文提供了一种有趣的方法来观察像素域中的深层特征。

通过逐层可视化卷积网络，ZFNet 可以调整各层的超参数，例如过滤器大小或 AlexNet 的步幅，并成功降低错误率。


参考：

- [https://mc.ai/paper-review-of-zfnet-the-winner-of-ilsvlc-2013-image-classification/](https://mc.ai/paper-review-of-zfnet-the-winner-of-ilsvlc-2013-image-classification/)
- [https://pechyonkin.me/architectures/zfnet/](https://pechyonkin.me/architectures/zfnet/)
- [https://blog.csdn.net/C_chuxin/article/details/82475913](https://blog.csdn.net/C_chuxin/article/details/82475913)
- [https://github.com/bigcindy/ZFNet](https://github.com/bigcindy/ZFNet) 翻译文
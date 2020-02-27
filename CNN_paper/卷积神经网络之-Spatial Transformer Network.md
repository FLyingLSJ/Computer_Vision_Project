# Spatial Transformer Network

论文地址：[https://arxiv.org/abs/1506.02025](https://arxiv.org/abs/1506.02025)

### 简述

Google DeepMind 出品的论文（alpha 狗就是他家的），STN 网络可以作为一个模块嵌入任何的网络，它有助于选择目标合适的区域并进行尺度变换，可以简化分类的流程并且提升分类的精度。

CNN 虽然具有一定的不变性，如平移不变性，但是其可能不具备某些不变性，比如：缩放不变性、旋转不变性。某些 CNN 网络学会对不同尺度的图像进行识别，那是因为训练的图像中就包含了不同尺度的图像，而不是CNN具有缩放不变性。

研究者认为，既然某些网络可能隐式的方式学会了某些变换，如缩放、平移等，那为什么不直接通过显式的方式让网络学会变换呢？所以学者们提出了 STN 网络来帮助网络学会对图像进行变换，帮助提升网络的性能。


### 空间变换知识

该论文主要涉及三种变换，分别是仿射变换、投影变换、薄板样条变换（Thin Plate Spline Transform）。

#### 仿射变换

仿射变换，又称仿射映射，是指在几何中，对一个向量空间进行一次线性变换并接上一个平移，变换为另一个向量空间。

变换的公式是


$$\left[\begin{array}{l} {\vec{y}} \\ {1} \end{array}\right]=\left[\begin{array}{cc} {A} & {\vec{b}} \\ {0, \dots, 0} & {1} \end{array}\right]\left[\begin{array}{l} {\vec{x}} \\ {1} \end{array}\right]$$


变换的方式包括 Translate（平移）、Scale（缩放）、Rotate（旋转）、Shear（裁剪）等方式，将公式中的矩阵 A 和向量 b 更换成下面的数，就可以进行对应方式的变换。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1579533626504-6bcd50cd-b67c-4884-a970-5a6df230e03f.png)



#### 投影变换

投影变换是仿射变换的一系列组合，但是还有投影的扭曲，投影变换有几个属性：1) 原点不一定要映射到原点 2) 直线变换后仍然是直线，但是一定是平行的 3) 变换的比例不一定要一致。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1579534064964-6ef0da8c-01f3-496b-9612-c8d20be31197.png)


#### 薄板样条变换  (TPS)

薄板样条函数 (TPS) 是一种很常见的插值方法。因为它一般都是基于 2D 插值，所以经常用在在图像配准中。在两张图像中找出 N 个匹配点，应用 TPS 可以将这 N 个点形变到对应位置，同时给出了整个空间的形变 (插值)。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1579605536397-9267dd68-1aa1-4acd-b6e7-0a3c91d6d5e1.png)

### STN 网络
STN 网络模型如下所示，包含三个部分：定位网络（Localisation network）、网格生成器（Grid generator）、采样器（Sampler）。

![STN网络结构图](https://cdn.nlark.com/yuque/0/2020/png/653487/1579605937220-cbbb99f6-549f-4d86-9d4e-e809e34ee30a.png)

#### Localisation network

Localisation network 用来生成仿射变换的系数，输入 U(可以是图片，也可以是特征图)是 C 通道，高 H，宽 W 的数据，输出是一个空间变换的系数$\theta$，$\theta$ 的维度大小根据变换类型而定，如果是仿射变换，则是一个 6 维的向量。

#### Grid generator

网格生成器，就是根据上面生成的 $\theta$ 参数，对输入进行变换，这样得到的就是原始图像或者特征图经过平移、旋转等变换的结果，转换公式如下：

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1579611053846-10c66701-d081-4b7f-9721-7d2277529298.png)

#### Sampler

根据 Grid generator 得到的结果，从中生成一个新的输出图片或者特征图 V，用于下一步操作

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1579610949219-96b5283d-11e6-419f-a985-bece6a47f059.png)



### 实验结果

#### MNIST

![不同模型，使用不同变换下 MNIST 数据的测试误差](https://cdn.nlark.com/yuque/0/2020/png/653487/1579612025379-59d614b8-8f25-470c-91f8-7d06ba5ff968.png)

**注意：上面的 FCN 指的是没有卷积的全连接网络，而不是全卷积网络**

从上面可以看出：ST-FCN 优于 FCN，ST-CNN 优于 CNN；ST-CNN 始终优于 ST-FCN。

![](https://cdn.nlark.com/yuque/0/2020/gif/653487/1579532097968-22fe2174-fa23-48ac-b0c5-5efaf43f5622.gif)

#### SVHN（街景门牌号）
![](https://cdn.nlark.com/yuque/0/2020/png/653487/1579612766479-1a002fb6-c13e-444b-91ec-346d458071ad.png)

#### 细粒度分类数据集（CUB-200-2011）

在细粒度数据集中，作者在网络中并行使用了多个 STN 网络，如下图，使用的是 2 个 STN 网络并行
![](https://cdn.nlark.com/yuque/0/2020/png/653487/1579612891563-562efa00-715e-4c5c-9375-905e4b8aff9b.png)

![在 CUB-200-2011 鸟类数据集上的测试精度](https://cdn.nlark.com/yuque/0/2020/png/653487/1579613189881-8e5b91ec-9c81-483e-a548-1af66d5e6af4.png)


可以看出，使用多个 STN 并行的网络，可以使精度达到不错的效果，4 个 STN 并行的网络效果更好。



实现代码

- PyTorch 框架实现：[https://github.com/fxia22/stn.pytorch](https://github.com/fxia22/stn.pytorch)
- [https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html) PyTorch1.4 支持STN
- lua 语言：[https://github.com/qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd)

参考资料：

- [https://towardsdatascience.com/review-stn-spatial-transformer-network-image-classification-d3cbd98a70aa](https://towardsdatascience.com/review-stn-spatial-transformer-network-image-classification-d3cbd98a70aa)
- [https://drive.google.com/file/d/0B1nQa_sA3W2iN3RQLXVFRkNXN0k/view](https://drive.google.com/file/d/0B1nQa_sA3W2iN3RQLXVFRkNXN0k/view) 实验效果视频
- [https://www.youtube.com/watch?v=SoCywZ1hZak](https://www.youtube.com/watch?v=SoCywZ1hZak) 李弘毅讲 STN 网络

![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1579531785033-827bb0cb-3ed4-4349-badc-ad90935cfeb6.jpeg#align=left&display=inline&height=8948&name=&originHeight=8948&originWidth=1600&size=0&status=done&style=none&width=1600)


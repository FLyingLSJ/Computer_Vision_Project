# BN-Inception / Inception-v2
![大纲](https://cdn.nlark.com/yuque/0/2020/png/653487/1580196109771-75b7b945-1a6b-4666-9ae6-5c7de34ad3e6.png)

### 简介

论文地址：[https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

Inception 的第二个版本也称作  BN-Inception，该文章的主要工作是引入了深度学习的一项重要的技术 **Batch Normalization (BN) 批处理规范化**。
BN 技术的使用，使得数据在从一层网络进入到另外一层网络之前进行规范化，可以获得更高的准确率和训练速度

题外话：BN-Inception 在 ILSVRC 竞赛数据集上分类错误率是 4.8%，但是该结果并没有提交给 ILSVRC 官方。

### 为什么需要 BN 技术？

BN 技术可以减少参数的尺度和初始化的影响，进而可以使用更高的学习率，也可以减少 Dropout 技术的使用

#### BN 有效性

网络的输入输出表达式一般表示为：$Y=F(W·X+b)$ ，其中 F 是 sigmoid 函数，如下图所示，蓝色虚线是 sigmoid 函数，橙色曲线是 sigmoid 函数的导数。从中可以看出，sigmoid 函数在两端容易使导数为 0，而且随着网络深度的加深，这种影响程度更严重，会导致训练速度变慢。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580193762618-7d8b1370-fbf1-4b14-91ba-41da35867a27.png)
如果将激活函数换成 ReLU(x)=max(x,0) 激活（见下图），可以解决 sigmoid 存在的问题，但是使用 Relu 也需要细心的设置学习率和进行参数初始化。


![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580194025644-05e3d94d-e2b3-455c-ae1e-8e1bac63d220.png)

随着训练的不断进行，数据的分布保持不变对训练是有利的，使用BN前后对训练的影响可以对照下图

![使用BN前后对训练的影响.png](https://cdn.nlark.com/yuque/0/2020/png/653487/1580194353960-d0c13e56-73d7-44fb-8718-2956ea7747fd.png)

### BN 原理

![Batch Normalization原理](https://cdn.nlark.com/yuque/0/2020/png/653487/1580194628611-3f9e89cf-5d40-460b-bd26-ea4c80ec3e94.png)

Batch Normalization  中的 batch 就是批量数据，即每一次优化时的样本数目，通常 BN 网络层用在**卷积层后**，用于重新调整数据分布。假设神经网络某层一个 batch 的输入为 X=[x1,x2,...,xn]，其中 xi 代表一个样本，n 为 batch size。步骤如下：

- 首先求解一个 batch 数据的均值 $\mu_{\mathcal{B}}$
- 求解一个 batch 的方差 $\sigma_{\mathcal{B}}^{2}$
- 然后对每一个数据进行规范化 $\widehat{x}_{i} \leftarrow \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}$$，$$\epsilon$是为了防止分母为 0
- 并使用其他可学习的参数 γ 和 β 进行缩放和平移，这样可以变换回原始的分布，实现恒等变换



最终得到的输出表达式是 **Y=F(BN(W ⋅ X+b))**

在测试阶段，我们利用 BN 训练好模型后，我们保留了每组 mini-batch 训练数据在网络中每一层的$\mu_{\mathcal{batch}}$  与 $\sigma_{\mathcal{batch}}^{2}$ 。此时我们使用整个样本的统计量来对 Test 数据进行归一化。

### 实验结果
#### MNIST 数据集
![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580196139481-56130c6d-a4b7-40ae-a761-90de375b4160.png)
在 MNIST 数据上使用一个简单的网络比较使用BN技术和未使用BN技术训练精度的差异，如上图(a)；上图 (b, c) 代表未使用 BN 技术和使用 BN 技术输入数据的分布，可以看出，使用 BN 技术，输入数据的分布更加稳定。

#### ILSVRC 数据集 

将 BN 运用到 GoogLeNet 网络上，同时将 Inception 模块中的 5×5 卷积替换成 2 个 3×3 卷积，将 5x5 卷积分解为两个 3x3 卷积运算，以提高计算速度。虽然这看似违反直觉，但 5x5 卷积比 3x3 卷积多 2.78 倍的参数量。因此，堆叠两个 3x3 卷积实际上可以提高性能。

![](https://cdn.nlark.com/yuque/0/2020/jpeg/653487/1580197332006-2f346b00-f75e-4e15-8f22-054a145958ac.jpeg)

在数据集 ILSVRC 上，使用 BN 技术并设计使用不同参数的 Inception 的网络，对比其精度，结果如下：使用 BN 技术，可以显著提高训练速度；对比 BN-×5 和 BN-×30，可以观察到，使用大的学习率可以提高训练速度。

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580197792197-e3d02b45-dddd-4969-8f46-a7976dc5aae0.png)
#### 与其他网络性能对比

![](https://cdn.nlark.com/yuque/0/2020/png/653487/1580198497421-9d4e1f26-0d1e-4324-a9f8-973e9f42b1de.png)

参考：

- [https://medium.com/@sh.tsang/review-batch-normalization-inception-v2-bn-inception-the-2nd-to-surpass-human-level-18e2d0f56651](https://medium.com/@sh.tsang/review-batch-normalization-inception-v2-bn-inception-the-2nd-to-surpass-human-level-18e2d0f56651)
- [https://mp.weixin.qq.com/s/Tuwg070YiXp5Rq4vULCy1w](https://mp.weixin.qq.com/s/Tuwg070YiXp5Rq4vULCy1w)
- [https://zhuanlan.zhihu.com/p/34879333](https://zhuanlan.zhihu.com/p/34879333)
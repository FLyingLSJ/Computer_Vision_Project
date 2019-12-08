数据标注工具  https://github.com/tzutalin/labelImg

**<u>请注意图片的格式，本项目默认是 jpg 格式，这个关系到其他脚本的处理</u>**



如果需要根据自己的数据生成合适的 Anchor 可以参考 `kmeans-anchor-boxes` 文件夹下的说明

### 1. 首先运行 1_init.py  

会初始化几个目录，手动将对应的文件放进对应的文件夹中，部分目录结构如下所示：

```bash
trainImage：训练集的图片
    - Abyssinian_1.jpg
    - Abyssinian_10.jpg
    - Abyssinian_11.jpg
    ...

validateImage：验证集的图片
    - Abyssinian_12.jpg
    - Abyssinian_13.jpg
    - Abyssinian_14.jpg
    ...

trainImageXML：训练集的 XML 文件
    - Abyssinian_1.xml
    - Abyssinian_10.xml
    - Abyssinian_11.xml
    ...

validateImageXML：验证集的 XML 文件
    - Abyssinian_12.xml
    - Abyssinian_13.xml
    - Abyssinian_14.xml
    ...

images：用来存放最终训练的图片
labels：用来存放最终边框标注的 txt 文件

```



### 2. 运行 2_createID.py 

会创建 **trainImageId.txt  validateImageId.txt** 文件，内容是图片的名称。每行一个。生成这两个文件是供  <u>3_trans.py</u>  调用使用的

例：

```bash
trainImageId.txt 内容如下
		Abyssinian_1.jpg
		Abyssinian_10.jpg
		Abyssinian_11.jpg
		...
```
### 3. 修改 3_trans.py 中的 classes = [] 顺序要和 data/custom/classes.names 文件一样

运行 3_trans.py ，会在 trainImage 和 validateImage 文件夹中生成 txt 文件，运行结果后的目录结构如下所示：例：
```bash
trainImage
		- Abyssinian_1.jpg
		- Abyssinian_1.txt
		- Abyssinian_10.jpg
		- Abyssinian_10.txt
		- Abyssinian_11.jpg
		- Abyssinian_11.txt
		...


validateImage
 		- Abyssinian_12.jpg
 		- Abyssinian_12.txt
 		- Abyssinian_13.jpg
 		- Abyssinian_13.txt
 		- Abyssinian_14.jpg
 		- Abyssinian_14.txt
```

### 4. 将 trainImage 和 validateImage 文件夹中的『图片』全部拷贝至 images 文件夹下,将 trainImage 和 validateImage 文件夹中的『txt 文件』全部拷贝至 labels 文件夹下

- https://blog.csdn.net/xiao_lxl/article/details/85342707 VOC 数据格式含义（生成的 txt 数据格式的含义）
  

```bash
生成的 txt 内容如下（举例一条）：
label <1> <2> <3> <4>

可以用以下公式简单验证一下生成的 txt 和与原始的 xml 文件是否转换正确：其中 label 是类别在 data/custom/classes.names 的索引, <> 代表缩放后的比例系数
    <1>*w = (xmax-xmin)/2 + xmin
    <2>*h = (ymax-ymin)/2 + ymin
    <3> = (xmax-xmin)/w
    <4> = (ymax-ymin)/h

```

### 5、运行 4_to_train_val_txt.py 会生成 train.txt 和 valid.txt 两个文件，内容如下：

注意：4_to_train_val_txt.py 会根据每个类别的数量选取各个类别 90% 的样本作为训练集， 10% 样本作为验证集，

**尽管前面区分了训练集和验证集的文件夹。若数据已经区分了训练集和验证集，建议重写 4_to_train_val_txt.py**
        
例：

```bash
	train.txt
		data/custom/images/Abyssinian_1.jpg
		data/custom/images/Abyssinian_10.jpg
		data/custom/images/Abyssinian_11.jpg
		...
	

	valid.txt
		data/custom/images/Abyssinian_12.jpg
		data/custom/images/Abyssinian_13.jpg
		data/custom/images/Abyssinian_14.jpg
		...
```
至此，数据集处理全部结束
1. ### 定义网络文件 (net.py)，并生成可视化文档  (visualize.py)

2. #### 训练网络

   1. 准备数据集，原始数据集准备，将数据集路径以及对应的标签放进文本，运行 move.py 脚本即可
   2. 创建一个神经网络并作可视化：net.py 
   3. 训练神经网络：train.py 
      1. 数据增强和数据预处理
      2. 数据加载器
      3. 是否可以使用 GPU
      4. 设置损失、优化器、学习率
      5. 训练模型
      6. 保存模型
      7. 保存历史训练记录

3. #### 预测

4. ### 细节处理
    部署一个网站：

  - [ ] 多个用户上传文件如何命名，管理
  - [ ] 没有识别到人脸怎么办

### 脚本说明

```
data
	数据集
	
data_collection # 数据收集
	image_class_split.py # 从原始数据集拆分成不同类别的数据（微笑和中性表情）
	image_downloader_gui_v1.0.5.rar  # 数据爬取 GUI 

face_detection 文件夹：
	face_keyPoint_detector.py  # 通过关键点和人脸检测，将嘴唇区域进行提取保存成图片
	
data_processing 文件夹
	# 在数据收集完成后，需要对数据进行清洗，清洗以后再进行下面步骤
	reformat_images.py # 设定图片格式 jpg
	# 需要做一步图片的重命名
	python resize_images.py [images_path] # 将图片整到同一尺寸
	split_train_val_test.py # 拆分数据集为训练集、测试集和验证集，可以设置分配比例，里面有点 bug 拆分的时候最后一个参数设置为 0 有可能也会将几张图片放到 测试集中

model_training 文件夹
	net.py # 自己设计神经网络
	visualize.py # 网络可视化
	train.py  # 训练
	inference.py # 模型预测测试
	
deploy_demo 文件夹
	face_detector_trained 文件夹# opencv 自带的检测器
	models 文件夹 # 训练的模型
	test_img 文件夹 # 测试的图片
	deploy_demo.py # 部署程序
	net.py # 模型的结构

server_demo 文件夹
	服务器部署
```

python 依赖包安装

- dlib

```bash
# 参考https://www.zhihu.com/question/34524316
#step 1. 安装相关依赖
# for macOS
brew install cmake
brew install boost
brew install boost-python --with-python3
# for Ubuntu
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
# for windows
#到 https://pypi.org/simple/dlib/ 下载 whl 文件 pip install *.whl 安装

#2. 安装 dlib
pip install dlib


```


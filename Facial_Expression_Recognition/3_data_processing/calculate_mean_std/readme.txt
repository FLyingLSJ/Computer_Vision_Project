本文件夹中代码的功能是计算图片的均值和方差，但是实际并不需要这么做，可以使用 ImageNet 数据集上的均值和方差，其他交给神经网络来学习
数据放置的格式
图片大小需要一致
文件夹下面需要有
	data
		class1
		class2
		class3
		......
		classes.txt
			# classes.txt 文件内容如下
			0 class1 
			1 class2
			2 class3
			# 或者，需要修改 classification_dataset.py 文件中的 _init_classes 函数中的 class_dict[c_list[1]] = c_list[0]
			class1 0
			class2 1
			class3 2
			
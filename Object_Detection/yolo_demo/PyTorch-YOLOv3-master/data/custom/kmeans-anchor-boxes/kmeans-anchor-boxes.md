根据自己的数据生成 anchor

标准的欧几里德距离导致较大的框比较小的框产生更多的误差

故使用：Jaccard index ，公式如下：

$$J(\mathbf{b_1}, \mathbf{b_2}) = \frac{\text{min}(w_1, w_2)\cdot\text{min}(h_1, h_2)}{w_1h_1 + w_2h_2 - \text{min}(w_1, w_2)\cdot\text{min}(h_1, h_2)}$$







代码参考：`gen_anchor.py` 文件，更换自己的 xml 所在的路径即可，会生成 9 个 anchor，结果类似如下：

```python
Boxes:
[[  5.2         10.60392157]
 [ 46.45333333  79.3470986 ]
 [ 20.8         35.15492958]
 [ 14.56        26.13612565]
 [ 29.81333333  50.375     ]
 [  8.09638554  16.52980132]
 [  6.35880933  13.73944954]
 [ 86.25823811 146.61285003]
 [ 10.56632173  20.69296741]]
Accuracy: 80.84%
Before Sort Ratios:
 [0.49, 0.59, 0.59, 0.56, 0.59, 0.49, 0.46, 0.59, 0.51]
After Sort Ratios:  
 [0.46, 0.49, 0.49, 0.51, 0.56, 0.59, 0.59, 0.59, 0.59]  
# 最后根据这个结果，从小到大去找对应的 Boxes，修改 config/*.cfg(你生成的配置文件是哪个就修改哪个) 文件的 [yolo] 网络下的 anchors 参数，总共有 3  个地方


```



参考：

- https://lars76.github.io/object-detection/k-means-anchor-boxes/  
- https://github.com/lars76/kmeans-anchor-boxes


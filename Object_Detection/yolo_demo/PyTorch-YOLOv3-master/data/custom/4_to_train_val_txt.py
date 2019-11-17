import os
import random



img_list = os.listdir("./images")

train_img_list = []
valid_img_list = []

classes_set = set([i.split("_")[0] for i in img_list])  # 每个类别的名称
for cls in classes_set:
    cls_list = list(filter(lambda x:x.startswith(cls), img_list))
    train_num = int(len(cls_list)*0.9)
    train_img_list += cls_list[:train_num]
    valid_img_list += cls_list[train_num:]


random.shuffle(train_img_list) # 打乱数据
random.shuffle(valid_img_list) 


print("num of train set is {} ".format(len(train_img_list)))
print("num of valid set is {} ".format(len(valid_img_list)))
print(f"total num of dataset is {len(train_img_list)+len(valid_img_list)}")

with open("train.txt", "a+") as f:
    for img in train_img_list:
        if img.endswith(".jpg"):
            f.write("data/custom/images/"+img+"\n")
print("train.txt create sucessful!")



with open("valid.txt", "a+") as f:
    for img in valid_img_list:
        if img.endswith(".jpg"):
            f.write("data/custom/images/"+img+"\n")
print("valid.txt create sucessful!")

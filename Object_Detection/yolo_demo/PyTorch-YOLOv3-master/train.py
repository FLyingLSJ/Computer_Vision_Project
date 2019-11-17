from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")  # 训练的轮次
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")  # 每次放进模型的批次
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")  # 累积多少部的梯度
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")  # yolo 配置文件路径
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")  # 也是配置文件，配置类别数、训练和测试集路径、类别名称文件路径等
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")  # 预训练模型的权重路径，最开始可以使用 yolov3_coco.weights 权重进行训练，也可以在训练过的模型基础上进行训练
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")  # 生成数据是 cpu 的线程数
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")  # 输入数据的尺寸，此值必须是 32 的整数倍
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")  # 
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")  # 用来记录训练的日志

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是不是有 GPU 可以使用

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)  # parse_data_config 在 utils/parse_config.py 文件中，类型是字典
    train_path = data_config["train"]  # 训练集的路径
    valid_path = data_config["valid"]  # 验证集的路径
    class_names = load_classes(data_config["names"])  #  load_classes 在  utils/utils.py 文件中，类型是列表，其中每个值是一个类别，如阿猫啊狗

    # Initiate model
    model = Darknet(opt.model_def).to(device)  # Darknet 在 model.py 文件中，这里可以有个  img_size 参数可以配置输入数据的尺寸
    model.apply(weights_init_normal)  #  weights_init_normal  在 utils/utils.py 文件中，对权重进行初始化

    # If specified we start from checkpoint
    # 从预训练模型进行训练，判断是 torch 模型 pth 文件还是 Darknet 预训练权重，两种权重的加载方式不同
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))  
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    # 数据加载器
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)  
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)

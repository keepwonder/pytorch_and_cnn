import copy
import time

import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from model import AlexNet


def train_val_data_process():
    train_data = FashionMNIST(root='../data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True
                              )

    train_data, val_data = random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=2)

    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=32,
                                shuffle=True,
                                num_workers=2)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用Adam优化器, 学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 损失函数为交叉熵损失韩式
    criterion = nn.CrossEntropyLoss()

    # 将模型放入到训练设备中
    model = model.to(device)

    # 复制当前模型的参数
    best_model_wts = copy.deepcopy((model.state_dict()))

    # 初始化参数

    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []

    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 50)

        # 初始化参数
        # 训练集损失
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0

        # 训练集样本数
        train_num = 0
        # 验证集样本数
        val_num = 0

        # 对每一个batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行汇总最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播和计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失值进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行汇总最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失
            loss = criterion(output, b_y)

            # 对损失值进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一个epoch的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(f'{epoch} train loss: {train_loss_all[-1]:.4f}, train acc: {train_acc_all[-1]:.4f}')
        print(f'{epoch} val loss: {val_loss_all[-1]:.4f}, val acc: {val_acc_all[-1]:.4f}')

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print(f'训练和验证耗费的时间{time_use // 60:.0f}m{time_use % 60:.0f}s')

    # 选择最优参数，保存最优参数的模型
    torch.save(best_model_wts, './best_model.pth')

    train_process = pd.DataFrame(data={'epoch': range(num_epochs),
                                       'train_loss_all': train_loss_all,
                                       'val_loss_all': val_loss_all,
                                       'train_acc_all': train_acc_all,
                                       'val_acc_all': val_acc_all,
                                       })

    return train_process


def draw_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载模型
    AlexNet = AlexNet()

    # 加载数据集
    train_data, val_data = train_val_data_process()

    # 模型训练
    train_process = train_model_process(AlexNet, train_data, val_data, num_epochs=20)

    # 画损失和准确度图
    draw_acc_loss(train_process)


"""
Epoch 0/19
--------------------------------------------------
0 train loss: 0.6189, train acc: 0.7645
0 val loss: 0.3700, val acc: 0.8630
训练和验证耗费的时间1m24s
Epoch 1/19
--------------------------------------------------
1 train loss: 0.3523, train acc: 0.8691
1 val loss: 0.3272, val acc: 0.8794
训练和验证耗费的时间2m45s
Epoch 2/19
--------------------------------------------------
2 train loss: 0.3061, train acc: 0.8866
2 val loss: 0.2970, val acc: 0.8921
训练和验证耗费的时间4m6s
Epoch 3/19
--------------------------------------------------
3 train loss: 0.2757, train acc: 0.8980
3 val loss: 0.2718, val acc: 0.8984
训练和验证耗费的时间5m27s
Epoch 4/19
--------------------------------------------------
4 train loss: 0.2576, train acc: 0.9056
4 val loss: 0.2724, val acc: 0.9021
训练和验证耗费的时间6m49s
Epoch 5/19
--------------------------------------------------
5 train loss: 0.2398, train acc: 0.9103
5 val loss: 0.2705, val acc: 0.9002
训练和验证耗费的时间8m9s
Epoch 6/19
--------------------------------------------------
6 train loss: 0.2296, train acc: 0.9147
6 val loss: 0.2735, val acc: 0.8992
训练和验证耗费的时间9m30s
Epoch 7/19
--------------------------------------------------
7 train loss: 0.2207, train acc: 0.9195
7 val loss: 0.2529, val acc: 0.9084
训练和验证耗费的时间10m51s
Epoch 8/19
--------------------------------------------------
8 train loss: 0.2056, train acc: 0.9237
8 val loss: 0.2622, val acc: 0.9085
训练和验证耗费的时间12m12s
Epoch 9/19
--------------------------------------------------
9 train loss: 0.2026, train acc: 0.9245
9 val loss: 0.2669, val acc: 0.9041
训练和验证耗费的时间13m33s
Epoch 10/19
--------------------------------------------------
10 train loss: 0.1883, train acc: 0.9296
10 val loss: 0.2585, val acc: 0.9126
训练和验证耗费的时间14m54s
Epoch 11/19
--------------------------------------------------
11 train loss: 0.1854, train acc: 0.9307
11 val loss: 0.2717, val acc: 0.9089
训练和验证耗费的时间16m15s
Epoch 12/19
--------------------------------------------------
12 train loss: 0.1833, train acc: 0.9329
12 val loss: 0.2810, val acc: 0.9099
训练和验证耗费的时间17m36s
Epoch 13/19
--------------------------------------------------
13 train loss: 0.1721, train acc: 0.9359
13 val loss: 0.3207, val acc: 0.9034
训练和验证耗费的时间18m57s
Epoch 14/19
--------------------------------------------------
14 train loss: 0.1716, train acc: 0.9368
14 val loss: 0.2831, val acc: 0.9123
训练和验证耗费的时间20m18s
Epoch 15/19
--------------------------------------------------
15 train loss: 0.1627, train acc: 0.9404
15 val loss: 0.2872, val acc: 0.9099
训练和验证耗费的时间21m39s
Epoch 16/19
--------------------------------------------------
16 train loss: 0.1584, train acc: 0.9427
16 val loss: 0.3037, val acc: 0.9050
训练和验证耗费的时间22m60s
Epoch 17/19
--------------------------------------------------
17 train loss: 0.1579, train acc: 0.9417
17 val loss: 0.3030, val acc: 0.9077
训练和验证耗费的时间24m21s
Epoch 18/19
--------------------------------------------------
18 train loss: 0.1526, train acc: 0.9436
18 val loss: 0.3212, val acc: 0.9095
训练和验证耗费的时间25m41s
Epoch 19/19
--------------------------------------------------
19 train loss: 0.1484, train acc: 0.9460
19 val loss: 0.3281, val acc: 0.9003
训练和验证耗费的时间27m2s
"""
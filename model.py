# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:26:05 2020

@author: zhijiezheng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

torch.manual_seed(1)
torch.cuda.manual_seed(1)
root = './mnist/'            # 存放数据的路径
batch_size = 64              # 每次训练的批量
hidden_units = 512           # 全连接层隐藏单元数
num_labels = 10              # 输出类别数
dropout_rate = 0.5           # Dropout概率
learning_rate = 0.1          # 初始学习率
learning_rate_decay = 0.90   # 学习率衰减系数
lr_decat_step = 400          # 学习率衰减轮数
weight_decay = 5e-4          # 正则化系数
momentum = 0.9               # 动量系数
epochs = 10                  # 训练轮数
verbose = True               # 显示训练过程
train_from_scratch = False   # 是否加载上一次训练结果

# 配置 GPU/CPU 设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 读取数据
train_dataset = datasets.MNIST(root = root,
                               train = True,
                               transform = transforms.ToTensor(),
                               download = True)

test_dataset = datasets.MNIST(root = root,
                              train = False,
                              transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True,
                                           num_workers = 2)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)

# 拉直
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)
   
# 构建网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                Flatten(),
                )
        self.fc1 = nn.Linear(64*7*7, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_labels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = F.dropout(x, dropout_rate, training = self.training)
        x = self.fc2(x)
        return x
    
# 配置模型，是否继续上一次的训练
model = Net()
if train_from_scratch == False:
    model.load_state_dict(torch.load('model.pth'))
model.to(device)

# 设置优化器
optimizer = optim.SGD(model.parameters(), lr = learning_rate,
                      momentum = momentum, weight_decay=weight_decay)

# 设置学习率衰减
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = lr_decat_step,
                                      gamma = learning_rate_decay)

# 设置损失函数
loss_func = nn.CrossEntropyLoss()

# 定义训练过程
def train(epoch, verbose = False):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        scheduler.step()
        loss = loss_func(output, target)

        if batch_idx % 200 == 0 & verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  

# 定义测试过程
def test():
    test_loss = 0.
    correct = 0.
    model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss.item(), correct, len(test_loader.dataset),
                100. * correct.type(torch.float) / len(test_loader.dataset)))


if __name__ == '__main__':
    print('Using Device:', device)
    for epoch in range(epochs):
        train(epoch, verbose = verbose)
    test()
    torch.save(model.state_dict(), 'model.pth')
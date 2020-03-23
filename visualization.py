# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:04:10 2020

@author: zhijiezheng
"""

from math import sqrt, ceil
import numpy as np
import matplotlib.pyplot as plt
from model import Net
import torch

# 定义绘图函数
def visualize_grid(Xs, ubound = 255.0, padding = 1):
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

# 加载训练模型
model = Net()
model.load_state_dict(torch.load('model.pth'))

# 读取第一个卷积层的权重
weight = model.conv1[0].weight.detach().numpy()

grid = visualize_grid(weight.transpose(0, 2, 3, 1))
plt.imshow(grid.squeeze().astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()
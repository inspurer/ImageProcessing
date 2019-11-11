# -*- coding: utf-8 -*-
# pc_type           lenovo
# create_time:      2019/11/8 21:21
# file_name:        1_2.py

import cv2

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set_context("paper") # 背景
sns.set_style('whitegrid') # 主题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题，这一句必须放在前两后面


# flags = 1 彩色图，flags = 0 灰度图
img = cv2.imread("img/test2.jpg",flags=0)

#灰度图像均衡处理
eq=cv2.equalizeHist(img)#直方图均衡化
merged = np.hstack([img,eq])
cv2.putText(merged, "before", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.putText(merged, "after", (520, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.imshow("Histogram Equalization",merged) # 合并两张图像

hist=cv2.calcHist([img],[0],None,[256],[0,256])
plt.subplot(1,2,1)
plt.xlabel("灰度级")
plt.ylabel("count(灰度级)")
plt.title("灰度直方图（均衡化变换前）")

plt.plot([i for i in range(0,256)],hist)

hist=cv2.calcHist([eq],[0],None,[256],[0,256])
plt.subplot(1,2,2)
plt.xlabel("灰度级")
plt.ylabel("count(灰度级)")
plt.title("灰度直方图（均衡化变换后）")

plt.plot([i for i in range(0,256)],hist)

plt.show()
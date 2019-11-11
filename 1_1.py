# -*- coding: utf-8 -*-
# pc_type           lenovo
# create_time:      2019/11/8 20:28
# file_name:        1_1.py

import cv2

import matplotlib.pyplot as plt
import seaborn as sns
# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set_context("paper") # 背景
sns.set_style('whitegrid') # 主题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题，这一句必须放在前两句后面


# flags = 1 彩色图，flags = 0 灰度图
img = cv2.imread("img/test1.jpg",flags=0)


fa,fb = 100,150
ga,gb = 50,200

# 处理前

# 返回ndarray 类型
hist=cv2.calcHist([img],[0],None,[256],[0,256])

plt.figure(figsize=(10, 6), dpi=100)


plt.subplot(1,2,1)

plt.imshow(img,cmap='gray')
plt.title('图像（变换前）')

plt.subplot(1,2,2)

plt.plot([i for i in range(0,256)],hist)

plt.plot([fa,fa,fb,fb],[hist.min(),hist.max(),hist.max(),hist.min()])
plt.xlabel("灰度级")
plt.ylabel("count(灰度级)")
plt.title("灰度直方图（变换前）")


alpha,beta,gamma = ga/fa,(gb-ga)/(fb-fa),(255-gb)/(255-fb)

h,w = img.shape[0],img.shape[1]


# 处理中

for hi in range(h):
    print('counting')
    for wi in range(w):
        if img[hi][wi]<fa:
            img[hi][wi] = alpha*img[hi][wi]
        elif img[hi][wi]<fb:
            img[hi][wi] = beta*(img[hi][wi] - fa) + ga
        else:
            img[hi][wi] = gamma*(img[hi][wi] - fb) + gb

# 处理后

# 返回ndarray 类型
hist=cv2.calcHist([img],[0],None,[256],[0,256])

plt.figure(figsize=(10, 6), dpi=100)


plt.subplot(1,2,1)

plt.imshow(img,cmap='gray')
plt.title('图像（变换后）')

plt.subplot(1,2,2)

plt.plot([i for i in range(0,256)],hist)

plt.plot([ga,ga,gb,gb],[hist.min(),hist.max(),hist.max(),hist.min()])
plt.xlabel("灰度级")
plt.ylabel("count(灰度级)")
plt.title("灰度直方图（变换后）")

plt.show()


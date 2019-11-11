# -*- coding: utf-8 -*-
# pc_type           lenovo
# create_time:      2019/11/9 8:48
# file_name:        2_2.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set_context("paper") # 背景
sns.set_style('whitegrid') # 主题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题，这一句必须放在前两后面

# flags = 1 彩色图，flags = 0 灰度图
img = cv2.imread("img/test6.tif",flags=0)
plt.subplot(2,3,1)
plt.imshow(img,cmap='gray')
plt.title("原图")

fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

plt.subplot(2,3,2)
plt.imshow(20*np.log(cv2.magnitude(fft_shift.real,fft_shift.imag)),cmap='gray')
plt.title("幅度谱")

plt.subplot(2,3,3)
plt.imshow(np.angle(fft_shift),cmap='gray')
plt.title("相位谱")

A = 1
img_restructed1 = np.fft.ifft2(A*np.exp(np.angle(fft)*1j))
img_restructed1 = np.real(img_restructed1)
plt.subplot(2,3,4)
plt.imshow(img_restructed1,cmap='gray')
plt.title("幅度谱为常数，仅利用相位谱重建")

img_restructed2 = np.fft.ifft2(np.abs(fft))
img_restructed2 = np.real(img_restructed2)
plt.subplot(2,3,5)
plt.imshow(img_restructed2,cmap='gray')
plt.title("相位谱为0，仅利用幅度谱重建")


mask = np.zeros((500,500))
# 在全黑 mask 上画白色多边形（内部填充）
polygons = np.array([[(225, 75), (225, 425), (275,425), (275,75)]])
cv2.fillPoly(mask, polygons, 255)
mask_fft = np.fft.fft2(mask)
img_restructed3 = np.fft.ifft2(np.abs(mask_fft)*np.exp(np.angle(fft)*1j))
img_restructed3 = np.real(img_restructed3)
plt.subplot(2,3,6)
plt.imshow(img_restructed3,cmap='gray')
plt.title("利用上一题幅度谱和本图相位谱重建")

plt.figure()
plt.imshow(img_restructed3,cmap='gray')


plt.show()


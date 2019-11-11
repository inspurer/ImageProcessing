# -*- coding: utf-8 -*-
# pc_type           lenovo
# create_time:      2019/11/8 21:44
# file_name:        2_1.py

import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set_context("paper") # 背景
sns.set_style('whitegrid') # 主题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题，这一句必须放在前两后面

img = np.zeros((500,500))
# 在全黑 mask 上画白色多边形（内部填充）
polygons = np.array([[(225, 75), (225, 425), (275,425), (275,75)]])

cv2.fillPoly(img, polygons, 255)
# numpy、scipy、opencv 都有对应的 fft 实现
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
# 将图像的低频部分移动到图像中心
dft_shift = np.fft.fftshift(dft)


# print(dft_shift,type(dft_shift))
plt.figure(figsize=(10, 6), dpi=100)
magnitude_spectrum=20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('input image（原图f）')
plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('magnitude spectrum（二维傅里叶变换幅度图）')


plt.figure(figsize=(10, 6), dpi=100)

h,w = img.shape

import copy
img1 = copy.deepcopy(img)
# print(img.dtype)

for hi in range(h):
    for wi in range(w):
        img1[hi][wi] = ((-1)**(hi+wi))*img1[hi][wi]


# 有 -0.0 也有 -255
img1 = np.clip(img1,0.,255.)
# 将图像的低频部分移动到图像中心
dft1 = cv2.dft(np.float32(img1),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift1 = np.fft.fftshift(dft1)

magnitude_spectrum1=20*np.log(cv2.magnitude(dft_shift1[:,:,0],dft_shift1[:,:,1]))
plt.subplot(1,2,1)
plt.imshow(img1,cmap='gray')
plt.title('input image（原图f1）')
plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum1,cmap='gray')
plt.title('magnitude spectrum（二维傅里叶变换幅度图）')


plt.figure(figsize=(10, 6), dpi=100)
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(img1, M, (w, h))
dft2 = cv2.dft(np.float32(rotated),flags=cv2.DFT_COMPLEX_OUTPUT)
# 将图像的低频部分移动到图像中心
dft_shift2 = np.fft.fftshift(dft2)

magnitude_spectrum2=20*np.log(cv2.magnitude(dft_shift2[:,:,0],dft_shift2[:,:,1]))
plt.subplot(1,2,1)
plt.imshow(rotated,cmap='gray')
plt.title('input image（原图f2）')
plt.subplot(1,2,2)
plt.imshow(magnitude_spectrum2,cmap='gray')
plt.title('magnitude spectrum（二维傅里叶变换幅度图）')

plt.show()



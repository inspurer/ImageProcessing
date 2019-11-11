# -*- coding: utf-8 -*-
# pc_type           lenovo
# create_time:      2019/11/10 9:13
# file_name:        4_1.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set_context("paper") # 背景
sns.set_style('whitegrid') # 主题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题，这一句必须放在前两后面


img1 = cv2.imread("img/test7.tif",flags=0)
img2 = cv2.imread("img/test8.tif",flags=0)

def my_grad(img):
    height,width = img.shape[0],img.shape[1]
    Grad = np.zeros(img.shape,dtype=np.int8)
    # 不对右、下边界作处理、影响不大
    for h in range(height-1):
        for w in range(width-1):
            gx = int(img[h,w+1])- int(img[h,w])
            gy = int(img[h+1,w]) -int(img[h,w])
            Grad[h][w] = abs(gx) + abs(gy)
    return Grad

def my_roberts(img):
    # Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Roberts

def my_prewitt(img):
    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt

def my_sobel(img):
    # Sobel算子
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 对x求一阶导
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Sobel

plt.figure()
plt.subplot(2,5,1)
plt.imshow(img1,cmap='gray')
plt.title("原图")
plt.subplot(2,5,2)
mg1 = my_grad(img1)
plt.imshow(mg1,cmap='gray')
plt.title("梯度算子")
plt.subplot(2,5,3)
mr1 = my_roberts(img1)
plt.imshow(mr1,cmap='gray')
plt.title("Roberts算子")
plt.subplot(2,5,4)
mp1 = my_prewitt(img1)
plt.imshow(mp1,cmap='gray')
plt.title("Prewitt算子")
plt.subplot(2,5,5)
ms1 = my_sobel(img1)
plt.imshow(ms1,cmap='gray')
plt.title("Sobel算子")

plt.subplot(2,5,6)
plt.imshow(img2,cmap='gray')
plt.title("原图")
plt.subplot(2,5,7)
mg2 = my_grad(img2)
plt.imshow(mg2,cmap='gray')
plt.title("梯度算子")
plt.subplot(2,5,8)
mr2 = my_roberts(img2)
plt.imshow(mr2,cmap='gray')
plt.title("Roberts算子")
plt.subplot(2,5,9)
mp2 = my_prewitt(img2)
plt.imshow(mp2,cmap='gray')
plt.title("Prewitt算子")
plt.subplot(2,5,10)
ms2 = my_sobel(img2)
plt.imshow(ms2,cmap='gray')
plt.title("Sobel算子")

def my_laplacian(img):
    dst = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    return Laplacian

plt.figure()
plt.subplot(2,2,1)
plt.imshow(img1,cmap='gray')
plt.title("原图")
plt.subplot(2,2,2)
ml1 = my_laplacian(img1)
plt.imshow(ml1,cmap='gray')
plt.title("Laplacian算子")
plt.subplot(2,2,3)
plt.imshow(img2,cmap='gray')
plt.title("原图")
plt.subplot(2,2,4)
ml2 = my_laplacian(img2)
plt.imshow(ml2,cmap='gray')
plt.title("Laplacian算子")

plt.figure()

def my_adaptivethreshold(img):
    result_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return result_img

plt.subplot(2,5,1)
plt.imshow(mg1,cmap='gray')
plt.title("梯度算子")
plt.subplot(2,5,2)
plt.imshow(mr1,cmap='gray')
plt.title("Roberts算子")
plt.subplot(2,5,3)
plt.imshow(mp1,cmap='gray')
plt.title("Prewitt算子")
plt.subplot(2,5,4)
plt.imshow(ms2,cmap='gray')
plt.title("Sobel算子")
plt.subplot(2,5,5)
plt.imshow(ml1,cmap='gray')
plt.title("Laplacian算子")
plt.subplot(2,5,6)
mg1 = np.uint8(mg1)
plt.imshow(my_adaptivethreshold(mg1),cmap='gray')
plt.title("梯度算子(自适应阈值处理)")
plt.subplot(2,5,7)
plt.imshow(my_adaptivethreshold(mr1),cmap='gray')
plt.title("Roberts算子(自适应阈值处理)")
plt.subplot(2,5,8)
plt.imshow(my_adaptivethreshold(mp1),cmap='gray')
plt.title("Prewitt算子(自适应阈值处理)")
plt.subplot(2,5,9)
plt.imshow(my_adaptivethreshold(ms1),cmap='gray')
plt.title("Sobel算子(自适应阈值处理)")
plt.subplot(2,5,10)
plt.imshow(my_adaptivethreshold(ml1),cmap='gray')
plt.title("Laplacian算子(自适应阈值处理)")
plt.show()


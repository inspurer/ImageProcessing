# -*- coding: utf-8 -*-
# author:           inspurer(月小水长)
# pc_type           lenovo
# create_time:      2019/11/10 21:05
# file_name:        4_2.py

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


img = cv2.imread("img/test7.tif",flags=0)

def my_sobel(img):
    # Sobel算子
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 对x求一阶导
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 对y求一阶导
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Sobel

def my_adaptivethreshold(img):
    result_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return result_img

plt.subplot(2,3,1)
plt.imshow(img,cmap='gray')
plt.title("原图")
plt.subplot(2,3,2)
ms = my_sobel(img)
plt.imshow(ms,cmap='gray')
plt.title("Sobel算子(未均值平滑)")
plt.subplot(2,3,3)
plt.imshow(my_adaptivethreshold(ms),cmap='gray')
plt.title("自适应阈值处理(未均值平滑)")

def fspeical_average(image,kernel):
    a = len(kernel)
    kernel = kernel/(a**2)
    step = a//2
    h,w = image.shape[0],image.shape[1]
    nh,nw = h+2*step,w+2*step
    lbimg = np.zeros((nh,nw), np.float32)
    tmpimg = np.zeros((nh,nw))
    newimg = np.array(image)
    tmpimg[step:nh - step, step:nw - step] = newimg[0:h, 0:w]
    for y in range(step, nh - step):
        for x in range(step, nw - step):
            lbimg[y, x] = np.sum(kernel * tmpimg[y - step:y + step + 1, x - step:x + step + 1])
    resultimg = np.array(lbimg[step:nh - step, step:nw - step], np.uint8)
    return resultimg

plt.subplot(2,3,4)
plt.imshow(img,cmap='gray')
plt.title("原图")
plt.subplot(2,3,5)
img = fspeical_average(img,kernel=np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]))
ms = my_sobel(img)
plt.imshow(ms,cmap='gray')
plt.title("Sobel算子(均值平滑)")
plt.subplot(2,3,6)
plt.imshow(my_adaptivethreshold(ms),cmap='gray')
plt.title("自适应阈值处理(均值平滑)")

plt.show()

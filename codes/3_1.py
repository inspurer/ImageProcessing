# -*- coding: utf-8 -*-
# pc_type           lenovo
# create_time:      2019/11/9 15:15
# file_name:        3_1.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set_context("paper") # 背景
sns.set_style('whitegrid') # 主题
sns.set(font='SimHei')  # 解决Seaborn中文显示问题，这一句必须放在前两后面

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gauss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值   mean = 0 是高斯白噪声
        var : 方差    方差越大，图像越模糊
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    # 把 out 的元素限制在 low_clip 和 1 之间
    out = np.clip(out, low_clip, 1.0)
    out = out*255
    #cv.imshow("gasuss", out)
    return out
from PIL import Image

# 解决 opencv 不能读取 gif
gif = cv2.VideoCapture('img/test3.gif')
ret,frame = gif.read()
img = Image.fromarray(frame)
# L : 灰度图  , RGB : RGB 彩色图
img = img.convert('L')
img = np.array(img)

sp_img = sp_noise(img,0.015)

gs_img = gauss_noise(img,var=0.02)

# 邻域平均法
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
# 中值滤波法
def fspeical_medium(image,a):
    step = a // 2
    h, w = image.shape[0], image.shape[1]
    nh, nw = h + 2 * step, w + 2 * step
    lbimg = np.zeros((nh, nw), np.float32)
    tmpimg = np.zeros((nh, nw))
    newimg = np.array(image)
    tmpimg[step:nh - step, step:nw - step] = newimg[0:h, 0:w]
    for y in range(step, nh - step):
        for x in range(step, nw - step):
            lbimg[y, x] = np.median(tmpimg[y - step:y + step + 1, x - step:x + step + 1])
    resultimg = np.array(lbimg[step:nh - step, step:nw - step], np.uint8)
    return resultimg

plt.figure()
plt.subplot(2,4,1)
plt.imshow(img,cmap='gray')
plt.title("原图")
plt.subplot(2,4,5)
plt.imshow(img,cmap='gray')
plt.title("原图")
plt.subplot(2,4,2)
plt.imshow(sp_img,cmap='gray')
plt.title("加椒盐噪声")
plt.subplot(2,4,3)
plt.imshow(fspeical_average(sp_img,kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])),cmap='gray')
plt.title("邻域平均法去椒盐噪声（3x3)")
plt.subplot(2,4,4)
plt.imshow(fspeical_average(sp_img,kernel=np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])),cmap='gray')
plt.title("邻域平均法去椒盐噪声（5x5)")
plt.subplot(2,4,6)
plt.imshow(gs_img,cmap='gray')
plt.title("加高斯噪声")
plt.subplot(2,4,7)
plt.imshow(fspeical_average(gs_img,kernel=np.array([[1,1,1],[1,1,1],[1,1,1]])),cmap='gray')
plt.title("邻域平均法去高斯噪声（3x3)")
plt.subplot(2,4,8)
plt.imshow(fspeical_average(gs_img,kernel=np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])),cmap='gray')
plt.title("邻域平均法去高斯噪声（5x5)")


plt.figure()
plt.subplot(2,4,1)
plt.imshow(img,cmap='gray')
plt.title("原图")
plt.subplot(2,4,5)
plt.imshow(img,cmap='gray')
plt.title("原图")
plt.subplot(2,4,2)
plt.imshow(sp_img,cmap='gray')
plt.title("加椒盐噪声")
plt.subplot(2,4,3)
plt.imshow(cv2.medianBlur(sp_img,3),cmap='gray')
plt.title("中值滤波法去椒盐噪声（3x3)")
plt.subplot(2,4,4)
plt.imshow(cv2.medianBlur(sp_img,5),cmap='gray')
plt.title("中值滤波法去椒盐噪声（5x5)")
plt.subplot(2,4,6)
plt.imshow(gs_img,cmap='gray')
plt.title("加高斯噪声")

plt.subplot(2,4,7)
plt.imshow(fspeical_medium(gs_img,3),cmap='gray')
plt.title("中值滤波法去高斯噪声（3x3)")
plt.subplot(2,4,8)
plt.imshow(fspeical_medium(gs_img,5),cmap='gray')
plt.title("中值滤波法去高斯噪声（5x5)")

# for h in range(gs_img.shape[0]):
#     for w in range(gs_img.shape[1]):
#         if gs_img[h][w]<0:
#             gs_img[h][w] = -gs_img[h][w]

# medianBlur 仅接收无符号整数类型元素
# gs_img = np.uint8(gs_img)
# print(gs_img)
# plt.subplot(2,4,7)
# print(sp_img,gs_img)
# plt.imshow(cv2.medianBlur(gs_img,3),cmap='gray')
# plt.title("中值滤波法去高斯噪声（3x3)")
# plt.subplot(2,4,8)
# plt.imshow(cv2.medianBlur(gs_img,5),cmap='gray')
# plt.title("中值滤波法去高斯噪声（5x5)")


plt.show()


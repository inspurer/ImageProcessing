# -*- coding: utf-8 -*-
# pc_type           lenovo
# create_time:      2019/11/9 21:18
# file_name:        3_2.py

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

from PIL import Image

# 解决 opencv 不能读取 gif
gif = cv2.VideoCapture('img/test5.gif')
ret,frame = gif.read()
img = Image.fromarray(frame)
# L : 灰度图  , RGB : RGB 彩色图
img = img.convert('L')
img = np.array(img)

def plot_fft(img,title):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将图像的低频部分移动到图像中心
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)

# print(dft_shift,type(dft_shift))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img,cmap='gray')
plt.title('input image（原图）')
plt.subplot(1,2,2)
plot_fft(img,title='magnitude spectrum（二维傅里叶变换幅度图）')


def ideal_low_pass_filter(img,r):
    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    # 设置低通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - r:crow + r, ccol - r:ccol + r] = 1

    # 掩膜图像和频谱图像乘积
    f = fshift * mask

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    return res
def ideal_high_pass_filter(img,r):
    # 傅里叶变换
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # 设置高通滤波器
    rows, cols = img.shape[0],img.shape[1]
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - r:crow + r, ccol - r:ccol + r] = 0

    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    # 最好转成 int
    return iimg
def guass_low_pass_filter(img,r):
    height,width = img.shape[0],img.shape[1]
    fft = np.fft.fft2(img)
    fftshift = np.fft.fftshift(fft)

    for i in range(height):
        for j in range(width):
            if (i - (height - 1) / 2) ** 2 + (j - (width - 1) / 2) ** 2 >= r ** 2:
                fftshift[i, j] = 0

    fftshift = np.fft.ifftshift(fftshift)
    ifft = np.fft.ifft2(fftshift)

    new = np.zeros((height, width), dtype="uint8")

    for i in range(height):
        for j in range(width):
            new[i, j] = ifft[i, j].real

    return new
def guass_high_pass_filter(img,sigma = 0.1):
    height, width = img.shape[0],img.shape[1]

    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)

    for i in range(height):
        for j in range(height):
            fft[i, j] *= (1 - np.exp(-((i - (height - 1) / 2) ** 2 + (j - (width - 1) / 2) ** 2) / 2 / sigma ** 2))

    fft = np.fft.ifftshift(fft)
    ifft = np.fft.ifft2(fft)

    ifft = np.real(ifft)
    max = np.max(ifft)
    min = np.min(ifft)

    res = np.zeros((height, width), dtype="uint8")

    for i in range(height):
        for j in range(width):
            res[i, j] = 255 * (ifft[i, j] - min) / (max - min)
    return res

plt.figure()

plt.subplot(2,4,1)
ilpf = ideal_low_pass_filter(img=img,r=30)
plt.imshow(ilpf,cmap='gray')
plt.title("理想低通滤波图像")
plt.subplot(2,4,5)
plot_fft(ilpf,title="理想低通滤波图像傅里叶变换频谱图")
plt.subplot(2,4,2)
ihpf = ideal_high_pass_filter(img=img,r=30)
plt.imshow(ihpf,cmap='gray')
plt.title("理想高通滤波图像")
plt.subplot(2,4,6)
plot_fft(ihpf,title="理想高通滤波图像傅里叶变换频谱图")
plt.subplot(2,4,3)
glpf = guass_low_pass_filter(img=img,r=30)
plt.imshow(glpf,cmap='gray')
plt.title("高斯低通滤波图像")
plt.subplot(2,4,7)
plot_fft(glpf,title="高斯低通滤波图像傅里叶变换频谱图")
plt.subplot(2,4,4)
ghpf = guass_high_pass_filter(img=img)
plt.imshow(ghpf,cmap='gray')
plt.title("高斯高通滤波图像")
plt.subplot(2,4,8)
plot_fft(glpf,title="高斯高通滤波图像傅里叶变换频谱图")
plt.show()


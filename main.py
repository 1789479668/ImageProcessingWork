import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import gray_hist_thresh
import median_filter
import fast_fourier_transform
import equalize
import roberts
def ImageProcessing(imgarray, type):
    if type == 'MedianFilter':
        return median_filter.MedianFilter(imgarray)
    if type == 'DFT':
        return fast_fourier_transform.fourier_transform(imgarray,'DFT')
    if type == 'IDFT':
        return fast_fourier_transform.fourier_transform(imgarray,'IDFT')
    if type == 'HE':
        return equalize.equalize_hist(imgarray)
    if type == 'Roberts':
        return roberts.Roberts(imgarray)
    if type == 'GHT':
        return gray_hist_thresh.threshTwoPeaks(imgarray)
'''
0.中值滤波：MedianFilter
1.离散傅里叶变换：DFT
2.离散逆傅里叶变换：IDFT
3.灰度直方图均衡: HE
4.Roberts算子边缘检测:Roberts
5.灰度直方图双峰法阈值分割:GHT
'''
#以下部分是通过matplotlib直接展示处理前和处理后图片，以此展示处理效果
type = ['MedianFilter',
        'DFT',
        'IDFT',
        'HE',
        'Roberts',
        'GHT']
#展示效果的默认图片路径
img_path = ['./src/salt_pepper_Miss.bmp',
            './src/TESTPAT1.TIF',
            '',
            './src/car.jpg',
            './src/Miss.bmp',
            './src/BOARD.TIF']

for i in range(6):
    imgarray = cv2.imread(img_path[i], cv2.IMREAD_GRAYSCALE)
    new_array = ImageProcessing(imgarray, type[i])

    #展示图像
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 展示原始图像
    plt.subplot(1, 2, 1), plt.imshow(imgarray, cmap='gray')
    plt.title('原始图像1'), plt.xticks([]), plt.yticks([])
    # 展示处理后图像
    plt.subplot(1, 2, 2), plt.imshow(new_array, cmap='gray')
    plt.title('处理图像1'), plt.xticks([]), plt.yticks([])
    plt.show()
    #任意键继续
    os.system('pause')
    plt.close()


# # 保存图片部分
#img_path = './src/IC.TIF'
#save_path = './src_save/g_Miss.BMP'
#
#'''对图片img，进行type类型的图像处理功能'''
#imgarray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 使用灰度读出，默认读出方式含有B,G,R三个通道，因为本就是灰度图，无需多此一举。
#new_array = ImageProcessing(imgarray,'Roberts')
# cv2.imwrite(save_path, new_array)
# # 显示运算后图片
# cv2.imshow('img',new_array)
# cv2.waitKey()
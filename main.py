import matplotlib.pyplot as plt
import cv2
import numpy as np

import median_filter
import fast_fourier_transform
import equalize
import roberts
import gray_hist_thresh
import twofliter

plt.rcParams['font.sans-serif'] = ['SimHei']

def ImageProcessing(imgarray, type):
    if type == 'MedianFilter':
        return median_filter.MedianFilter(imgarray)
    #还有对应的DFT和IDFT过程，但是FFT和DFT输出基本一致，若不符合FFT条件（为偶数矩阵）就使用DFT，故没有单独的列出来。
    if type == 'FFT':
        return fast_fourier_transform.fourier_transform(imgarray,'FFT')
    #逆变换只能基于正变换运行过程中复数矩阵时进行逆变换，如果是可以show出来的振幅图，就无法做傅里叶逆变换了
    #因为shift和iffshift，是一个逆过程，但是shift后的复数矩阵转化为振幅图，与震幅图直接iffshift不是逆过程。
    #应该是震幅图转换为原来的复数矩阵后再iffshift是与之对于的逆过程，但是无法做到震幅图转化为原来的那个复数矩阵
    #因此傅里叶逆变换应该结合滤波一起使用，设定滤波函数再使用逆变换才有意义。否则就是输入原图再输出原图。
    if type == 'IFFT':
        return np.abs(fast_fourier_transform.dft(fast_fourier_transform.dft(imgarray,'FFT'), 'IFFT'))
    if type == 'HE':
        return equalize.equalize_hist(imgarray)
    if type == 'Roberts':
        return roberts.Roberts(imgarray)
    if type == 'GHT':
        return gray_hist_thresh.threshTwoPeaks(imgarray)
    if type == 'BHPF':#滤波效果体现IFFT作用
        #巴沃斯特高通滤波,默认初始频率10
        return twofliter.freq_filter(imgarray,twofliter.BHPF(imgarray, 10))
    if type == 'GLPF':
        #高斯低通滤波,默认初始频率10
        return twofliter.freq_filter(imgarray,twofliter.GLPF(imgarray,10))


'''
0.中值滤波：MedianFilter
1.快速傅里叶变换：FFT
2.快速傅里叶逆变换：IFFT
3.灰度直方图均衡: HE
4.Roberts算子边缘检测:Roberts
5.灰度直方图双峰法阈值分割:GHT
6.巴沃斯特高通滤波:BHPF
7.高斯低通滤波:GLPF
快速傅里叶变换：FFT
快速傅里叶逆变换：IFFT
'''
#以下部分是通过matplotlib直接展示处理前和处理后图片，以此展示处理效果,同时在src_save文件夹储存运行后的图片
type = ['MedianFilter',
        'FFT',
        'IFFT',
        'HE',
        'Roberts',
        'GHT',
        'BHPF',
        'GLPF']
#展示效果的默认图片路径，IDFT还未改善，不知道为什么rect反而运行时间很久，是否保留待定
img_path = ['./src/salt_pepper_Miss.bmp',
            './src/rect.bmp',
            './src/TESTPAT2.TIF',
            './src/car.jpg',
            './src/Miss.bmp',
            './src/cameraman.tif',
            './src/blood1.BMP',
            './src/blood1.BMP']
title = ['中值滤波图像',
         '快速傅里叶变换图像',
         '快速傅里叶逆变换图像',
         '直方图均衡图像',
         'Roberts边缘检测图像',
         '双峰法阈值分割图像',
         '巴沃斯特高通滤波图像',
         '高斯低通滤波图像']

for i in range(8):
    imgarray = cv2.imread(img_path[i], cv2.IMREAD_GRAYSCALE)
    new_array = ImageProcessing(imgarray, type[i])
    save_path = f'./src_save/{type[i]}.BMP'

    #opencv2的imshow表现似乎有些问题，没办法显示有些变化。
    # imgs = np.hstack([imgarray, new_array])
    # cv2.imshow('displace', imgs)
    # cv2.waitKey()

#展示图象
    '''
    存在问题：car图片负责的直方图均衡,前后对比直接打开文件才能显示。使用显示手段无法展示未均衡前的图片。
    '''
    # 展示原始图像
    plt.subplot(1, 2, 1), plt.imshow(imgarray, cmap='gray')
    plt.title('原始图像'), plt.xticks([]), plt.yticks([])
    # 展示处理后图像
    plt.subplot(1, 2, 2), plt.imshow(new_array, cmap='gray')
    plt.title(f'{title[i]}'), plt.xticks([]), plt.yticks([])
    plt.show()
    #关掉视图后继续播放后续的
#保存图片

    cv2.imwrite(save_path, new_array)


'''单个试验部分'''
# img_path = './src/car.TIF'
# save_path = './src_save/g_Miss.BMP'
# # 保存图片部分
# '''对图片img，进行type类型的图像处理功能'''
# imgarray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 使用灰度读出，默认读出方式含有B,G,R三个通道，因为本就是灰度图，无需多此一举。
# new_array = ImageProcessing(imgarray,'Roberts')
# cv2.imwrite(save_path, new_array)
#
# # 显示运算后图片
# cv2.imshow('img',new_array)
# 任意键继续
# cv2.waitKey()
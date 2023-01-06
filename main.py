import numpy as np
import cv2
import median_filter
import fast_fourier_transform

def ImageProcessing(img, type):
    '''对图片img，进行type类型的图像处理功能'''
    imgarray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)#使用灰度读出，默认读出方式含有B,G,R三个通道，因为本就是灰度图，无需多此一举。
    if type == 'MedianFilter':
        return median_filter.MedianFilter(imgarray)
    if type == 'DFT':
        return fast_fourier_transform.fourier_transform(imgarray,'DFT')
    if type == 'IDFT':
        return fast_fourier_transform.fourier_transform(imgarray,'IDFT')
'''
中值滤波：MedianFilter
离散傅里叶变换：DFT
离散逆傅里叶变换：IDFT
'''
img_path = './src/Miss.bmp'
save_path = './src_save/g_Miss.BMP'

new_array = ImageProcessing(img_path,'DFT')
cv2.imwrite(save_path, new_array)
# cv2.imshow('img',new_array)
# cv2.waitKey()
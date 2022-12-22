import numpy as np
import cv2
"""
计算离散傅里叶变换
"""
img = './src/salt_pepper_Miss.bmp'
path = './src_save/m_Miss.bmp'

imgarray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
print(imgarray.shape[0])
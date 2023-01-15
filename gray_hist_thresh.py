# coding:utf8
import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram(grayfig):
    x = grayfig.shape[0]
    y = grayfig.shape[1]
    hist = np.zeros(256)
    for i in range(x):
        for j in range(y):
            hist[grayfig[i][j]] += 1
    return hist

def threshTwoPeaks(image):

    # 计算灰度直方图
    hist = histogram(image)
    # 寻找灰度直方图的最大峰值对应的灰度值
    max_Location = np.where(hist == np.max(hist)) #maxLoc 中存放位置
    firstPeak = max_Location[0][0]
    # 寻找灰度直方图的第二个峰值对应的灰度值
    elementList = np.arange(256,dtype = np.uint64)
    measureDists = np.power(elementList - firstPeak,2) * hist

    max_Location2 = np.where(measureDists == np.max(measureDists))
    secondPeak = max_Location2[0][0]

    # 找到两个峰值之间的最小值对应的灰度值，作为阈值标准
    thresh = 0
    if secondPeak > firstPeak:
        firstPeak,secondPeak = secondPeak,firstPeak
    #在两峰之间找到最低点
    temp = hist[secondPeak:firstPeak]
    minloc = np.where(temp == np.min(temp))
    thresh = secondPeak + minloc[0][0] + 1
    # 找到阈值之后进行阈值处理，得到二值图
    threshImage_out = image.copy()
    # 大于阈值的都设置为255
    threshImage_out[threshImage_out > thresh] = 255
    # 小于阈值的都设置为0
    threshImage_out[threshImage_out <= thresh] = 0
    #thresh反映阈值的位置
    return threshImage_out
import cv2
import numpy as np

def MedianFilter(imgarray, k=3, padding = None):
    H, W = imgarray.shape
    #灰度图像数据结构即(H,W)，BGR结构为(H,W,C)

    if not padding:
        edge = int((k-1)/2)
        #设定邻域大小
        if H - 1 - edge <= edge or W - 1 - edge <= edge:
            print("参数k设置得太大了")
            return None
        new_array = np.zeros((H, W),'uint8')
        #使用方法为np.zeros(shape,dtype)，shape，新数组的维度，dtype新数组的数据类型。
        #否则报错：Cannot interpret '256' as a data type
        for i in range(H):
            for j in range(W):
                if i <= edge - 1 or i >= H - 1 - edge or j <= edge - 1 or j >= H - edge - 1:
                    #边缘处不进行滤波（防止无像素位置的影响，如64*64图片，即(2,2)到(63,63）
                    new_array[i,j] = imgarray[i,j]
                else:
                    new_array[i,j] = np.median(imgarray[i-edge:i+edge+1, j-edge:j+edge+1])
                    #使用median直接进行选取领域所有像素值的中间值代替当前点的像素值。
                    #使用new_array来储存运算后结果，而不是直接覆盖imgarray，防止连续作用发生
        return new_array

import cv2
import numpy as np

#np.pi=3.141592653589793
'''
二维傅里叶变换分解为两个维度上分别做一次一维傅里叶变换
p180
'''

def DFT(img, path):
    imgarray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', imgarray)
    #imgarray是一个(H,W)矩阵
    H, W = imgarray.shape
    # print(imgarray)
    # G = Fx*imgarray*Fy（矩阵运算）
        # for i in range(H):
        #     for k in range(W):
        #         Fx = 1/np.sqrt(H)*np.exp(-1j*2*np.pi*i*k/(np.sqrt(H)*np.sqrt(W)))
        #         Fy = 1/np.sqrt(W)*np.exp(-1j*2*np.pi*i*k/(np.sqrt(H)*np.sqrt(W)))
        # #傅里叶变换的核
        #         Garray = np.dot(np.dot(Fx,imgarray),Fy)
    F = 1/np.sqrt(H)*np.exp(-1j*2*np.pi*i*k/H)
    cv2.imshow('FFT', Garray)
    cv2.waitKey(0)

img_path = './src/salt_pepper_Miss.bmp'
save_path = './src_save/m_Miss.bmp'
DFT(img_path, save_path)
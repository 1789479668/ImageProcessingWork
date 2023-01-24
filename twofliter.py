import cv2
import matplotlib.pyplot as plt
import numpy as np
import fast_fourier_transform
def GLPF(image,d0):
    #高斯低通滤波器
    #f0即截止频率
    H = np.zeros_like(image,float)
    M, N = image.shape
    #找到中心点作为原点
    mid_x = M/2
    mid_y = N/2
    for x in range(0, M):
        for y in range(0, N):
            d = np.sqrt((x - mid_x)**2 + (y - mid_y) ** 2)
            H[x, y] = np.exp(-d/(2*d0))
    return H

def BHPF(image,d0):#巴特沃斯高通滤波器
    H = np.zeros_like(image,float)
    M,N = image.shape
    mid_x = int(M/2)
    mid_y = int(N/2)
    for x in range(0, M):
        for y in range(0, N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            H[x,y] = 1/(1+(d0/(d**2)))
    return H

def freq_filter(img ,filter):
    fftimg = fast_fourier_transform.fft_2d(img)
    fftimgshift = np.fft.fftshift(fftimg)
    #进行傅里叶变换，获得代表频谱的复数矩阵
    handle_fftImgShift1 = fftimgshift*filter
    #在频谱中进行滤波，使用滤波器filter
    handle_fftImgShift2 = np.fft.ifftshift(handle_fftImgShift1)
    handle_fftImgShift3 = fast_fourier_transform.ifft_2d(handle_fftImgShift2)
    handle_fftImgShift4 = np.abs(handle_fftImgShift3)
    #傅里叶逆变换，并且取模获得振幅图
    return handle_fftImgShift4


#
# img = cv2.imread('src/blood1.BMP', cv2.IMREAD_GRAYSCALE)
# img = freq_filter(img ,BHPF(img, 10))
#
# plt.imshow(img,cmap='gray')
# plt.show()
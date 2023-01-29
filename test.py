import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def histogram(img):
    #计算灰度直方图，统计img中坐标为i，j的灰度值数量。hist为一维矩阵，大小为1x256，数值代表频率。
    x = img.shape[0]
    y = img.shape[1]
    hist = np.zeros(256)
    for i in range(x):
        for j in range(y):
            hist[img[i][j]] += 1
    return hist

img = 'src/cameraman.tif'
imgarray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)  # 打开图片,并把图片转换为np矩阵
ret = histogram(imgarray)
# plt.title("Histogram")
# plt.plot(ret)
# plt.axis([0, 255, 0, np.max(ret) * 1.05])
# plt.xlabel("Gray Level")
# plt.ylabel("Frequence")
plt.figure(figsize=(300,100), dpi=10)
plt.xlabel('Gray level')
plt.ylabel('Frequence')
plt.grid()
plt.bar(range(256), ret)
plt.show()

# #读取图像
# img = cv.imread('src/rect.tif', 0)
# fff = cv.imread('src/FFT.BMP', 0)
#
# #傅里叶变换
# def testfft(img):
#     f = np.fft.fft2(img)
#     fshift = np.fft.fftshift(f)
#     res = np.log(np.abs(fshift))*20
#     return res
# #傅里叶逆变换
# def testifft(img):
#     ishift = np.fft.ifftshift(fff)
#     iimg = np.fft.ifft2(ishift)
#     iimg = np.abs(iimg)
#     return iimg
#
#
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
# # 这里构建振幅图的公式没学过
# res = 20*np.log(np.abs(fshift))
#
# ishift = np.fft.ifftshift(res)
# iimg = np.fft.ifft2(ishift)
# iimg = np.log(abs(iimg))*20
# #展示结果
# plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Fourier Image')
# plt.axis('off')
# plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
# plt.axis('off')
# plt.show()

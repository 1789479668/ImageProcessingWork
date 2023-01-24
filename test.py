import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#读取图像
img = cv.imread('src/rect.tif', 0)
fff = cv.imread('src/FFT.BMP', 0)

#傅里叶变换
def testfft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    res = np.log(np.abs(fshift))*20
    return res
#傅里叶逆变换
def testifft(img):
    ishift = np.fft.ifftshift(fff)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    return iimg


f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# 这里构建振幅图的公式没学过
res = 20*np.log(np.abs(fshift))

ishift = np.fft.ifftshift(res)
iimg = np.fft.ifft2(ishift)
iimg = np.log(abs(iimg))*20
#展示结果
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Fourier Image')
plt.axis('off')
plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
plt.axis('off')
plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import fast_fourier_transform
def GLPF(image,d0=10):
    # 高斯低通滤波器
    # f0即截止频率
    H = np.zeros_like(image,float)
    M, N = image.shape
    # 找到中心点作为原点
    mid_x = M/2
    mid_y = N/2
    for x in range(0, M):
        for y in range(0, N):
            d = np.sqrt((x - mid_x)**2 + (y - mid_y) ** 2)
            H[x, y] = np.exp(-d/(2*d0))
    return H

def BHPF(image,d0=10):#巴特沃斯高通滤波器
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
    # 进行傅里叶变换，获得代表频谱的复数矩阵
    fftimg = fast_fourier_transform.fft_2d(img)
    fftimgshift = np.fft.fftshift(fftimg)
    # 在频谱中进行滤波，使用滤波器filter（空域卷积即频域之积）
    fftimgshift1 = fftimgshift*filter
    # 傅里叶逆变换，并且取模获得振幅图
    fftimgshift2 = np.fft.ifftshift(fftimgshift1)
    fftimgshift3 = fast_fourier_transform.ifft_2d(fftimgshift2)
    newimg = np.abs(fftimgshift3)
    return newimg

'''
单独运行该模块功能，则将下面的注释消除，图片路径在第49行中修改。
'''
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 读取图像
# img = cv2.imread('./src/blood1.BMP',cv2.IMREAD_GRAYSCALE)
# # BHPF即巴沃斯特高通滤波，BHPF(img,d0)，d0默认值为10，可以自行修改。
# # GLPF即高斯低通滤波，GLPF(img,d0)，d0默认值为10，可以自行修改。
# newimg = freq_filter(img, BHPF(img))
# # 显示图形
# titles = [u'原始图像', u'滤波结果']
# images = [img, newimg]
# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

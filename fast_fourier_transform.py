import cv2
import numpy as np

#np.pi=3.141592653589793
'''
二维傅里叶变换分解为两个维度上分别做一次一维傅里叶变换，数字图像处理p180
(x,y)，原图像坐标
(u,v)，变换后坐标（频谱）
'''

def fourier_transform(imgarray, type):
    '''获得所给图片的傅里叶频谱'''
    imgarray = dft(imgarray, type)
    #得到傅里叶谱，但此时低频成分在四角而非中间，用fftshift进行移动
    imgarray = np.fft.fftshift(imgarray)
    #参考opencv中文手册p147代码中的振幅图构造公式，abs取模
    imgarray = 20*np.log(np.abs(imgarray))
    return imgarray

def dft(imgarray, type):
    '用以选择类型,调用正变换或者逆变换'
    if type == 'DFT':
        return dft_2d(imgarray)
    elif type == 'IDFT':
        return idft_2d(imgarray)
    elif type == 'FFT':
        return fft_2d(imgarray)
    elif type == 'IFFT':
        return ifft_2d(imgarray)

def dft_1d(imgarray):
    '''计算一维傅里叶变换'''
    N = imgarray.shape[0]
    #创建一个N元素的一维矩阵，并且重塑为2d矩阵形式，即1xN矩阵，[[0,1,2,...,N-1]]
    x = np.arange(N).reshape((1, N))
    #转置x（作频谱坐标）
    u = x.reshape((N, 1))
    # 傅里叶变换核 e ^ (-j * 2 * π * u * x / N)
    test_UX = np.dot(u, x)
    E = np.exp(-1j * 2 * np.pi * np.dot(u, x) / N)
    test_v = np.dot(E, imgarray)
    return np.dot(E, imgarray)

def idft_1d(imgarray):
    N = imgarray.shape[0]
    # 创建一个N元素的一维矩阵，并且重塑为2d矩阵形式，即1xN矩阵，[[0,1,2,...,N-1]]
    x = np.arange(N).reshape((1, N))
    # 转置x（作频谱坐标）
    u = x.reshape((N, 1))
    # 傅里叶逆变换核 e ^ (j * 2 * π * u * x / N)
    E = np.exp(1j * 2 * np.pi * np.dot(u, x) / N)
    return np.dot(E, imgarray) / N

def dft_2d(imgarray):
    height,width = imgarray.shape
    # 创建频谱代表的矩阵,同时该矩阵为复矩阵元素为[0.+0.j]，type为complex128
    garray = np.zeros((height, width), dtype=complex)
    # 拆分为x，y两个方向的运算，详情见wps的180
    for row in range(height):
        # 对每行进行傅里叶变换，imgarray[row]即imgarray[row, :],是一个一维的矩阵，不是1xN的二维矩阵
        garray[row, :] = dft_1d(imgarray[row])
    for col in range(width):
        garray[:, col] = dft_1d(garray[:, col])
    return garray

def idft_2d(imgarray):
    height,width = imgarray.shape
    # 创建频谱代表的矩阵
    garray = np.zeros((height, width), dtype=complex)
    # 拆分为x，y两个方向的运算，详情见wps的180
    for row in range(height):
        # 对每行进行傅里叶变换，imgarray[row]即imgarray[row, :],是一个一维的矩阵，不是1xN的二维矩阵
        garray[row, :] = idft_1d(imgarray[row])
    for col in range(width):
        garray[:, col] = idft_1d(garray[:, col])
    return garray

def fft_1d(imgarray):
    imgarray = np.asarray(imgarray, dtype=complex)
    N = imgarray.shape[0]
    # 不符合条件的直接使用dft
    if N % 2 > 0:
        return dft_1d(imgarray)
    # 符合偶数个条件的使用快速傅里叶变换
    else:
        # 从第一个开始，步长2，即奇数部分
        even_part = fft_1d(imgarray[::2])
        # 从第二个开始，步长2，即偶数部分
        odd_part = fft_1d(imgarray[1::2])
        E = np.exp(-1j * 2 * np.pi * np.arange(N) / N)
        return np.concatenate([even_part + E[: N // 2] * odd_part,
                            even_part + E[N // 2 :] * odd_part])

def ifft_1d(imgarray):
    def rec(imgarray):
        imgarray = np.asarray(imgarray, dtype=complex)
        N = imgarray.shape[0]
        if N % 2 > 0:
            return idft_1d(imgarray) * N
        else:
            even_part = rec(imgarray[::2])
            odd_part = rec(imgarray[1::2])
            factor = np.exp(1j * 2 * np.pi * np.arange(N) / N)
            #拼接
            return np.concatenate([even_part + factor[: N // 2] * odd_part,
                                even_part + factor[N // 2 :] * odd_part])
    return rec(imgarray) / imgarray.shape[0]

def fft_2d(imgarray):
    M, N = imgarray.shape
    garray = np.zeros((M, N), dtype=complex)
    for row in range(M):
        garray[row, :] = fft_1d(imgarray[row])
    for col in range(N):
        garray[:, col] = fft_1d(garray[:, col])
    return garray

def ifft_2d(imgarray):
    M, N = imgarray.shape
    garray = np.zeros((M, N), dtype=complex)
    for row in range(M):
        garray[row, :] = ifft_1d(imgarray[row])
    for col in range(N):
        garray[:, col] = ifft_1d(garray[:, col])
    return garray

'''
单独运行该模块功能，则将下面的注释消除，图片路径在第133行中修改。
'''
# import matplotlib.pyplot as plt
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 读取图像
# img = cv2.imread('./src/rect.bmp',cv2.IMREAD_GRAYSCALE)
# # 可以运行的类型有DFT、IDFT、FFT、IFFT，自行修改标签运行。
# newimg = fourier_transform(img, 'DFT')
# # 显示图形
# titles = [u'原始图像', u'傅里叶变换结果']
# images = [img, newimg]
# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
import cv2
import numpy as np

#np.pi=3.141592653589793
'''
二维傅里叶变换分解为两个维度上分别做一次一维傅里叶变换，p180
(x,y)，原图像坐标
(u,v)，变换后坐标（频谱）
'''

def fourier_transform(imgarray, type):
    '''获得所给图片的傅里叶频谱'''
    imgarray = dft2d(imgarray, type)
    #得到傅里叶谱，但此时低频成分在四角而非中间，用fftshift进行移动
    imgarray = np.fft.fftshift(imgarray)
    imgarray = np.log(1 + np.abs(imgarray))
    imgarray = quantize(imgarray)
    return imgarray
def dft2d(imgarray, type):
    '用以选择类型,调用正变换或者逆变换'
    if type == 'DFT':
        return dft_2d(imgarray)
    elif type == 'IDFT':
        return idft_2d(imgarray)
def quantize(array):
    '''归一化，调整频谱图的对比度'''
    H, W = array.shape
    factor = (array.max() - array.min()) / 256
    #设置权重因子
    for row in range(H):
        for col in range(W):
            array[row, col] = round(array[row, col] / factor)
    return array
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
    #创建一个N元素的一维矩阵，并且重塑为2d矩阵形式，即1xN矩阵，[[0,1,2,...,N-1]]
    x = np.arange(N).reshape((1, N))
    #转置x（作频谱坐标）
    u = x.reshape((N, 1))
    # 傅里叶逆变换核 e ^ (j * 2 * π * u * x / N)
    E = np.exp(1j * 2 * np.pi * np.dot(u, x) / N)
    return np.dot(E, imgarray) / N

def dft_2d(imgarray):
    height,width = imgarray.shape
    #创建频谱代表的矩阵,同时该矩阵为复矩阵元素为[0.+0.j]，type为complex128
    garray = np.zeros((height, width), dtype=complex)
    #拆分为x，y两个方向的运算，详情见wps的180
    for row in range(height):
        #对每行进行傅里叶变换，imgarray[row]即imgarray[row, :],是一个一维的矩阵，不是1xN的二维矩阵
        garray[row, :] = dft_1d(imgarray[row])
    for col in range(width):
        garray[:, col] = dft_1d(garray[:, col])
    return garray

def idft_2d(imgarray):
    height,width = imgarray.shape
    #创建频谱代表的矩阵
    garray = np.zeros((height, width), dtype=complex)
    #拆分为x，y两个方向的运算，详情见wps的180
    for row in range(height):
        #对每行进行傅里叶变换，imgarray[row]即imgarray[row, :],是一个一维的矩阵，不是1xN的二维矩阵
        garray[row, :] = idft_1d(imgarray[row])
    for col in range(width):
        garray[:, col] = idft_1d(garray[:, col])
    return garray


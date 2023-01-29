import numpy as np

def histogram(img):
    #计算灰度直方图，统计img中坐标为i，j的灰度值数量。hist为一维矩阵，大小为1x256，数值代表频率。
    x = img.shape[0]
    y = img.shape[1]
    hist = np.zeros(256)
    for i in range(x):
        for j in range(y):
            hist[img[i][j]] += 1
    return hist

def threshTwoPeaks(image):

    # 计算灰度直方图
    hist = histogram(image)
    # 寻找灰度直方图的最大峰值对应的灰度值
    max_Location = np.where(hist == np.max(hist)) #maxLoc 中存放位置
    firstPeak = max_Location[0][0]
    # 寻找灰度直方图的第二个峰值对应的灰度值
    elementList = np.arange(256,dtype = np.uint64)
    #公式意义，离第一峰值越远，hist频率的权重更大（防止第二峰值就在第一峰值领域的情况）
    #但也导致第二峰值获取不够准确，但寻找的是两峰值间的最小值，只需要找到大概的范围
    #最小值就可以直接通过函数找到
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

'''
单独运行该模块功能，则将下面的注释消除，图片路径在第55行中修改。
'''
# import matplotlib.pyplot as plt
# import cv2
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 读取图像
# img = cv2.imread('./src/blood1.BMP',cv2.IMREAD_GRAYSCALE)
# newimg = threshTwoPeaks(img)
# # 显示图形
# titles = [u'原始图像', u'阈值分割结果']
# images = [img, newimg]
# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
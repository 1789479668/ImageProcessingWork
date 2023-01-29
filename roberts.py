import cv2
import numpy as np
import matplotlib.pyplot as plt

def Roberts(imgarray):
    # 定义Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    # 使用自定义的x、y卷积核（即roberts算子），空域卷积等于频域之积
    x = cv2.filter2D(imgarray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(imgarray, cv2.CV_16S, kernely)
# 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Roberts

'''
单独运行该模块功能，则将下面的注释消除，图片路径在第25行中修改。
'''
# # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 读取图像
# img = cv2.imread('./src/Miss.bmp',cv2.IMREAD_GRAYSCALE)
# newimg = Roberts(img)
# # 显示图形
# titles = [u'原始图像', u'Roberts算子']
# images = [img, newimg]
# for i in range(2):
#     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


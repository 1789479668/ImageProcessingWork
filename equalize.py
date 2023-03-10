import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_hist(imgarray):

    # 灰度直方图的矩阵
    hist = np.zeros(256)
    row = imgarray.shape[0]
    col = imgarray.shape[1]
    for i in range(row):
        for j in range(col):
            #每有相应灰度值，hist[该点灰度值]+1，进行统计
            hist[imgarray[i][j]] += 1
    #对hist作归一化，
    hist = hist / (row*col)
    #累计频率矩阵
    acc_freq = np.zeros(256)
    acc_freq[0] = hist[0]
    for i in range(1, 256):
        acc_freq[i] = acc_freq[i - 1] + hist[i]
    # 变换矩阵，累计频率*(灰度值范围-1)，再四舍五入，就是原灰度值的替换值。
    trans = np.round((acc_freq * 255),0)
    # 直方图均衡
    #对imgarray的第i行j列，第i行含有数据row(一维数组)，gray_data是组成row的元素。
    #即i，j是坐标，graydata是对应坐标的灰度值
    '''
    [[ 0  1  2  3  4],
     [ 5  6  7  8  9],
     [10 11 12 13 14],
     [15 16 17 18 19],
     [20 21 22 23 24]]
     则i=0时，row=[0 1 2 3 4],
        i=0，j=1，则graydata = 0，j=2，graydata = 2
       i=2时，row=[10 11 12 13 14]
        i=2,j=1，则graydata=10，j=2，graydata=11
    '''
    for i, row in enumerate(imgarray):
        for j, gray_data in enumerate(row):
            imgarray[i][j] = trans[gray_data]
    return imgarray
'''
单独运行该模块功能，则将下面的注释消除，读取图片路径在第46行中修改，保存图片路径在47行修改，默认为src_save文件夹，HEsave文件
可能是opencv的问题，无法反映灰度变化，因此检验该部分需保存图片后进行对比。
'''
# img = 'src/car.jpg'
# save_path = './src_save/HEsave.bmp'
# # 图像矩阵
#
# imgarray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)#打开图片,并把图片转换为np矩阵
# # 直方图均衡
# output_img = equalize_hist(imgarray)
# # 储存，save_path为地址，默认在src_save文件夹内，HEsave文件
# cv2.imwrite(save_path, output_img)

import cv2
import numpy as np

def AddNoise(img, path ,probility = 0.05, method = "salt_pepper"):
    '''
    :param img:图片
    :param path:处理后保存的地址
    :param probility:添加椒盐噪声的比例
    :param method:处理方法
    :return:
    '''
    imgarray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)#打开图片,并把图片转换为np矩阵
    H, W = imgarray.shape#将图片的长宽导出

    for i in range(H):
        for j in range(W):
            if np.random.random(1) < probility:
                #添加0.05占比的噪点
                if np.random.rand(1) < 0.5:#0.5加0，0.5加255
                    imgarray[i,j] = 0
                else:
                    imgarray[i,j] = 255
    cv2.imwrite(path, imgarray)

img_path = './src/Miss.bmp'
save_path = './src_save/salt_pepper_Miss.bmp'

AddNoise(img_path,save_path)